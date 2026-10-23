"""Immutable, versioned data bundles: parquet tables + manifest.json with checksums.

A bundle is written to `<out_dir>/<bundle_id>.partial/` and renamed to `<out_dir>/<bundle_id>/` only after
every file and the manifest are on disk, so a reader never sees a half-written bundle. read_bundle()
re-verifies every checksum and row count and refuses partial or tampered bundles.

Tables (bundle format v2):
    players          bootstrap elements (one row per player)
    teams            bootstrap teams
    events           bootstrap events (nested fields JSON-encoded)
    bootstrap_misc   every other bootstrap key as (key, json): chips, game_settings, element_types, ...
    fixtures         all fixtures incl. finished results (team_h_score, team_a_score, finished, kickoff_time)
    gw_live          finished GWs stacked: one row per (gw, element) with every live stat
    gw_live_explain  long per-fixture scoring lines: (gw, element, fixture, identifier, value, points)
    element_history  element-summary/{id} history of EVERY player: one row per (element, round, fixture)
    round_prices     (element, round, value, source): history `value` per round, plus now_cost as the
                     upcoming round (source "now_cost"); replaces the planning state's live price calls

players / teams / events / bootstrap_misc / fixtures are written LOSSLESSLY from the JSON records (one Arrow
array per key, nulls kept, nested values JSON-encoded and listed under `json_columns` in the manifest), so
records_from_parquet() gives back the API's own values; the app rebuilds bootstrap-static from them.

The manifest also carries next_event / next_deadline_time and `pre_deadline`: True when the bundle is created
30-90 minutes before the next deadline (PRE_DEADLINE_WINDOW_MIN). The hourly publishing workflow
(.github/workflows/bundle.yml) tags such a bundle `deadline-gw<N>` and keeps it permanently; if GitHub's cron
delay leaves no bundle in that window, the week is recorded as missing.

Polite API load (hourly runs): element summaries (one call per player) are refetched only when the previous
release's element_history is older than 6 h, a GW has finished since it was fetched, or there is no previous
release; otherwise element_history is reused from the previous release and round_prices is rebuilt from it
plus the bootstrap now_cost. The manifest records element_history_source = "fetched" | "reused:<bundle_id>"
(the bundle that fetched it), element_history_fetched_at and element_history_reason.

Optional model tables (scripts/build_bundle.py --with-model; fpl_rocket.model): xp, xp_sims, xp_fixtures, with
model_version / model_fitted_on / model_features in the manifest. The app does not read them yet.

Optional news tables (scripts/build_bundle.py --news, on by default; fpl_rocket.news):
    status_history   FPL status / chance / news changes between consecutive bundles, carried forward from the
                     previous release (one row per element and observation where something changed)
    availability     per player: news recency (days since status change / news_added, transitions, stale
                     flag),
                     rule-parsed news, the GW+1 caps cap_fpl / cap_model_only / cap_with_news and, with the
                     model,
                     p_play_unconstrained / p_play_model_only / p_play_with_news
    news_items       matched news items (FPL news text + the public injury table) with the rule that read them
    news_unmatched   items no player could be matched to (kept for inspection, never guessed)
with news_* / status_history_* fields in the manifest; a news failure is recorded as news_error.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from .api import FPLAPIError, FPLClient
from .config import FETCH_WORKERS
from .rules import get_rules

MANIFEST = "manifest.json"
PARTIAL_SUFFIX = ".partial"
FORMAT_VERSION = 2
CRON_INTERVAL_H = 1  # .github/workflows/bundle.yml runs hourly (at :40)
# pre_deadline = the bundle created 30-90 minutes before the next deadline (published as deadline-gw<N>)
PRE_DEADLINE_WINDOW_MIN = (30, 90)
# element summaries (668 calls) are refetched only when the previous release's copy is older than this,
# a GW has finished since, or there is no previous release (element_history_decision)
ELEMENT_HISTORY_MAX_AGE_H = 6.0
# the joint simulation sample is large and highly repetitive: zstd keeps it small
PARQUET_COMPRESSION = {"xp_sims": "zstd"}
JSON_COLS_KEY = b"fpl_rocket.json_columns"
# tables the app needs (data.py downloads only these); element_history / gw_live_explain are for models
APP_TABLES = ("players", "teams", "events", "bootstrap_misc", "fixtures", "gw_live", "round_prices")
NUMERIC_STRINGS = (
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
)

JSON = dict[str, Any]
Table = pd.DataFrame | pa.Table


class BundleError(RuntimeError):
    """The bundle is missing, partial, or fails validation."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================== lossless JSON <-> Arrow
def records_table(records: list[JSON]) -> pa.Table:
    """Arrow table from JSON records without losing types: ints stay ints, nulls stay null.

    Dict/list values are JSON-encoded; a column whose scalar types cannot share one Arrow type (e.g. int and
    str) is JSON-encoded as a whole. Encoded column names are stored in the schema metadata.
    """
    keys: list[str] = []
    for r in records:
        keys.extend(k for k in r if k not in keys)
    arrays, json_cols = [], []
    for k in keys:
        values = [r.get(k) for r in records]
        nested = any(isinstance(v, dict | list) for v in values)
        arr = None
        if not nested:
            try:
                arr = pa.array(values)
            except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, OverflowError):
                arr = None
        if arr is None:
            encoded = [None if v is None else json.dumps(v, ensure_ascii=False) for v in values]
            arr = pa.array(encoded, pa.string())
            json_cols.append(k)
        elif pa.types.is_null(arr.type):
            arr = arr.cast(pa.string())  # all-null column: give parquet a concrete type
        arrays.append(arr)
    table = pa.Table.from_arrays(arrays, names=keys) if keys else pa.table({})
    return table.replace_schema_metadata({JSON_COLS_KEY: json.dumps(json_cols)})


def json_columns(table: pa.Table) -> list[str]:
    meta = table.schema.metadata or {}
    return json.loads(meta.get(JSON_COLS_KEY, b"[]"))


def table_records(table: pa.Table, json_cols: list[str] | None = None) -> list[JSON]:
    """Inverse of records_table: JSON records with nested values decoded."""
    cols = json_columns(table) if json_cols is None else json_cols
    rows = table.to_pylist()
    for r in rows:
        for c in cols:
            if r.get(c) is not None:
                r[c] = json.loads(r[c])
    return rows


def records_from_parquet(data: bytes, json_cols: list[str] | None = None) -> list[JSON]:
    return table_records(pq.read_table(io.BytesIO(data)), json_cols)


# ============================================================================== table builders
def live_tables(gw_live: dict[int, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(gw_live, gw_live_explain) DataFrames from {gw: event/{gw}/live JSON}."""
    stats_rows, explain_rows = [], []
    for gw, live in sorted(gw_live.items()):
        for el in live["elements"]:
            stats_rows.append({"gw": gw, "element": el["id"], **el["stats"]})
            for fx in el.get("explain") or []:
                for s in fx.get("stats", []):
                    explain_rows.append(
                        {
                            "gw": gw,
                            "element": el["id"],
                            "fixture": fx.get("fixture"),
                            "identifier": s["identifier"],
                            "value": s["value"],
                            "points": s["points"],
                        }
                    )
    stats = pd.DataFrame(stats_rows)
    for col in NUMERIC_STRINGS:
        if col in stats:
            stats[col] = pd.to_numeric(stats[col], errors="coerce")
    return stats, pd.DataFrame(explain_rows)


ELEMENT_HISTORY_LEAD = (
    "element",
    "round",
    "fixture",
    "opponent_team",
    "was_home",
    "kickoff_time",
    "minutes",
)


def element_history_frame(summaries: dict[int, JSON]) -> pd.DataFrame:
    """One row per (element, round, fixture) from {element_id: element-summary JSON} (every stat kept)."""
    rows = []
    for pid, summary in sorted(summaries.items()):
        for h in summary.get("history") or []:
            rows.append({**h, "element": int(h.get("element", pid))})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["element", "round", "value"])
    for col in NUMERIC_STRINGS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    lead = [c for c in ELEMENT_HISTORY_LEAD if c in df]
    df = df[lead + [c for c in df.columns if c not in lead]]
    sort = [c for c in ("element", "round", "fixture") if c in df]
    return df.sort_values(sort, kind="stable").reset_index(drop=True)


def round_prices_frame(element_history: pd.DataFrame, bootstrap: JSON) -> pd.DataFrame:
    """(element, round, value, source): one price per player per round played, plus the upcoming round.

    A double GW has two history rows with the same `value`; the first is kept. The upcoming round (events
    is_next) gets every player's now_cost, source "now_cost", unless history already has that round.
    """
    cols = ["element", "round", "value", "source"]
    hist = pd.DataFrame(columns=cols)
    if not element_history.empty and {"element", "round", "value"} <= set(element_history.columns):
        hist = (
            element_history[["element", "round", "value"]]
            .dropna()
            .astype({"element": "int64", "round": "int64", "value": "int64"})
            .drop_duplicates(["element", "round"], keep="first")
            .assign(source="history")
        )
    nxt = [int(e["id"]) for e in bootstrap.get("events", []) if e.get("is_next")]
    upcoming = pd.DataFrame(columns=cols)
    if nxt:
        upcoming = pd.DataFrame(
            [
                {"element": int(p["id"]), "round": nxt[0], "value": int(p["now_cost"]), "source": "now_cost"}
                for p in bootstrap.get("elements", [])
            ],
            columns=cols,
        )
        have = set(zip(hist["element"], hist["round"], strict=True))
        keys = zip(upcoming["element"], upcoming["round"], strict=True)
        upcoming = upcoming[[k not in have for k in keys]]
    out = pd.concat([hist, upcoming], ignore_index=True) if len(upcoming) else hist
    out = out.astype({"element": "int64", "round": "int64", "value": "int64", "source": "string"})
    return out.sort_values(["element", "round"], kind="stable").reset_index(drop=True)[cols]


def round_prices_lookup(round_prices: pd.DataFrame) -> dict[int, dict[int, int]]:
    """{element: {round: value}} for state.fetch_inputs (elements absent here fall back to live calls)."""
    out: dict[int, dict[int, int]] = {}
    for e, r, v in zip(round_prices["element"], round_prices["round"], round_prices["value"], strict=True):
        out.setdefault(int(e), {})[int(r)] = int(v)
    return out


# ============================================================================== element-summary fan-out
class _Pacer:
    """Process-wide polite pacing: request starts are spaced at least `interval` seconds apart."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        if slot > now:
            time.sleep(slot - now)


def fetch_element_summaries(
    client_factory: Callable[[], FPLClient],
    ids: list[int],
    *,
    workers: int = FETCH_WORKERS,
    min_interval: float = 0.08,
) -> tuple[dict[int, JSON], list[int]]:
    """Fetch element-summary/{id} for every id on a bounded pool (one client per thread, paced).

    Each call already retries 403/429/5xx with backoff (FPLClient); ids still failing are retried once
    sequentially. Returns ({id: summary}, missing ids).
    """
    local = threading.local()
    pacer = _Pacer(min_interval)

    def one(pid: int) -> tuple[int, JSON | None]:
        if not hasattr(local, "client"):
            local.client = client_factory()
        pacer.wait()
        try:
            return pid, local.client.element_summary(pid)
        except FPLAPIError:
            return pid, None

    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="bundle-fetch") as pool:
        results = dict(pool.map(one, ids))
    missing = [pid for pid, body in results.items() if body is None]
    if missing:
        retry = client_factory()
        for pid in missing:
            pacer.wait()
            with contextlib.suppress(FPLAPIError):
                results[pid] = retry.element_summary(pid)
    got = {pid: body for pid, body in results.items() if body is not None}
    return got, sorted(pid for pid in ids if pid not in got)


def _clone_factory(client: FPLClient) -> Callable[[], FPLClient]:
    def make() -> FPLClient:
        return FPLClient(
            client.cache_dir,
            offline=client.offline,
            timeout=client.timeout,
            backoff=client.backoff,
            base_url=client.base_url,
        )

    return make


# ============================================================================== previous release
# tables of the previous release carried forward when present (missing or unreadable -> simply absent)
PREVIOUS_OPTIONAL_TABLES = ("status_history",)


@dataclass
class PreviousRelease:
    """The latest published bundle: its manifest, element_history and the optional carried-forward tables
    (PREVIOUS_OPTIONAL_TABLES), all checksum-verified."""

    manifest: JSON
    element_history: pd.DataFrame
    extra: dict[str, pd.DataFrame] = field(default_factory=dict)


def _read_release_file(source: str, name: str, bundle_id: str | None = None) -> bytes:
    """manifest.json of the latest release (bundle_id None) or one asset of release bundle-<bundle_id>;
    `source` is the releases base URL or a local bundle directory (tests, offline runs)."""
    if not source.startswith(("https://", "http://")):
        return (Path(source) / name).read_bytes()
    tag = "latest/download" if bundle_id is None else f"download/bundle-{bundle_id}"
    resp = requests.get(
        f"{source.rstrip('/')}/{tag}/{name}", timeout=(5.0, 60.0), headers={"User-Agent": "fpl-rocket-bundle"}
    )
    resp.raise_for_status()
    return resp.content


def fetch_previous_release(source: str | None) -> tuple[PreviousRelease | None, str]:
    """(previous release, note). None when there is none or it is unusable (the note says why)."""
    if not source:
        return None, "no previous release source"
    try:
        manifest = json.loads(_read_release_file(source, MANIFEST))
        name = "element_history.parquet"
        data = _read_release_file(source, name, str(manifest["bundle_id"]))
        verified_file(manifest, name, data)
        extra: dict[str, pd.DataFrame] = {}
        listed = {f.get("table") for f in manifest.get("files", [])}
        for table in PREVIOUS_OPTIONAL_TABLES:
            if table not in listed:
                continue
            with contextlib.suppress(Exception):  # optional: a bad copy only means "start afresh"
                raw = _read_release_file(source, f"{table}.parquet", str(manifest["bundle_id"]))
                verified_file(manifest, f"{table}.parquet", raw)
                extra[table] = pd.read_parquet(io.BytesIO(raw))
        return PreviousRelease(manifest, pd.read_parquet(io.BytesIO(data)), extra), "ok"
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else None
        return None, "no previous release" if code == 404 else f"previous release unreadable (HTTP {code})"
    except Exception as exc:  # noqa: BLE001 - any failure means: fetch fresh
        return None, f"previous release unusable: {exc!r}"


def element_history_origin(manifest: JSON) -> tuple[str, str | None, list[int] | None]:
    """(origin bundle id, fetched_at, finished GWs at fetch) of a manifest's element_history. Manifests
    written before the reuse rule existed fetched it themselves (snapshot times in source_snapshots)."""
    src = str(manifest.get("element_history_source") or "fetched")
    origin = src.split(":", 1)[1] if src.startswith("reused:") else str(manifest.get("bundle_id"))
    snaps = manifest.get("source_snapshots") or {}
    fetched_at = manifest.get("element_history_fetched_at") or snaps.get("element_summary")
    finished = manifest.get("element_history_finished_gws")
    if finished is None and "finished_gws" in snaps:
        finished = [int(g) for g in str(snaps["finished_gws"]).split(",") if g.strip()]
    return origin, fetched_at, finished


def element_history_decision(
    previous: JSON | None,
    finished_now: list[int],
    now: datetime,
    max_age_h: float = ELEMENT_HISTORY_MAX_AGE_H,
) -> tuple[bool, str]:
    """(refetch?, reason). The 668 element-summary calls are made only when (a) there is no previous release,
    (b) its element_history is older than max_age_h, or (c) a GW has finished since it was fetched;
    otherwise the previous release's element_history is reused (prices patched from bootstrap now_cost)."""
    if previous is None:
        return True, "no previous release"
    _, fetched_at, finished = element_history_origin(previous)
    if not fetched_at or finished is None:
        return True, "previous manifest lacks element_history provenance"
    age = now - datetime.fromisoformat(str(fetched_at).replace("Z", "+00:00"))
    if age > timedelta(hours=max_age_h):
        return True, f"previous element_history is {age.total_seconds() / 3600:.1f} h old (> {max_age_h:g} h)"
    if sorted(int(g) for g in finished) != sorted(int(g) for g in finished_now):
        return True, "a gameweek has finished since the previous element_history was fetched"
    return False, f"reused: {age.total_seconds() / 3600:.1f} h old, no gameweek finished since"


# ============================================================================== build
@dataclass
class BuildResult:
    tables: dict[str, Table]
    snapshots: dict[str, str]
    extra: dict[str, Any] = field(default_factory=dict)


def deadline_info(events: list[JSON], now: datetime) -> JSON:
    """next_event, next_deadline_time and pre_deadline: deadline - now in (30 min, 90 min]."""
    nxt = next((e for e in events if e.get("is_next")), None)
    if nxt is None:
        return {"next_event": None, "next_deadline_time": None, "pre_deadline": False}
    deadline = datetime.fromisoformat(str(nxt["deadline_time"]).replace("Z", "+00:00"))
    left = deadline - now
    lo, hi = PRE_DEADLINE_WINDOW_MIN
    return {
        "next_event": int(nxt["id"]),
        "next_deadline_time": deadline.isoformat(timespec="seconds"),
        "pre_deadline": timedelta(minutes=lo) < left <= timedelta(minutes=hi),
    }


def build_tables(
    client: FPLClient,
    now: Callable[[], datetime] = _utc_now,
    *,
    element_ids: list[int] | None = None,
    client_factory: Callable[[], FPLClient] | None = None,
    max_missing_frac: float = 0.02,
    min_interval: float = 0.08,
    previous: PreviousRelease | None = None,
) -> BuildResult:
    """Fetch the shared data and return the tables, source snapshot times and manifest extras.

    element_ids limits the element-summary fan-out (tests); default = every player in bootstrap. Raises
    BundleError when more than max_missing_frac of the summaries cannot be fetched. `previous` (the latest
    published release) lets element_history be reused under element_history_decision().
    """
    snaps: dict[str, str] = {}
    t0 = time.monotonic()
    bootstrap = client.bootstrap()
    snaps["bootstrap-static"] = now().isoformat(timespec="seconds")
    fixtures = client.fixtures()
    snaps["fixtures"] = now().isoformat(timespec="seconds")
    finished = [e["id"] for e in bootstrap["events"] if e.get("finished")]
    lives = {gw: client.event_live(gw) for gw in finished}
    snaps["event_live"] = now().isoformat(timespec="seconds")
    gw_live, explain = live_tables(lives)

    refetch, reason = element_history_decision(previous.manifest if previous else None, finished, now())
    if refetch:
        ids = element_ids if element_ids is not None else [int(p["id"]) for p in bootstrap["elements"]]
        summaries, missing = fetch_element_summaries(
            client_factory or _clone_factory(client), ids, min_interval=min_interval
        )
        snaps["element_summary"] = now().isoformat(timespec="seconds")
        if ids and len(missing) > max_missing_frac * len(ids):
            raise BundleError(f"{len(missing)} of {len(ids)} element summaries failed: {missing[:20]}")
        history = element_history_frame(summaries)
        provenance = {
            "element_history_source": "fetched",
            "element_history_fetched_at": snaps["element_summary"],
            "element_history_finished_gws": finished,
            "element_history_reason": reason,
            "element_summary_missing": missing,
            "element_summary_requested": len(ids),
        }
    else:
        assert previous is not None
        origin, fetched_at, _ = element_history_origin(previous.manifest)
        history = previous.element_history
        snaps["element_summary"] = str(fetched_at)
        provenance = {
            "element_history_source": f"reused:{origin}",
            "element_history_fetched_at": fetched_at,
            "element_history_finished_gws": finished,
            "element_history_reason": reason,
            "element_history_previous_bundle": previous.manifest.get("bundle_id"),
            "element_summary_missing": previous.manifest.get("element_summary_missing", []),
            "element_summary_requested": 0,
        }

    misc = [
        {"key": k, "json": json.dumps(v, ensure_ascii=False)}
        for k, v in bootstrap.items()
        if k not in ("elements", "teams", "events")
    ]
    tables: dict[str, Table] = {
        "players": records_table(bootstrap["elements"]),
        "teams": records_table(bootstrap["teams"]),
        "events": records_table(bootstrap["events"]),
        "bootstrap_misc": records_table(misc),
        "fixtures": records_table(fixtures),
        "gw_live": gw_live,
        "gw_live_explain": explain,
        "element_history": history,
        # history prices + the upcoming round from bootstrap now_cost (fresh even when history is reused)
        "round_prices": round_prices_frame(history, bootstrap),
    }
    snaps["finished_gws"] = ",".join(map(str, finished))
    extra = {**provenance, "fetch_seconds": round(time.monotonic() - t0, 1)}
    return BuildResult(tables, snaps, extra)


def _rows(t: Table) -> int:
    return t.num_rows if isinstance(t, pa.Table) else len(t)


def write_tables(
    tables: dict[str, Table],
    out_dir: str | Path,
    snapshots: dict[str, str],
    *,
    bundle_id: str | None = None,
    extra: dict[str, Any] | None = None,
    created: datetime | None = None,
) -> Path:
    """Write tables as one bundle under out_dir; returns the final bundle directory."""
    created = created or _utc_now()
    bundle_id = bundle_id or created.strftime("%Y%m%dT%H%M%SZ")
    root = Path(out_dir)
    final = root / bundle_id
    if final.exists():
        raise BundleError(f"bundle {final} already exists (bundles are immutable)")
    tmp = root / (bundle_id + PARTIAL_SUFFIX)
    tmp.mkdir(parents=True, exist_ok=False)
    files = []
    for name, t in tables.items():
        path = tmp / f"{name}.parquet"
        if isinstance(t, pa.Table):
            pq.write_table(t, path)
            jcols = json_columns(t)
        else:
            t.to_parquet(path, index=False, compression=PARQUET_COMPRESSION.get(name, "snappy"))
            jcols = []
        files.append(
            {
                "name": path.name,
                "table": name,
                "rows": _rows(t),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "json_columns": jcols,
            }
        )
    manifest = {
        "bundle_id": bundle_id,
        "format_version": FORMAT_VERSION,
        "created_at": created.isoformat(timespec="seconds"),
        "source_snapshots": snapshots,
        "rules_version": get_rules().RULES_VERSION,
        "model_version": None,
        "counts": {f["table"]: f["rows"] for f in files},
        "files": files,
        **(extra or {}),
    }
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, final)
    return final


ModelFn = Callable[[dict[str, Table]], tuple[dict[str, Table], dict[str, Any]]]
NewsFn = Callable[
    [dict[str, Table], "PreviousRelease | None", datetime], tuple[dict[str, Table], dict[str, Any]]
]


def write_bundle(
    out_dir: str | Path,
    client: FPLClient | None = None,
    *,
    now: Callable[[], datetime] = _utc_now,
    previous: PreviousRelease | None = None,
    model: ModelFn | None = None,
    news: NewsFn | None = None,
    **build_kw: Any,
) -> Path:
    """Pull the shared data (bootstrap, fixtures, finished GWs' live data, element summaries unless reused
    from `previous`) and write one immutable bundle whose manifest carries the deadline flags.

    `model(tables) -> (extra tables, manifest fields)` adds the xP tables (scripts/build_bundle.py
    --with-model); a model failure is recorded as model_error and the data bundle is still written.
    `news(tables, previous, now) -> (news tables, manifest fields)` runs first (fpl_rocket.news; the model
    reads
    its `availability`); a news failure is recorded as news_error."""
    result = build_tables(client or FPLClient(), now, previous=previous, **build_kw)
    tables = dict(result.tables)
    news_extra: dict[str, Any] = {}
    if news is not None:
        try:
            news_tables, news_extra = news(tables, previous, now())
            tables.update(news_tables)
        except Exception as exc:  # noqa: BLE001 - shared data must still publish; the failure is recorded
            news_extra = {"news_error": f"{type(exc).__name__}: {exc}"}
    model_extra: dict[str, Any] = {}
    if model is not None:
        try:
            extra_tables, model_extra = model(tables)
            tables.update(extra_tables)
        except Exception as exc:  # noqa: BLE001 - shared data must still publish; the failure is recorded
            model_extra = {"model_version": None, "model_error": f"{type(exc).__name__}: {exc}"}
    created = now()
    events = table_records(result.tables["events"]) if isinstance(result.tables["events"], pa.Table) else []
    extra = {
        **result.extra,
        **news_extra,
        **model_extra,
        **deadline_info(events, created),
        "pre_deadline_window_minutes": list(PRE_DEADLINE_WINDOW_MIN),
        "cron_interval_hours": CRON_INTERVAL_H,
    }
    return write_tables(tables, out_dir, result.snapshots, extra=extra, created=created)


# ============================================================================== read
def read_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    d = Path(bundle_dir)
    if d.name.endswith(PARTIAL_SUFFIX):
        raise BundleError(f"{d} is a partial bundle")
    m = d / MANIFEST
    if not m.exists():
        raise BundleError(f"{d} has no {MANIFEST}: partial or not a bundle")
    return json.loads(m.read_text(encoding="utf-8"))


def read_bundle(bundle_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Validate checksums and row counts, then return {table: DataFrame}. Refuses partial bundles."""
    d = Path(bundle_dir)
    manifest = read_manifest(d)
    out: dict[str, pd.DataFrame] = {}
    for f in manifest["files"]:
        path = d / f["name"]
        if not path.exists():
            raise BundleError(f"{path} listed in manifest but missing")
        if _sha256(path) != f["sha256"]:
            raise BundleError(f"{path} checksum mismatch")
        df = pd.read_parquet(path)
        if len(df) != f["rows"]:
            raise BundleError(f"{path} has {len(df)} rows, manifest says {f['rows']}")
        out[f["table"]] = df
    return out


def latest_bundle(out_dir: str | Path) -> Path | None:
    """Newest complete bundle directory (ignores *.partial), or None."""
    root = Path(out_dir)
    if not root.exists():
        return None
    done = sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and not p.name.endswith(PARTIAL_SUFFIX) and (p / MANIFEST).exists()
    )
    return done[-1] if done else None


# ============================================================================== app payload
@dataclass
class AppShared:
    """What the app reads from one bundle: everything the shared loaders serve, from a single run."""

    bundle_id: str
    created_at: str
    bootstrap: JSON
    fixtures: list[JSON]
    live: pd.DataFrame  # views.live_frame layout: gw, element, LIVE_STATS (ints)
    round_prices: dict[int, dict[int, int]]


def verified_file(manifest: JSON, name: str, data: bytes) -> JSON:
    """The manifest entry of `name` after checking size and sha256 of the downloaded bytes."""
    entry = next((f for f in manifest.get("files", []) if f["name"] == name), None)
    if entry is None:
        raise BundleError(f"{name} not listed in manifest {manifest.get('bundle_id')}")
    if sha256_bytes(data) != entry["sha256"]:
        raise BundleError(f"{name} checksum mismatch (bundle {manifest.get('bundle_id')})")
    return entry


def app_shared(manifest: JSON, files: dict[str, bytes], live_stats: tuple[str, ...]) -> AppShared:
    """Rebuild bootstrap-static, fixtures, the live table and round prices from verified bundle files."""
    if int(manifest.get("format_version", 1)) < FORMAT_VERSION:
        raise BundleError(f"bundle {manifest.get('bundle_id')} has format v1 (no round_prices)")
    tables: dict[str, Any] = {}
    for table in APP_TABLES:
        name = f"{table}.parquet"
        if name not in files:
            raise BundleError(f"{name} missing from bundle {manifest.get('bundle_id')}")
        entry = verified_file(manifest, name, files[name])
        if table in ("gw_live", "round_prices"):
            tables[table] = pd.read_parquet(io.BytesIO(files[name]))
        else:
            tables[table] = records_from_parquet(files[name], entry.get("json_columns"))
        if len(tables[table]) != entry["rows"]:
            raise BundleError(f"{name} row count differs from manifest")
    bootstrap = {r["key"]: json.loads(r["json"]) for r in tables["bootstrap_misc"]}
    bootstrap.update(elements=tables["players"], teams=tables["teams"], events=tables["events"])
    finished = {int(e["id"]) for e in bootstrap["events"] if e.get("finished")}
    g = tables["gw_live"]
    live = pd.DataFrame(columns=["gw", "element", *live_stats])
    if not g.empty:
        g = g[g["gw"].isin(finished)]
        live = g.reindex(columns=["gw", "element", *live_stats]).fillna(0).astype("int64")
        live = live.reset_index(drop=True)
    return AppShared(
        bundle_id=str(manifest["bundle_id"]),
        created_at=str(manifest["created_at"]),
        bootstrap=bootstrap,
        fixtures=tables["fixtures"],
        live=live,
        round_prices=round_prices_lookup(tables["round_prices"]),
    )
