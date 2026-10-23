"""Cached data access for the Streamlit app.

Two cache layers (docs/PLAN_v2.md section 2), both in-process only (nothing written to disk in production):

* SHARED (ttl 600 s, same for every visitor): bootstrap, fixtures, live stats of finished GWs, element
  summaries. Served from the latest published BUNDLE (GitHub Release, .github/workflows/bundle.yml) when one
  is available: manifest.json (cached 300 s, failures too) -> every app table downloaded from that release's
  tag, sha256-verified, held 600 s per bundle_id. Any failure (no release yet, checksum mismatch, timeout,
  bundle older than BUNDLE_MAX_AGE_H) falls back to the live-API loaders below and the sidebar says
  "live API fallback". One bundle serves all shared loaders, so files of different runs are never mixed.
* PER-ID (ttl 300 s, keyed by the team ID): entry, history, picks of a GW, transfers, and the planning state.
  The planning state is built ONCE per ID from the cached per-ID pieces (reusing state.fetch_inputs through a
  cache-backed client adapter), so a cache miss never fans out into duplicate API calls.

Each loader returns its payload; fetch times are kept alongside so the sidebar can show "shared data as of"
and "team data as of". Exceptions (FPLAPIError, EntryNotFound, PlanningStateError) propagate to the page,
which degrades per feature (ui.guard). Failed calls are never cached.

MODEL (experimental, Phase B): xp.parquet of the SAME bundle (sha256-checked, held per bundle_id); only
when shared data comes from a bundle. A missing column `low_history` is treated as all False. Plans are
cached by (team ID, planning-state hash, bundle id, settings). The prospective registry's scores are read
from the spike-log branch (raw URL, cached 600 s, failures too).

Environment (tests / offline replay only): FPL_ROCKET_CACHE_DIR = raw JSON cache directory,
FPL_ROCKET_OFFLINE=1 = never touch the network (bundles disabled unless a source is set),
FPL_ROCKET_BUNDLE_SOURCE = releases base URL, a local bundle directory, or "off",
FPL_ROCKET_REGISTRY_SCORES = registry_scores.csv path or URL, or "off" (offline default: off).
"""

from __future__ import annotations

import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

from . import views
from .api import FPLClient
from .bundle import APP_TABLES, MANIFEST, AppShared, app_shared, verified_file
from .config import (
    BUNDLE_MAX_AGE_H,
    BUNDLE_TIMEOUT_S,
    FETCH_WORKERS,
    MANIFEST_TTL_S,
    MAX_TEAM_ID,
    RELEASES_URL,
    SHARED_TTL_S,
    TEAM_CACHE_ENTRIES,
    TEAM_TTL_S,
)
from .state import (
    PlanningState,
    build_planning_state,
    fetch_inputs,
    initial_squad_ids,
    missing_round_prices,
)

JSON = dict[str, Any]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


_local = threading.local()
# One bounded pool for the whole process: its threads keep their HTTP connections alive, and it caps the
# number of parallel calls this app makes to the FPL API across all sessions.
_POOL = ThreadPoolExecutor(max_workers=FETCH_WORKERS, thread_name_prefix="fpl-fetch")


def client() -> FPLClient:
    """One client per thread (requests.Session is not thread-safe; reusing it keeps TLS connections warm)."""
    cache = os.environ.get("FPL_ROCKET_CACHE_DIR") or None
    offline = os.environ.get("FPL_ROCKET_OFFLINE") == "1"
    key = (cache, offline)
    if getattr(_local, "key", None) != key:
        _local.client = FPLClient(cache, offline=offline, backoff=0.5)
        _local.key = key
    return _local.client


def _parallel(fn, items: list[Any]) -> list[Any]:
    """Map fn over items on the shared pool. fn must not itself submit to the pool (no nesting)."""
    if len(items) <= 1:
        return [fn(i) for i in items]
    return list(_POOL.map(fn, items))


# ============================================================================================ bundles
def bundle_source() -> str | None:
    """Where bundles come from: FPL_ROCKET_BUNDLE_SOURCE, else the GitHub releases (None = disabled)."""
    src = os.environ.get("FPL_ROCKET_BUNDLE_SOURCE")
    if src:
        return None if src.lower() == "off" else src
    if os.environ.get("FPL_ROCKET_OFFLINE") == "1":
        return None
    return RELEASES_URL


def _is_remote(source: str) -> bool:
    return source.startswith(("https://", "http://"))


XP_FILE = "xp.parquet"  # the model's table read by the app (section "model (xP)" below)
# EVERY bundle asset the app may download: manifest.json, the shared tables and the one-GW-per-row xP table.
# The model's joint simulations (xp_sims.parquet, ~N x players x 6 GWs rows) and xp_fixtures are NEVER read by
# the app; _read refuses any other name (tests/test_app_planner.py checks it).
APP_ASSETS: tuple[str, ...] = (MANIFEST, *(f"{t}.parquet" for t in APP_TABLES), XP_FILE)


def _read(source: str, name: str, bundle_id: str | None = None) -> bytes:
    """manifest.json of the latest release (bundle_id None) or one asset of release bundle-<bundle_id>.

    Assets are read from the manifest's own tag, so a release published meanwhile cannot mix in. Only names
    in APP_ASSETS are ever read.
    """
    if name not in APP_ASSETS:
        raise ValueError(f"{name} is not an app asset")
    if not _is_remote(source):
        return (Path(source) / name).read_bytes()
    base = source.rstrip("/")
    tag = "latest/download" if bundle_id is None else f"download/bundle-{bundle_id}"
    url = f"{base}/{tag}/{name}"
    resp = requests.get(url, timeout=BUNDLE_TIMEOUT_S, headers={"User-Agent": "fpl-rocket-app"})
    resp.raise_for_status()
    return resp.content


def _err(exc: BaseException) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        code = exc.response.status_code
        return "no bundle release published yet" if code == 404 else f"bundle download failed (HTTP {code})"
    if isinstance(exc, requests.Timeout):
        return "bundle download timed out"
    if isinstance(exc, FileNotFoundError):
        return "no bundle found"
    return f"bundle rejected: {exc}" if str(exc) else f"bundle rejected ({type(exc).__name__})"


@st.cache_data(ttl=MANIFEST_TTL_S, show_spinner=False, max_entries=4)
def _manifest(source: str) -> tuple[JSON | None, str | None]:
    """(manifest, None) or (None, reason). Failures are cached too: an outage costs one timeout per 300 s."""
    try:
        m = json.loads(_read(source, MANIFEST))
        created = datetime.fromisoformat(str(m["created_at"]).replace("Z", "+00:00"))
        if datetime.now(UTC) - created > timedelta(hours=BUNDLE_MAX_AGE_H):
            return None, f"latest bundle {m['bundle_id']} is older than {BUNDLE_MAX_AGE_H} h"
        return m, None
    except Exception as exc:  # noqa: BLE001 - any failure means fallback
        return None, _err(exc)


@st.cache_resource(ttl=SHARED_TTL_S, show_spinner="Loading shared data bundle...", max_entries=2)
def _bundle(source: str, bundle_id: str, _manifest: JSON) -> tuple[AppShared | None, str | None]:
    """Download and verify every app table of one bundle. Held once per process and never handed out
    directly (the accessors below give copies through st.cache_data). Failures are held for the ttl too."""
    try:
        names = [f"{t}.parquet" for t in APP_TABLES]
        blobs = _parallel(lambda n: _read(source, n, bundle_id), names)
        return app_shared(_manifest, dict(zip(names, blobs, strict=True)), views.LIVE_STATS), None
    except Exception as exc:  # noqa: BLE001 - any failure means fallback
        return None, _err(exc)


def _active() -> tuple[AppShared | None, JSON]:
    """(verified bundle or None, status) - the status feeds the sidebar."""
    src = bundle_source()
    if src is None:
        return None, {"mode": "live"}
    m, err = _manifest(src)
    if m is None:
        return None, {"mode": "fallback", "reason": err}
    b, err = _bundle(src, str(m["bundle_id"]), m)
    if b is None:
        return None, {"mode": "fallback", "reason": err}
    return b, {"mode": "bundle", "bundle_id": b.bundle_id, "created_at": b.created_at}


def shared_status() -> JSON:
    """{"mode": "bundle" | "live" | "fallback", ...}: bundle in use, bundles disabled, or bundle unusable."""
    return _active()[1]


@st.cache_data(ttl=SHARED_TTL_S, show_spinner=False, max_entries=8)
def _bundle_part(bundle_id: str, part: str, _b: AppShared) -> Any:
    """A copy of one part of the bundle (pages may mutate what they get; the held bundle stays intact)."""
    return getattr(_b, part)


def _from_bundle(part: str) -> Any:
    b, _ = _active()
    return None if b is None else _bundle_part(b.bundle_id, part, b)


# ============================================================================================ model (xP)
@st.cache_data(ttl=SHARED_TTL_S, show_spinner="Loading the xP table...", max_entries=2)
def _xp(source: str, bundle_id: str, _manifest: JSON) -> tuple[pd.DataFrame | None, str | None]:
    """(xp table, None) or (None, reason): xp.parquet of bundle `bundle_id`, sha256-checked."""
    names = {f["name"] for f in _manifest.get("files", [])}
    if XP_FILE not in names:
        err = _manifest.get("model_error")
        return None, f"bundle {bundle_id} has no xP table" + (f" (model error: {err})" if err else "")
    try:
        raw = _read(source, XP_FILE, bundle_id)
        verified_file(_manifest, XP_FILE, raw)
        df = pd.read_parquet(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 - the model is optional; pages degrade
        return None, _err(exc)
    if "low_history" not in df:
        df["low_history"] = False
    df["low_history"] = df["low_history"].fillna(False).astype(bool)
    return df, None


def _xp_source() -> tuple[str | None, JSON | None, AppShared | None, str | None]:
    b, status = _active()
    if b is None:
        why = status.get("reason") or "shared data come from the live API"
        return None, None, None, f"no model bundle ({why})"
    src = bundle_source()
    m, err = _manifest(src) if src else (None, "no bundle source")
    if m is None or str(m.get("bundle_id")) != b.bundle_id:
        return None, None, None, err or "bundle changed; reload the page"
    return src, m, b, None


def model_status() -> JSON:
    """{"available", "reason", "model_version", "version" (e.g. v0.1), "bundle_id", "created_at"} of the xP
    table of the bundle in use (only when shared data comes from a bundle)."""
    src, m, b, err = _xp_source()
    if src is None or m is None or b is None:
        return {"available": False, "reason": err}
    df, err = _xp(src, b.bundle_id, m)
    version = str(m.get("model_version") or "")
    return {
        "available": df is not None,
        "reason": err,
        "model_version": version or None,
        "version": version.split("-")[0] if version else None,
        "bundle_id": b.bundle_id,
        "created_at": b.created_at,
    }


def xp_table() -> pd.DataFrame | None:
    """The xP table of the bundle in use, or None (model_status says why)."""
    src, m, b, _ = _xp_source()
    if src is None or m is None or b is None:
        return None
    return _xp(src, b.bundle_id, m)[0]


@st.cache_data(ttl=TEAM_TTL_S, show_spinner="Solving the plan...", max_entries=TEAM_CACHE_ENTRIES)
def plan(team_id: int, state_key: tuple, bundle_id: str, settings: Any, _inp: Any) -> Any:
    """A planner.Plan cached by (team ID, planning-state hash, bundle id, objective settings)."""
    from .planner import solve

    return solve(_inp, settings)


@st.cache_data(ttl=TEAM_TTL_S, show_spinner="Solving the chip scenarios...", max_entries=TEAM_CACHE_ENTRIES)
def chip_plans(team_id: int, state_key: tuple, bundle_id: str, settings: Any, _inp: Any) -> Any:
    """(plan without chips, [ChipScenario]) cached like plan()."""
    from .planner import chip_scenarios

    return chip_scenarios(_inp, settings)


def registry_scores_source() -> str | None:
    src = os.environ.get("FPL_ROCKET_REGISTRY_SCORES")
    if src:
        return None if src.lower() == "off" else src
    if os.environ.get("FPL_ROCKET_OFFLINE") == "1":
        return None
    from .registry import SCORES_URL

    return SCORES_URL


@st.cache_data(ttl=SHARED_TTL_S, show_spinner=False, max_entries=2)
def _registry_scores(source: str) -> tuple[pd.DataFrame | None, str | None]:
    try:
        if _is_remote(source):
            resp = requests.get(source, timeout=BUNDLE_TIMEOUT_S, headers={"User-Agent": "fpl-rocket-app"})
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content)), None
        return pd.read_csv(source), None
    except (requests.HTTPError, FileNotFoundError):
        return None, "no scores yet"
    except Exception as exc:  # noqa: BLE001 - the page shows the reason
        return None, f"scores unavailable ({type(exc).__name__})"


def registry_scores() -> tuple[pd.DataFrame | None, str | None]:
    """(registry_scores.csv, None) or (None, reason): the prospective registry's scorecard."""
    src = registry_scores_source()
    if src is None:
        return None, "no scores yet"
    return _registry_scores(src)


# ============================================================================================ shared layer
@st.cache_data(ttl=SHARED_TTL_S, show_spinner="Loading FPL data...", max_entries=2)
def _bootstrap() -> tuple[JSON, str]:
    return client().bootstrap(), _now()


def bootstrap() -> JSON:
    b = _from_bundle("bootstrap")
    return b if b is not None else _bootstrap()[0]


def shared_as_of() -> str:
    b, _ = _active()
    return b.created_at if b is not None else _bootstrap()[1]


def _shared_key() -> str:
    b, _ = _active()
    return f"bundle {b.bundle_id}" if b is not None else f"live {_bootstrap()[1]}"


@st.cache_data(ttl=SHARED_TTL_S, show_spinner=False, max_entries=2)
def _players(key: str) -> pd.DataFrame:  # keyed by bundle id / bootstrap fetch time so both expire together
    return views.players_frame(bootstrap())


def players() -> pd.DataFrame:
    return _players(_shared_key())


@st.cache_data(ttl=SHARED_TTL_S, show_spinner=False, max_entries=2)
def _fixtures() -> list[JSON]:
    return client().fixtures()


def fixtures() -> list[JSON]:
    f = _from_bundle("fixtures")
    return f if f is not None else _fixtures()


@st.cache_data(ttl=SHARED_TTL_S, show_spinner="Loading gameweek stats...", max_entries=2)
def _live_table(gws: tuple[int, ...]) -> pd.DataFrame:
    def one(gw: int) -> JSON:
        return client().event_live(gw)

    return views.live_frame(dict(zip(gws, _parallel(one, list(gws)), strict=True)))


def live_table() -> pd.DataFrame:
    """Live stats of every finished GW (one row per GW x player)."""
    live = _from_bundle("live")
    return live if live is not None else _live_table(tuple(views.finished_gws(bootstrap())))


@st.cache_data(ttl=SHARED_TTL_S, show_spinner=False, max_entries=1000)
def element_summary(player_id: int) -> JSON:
    return client().element_summary(int(player_id))


# ============================================================================================ per-ID layer
def parse_team_id(text: str | None) -> tuple[int | None, str | None]:
    """(team_id, None) for a valid ID, else (None, error message). No network access."""
    raw = (text or "").strip()
    if not raw:
        return None, "Enter a team ID."
    if not raw.isdigit():
        return None, "A team ID contains digits only (find it in the URL of your Points page)."
    value = int(raw)
    if value <= 0 or value > MAX_TEAM_ID:
        return None, f"{raw} is not a valid team ID."
    return value, None


@st.cache_data(ttl=TEAM_TTL_S, show_spinner=False, max_entries=TEAM_CACHE_ENTRIES)
def _entry(team_id: int) -> tuple[JSON, str]:
    return client().entry(team_id), _now()


def entry(team_id: int) -> JSON:
    return _entry(team_id)[0]


def team_as_of(team_id: int) -> str:
    return _entry(team_id)[1]


@st.cache_data(ttl=TEAM_TTL_S, show_spinner=False, max_entries=TEAM_CACHE_ENTRIES)
def history(team_id: int) -> JSON:
    return client().entry_history(team_id)


@st.cache_data(ttl=TEAM_TTL_S, show_spinner=False, max_entries=TEAM_CACHE_ENTRIES)
def transfers(team_id: int) -> list[JSON]:
    return client().entry_transfers(team_id)


@st.cache_data(ttl=TEAM_TTL_S, show_spinner=False, max_entries=4 * TEAM_CACHE_ENTRIES)
def picks(team_id: int, gw: int) -> JSON | None:
    """Picks of one GW (None when not yet published)."""
    return client().entry_picks(team_id, gw)


def last_published_gw(team_id: int) -> int | None:
    cur = history(team_id).get("current") or []
    return int(max(h["event"] for h in cur)) if cur else None


class _CachedClient:
    """Adapter giving state.fetch_inputs the FPLClient interface, served from the caches above."""

    def entry(self, team_id: int) -> JSON:
        return entry(team_id)

    def entry_history(self, team_id: int) -> JSON:
        return history(team_id)

    def entry_picks(self, team_id: int, gw: int) -> JSON | None:
        return picks(team_id, gw)

    def entry_transfers(self, team_id: int) -> list[JSON]:
        return transfers(team_id)

    def bootstrap(self) -> JSON:
        return bootstrap()

    def element_summary(self, player_id: int) -> JSON:
        return element_summary(player_id)


def _prefetch_element_summaries(ids: list[int]) -> None:
    """Warm the shared element-summary cache in parallel (the planning state needs ~15 of them)."""
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

    ctx = get_script_run_ctx()

    def one(pid: int) -> None:
        if ctx is not None:
            add_script_run_ctx(threading.current_thread(), ctx)
        element_summary(pid)

    _parallel(one, ids)


@st.cache_data(
    ttl=TEAM_TTL_S, show_spinner="Reconstructing the planning state...", max_entries=TEAM_CACHE_ENTRIES
)
def planning_state(team_id: int) -> PlanningState:
    """Planning state for the upcoming GW (raises PlanningStateError / FPLAPIError).

    Initial-squad prices come from the bundle's round_prices; only players missing there cost a live
    element-summary call (all 15 when no bundle is available).
    """
    b, _ = _active()
    round_prices = b.round_prices if b is not None else None
    hist = history(team_id)
    cur = hist.get("current") or []
    if cur:
        started = int(entry(team_id).get("started_event") or cur[0]["event"])
        ids = initial_squad_ids({started: picks(team_id, started)}, started)
        missing = missing_round_prices(ids, round_prices)
        if missing:
            _prefetch_element_summaries(missing)
    inputs = fetch_inputs(_CachedClient(), team_id, round_prices=round_prices)
    as_of = (
        f"bundle {b.bundle_id} (created {b.created_at})"
        if b is not None
        else f"bootstrap fetched {shared_as_of()}"
    )
    return build_planning_state(**inputs, prices_as_of=as_of)


@st.cache_data(
    ttl=TEAM_TTL_S, show_spinner="Loading picks for every gameweek...", max_entries=TEAM_CACHE_ENTRIES
)
def all_picks(team_id: int, gws: tuple[int, ...]) -> dict[int, JSON | None]:
    """Picks for many GWs in one parallel fetch (used lazily by the captain-hindsight section)."""

    def one(gw: int) -> JSON | None:
        return client().entry_picks(team_id, gw)

    return dict(zip(gws, _parallel(one, list(gws)), strict=True))


@st.cache_data(ttl=TEAM_TTL_S, show_spinner=False, max_entries=TEAM_CACHE_ENTRIES)
def league_standings_page1(league_id: int) -> JSON:
    return client().league_classic_standings(league_id, 1)


def clear_all() -> None:
    """Drop every cached payload (tests and timing measurements)."""
    st.cache_data.clear()
    st.cache_resource.clear()
