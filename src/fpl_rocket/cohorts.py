"""Frozen prospective cohorts of real managers (docs/PLAN_v3_blind_follow.md sections 2 and 5, Phase 1e).

The prospective comparison with real managers uses cohorts whose membership is fixed BEFORE the scored period,
chosen only with information available at the freeze, and results are reported as "percentile within this
cohort" - never as an overall FPL percentile.

FREEZE (scripts/cohort_freeze.py, run once):
    random  draw candidate entry ids uniformly (without replacement) from 1..total_players (bootstrap at the
            freeze) with a seeded generator; a candidate is kept when it exists and is ACTIVE = has a history
            row for each of the last 3 finished gameweeks (ACTIVE_LAST_N); the first COHORT_SIZE that qualify,
            in draw order, are the cohort. Every rejection is counted (not_found, inactive).
    top10k  COHORT_SIZE positions drawn uniformly without replacement from a canonical top-10,000 list of the
            overall league (leagues-classic/314): all 200 pages fetched in TOP_SNAPSHOTS passes, the distinct
            entries with rank <= 10,000 ordered by (rank, entry id). The league's order within ties changes
            between page fetches, so a single pass both repeats and misses entries (see freeze_top). No
            activity rule (a top-10k entry is active by construction).
    For every member: entry id, overall rank and total points after the last finished GW, that GW, the chips
    already used (name + GW), team value and bank after that GW (FPL units: tenths of a million). Everything
    is read from rows of GWs <= the last finished GW, so a freeze that runs into the next deadline cannot see
    it.
    NO names, no regions are stored (also not in the local cache).
    Output on the data-only `spike-log` branch: registry/cohorts/cohort_<YYYYMMDD>.json (immutable) and
    cohort_<YYYYMMDD>_manifest.json (seed, times, GWs, rules, counts, request log, sha256 of the cohort file).

SCORE (scripts/cohort_score.py, daily workflow cohort_score.yml): for every finished and data-checked GW after
the freeze GW not yet scored, ONE entry/{id}/history/ call per member (it covers every pending GW), staged on
disk; only when every member has been fetched are the rows appended to cohort_scores.csv and
cohort_summary.csv recomputed (atomic replace). Members are never dropped: a member without a row for the GW
(status no_row) or whose entry vanished (status not_found) keeps a row with empty points. The scorer refuses
to run when the cohort file's sha256 differs from the manifest's (or from KNOWN_FREEZES below).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

JSON = dict[str, Any]

RANDOM, TOP10K = "random", "top10k"
COHORTS = (RANDOM, TOP10K)
COHORT_SIZE = 1500
TOP_N = 10_000
OVERALL_LEAGUE = 314
PAGE_SIZE = 50
ACTIVE_LAST_N = 3
TOP_SNAPSHOTS = 2  # full passes over the top-10k standings pages (see freeze_top)
MAX_RANDOM_CANDIDATES = 8000  # hard stop for the random draw (budget guard; reported if reached)
SEED_SCHEME = (
    "random.Random(int(sha256(f'{seed}:{cohort}').hexdigest()[:16], 16)), cohort in {random, top10k}"
)
RULES = {
    "random": (
        "candidate ids drawn uniformly without replacement from 1..total_players (bootstrap at the freeze); "
        "kept if entry/{id}/history/ exists (404 = not_found) and has a row for each of the last "
        f"{ACTIVE_LAST_N} finished gameweeks (else inactive); the first {COHORT_SIZE} kept in draw order"
    ),
    "top10k": (
        f"all {TOP_N // PAGE_SIZE} pages of the top {TOP_N} of leagues-classic/{OVERALL_LEAGUE} fetched in "
        f"{TOP_SNAPSHOTS} snapshots (the order within ties changes between fetches); canonical list = the "
        f"distinct entries with rank <= {TOP_N} seen in any snapshot, ordered by (rank, entry id), cut at "
        f"{TOP_N}; {COHORT_SIZE} positions drawn uniformly without replacement from 1..len(list); no "
        "activity rule"
    ),
    "at_freeze": (
        "overall_rank, total_points, value, bank = the history row of the last finished GW; chips_used = "
        "history chips with event <= that GW; nothing from later GWs is read"
    ),
    "privacy": "entry ids and FPL numbers only: no names, no regions",
    "scoring": (
        "per finished, data-checked GW after the freeze GW: net points = points - event_transfers_cost; "
        "members are never dropped (no_row / not_found rows keep empty points)"
    ),
}
# sha256 of the published cohort files: a second record next to the manifest's (code on main)
KNOWN_FREEZES: dict[str, str] = {
    "20261106": "8c625534bfc9de27c0974303168038b628d7ed92e4c9c2c9e05df2ffc22f51c5",
}

COHORTS_DIR = "registry/cohorts"  # path on the spike-log branch
SCORES_NAME, SUMMARY_NAME = "cohort_scores.csv", "cohort_summary.csv"
RAW_BASE = "https://raw.githubusercontent.com/drdata2018/fpl_rocket/spike-log/"
SCORES_URL = RAW_BASE + COHORTS_DIR + "/" + SCORES_NAME
SUMMARY_URL = RAW_BASE + COHORTS_DIR + "/" + SUMMARY_NAME

SCORE_COLUMNS = (
    "freeze_id",
    "gw",
    "cohort",
    "entry_id",
    "status",  # ok | no_row | not_found
    "net_points",  # points - event_transfers_cost
    "points",
    "hits",  # number of 4-point hits (event_transfers_cost / 4)
    "hit_cost",
    "transfers",
    "total_points",
    "overall_rank",
    "chip",
    "cum_net_points",  # total_points(gw) - total_points at the freeze
    "scored_at",
)
SCORE_KEY = ("freeze_id", "gw", "cohort", "entry_id")
CHIP_NAMES = ("wildcard", "freehit", "bboost", "3xc")
QUANTILES = (("p10", 0.10), ("p25", 0.25), ("median", 0.50), ("p75", 0.75), ("p90", 0.90))
HISTORY_FIELDS = (
    "event",
    "points",
    "total_points",
    "overall_rank",
    "bank",
    "value",
    "event_transfers",
    "event_transfers_cost",
)


class CohortTampered(RuntimeError):
    """The cohort file does not match the sha256 recorded at the freeze."""


# ============================================================================================ freeze
def rng_for(seed: int, cohort: str) -> random.Random:
    digest = hashlib.sha256(f"{int(seed)}:{cohort}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def random_candidates(seed: int, total_players: int) -> Iterator[int]:
    """Distinct entry ids drawn uniformly from 1..total_players, in a fixed order for the seed."""
    rng, seen = rng_for(seed, RANDOM), set()
    while len(seen) < total_players:
        cid = rng.randint(1, int(total_players))
        if cid not in seen:
            seen.add(cid)
            yield cid


def top_positions(seed: int, n: int = COHORT_SIZE, top: int = TOP_N) -> list[int]:
    """n distinct overall-league positions from 1..top (sorted)."""
    return sorted(rng_for(seed, TOP10K).sample(range(1, top + 1), n))


def pages_for(positions: Iterable[int], page_size: int = PAGE_SIZE) -> list[int]:
    return sorted({(int(p) - 1) // page_size + 1 for p in positions})


def last_finished_gw(bootstrap: JSON) -> int:
    done = [int(e["id"]) for e in bootstrap.get("events", []) if e.get("finished")]
    if not done:
        raise ValueError("no finished gameweek yet")
    return max(done)


def compact_history(history: JSON | None) -> JSON | None:
    """The fields the cohorts need from entry/{id}/history/ (no 'past', no names)."""
    if history is None:
        return None
    return {
        "current": [{k: r.get(k) for k in HISTORY_FIELDS} for r in history.get("current", [])],
        "chips": [{"name": c.get("name"), "event": c.get("event")} for c in history.get("chips", [])],
    }


def is_active(history: JSON | None, last_gw: int, n: int = ACTIVE_LAST_N) -> bool:
    """ACTIVE = a history row for each of the last n finished GWs (last_gw - n + 1 .. last_gw)."""
    if not history:
        return False
    events = {int(r["event"]) for r in history.get("current", [])}
    return all(g in events for g in range(last_gw - n + 1, last_gw + 1))


def member_record(entry_id: int, history: JSON | None, freeze_gw: int) -> JSON:
    """The member's state after the freeze GW, read only from rows of GWs <= freeze_gw."""
    rows = {int(r["event"]): r for r in (history or {}).get("current", [])}
    row = rows.get(int(freeze_gw)) or {}
    chips = sorted(
        (
            {"name": c["name"], "gw": int(c["event"])}
            for c in (history or {}).get("chips", [])
            if c.get("event") is not None and int(c["event"]) <= freeze_gw
        ),
        key=lambda c: (c["gw"], c["name"]),
    )
    return {
        "entry_id": int(entry_id),
        "freeze_gw": int(freeze_gw),
        "overall_rank": row.get("overall_rank"),
        "total_points": row.get("total_points"),
        "value": row.get("value"),
        "bank": row.get("bank"),
        "chips_used": chips,
    }


class PoliteSession(requests.Session):
    """A requests session that keeps >= min_interval seconds between requests, cools down after 429/403
    and counts requests by status (the FPL client adds retries with exponential backoff on top)."""

    def __init__(self, min_interval: float = 1.0, cooldown: float = 30.0, sleep: Any = time.sleep) -> None:
        super().__init__()
        self.min_interval = float(min_interval)
        self.cooldown = float(cooldown)
        self._sleep = sleep
        self._last = 0.0
        self.n_requests = 0
        self.by_status: Counter[str] = Counter()

    def get(self, url: str | bytes, **kwargs: Any) -> requests.Response:  # type: ignore[override]
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            self._sleep(wait)
        self.n_requests += 1
        try:
            resp = super().get(url, **kwargs)
        except requests.RequestException:
            self.by_status["error"] += 1
            raise
        finally:
            self._last = time.monotonic()
        self.by_status[str(resp.status_code)] += 1
        if resp.status_code in (403, 429) and self.cooldown > 0:
            self._sleep(self.cooldown)
        return resp


class JsonlCache:
    """Append-only {key: record} cache (one JSON per line); resumable after an interruption."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[str(rec["key"])] = rec["value"]

    def __contains__(self, key: object) -> bool:
        return str(key) in self.data

    def get(self, key: object) -> Any:
        return self.data.get(str(key))

    def put(self, key: object, value: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": str(key), "value": value}, sort_keys=True) + "\n")
        self.data[str(key)] = value


def fetch_history(client: Any, entry_id: int) -> JSON | None:
    """Compact history, or None when the entry does not exist (404)."""
    return compact_history(client.get(f"entry/{int(entry_id)}/history/", allow_404=True))


def freeze_random(
    client: Any,
    seed: int,
    total_players: int,
    freeze_gw: int,
    cache: JsonlCache,
    size: int = COHORT_SIZE,
    max_candidates: int = MAX_RANDOM_CANDIDATES,
    progress: Any = None,
) -> tuple[list[JSON], JSON]:
    """(members, counts) of the random cohort. cache: candidate id -> {"status", "history"}."""
    members: list[JSON] = []
    counts: Counter[str] = Counter()
    drawn = 0
    for cid in random_candidates(seed, total_players):
        if len(members) >= size or drawn >= max_candidates:
            break
        drawn += 1
        rec = cache.get(cid)
        if rec is None:
            hist = fetch_history(client, cid)
            if hist is None:
                rec = {"status": "not_found", "history": None}
            elif is_active(hist, freeze_gw):
                rec = {"status": "ok", "history": hist}
            else:
                rec = {"status": "inactive", "history": None}
            cache.put(cid, rec)
        counts[rec["status"]] += 1
        if rec["status"] == "ok":
            members.append({**member_record(cid, rec["history"], freeze_gw), "draw_index": drawn})
        if progress and drawn % 100 == 0:
            progress(f"random: {drawn} drawn, {len(members)} kept")
    stats = {
        "members": len(members),
        "candidates_drawn": drawn,
        "accepted": counts["ok"],
        "rejected_not_found": counts["not_found"],
        "rejected_inactive": counts["inactive"],
        "acceptance_rate": round(counts["ok"] / drawn, 4) if drawn else None,
        "max_candidates_reached": len(members) < size,
    }
    return members, stats


def freeze_top(
    client: Any,
    seed: int,
    freeze_gw: int,
    page_caches: list[JsonlCache],
    hist_cache: JsonlCache,
    size: int = COHORT_SIZE,
    top: int = TOP_N,
    progress: Any = None,
) -> tuple[list[JSON], JSON]:
    """(members, counts) of the top-10k cohort.

    The overall league's order WITHIN a tie (same points, same rank) changes between page fetches (pages are
    served from caches refreshed at different times), so a position read from one page fetch is not a stable
    identity: one fetch pass of the 200 pages repeated some entries and missed others. Hence: every page of
    the top is fetched once per snapshot (page_caches: one cache per snapshot, page -> rows), the canonical
    list = the distinct entries with rank <= top seen in any snapshot, ordered by (rank, entry id) and cut at
    `top`, and the positions are drawn uniformly from 1..len(canonical list)."""
    pages = list(range(1, (top - 1) // PAGE_SIZE + 2))
    updated: set[str] = set()
    seen: dict[int, JSON] = {}
    inconsistent: set[int] = set()
    for s, cache in enumerate(page_caches, 1):
        for i, page in enumerate(pages, 1):
            if page not in cache:
                body = client.league_classic_standings(OVERALL_LEAGUE, page)
                rows = [
                    {k: r.get(k) for k in ("entry", "rank", "rank_sort", "total")}
                    for r in body["standings"]["results"]
                ]
                cache.put(page, {"rows": rows, "last_updated_data": body.get("last_updated_data")})
            rec = cache.get(page)
            updated.add(str(rec.get("last_updated_data")))
            for r in rec["rows"]:
                eid = int(r["entry"])
                if eid in seen and (seen[eid]["rank"], seen[eid]["total"]) != (r["rank"], r["total"]):
                    inconsistent.add(eid)
                seen.setdefault(eid, {"entry": eid, "rank": int(r["rank"]), "total": r["total"]})
            if progress and i % 50 == 0:
                progress(f"top10k: snapshot {s}, {i}/{len(pages)} pages")
    canonical = sorted((r for r in seen.values() if r["rank"] <= top), key=lambda r: (r["rank"], r["entry"]))
    canonical = canonical[:top]
    if len(canonical) < size:
        raise ValueError(f"only {len(canonical)} distinct top-{top} entries observed")
    positions = top_positions(seed, size, len(canonical))
    members = []
    for n, pos in enumerate(positions, 1):
        st = canonical[pos - 1]
        eid = int(st["entry"])
        if eid not in hist_cache:
            hist_cache.put(eid, fetch_history(client, eid))
        hist = hist_cache.get(eid)
        rec = member_record(eid, hist, freeze_gw)
        if rec["total_points"] is None:  # no history row for the freeze GW: the standings' own numbers
            rec.update(overall_rank=st["rank"], total_points=st["total"])
        members.append({**rec, "position": pos, "rank_at_freeze": st["rank"], "standings_total": st["total"]})
        if progress and n % 100 == 0:
            progress(f"top10k: {n}/{len(positions)} histories")
    stats = {
        "members": len(members),
        "positions_drawn": len(positions),
        "snapshots": len(page_caches),
        "pages_per_snapshot": len(pages),
        "page_requests": len(pages) * len(page_caches),
        "rows_seen": sum(len(c.get(p)["rows"]) for c in page_caches for p in pages),
        "distinct_entries_seen": len(seen),
        "canonical_list_size": len(canonical),
        "entries_rank_or_total_inconsistent": len(inconsistent),
        "distinct_members": len({m["entry_id"] for m in members}),
        "standings_last_updated_data": sorted(updated),
    }
    return members, stats


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cohort_bytes(doc: JSON) -> bytes:
    """Deterministic serialisation: header fields, then one member per line."""
    head = {k: v for k, v in doc.items() if k != "cohorts"}
    lines = ["{", f' "header": {json.dumps(head, sort_keys=True)},', ' "cohorts": {']
    names = list(doc["cohorts"])
    for i, name in enumerate(names):
        ms = doc["cohorts"][name]
        lines.append(f"  {json.dumps(name)}: [")
        lines += [
            "   " + json.dumps(m, sort_keys=True) + ("," if j < len(ms) - 1 else "") for j, m in enumerate(ms)
        ]
        lines.append("  ]" + ("," if i < len(names) - 1 else ""))
    lines += [" }", "}"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def freeze_paths(out_root: str | Path, freeze_id: str) -> tuple[Path, Path]:
    d = Path(out_root) / COHORTS_DIR
    return d / f"cohort_{freeze_id}.json", d / f"cohort_{freeze_id}_manifest.json"


def write_freeze(out_root: str | Path, freeze_id: str, doc: JSON, manifest: JSON) -> tuple[Path, Path, str]:
    """Write the cohort file and its manifest (refuses to overwrite an existing, different cohort file)."""
    cpath, mpath = freeze_paths(out_root, freeze_id)
    data = cohort_bytes(doc)
    if cpath.exists() and cpath.read_bytes() != data:
        raise CohortTampered(f"{cpath} exists with different content: a frozen cohort is never rewritten")
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_bytes(data)
    sha = sha256_bytes(data)
    manifest = {**manifest, "freeze_id": freeze_id, "cohort_file": cpath.name, "sha256": sha}
    mpath.write_bytes((json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    return cpath, mpath, sha


# ============================================================================================ load
def load_frozen(cohorts_dir: str | Path, known: dict[str, str] | None = None) -> list[tuple[JSON, JSON]]:
    """[(manifest, cohort doc)] for every cohort_*_manifest.json in cohorts_dir, verified: raises
    CohortTampered when a cohort file's sha256 differs from its manifest or from `known` (KNOWN_FREEZES)."""
    known = KNOWN_FREEZES if known is None else known
    out = []
    for mpath in sorted(Path(cohorts_dir).glob("cohort_*_manifest.json")):
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        cpath = mpath.parent / manifest["cohort_file"]
        if not cpath.exists():
            raise CohortTampered(f"{cpath.name} is missing")
        raw = cpath.read_bytes()
        sha = sha256_bytes(raw)
        if sha != manifest.get("sha256"):  # a Windows checkout may turn the LF file into CRLF: only that
            sha = sha256_bytes(raw.replace(b"\r\n", b"\n"))
        fid = str(manifest["freeze_id"])
        if sha != manifest.get("sha256"):
            raise CohortTampered(f"{cpath.name}: sha256 {sha} != manifest {manifest.get('sha256')}")
        if fid in known and known[fid] != sha:
            raise CohortTampered(f"{cpath.name}: sha256 {sha} != KNOWN_FREEZES {known[fid]}")
        doc = json.loads(cpath.read_text(encoding="utf-8"))
        out.append((manifest, doc))
    return out


# ============================================================================================ score
def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def finished_gws(bootstrap: JSON) -> list[int]:
    """GWs whose points are final: finished and data_checked."""
    return sorted(
        int(e["id"]) for e in bootstrap.get("events", []) if e.get("finished") and e.get("data_checked")
    )


def read_scores(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or not p.stat().st_size:
        return pd.DataFrame(columns=list(SCORE_COLUMNS))
    return pd.read_csv(p, dtype={"freeze_id": str, "chip": str}, keep_default_na=True).reindex(
        columns=list(SCORE_COLUMNS)
    )


def scored_gws(scores: pd.DataFrame, freeze_id: str) -> set[int]:
    s = scores[scores["freeze_id"].astype(str) == str(freeze_id)]
    return {int(g) for g in s["gw"].dropna().unique()}


def pending_gws(bootstrap: JSON, freeze_gw: int, scored: set[int]) -> list[int]:
    return [g for g in finished_gws(bootstrap) if g > int(freeze_gw) and g not in scored]


def score_rows(
    freeze_id: str,
    cohorts: dict[str, list[JSON]],
    histories: dict[int, JSON | None],
    gws: Iterable[int],
    scored_at: str | None = None,
) -> list[JSON]:
    """One row per (gw, cohort, member) from compact histories (None = entry not found)."""
    scored_at = scored_at or _now()
    rows = []
    for gw in sorted(gws):
        for cohort, members in cohorts.items():
            for m in members:
                eid = int(m["entry_id"])
                base = {"freeze_id": str(freeze_id), "gw": int(gw), "cohort": cohort, "entry_id": eid}
                hist = histories.get(eid)
                if hist is None:
                    rows.append({**base, "status": "not_found", "chip": "", "scored_at": scored_at})
                    continue
                r = next((x for x in hist.get("current", []) if int(x["event"]) == int(gw)), None)
                chip = next((c["name"] for c in hist.get("chips", []) if c.get("event") == gw), "")
                if r is None:
                    rows.append({**base, "status": "no_row", "chip": chip, "scored_at": scored_at})
                    continue
                cost = int(r.get("event_transfers_cost") or 0)
                pts = int(r.get("points") or 0)
                total = r.get("total_points")
                frozen_total = m.get("total_points")
                rows.append(
                    {
                        **base,
                        "status": "ok",
                        "net_points": pts - cost,
                        "points": pts,
                        "hits": cost // 4,
                        "hit_cost": cost,
                        "transfers": r.get("event_transfers"),
                        "total_points": total,
                        "overall_rank": r.get("overall_rank"),
                        "chip": chip,
                        "cum_net_points": (
                            int(total) - int(frozen_total)
                            if total is not None and frozen_total is not None
                            else None
                        ),
                        "scored_at": scored_at,
                    }
                )
    return rows


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """Per (freeze_id, gw, cohort): n members, n scored, mean and quantiles of net points and of cumulative
    net points since the freeze, and the share of members playing each chip (over all n members)."""
    cols = ["freeze_id", "gw", "cohort", "n", "n_scored", "mean"]
    cols += [q for q, _ in QUANTILES]
    cols += [f"share_{c}" for c in CHIP_NAMES]
    cols += ["cum_n_scored", "cum_mean"] + [f"cum_{q}" for q, _ in QUANTILES]
    if scores.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for (fid, gw, cohort), g in scores.groupby(["freeze_id", "gw", "cohort"], sort=True):
        net = pd.to_numeric(g["net_points"], errors="coerce").dropna()
        cum = pd.to_numeric(g["cum_net_points"], errors="coerce").dropna()
        chip = g["chip"].fillna("").astype(str)
        row: JSON = {"freeze_id": fid, "gw": int(gw), "cohort": cohort, "n": len(g), "n_scored": len(net)}
        row["mean"] = round(float(net.mean()), 3) if len(net) else None
        for q, v in QUANTILES:
            row[q] = float(net.quantile(v)) if len(net) else None
        for c in CHIP_NAMES:
            row[f"share_{c}"] = round(float((chip == c).sum()) / len(g), 4)
        row["cum_n_scored"] = len(cum)
        row["cum_mean"] = round(float(cum.mean()), 3) if len(cum) else None
        for q, v in QUANTILES:
            row[f"cum_{q}"] = float(cum.quantile(v)) if len(cum) else None
        out.append(row)
    return pd.DataFrame(out).reindex(columns=cols)


def publish(cohorts_dir: str | Path, rows: list[JSON]) -> tuple[int, pd.DataFrame]:
    """Append rows (new keys only) to cohort_scores.csv and rewrite cohort_summary.csv, each by an atomic
    replace. Returns (rows added, the whole scores table)."""
    d = Path(cohorts_dir)
    old = read_scores(d / SCORES_NAME)
    new = pd.DataFrame(rows).reindex(columns=list(SCORE_COLUMNS))
    seen = set(_keys(old))
    new = new[[k not in seen for k in _keys(new)]]
    if new.empty:
        out = old
    elif old.empty:
        out = new
    else:
        out = pd.concat([old, new], ignore_index=True)
    out = out.sort_values(["freeze_id", "gw", "cohort", "entry_id"], kind="stable").reset_index(drop=True)
    _atomic_csv(out, d / SCORES_NAME)
    _atomic_csv(summarize(out), d / SUMMARY_NAME)
    return len(new), out


def _keys(df: pd.DataFrame) -> list[tuple[str, int, str, int]]:
    if df.empty:
        return []
    return list(
        zip(
            df["freeze_id"].astype(str),
            pd.to_numeric(df["gw"]).astype(int),
            df["cohort"].astype(str),
            pd.to_numeric(df["entry_id"]).astype(int),
            strict=True,
        )
    )


def stage_histories(
    client: Any, entry_ids: Iterable[int], stage_dir: str | Path, progress: Any = None
) -> tuple[dict[int, JSON | None], list[int]]:
    """Fetch (or reuse from stage_dir) every member's compact history. Returns (histories, failed ids);
    the caller publishes only when nothing failed."""
    stage = JsonlCache(Path(stage_dir) / "histories.jsonl")
    out: dict[int, JSON | None] = {}
    failed = []
    ids = sorted({int(e) for e in entry_ids})
    for n, eid in enumerate(ids, 1):
        if eid not in stage:
            try:
                stage.put(eid, fetch_history(client, eid))
            except Exception:  # noqa: BLE001 - retried by the next run; nothing is published meanwhile
                failed.append(eid)
                continue
        out[eid] = stage.get(eid)
        if progress and n % 250 == 0:
            progress(f"{n}/{len(ids)} histories")
    return out, failed


# ============================================================================================ read
def load_scores(source: str | Path | pd.DataFrame | None = None) -> pd.DataFrame:
    """The published scores (default: the spike-log raw URL), or a local path, or a frame as is."""
    if isinstance(source, pd.DataFrame):
        return source
    src = str(source or SCORES_URL)
    return pd.read_csv(src, dtype={"freeze_id": str, "chip": str})


def percentile_of(
    points: float,
    gw: int,
    cohort: str,
    cumulative: bool = False,
    *,
    scores: str | Path | pd.DataFrame | None = None,
    freeze_id: str | None = None,
) -> float:
    """Percentile (0-100, mid-rank for ties) of `points` WITHIN the frozen cohort for GW gw: net GW points, or
    cumulative net points since the freeze when cumulative=True. Members without points for the GW
    (no_row / not_found) are not in the denominator. freeze_id defaults to the latest freeze in the scores.
    This is a within-cohort percentile, never an overall FPL percentile."""
    df = load_scores(scores)
    if freeze_id is None:
        if df.empty:
            raise ValueError("no cohort scores published yet")
        freeze_id = str(sorted(df["freeze_id"].astype(str).unique())[-1])
    g = df[
        (df["freeze_id"].astype(str) == str(freeze_id))
        & (pd.to_numeric(df["gw"]) == int(gw))
        & (df["cohort"].astype(str) == cohort)
    ]
    vals = pd.to_numeric(g["cum_net_points" if cumulative else "net_points"], errors="coerce").dropna()
    if vals.empty:
        raise ValueError(f"no scores for freeze {freeze_id}, GW{gw}, cohort {cohort}")
    below = float((vals < points).sum())
    equal = float((vals == points).sum())
    return round(100.0 * (below + 0.5 * equal) / len(vals), 2)
