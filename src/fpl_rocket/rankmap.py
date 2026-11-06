"""Real final-rank mapping: season total points -> overall rank, from real managers' past seasons.

Source: entry/{id}/history/ `past` = one row per earlier season the manager played (season_name, total_points,
rank = the FINAL overall rank). Only those three numbers and the entry id are kept: no names, no regions.

Entries fetched (scripts/rank_map.py fetch; cache data/raw/rankmap/histories.jsonl, append-only, resumable,
>= 1 s between requests):
    cohort   the members of the frozen prospective cohorts (fpl_rocket.cohorts; random + top10k, freeze
             20261106). Their cached histories (data/raw/cohorts) were compacted WITHOUT `past`, so each
             member
             is fetched once more for it.
    random   extra entry ids drawn uniformly without replacement from 1..total_players at the freeze with a
             seeded generator (seed scheme below), cohort ids excluded; a 404 is recorded as not_found.
The top10k cohort (top 10,000 of the 2026-27 overall league at the freeze) is a SELECTED sample: its past
seasons concentrate on good ranks. That does not bias the points -> rank mapping (every pair is an exact
point of the official ranking of that season), it only decides where the anchors are dense.

Mapping per season: pairs (points, rank) -> monotonic consistency check (a higher total must rank strictly
better than every lower total; equal totals may carry different ranks because FPL breaks ties, counted
separately) -> isotonic fit of log10(rank) on points (decreasing; one anchor per total at the middle of its
tie range in log10), evaluated by linear interpolation between the fitted anchors. place(): estimated rank of
a season total, the band = the ranks bounding it from the nearest observed anchor above (better) and below
(worse) - within the tie range when the total itself was observed - and the percentile rank / entrants.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
# Entrants per season = bootstrap-static `total_players` in the last archived snapshot after GW38 (community
# bootstrap archive data/raw/fplcache, file named below); identical to the season totals published by
# allaboutfpl.com ("How Many Play FPL?", 2026-07: 9.17m, 11.45m, 10.91m = 10,905,021, 11.50m, 13.1m).
ENTRANTS = {
    "2021-22": (9_167_407, "fplcache 2022/7/5/1248 bootstrap total_players"),
    "2022-23": (11_447_257, "fplcache 2023/7/5/1244 bootstrap total_players"),
    "2023-24": (10_905_021, "fplcache 2024/7/17/0627 bootstrap total_players"),
    "2024-25": (11_501_826, "fplcache 2025/7/20/1835 bootstrap total_players"),
    "2025-26": (13_107_732, "fplcache 2026/7/23/0331 bootstrap total_players"),
}
INTEREST = (2100, 2450)
RANDOM_SEED_SCHEME = "random.Random(int(sha256(f'{seed}:rankmap').hexdigest()[:16], 16))"
CACHE_NAME = "histories.jsonl"


def season_key(season_name: str) -> str:
    """'2023/24' (FPL `past`) -> '2023-24'."""
    s = str(season_name).strip().replace("/", "-")
    return s


def compact_past(history: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """[{season, points, rank}] from an entry/{id}/history/ body (None when the entry does not exist)."""
    if history is None:
        return None
    out = []
    for p in history.get("past", []) or []:
        if p.get("total_points") is None or p.get("rank") is None:
            continue
        out.append(
            {"season": season_key(p["season_name"]), "points": int(p["total_points"]), "rank": int(p["rank"])}
        )
    return out


def random_ids(seed: int, total_players: int, n: int, exclude: Iterable[int] = ()) -> list[int]:
    """n distinct ids from 1..total_players in a fixed order for the seed, skipping `exclude`."""
    digest = hashlib.sha256(f"{int(seed)}:rankmap".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    skip, out, seen = set(int(e) for e in exclude), [], set()
    while len(out) < n and len(seen) < total_players:
        cid = rng.randint(1, int(total_players))
        if cid in seen:
            continue
        seen.add(cid)
        if cid not in skip:
            out.append(cid)
    return out


def fetch(
    client: Any,
    ids: Iterable[int],
    cache: Any,
    source: str,
    *,
    log: Callable[[str], None] = print,
    max_consecutive_errors: int = 20,
) -> dict[str, int]:
    """Fetch the `past` of every id not yet cached (cache: cohorts.JsonlCache). Errors are counted and the id
    is left uncached (a later run retries it); too many consecutive errors stop the run."""
    counts = {"cached": 0, "ok": 0, "not_found": 0, "error": 0}
    streak = 0
    todo = [int(e) for e in ids]
    for i, eid in enumerate(todo, 1):
        if eid in cache:
            counts["cached"] += 1
            continue
        try:
            body = client.get(f"entry/{eid}/history/", allow_404=True)
        except Exception as exc:  # noqa: BLE001 - logged, retried by the next run
            counts["error"] += 1
            streak += 1
            log(f"{source} {eid}: {exc!r}")
            if streak >= max_consecutive_errors:
                log(f"{streak} consecutive errors: stopping (resume later)")
                break
            continue
        streak = 0
        past = compact_past(body)
        status = "not_found" if past is None else "ok"
        cache.put(eid, {"status": status, "source": source, "past": past or []})
        counts[status] += 1
        if i % 200 == 0:
            log(f"{source}: {i}/{len(todo)} {counts}")
    return counts


def pairs_from_cache(records: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, rec in records.items():
        if not rec or rec.get("status") != "ok":
            continue
        for p in rec.get("past", []):
            rows.append(
                {
                    "entry": int(key),
                    "source": rec.get("source", ""),
                    "season": p["season"],
                    "points": p["points"],
                    "rank": p["rank"],
                }
            )
    return pd.DataFrame(rows, columns=["entry", "source", "season", "points", "rank"])


def monotonic_violations(pairs: pd.DataFrame) -> int:
    """Inversions between neighbouring points totals: a total whose best rank is not worse than the worst
    rank of the next higher total. Equal totals with different ranks are NOT violations (FPL breaks ties:
    tie_groups counts them)."""
    g = pairs.groupby("points")["rank"].agg(["min", "max"]).sort_index(ascending=False)
    return int((g["min"].to_numpy()[1:] <= g["max"].to_numpy()[:-1]).sum())


def tie_groups(pairs: pd.DataFrame) -> int:
    """Points totals observed with more than one distinct final rank (ties broken by FPL)."""
    return int((pairs.groupby("points")["rank"].nunique() > 1).sum())


@dataclass
class RankCurve:
    season: str
    points: np.ndarray  # increasing anchor points
    log_rank: np.ndarray  # fitted log10(rank) at the anchors (non-increasing)
    raw: pd.DataFrame  # the observed pairs (points, rank) of the season, deduplicated
    violations: int

    @property
    def n_pairs(self) -> int:
        return int(self.raw["n"].sum())

    def rank_at(self, pts: float) -> float:
        return float(10 ** np.interp(pts, self.points, self.log_rank))

    def place(self, pts: float, entrants: float | None = None) -> dict[str, Any]:
        """Band: a total below an anchor ranks worse than that anchor's worst rank; a total above an anchor
        ranks better than its best rank; a total equal to an anchor lies within the anchor's tie range."""
        r = self.raw
        above = r[r["points"] >= pts].sort_values("points").head(1)  # nearest anchor with >= points
        below = r[r["points"] <= pts].sort_values("points").tail(1)  # nearest anchor with <= points
        best = worst = None
        if len(above):
            a = above.iloc[0]
            best = int(a["rank"] if a["points"] == pts else a["rank_max"])
        if len(below):
            b = below.iloc[0]
            worst = int(b["rank_max"] if b["points"] == pts else b["rank"])
        out: dict[str, Any] = {
            "season": self.season,
            "points": pts,
            "in_range": bool(len(above) and len(below)),
            "anchor_above_points": int(above["points"].iloc[0]) if len(above) else None,
            "rank_best": best if best is not None else 1,
            "anchor_below_points": int(below["points"].iloc[0]) if len(below) else None,
            "rank_worst": worst,
            "anchors_within_25": int(r.loc[(r["points"] - pts).abs() <= 25, "n"].sum()),
        }
        out["rank_est"] = int(round(self.rank_at(pts))) if out["in_range"] else None
        if entrants:
            for k in ("rank_est", "rank_best", "rank_worst"):
                v = out.get(k)
                out[k.replace("rank", "pct")] = None if v is None else 100.0 * v / float(entrants)
        return out


def fit_curve(pairs: pd.DataFrame, season: str) -> RankCurve:
    from sklearn.isotonic import IsotonicRegression

    d = pairs[pairs["season"] == season]
    if d.empty:
        raise ValueError(f"no pairs for {season}")
    viol = monotonic_violations(d)
    g = (
        d.groupby("points", as_index=False)
        .agg(rank=("rank", "min"), rank_max=("rank", "max"), n=("rank", "size"))
        .sort_values("points")
        .reset_index(drop=True)
    )
    x = g["points"].to_numpy(float)
    y = 0.5 * (np.log10(g["rank"].to_numpy(float)) + np.log10(g["rank_max"].to_numpy(float)))
    iso = IsotonicRegression(increasing=False).fit(x, y, sample_weight=g["n"].to_numpy(float))
    return RankCurve(season, x, iso.predict(x), g, viol)


def load_cache(path: str | Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[str(rec["key"])] = rec["value"]
    return out
