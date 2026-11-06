"""Sampled real managers by rank tier from github.com/sertalpbilal/fpl_optimized (branch `webpage`,
sample/<season>/<gw>/fpl_sampled.json; seasons 2023-24, 2024-25, 2026-27 - 2025-26 is absent from the tree).

How the sample is drawn (src/collect.py `sample_fpl_teams`, branch `engine`, read 2026-11-06):
    "Overall": `random.sample(range(1, max(total_players, 4000000)), 666)` entry ids, uniformly over the id
        space (ids that fail to load are dropped, so fewer than 666 remain);
    tiers 100 / 1000 / 10000 / 100000 / 1000000 (from GW2 only: `if int(gw) != 1`): `pairs = [[100, 80],
        [1000, 278], [10000, 370], [100000, 383], [1000000, 385]]`, `random.sample(range(1, target+1),
        nsample)` RANKS, each rank resolved to an entry through the overall league 314 standings page
        `((rank-1)//50)+1` AT THE TIME THE SCRIPT RUNS; tiers are drawn independently, so they are nested by
        definition (top 100 is inside top 1000) and an entry can appear in two tiers by chance;
    named mini-leagues (all members): the keys vary by season.
The picks are `entry/{id}/event/{gw}/picks/` (revealed after the GW deadline). The rank is the live overall
rank
when the job runs (manual `workflow_dispatch` with a GW input), NOT the final rank - but a file can be
re-committed by a later re-run (e.g. a 2024-12-10 run rewrote 2024-25 GW1 and GW10 with December ranks), so
the
version matters: the COMPARATOR uses the earliest committed version of each GW file; the LAGGED tables use the
last version committed at or before the issue time T of the NEXT gameweek (the only one a policy could have
read).

Privacy: only the entry id and the picks are kept (no team names, no regions). Untrusted data: json only.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

SOURCE = "fpl_optimized"
REPO_URL = "https://github.com/sertalpbilal/fpl_optimized"
BRANCH = "webpage"
RANK_TIERS = ("100", "1000", "10000", "100000", "1000000")


def parse_sample(raw: bytes | str, season: str, gw: int, version_at) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One fpl_sampled.json -> (managers, picks)."""
    data: dict[str, list[dict[str, Any]]] = json.loads(raw)
    mrows, prows = [], []
    for tier, entries in data.items():
        for e in entries or []:
            if not isinstance(e, dict) or not isinstance(e.get("data"), dict):
                continue
            team = e.get("team") or {}
            d = e["data"]
            picks = d.get("picks") or []
            if not picks:
                continue
            hist = d.get("entry_history") or {}
            entry = team.get("id")
            cap = next((p["element"] for p in picks if p.get("is_captain")), None)
            vice = next((p["element"] for p in picks if p.get("is_vice_captain")), None)
            bench = [
                p["element"]
                for p in sorted(picks, key=lambda p: p.get("position", 0))
                if p.get("position", 0) > 11
            ]
            mrows.append(
                {
                    "season": season,
                    "gw": gw,
                    "tier": str(tier),
                    "tier_kind": "rank"
                    if str(tier) in RANK_TIERS
                    else ("random" if tier == "Overall" else "league"),
                    "entry": entry,
                    "chip": d.get("active_chip"),
                    "captain": cap,
                    "vice_captain": vice,
                    "bench_order": json.dumps(bench),
                    "gw_points": hist.get("points"),
                    "gw_overall_rank": hist.get("overall_rank"),  # rank after this GW
                    "rank_at_fetch": team.get("summary_overall_rank"),  # rank when the sampler ran
                    "event_transfers": hist.get("event_transfers"),
                    "event_transfers_cost": hist.get("event_transfers_cost"),
                    "version_committed_at": version_at,
                }
            )
            for p in picks:
                prows.append(
                    {
                        "season": season,
                        "gw": gw,
                        "tier": str(tier),
                        "entry": entry,
                        "element": p.get("element"),
                        "position": p.get("position"),
                        "multiplier": p.get("multiplier"),
                        "is_captain": bool(p.get("is_captain")),
                        "is_vice_captain": bool(p.get("is_vice_captain")),
                    }
                )
    m = pd.DataFrame(mrows)
    p = pd.DataFrame(prows)
    for c in (
        "entry",
        "captain",
        "vice_captain",
        "gw_points",
        "gw_overall_rank",
        "rank_at_fetch",
        "event_transfers",
        "event_transfers_cost",
    ):
        if c in m:
            m[c] = pd.to_numeric(m[c], errors="coerce").astype("Int64")
    for c in ("entry", "element", "position", "multiplier"):
        if c in p:
            p[c] = pd.to_numeric(p[c], errors="coerce").astype("Int64")
    return m, p


def aggregate(managers: pd.DataFrame, picks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per player: ownership / starting / captaincy share, per tier: managers and chip shares) per
    (season, gw,
    tier)."""
    keys = ["season", "gw", "tier"]
    n = managers.groupby(keys).size().rename("managers")
    pk = picks.assign(started=picks["multiplier"].fillna(0) > 0)
    g = pk.groupby([*keys, "element"])
    pl = pd.DataFrame(
        {
            "owners": g["entry"].nunique(),
            "starters": g["started"].sum(),
            "captains": g["is_captain"].sum(),
            "vice_captains": g["is_vice_captain"].sum(),
        }
    ).reset_index()
    pl = pl.merge(n.reset_index(), on=keys)
    pl["ownership"] = pl["owners"] / pl["managers"]
    pl["starting_share"] = pl["starters"] / pl["managers"]
    pl["captaincy_share"] = pl["captains"] / pl["managers"]
    chips = managers.assign(chip=managers["chip"].fillna("none")).pivot_table(
        index=keys, columns="chip", values="entry", aggfunc="count", fill_value=0
    )
    chips = chips.div(n, axis=0).add_prefix("chip_share_")
    tiers = n.to_frame().join(chips).reset_index()
    return pl, tiers


def lag(df: pd.DataFrame) -> pd.DataFrame:
    """Shift a per-GW table to the NEXT gameweek: row gw g describes source_gw g-1 (usable at T of g)."""
    out = df.rename(columns={"gw": "source_gw"}).copy()
    out.insert(1, "gw", out["source_gw"] + 1)
    return out


def tier_overlap(managers: pd.DataFrame) -> pd.DataFrame:
    """Entries present in more than one tier of the same (season, gw)."""
    m = managers.dropna(subset=["entry"])
    c = m.groupby(["season", "gw", "entry"])["tier"].nunique()
    return (
        c[c > 1]
        .reset_index(name="tiers")
        .groupby(["season", "gw"])
        .size()
        .rename("entries_in_2plus_tiers")
        .reset_index()
    )
