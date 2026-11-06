"""Reconcile rules-engine points with the API's total_points for one gameweek."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .rules import get_rules


@dataclass(frozen=True)
class Mismatch:
    element: int
    element_type: int
    api_total: int
    computed_total: int
    breakdown: dict[str, int]
    api_explain: list[dict[str, Any]]


def reconcile_gw(
    live: dict[str, Any], bootstrap: dict[str, Any], season: str = "2026-27"
) -> tuple[int, list[Mismatch]]:
    """Recompute every player's points from `live` stats. Returns (players checked, mismatches)."""
    rules = get_rules(season)
    element_type = {p["id"]: p["element_type"] for p in bootstrap["elements"]}
    mismatches: list[Mismatch] = []
    checked = 0
    for el in live["elements"]:
        et = element_type.get(el["id"])
        if et is None:  # player removed from the game after this GW; cannot classify
            continue
        checked += 1
        breakdown = rules.points_from_live(el, et)
        api_total = int(el["stats"]["total_points"])
        if breakdown["total"] != api_total:
            mismatches.append(
                Mismatch(el["id"], et, api_total, breakdown["total"], breakdown, el.get("explain", []))
            )
    return checked, mismatches


# ================================================================================ archive, every season
ARCHIVE_STATS = (
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "defensive_contribution",
)


def reconcile_archive(df: Any, season: str, gws: Any = None) -> Any:
    """Recompute every per-fixture archive row (fpl_rocket.archive schema) of `season` with that season's
    scoring table and compare with the row's total_points. Returns a frame: season, gw, element, fixture,
    element_type, archive, computed, diff (all rows; filter diff != 0 for mismatches)."""

    rules = get_rules(season)
    s = df[(df["season"] == season) & (df["element_type"].isin((1, 2, 3, 4)))]
    if gws is not None:
        s = s[s["gw"].isin(list(gws))]
    stats = s[[c for c in ARCHIVE_STATS if c in s]].astype("Float64").fillna(0).astype(int)
    et = s["element_type"].astype(int).to_numpy()
    computed = [
        rules.points_from_stats(row, int(t))["total"]
        for row, t in zip(stats.to_dict("records"), et, strict=True)
    ]
    out = s[["season", "gw", "element", "fixture", "element_type"]].copy()
    out["archive"] = s["total_points"].astype("Float64").fillna(0).astype(int).to_numpy()
    out["computed"] = computed
    out["diff"] = out["computed"] - out["archive"]
    return out.reset_index(drop=True)
