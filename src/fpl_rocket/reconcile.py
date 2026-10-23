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
