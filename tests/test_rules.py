"""Scoring rules: unit cases + full reconciliation of a recorded gameweek."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fpl_rocket.reconcile import reconcile_gw
from fpl_rocket.rules import get_rules
from fpl_rocket.rules.season_2026_27 import (
    DEF,
    FWD,
    GK,
    MID,
    points_from_live,
    points_from_stats,
    selling_price,
    transfer_rules_from_bootstrap,
)

FIX = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_recorded_gw7_reconciles_to_zero_mismatches():
    live, bootstrap = _load("event_7_live.json"), _load("bootstrap_trim.json")
    checked, mismatches = reconcile_gw(live, bootstrap)
    assert checked >= 600
    assert mismatches == [], mismatches[:5]


@pytest.mark.parametrize(
    "stats,et,total",
    [
        ({"minutes": 0, "bonus": 0}, MID, 0),  # no appearance, nothing
        ({"minutes": 59, "clean_sheets": 1}, DEF, 1),  # CS needs 60 minutes
        ({"minutes": 60, "clean_sheets": 1}, DEF, 6),
        ({"minutes": 90, "clean_sheets": 1}, MID, 3),
        ({"minutes": 90, "clean_sheets": 1}, FWD, 2),
        ({"minutes": 90, "goals_scored": 1}, GK, 8),
        ({"minutes": 90, "goals_scored": 2, "assists": 1}, FWD, 2 + 8 + 3),
        ({"minutes": 90, "saves": 8, "goals_conceded": 5, "penalties_saved": 1}, GK, 2 + 2 - 2 + 5),
        ({"minutes": 90, "goals_conceded": 3, "yellow_cards": 1, "own_goals": 1}, DEF, 2 - 1 - 1 - 2),
        ({"minutes": 30, "red_cards": 1, "penalties_missed": 1}, FWD, 1 - 3 - 2),
        ({"minutes": 90, "bonus": 3}, MID, 5),
        ({"minutes": 90, "defensive_contribution": 10}, DEF, 4),  # DEF threshold 10
        ({"minutes": 90, "defensive_contribution": 9}, DEF, 2),
        ({"minutes": 90, "defensive_contribution": 11}, MID, 2),  # MID threshold 12
        ({"minutes": 90, "defensive_contribution": 12}, FWD, 4),
        ({"minutes": 90, "defensive_contribution": 20}, GK, 2),  # never for GK
        ({"minutes": 90, "goals_conceded": 4}, MID, 2),  # conceded only hurts GK/DEF
    ],
)
def test_points_from_stats(stats, et, total):
    assert points_from_stats(stats, et)["total"] == total


def test_double_gameweek_scored_per_fixture():
    # two fixtures with 1 goal conceded each: aggregate 2 would cost -1, per fixture it costs nothing
    el = {
        "stats": {"minutes": 180, "goals_conceded": 2, "total_points": 4},
        "explain": [
            {"fixture": 1, "stats": [{"identifier": "minutes", "value": 90, "points": 2}]},
            {"fixture": 2, "stats": [{"identifier": "minutes", "value": 90, "points": 2}]},
        ],
    }
    assert points_from_live(el, DEF)["total"] == 4


def test_selling_price():
    assert selling_price(50, 55) == 52
    assert selling_price(50, 51) == 50
    assert selling_price(50, 48) == 48
    assert selling_price(50, 50) == 50


def test_rules_registry_and_bootstrap_settings():
    assert get_rules("2026_27").SEASON == "2026-27"
    with pytest.raises(ValueError):
        get_rules("1999-00")
    tr = transfer_rules_from_bootstrap(_load("bootstrap_trim.json"))
    assert tr.max_bank == 5 and tr.transfers_cap == 20 and tr.sell_on_fee == 0.5
    windows = get_rules().chip_windows(_load("bootstrap_trim.json"))
    assert sum(w["name"] == "wildcard" for w in windows) == 2
