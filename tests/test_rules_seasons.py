"""Season rules 2021-22 .. 2026-27: chip windows, free-transfer caps and chip-week behaviour, special events,
scoring differences between seasons."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_rocket.rules import SEASONS, get_rules, season_rules
from fpl_rocket.rules.common import DEF, FWD, GK, MID


def _windows(season: str) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    for c in season_rules(season).chips:
        out.setdefault(c.name, []).append((c.start_event, c.stop_event))
    return {k: sorted(v) for k, v in out.items()}


def test_every_season_has_the_same_interface():
    assert SEASONS == ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27")
    names = [
        "SEASON",
        "RULES_VERSION",
        "SCORING",
        "TRANSFERS",
        "SQUAD",
        "CHIPS",
        "SPECIAL",
        "RULES",
        "SOURCES",
        "KNOWN_UNCERTAIN",
        "points_from_stats",
        "points_from_live",
        "fixture_stats_from_explain",
        "selling_price",
        "chip_windows",
        "transfer_rules_from_bootstrap",
    ]
    for s in SEASONS:
        m = get_rules(s)
        missing = [n for n in names if not hasattr(m, n)]
        assert not missing, (s, missing)
        assert m.RULES.season == s


@pytest.mark.parametrize(
    "season,expected",
    [
        (
            "2021-22",
            {
                "wildcard": [(2, 20), (21, 38)],
                "freehit": [(2, 38), (20, 38)],
                "bboost": [(1, 38)],
                "3xc": [(1, 38)],
            },
        ),
        (
            "2022-23",
            {"wildcard": [(2, 16), (18, 38)], "freehit": [(2, 38)], "bboost": [(1, 38)], "3xc": [(1, 38)]},
        ),
        (
            "2023-24",
            {"wildcard": [(2, 20), (21, 38)], "freehit": [(2, 38)], "bboost": [(1, 38)], "3xc": [(1, 38)]},
        ),
        (
            "2024-25",
            {
                "wildcard": [(2, 19), (20, 38)],
                "freehit": [(2, 38)],
                "bboost": [(1, 38)],
                "3xc": [(1, 38)],
                "manager": [(24, 38)],
            },
        ),
        (
            "2025-26",
            {
                "wildcard": [(2, 19), (20, 38)],
                "freehit": [(2, 19), (20, 38)],
                "bboost": [(1, 19), (20, 38)],
                "3xc": [(1, 19), (20, 38)],
            },
        ),
    ],
)
def test_chip_windows(season, expected):
    assert _windows(season) == expected


def test_assistant_manager_unsupported_and_extra_free_hit_announced_mid_season():
    am = [c for c in season_rules("2024-25").chips if c.name == "manager"]
    assert am and not am[0].supported
    r = season_rules("2021-22")
    before = r.chips_known_at(pd.Timestamp("2021-12-01", tz="UTC"))
    after = r.chips_known_at(pd.Timestamp("2021-12-28", tz="UTC"))
    assert sum(c.name == "freehit" for c in before) == 1
    assert sum(c.name == "freehit" for c in after) == 2


@pytest.mark.parametrize(
    "season,cap,policy",
    [
        ("2021-22", 2, "reset_one"),
        ("2022-23", 2, "reset_one"),
        ("2023-24", 2, "reset_one"),
        ("2024-25", 5, "carry"),
        ("2025-26", 5, "carry"),
        ("2026-27", 5, "carry"),
    ],
)
def test_free_transfer_caps_and_chip_weeks(season, cap, policy):
    r = season_rules(season)
    assert r.transfers.max_bank == cap and r.transfers.chip_week_policy == policy
    f = 1
    for gw in range(2, 12):  # never transfer: the balance grows to the cap
        f = r.next_free_transfers(gw, f, 0, None)
    assert f == cap
    assert r.next_free_transfers(1, 1, 15, None) == 1  # GW1: unlimited, 1 entering GW2
    assert r.next_free_transfers(5, cap, 0, "wildcard") == (1 if policy == "reset_one" else cap)
    assert r.next_free_transfers(5, cap, 7, "freehit") == (1 if policy == "reset_one" else cap)
    assert r.hits(5, 1, 3, None) == 2 and r.hits(5, 1, 9, "wildcard") == 0 and r.hits(1, 1, 15, None) == 0
    assert r.next_free_transfers(5, 2, 3, None) == 1  # 2 free, 3 made (1 hit) -> 0 + 1


def test_world_cup_break_2022_23():
    r = season_rules("2022-23")
    assert r.free_transfer_week(17, None) and r.hits(17, 1, 11, None) == 0
    assert (
        not r.chip_allowed("wildcard", 17) and r.chip_allowed("freehit", 17) and r.chip_allowed("bboost", 17)
    )
    assert r.next_free_transfers(17, 1, 0, None) == 2  # no transfer in the break: 2 entering GW18
    assert r.next_free_transfers(17, 2, 6, None) == 1
    assert 7 in r.special.void_gws  # Queen's death: GW7 postponed
    assert season_rules("2023-24").free_transfer_week(17, None) is False


def test_afcon_top_up_2025_26():
    r = season_rules("2025-26")
    assert r.next_free_transfers(15, 1, 1, None) == 5  # topped up to 5 entering GW16
    assert r.next_free_transfers(14, 1, 1, None) == 1
    assert season_rules("2024-25").next_free_transfers(15, 1, 1, None) == 1


def test_scoring_by_season():
    gk_goal = {"minutes": 90, "goals_scored": 1}
    assert get_rules("2023-24").points_from_stats(gk_goal, GK)["total"] == 8  # 6 + 2
    assert get_rules("2024-25").points_from_stats(gk_goal, GK)["total"] == 12  # 10 + 2
    dc = {"minutes": 90, "defensive_contribution": 12}
    for s in ("2021-22", "2022-23", "2023-24", "2024-25"):
        assert get_rules(s).points_from_stats(dc, MID)["total"] == 2, s
    for s in ("2025-26", "2026-27"):
        assert get_rules(s).points_from_stats(dc, MID)["total"] == 4, s
        assert (
            get_rules(s).points_from_stats({"minutes": 90, "defensive_contribution": 10}, DEF)["total"] == 4
        )
    # a booking of an unused substitute counts (archive reconciliation)
    for s in SEASONS:
        assert get_rules(s).points_from_stats({"minutes": 0, "yellow_cards": 1}, FWD)["total"] == -1
        assert get_rules(s).points_from_stats({"minutes": 0, "bonus": 0}, FWD)["total"] == 0


def test_selling_price_same_rule_every_season():
    for s in SEASONS:
        m = get_rules(s)
        assert (
            m.selling_price(50, 55) == 52 and m.selling_price(50, 48) == 48 and m.selling_price(50, 51) == 50
        )
