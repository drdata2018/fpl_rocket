"""Pure page builders (fpl_rocket.views): synthetic inputs, including blank and double gameweeks."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from fpl_rocket import views
from fpl_rocket.data import parse_team_id

TEAMS = [
    {"id": 1, "name": "Arsenal", "short_name": "ARS"},
    {"id": 2, "name": "Bournemouth", "short_name": "BOU"},
    {"id": 3, "name": "Chelsea", "short_name": "CHE"},
    {"id": 4, "name": "Dover", "short_name": "DOV"},
]


def _el(pid, team, et, cost=50, own="1.0", pts=0, **kw):
    return {
        "id": pid,
        "web_name": f"P{pid}",
        "team": team,
        "element_type": et,
        "now_cost": cost,
        "selected_by_percent": own,
        "total_points": pts,
        "status": kw.pop("status", "a"),
        "news": kw.pop("news", ""),
        **kw,
    }


def _boot(next_gw=3, elements=None):
    events = [
        {
            "id": g,
            "finished": g < next_gw,
            "is_next": g == next_gw,
            "average_entry_score": 50 + g,
            "highest_score": 100,
        }
        for g in range(1, 39)
    ]
    return {"teams": TEAMS, "events": events, "elements": elements or [], "total_players": 1000}


def _fx(fid, gw, h, a, hs=None, as_=None, hd=2, ad=4):
    return {
        "id": fid,
        "event": gw,
        "team_h": h,
        "team_a": a,
        "team_h_score": hs,
        "team_a_score": as_,
        "finished": hs is not None,
        "team_h_difficulty": hd,
        "team_a_difficulty": ad,
        "kickoff_time": f"2026-08-{10 + fid:02d}T15:00:00Z",
    }


# ------------------------------------------------------------------------------------------------ standings
def test_standings_points_gd_and_order():
    fx = [
        _fx(1, 1, 1, 2, 3, 0),
        _fx(2, 1, 3, 4, 1, 1),
        _fx(3, 2, 2, 3, 2, 1),
        _fx(4, 3, 1, 3),
    ]  # last unplayed
    t = views.standings(fx, _boot())
    ars = t[t["Team"] == "Arsenal"].iloc[0]
    assert (ars["P"], ars["W"], ars["GF"], ars["GA"], ars["GD"], ars["Pts"]) == (1, 1, 3, 0, 3, 3)
    assert list(t["Team"])[:2] == ["Arsenal", "Bournemouth"]  # 3 pts GD+3 vs 3 pts GD-2
    assert t["P"].sum() == 6 and list(t["Pos"]) == [1, 2, 3, 4]


# ------------------------------------------------------------------------------------------------ fixtures
def test_fixture_grid_double_and_blank():
    fx = [_fx(1, 3, 1, 2), _fx(2, 3, 1, 3, hd=4, ad=2), _fx(3, 4, 2, 3)]  # ARS double in GW3, blank GW4
    text, diff = views.fixture_grid(fx, _boot(), 3, 2)
    assert text.loc["Arsenal", "GW3"] == "BOU(H) + CHE(H)"
    assert diff.loc["Arsenal", "GW3"] == pytest.approx(3.0)  # mean of 2 and 4
    assert text.loc["Arsenal", "GW4"] == "-" and math.isnan(diff.loc["Arsenal", "GW4"])
    assert text.loc["Dover", "GW3"] == "-"
    assert list(text.columns) == ["GW3", "GW4"]


def test_fixture_strip_text_marks_blanks():
    fx = [_fx(1, 3, 1, 2), _fx(2, 5, 3, 1, hd=3, ad=5)]
    by_team = views.fixtures_by_team(fx, 3, 5)
    s = views.fixture_strip(1, by_team, [3, 4, 5], views.team_short(_boot()))
    assert s == "BOU(H) 2 · GW4 blank · CHE(A) 5"


# ------------------------------------------------------------------------------------------------ players
def _players():
    els = [
        _el(1, 1, 1, 45, "30.5", 40),
        _el(2, 1, 2, 55, "12.0", 35, cost_change_event=1, transfers_in_event=500, transfers_out_event=100),
        _el(3, 2, 3, 80, "5.5", 50, status="i", news="Knee injury", news_added="2026-10-01T10:00:00Z"),
        _el(4, 3, 4, 100, "2.0", 20, cost_change_event=-1),
        _el(5, 4, 3, 60, "9.9", 60, news="Late fitness test", news_added="2026-10-05T10:00:00Z"),
    ]
    boot = _boot(elements=els)
    return views.players_frame(boot), boot


def test_players_frame_units():
    p, _ = _players()
    r = p.set_index("id").loc[2]
    assert r["price"] == 5.5 and r["ownership"] == 12.0 and r["pos"] == "DEF" and r["team"] == "ARS"
    assert r["cost_change_event"] == pytest.approx(0.1) and r["net_transfers_event"] == 400


def test_price_changes_availability_template():
    p, _ = _players()
    risers, fallers = views.price_changes(p)
    assert list(risers["name"]) == ["P2"] and list(fallers["name"]) == ["P4"]
    av = views.availability(p)
    assert list(av["name"]) == ["P5", "P3"]  # newest news first; status 'a' with news is included
    assert av.iloc[1]["status_label"] == "Injured"
    tpl = views.template_squad(p)
    assert list(tpl["pos"]) == ["GKP", "DEF", "MID", "MID", "FWD"]  # fewer players than the shape


def test_leaders_position_filter():
    p, _ = _players()
    top = views.leaders(p, "Points", "MID")
    assert list(top["name"]) == ["P5", "P3"]


def test_differentials_threshold_filters_and_sort():
    p, boot = _players()
    live = views.live_frame(
        {g: {"elements": [{"id": 5, "stats": {"minutes": 90, "total_points": 6}}]} for g in (1, 2)}
    )
    fx = [_fx(1, 3, 4, 1), _fx(2, 4, 2, 4)]
    d = views.differentials(p, live, fx, boot, max_ownership=10.0)
    assert list(d["name"]) == ["P5", "P3", "P4"]  # < 10% ownership, total points desc
    assert d.iloc[0]["minutes_last5"] == 180
    assert d.iloc[0]["next_fixtures"].startswith("ARS(H) 2 · BOU(A) 4")
    assert list(views.differentials(p, live, fx, boot, max_ownership=10.0, pos="FWD")["name"]) == ["P4"]
    assert views.differentials(p, live, fx, boot, price_range=(5.0, 7.0), clubs=["DOV"])["name"].tolist() == [
        "P5"
    ]


def test_compare_table_formats():
    p, _ = _players()
    t = views.compare_table(p, [2, 3])
    assert t.loc["Price", "P2 (ARS)"] == "5.5" and t.loc["Ownership", "P3 (BOU)"] == "5.5%"
    assert t.shape == (len(views.COMPARE_METRICS), 2)


# ------------------------------------------------------------------------------------------------ my team
def _picks(ids, captain, vice, chip=None):
    return {
        "active_chip": chip,
        "picks": [
            {"element": e, "position": i + 1, "is_captain": e == captain, "is_vice_captain": e == vice}
            for i, e in enumerate(ids)
        ],
    }


def test_captain_gap_and_unfinished_gw_skipped():
    p, _ = _players()
    live = views.live_frame(
        {
            1: {
                "elements": [
                    {"id": e, "stats": {"minutes": 90, "total_points": pts}}
                    for e, pts in [(1, 2), (2, 9), (3, 5)]
                ]
            },
        }
    )
    pk = {1: _picks([1, 2, 3], captain=3, vice=2), 2: _picks([1, 2, 3], 3, 2)}
    g = views.captain_gap(pk, live, p)
    assert len(g) == 1  # GW2 has no live data yet
    r = g.iloc[0]
    assert (r["captain"], r["captain_points"], r["best_xi_player"], r["best_points"], r["gap"]) == (
        "P3",
        5,
        "P2",
        9,
        4,
    )


def test_transfer_ledger_window_and_free_hit():
    p, _ = _players()
    live = views.live_frame(
        {
            g: {
                "elements": [{"id": 2, "stats": {"total_points": 5}}, {"id": 4, "stats": {"total_points": 1}}]
            }
            for g in (1, 2, 3)
        }
    )
    transfers = [
        {
            "element_in": 2,
            "element_out": 4,
            "element_in_cost": 55,
            "element_out_cost": 100,
            "event": 2,
            "time": "a",
        },
        {
            "element_in": 4,
            "element_out": 2,
            "element_in_cost": 100,
            "element_out_cost": 55,
            "event": 3,
            "time": "b",
        },
        {
            "element_in": 3,
            "element_out": 1,
            "element_in_cost": 80,
            "element_out_cost": 45,
            "event": 4,
            "time": "c",
        },
    ]
    hist = {"chips": [{"name": "freehit", "event": 3}], "current": [{"event": 2, "event_transfers_cost": 4}]}
    led = views.transfer_ledger(transfers, hist, live, p, last_finished=3)
    assert led.iloc[0]["window"] == "GW2-GW3" and led.iloc[0]["in_minus_out"] == 10 - 2
    assert led.iloc[0]["gw_hit_cost"] == 4
    assert led.iloc[1]["window"] == "GW3-GW3" and led.iloc[1]["chip"] == "Free Hit"
    assert led.iloc[2]["window"] == "not started" and pd.isna(led.iloc[2]["in_minus_out"])


def test_league_gap_uses_page_one_leader():
    page = {
        "standings": {
            "results": [
                {"entry": 9, "rank": 1, "rank_sort": 1, "total": 500},
                {"entry": 7, "rank": 2, "rank_sort": 2, "total": 480},
            ]
        }
    }
    assert views.league_gap(page, 7, 470) == {"leader_total": 500, "gap": 20, "my_total": 480}
    assert views.league_gap(page, 8, 470)["gap"] == 30  # not on page 1: falls back to the entry's total
    assert views.league_gap({"standings": {"results": []}}, 7, 1)["leader_total"] is None


def test_pitch_rows_and_badges():
    p, _ = _players()
    live = views.live_frame({1: {"elements": [{"id": 3, "stats": {"total_points": 7}}]}})
    rows = views.pitch(_picks([1, 2, 3, 4, 5], captain=3, vice=2), p, live, 1)
    assert [c["name"] for c in rows["MID"]] == ["P3", "P5"] and rows["MID"][0]["badge"] == "C"
    assert rows["MID"][0]["points"] == 7 and rows["DEF"][0]["badge"] == "VC"


def test_gw_frame_joins_average():
    hist = {
        "current": [
            {
                "event": 1,
                "points": 60,
                "event_transfers_cost": 4,
                "points_on_bench": 3,
                "overall_rank": 10,
                "total_points": 56,
                "value": 1000,
                "bank": 5,
            }
        ]
    }
    g = views.gw_frame(hist, _boot())
    r = g.iloc[0]
    assert (r["gw_average"], r["hits"], r["net_points"], r["bench"], r["value"], r["bank"]) == (
        51,
        4,
        56,
        3,
        100.0,
        0.5,
    )


# ------------------------------------------------------------------------------------------------ team id
@pytest.mark.parametrize(
    "raw, ok",
    [
        ("7630179", True),
        (" 42 ", True),
        ("0", False),
        ("abc", False),
        ("", False),
        ("-5", False),
        ("1e5", False),
        ("99999999999", False),
    ],
)
def test_parse_team_id(raw, ok):
    tid, err = parse_team_id(raw)
    assert (tid is not None) is ok and (err is None) is ok
