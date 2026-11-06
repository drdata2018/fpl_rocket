"""Blind replay engine: legality rejections, scoring (auto-subs, vice, chips, hits, free-hit restore), the
chip rule (triggers and last eligible GW), the postponed-match exclusion on a real rearranged fixture,
determinism and the leakage test on recorded fixtures (tests/fixtures/replay, built by build_fixture.py)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fpl_rocket.replay import chips as chip_rule
from fpl_rocket.replay import engine, forecast
from fpl_rocket.replay.engine import IllegalPackage, ManagerState, Package
from fpl_rocket.replay.inputs import HistoryStore, corrupt_copy, load_inputs
from fpl_rocket.replay.policies import make_policies
from fpl_rocket.rules import season_rules

FIX = Path(__file__).parent / "fixtures" / "replay"
ASKNOWN = FIX / "asknown_mini"
GW = 5  # the recorded as-known slice (build_fixture.py)
HISTORY = FIX / "history_mini.parquet"
FULL_GW17 = Path(__file__).resolve().parents[1] / "data" / "asknown" / "fixtures" / "2021-22" / "gw17.parquet"

# synthetic world: squad 1-15 (GK 1-2, DEF 3-7, MID 8-12, FWD 13-15) from 15 clubs, plus pool players
TYPES = {1: 1, 2: 1, **{e: 2 for e in range(3, 8)}, **{e: 3 for e in range(8, 13)}, 13: 4, 14: 4, 15: 4}
TEAMS = {e: e for e in range(1, 16)}
EXTRA = {  # element: (type, team, cost, buyable)
    16: (2, 1, 45, True),
    17: (3, 1, 60, True),
    18: (3, 1, 60, True),
    19: (3, 2, 400, True),
    20: (1, 3, 40, True),
    21: (4, 1, 55, True),
    22: (3, 4, 50, False),
}
XI = [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14]
BENCH = [2, 7, 12, 15]


def players() -> pd.DataFrame:
    rows = [
        {"element": e, "element_type": t, "team": TEAMS[e], "now_cost": 50, "buyable": True}
        for e, t in TYPES.items()
    ]
    rows += [
        {"element": e, "element_type": t, "team": c, "now_cost": v, "buyable": b}
        for e, (t, c, v, b) in EXTRA.items()
    ]
    return pd.DataFrame(rows)


def state(season: str = "2025-26", ft: int = 1, **kw) -> ManagerState:
    st = engine.new_state(season_rules(season))
    st.squad = {e: 50 for e in TYPES}
    st.bank = 250
    st.free_transfers = ft
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def pkg(season="2025-26", gw=10, t_in=(), t_out=(), xi=XI, bench=BENCH, cap=13, vice=8, chip=None, hits=0):
    return Package(season, gw, list(t_in), list(t_out), list(xi), list(bench), cap, vice, chip, hits)


def results(points=None, minutes=None) -> pd.DataFrame:
    els = list(TYPES) + list(EXTRA)
    pts = {e: 2 for e in els} | (points or {})
    mins = {e: 90 for e in els} | (minutes or {})
    return pd.DataFrame({"points": [pts[e] for e in els], "minutes": [mins[e] for e in els]}, index=els)


# ======================================================================================= legality
def test_legal_package_executes():
    rules = season_rules("2025-26")
    new, row = engine.step(
        rules,
        state(),
        players(),
        pkg(t_in=[16], t_out=[3], xi=[1, 16, 4, 5, 6, 8, 9, 10, 11, 13, 14]),
        results(),
    )
    assert 16 in new.squad and 3 not in new.squad and new.bank == 250 + 50 - 45
    assert row["points"] == 11 * 2 + 2 and row["hits"] == 0 and new.free_transfers == 1


@pytest.mark.parametrize(
    "kw,msg",
    [
        ({"t_in": [19], "t_out": [8], "xi": [1, 3, 4, 5, 6, 19, 9, 10, 11, 13, 14]}, "over budget"),
        (
            {
                "t_in": [16, 17, 21],
                "t_out": [3, 8, 13],
                "hits": 2,
                "xi": [1, 16, 4, 5, 6, 17, 9, 10, 11, 21, 14],
                "cap": 21,
            },
            "players from club",
        ),
        ({"xi": [1, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15], "bench": [2, 5, 6, 7]}, "formation"),
        ({"bench": [7, 2, 12, 15]}, "reserve goalkeeper"),
        ({"chip": "3xc", "gw": 10}, None),  # legal control (held)
        (
            {"t_in": [16, 17], "t_out": [3, 8], "xi": [1, 16, 4, 5, 6, 17, 9, 10, 11, 13, 14]},
            "declared 0 hits",
        ),
        ({"t_in": [22], "t_out": [8], "xi": [1, 3, 4, 5, 6, 22, 9, 10, 11, 13, 14]}, "not selectable"),
        ({"t_in": [3], "t_out": [4]}, "already owned"),
        ({"t_in": [99], "t_out": [4]}, "not in the game"),
        ({"cap": 2}, "captain"),
        ({"cap": 13, "vice": 13}, "captain"),
        ({"xi": XI[:10] + [2], "bench": [14, 7, 12, 15]}, "first bench slot"),
    ],
)
def test_illegal_packages_are_rejected(kw, msg):
    rules = season_rules("2025-26")
    p = pkg(**kw)
    if msg is None:
        engine.validate(rules, state(), players(), p)
        return
    with pytest.raises(IllegalPackage, match=msg):
        engine.validate(rules, state(), players(), p)


def test_chip_legality():
    r21, r22, r25 = season_rules("2021-22"), season_rules("2022-23"), season_rules("2025-26")
    with pytest.raises(IllegalPackage, match="not held"):  # first-half bench boost is GW1-19 only
        engine.validate(
            r25, state(chips=[c for c in r25.chips if c.number == 1]), players(), pkg(gw=25, chip="bboost")
        )
    with pytest.raises(IllegalPackage, match="consecutive"):
        engine.validate(
            r21, state("2021-22", last_fh_gw=24), players(), pkg("2021-22", gw=25, chip="freehit")
        )
    with pytest.raises(IllegalPackage, match="not held|not allowed"):  # World Cup GW17: no wildcard
        engine.validate(r22, state("2022-23"), players(), pkg("2022-23", gw=17, chip="wildcard"))
    am = state("2024-25")
    with pytest.raises(IllegalPackage, match="unsupported"):
        engine.validate(season_rules("2024-25"), am, players(), pkg("2024-25", gw=30, chip="manager"))
    # 2022-23 GW17: transfers free and unlimited
    engine.validate(
        r22,
        state("2022-23"),
        players(),
        pkg(
            "2022-23", gw=17, t_in=[16, 17], t_out=[3, 8], xi=[1, 16, 4, 5, 6, 17, 9, 10, 11, 13, 14], vice=9
        ),
    )


# ======================================================================================= scoring
def test_auto_subs_and_vice_captain():
    rules = season_rules("2025-26")
    res = results(points={1: 0, 3: 0, 13: 0, 8: 7, 7: 5, 2: 3}, minutes={1: 0, 3: 0, 13: 0})
    _, row = engine.step(rules, state(), players(), pkg(cap=13, vice=8), res)
    # GK1 (0 min) -> reserve GK2 (3); DEF3 -> DEF7 (5, first outfield bench player); FWD13 -> MID12 (4-5-1 is
    # legal); FWD15 unused. Captain 13 did not play -> the vice (8, 7 points) is doubled.
    assert row["auto_subs"] == 3 and row["armband"] == 8
    assert row["points"] == 3 + 5 + 3 * 2 + 7 * 2 + 3 * 2 + 2 + 2


def test_chips_scoring_hits_and_free_hit_restore():
    rules = season_rules("2025-26")
    res = results(points={13: 10})
    _, tc = engine.step(rules, state(), players(), pkg(chip="3xc"), res)
    _, plain = engine.step(rules, state(), players(), pkg(), res)
    assert tc["points"] - plain["points"] == 10 and tc["chip_gain"] == 10
    _, bb = engine.step(rules, state(), players(), pkg(chip="bboost"), res)
    assert bb["points"] - plain["points"] == 8 and bb["chip_gain"] == 8
    _, hit = engine.step(
        rules,
        state(ft=1),
        players(),
        pkg(t_in=[16, 21], t_out=[3, 14], hits=1, xi=[1, 16, 4, 5, 6, 8, 9, 10, 11, 13, 21]),
        res,
    )
    assert hit["net"] == hit["points"] - 4 and hit["hits"] == 1
    st = state(ft=3)
    new, fh = engine.step(
        rules,
        st,
        players(),
        pkg(chip="freehit", t_in=[16, 21], t_out=[3, 14], xi=[1, 16, 4, 5, 6, 8, 9, 10, 11, 13, 21]),
        res,
    )
    assert new.squad == st.squad and new.bank == st.bank  # restored after the GW
    assert new.free_transfers == 3 and new.last_fh_gw == 10  # 2025-26 "carry": banked balance kept
    assert isinstance(fh["chip_gain"], int)
    old = state("2023-24", ft=2)
    new, _ = engine.step(season_rules("2023-24"), old, players(), pkg("2023-24", chip="wildcard"), res)
    assert new.free_transfers == 1  # before 2024-25 a wildcard loses the banked transfers


def test_void_gw_reinstates_chips_and_scores_zero():
    rules = season_rules("2022-23")
    st = state("2022-23")
    new, row = engine.step(rules, st, players(), pkg("2022-23", gw=7, chip="bboost"), results())
    assert row["points"] == 0 and row["chip_reinstated"]
    assert any(c.name == "bboost" for c in new.chips)


# ======================================================================================= chip rule
def test_last_eligible_schedule_and_forced_chip():
    r = season_rules("2025-26")
    first = [c for c in r.chips if c.number == 1]
    sched = chip_rule.schedule(r, first, 16, None)
    by = {k.split("#")[0]: g for k, g in sched.items()}
    assert by == {"3xc": 19, "bboost": 18, "freehit": 17, "wildcard": 16}
    never = {k: (lambda: (False, 0.0)) for k in chip_rule.PRIORITY}
    chip, rec = chip_rule.decide(r, first, 16, None, never)
    assert chip == "wildcard" and rec["reason"] == "last eligible GW"
    chip, _ = chip_rule.decide(r, first, 15, None, never)
    assert chip is None
    # 2021-22 end of season: two free hits must not be consecutive
    r21 = season_rules("2021-22")
    held = [c for c in r21.chips if c.stop_event == 38]
    s21 = chip_rule.schedule(r21, held, 30, None)
    fh = sorted(g for k, g in s21.items() if k.startswith("freehit"))
    assert fh[1] - fh[0] >= 2 and len(set(s21.values())) == len(s21)


def test_triggers_in_priority_order_and_hold_without_wildcard():
    r = season_rules("2024-25")
    held = list(r.chips)
    calls = []

    def t(name, fire):
        def f():
            calls.append(name)
            return fire, 1.0

        return f

    trig = {
        "wildcard": t("wildcard", False),
        "freehit": t("freehit", False),
        "bboost": t("bboost", True),
        "3xc": t("3xc", True),
    }
    chip, rec = chip_rule.decide(r, held, 10, None, trig)
    assert chip == "bboost" and calls == ["wildcard", "freehit", "bboost"] and rec["reason"] == "trigger"
    chip, _ = chip_rule.decide(
        r, held, 10, None, {"wildcard": t("wildcard", True)}, allowed=("freehit", "bboost", "3xc")
    )
    assert chip is None
    assert (
        chip_rule.decide(season_rules("2022-23"), list(season_rules("2022-23").chips), 7, None, trig)[0]
        is None
    )


def test_chip_thresholds_are_the_prespecified_defaults():
    c = chip_rule.CHIP_RULE
    assert (c.bb_bench_xp, c.tc_captain_xp, c.fh_blank_players, c.wc_gain, c.wc_horizon) == (
        12.0,
        9.0,
        4,
        12.0,
        5,
    )


# ======================================================================================= inputs at T
def test_postponed_match_excluded_until_played():
    """Fixture 166 (Leicester v Spurs) was in GW17 as known at the GW17 deadline (fixtures_as_known), was
    postponed and was played on 2022-01-19, after the GW22 deadline: it must be absent from the history at
    every issue time GW18..GW22 (while the GW17 matches that were played are present) and present at GW23's;
    decided by kickoff + 3 h <= T."""
    if FULL_GW17.exists():  # the as-known schedule at the GW17 deadline (full archive only)
        fx17 = pd.read_parquet(FULL_GW17)
        assert int(fx17.loc[fx17["fixture"] == 166, "event"].iloc[0]) == 17
    store = HistoryStore(HISTORY)
    dl = pd.read_parquet(ASKNOWN / "deadlines.parquet").set_index("gw")
    played17 = set(store.df.loc[store.df["gw"] == 17, "fixture"])
    for gw in range(18, 23):
        h = store.known_at("2021-22", dl.at[gw, "issue_time_utc"])
        assert {162, 164, 166}.isdisjoint(set(h["fixture"])), gw  # all three postponed from GW17
        assert played17 <= set(h["fixture"]), gw
    t23 = dl.at[23, "issue_time_utc"]
    h23 = store.known_at("2021-22", t23)
    assert {162, 166} <= set(h23["fixture"]) and 164 not in set(h23["fixture"])  # 164 played 5 Feb
    assert (pd.to_datetime(h23["known_at"], utc=True) <= t23).all()


def _decide_all(inp, fc, st_by_policy):
    rules = season_rules(inp.season)
    out = {}
    for name, pol in make_policies().items():
        if name == "minimal_model":
            continue
        out[name] = pol.decide(fc, st_by_policy[name].copy(), rules, inp).key()
    return out


def _states(inp, fc):
    rules = season_rules(inp.season)
    states = {}
    for name, pol in make_policies().items():
        st = engine.new_state(rules, inp.issue_time)
        first = pol.decide(fc, st, rules, inp) if name != "minimal_model" else None
        if first is not None:
            st.squad = {e: int(fc.price()[e]) for e in first.transfers_in}
            st.bank = 1000 - sum(st.squad.values())
            st.free_transfers = 2
        states[name] = st
    return states


def test_determinism_and_leak_test_on_recorded_fixtures(tmp_path):
    store = HistoryStore(HISTORY)
    inp = load_inputs("2021-22", GW, store, ASKNOWN)
    assert inp.violations() == {}
    fc = forecast.build(inp, sources=("baseline",))
    states = _states(inp, fc)
    a = _decide_all(inp, fc, states)
    b = _decide_all(
        inp, forecast.build(load_inputs("2021-22", GW, store, ASKNOWN), sources=("baseline",)), states
    )
    assert a == b  # deterministic
    # leakage: corrupt every row known after T (GW5 and GW6 points, GW17 matches, later deadlines)
    ak, hist, changed = corrupt_copy(ASKNOWN, HISTORY, "2021-22", inp.issue_time, tmp_path / "leak")
    assert changed > 0
    store2 = HistoryStore(hist)
    assert len(store2.df) == len(store.df)
    inp2 = load_inputs("2021-22", GW, store2, ak)
    assert inp2.violations() == {}
    assert inp2.manifest()["asknown_digest"] == inp.manifest()["asknown_digest"]
    assert inp2.manifest()["history_digest"] == inp.manifest()["history_digest"]
    c = _decide_all(inp2, forecast.build(inp2, sources=("baseline",)), states)
    assert c == a
    # control: the same corruption one week EARLIER (points visible at T are mutated) changes the inputs
    ak3, hist3, _ = corrupt_copy(
        ASKNOWN, HISTORY, "2021-22", inp.issue_time - pd.Timedelta(days=7), tmp_path / "c"
    )
    inp3 = load_inputs("2021-22", GW, HistoryStore(hist3), ak3)
    assert inp3.manifest()["history_digest"] != inp.manifest()["history_digest"]
