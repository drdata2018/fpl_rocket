"""Step 4e: reconciliation variants (October inputs, played + flags off), captain swap, seed diagnostic and
the points decomposition."""

from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fpl_rocket.replay import decompose as D
from fpl_rocket.replay import engine, seeds, variants
from fpl_rocket.replay.forecast import Forecasts
from fpl_rocket.rules import season_rules

# synthetic world: GK 1-2, DEF 3-7, MID 8-12, FWD 13-15, one club each; pool 16-20
TYPES = {1: 1, 2: 1, **{e: 2 for e in range(3, 8)}, **{e: 3 for e in range(8, 13)}, 13: 4, 14: 4, 15: 4}
TYPES |= {16: 2, 17: 3, 18: 4, 19: 1, 20: 3}
XI = [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14]
BENCH = [2, 7, 12, 15]
LIMITS = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}


def _fc(gws=(1,)) -> Forecasts:
    els = sorted(TYPES)
    p = pd.DataFrame(
        {
            "element": els,
            "element_type": [TYPES[e] for e in els],
            "team": els,
            "now_cost": [50] * len(els),
            "buyable": [True] * len(els),
            "unavailable": [False] * len(els),
            "code": els,
        }
    )
    base = {(e, g): 2.0 + e / 100 for e in els for g in gws}
    base[(3, gws[0])] = 9.5  # the form baseline's top player is DEF 3
    model = {(e, g): 3.0 + e / 100 for e in els for g in gws}
    model[(3, gws[0])] = 3.12  # in the model's XI, below its captain (15) and vice (14)
    return Forecasts(
        "2024-25",
        gws[0],
        list(gws),
        p,
        {(e, g): 1 for e in els for g in gws},
        frozenset({20}),
        {"baseline": base, "model": model},
    )


# ================================================================================= variants
def test_reconciliation_variants_and_their_tables():
    oc = variants.by_name("oct_conditions")
    assert (oc.model_flags, oc.form_flags, oc.chips, oc.imposed_start, oc.fixtures) == (
        False,
        False,
        False,
        True,
        "played",
    )
    assert oc.imposed_rule == "october" and oc.table_keys() == ("model_played_uncapped", "nfix_played")
    assert variants.by_name("oct_conditions_known").table_keys() == ("model_known_uncapped",)
    oi = variants.by_name("oct_inputs")
    assert oi.inputs == "october" and not oi.needs_model_tables() and oi.table_keys() == ()
    assert variants.by_name("fixtures_played").table_keys() == ("model_played_capped", "nfix_played")
    assert (
        variants.by_name("chips_off").table_keys() == ()
        and not variants.by_name("chips_off").changes_forecasts()
    )
    assert not variants.CAPTAIN_SWAP.changes_forecasts() and variants.by_name("captain_swap").captain_swap
    # the 4d variants keep their meaning
    assert variants.by_name("imposed_start").imposed_rule == "form_gw1"


def test_played_without_flags_reads_the_uncapped_played_table(monkeypatch):
    from fpl_rocket.replay import forecast

    seen = {}

    def fake_baseline(players, hist, nfix, gws):
        seen["unavailable"] = bool(players["unavailable"].any())
        seen["nfix"] = nfix
        return {"form": 1.0}

    monkeypatch.setattr(forecast, "baseline_xp", fake_baseline)
    fc = _fc()
    fc.players.loc[0, "unavailable"] = True
    tabs = {
        "model_played_uncapped": {k: v + 2 for k, v in fc.tables["model"].items()},
        "model_played_capped": {k: v + 1 for k, v in fc.tables["model"].items()},
        "nfix_played": {k: 2 for k in fc.n_fixtures},
    }
    out = variants.apply(variants.by_name("oct_conditions"), fc, SimpleNamespace(history=None), tabs)
    assert out.tables["model"] == tabs["model_played_uncapped"] and out.n_fixtures == tabs["nfix_played"]
    # the form baseline is recomputed on the played fixture counts without the availability zeroing
    assert out.tables["baseline"] == {"form": 1.0} and seen == {
        "unavailable": False,
        "nfix": tabs["nfix_played"],
    }
    assert fc.tables["model"][(1, 1)] == 3.01


def test_variant_tables_cache_rebuilds_when_a_needed_table_is_missing(tmp_path):
    inp = SimpleNamespace(season="2024-25", gw=3)
    p = variants.variant_tables_path(tmp_path, "2024-25", 3)
    p.parent.mkdir(parents=True)
    old = {"digest": "d", "version": variants.VARIANT_TABLES_VERSION, "model_known_uncapped": {}}
    with p.open("wb") as fh:
        pickle.dump(old, fh)
    got = variants.load_or_build_tables(inp, "d", tmp_path, build=False, need=("model_known_uncapped",))
    assert got is not None and "model_played_uncapped" not in got
    with pytest.raises(FileNotFoundError):
        variants.load_or_build_tables(inp, "d", tmp_path, build=False, need=("model_played_uncapped",))
    with pytest.raises(FileNotFoundError):
        variants.load_or_build_tables(inp, "other", tmp_path, build=False, need=("model_known_uncapped",))


def test_october_xp_carries_the_last_prediction_and_counts_fixtures():
    per = {1: pd.Series({10: 2.0, 11: 1.0}), 2: pd.Series({10: 3.0}), 3: pd.Series({10: 9.0})}
    nfix = {2: {10: 1, 11: 1}, 3: {10: 2, 11: 0}, 4: {10: 0, 11: 1}}
    out = variants.october_xp(per, nfix, 2, [2, 3, 4])
    # GW 3's prediction is not known at GW 2; player 11 keeps his GW1 prediction
    assert out == {(10, 2): 3.0, (11, 2): 1.0, (10, 3): 6.0, (11, 4): 1.0}


def test_october_forecast_replaces_tables_low_set_and_pool():
    fc = _fc(gws=(2, 3))
    october = {
        "preds": {
            "model": {2: pd.Series({e: 1.0 for e in TYPES})},
            "baseline": {1: pd.Series({e: 0.5 for e in TYPES})},
        },
        "lows": {2: frozenset({5})},
        "nfix": {2: {e: 1 for e in TYPES if e != 16}, 3: {e: 2 for e in TYPES}},
        "pool": {2: frozenset(e for e in TYPES if e != 17)},
    }
    v = variants.by_name("oct_inputs")
    with pytest.raises(ValueError):
        variants.apply(v, fc, None)
    with pytest.raises(ValueError):
        variants.apply(variants.Variant("x", inputs="october", fixtures="played"), fc, None, october=october)
    out = variants.apply(v, fc, None, october=october)
    assert out.tables["model"][(1, 2)] == 1.0 and out.tables["model"][(1, 3)] == 2.0
    assert (16, 2) not in out.tables["model"] and out.tables["baseline"][(4, 3)] == 1.0
    assert out.low == frozenset({5}) and out.n_fixtures[(16, 2)] == 0 and out.n_fixtures[(16, 3)] == 2
    assert not bool(out.players.set_index("element").at[17, "buyable"])
    assert bool(fc.players.set_index("element").at[17, "buyable"])  # the frozen forecasts are untouched
    assert fc.tables["model"][(1, 2)] == 3.01


def test_no_topup_variant_removes_only_the_afcon_top_up():
    v = variants.by_name("oct_inputs_no_topup")
    assert not v.ft_topup and v.inputs == "october"
    r = season_rules("2025-26")
    assert r.next_free_transfers(15, 1, 1, None) == 5  # AFCON top-up entering GW16
    nr = variants.rules_for(v, r)
    assert nr.next_free_transfers(15, 1, 1, None) == 1 and nr.next_free_transfers(15, 2, 0, None) == 3
    assert variants.rules_for(variants.by_name("oct_inputs"), r) is r
    r24 = season_rules("2024-25")
    assert variants.rules_for(v, r24) is r24


# ================================================================================= captain swap
def test_captain_swap_uses_the_other_policys_xp_on_the_own_xi():
    imposed = sorted(e for e in TYPES if e <= 15)
    v = variants.Variant("swap_imposed", imposed_start=True, captain_swap=True)
    pols = variants.make_variant_policies(v, imposed)
    fc = _fc()
    rules = season_rules("2024-25")
    st = engine.new_state(rules)
    model_pol = pols["minimal_model"]
    own = model_pol._imposed_gw1(fc, st, rules)
    assert (own.captain, own.vice) == (15, 14)  # the model's own pair
    # triple-captain trigger reads the form xP of the form-chosen captain (DEF 3: 9.5 >= 9)
    trig = model_pol.chip_triggers(fc, st, rules, own.bench, own.captain)
    assert trig["3xc"]() == (True, 9.5)
    pkg = model_pol._swap_captain(fc, own)
    assert (pkg.captain, pkg.vice) == (3, 15) and pkg.notes[-1]["captain_swap"]["own"] == [15, 14]
    assert model_pol.settings()["captain_source"] == "baseline"
    engine.validate(rules, st, fc.players, pkg)  # still a legal package
    # FORM_RESTRAINED captains by the model: the model's best XI player (15), vice 14
    form_pol = pols["form_restrained"]
    fpkg = form_pol._swap_captain(fc, form_pol._imposed_gw1(fc, st, rules))
    assert (fpkg.captain, fpkg.vice) == (15, 14)
    # without the switch nothing changes
    plain = variants.make_variant_policies(variants.Variant("i", imposed_start=True), imposed)[
        "minimal_model"
    ]
    p2 = plain._swap_captain(fc, plain._imposed_gw1(fc, st, rules))
    assert (p2.captain, p2.vice) == (15, 14)
    assert plain.chip_triggers(fc, st, rules, p2.bench, p2.captain)["3xc"]() == (False, 3.15)


# ================================================================================= seeds
def test_seed_scheme_and_package_changes():
    assert seeds.mc_seed(0, 7) == 20261023 + 7
    assert len({seeds.mc_seed(k, g) for k in range(9) for g in range(1, 39)}) == 9 * 38
    base = {"transfers_in": [1], "transfers_out": [2], "xi": XI, "bench": BENCH, "captain": 13, "vice": 8}
    assert not seeds.package_changes(base, dict(base))["any"]
    ch = seeds.package_changes(base, {**base, "vice": 9, "bench": [2, 12, 7, 15], "chip": "3xc"})
    assert ch == {"transfers": False, "captain": True, "xi_bench": True, "chip": True, "any": True}
    assert seeds.package_changes(base, {**base, "transfers_in": [3]})["transfers"]


def test_seed_standard_errors():
    rows = {r["m"]: r for r in seeds.se_by_count([1, 3, 1, 3, 1, 3, 1, 3])}
    sd = float(np.std([1, 3] * 4, ddof=1))
    assert rows[1]["se"] == pytest.approx(sd) and rows[4]["se"] == pytest.approx(sd / 2)
    assert rows[2]["groups"] == 4 and rows[2]["empirical_sd"] == 0.0
    assert np.isnan(rows[8]["empirical_sd"]) and rows[8]["groups"] == 1


def test_seed_cache_adds_missing_indices_and_is_keyed_by_digest(tmp_path, monkeypatch):
    calls = []

    def fake_build(inp, ks):
        calls.append(list(ks))
        return {k: {(1, inp.gw): float(k)} for k in ks}

    monkeypatch.setattr(seeds, "build_seed_tables", fake_build)
    inp = SimpleNamespace(season="2023-24", gw=5)
    got = seeds.load_or_build(inp, "d1", [0, 1], tmp_path)
    assert got == {0: {(1, 5): 0.0}, 1: {(1, 5): 1.0}} and calls == [[0, 1]]
    assert seeds.load_or_build(inp, "d1", [1], tmp_path, build=False) == {1: {(1, 5): 1.0}}
    seeds.load_or_build(inp, "d1", [0, 2], tmp_path)
    assert calls[-1] == [2]
    with pytest.raises(FileNotFoundError):
        seeds.load_or_build(inp, "d2", [1], tmp_path, build=False)  # another input digest: not reused
    assert seeds.seed_tables_path(tmp_path, "2023-24", 5).parent.name == "2023-24"
    fc = _fc()
    out = seeds.with_model_table(fc, {(1, 1): 7.0}, 3)
    assert out.tables["model"] == {(1, 1): 7.0} and out.tables["baseline"] is fc.tables["baseline"]
    assert out.meta["mc_seed"] == seeds.mc_seed(3, 1) and fc.tables["model"][(1, 1)] == 3.01


# ================================================================================= decomposition
def _pkg(chip=None, captain=13, vice=8):
    return {"xi": XI, "bench": BENCH, "captain": captain, "vice": vice, "chip": chip}


def test_components_add_up_and_follow_the_convention():
    pts = {e: 2 for e in TYPES} | {13: 10, 8: 6, 7: 5, 12: 3}
    mins = {e: 90 for e in TYPES}
    c = D.gw_components(_pkg(), pts, mins, TYPES, LIMITS)
    assert (c["lineup"], c["captain"], c["chip"], c["total"]) == (2 * 9 + 10 + 6, 10, 0, 44)
    assert c["captain_top_of_xi"] and not c["vice_took_armband"]
    # captain absent: a bench player comes in, the vice's points are the captain component
    c = D.gw_components(_pkg(), pts, mins | {13: 0}, TYPES, LIMITS)
    assert c["captain"] == 6 and c["vice_took_armband"] and c["lineup"] == 2 * 9 + 6 + 5
    assert c["auto_subs"] == 1 and not c["captain_played"]
    # bench boost: the bench is the chip; triple captain: the third multiple
    c = D.gw_components(_pkg("bboost"), pts, mins, TYPES, LIMITS)
    assert (c["lineup"], c["chip"]) == (34, 2 + 5 + 3 + 2)
    c = D.gw_components(_pkg("3xc"), pts, mins, TYPES, LIMITS)
    assert (c["captain"], c["chip"], c["total"]) == (10, 10, 54)
    c = D.gw_components(_pkg("wildcard"), pts, mins, TYPES, LIMITS, hits=1)
    assert c["flag_squad_chip"] and c["chip"] == 0 and c["hits"] == -4 and c["total"] == 40
    c = D.gw_components(_pkg("3xc"), pts, mins, TYPES, LIMITS, void=True)
    assert c["total"] == 0 and c["chip_effective"] == ""


def test_policy_components_from_a_log_and_pairing(tmp_path):
    path = tmp_path / "p.jsonl"
    rows = []
    for gw, chip, net in ((1, None, 44), (2, "3xc", 54)):
        pkg = {**_pkg(chip), "scores": {"13": 7.0, "8": 5.5, "9": 6.0}}
        rows.append(json.dumps({"event": "decision", "gw": gw, "package": pkg}))
        rows.append(json.dumps({"event": "score", "gw": gw, "row": {"net": net, "hits": 0}}))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    res = pd.DataFrame(
        {"points": {e: 2 for e in TYPES} | {13: 10, 8: 6}, "minutes": dict.fromkeys(TYPES, 90)}
    )
    df = D.policy_components(path, "2024-25", lambda gw: res, TYPES, LIMITS)
    assert list(df["check"]) == [0, 0] and list(df["pred_second_el"]) == [9, 9]
    assert df["pred_captain"].iloc[0] == 7.0 and df["pred_vice"].iloc[0] == 5.5
    other = df.assign(lineup=df["lineup"] - 1, net=df["net"] - 1)
    d = D.paired(df, other)
    assert list(d["lineup"]) == [1, 1] and list(d["net"]) == [1, 1] and list(d["captain"]) == [0, 0]


def test_leave_largest_out_and_margin_bins():
    r = D.leave_largest_out([5, -2, 10, 1], 1)
    assert r == {"k": 1, "gws": 3, "total": 4.0, "mean": pytest.approx(4 / 3)}
    assert D.leave_largest_out([5, -2], 0)["total"] == 3.0
    df = pd.DataFrame(
        {
            "gw": range(6),
            "void": False,
            "pred_captain": [1.0, 1.1, 2.0, 2.5, 5.0, 6.0],
            "pred_vice": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "captain_points": [2, 1, 6, 2, 12, 9],
            "vice_points": [2, 2, 2, 2, 2, 2],
        }
    )
    b = D.margin_bins(df, 3)
    assert list(b["gws"]) == [2, 2, 2] and list(b["real_mean"]) == [-0.5, 2.0, 8.5]
    assert D.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_crowd_captains_after_the_fact_with_gw1_from_the_lagged_table():
    after = pd.DataFrame(
        {
            "season": "2024-25",
            "gw": [2, 2, 2],
            "tier": "10000",
            "element": [5, 7, 9],
            "captains": [40, 40, 3],
            "managers": 370,
        }
    )
    lagged = pd.DataFrame(
        {
            "season": "2024-25",
            "gw": [2, 2, 3],
            "source_gw": [1, 1, 2],
            "tier": "10000",
            "element": [4, 6, 9],
            "captains": [10, 50, 99],
            "managers": 370,
        }
    )
    top = D.crowd_captains(after, lagged).set_index("gw")
    assert top.at[1, "crowd_captain"] == 6 and top.at[1, "source"] == "lagged"
    assert (
        top.at[2, "crowd_captain"] == 5 and top.at[2, "tied"] == 1 and top.at[2, "source"] == "after_the_fact"
    )
