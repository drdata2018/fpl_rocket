"""v0.2 pieces on synthetic data: low_history, newcomer fallbacks, DefCon prior, bonus modes, component
points, and the diagnostics (bootstrap, sign test, comparators, GK diagnostic, intervals, folds, horizons,
stability)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from fpl_rocket.backtest import FormPer90Baseline, run_backtest  # noqa: E402
from fpl_rocket.model import diagnostics as D  # noqa: E402
from fpl_rocket.model.bonus import bonus_link_table  # noqa: E402
from fpl_rocket.model.config import ModelConfig  # noqa: E402
from fpl_rocket.model.features import history_gw_counts, low_history, player_states  # noqa: E402
from fpl_rocket.model.newcomers import NewcomerPrior, gws_before, price_band, shrink_to_band  # noqa: E402
from fpl_rocket.model.pipeline import XP_COLUMNS, XPModel  # noqa: E402
from fpl_rocket.model.simulate import COMPONENTS, LINE_STATS, component_points, score_lines  # noqa: E402
from fpl_rocket.rules import season_2026_27  # noqa: E402
from model_synth import SEASON1, SEASON2, clubs, world  # noqa: E402

FAST = ModelConfig(n_sims=150, horizons=(1,), hgb_max_iter=30, bonus_max_iter=30)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    w = world()
    # newcomers: slot 14 of club T00 appears only from SEASON2 GW6 (new code) and plays 90 from GW7
    new = (w["season"] == SEASON2) & (w["code"] == 10_014)
    w.loc[new, "code"] = 99_999
    w = w[~(new & (w["gw"] < 6))].copy()
    late = (w["code"] == 99_999) & (w["gw"] >= 7)
    w.loc[late, ["minutes", "starts"]] = [90, 1]
    w.loc[w["code"] == 99_999, "value"] = 100
    return w.reset_index(drop=True)


@pytest.fixture(scope="module")
def fitted(df):
    train = df[(df["season"] == SEASON1) | (df["gw"] < 8)]
    target = df[(df["season"] == SEASON2) & (df["gw"] == 8)]
    model = XPModel(FAST).fit(
        train, season=SEASON2, ref_time=target["kickoff_time"].min(), season_teams=clubs(SEASON2)
    )
    return model, target.drop(columns=["minutes", "total_points"]).assign(horizon=1)


# ------------------------------------------------------------------------------------------ low_history
def test_history_counts_and_low_history_flag():
    hist = pd.DataFrame(
        {
            "season": [SEASON1] * 3 + [SEASON2] * 2,
            "gw": [37, 38, 38, 1, 2],
            "element": [1, 1, 1, 2, 3],
            "code": [10, 10, 10, 11, 12],  # code 10: two fixtures in GW38 count once
        }
    )
    assert history_gw_counts(hist).to_dict() == {10: 2, 11: 1, 12: 1}
    tg = pd.DataFrame({"element": [1, 2, 9], "code": [10, 11, 99]})
    n, low = low_history(tg, hist, 2)
    assert n.tolist() == [2, 1, 0] and low.tolist() == [False, True, True]


def test_player_states_bin_counts():
    rows = pd.DataFrame(
        {
            "season": [SEASON2] * 4,
            "gw": [1, 2, 2, 3],
            "element": [1] * 4,
            "code": [7] * 4,
            "team": ["A"] * 4,
            "fixture": [1, 2, 3, 4],
            "minutes": [90, 0, 30, 0],
            "starts": [1, 0, 0, 0],
        }
    )
    st = player_states(rows).set_index("t")
    gw2 = st.loc[season_t_(2)]
    assert (gw2["bin0_1"], gw2["bin1_1"], gw2["bin2_1"]) == (1.0, 1.0, 0.0)  # double GW: 0 and 30
    gw3 = st.loc[season_t_(3)]
    assert (gw3["bin0_5"], gw3["bin1_5"], gw3["bin2_5"]) == (2.0, 1.0, 1.0)


def season_t_(gw: int) -> int:
    return int(SEASON2[:4]) * 100 + gw


def test_backtest_predictions_carry_low_history(df):
    small = df[df["gw"] <= 8]
    res = run_backtest(small, [SEASON2], FormPer90Baseline())
    p = res.predictions
    assert {"low_history", "n_hist"} <= set(p.columns) and p["low_history"].dtype == bool
    newbie = p[p["element"] == p.loc[p["n_hist"] == 0, "element"].iloc[0]]
    assert newbie["low_history"].iloc[0]
    assert not p.loc[p["n_hist"] >= 3, "low_history"].any()


# ------------------------------------------------------------------------------------------ newcomers
def test_price_band_and_gws_before():
    assert price_band(pd.Series([45, 50, 59, 60, 75, 99, 100, np.nan]), (50, 60, 75, 100)).tolist() == [
        0,
        1,
        1,
        2,
        3,
        3,
        4,
        -1,
    ]
    h = pd.DataFrame({"season": [SEASON2] * 4, "gw": [1, 2, 2, 5], "code": [1, 1, 1, 1], "element": [1] * 4})
    assert gws_before(h).tolist() == [0, 1, 1, 2]


def test_newcomer_prior_changes_only_low_history_rows(fitted):
    model, target = fitted
    base, _ = model.with_config(newcomer_fallback="none").lines(target)
    pri, _ = model.with_config(newcomer_fallback="prior").lines(target)
    low = base["low_history"].to_numpy(bool)
    assert low.any() and (~low).any()
    cols = ["p0", "p1", "p2", "share_g", "share_a"]
    assert np.allclose(base.loc[~low, cols], pri.loc[~low, cols])
    assert not np.allclose(base.loc[low, cols], pri.loc[low, cols])
    assert np.allclose(pri[["p0", "p1", "p2"]].sum(axis=1), 1.0)
    # the newcomer's own current-season bins (GW6: 0 min, GW7: 90) + the band prior worth one fixture
    nb = pri[pri["code"] == 99_999].iloc[0]
    prior = model.newcomer.minutes_prior(np.array([nb["element_type"]]), np.array([nb["price_band"]]))[0]
    assert nb["price_band"] == 4
    assert [nb["p0"], nb["p1"], nb["p2"]] == pytest.approx(((np.array([1, 0, 1]) + prior) / 3).tolist())


def test_newcomer_prior_uses_newcomer_rows_by_band():
    rows = []
    for season, gws in ((SEASON1, range(1, 11)), (SEASON2, range(1, 4))):
        for gw in gws:
            for el, value, mins in ((1, 45, 0), (2, 100, 90)):
                code = el * 1000 + (gw if season == SEASON2 else 0)  # SEASON2: a new code every GW
                rows.append(
                    {
                        "season": season,
                        "gw": gw,
                        "element": el,
                        "code": code,
                        "element_type": 3,
                        "value": value,
                        "minutes": mins,
                        "expected_goals": 0.5 if mins else 0.0,
                        "expected_assists": 0.0,
                    }
                )
    h = pd.DataFrame(rows)
    pri = NewcomerPrior(ModelConfig(newcomer_prior_min_rows=2)).fit(h)
    assert pri.n_newcomer_rows == 6  # SEASON2 rows only (SEASON1 GW <= 5 is the warm-up, later GWs not new)
    cheap = pri.minutes_prior(np.array([3]), np.array([0]))[0]
    dear = pri.minutes_prior(np.array([3]), np.array([4]))[0]
    assert cheap.tolist() == [1.0, 0.0, 0.0] and dear.tolist() == [0.0, 0.0, 1.0]


def test_shrink_and_withhold_fallbacks(fitted):
    model, target = fitted
    none = model.with_config(newcomer_fallback="none").predict(target, n_sims=100, sims_frame=False).xp
    held = model.with_config(newcomer_fallback="withhold").predict(target, n_sims=100, sims_frame=False).xp
    assert held.loc[held["low_history"], "xp"].isna().all()
    assert np.allclose(held.loc[~held["low_history"], "xp"], none.loc[~none["low_history"], "xp"])
    x = pd.DataFrame(
        {
            "gw": [1, 1, 1],
            "element_type": [3, 3, 3],
            "price_band": [2, 2, 2],
            "n_fixtures": [1, 1, 2],
            "n_hist": [10, 10, 0],
            "low_history": [False, False, True],
            "xp": [2.0, 4.0, 0.5],
        }
    )
    out = shrink_to_band(x, ModelConfig())
    assert out.tolist() == [2.0, 4.0, pytest.approx(6.0)]  # band mean 3 per fixture x 2 fixtures, w = 0


def test_xp_output_has_low_history_and_with_config_shares_components(fitted):
    model, target = fitted
    other = model.with_config(bonus_sim="linked")
    assert other.minutes is model.minutes and other.cfg.bonus_sim == "linked"
    assert model.cfg.bonus_sim == "poisson"
    xp = other.predict(target, n_sims=100).xp
    assert tuple(xp.columns) == XP_COLUMNS and xp["low_history"].dtype == bool
    assert xp["low_history"].any()
    with pytest.raises(ValueError):
        replace(model.cfg, newcomer_fallback="bogus")


# ------------------------------------------------------------------------------------------ DefCon prior
def test_defcon_prior_applies_only_without_history_and_with_defcon_scoring(df):
    s1 = df[df["season"] == SEASON1]  # SEASON1 rows carry no DefCon counts
    target = df[(df["season"] == SEASON2) & (df["gw"] == 1)].assign(horizon=1)
    kw = {"ref_time": target["kickoff_time"].min(), "season_teams": clubs(SEASON2)}
    cfg = FAST
    model = XPModel(cfg).fit(s1, season=SEASON2, **kw)
    assert model.rates.dc_positions == set()
    rows, _ = model.lines(target)
    want = rows["element_type"].map(cfg.defcon_prior90)
    assert np.allclose(rows["dc90"], want)
    off, _ = model.with_config(defcon_fallback=False).lines(target)
    assert (off["dc90"] == 0).all()
    old = XPModel(cfg).fit(s1, season="2024-25", **kw)  # a season before DefCon scoring: no prior
    rows_old, _ = old.lines(target.assign(season="2024-25"))
    assert (rows_old["dc90"] == 0).all()


# ------------------------------------------------------------------------------------------ simulation
def test_component_points_sum_to_the_rules_engine():
    rng = np.random.default_rng(5)
    n = 600
    lines = np.column_stack(
        [
            rng.integers(1, 5, n),
            rng.choice([0, 30, 88], n),
            rng.integers(0, 3, n),
            rng.integers(0, 3, n),
            rng.integers(0, 2, n),
            rng.integers(0, 5, n),
            rng.integers(0, 9, n),
            rng.integers(0, 2, n),
            rng.integers(0, 2, n),
            rng.integers(0, 4, n),
            rng.choice([0, 10, 12], n),
        ]
    )
    comp = component_points(*lines.T, season="2026-27")
    assert set(comp) == set(COMPONENTS)
    total = sum(comp.values())
    assert np.allclose(total, score_lines(lines, "2026-27"))
    names = [k for k, _ in LINE_STATS]
    row = lines[0]
    stats = dict(zip(names[1:], map(int, row[1:]), strict=True))
    assert total[0] == season_2026_27.points_from_stats(stats, int(row[0]))["total"]


def test_bonus_link_table_and_linked_simulation_keep_the_mean():
    rows = pd.DataFrame(
        {
            "element_type": [3] * 400,
            "goals_scored": [1] * 100 + [0] * 300,
            "assists": [0] * 400,
            "clean_sheets": [0] * 400,
            "minutes": [90] * 400,
            "bonus": [2] * 100 + [0] * 300,
        }
    )
    t = bonus_link_table(rows, pseudo_rows=0.0)
    assert t[3, 1, 0, 0] == pytest.approx(4.0) and t[3, 0, 0, 0] == pytest.approx(0.0, abs=1e-9)
    assert t[2, 1, 1, 1] == 1.0  # empty cell
    from fpl_rocket.model.simulate import simulate_gw
    from test_model import _sim_rows

    sim_rows, fx = _sim_rows()
    out = {}
    for mode in ("round", "poisson", "linked"):
        link = bonus_link_table(rows) if mode == "linked" else None
        sim = simulate_gw(
            sim_rows,
            fx,
            n_sims=20000,
            rng=np.random.default_rng(3),
            season="2026-27",
            assist_rate=0.9,
            bonus_sim=mode,
            bonus_link=link,
        )
        out[mode] = sim
        assert (sim["bonus"][sim["minutes"] == 0] == 0).all() and sim["bonus"].max() <= 3
    for mode in ("poisson", "linked"):
        assert out[mode]["bonus"].mean() == pytest.approx(out["round"]["bonus"].mean(), rel=0.05)
    linked = out["linked"]
    scored = linked["goals"] > 0
    assert linked["bonus"][scored].mean() > 2 * linked["bonus"][~scored & (linked["minutes"] > 0)].mean()
    with pytest.raises(ValueError):
        simulate_gw(
            sim_rows,
            fx,
            n_sims=10,
            rng=np.random.default_rng(0),
            season="2026-27",
            assist_rate=0.9,
            bonus_sim="bogus",
        )


# ------------------------------------------------------------------------------------------ statistics
def test_block_bootstrap_ci():
    m, lo, hi = D.block_bootstrap_ci([np.full(10, 2.0), np.full(7, 2.0)])
    assert (m, lo, hi) == (2.0, 2.0, 2.0)
    rng = np.random.default_rng(1)
    x = [rng.normal(1.0, 1.0, 38), rng.normal(1.0, 1.0, 38)]
    m, lo, hi = D.block_bootstrap_ci(x, n_boot=500)
    assert lo < m < hi and lo < 1.0 < hi and m == pytest.approx(np.concatenate(x).mean())
    # a short series (6 GWs) works with block 4; NaNs are dropped
    m, lo, hi = D.block_bootstrap_ci([np.array([1, 2, np.nan, 3, 4, 5, 6.0])])
    assert m == pytest.approx(3.5) and lo <= m <= hi
    assert np.isnan(D.block_bootstrap_ci([np.array([np.nan])])[0])


def test_sign_test_and_paired_comparison():
    k, n, p = D.sign_test(np.array([-1, -2, -0.5, 0, 1.0]), lower_is_better=True)
    assert (k, n) == (3, 4) and 0 < p <= 1
    rows = []
    for season in ("A", "B"):
        for gw in range(1, 11):
            rows.append({"season": season, "gw": gw, "predictor": "m", "rmse": 1.0, "spearman_played": 0.5})
            rows.append({"season": season, "gw": gw, "predictor": "b", "rmse": 1.5, "spearman_played": 0.3})
    t = D.paired_comparison(
        pd.DataFrame(rows), "m", "b", metrics=(("rmse", True), ("spearman_played", False))
    )
    r = t.set_index("metric")
    assert r.loc["rmse", "mean_diff"] == pytest.approx(-0.5) and r.loc["rmse", "a_better_gws"] == 20
    assert r.loc["spearman_played", "mean_diff"] == pytest.approx(0.2)
    assert r.loc["rmse", "sign_p"] < 1e-4 and r.loc["rmse", "gws"] == 20


# ------------------------------------------------------------------------------------------ minutes
def test_persistence_and_frequency_comparators():
    rows = pd.DataFrame(
        {
            "bin0_1": [0, 1, 1, np.nan],
            "bin1_1": [0, 1, 0, np.nan],
            "bin2_1": [1, 0, 0, np.nan],
            "bin0_5": [0, 2, 5, np.nan],
            "bin1_5": [1, 2, 0, np.nan],
            "bin2_5": [4, 1, 0, np.nan],
        }
    )
    p = D.persistence_probs(rows)
    assert p.argmax(axis=1).tolist() == [2, 1, 0, 0] and np.allclose(p.max(axis=1), 0.9)
    base = np.array([0.6, 0.1, 0.3])
    f = D.frequency_probs(rows, base)
    assert f[0].tolist() == pytest.approx([0.5 / 6.5, 1.5 / 6.5, 4.5 / 6.5])
    assert f[3].tolist() == base.tolist() and np.allclose(f.sum(axis=1), 1.0)
    assert D.log_loss_probs(np.array([[1.0, 0, 0]]), np.array([0])) == pytest.approx(0.0, abs=1e-9)
    assert D.log_loss_probs(np.array([[0.5, 0.25, 0.25]]), np.array([1])) == pytest.approx(np.log(4))


# ------------------------------------------------------------------------------------------ goalkeepers
def test_gk_diagnostic_collapse_test():
    g = pd.DataFrame(
        {
            "season": "S",
            "gw": 1,
            "element_type": 1,
            "element": range(6),
            "actual": [6.0, 2.0, 0, 0, 0, 0],
            "minutes": [90, 90, 0, 0, 0, 0],
            "xp_m": [3.0, 3.5, 0.03, 0.01, 0.04, 0.02],
            "pred_base": [3.0, 3.5, 0, 0, 0, 0],
            "started_last": [True, True, False, False, False, False],
            "zero_last3": [False, False, True, True, True, True],
            "p_play": [0.95, 0.9, 0.05, 0.02, 0.04, 0.03],
        }
    )
    r = D.gk_diagnostic(g, "xp_m").iloc[0]
    assert r["actual_zero"] == 4 and r["base_pred_zero"] == 4 and r["n_contender"] == 2
    assert r["sp_all_base"] > r["sp_all_model"]  # the model breaks the tie of the zero group arbitrarily
    assert r["sp_all_model_collapsed"] == pytest.approx(r["sp_all_base"])  # ties explain the gap


# ------------------------------------------------------------------------------------------ intervals
def test_pinball_exceedance_and_naive_quantiles():
    assert D.pinball(np.array([3.0]), np.array([1.0]), 0.8) == pytest.approx(1.6)
    assert D.pinball(np.array([0.0]), np.array([1.0]), 0.8) == pytest.approx(0.2)
    f = pd.DataFrame(
        {
            "actual": [0, 2, 5, 9],
            "p20": [1, 1, 1, 1],
            "p80": [6, 6, 6, 6],
            "p_below": [0.1] * 4,
            "p_above": [0.15] * 4,
            "p_play": [0.95, 0.6, 0.2, 0.92],
        }
    )
    t = D.exceedance_table(f, "", []).iloc[0]
    assert (t["below_p20"], t["inside"], t["above_p80"]) == (0.25, 0.5, 0.25)
    assert t["model_above_p80"] == pytest.approx(0.15)
    assert D.play_group(f["p_play"]).tolist() == [">=0.9", "0.5-0.9", "<0.5", ">=0.9"]
    train = pd.DataFrame(
        {
            "season": "S",
            "gw": list(range(1, 11)),
            "element": 1,
            "element_type": 2,
            "total_points": [0, 0, 0, 1, 2, 2, 2, 6, 8, 2],
        }
    )
    q = D.naive_quantiles(train).iloc[0]
    assert (q["nq20"], q["nq50"], q["nq80"]) == (0.0, 2.0, 2.0)


def test_actual_components_reconcile_with_total_points(df):
    target = df[(df["season"] == SEASON2) & (df["gw"] == 3)]
    a = D.actual_components(target)
    tot = target.groupby("element")["total_points"].sum()
    comp = a.set_index("element")[[f"A_{c}" for c in COMPONENTS] + ["A_other"]].sum(axis=1)
    assert np.allclose(comp.reindex(tot.index), tot)
    assert (a["A_other"] == 0).all()  # synthetic points come from the same stats


# ------------------------------------------------------------------------------------------ folds
def test_run_fold_variants_comparators_and_components(df):
    v01 = {"newcomer_fallback": "none", "bonus_sim": "round", "defcon_fallback": False}
    vs = [
        D.Variant("v01", v01, collect=True),
        D.Variant("prior", {**v01, "newcomer_fallback": "prior"}),
        D.Variant("noshrink", v01, refit="noshrink"),
        D.Variant("naive", v01, refit="naivegoals"),
        D.Variant("pers", v01, refit="persistence"),
        D.Variant("noprev", v01, refit="noprev"),
    ]
    out = D.run_fold(df[df["gw"] <= 8], SEASON2, 8, vs, cfg=FAST, n_sims=100)
    p, lines, met = out["players"], out["lines"], out["metrics"]
    assert set(met["predictor"]) == {"baseline", "v01", "prior", "noshrink", "naive", "pers", "noprev"}
    assert p["low_history"].any() and {"xp_v01", "v01__p80", "v01__E_goals", "A_goals", "nq80"} <= set(
        p.columns
    )
    comps = p[[f"v01__E_{c}" for c in COMPONENTS]].sum(axis=1)
    assert np.allclose(comps, p["xp_v01"])  # component expectations add up to xP
    assert p["v01__p_above"].between(0, 0.2 + 1e-9).all()  # integer quantile: at most 20% above
    for k in ("hgb", "pers", "freq"):
        assert np.allclose(lines[[f"{k}0", f"{k}1", f"{k}2"]].sum(axis=1), 1.0)
    assert set(lines["bin"]) <= {0, 1, 2}
    gk = D.gk_diagnostic(p.assign(p_play=p["v01__p_play"]), "xp_v01")
    assert len(gk) == 1 and gk["n_gk"].iloc[0] == (p["element_type"] == 1).sum()
    # the persistence ablation equals the persistence comparator on the classifier's lines
    assert not np.allclose(p["xp_pers"], p["xp_v01"])
    # GW1 has no current-season rows: the no-previous-season ablation cannot predict it
    small = df[(df["season"] == SEASON1) | (df["gw"] <= 2)]
    out1 = D.run_fold(small, SEASON2, 1, [D.Variant("noprev", v01, refit="noprev")], cfg=FAST)
    assert out1["players"]["xp_noprev"].isna().all()


def test_horizon_origin_and_stability(df):
    cfg = replace(FAST, horizons=(1, 2, 3))
    h = D.run_horizon_origin(df, SEASON2, 6, cfg, max_h=3, n_sims=80)
    assert sorted(h["h"].unique()) == [1, 2, 3] and set(h["gw"]) == {7, 8, 9}
    assert {"model", "naive", "base", "actual"} <= set(h.columns)
    h1 = h[h["h"] == 1].set_index("element")
    assert np.allclose(h1["naive"], h1["model"])  # horizon 1: the naive IS the model
    # a double GW scales the naive by the fixture count
    for g in (2, 3):
        x = h[h["h"] == g].set_index("element")
        common = x.index.intersection(h1.index)
        ratio = x.loc[common, "naive"] / h1.loc[common, "model"].where(h1.loc[common, "model"] > 0)
        nf = x.loc[common, "n_fixtures"] / h1.loc[common, "n_fixtures"]
        ok = ratio.notna()
        assert np.allclose(ratio[ok], nf[ok])
    st = D.run_stability(df[df["gw"] <= 8], SEASON2, 8, FAST, [(100, 1), (100, 2), (400, 1)])
    c = D.stability_compare(st, "n100_s1", "n100_s2", top=10)
    assert c["max_abs_diff"] > 0 and 0 <= c["top_changed"] <= 10
    assert D.stability_compare(st, "n100_s1", "n100_s1")["max_abs_diff"] == 0
