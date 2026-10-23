"""xP model components on synthetic data: minutes, team goals, shares, joint simulation, scoring, outputs."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from fpl_rocket.backtest import FormPer90Baseline, run_backtest  # noqa: E402
from fpl_rocket.bundle import read_bundle, read_manifest, write_tables  # noqa: E402
from fpl_rocket.model.config import ModelConfig  # noqa: E402
from fpl_rocket.model.features import attach_states, player_states, season_t  # noqa: E402
from fpl_rocket.model.minutes import availability_factor  # noqa: E402
from fpl_rocket.model.pipeline import (  # noqa: E402
    XP_COLUMNS,
    ModelPredictor,
    XPModel,
    bundle_history_rows,
    bundle_targets,
    run_bundle_model,
)
from fpl_rocket.model.rates import PlayerRates  # noqa: E402
from fpl_rocket.model.simulate import LINE_STATS, ROW_COLUMNS, score_lines, simulate_gw  # noqa: E402
from fpl_rocket.model.team_goals import (  # noqa: E402
    DixonColes,
    league_table,
    results_from_rows,
    score_matrix,
)
from fpl_rocket.rules import season_2026_27  # noqa: E402
from model_synth import SEASON1, SEASON2, bundle_tables, clubs, strengths, world  # noqa: E402

FAST = ModelConfig(n_sims=200, horizons=(1, 2, 3), hgb_max_iter=50, bonus_max_iter=50)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return world()


@pytest.fixture(scope="module")
def fitted(df) -> tuple[XPModel, pd.DataFrame]:
    """Model fitted on SEASON1 + SEASON2 GW1-7; targets = SEASON2 GW8 (horizon 1)."""
    train = df[(df["season"] == SEASON1) | (df["gw"] < 8)]
    target = df[(df["season"] == SEASON2) & (df["gw"] == 8)]
    ref = target["kickoff_time"].min()
    model = XPModel(FAST).fit(train, season=SEASON2, ref_time=ref, season_teams=clubs(SEASON2))
    return model, target.drop(columns=["minutes", "total_points"]).assign(horizon=1)


# ------------------------------------------------------------------------------------------ team goals
def _dc_world(n_rounds: int = 30, seed: int = 1) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    teams = [f"C{i}" for i in range(12)]
    att = {t: 0.4 - 0.07 * i for i, t in enumerate(teams)}
    dfc = {t: -0.3 + 0.05 * i for i, t in enumerate(teams)}
    rows, kick = [], pd.Timestamp("2025-08-01", tz="UTC")
    fixture = 0
    for r in range(n_rounds):
        order = rng.permutation(teams)
        for k in range(6):
            h, a = order[2 * k], order[2 * k + 1]
            fixture += 1
            lam = np.exp(0.1 + 0.3 + att[h] + dfc[a])
            mu = np.exp(0.1 + att[a] + dfc[h])
            rows.append(
                {
                    "season": SEASON2,
                    "fixture": fixture,
                    "gw": r + 1,
                    "kickoff_time": kick + pd.Timedelta(days=3 * r),
                    "home": h,
                    "away": a,
                    "hg": rng.poisson(lam),
                    "ag": rng.poisson(mu),
                }
            )
    return pd.DataFrame(rows), att


def test_dixon_coles_recovers_strengths_and_home_advantage():
    res, att = _dc_world()
    dc = DixonColes(ModelConfig(dc_xi=0.0)).fit(
        res, ref_time=pd.Timestamp("2027-01-01", tz="UTC"), season=SEASON2
    )
    assert dc.converged
    est = np.array([dc.attack[t] for t in att])
    assert np.corrcoef(est, np.array(list(att.values())))[0, 1] > 0.9
    assert 0.15 < dc.home < 0.45
    m = score_matrix(*dc.rates("C0", "C11"), dc.rho)
    assert m.sum() == pytest.approx(1.0) and m.shape == (11, 11)
    fx = dc.fixture_frame(pd.DataFrame({"fixture": [1], "home": ["C0"], "away": ["C11"]}))
    assert fx["lambda_home"].iloc[0] > fx["lambda_away"].iloc[0]
    gc = fx[[f"p_home_gc_{k}" for k in range(7)]].sum(axis=1).iloc[0]
    assert gc == pytest.approx(1.0) and fx["p_cs_home"].iloc[0] == pytest.approx(fx["p_home_gc_0"].iloc[0])


def test_dixon_coles_gradient_is_exact():
    from scipy.optimize import check_grad

    res, _ = _dc_world(10)
    nll, theta0, _, _, _ = DixonColes().objective(res, pd.Timestamp("2026-06-01", tz="UTC"), SEASON2, None)
    theta = theta0 + np.random.default_rng(0).normal(0, 0.1, len(theta0))
    theta[-1] = 0.05
    assert check_grad(lambda z: nll(z)[0], lambda z: nll(z)[1], theta) < 1e-3


def test_time_decay_downweights_old_matches():
    res, _ = _dc_world()
    ref = res["kickoff_time"].max() + pd.Timedelta(days=1)
    flipped = res.copy()
    old = flipped["gw"] <= 15  # in old matches C0 is terrible
    flipped.loc[old & (flipped["home"] == "C0"), "hg"] = 0
    flipped.loc[old & (flipped["away"] == "C0"), "ag"] = 0
    a_decay = DixonColes(ModelConfig(dc_xi=0.05)).fit(flipped, ref_time=ref, season=SEASON2).attack["C0"]
    a_flat = DixonColes(ModelConfig(dc_xi=0.0)).fit(flipped, ref_time=ref, season=SEASON2).attack["C0"]
    assert a_decay > a_flat


def test_promoted_clubs_start_at_bottom_three_mean(df):
    res = results_from_rows(df[df["season"] == SEASON1])
    ref = pd.Timestamp(datetime(2026, 8, 1, tzinfo=UTC))
    dc = DixonColes().fit(res, ref_time=ref, season=SEASON2, season_teams=clubs(SEASON2))
    bottom = list(league_table(res)["team"].iloc[-3:])
    assert dc.promoted == ["P00", "P01", "P02"] and dc.bottom_three == bottom
    m_att = np.mean([dc.attack[t] for t in bottom])
    m_def = np.mean([dc.defence[t] for t in bottom])
    for p in dc.promoted:  # no current-season match yet: the prior alone decides
        assert dc.attack[p] == pytest.approx(m_att, abs=0.02) and dc.defence[p] == pytest.approx(
            m_def, abs=0.02
        )


def test_results_from_rows_counts_opponent_own_goals():
    rows = pd.DataFrame(
        {
            "season": ["s"] * 4,
            "fixture": [1] * 4,
            "gw": [1] * 4,
            "kickoff_time": [pd.Timestamp("2026-08-01", tz="UTC")] * 4,
            "team": ["A", "A", "B", "B"],
            "was_home": [True, True, False, False],
            "goals_scored": [2, 0, 1, 0],
            "own_goals": [0, 1, 0, 1],
        }
    )
    r = results_from_rows(rows).iloc[0]
    assert (r["home"], r["away"], r["hg"], r["ag"]) == ("A", "B", 3, 2)


# ------------------------------------------------------------------------------------------ minutes
def test_states_never_use_the_target_or_later_gws():
    rows = pd.DataFrame(
        {
            "season": [SEASON2] * 4,
            "gw": [1, 2, 3, 4],
            "element": [1] * 4,
            "code": [7] * 4,
            "team": ["A"] * 4,
            "fixture": [1, 2, 3, 4],
            "minutes": [90, 0, 90, 45],
            "starts": [1, 0, 1, 0],
        }
    )
    states = player_states(rows)
    tg = pd.DataFrame(
        {"pkey": [7, 7, 7], "season": [SEASON2] * 3, "t": [season_t(SEASON2, 4)] * 3, "horizon": [1, 2, 3]}
    )
    out = attach_states(tg, states)
    assert list(out["state_t"]) == [season_t(SEASON2, g) for g in (3, 2, 1)]
    assert out["min_pf_1"].tolist() == [90.0, 0.0, 90.0]
    assert out["min_pf_3"].iloc[0] == pytest.approx(60.0)  # GW1-3, never GW4


def test_minutes_model_separates_regulars_bench_and_unused(fitted):
    model, target = fitted
    rows, _ = model.lines(target)
    slot = rows["element"] % 100
    regular = rows[slot.isin([0, 2, 7, 12])]
    unused = rows[slot.isin([1, 6, 14])]
    sub = rows[slot == 11]
    assert (regular["p2"] > 0.8).all()
    assert (unused["p0"] > 0.8).all()
    assert (sub["p1"] > 0.6).all() and sub["m_short"].between(10, 45).all()
    assert np.allclose(rows[["p0", "p1", "p2"]].sum(axis=1), 1.0)


def test_availability_factor_live_flags():
    f = availability_factor(
        pd.Series(["a", "d", "i", "i", "u", "n"]),
        pd.Series([None, 75, 0, None, None, None]),
        pd.Series([1, 1, 1, 3, 1, 1]),
        decay=0.5,
    )
    assert f.tolist() == [1.0, 0.75, 0.0, pytest.approx(0.75), 0.0, 0.0]
    assert (availability_factor(None, None, pd.Series([1, 2]), 0.5) == 1.0).all()


# ------------------------------------------------------------------------------------------ shares
def test_shares_shrink_to_position_prior():
    cfg = ModelConfig()
    base = {"season": SEASON2, "gw": 1, "team": "A", "fixture": 1, "was_home": True, "goals_scored": 0}
    rows = pd.DataFrame(
        [
            {**base, "element": 1, "code": 1, "element_type": 4, "minutes": 900, "expected_goals": 9.0},
            {**base, "element": 2, "code": 2, "element_type": 4, "minutes": 90, "expected_goals": 0.0},
            {**base, "element": 3, "code": 3, "element_type": 4, "minutes": 0, "expected_goals": 0.0},
        ]
    ).assign(expected_assists=0.0, yellow_cards=0, red_cards=0, saves=0, own_goals=0, assists=0)
    r = PlayerRates(cfg).fit(rows, SEASON2)
    prior = r.priors.loc[4, "xg90"]
    assert prior == pytest.approx(9.0 / 990 * 90)
    p = cfg.share_pseudo_minutes
    assert r.players.loc[1, "xg90"] == pytest.approx((9.0 + prior * p / 90) / ((900 + p) / 90))
    assert r.players.loc[2, "xg90"] == pytest.approx((0.0 + prior * p / 90) / ((90 + p) / 90))
    assert r.players.loc[3, "xg90"] == pytest.approx(prior)  # no minutes: the prior
    new = pd.DataFrame({"element": [99], "code": [99], "element_type": [4], "team": ["A"]})
    par = r.params(new)
    assert par["xg90"].iloc[0] == pytest.approx(prior)
    assert par["share_g"].iloc[0] == pytest.approx(prior / r.team_xg["A"])


def test_model_shares_follow_the_synthetic_truth(fitted):
    model, target = fitted
    rows, _ = model.lines(target)
    by_slot = rows.assign(slot=rows["element"] % 100).groupby("slot")["share_g"].mean()
    assert by_slot[12] > by_slot[13] > by_slot[7] > by_slot[2]  # FWD1 > FWD2 > MID > DEF


# ------------------------------------------------------------------------------------------ simulation
def _sim_rows(n_teams: int = 2, per_team: int = 6, share: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for t in range(n_teams):
        for j in range(per_team):
            rows.append(
                {
                    "element": t * 100 + j,
                    "fixture": 1,
                    "was_home": t == 0,
                    "element_type": 1 if j == 0 else 3,
                    "p0": 0.1,
                    "p1": 0.2,
                    "p2": 0.7,
                    "m_short": 30.0,
                    "m_long": 88.0,
                    "share_g": share,  # 6 x 0.5 = 3 > 1: weights must be normalised
                    "share_a": share,
                    "yc90": 0.2,
                    "rc90": 0.01,
                    "saves_mu90": 3.0 if j == 0 else 0.0,
                    "dc90": 9.0,
                    "dc_threshold": np.nan if j == 0 else 12.0,
                    "bonus_short": 0.2,
                    "bonus_long": 0.6,
                }
            )
    fx = pd.DataFrame({"fixture": [1], "lambda_home": [2.5], "lambda_away": [2.0], "rho": [-0.05]})
    return pd.DataFrame(rows, columns=list(ROW_COLUMNS)), fx


def test_multinomial_allocation_never_exceeds_team_goals():
    rows, fx = _sim_rows()
    rng = np.random.default_rng(0)
    sim = simulate_gw(rows, fx, n_sims=2000, rng=rng, season="2026-27", assist_rate=0.9)
    home = rows["was_home"].to_numpy()
    scored = np.r_[sim["goals"][home].sum(axis=0)[None], sim["goals"][~home].sum(axis=0)[None]]
    assists = np.r_[sim["assists"][home].sum(axis=0)[None], sim["assists"][~home].sum(axis=0)[None]]
    totals = sim["fixture_goals"][0]  # (2, S) simulated team goals (home, away)
    assert (scored <= totals).all() and (assists <= totals).all()
    assert (sim["goals"] <= np.where(home[:, None], totals[0], totals[1])).all()
    assert (sim["goals"][sim["minutes"] == 0] == 0).all() and (sim["assists"][sim["minutes"] == 0] == 0).all()
    assert scored.sum() > 0.8 * totals.sum()  # nearly every goal attributed when weights exceed 1


def test_clean_sheet_requires_60_minutes_and_no_goals_against():
    rows, fx = _sim_rows()
    sim = simulate_gw(rows, fx, n_sims=2000, rng=np.random.default_rng(1), season="2026-27", assist_rate=0.9)
    home = rows["was_home"].to_numpy()
    goals = sim["fixture_goals"][0]
    against = np.where(home[:, None], goals[1], goals[0])
    assert (sim["goals_conceded"][sim["minutes"] >= 60] == against[sim["minutes"] >= 60]).all()
    cs = sim["cs_points"].astype(bool)
    assert (sim["minutes"][cs] >= 60).all() and (against[cs] == 0).all()
    assert ((sim["minutes"] >= 60) & (against == 0) == cs).all()


def test_simulated_lines_are_scored_by_the_rules_engine(monkeypatch):
    calls = []

    def fake(stats, element_type, table=season_2026_27.SCORING):
        calls.append((element_type, dict(stats)))
        return {"total": 7 if stats.get("minutes", 0) > 0 else 0}

    monkeypatch.setattr(season_2026_27, "points_from_stats", fake)
    rows, fx = _sim_rows()
    sim = simulate_gw(rows, fx, n_sims=300, rng=np.random.default_rng(2), season="2026-27", assist_rate=0.9)
    assert calls, "the rules engine was never called"
    assert set(np.unique(sim["points"])) <= {0, 7}
    assert ((sim["points"] == 7) == (sim["minutes"] > 0)).all()
    assert len(calls) == len({(e, tuple(sorted(s.items()))) for e, s in calls})  # once per distinct line


def test_score_lines_match_the_rules_engine_line_by_line():
    rng = np.random.default_rng(3)
    n = 500
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
    pts = score_lines(lines, "2026-27")
    names = [k for k, _ in LINE_STATS]
    for row, p in zip(lines, pts, strict=True):
        stats = dict(zip(names[1:], map(int, row[1:]), strict=True))
        assert p == season_2026_27.points_from_stats(stats, int(row[0]))["total"]


# ------------------------------------------------------------------------------------------ outputs
def test_xp_output_schema_and_sample_consistency(fitted):
    model, target = fitted
    out = model.predict(target, n_sims=200)
    xp, sims = out.xp, out.sims
    assert tuple(xp.columns) == XP_COLUMNS
    assert not xp.duplicated(["element", "gw"]).any()
    assert (xp["p20"] <= xp["p50"]).all() and (xp["p50"] <= xp["p80"]).all()
    assert xp[["p_play", "p_60", "p_cs"]].stack().between(0, 1).all()
    assert (xp["p_60"] <= xp["p_play"] + 1e-12).all()
    assert list(sims.columns) == ["element", "gw", "sim", "points"] and sims["points"].dtype == np.int8
    assert len(sims) == len(xp) * 200
    mean = sims.groupby(["element", "gw"])["points"].mean()
    j = xp.set_index(["element", "gw"])["xp"]
    assert np.allclose(j.sort_index(), mean.sort_index())
    tf = out.fixtures
    assert len(tf) == 2 * target["fixture"].nunique() and {"xg_for", "xg_against", "p_cs"} <= set(tf.columns)


def test_bundle_model_outputs_and_manifest_fields(df, tmp_path):
    tables = bundle_tables(df, next_gw=8)
    hist = bundle_history_rows(tables, SEASON2)
    assert set(hist["gw"]) == set(range(1, 8)) and hist["team"].notna().all()
    targets, players = bundle_targets(tables, n_gws=3, season=SEASON2)
    assert sorted(targets["horizon"].unique()) == [1, 2, 3]
    in_progress = {**tables, "events": tables["events"].assign(finished=tables["events"]["id"] < 7)}
    assert sorted(bundle_targets(in_progress, n_gws=3, season=SEASON2)[0]["horizon"].unique()) == [2, 3, 4]
    cfg = ModelConfig(n_sims=50, horizons=(1, 2, 3), hgb_max_iter=30, bonus_max_iter=30)
    now = datetime(2026, 10, 10, tzinfo=UTC)
    out, man = run_bundle_model(
        tables, df[df["season"] == SEASON1], season=SEASON2, now=now, cfg=cfg, n_gws=3
    )
    xp = out["xp"]
    assert set(xp["gw"]) == {8, 9, 10} and len(xp) == 3 * len(players)  # every player, every GW
    gw8 = xp[xp["gw"] == 8]
    assert set(gw8["n_fixtures"]) == {0, 1, 2}  # blank, single and double GW handled
    assert (gw8.loc[gw8["n_fixtures"] == 0, "xp"] == 0).all()
    injured = xp[(xp["element"] % 100 == 2) & (xp["gw"] == 8)]
    assert (injured["p_play"] == 0).all()  # live flag: chance_of_playing_next_round 0
    assert xp.loc[xp["element"] % 100 == 12, "penalties_order"].eq(1.0).all()
    assert man["model_version"].startswith("v0.1-") and man["model_n_sims"] == 50
    assert man["model_fitted_on"]["seasons"] == [SEASON1, SEASON2] and man["model_fitted_on"]["promoted"]
    assert "minutes" in man["model_features"] and man["model_hyperparameters"]["dc_xi"] == 0.0065
    path = write_tables({**out}, tmp_path, {}, bundle_id="M1", extra=man)
    m = read_manifest(path)
    tables_back = read_bundle(path)
    assert {"xp", "xp_sims", "xp_fixtures"} <= set(tables_back) and m["model_version"] == man["model_version"]
    assert tables_back["xp_sims"]["points"].dtype == np.int8


def test_model_predictor_runs_in_the_backtest_harness(df):
    small = df[df["gw"] <= 9]
    res = run_backtest(small, [SEASON2], ModelPredictor(FAST, n_sims=100))
    base = run_backtest(small, [SEASON2], FormPer90Baseline())
    assert len(res.folds) == len(base.folds) == 9
    assert {"players", "minutes", "fixtures", "timing"} <= set(res.details)
    assert res.details["timing"]["fit_s"].gt(0).all()
    # GW1 of SEASON2 is predicted from SEASON1 only; promoted clubs exist
    assert set(res.details["fixtures"]["home"]) | set(res.details["fixtures"]["away"]) >= {"P00"}
    assert strengths()["T00"] > strengths()["P00"]
