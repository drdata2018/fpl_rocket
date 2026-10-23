"""Decision-level diagnostics (step 3c) on synthetic data: captaincy decomposition, top-k candidates, the
squared-error decomposition, reliability / Brier, fixture-combined probabilities, the baseline rate, and the
decision fold / captain seed study on the synthetic league."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from fpl_rocket.backtest import FormPer90Baseline, xi_points  # noqa: E402
from fpl_rocket.model import decisions as X  # noqa: E402
from fpl_rocket.model.config import ModelConfig  # noqa: E402
from model_synth import SEASON2, world  # noqa: E402

FAST = ModelConfig(n_sims=150, horizons=(1,), hgb_max_iter=30, bonus_max_iter=30)


def squad(pred: list[float], actual: list[float], minutes: list[float] | None = None) -> pd.DataFrame:
    """15 players of 15 different clubs: 2 GK, 5 DEF, 5 MID, 3 FWD (no club cap binds)."""
    et = [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4]
    return pd.DataFrame(
        {
            "element": np.arange(1, 16),
            "element_type": et,
            "team_id": np.arange(1, 16),
            "pred": pred,
            "actual": actual,
            "minutes": minutes if minutes is not None else [90.0] * 15,
        }
    )


def test_captain_decomposition_adds_up_and_ranks_the_captain():
    pred = [5, 1, 4, 3.9, 3.8, 1, 1, 9, 6, 5.5, 1, 1, 7, 2, 1]
    actual = [2.0, 0, 6, 1, 2, 0, 3, 12, 2, 2, 15, 1, 5, 2, 0]
    f = squad(pred, actual)
    d = X.captain_decomposition(f, "pred")
    total, cap = xi_points(f)
    assert d["total"] == total and d["captain"] == cap == 8
    assert d["xi_no_cap"] + d["cap_bonus"] == d["total"] and d["cap_bonus"] == 12
    # 12 points: one player (15) above -> rank 2, top 10 / 50, mid-rank percentile (13 below) / 15
    assert d["cap_rank"] == 2 and d["cap_top10"] and d["cap_top50"]
    assert d["cap_pct"] == pytest.approx((13 + 0.5) / 15)
    # a captain who did not play: 0 bonus, ranked by his 0 among the players who did
    mins = [90.0] * 15
    mins[7] = 0.0
    d0 = X.captain_decomposition(squad(pred, [*actual[:7], 0.0, *actual[8:]], mins), "pred")
    assert not d0["cap_played"] and d0["cap_bonus"] == 0 and d0["n_played"] == 14
    assert d0["cap_rank"] == 1 + sum(a > 0 for i, a in enumerate(actual) if i != 7)


def test_top_candidates():
    f = squad(list(range(15, 0, -1)), [float(x) for x in range(15)], [0.0] + [90.0] * 14)
    t = X.top_candidates(f, "pred").set_index("k")
    assert t.loc[1, "pred_mean"] == 15 and t.loc[1, "actual_mean"] == 0 and t.loc[1, "played_share"] == 0
    assert t.loc[3, "actual_mean"] == pytest.approx(1.0) and t.loc[5, "pred_mean"] == pytest.approx(13.0)


def test_error_decomposition_adds_up():
    rng = np.random.default_rng(0)
    n = 300
    f = pd.DataFrame(
        {
            "actual": rng.poisson(2, n).astype(float),
            "a": rng.normal(2, 1, n),
            "b": rng.normal(2, 2, n),
            "mclass": X.minutes_class(rng.integers(0, 3, n)),
        }
    )
    t = X.error_decomposition(f, "a", "b").set_index("mclass")
    parts = t.drop(index="all")
    assert set(parts.index) == set(X.MINUTES_CLASSES)
    mse_a = np.mean((f["a"] - f["actual"]) ** 2)
    mse_b = np.mean((f["b"] - f["actual"]) ** 2)
    assert parts["mse_contrib_a"].sum() == pytest.approx(mse_a)
    assert parts["mse_reduction"].sum() == pytest.approx(mse_b - mse_a)
    assert parts["share_of_reduction"].sum() == pytest.approx(1.0)
    assert t.loc["all", "rmse_a"] == pytest.approx(np.sqrt(mse_a)) and t.loc["all", "n"] == n
    # a group where both are equal carries no reduction
    g = f.assign(b=np.where(f["mclass"] == "0 min", f["a"], f["b"]))
    assert X.error_decomposition(g, "a", "b").set_index("mclass").loc["0 min", "mse_reduction"] == 0


def test_minutes_class_double_gw():
    assert list(X.minutes_class([0, 1, 2])) == ["0 min", "1-59 min", "60+ min"]


def test_reliability_and_brier():
    p = np.array([0.05, 0.05, 0.95, 0.95, 0.55, 1.0])
    y = np.array([0, 1, 1, 1, 0, 1])
    t = X.reliability_table(p, y).set_index("bin")
    assert t["n"].sum() == 6 and len(t) == 10
    assert t.loc["0.0-0.1", "observed"] == 0.5 and t.loc["0.9-1.0", "n"] == 3  # 1.0 falls in the last bin
    assert t.loc["0.5-0.6", "gap"] == pytest.approx(-0.55) and np.isnan(t.loc["0.2-0.3", "observed"])
    assert X.brier(p, y) == pytest.approx(np.mean((p - y) ** 2))
    assert X.brier([1, 0], [1, 0]) == 0


def test_combine_fixture_probs_double_gw():
    lines = pd.DataFrame(
        {"element": [1, 2, 2], "q0": [0.2, 0.5, 0.4], "q1": [0.3, 0.2, 0.2], "q2": [0.5, 0.3, 0.4]}
    )
    c = X.combine_fixture_probs(lines, "q")
    assert c.loc[1, "q_play"] == pytest.approx(0.8) and c.loc[1, "q_60"] == pytest.approx(0.5)
    assert c.loc[2, "q_play"] == pytest.approx(1 - 0.5 * 0.4) and c.loc[2, "q_60"] == pytest.approx(
        1 - 0.7 * 0.6
    )


def test_baseline_p90_is_the_baseline_rate():
    df = world(n_gws=6)
    train = df[(df["season"] != SEASON2) | (df["gw"] < 4)]
    p90 = X.baseline_p90(train)
    base = FormPer90Baseline()
    base.fit(train)
    common = base.xp_per_fixture.index[base.xp_per_fixture > 0]
    # regulars play 90 every GW: availability 1, so the baseline xP per fixture IS the rate
    regular = [k for k in common if abs(base.xp_per_fixture[k] - p90[k]) < 1e-9]
    assert len(regular) > 100
    assert (p90.reindex(common) >= base.xp_per_fixture[common] - 1e-9).all()  # availability <= 1
    assert X.position_p90(train).index.tolist() == [1, 2, 3, 4]


def test_xi_changed():
    assert X.xi_changed("1 2 3", "1 2 3") == 0 and X.xi_changed("1 2 3", "1 2 4") == 1


@pytest.fixture(scope="module")
def small() -> pd.DataFrame:
    return world(n_gws=8)


def test_decision_fold(small):
    out = X.run_decision_fold(small, SEASON2, 8, cfg=FAST)
    p, t2 = out["players"], out["top2"]
    assert len(t2) == 1 and t2["xp_top1"].iloc[0] >= t2["xp_top2"].iloc[0] and t2["gap_se"].iloc[0] > 0
    assert t2["gap"].iloc[0] == pytest.approx(t2["xp_top1"].iloc[0] - t2["xp_top2"].iloc[0])
    for c in ("p_play", "p_60", "pers_play", "pers_60", "freq_play", "freq_60"):
        assert p[c].between(0, 1).all(), c
    assert (p["p_60"] <= p["p_play"] + 1e-12).all() and (p["pers_60"] <= p["pers_play"] + 1e-12).all()
    assert set(p["max_bin"]) <= {0, 1, 2} and ((p["max_bin"] > 0) == (p["minutes"] > 0)).all()
    # conditional on 60+ minutes the xP is at least the appearance points and above the unconditional xP
    ok = p["n_sims60"] > 20
    assert (p.loc[ok, "cond60"] >= p.loc[ok, "xp"] - 1e-9).all()
    assert np.allclose(p["pred_base"], p["pred_base"].clip(lower=0))


def test_captain_seeds(small):
    st = X.run_captain_seeds(small, SEASON2, 8, FAST, [(200, 1), (200, 1), (100, 2)])
    assert len(st) == 3 and (st["ref_a"] == st["top1"].iloc[0]).all()
    a, b = st.iloc[0], st.iloc[1]  # the same setting twice: identical runs
    assert a["xi"] == b["xi"] and a["ref_diff"] == b["ref_diff"] and a["captain"] == b["captain"]
    assert a["ref_diff"] == pytest.approx(a["gap_top2"]) and len(a["xi"].split()) == 11
    assert (st["ref_diff_se"] > 0).all()
