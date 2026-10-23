"""Backtest harness on synthetic data: rolling origin, baseline predictor, metrics, decision metric."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fpl_rocket.backtest import (
    Fold,
    FormPer90Baseline,
    best_xi,
    calibration_table,
    gw_average_proxy,
    point_metrics,
    previous_season,
    rolling_origin,
    run_backtest,
    simulate_matched_policy,
    spearman,
    summarise,
    xi_points,
)


def _row(season, gw, element, minutes, points, *, code=None, et=3, team=1, fixture=None, selected=100):
    return {
        "season": season,
        "gw": gw,
        "element": element,
        "code": code if code is not None else 1000 + element,
        "element_type": et,
        "team_id": team,
        "fixture": fixture if fixture is not None else gw * 100 + team,
        "minutes": minutes,
        "total_points": points,
        "selected": selected,
    }


def _world() -> pd.DataFrame:
    """Two seasons x 4 GWs; player 1 = steady starter (90', 6 pts), 2 = sub (30', 1 pt), 3 = injured."""
    rows = []
    for season in ("2023-24", "2024-25"):
        for gw in range(1, 5):
            rows.append(_row(season, gw, 1, 90, 6))
            rows.append(_row(season, gw, 2, 30, 1, team=2))
            rows.append(_row(season, gw, 3, 0, 0, team=3))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ iterator
def test_rolling_origin_train_is_strictly_before_target_plus_previous_season():
    df = _world()
    folds = list(rolling_origin(df, ["2023-24", "2024-25"]))
    # 2023-24 GW1 has no train rows (no 2022-23) and is skipped
    assert [(f.season, f.target_gw) for f in folds] == [("2023-24", g) for g in (2, 3, 4)] + [
        ("2024-25", g) for g in (1, 2, 3, 4)
    ]
    for f in folds:
        same = f.train[f.train["season"] == f.season]
        assert (same["gw"] < f.target_gw).all()
        assert set(f.train["season"]) <= {f.season, previous_season(f.season)}
        assert (f.target["gw"] == f.target_gw).all() and (f.target["season"] == f.season).all()
    last = folds[-1]
    assert len(last.train) == 12 + 9  # whole previous season + GW1-3


def test_target_players_carry_no_outcomes_and_sum_double_gws():
    t = pd.DataFrame([_row("2024-25", 5, 1, 90, 6, fixture=1), _row("2024-25", 5, 1, 90, 2, fixture=2)])
    fold = Fold("2024-25", 5, t.iloc[0:0], t)
    tp = fold.target_players()
    assert "total_points" not in tp and "minutes" not in tp
    assert tp.loc[0, "n_fixtures"] == 2 and fold.actuals().loc[1] == 8


def test_previous_season_label():
    assert previous_season("2025-26") == "2024-25" and previous_season("2000-01") == "1999-00"


# ------------------------------------------------------------------------------------------ baseline
def test_baseline_formula():
    rows = [_row("2024-25", g, 1, m, p) for g, m, p in [(1, 90, 2), (2, 90, 10), (3, 45, 3), (4, 0, 0)]]
    rows += [_row("2024-25", g, 2, 0, 0, team=2) for g in (1, 2, 3, 4)]  # no minutes in last 3 -> 0
    rows += [_row("2024-25", g, 3, 90 if g == 1 else 0, 5 if g == 1 else 0, team=3) for g in (1, 2, 3, 4)]
    train = pd.DataFrame(rows)
    target = pd.DataFrame([_row("2024-25", 5, e, 0, 0, team=e) for e in (1, 2, 3, 9)])
    target.loc[target["element"] == 1, "fixture"] = [777]
    target = pd.concat([target, pd.DataFrame([_row("2024-25", 5, 1, 0, 0, fixture=778)])])  # double GW
    b = FormPer90Baseline()
    b.fit(train)
    tp = Fold("2024-25", 5, train, target).target_players()
    xp = dict(zip(tp["element"], b.predict(tp, 5), strict=True))
    p90 = (2 + 10 + 3) / (90 + 90 + 45) * 90  # last 5 played GWs
    avail = (90 + 90 + 45 + 0) / (90 * 4)  # last 5 GWs (only 4 exist)
    assert xp[1] == pytest.approx(p90 * avail * 2)  # two fixtures
    assert xp[2] == 0.0  # never played
    assert xp[3] == 0.0  # no minutes in the last 3 GWs
    assert xp[9] == 0.0  # unknown player


def test_baseline_uses_previous_season_through_code():
    rows = [_row("2023-24", g, 50, 90, 4, code=7) for g in (35, 36, 37, 38)]
    b = FormPer90Baseline()
    b.fit(pd.DataFrame(rows))
    tp = pd.DataFrame({"element": [3], "code": [7], "n_fixtures": [1]})  # new season-local id, same code
    assert b.predict(tp, 1).iloc[0] == pytest.approx(4.0)


# ------------------------------------------------------------------------------------------ metrics
def test_point_metrics_and_spearman():
    pred, actual = pd.Series([1.0, 2.0, 3.0, 4.0]), pd.Series([1.0, 3.0, 2.0, 6.0])
    m = point_metrics(pred, actual)
    assert m["rmse"] == pytest.approx(math.sqrt((0 + 1 + 1 + 4) / 4))
    assert m["mae"] == pytest.approx(1.0)
    assert m["spearman"] == pytest.approx(0.8)
    assert math.isnan(spearman(pd.Series([1, 1, 1]), pd.Series([1, 2, 3])))


def test_calibration_table_deciles():
    rng = np.random.default_rng(0)
    pred = pd.Series(rng.uniform(0, 10, 1000))
    cal = calibration_table(pred, pred + 1.0)
    assert len(cal) == 10 and cal["n"].sum() == 1000
    assert (cal["gap"].round(9) == 1.0).all() and cal["pred_mean"].is_monotonic_increasing


def _players(n_per_type=6, teams=10):
    rows, e = [], 0
    for et, base in ((1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)):
        for i in range(n_per_type):
            e += 1
            rows.append(
                {"element": e, "element_type": et, "team_id": e % teams, "pred": base + i, "actual": i}
            )
    return pd.DataFrame(rows)


def test_best_xi_is_legal_and_greedy():
    p = _players()
    xi = best_xi(p)
    counts = xi["element_type"].value_counts().to_dict()
    assert len(xi) == 11 and counts[1] == 1 and 3 <= counts[2] <= 5 and 2 <= counts[3] <= 5
    assert 1 <= counts[4] <= 3 and xi["team_id"].value_counts().max() <= 3
    assert counts == {1: 1, 2: 3, 3: 4, 4: 3}  # highest predictions within the formation limits


def test_best_xi_respects_club_cap():
    p = _players()
    p["team_id"] = 1  # one club only -> at most 3 players
    p.loc[p.index[::2], "team_id"] = np.arange(len(p.index[::2])) + 2
    xi = best_xi(p)
    assert (xi["team_id"] == 1).sum() <= 3


def test_xi_points_doubles_the_top_prediction():
    p = _players()
    pts, cap = xi_points(p)
    xi = best_xi(p)
    top = xi.sort_values("pred", ascending=False).iloc[0]
    assert cap == top["element"] and pts == xi["actual"].sum() + top["actual"]


def test_gw_average_proxy_is_ownership_weighted_squad():
    t = pd.DataFrame(
        [
            _row("2024-25", 1, 1, 90, 10, selected=300),
            _row("2024-25", 1, 2, 90, 2, selected=100),
            _row("2024-25", 1, 3, 0, 0, et=5, selected=10**6),  # assistant manager: ignored
        ]
    )
    assert gw_average_proxy(t) == pytest.approx(15 * (300 * 10 + 100 * 2) / 400)


def test_run_backtest_end_to_end_and_summary():
    df = _world()
    res = run_backtest(df, ["2024-25"], FormPer90Baseline(), {("2024-25", 2): 50.0})
    assert list(res.folds["gw"]) == [1, 2, 3, 4]
    assert set(res.folds["gw_average_source"]) == {"official", "proxy"}
    assert res.folds.loc[res.folds["gw"] == 2, "gw_average"].iloc[0] == 50.0
    # player 1: 6 per 90 at 90' -> xP 6, exact; player 2: 3 per 90 x 1/3 = 1, exact; player 3: 0
    assert (res.folds["rmse"].abs() < 1e-9).all() and (res.folds["mae"].abs() < 1e-9).all()
    s = summarise(res)
    assert s.loc[0, "folds"] == 4 and s.loc[0, "rmse"] == pytest.approx(0.0)
    assert {"spearman_mean", "xi_minus_avg_mean", "xi_minus_avg_sd"} <= set(s.columns)
    assert set(res.calibration["season"]) == {"2024-25", "all"}


def test_matched_policy_is_a_stub():
    with pytest.raises(NotImplementedError):
        simulate_matched_policy()
