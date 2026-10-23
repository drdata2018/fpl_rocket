"""Matched-policy simulator on synthetic seasons: same start for every policy, legal squads, budget, FT/hits
accounting, HOLD never transfers, statistics (drawdown, moving-block bootstrap)."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from fpl_rocket import matched
from fpl_rocket.backtest import FormPer90Baseline

CLUBS = 20
SHAPE = {1: 2, 2: 5, 3: 5, 4: 3}
GWS = 5


def _archive(seed: int = 3) -> pd.DataFrame:
    """Two seasons (2023-24 history, 2024-25 simulated), 20 clubs x 15 players, a rising star in 2024-25."""
    rng = random.Random(seed)
    rows = []
    quality = {}
    pid = 0
    meta = {}
    for club in range(1, CLUBS + 1):
        for et, n in SHAPE.items():
            for _ in range(n):
                pid += 1
                quality[pid] = rng.random() * (3 if et > 1 else 1.5)
                meta[pid] = (et, club, 40 + int(quality[pid] * 25) + 5 * (et - 1))
    star = next(p for p in meta if meta[p][0] == 3 and meta[p][1] == 2)
    for s_i, season in enumerate(("2023-24", "2024-25")):
        for gw in range(1, GWS + 1):
            for p, (et, club, price) in meta.items():
                q = quality[p] + (4.0 if (p == star and season == "2024-25" and gw >= 2) else 0.0)
                played = rng.random() < 0.85
                pts = max(0, round(q + rng.gauss(0, 1.5))) + 2 if played else 0
                opp = club % CLUBS + 1
                rows.append(
                    {
                        "season": season,
                        "gw": gw,
                        "element": p,
                        "code": 10_000 + p,
                        "element_type": et,
                        "team_id": club,
                        "team": f"T{club}",
                        "opponent_id": opp,
                        "opponent": f"T{opp}",
                        "was_home": club % 2 == 0,
                        "fixture": s_i * 1000 + gw * 100 + min(club, opp),
                        "kickoff_time": pd.Timestamp(f"202{3 + s_i}-08-{10 + gw}", tz="UTC"),
                        "minutes": 90 if played else 0,
                        "total_points": pts,
                        "selected": 1000 + int(q * 1000),
                        "value": price + (gw - 1) * (1 if p == star else 0),
                    }
                )
    df = pd.DataFrame(rows)
    df.attrs["star"] = star
    return df


@pytest.fixture(scope="module")
def result():
    df = _archive()
    preds = {"model": FormPer90Baseline(form_window=2), "baseline": FormPer90Baseline()}
    rows = matched.run_season(df, "2024-25", preds, time_limit=20)
    return df, rows


def test_every_policy_starts_from_the_same_squad(result):
    _, rows = result
    gw1 = rows[rows["gw"] == 1]
    assert set(gw1["policy"]) == set(matched.POLICIES)
    assert gw1["squad"].nunique() == 1 and gw1["bank"].nunique() == 1
    assert (gw1["transfers"] == 0).all() and (gw1["hits"] == 0).all()


def test_squads_legal_budget_and_free_transfer_accounting(result):
    df, rows = result
    meta = df.drop_duplicates("element").set_index("element")
    for policy, g in rows.groupby("policy"):
        g = g.sort_values("gw")
        ft_expected = None
        for r in g.itertuples(index=False):
            squad = [int(e) for e in r.squad.split(",")]
            assert len(squad) == 15
            assert Counter(meta.loc[squad, "element_type"]) == Counter(SHAPE)
            assert max(Counter(meta.loc[squad, "team_id"]).values()) <= 3
            assert r.bank >= 0
            if r.gw == 1:
                assert r.free_transfers == 1
            else:
                assert r.free_transfers == ft_expected
                assert r.hits == max(0, r.transfers - r.free_transfers)
                assert r.transfers <= matched.MAX_TRANSFERS
            assert r.net == r.points - 4 * r.hits
            ft_expected = (
                1 if r.gw == 1 else min(5, r.free_transfers - min(r.transfers, r.free_transfers) + 1)
            )
        if policy == "hold":
            assert g["transfers"].sum() == 0 and g["squad"].nunique() == 1


def test_planner_policies_react_to_a_rising_star(result):
    df, rows = result
    star = df.attrs["star"]
    bought = rows[(rows["policy"] != "hold") & rows["squad"].str.split(",").apply(lambda s: str(star) in s)]
    assert not bought.empty  # at least one planner policy picks up the in-form player


def test_summary_and_paired_bootstrap(result):
    _, rows = result
    summary, paired = matched.summarise(rows, resamples=200)
    assert set(summary["policy"]) == set(matched.POLICIES)
    hold = summary[summary["policy"] == "hold"].iloc[0]
    assert hold["transfers"] == 0 and hold["hits"] == 0
    p = paired.iloc[0]
    assert p["ci_low"] <= p["mean_diff_per_gw"] <= p["ci_high"]
    assert p["gws"] == GWS


def test_max_drawdown_and_block_bootstrap():
    assert matched.max_drawdown(np.array([5.0, -2.0, -4.0, 3.0, -10.0])) == pytest.approx(13.0)
    assert matched.max_drawdown(np.array([1.0, 2.0])) == 0.0
    const = matched.block_bootstrap_mean(np.full(10, 3.0), block=4, resamples=50)
    assert np.allclose(const, 3.0)
    x = np.arange(20, dtype=float)
    a = matched.block_bootstrap_mean(x, resamples=100, seed=1)
    b = matched.block_bootstrap_mean(x, resamples=100, seed=1)
    assert np.array_equal(a, b) and abs(a.mean() - x.mean()) < 2


def test_low_history_players_are_flagged_until_three_gws():
    df = _archive()
    newcomer = int(df["element"].max())
    df = df[~((df["season"] == "2023-24") & (df["element"] == newcomer))]
    lows = matched.low_history_sets(df, "2024-25")
    assert newcomer in lows[1] and newcomer in lows[3] and newcomer not in lows[4]
    assert len(lows[1]) == 1
