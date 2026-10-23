"""Retrospective backtest harness (docs/PLAN_v2.md section 4, "retrospective track").

RETROSPECTIVE: the archive holds end-of-GW data, so some inputs (e.g. the player pool, positions) may not
match what was known at the deadline. Every number produced here is labelled accordingly.

* rolling_origin(): for each (season, target_gw) the train rows are every GW strictly before target_gw in that
  season plus the whole previous season when it is in the frame; the target rows are the target GW.
* Predictor: fit(train) -> predict(target_players, target_gw) -> expected points per player for that GW
  (target_players carries no outcome columns).
* FormPer90Baseline ("form per 90 x availability"), per player keyed by the cross-season `code`:
      p90   = points / minutes * 90 over the player's last 5 GWs with minutes > 0 (ratio of sums)
      avail = minutes / (90 * fixtures) over the player's last 5 GWs (any minutes)
      xP    = p90 * avail * fixtures in the target GW (a double GW counts twice, a blank is not a target)
      xP    = 0 when the player had no minutes in his last 3 GWs (or no history at all)
* Metrics per fold over outfield players + GKs (element_type 1-4): RMSE, MAE, within-GW Spearman (average
  ranks; also among players who played, since the 0-minute mass inflates it), and a pooled calibration
  table by predicted-points decile.
* Decision metric WITHOUT transfers (an UNMATCHED comparison): the best legal XI by prediction from all
  players with a fixture (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD, max 3 per club, no budget) + captain = highest
  prediction, scored with actual points; compared with the GW average: events[].average_entry_score of the
  current bootstrap for the current season, otherwise a PROXY from the archive (gw_average_proxy: 15 x the
  ownership-weighted mean points of all players that GW).
* The matched-policy simulator (same start squad, budget, FT and chips; hits) is a stub until the planner.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

OUTCOME_COLUMNS = (
    "total_points",
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "defensive_contribution",
    "clearances_blocks_interceptions",
    "recoveries",
    "tackles",
)
PLAYER_TYPES = (1, 2, 3, 4)
FORMATION_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
FORMATION_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
XI_SIZE = 11
MAX_PER_CLUB = 3
PROXY_SQUAD = 15  # gw_average_proxy: an average squad of 15 owned players


@dataclass
class Fold:
    season: str
    target_gw: int
    train: pd.DataFrame  # fixture-level archive rows (incl. outcomes)
    target: pd.DataFrame  # fixture-level archive rows of the target GW (incl. outcomes)

    def target_players(self) -> pd.DataFrame:
        """One row per player with a fixture in the target GW, WITHOUT outcome columns."""
        t = self.target
        g = t.groupby(["season", "gw", "element"], as_index=False, sort=True).agg(
            code=("code", "first"),
            element_type=("element_type", "first"),
            team_id=("team_id", "first"),
            n_fixtures=("fixture", "nunique"),
        )
        return g

    def actuals(self) -> pd.Series:
        """Actual GW points per element (double GWs summed), aligned with target_players()."""
        pts = self.target.groupby("element", sort=True)["total_points"].sum()
        return pts.astype(float)


class Predictor(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> None: ...

    def predict(self, target_players: pd.DataFrame, target_gw: int) -> pd.Series:
        """Expected points per player for target_gw, index aligned with target_players."""
        ...


def season_order(seasons: list[str] | pd.Series) -> dict[str, int]:
    return {s: i for i, s in enumerate(sorted(set(seasons)))}


def previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{(start + 1) % 100:02d}"


def rolling_origin(
    df: pd.DataFrame, seasons: list[str], *, include_previous: bool = True, min_gw: int = 1
) -> Iterator[Fold]:
    """Folds in (season, target_gw) order; folds with an empty train set are skipped."""
    for season in seasons:
        s = df[df["season"] == season]
        prev = df[df["season"] == previous_season(season)] if include_previous else df.iloc[0:0]
        for gw in sorted(int(g) for g in s["gw"].dropna().unique()):
            if gw < min_gw:
                continue
            train = pd.concat([prev, s[s["gw"] < gw]], ignore_index=True)
            if train.empty:
                continue
            yield Fold(season, gw, train, s[s["gw"] == gw].reset_index(drop=True))


# ============================================================================================ baseline
def gw_history(train: pd.DataFrame) -> pd.DataFrame:
    """Per (player code, season, gw): points, minutes, fixtures - in chronological order."""
    key = train["code"].fillna(-train["element"]).astype("int64")  # code missing: season-local fallback
    g = (
        train.assign(pkey=key)
        .groupby(["pkey", "season", "gw"], as_index=False, sort=False)
        .agg(points=("total_points", "sum"), minutes=("minutes", "sum"), fixtures=("fixture", "nunique"))
    )
    order = season_order(g["season"])
    g["t"] = g["season"].map(order) * 100 + g["gw"].astype("int64")
    return g.sort_values(["pkey", "t"], kind="stable").reset_index(drop=True)


class FormPer90Baseline:
    """Form per 90 x availability (see module docstring)."""

    name = "baseline"

    def __init__(self, form_window: int = 5, zero_window: int = 3) -> None:
        self.form_window = form_window
        self.zero_window = zero_window
        self.xp_per_fixture: pd.Series = pd.Series(dtype=float)

    def fit(self, train: pd.DataFrame) -> None:
        h = gw_history(train)
        h["minutes"] = h["minutes"].fillna(0).astype(float)
        h["points"] = h["points"].fillna(0).astype(float)
        rank_from_end = h.groupby("pkey").cumcount(ascending=False)
        last_n = h[rank_from_end < self.form_window]
        win = last_n.groupby("pkey")[["minutes", "fixtures"]].sum()
        avail = win["minutes"] / (90.0 * win["fixtures"].clip(lower=1))
        played = h[h["minutes"] > 0]
        played_last = played[played.groupby("pkey").cumcount(ascending=False) < self.form_window]
        sums = played_last.groupby("pkey")[["points", "minutes"]].sum()
        p90 = sums["points"] / sums["minutes"] * 90.0
        recent = h[rank_from_end < self.zero_window].groupby("pkey")["minutes"].sum()
        xp = (p90.reindex(avail.index).fillna(0.0) * avail).where(recent.reindex(avail.index) > 0, 0.0)
        self.xp_per_fixture = xp.fillna(0.0)

    def predict(self, target_players: pd.DataFrame, target_gw: int) -> pd.Series:
        key = target_players["code"].fillna(-target_players["element"]).astype("int64")
        per_fixture = key.map(self.xp_per_fixture).fillna(0.0).astype(float)
        return (per_fixture * target_players["n_fixtures"].astype(float)).rename("xp")


PREDICTORS: dict[str, type] = {"baseline": FormPer90Baseline}


# ============================================================================================ metrics
def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation with average ranks for ties (no scipy). NaN when a side is constant."""
    ra, rb = pd.Series(a).rank(method="average"), pd.Series(b).rank(method="average")
    if ra.std() == 0 or rb.std() == 0 or len(ra) < 2:
        return float("nan")
    return float(np.corrcoef(ra.to_numpy(float), rb.to_numpy(float))[0, 1])


def point_metrics(pred: pd.Series, actual: pd.Series) -> dict[str, float]:
    err = pred.to_numpy(float) - actual.to_numpy(float)
    return {
        "n": len(err),
        "rmse": float(math.sqrt(np.mean(err**2))) if len(err) else float("nan"),
        "mae": float(np.mean(np.abs(err))) if len(err) else float("nan"),
        "spearman": spearman(pred, actual),
    }


def calibration_table(pred: pd.Series, actual: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs mean actual points per predicted-points decile (ties broken by rank order)."""
    df = pd.DataFrame({"pred": pred.to_numpy(float), "actual": actual.to_numpy(float)})
    df["decile"] = pd.qcut(df["pred"].rank(method="first"), bins, labels=False) + 1
    out = df.groupby("decile").agg(
        n=("pred", "size"),
        pred_mean=("pred", "mean"),
        actual_mean=("actual", "mean"),
        pred_min=("pred", "min"),
        pred_max=("pred", "max"),
    )
    out["gap"] = out["actual_mean"] - out["pred_mean"]
    return out.reset_index()


def best_xi(players: pd.DataFrame, score_col: str = "pred") -> pd.DataFrame:
    """Greedy legal XI by `score_col`: formation minimums first, then best remaining within the maxima;
    at most 3 per club (greedy, so with club caps it is a close approximation, not a proven optimum)."""
    ranked = players.sort_values(score_col, ascending=False, kind="stable")
    chosen: list[int] = []
    by_type: Counter = Counter()
    by_club: Counter = Counter()

    def take(idx: int, row: pd.Series) -> None:
        chosen.append(idx)
        by_type[int(row["element_type"])] += 1
        by_club[int(row["team_id"])] += 1

    for et, need in FORMATION_MIN.items():
        for idx, row in ranked[ranked["element_type"] == et].iterrows():
            if by_type[et] >= need:
                break
            if by_club[int(row["team_id"])] < MAX_PER_CLUB:
                take(idx, row)
    for idx, row in ranked.iterrows():
        if len(chosen) >= XI_SIZE:
            break
        et = int(row["element_type"])
        if idx in chosen or by_type[et] >= FORMATION_MAX[et] or by_club[int(row["team_id"])] >= MAX_PER_CLUB:
            continue
        take(idx, row)
    return players.loc[chosen]


def xi_points(players: pd.DataFrame) -> tuple[float, int]:
    """(actual points of the predicted-best XI with the top prediction captained (x2), captain element)."""
    xi = best_xi(players)
    cap = xi.sort_values("pred", ascending=False, kind="stable").iloc[0]
    return float(xi["actual"].sum() + cap["actual"]), int(cap["element"])


def gw_average_proxy(target: pd.DataFrame) -> float:
    """PROXY for the average manager score when the official one is unknown (older seasons):
    PROXY_SQUAD x ownership-weighted mean GW points = the points of an "average squad" of 15 owned players
    (players with a fixture in that GW; double GWs summed). Checked against the official
    average_entry_score of 2026-27 GW1-6: 53.4/50, 75.8/81, 51.4/51, 69.9/69, 50.3/48, 59.7/58 (MAE 2.3).
    The unweighted alternative (12 x mean points of players who played) was 23 points too low on average."""
    t = target[target["element_type"].isin(PLAYER_TYPES)]
    g = t.groupby("element").agg(points=("total_points", "sum"), selected=("selected", "max"))
    g = g.dropna()
    weight = g["selected"].astype(float)
    if weight.sum() <= 0:
        return float("nan")
    return float(PROXY_SQUAD * (weight * g["points"].astype(float)).sum() / weight.sum())


def simulate_matched_policy(*_args, **_kwargs):
    """TODO(planner step, Phase B): matched-policy comparison - from the SAME starting squad, budget, FT
    balance and chip state, apply identical legal rules (transfers, 4-point hits, FT banking) to (a) the
    model planner, (b) the last-5-form picker, (c) hold + best XI; report net points after hits, transfer
    count and drawdowns. Needs the planner (CBC via PuLP) first."""
    raise NotImplementedError("matched-policy simulator arrives with the planner (Phase B step 2+)")


# ============================================================================================ run
@dataclass
class BacktestResult:
    folds: pd.DataFrame  # one row per (season, gw)
    calibration: pd.DataFrame  # per season and pooled, by decile
    predictions: pd.DataFrame  # per (season, gw, element): pred, actual


def run_backtest(
    df: pd.DataFrame,
    seasons: list[str],
    predictor: Predictor,
    gw_averages: dict[tuple[str, int], float] | None = None,
) -> BacktestResult:
    """Run every fold; gw_averages = {(season, gw): official average entry score} where known."""
    gw_averages = gw_averages or {}
    fold_rows, preds = [], []
    for fold in rolling_origin(df, seasons):
        tp = fold.target_players()
        tp = tp[tp["element_type"].isin(PLAYER_TYPES)].reset_index(drop=True)
        if tp.empty:
            continue
        predictor.fit(fold.train)
        pred = predictor.predict(tp, fold.target_gw).reset_index(drop=True)
        actual = tp["element"].map(fold.actuals()).fillna(0.0).astype(float)
        m = point_metrics(pred, actual)
        played = (tp["element"].map(fold.target.groupby("element")["minutes"].sum()).fillna(0) > 0).to_numpy()
        m["spearman_played"] = spearman(pred[played], actual[played])  # without the 0-minute mass
        frame = tp.assign(pred=pred.to_numpy(float), actual=actual.to_numpy(float))
        xi, cap = xi_points(frame)
        official = gw_averages.get((fold.season, fold.target_gw))
        avg = official if official is not None else gw_average_proxy(fold.target)
        fold_rows.append(
            {
                "season": fold.season,
                "gw": fold.target_gw,
                **m,
                "xi_points": xi,
                "captain": cap,
                "gw_average": avg,
                "gw_average_source": "official" if official is not None else "proxy",
                "xi_minus_average": xi - avg,
                "train_rows": len(fold.train),
            }
        )
        preds.append(frame[["season", "gw", "element", "element_type", "pred", "actual"]])
    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    calib = []
    if not predictions.empty:
        for season, g in predictions.groupby("season"):
            calib.append(calibration_table(g["pred"], g["actual"]).assign(season=season))
        calib.append(calibration_table(predictions["pred"], predictions["actual"]).assign(season="all"))
    calibration = pd.concat(calib, ignore_index=True) if calib else pd.DataFrame()
    return BacktestResult(folds, calibration, predictions)


def summarise(result: BacktestResult) -> pd.DataFrame:
    """Per season: pooled RMSE / MAE over all player-GWs, mean within-GW Spearman, decision metric."""
    rows = []
    for season, f in result.folds.groupby("season"):
        p = result.predictions[result.predictions["season"] == season]
        err = p["pred"] - p["actual"]
        rows.append(
            {
                "season": season,
                "folds": len(f),
                "player_gws": len(p),
                "rmse": float(np.sqrt((err**2).mean())),
                "mae": float(err.abs().mean()),
                "spearman_mean": float(f["spearman"].mean()),
                "spearman_sd": float(f["spearman"].std()),
                "spearman_played_mean": float(f["spearman_played"].mean()),
                "xi_points_mean": float(f["xi_points"].mean()),
                "gw_average_mean": float(f["gw_average"].mean()),
                "xi_minus_avg_mean": float(f["xi_minus_average"].mean()),
                "xi_minus_avg_sd": float(f["xi_minus_average"].std()),
                "xi_minus_avg_min": float(f["xi_minus_average"].min()),
                "xi_minus_avg_max": float(f["xi_minus_average"].max()),
                "average_source": ",".join(sorted(set(f["gw_average_source"]))),
            }
        )
    return pd.DataFrame(rows)
