"""Retrospective evaluation of the xP model beyond point errors: minutes log-loss and calibration, interval
coverage, team-goals log-loss against a naive league-average model. Inputs are ModelPredictor.last_details
collected by backtest.run_backtest (BacktestResult.details) plus the archive rows for the actual outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import minutes_bin
from .team_goals import poisson_pmf, results_from_rows, score_log_prob

EPS = 1e-12


def minutes_frame(details_minutes: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    """Predicted minutes-bin probabilities joined with the actual minutes of each player-fixture."""
    act = archive[["season", "gw", "element", "fixture", "minutes"]].copy()
    act["actual_minutes"] = pd.to_numeric(act.pop("minutes"), errors="coerce").fillna(0).astype(float)
    for c in ("gw", "element", "fixture"):
        act[c] = act[c].astype("int64")
    d = details_minutes.copy()
    for c in ("gw", "element", "fixture"):
        d[c] = d[c].astype("int64")
    m = d.merge(act, on=["season", "gw", "element", "fixture"], how="inner")
    m["bin"] = minutes_bin(m["actual_minutes"])
    m["rotation_risk"] = m["starts_last_5"].between(1, 4)
    return m


def log_loss3(m: pd.DataFrame) -> float:
    p = m[["p0", "p1", "p2"]].to_numpy(float)
    p = np.clip(p / p.sum(axis=1, keepdims=True), EPS, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(m)), m["bin"].to_numpy(int)])))


def base_rate_log_loss(m: pd.DataFrame, reference: pd.DataFrame | None = None) -> float:
    """Log-loss of a constant prediction = the class frequencies of `reference` (default: m itself, i.e. an
    in-sample upper bound for any constant predictor)."""
    ref = m if reference is None else reference
    freq = np.bincount(ref["bin"].to_numpy(int), minlength=3) / max(len(ref), 1)
    freq = np.clip(freq, EPS, 1)
    return float(-np.mean(np.log(freq[m["bin"].to_numpy(int)])))


def calibration(pred: pd.Series, outcome: pd.Series, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"pred": pred.to_numpy(float), "obs": outcome.to_numpy(float)})
    df["decile"] = pd.qcut(df["pred"].rank(method="first"), bins, labels=False) + 1
    out = df.groupby("decile").agg(n=("pred", "size"), pred_mean=("pred", "mean"), obs_rate=("obs", "mean"))
    out["gap"] = out["obs_rate"] - out["pred_mean"]
    return out.reset_index()


def minutes_report(m: pd.DataFrame) -> dict[str, object]:
    """Log-loss (model vs in-sample class frequencies), and calibration of P(play) and P(60+), overall and
    for rotation risks (1-4 starts in the last 5 GWs)."""
    out: dict[str, object] = {}
    for label, sub in (("all", m), ("rotation_risk", m[m["rotation_risk"]])):
        if sub.empty:
            continue
        play = sub["p1"] + sub["p2"]
        out[label] = {
            "n": len(sub),
            "log_loss": log_loss3(sub),
            "base_rate_log_loss": base_rate_log_loss(sub),
            "brier_play": float(np.mean((play - (sub["bin"] > 0)) ** 2)),
            "mean_p_play": float(play.mean()),
            "obs_play": float((sub["bin"] > 0).mean()),
            "mean_p_60": float(sub["p2"].mean()),
            "obs_60": float((sub["bin"] == 2).mean()),
            "calib_play": calibration(play, (sub["bin"] > 0).astype(float)),
            "calib_60": calibration(sub["p2"], (sub["bin"] == 2).astype(float)),
        }
    return out


def interval_coverage(players: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Share of actual points inside [p20, p80] (inclusive) among players who played, plus below / above.
    Nominal content of a continuous 20-80 interval is 60%; integer points make inclusive coverage higher."""
    p = players[["season", "gw", "element", "p20", "p80", "element_type"]].merge(
        predictions[["season", "gw", "element", "actual", "minutes"]], on=["season", "gw", "element"]
    )
    p = p[p["minutes"] > 0]
    rows = []
    for (season, et), g in pd.concat([p, p.assign(element_type=0)]).groupby(["season", "element_type"]):
        rows.append(
            {
                "season": season,
                "element_type": int(et),
                "n": len(g),
                "inside": float(((g["actual"] >= g["p20"]) & (g["actual"] <= g["p80"])).mean()),
                "below_p20": float((g["actual"] < g["p20"]).mean()),
                "above_p80": float((g["actual"] > g["p80"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def team_goals_report(fixtures: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    """Per season: exact-score log-loss and per-team goals log-loss of Dixon-Coles vs naive league-average
    goals (independent Poisson with the training mean home / away goals)."""
    keys = fixtures[["season", "gw"]].drop_duplicates()
    rows_needed = archive.merge(keys, on=["season", "gw"])
    actual = results_from_rows(rows_needed)[["season", "fixture", "hg", "ag"]]
    f = fixtures.merge(actual, on=["season", "fixture"], how="inner")
    out = []
    for season, g in f.groupby("season"):
        dc_score, nv_score, dc_team, nv_team = [], [], [], []
        for r in g.itertuples(index=False):
            dc_score.append(score_log_prob(r.lambda_home, r.lambda_away, r.rho, int(r.hg), int(r.ag)))
            nv_score.append(score_log_prob(r.naive_home, r.naive_away, 0.0, int(r.hg), int(r.ag)))
            for lam, nv, goals in ((r.lambda_home, r.naive_home, r.hg), (r.lambda_away, r.naive_away, r.ag)):
                k = int(goals)
                dc_team.append(np.log(max(poisson_pmf(lam, k + 1)[k], EPS)))
                nv_team.append(np.log(max(poisson_pmf(nv, k + 1)[k], EPS)))
        out.append(
            {
                "season": season,
                "fixtures": len(g),
                "dc_score_log_loss": -float(np.mean(dc_score)),
                "naive_score_log_loss": -float(np.mean(nv_score)),
                "dc_team_goals_log_loss": -float(np.mean(dc_team)),
                "naive_team_goals_log_loss": -float(np.mean(nv_team)),
                "mean_rho": float(g["rho"].mean()),
            }
        )
    return pd.DataFrame(out)
