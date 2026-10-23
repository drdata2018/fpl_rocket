"""Retrospective backtest of a predictor on the normalised archive (scripts/load_archive.py first).

Usage: uv run python -I scripts/run_backtest.py --predictor baseline --seasons 2024-25 2025-26
       uv run python -I scripts/run_backtest.py --predictor model --seasons 2024-25 2025-26 2026-27

Writes data/backtests/<timestamp>_<predictor>.csv (one row per season x GW), ..._calibration.csv,
..._summary.csv and ..._by_position.csv, and prints the per-season summary. Labelled RETROSPECTIVE: the
archive is end-of-GW data. The decision metric (best XI + captain by prediction vs the GW average) is an
UNMATCHED comparison: no squad, budget or transfers. GW averages are official (bootstrap
events[].average_entry_score) for the current season and a proxy (backtest.gw_average_proxy) for older
seasons.

--predictor model also runs the baseline on the same folds and writes the model evaluation (minutes
log-loss and calibration, interval coverage, team-goals log-loss vs naive, runtime per fit) to
..._model_*.csv and data/backtests/<timestamp>_model_summary.md (comparison relative to the baseline).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpl_rocket.backtest import (  # noqa: E402
    PREDICTORS,
    BacktestResult,
    FormPer90Baseline,
    run_backtest,
    summarise,
    summarise_by_position,
)
from fpl_rocket.rules import CURRENT_SEASON  # noqa: E402

POS = {0: "all", 1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def official_averages(seasons: list[str]) -> dict[tuple[str, int], float]:
    """Current-season GW averages from the live bootstrap (other seasons have none)."""
    if CURRENT_SEASON not in seasons:
        return {}
    try:
        from fpl_rocket.api import FPLClient

        events = FPLClient().bootstrap()["events"]
    except Exception as exc:  # noqa: BLE001 - fall back to the proxy, loudly
        print(f"warning: bootstrap unavailable ({exc!r}); {CURRENT_SEASON} uses the proxy average")
        return {}
    return {
        (CURRENT_SEASON, int(e["id"])): float(e["average_entry_score"])
        for e in events
        if e.get("finished") and e.get("average_entry_score") is not None
    }


def write_result(base: Path, result: BacktestResult) -> pd.DataFrame:
    summary = summarise(result)
    result.folds.to_csv(f"{base}.csv", index=False)
    result.calibration.to_csv(f"{base}_calibration.csv", index=False)
    summary.to_csv(f"{base}_summary.csv", index=False)
    summarise_by_position(result).to_csv(f"{base}_by_position.csv", index=False)
    return summary


# ============================================================================================ model report
def _fmt(x: float, nd: int = 3) -> str:
    return "n/a" if pd.isna(x) else f"{x:.{nd}f}"


def _rel(model: float, base: float, lower_better: bool) -> str:
    if pd.isna(model) or pd.isna(base) or base == 0:
        return ""
    pct = (model - base) / abs(base) * 100
    better = (pct < 0) == lower_better
    return f"{pct:+.1f}% ({'better' if better and pct != 0 else 'worse' if pct != 0 else 'equal'})"


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def model_report(
    stamp: str,
    out: Path,
    archive: pd.DataFrame,
    model: BacktestResult,
    base: BacktestResult,
    seasons: list[str],
    runtime_s: float,
) -> Path:
    from fpl_rocket.model.config import BACKTEST_CONFIG, MODEL_BASE_VERSION
    from fpl_rocket.model.evaluate import interval_coverage, minutes_frame, minutes_report, team_goals_report
    from fpl_rocket.model.pipeline import git_sha_short

    ms, bs = summarise(model), summarise(base)
    mpos, bpos = summarise_by_position(model), summarise_by_position(base)
    d = model.details
    mins = minutes_frame(d["minutes"], archive)
    cover = interval_coverage(d["players"], model.predictions)
    goals = team_goals_report(d["fixtures"], archive)
    timing = d["timing"]
    prefix = out / f"{stamp}_model"
    cover.to_csv(f"{prefix}_coverage.csv", index=False)
    goals.to_csv(f"{prefix}_team_goals.csv", index=False)
    timing.to_csv(f"{prefix}_timing.csv", index=False)
    mins.to_parquet(f"{prefix}_minutes.parquet", index=False)

    lines = [
        f"# xP model {MODEL_BASE_VERSION}-{git_sha_short()} vs baseline - RETROSPECTIVE backtest ({stamp})",
        "",
        "**Retrospective track** (docs/PLAN_v2.md section 4): rolling origin, each fold fits on every GW",
        "before the target in its season plus the whole previous season, and predicts the target GW (horizon",
        "1). The archive is end-of-GW data, so it uses information that may not have been available at the",
        "deadline. Inputs without true historical snapshots: the player pool and positions of the target GW",
        "(players listed in the archive that GW), club assignment per fixture, the fixture list itself",
        "(postponements known after the fact). The live-only minutes inputs (status,",
        "chance_of_playing_next_round) do NOT exist historically:",
        "this backtest runs WITHOUT them (the live bundle applies them as an availability cap). Defensive",
        "contributions exist from 2025-26 only (2024-25 had no DefCon scoring; 2025-26 GW1 has no DefCon",
        "history to learn from). 2026-27 covers GW1-6 only and is shown for reference.",
        "",
        f"Seasons: {', '.join(seasons)}. Simulations per fold: {BACKTEST_CONFIG.n_sims}. Total runtime"
        f" {runtime_s / 60:.1f} min.",
        "",
        "## Point accuracy and decision metric (per season)",
        "",
    ]
    rows = []
    metrics = [
        ("RMSE", "rmse", True),
        ("MAE", "mae", True),
        ("Spearman within GW (all)", "spearman_mean", False),
        ("Spearman within GW (played)", "spearman_played_mean", False),
        ("best XI + C minus GW avg (mean)", "xi_minus_avg_mean", False),
        ("best XI + C minus GW avg (sd)", "xi_minus_avg_sd", None),
    ]
    for season in seasons:
        m = ms[ms["season"] == season]
        b = bs[bs["season"] == season]
        if m.empty:
            continue
        for label, col, lower in metrics:
            mv, bv = float(m[col].iloc[0]), float(b[col].iloc[0])
            rows.append(
                {
                    "season": season,
                    "metric": label,
                    "baseline": _fmt(bv),
                    "model": _fmt(mv),
                    "model vs baseline": "" if lower is None else _rel(mv, bv, lower),
                }
            )
        src = m["average_source"].iloc[0]
        rows.append(
            {
                "season": season,
                "metric": "GW average source",
                "baseline": src,
                "model": src,
                "model vs baseline": "",
            }
        )
    lines += [_md_table(pd.DataFrame(rows)), ""]
    lines += [
        "The decision metric (best legal XI + captain by prediction, no squad / budget / transfers, minus",
        "the GW average) is an UNMATCHED comparison, not a policy result; the GW average is a proxy before",
        "2026-27.",
        "",
        "## By position (pooled RMSE / MAE, mean within-GW Spearman over all listed players)",
        "",
    ]
    pos_rows = []
    for _, r in mpos.iterrows():
        b = bpos[(bpos["season"] == r["season"]) & (bpos["element_type"] == r["element_type"])].iloc[0]
        pos_rows.append(
            {
                "season": r["season"],
                "pos": POS[int(r["element_type"])],
                "n": int(r["player_gws"]),
                "RMSE base / model": (
                    f"{b['rmse']:.3f} / {r['rmse']:.3f} ({_rel(r['rmse'], b['rmse'], True)})"
                ),
                "MAE base / model": f"{b['mae']:.3f} / {r['mae']:.3f}",
                "Spearman base / model": f"{b['spearman_mean']:.3f} / {r['spearman_mean']:.3f}",
            }
        )
    lines += [_md_table(pd.DataFrame(pos_rows)), ""]

    lines += ["## Calibration of xP by predicted decile (all seasons pooled)", ""]
    cal_m = model.calibration[model.calibration["season"] == "all"]
    cal_b = base.calibration[base.calibration["season"] == "all"]
    cal = pd.DataFrame(
        {
            "decile": cal_m["decile"].to_numpy(),
            "model pred": cal_m["pred_mean"].round(2).to_numpy(),
            "model actual": cal_m["actual_mean"].round(2).to_numpy(),
            "baseline pred": cal_b["pred_mean"].round(2).to_numpy(),
            "baseline actual": cal_b["actual_mean"].round(2).to_numpy(),
        }
    )
    lines += [_md_table(cal), ""]

    lines += [
        "## Interval coverage (players who played; actual points inside the simulated [p20, p80])",
        "",
        "Nominal content of a continuous 20-80 interval is 60%; points are integers, so inclusive coverage",
        "of a well-calibrated discrete sample is somewhat higher.",
        "",
    ]
    cv = cover.copy()
    cv["pos"] = cv["element_type"].map(POS)
    cv = cv[["season", "pos", "n", "inside", "below_p20", "above_p80"]]
    for c in ("inside", "below_p20", "above_p80"):
        cv[c] = (cv[c] * 100).round(1).astype(str) + "%"
    lines += [_md_table(cv), ""]

    lines += ["## Minutes model (per player-fixture, horizon 1, WITHOUT live status / chance of playing)", ""]
    mr_rows = []
    reports = {}
    for season, g in [*mins.groupby("season"), ("all", mins)]:
        rep = minutes_report(g)
        reports[season] = rep
        for label, r in rep.items():
            mr_rows.append(
                {
                    "season": season,
                    "group": label,
                    "n": r["n"],
                    "log-loss": f"{r['log_loss']:.3f}",
                    "constant (in-sample freq) log-loss": f"{r['base_rate_log_loss']:.3f}",
                    "Brier P(play)": f"{r['brier_play']:.3f}",
                    "P(play) mean pred / obs": f"{r['mean_p_play']:.3f} / {r['obs_play']:.3f}",
                    "P(60+) mean pred / obs": f"{r['mean_p_60']:.3f} / {r['obs_60']:.3f}",
                }
            )
    lines += [
        _md_table(pd.DataFrame(mr_rows)),
        "",
        "rotation_risk = players with 1-4 starts in their last 5 GWs before the target.",
        "",
    ]
    for label in ("all", "rotation_risk"):
        rep = reports["all"].get(label)
        if not rep:
            continue
        for name, key in (("P(play)", "calib_play"), ("P(60+)", "calib_60")):
            c = rep[key].copy()
            c["pred_mean"] = c["pred_mean"].round(3)
            c["obs_rate"] = c["obs_rate"].round(3)
            c["gap"] = c["gap"].round(3)
            lines += [f"Calibration of {name}, {label}, all seasons pooled:", "", _md_table(c), ""]

    lines += ["## Team goals: Dixon-Coles vs naive league-average goals (log-loss, lower is better)", ""]
    g2 = goals.copy()
    for c in g2.columns:
        if c.endswith("log_loss") or c == "mean_rho":
            g2[c] = g2[c].round(4)
    lines += [
        _md_table(g2),
        "",
        "score = exact scoreline (DC score matrix vs independent Poisson with the training mean home / away",
        "goals); team goals = each side's goal count under its marginal Poisson rate.",
        "",
    ]

    t = timing
    lines += [
        "## Runtime per fit (this machine)",
        "",
        f"fit (all components) mean {t['fit_s'].mean():.2f} s, median {t['fit_s'].median():.2f} s,"
        f" max {t['fit_s'].max():.2f} s; predict incl. {BACKTEST_CONFIG.n_sims} simulations mean"
        f" {t['predict_s'].mean():.2f} s; folds {len(t)}.",
        f"Components (mean): team goals {t['fit_team_goals_s'].mean():.2f} s,"
        f" minutes {t['fit_minutes_s'].mean():.2f} s,"
        f" rates + bonus {t['fit_rates_bonus_s'].mean():.2f} s, simulation {t['simulate_s'].mean():.2f} s.",
        "",
        "## Hyperparameters (src/fpl_rocket/model/config.py; fixed a priori, except newcomer_fallback and",
        "bonus_sim, chosen on 2023-24 only; none tuned on the seasons above)",
        "",
    ]
    hp = {k: v for k, v in BACKTEST_CONFIG.as_dict().items() if not k.endswith("features")}
    lines += [f"- {k} = {v}" for k, v in hp.items()]
    lines += [
        "",
        "(The bundle trains the minutes model on horizons 1-6; this backtest scores horizon 1, so it trains",
        "horizon 1 only. Everything else is identical.)",
    ]
    path = out / f"{stamp}_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictor", default="baseline", choices=sorted(PREDICTORS))
    ap.add_argument("--seasons", nargs="+", default=["2024-25", "2025-26"])
    ap.add_argument("--archive", default="data/archive/all_gws.parquet")
    ap.add_argument("--out", default="data/backtests")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.archive)
    averages = official_averages(args.seasons)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    t0 = time.perf_counter()
    result = run_backtest(df, args.seasons, PREDICTORS[args.predictor](), averages)
    runtime = time.perf_counter() - t0
    base = out / f"{stamp}_{args.predictor}"
    summary = write_result(base, result)

    pd.set_option("display.width", 200)
    print(f"RETROSPECTIVE backtest, predictor={args.predictor}, seasons={args.seasons} ({runtime:.0f} s)")
    print(f"results: {base}.csv (+ _calibration.csv, _summary.csv, _by_position.csv)\n")
    print(summary.round(3).to_string(index=False))
    pooled = result.calibration[result.calibration["season"] == "all"]
    print("\ncalibration by predicted-points decile (all seasons):")
    print(pooled.drop(columns="season").round(2).to_string(index=False))
    if args.predictor == "model":
        baseline = run_backtest(df, args.seasons, FormPer90Baseline(), averages)
        write_result(out / f"{stamp}_baseline", baseline)
        report = model_report(stamp, out, df, result, baseline, args.seasons, runtime)
        print(f"\nmodel report: {report}")
    print(
        "\nDecision metric = best legal XI + captain by prediction (no squad, budget or transfers) minus the"
        " GW average: an UNMATCHED comparison, not a policy result."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
