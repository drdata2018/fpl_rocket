"""Decision-level diagnostics of the frozen v0.2 model (step 3c; RETROSPECTIVE, archive = end-of-GW data).

Measurements only, no modelling choices: 2024-25 + 2025-26 (76 folds) and 2026-27 (current season,
separately).

  uv run python -I scripts/run_decision_diagnostics.py [--workers 8] [--bundle data/bundles/<id>]
      1. captaincy decomposition (XI without the captain bonus / the bonus / total, model minus baseline with
         block-bootstrap CIs; captain rank; top-1/3/5 calibration; top-2 gap vs its Monte Carlo SE)
      2. captain uncertainty: 10 fixed folds x (N 4000 seeds 1-2, N 1000 seeds 1-5); bundle build cost of
         N 4000 vs 1000 (timed on --bundle when given)
      3. squared-error decomposition by minutes class (0 / 1-59 / 60+) and the "points given 60+" RMSE
      4. P(play) / P(60+) reliability and Brier score vs persistence and 5-GW frequency
      5. cold start: GW1-3 of each season
      6. multiweek: per-origin paired differences of the 3- and 6-GW window RMSE
  --summarise <ts>   rebuild the report from the raw frames of an earlier run (no refitting).
Writes data/backtests/<ts>_v02_decisions_*.csv and <ts>_v02_decisions_summary.md. Set PYTHONUTF8=1 on
Windows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TEST_SEASONS = ["2024-25", "2025-26"]
CURRENT = "2026-27"
HORIZON_MAX = 6
# captain-uncertainty folds, fixed before any result: every 5th GW of 2025-26 plus three of 2024-25
CAPTAIN_FOLDS = [("2025-26", g) for g in (5, 10, 15, 20, 25, 30, 35)] + [("2024-25", g) for g in (12, 24, 36)]
# the first setting defines the reference captain pair (the most precise run)
CAPTAIN_SETTINGS = [(4000, 1), (4000, 2), (1000, 1), (1000, 2), (1000, 3), (1000, 4), (1000, 5)]
POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
GROUPS = (
    ("2024-25 + 2025-26 (76 GWs)", TEST_SEASONS),
    ("2024-25", ["2024-25"]),
    ("2025-26", ["2025-26"]),
    ("2026-27 (current, 6 GWs)", [CURRENT]),
)

_DF: pd.DataFrame | None = None


# ============================================================================================ workers
def _init(archive: str) -> None:
    global _DF
    _DF = pd.read_parquet(archive)


def _task(kind: str, args: tuple):
    import warnings

    warnings.filterwarnings("ignore")
    from fpl_rocket.model.config import BACKTEST_CONFIG
    from fpl_rocket.model.decisions import run_captain_seeds, run_decision_fold
    from fpl_rocket.model.diagnostics import run_horizon_origin

    t0 = time.perf_counter()
    if kind == "fold":
        out = run_decision_fold(_DF, *args)
    elif kind == "horizon":
        cfg = replace(BACKTEST_CONFIG, horizons=tuple(range(1, HORIZON_MAX + 1)))
        out = {"horizon": run_horizon_origin(_DF, args[0], args[1], cfg, HORIZON_MAX)}
    elif kind == "captain":
        out = {"captain": run_captain_seeds(_DF, args[0], args[1], BACKTEST_CONFIG, CAPTAIN_SETTINGS)}
    else:
        raise ValueError(kind)
    return kind, args, out, time.perf_counter() - t0


def run_tasks(tasks: list[tuple[str, tuple]], archive: str, workers: int) -> dict[str, list[pd.DataFrame]]:
    frames: dict[str, list[pd.DataFrame]] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(archive,)) as ex:
        futures = {ex.submit(_task, k, a): (k, a) for k, a in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                kind, key, out, secs = fut.result()
            except Exception as exc:  # noqa: BLE001 - record the failed task, keep the others
                k, a = futures[fut]
                print(f"  [{i}/{len(tasks)}] FAILED {k} {a}: {exc!r}", flush=True)
                frames.setdefault("failed", []).append(
                    pd.DataFrame([{"kind": k, "task": str(a), "error": repr(exc)}])
                )
                continue
            for name, f in out.items():
                frames.setdefault(name, []).append(f)
            print(f"  [{i}/{len(tasks)}] {kind} {key} {secs:.0f} s", flush=True)
    return frames


# ============================================================================================ helpers
def md_table(df: pd.DataFrame, nd: int = 3) -> str:
    def fmt(v):
        if isinstance(v, bool | np.bool_):
            return str(v)
        if isinstance(v, float | np.floating):
            return "n/a" if np.isnan(v) else f"{v:.{nd}f}"
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def rmse(e) -> float:
    e = np.asarray(e, dtype=float)
    return float(np.sqrt(np.mean(e**2))) if len(e) else float("nan")


def paired_rows(per_gw: pd.DataFrame, a: str, b: str, metrics: list[str], lower: bool = False) -> list[dict]:
    """per_gw: one row per (season, gw) with {metric}_{a} and {metric}_{b}. Mean a - b over the GWs with a
    block-bootstrap 95% CI (block 4, 2000, blocks within a season) and the sign test."""
    from fpl_rocket.model.diagnostics import block_bootstrap_ci, sign_test

    rows = []
    for m in metrics:
        d = (per_gw[f"{m}_{a}"] - per_gw[f"{m}_{b}"]).to_numpy(float)
        groups = [d[(per_gw["season"] == s).to_numpy()] for s in sorted(per_gw["season"].unique())]
        mean, lo, hi = block_bootstrap_ci(groups)
        k, n, p = sign_test(d, lower)
        rows.append(
            {
                "metric": m,
                "gws": len(d),
                f"mean_{a}": float(per_gw[f"{m}_{a}"].mean()),
                f"mean_{b}": float(per_gw[f"{m}_{b}"].mean()),
                "mean_diff": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                f"{a}_better_gws": k,
                "non_tied_gws": n,
                "sign_p": p,
            }
        )
    return rows


# ============================================================================================ 1. captaincy
def captain_tables(players: pd.DataFrame, top2: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from fpl_rocket.model.decisions import captain_decomposition, top_candidates

    per_gw, topk = [], []
    for (season, gw), g in players.groupby(["season", "gw"], sort=True):
        r = {"season": season, "gw": gw}
        for name, col in (("model", "xp"), ("base", "pred_base")):
            for k, v in captain_decomposition(g, col).items():
                r[f"{k}_{name}"] = v
            topk.append(top_candidates(g, col).assign(season=season, gw=gw, predictor=name))
        per_gw.append(r)
    per_gw = pd.DataFrame(per_gw)
    paired = []
    for label, seasons in GROUPS:
        sub = per_gw[per_gw["season"].isin(seasons)]
        for row in paired_rows(sub, "model", "base", ["xi_no_cap", "cap_bonus", "total"]):
            paired.append({"folds": label, **row})
    quality = []
    for label, seasons in GROUPS:
        sub = per_gw[per_gw["season"].isin(seasons)]
        for name in ("model", "base"):
            quality.append(
                {
                    "folds": label,
                    "predictor": name,
                    "gws": len(sub),
                    "cap_played": float(sub[f"cap_played_{name}"].mean()),
                    "cap_pct_mean": float(sub[f"cap_pct_{name}"].mean()),
                    "cap_rank_median": float(sub[f"cap_rank_{name}"].median()),
                    "top10_share": float(sub[f"cap_top10_{name}"].mean()),
                    "top50_share": float(sub[f"cap_top50_{name}"].mean()),
                    "cap_pred_mean": float(sub[f"cap_pred_{name}"].mean()),
                    "cap_actual_mean": float(sub[f"cap_bonus_{name}"].mean()),
                    "same_captain_share": float((sub["captain_model"] == sub["captain_base"]).mean()),
                }
            )
    tk = pd.concat(topk, ignore_index=True)
    tk = tk.assign(group=np.where(tk["season"] == CURRENT, CURRENT, tk["season"]))
    calib = []
    for (grp, pred, k), g in [
        *tk.groupby(["group", "predictor", "k"]),
        *(
            (("2024-25 + 2025-26", p, k), x)
            for (p, k), x in tk[tk["season"] != CURRENT].groupby(["predictor", "k"])
        ),
    ]:
        calib.append(
            {
                "season": grp,
                "predictor": pred,
                "k": int(k),
                "gws": len(g),
                "pred_mean": float(g["pred_mean"].mean()),
                "actual_mean": float(g["actual_mean"].mean()),
                "actual_over_pred": float(g["actual_mean"].mean() / g["pred_mean"].mean()),
                "played_share": float(g["played_share"].mean()),
            }
        )
    t2 = top2.assign(z=top2["gap"] / top2["gap_se"])
    t2 = t2.assign(group=np.where(t2["season"] == CURRENT, CURRENT, "2024-25 + 2025-26"))
    gap = (
        t2.groupby("group")
        .agg(
            gws=("gap", "size"),
            gap_median=("gap", "median"),
            gap_mean=("gap", "mean"),
            se_median=("gap_se", "median"),
            within_1se=("z", lambda z: float((z < 1).mean())),
            within_2se=("z", lambda z: float((z < 2).mean())),
        )
        .reset_index()
    )
    return {
        "captain_per_gw": per_gw,
        "captain_paired": pd.DataFrame(paired),
        "captain_quality": pd.DataFrame(quality),
        "captain_topk_calibration": pd.DataFrame(calib),
        "captain_top2_gap": gap,
        "captain_top2_per_gw": top2,
    }


# ============================================================================================ 2. seeds / N
def seed_tables(cap: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from fpl_rocket.model.decisions import xi_changed

    per_fold, rows = [], []
    for (season, gw), g in cap.groupby(["season", "gw"], sort=True):
        r = {"season": season, "gw": gw, "ref_pair": f"{g['ref_a'].iloc[0]} vs {g['ref_b'].iloc[0]}"}
        for n in (1000, 4000):
            x = g[g["n_sims"] == n].reset_index(drop=True)
            pairs = [(i, j) for i in range(len(x)) for j in range(i + 1, len(x))]
            r[f"n{n}_runs"] = len(x)
            r[f"n{n}_distinct_captains"] = int(x["captain"].nunique())
            r[f"n{n}_captain_change_pairs"] = float(
                np.mean([x["captain"][i] != x["captain"][j] for i, j in pairs]) if pairs else np.nan
            )
            changed = [xi_changed(x["xi"][i], x["xi"][j]) for i, j in pairs]
            r[f"n{n}_xi_changed_mean"] = float(np.mean(changed)) if changed else np.nan
            r[f"n{n}_xi_changed_max"] = int(np.max(changed)) if changed else 0
            r[f"n{n}_ref_diff_mean"] = float(x["ref_diff"].mean())
            r[f"n{n}_ref_diff_sd_seeds"] = float(x["ref_diff"].std(ddof=1)) if len(x) > 1 else np.nan
            r[f"n{n}_ref_diff_min"] = float(x["ref_diff"].min())
            r[f"n{n}_ref_diff_max"] = float(x["ref_diff"].max())
            r[f"n{n}_ref_diff_se_mc"] = float(x["ref_diff_se"].mean())
            r[f"n{n}_ref_sign_flips"] = int((np.sign(x["ref_diff"]) != np.sign(g["ref_diff"].iloc[0])).sum())
        per_fold.append(r)
    pf = pd.DataFrame(per_fold)
    for n in (1000, 4000):
        rows.append(
            {
                "n_sims": n,
                "folds": len(pf),
                "runs_per_fold": int(pf[f"n{n}_runs"].iloc[0]),
                "folds_with_captain_change": int((pf[f"n{n}_distinct_captains"] > 1).sum()),
                "captain_change_share_of_pairs": float(pf[f"n{n}_captain_change_pairs"].mean()),
                "xi_changed_mean": float(pf[f"n{n}_xi_changed_mean"].mean()),
                "xi_changed_max": int(pf[f"n{n}_xi_changed_max"].max()),
                "ref_diff_sd_seeds_mean": float(pf[f"n{n}_ref_diff_sd_seeds"].mean()),
                "ref_diff_se_mc_mean": float(pf[f"n{n}_ref_diff_se_mc"].mean()),
                "ref_abs_diff_median": float(pf[f"n{n}_ref_diff_mean"].abs().median()),
            }
        )
    # cross-N: does N 1000 (seed 1) pick the N 4000 (seed 1) captain / XI?
    cross = []
    for (season, gw), g in cap.groupby(["season", "gw"], sort=True):
        ref = g[(g["n_sims"] == 4000)].iloc[0]
        for _, x in g[g["n_sims"] == 1000].iterrows():
            cross.append(
                {
                    "season": season,
                    "gw": gw,
                    "seed": x["seed"],
                    "same_captain_as_n4000": x["captain"] == ref["captain"],
                    "xi_changed_vs_n4000": xi_changed(x["xi"], ref["xi"]),
                }
            )
    cross = pd.DataFrame(cross)
    return {"seeds_per_fold": pf, "seeds_summary": pd.DataFrame(rows), "seeds_vs_n4000": cross}


def bundle_timing(bundle_dir: str, archive: str, scratch: Path, repeats: int = 2) -> pd.DataFrame:
    """Wall time of run_bundle_model (fit + 6 GWs x all players) and of writing xp_sims.parquet (zstd, as
    the bundle does) at N 1000 and N 4000 on a recorded bundle."""
    import warnings

    warnings.filterwarnings("ignore")
    from fpl_rocket.bundle import PARQUET_COMPRESSION, read_bundle
    from fpl_rocket.model.config import DEFAULT_CONFIG
    from fpl_rocket.model.pipeline import run_bundle_model

    tables = read_bundle(bundle_dir)
    tables = {k: v for k, v in tables.items() if not k.startswith("xp")}
    arch = pd.read_parquet(archive)
    rows = []
    for rep in range(repeats):
        for n in (1000, 4000):
            t0 = time.perf_counter()
            res, man = run_bundle_model(tables, arch, cfg=replace(DEFAULT_CONFIG, n_sims=n))
            t1 = time.perf_counter()
            path = scratch / f"xp_sims_n{n}.parquet"
            res["xp_sims"].to_parquet(path, index=False, compression=PARQUET_COMPRESSION["xp_sims"])
            t2 = time.perf_counter()
            rows.append(
                {
                    "n_sims": n,
                    "repeat": rep + 1,
                    "model_s": t1 - t0,
                    "simulate_s": man["model_timings_s"]["simulate_s"],
                    "fit_s": sum(v for k, v in man["model_timings_s"].items() if k.startswith("fit_")),
                    "write_sims_s": t2 - t1,
                    "xp_sims_rows": len(res["xp_sims"]),
                    "xp_sims_mb": path.stat().st_size / 1e6,
                    "sims_memory_mb": res["xp_sims"].memory_usage(deep=True).sum() / 1e6,
                    "gws": len(man["model_fitted_on"]["gws_predicted"]),
                }
            )
            print(f"  bundle timing N={n} rep {rep + 1}: {t2 - t0:.1f} s", flush=True)
            path.unlink(missing_ok=True)
    return pd.DataFrame(rows)


# ============================================================================================ 3. errors
def error_tables(players: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from fpl_rocket.model.decisions import error_decomposition, minutes_class

    p = players.assign(mclass=minutes_class(players["max_bin"]))
    dec = []
    for label, seasons in GROUPS:
        t = error_decomposition(p[p["season"].isin(seasons)], "xp", "pred_base")
        t.insert(0, "folds", label)
        dec.append(t)
    sixty = p[p["max_bin"] == 2].copy()
    # no simulation with 60+ minutes: given any minutes, else the fold's position mean given 60+
    pos_mean = sixty.groupby(["season", "gw", "element_type"])["cond60"].transform("mean")
    sixty["model_cond"] = sixty["cond60"].fillna(sixty["condplay"]).fillna(pos_mean)
    cond = []
    for label, seasons in GROUPS:
        s = sixty[sixty["season"].isin(seasons)]
        for pos, g in [("all", s), *((POS[int(k)], x) for k, x in s.groupby("element_type"))]:
            y = g["actual"]
            cond.append(
                {
                    "folds": label,
                    "pos": pos,
                    "n": len(g),
                    "mean_actual": float(y.mean()),
                    "rmse_model_cond60": rmse(g["model_cond"] - y),
                    "rmse_base_p90": rmse(g["base_cond"] - y),
                    "bias_model_cond60": float((g["model_cond"] - y).mean()),
                    "bias_base_p90": float((g["base_cond"] - y).mean()),
                    "rmse_model_xp": rmse(g["xp"] - y),
                    "rmse_base_xp": rmse(g["pred_base"] - y),
                    "model_cond_fallbacks": int(g["cond60"].isna().sum()),
                }
            )
    return {"error_decomposition": pd.concat(dec, ignore_index=True), "error_given_60": pd.DataFrame(cond)}


# ================================================================================ 4. probabilities
def probability_tables(players: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from fpl_rocket.model.decisions import brier, reliability_table

    p = players.assign(
        y_play=(players["max_bin"] > 0).astype(float), y_60=(players["max_bin"] == 2).astype(float)
    )
    seasons = [*((s, [s]) for s in [*TEST_SEASONS, CURRENT]), ("2024-25 + 2025-26", TEST_SEASONS)]
    rel, br = [], []
    for label, ss in seasons:
        g = p[p["season"].isin(ss)]
        for target, ycol in (("P(play)", "y_play"), ("P(60+)", "y_60")):
            suffix = "play" if ycol == "y_play" else "60"
            t = reliability_table(g[f"p_{suffix}"], g[ycol])
            t.insert(0, "probability", target)
            t.insert(0, "season", label)
            rel.append(t)
            br.append(
                {
                    "season": label,
                    "probability": target,
                    "n": len(g),
                    "base_rate": float(g[ycol].mean()),
                    "brier_model": brier(g[f"p_{suffix}"], g[ycol]),
                    "brier_persistence": brier(g[f"pers_{suffix}"], g[ycol]),
                    "brier_freq_5gw": brier(g[f"freq_{suffix}"], g[ycol]),
                    "brier_constant": brier(np.full(len(g), g[ycol].mean()), g[ycol]),
                }
            )
    return {"prob_reliability": pd.concat(rel, ignore_index=True), "prob_brier": pd.DataFrame(br)}


# ============================================================================================ 5. cold start
def cold_start_table(players: pd.DataFrame) -> pd.DataFrame:
    from fpl_rocket.backtest import spearman

    rows = []
    for season, s in players.groupby("season"):
        parts = [(f"GW{g}", s[s["gw"] == g]) for g in (1, 2, 3)]
        parts.append(("GW1-3", s[s["gw"] <= 3]))
        parts.append((f"GW4-{int(s['gw'].max())}", s[s["gw"] >= 4]))
        for label, g in parts:
            if g.empty:
                continue
            r = {"season": season, "gws": label, "n": len(g), "low_history_share": np.nan}
            for name, col in (("model", "xp"), ("base", "pred_base")):
                r[f"rmse_{name}"] = rmse(g[col] - g["actual"])
                sp = [
                    spearman(x.loc[x["minutes"] > 0, col], x.loc[x["minutes"] > 0, "actual"])
                    for _, x in g.groupby("gw")
                ]
                r[f"sp_played_{name}"] = float(np.nanmean(sp))
            r["rmse_diff"] = r["rmse_model"] - r["rmse_base"]
            r["sp_diff"] = r["sp_played_model"] - r["sp_played_base"]
            r.pop("low_history_share")
            rows.append(r)
    return pd.DataFrame(rows)


# ============================================================================================ 6. multiweek
def multiweek_tables(h: pd.DataFrame) -> dict[str, pd.DataFrame]:
    per_origin = []
    for window in (3, 6):
        w = h[h["h"] <= window]
        agg = (
            w.groupby(["season", "origin", "element"])[["model", "naive", "base", "actual"]]
            .sum()
            .reset_index()
        )
        for (season, origin), g in agg.groupby(["season", "origin"]):
            per_origin.append(
                {
                    "season": season,
                    "gw": origin,  # the origin plays the role of the GW index for the block bootstrap
                    "window_gws": window,
                    "n": len(g),
                    **{f"rmse_{c}": rmse(g[c] - g["actual"]) for c in ("model", "naive", "base")},
                }
            )
    po = pd.DataFrame(per_origin)
    rows = []
    for window in (3, 6):
        x = po[po["window_gws"] == window]
        for label, seasons in (("pooled", TEST_SEASONS), *((s, [s]) for s in TEST_SEASONS)):
            sub = x[x["season"].isin(seasons)]
            for b in ("naive", "base"):
                for r in paired_rows(sub, "model", b, ["rmse"], lower=True):
                    r.pop("metric")
                    r["mean_other"] = r.pop(f"mean_{b}")
                    rows.append({"window_gws": window, "seasons": label, "vs": b, **r})
    return {
        "multiweek_per_origin": po.rename(columns={"gw": "origin"}),
        "multiweek_paired": pd.DataFrame(rows),
    }


# ============================================================================================ report
def write_csvs(prefix: str, tables: dict[str, pd.DataFrame]) -> None:
    for name, t in tables.items():
        t.to_csv(f"{prefix}_{name}.csv", index=False)


def summarise(out: Path, stamp: str, runtime_s: float | None) -> Path:
    from fpl_rocket.model.config import BACKTEST_CONFIG as C
    from fpl_rocket.model.config import MODEL_BASE_VERSION
    from fpl_rocket.model.pipeline import git_sha_short

    pre = out / f"{stamp}_v02_decisions"
    players = pd.read_parquet(f"{pre}_raw_players.parquet")
    top2 = pd.read_parquet(f"{pre}_raw_top2.parquet")
    horizon = pd.read_parquet(f"{pre}_raw_horizon.parquet")
    cap = pd.read_parquet(f"{pre}_raw_captain.parquet")
    timing_path = Path(f"{pre}_bundle_timing.csv")
    timing = pd.read_csv(timing_path) if timing_path.exists() else pd.DataFrame()

    t1 = captain_tables(players, top2)
    t2 = seed_tables(cap)
    t3 = error_tables(players)
    t4 = probability_tables(players)
    t5 = cold_start_table(players)
    t6 = multiweek_tables(horizon)
    tables = {**t1, **t2, **t3, **t4, "cold_start": t5, **t6}
    write_csvs(str(pre), tables)
    tsum = (
        timing.groupby("n_sims")[
            ["model_s", "simulate_s", "fit_s", "write_sims_s", "xp_sims_mb", "sims_memory_mb"]
        ]
        .mean()
        .reset_index()
        if len(timing)
        else pd.DataFrame()
    )
    test = players[players["season"].isin(TEST_SEASONS)]
    vs4 = t2["seeds_vs_n4000"]
    xi_check = float(t1["captain_per_gw"].loc[lambda d: d["season"].isin(TEST_SEASONS), "total_model"].mean())

    L: list[str] = [
        f"# xP model {MODEL_BASE_VERSION}-{git_sha_short()}: decision-level diagnostics ({stamp})",
        "",
        "RETROSPECTIVE track (vaastav archive, end-of-GW data; see the v0.2 summary for the caveats).",
        f"The frozen v0.2 model (`newcomer_fallback = {C.newcomer_fallback}`, `bonus_sim = {C.bonus_sim}`,",
        f"N = {C.n_sims},",
        "seed = config seed + GW: the SAME predictions as the v0.2 report) and the form-per-90 baseline on",
        f"2024-25 + 2025-26 ({test.groupby(['season', 'gw']).ngroups} folds) and 2026-27 GW1-6 (separately).",
        "No modelling choices were made in this step. Check: mean XI points of the model on the 76 folds",
        f"= {xi_check:.3f} (v0.2 report: 60.526).",
        "",
        "Paired differences: mean over GWs, 95% CI from a moving-block bootstrap (block 4, 2000 resamples,",
        "blocks within a season), two-sided sign test. A block-4 bootstrap over the 6 GWs of 2026-27 has too",
        "few distinct resamples to be meaningful (shown for completeness).",
        "",
    ]
    if runtime_s is not None:
        L += [f"Runtime: {runtime_s / 60:.1f} min (parallel workers; bundle timing sequential).", ""]
    L += [
        "## 1. Captaincy decomposition",
        "",
        "XI = the greedy legal best XI by prediction (backtest.best_xi), captain = its highest",
        "prediction, no",
        "vice-captain (a captain who does not play scores 0 bonus), exactly the xi_points of the v0.2",
        "report.",
        "xi_no_cap = actual points of the XI; cap_bonus = the captain's actual points (the doubling);",
        "total =",
        "their sum. Differences are model minus baseline.",
        "",
        md_table(t1["captain_paired"]),
        "",
        "Captain quality. cap_pct = mid-rank percentile of the captain's actual points among the GW's",
        "players",
        "who played (1 = best); cap_rank = 1 + players who played with more points; top10 / top50 = cap_rank",
        "<= 10 / 50; same_captain_share = model and baseline captain the same player.",
        "",
        md_table(t1["captain_quality"]),
        "",
        "Top of the distribution: the top-1 / 3 / 5 players by prediction, mean prediction vs mean actual",
        "points (per GW, then averaged).",
        "",
        md_table(t1["captain_topk_calibration"]),
        "",
        "Gap between the two highest xP of each GW (the captain choice) and its Monte Carlo SE from the",
        "paired",
        "simulations (common draws) at N = 1000; within_1se / within_2se = share of GWs where the gap is",
        "smaller than 1 / 2 SE (the order of the top two is then a coin flip of the simulation).",
        "",
        md_table(t1["captain_top2_gap"]),
        "",
        "## 2. Captain and XI under simulation noise",
        "",
        f"10 fixed folds ({', '.join(f'{s} GW{g}' for s, g in CAPTAIN_FOLDS)}), each fitted once and",
        "simulated",
        "with N = 4000 (seeds 1, 2) and N = 1000 (seeds 1-5). Reference pair (A, B) = the top-2 xP of the",
        "N = 4000 seed-1 run; ref_diff = xP(A) - xP(B) per run. captain_change = share of seed pairs",
        "(same N)",
        "with a different captain; xi_changed = XI members that differ between two runs of the same N.",
        "",
        md_table(t2["seeds_summary"]),
        "",
        md_table(
            t2["seeds_per_fold"][
                [
                    "season",
                    "gw",
                    "n1000_distinct_captains",
                    "n4000_distinct_captains",
                    "n1000_ref_diff_mean",
                    "n1000_ref_diff_sd_seeds",
                    "n1000_ref_diff_min",
                    "n1000_ref_diff_max",
                    "n4000_ref_diff_mean",
                    "n1000_ref_diff_se_mc",
                    "n4000_ref_diff_se_mc",
                    "n1000_xi_changed_mean",
                    "n4000_xi_changed_mean",
                ]
            ]
        ),
        "",
        "N = 1000 runs against the N = 4000 seed-1 run: same captain in "
        f"{vs4['same_captain_as_n4000'].mean():.0%} of runs; XI members changed: mean "
        f"{vs4['xi_changed_vs_n4000'].mean():.2f}, max {vs4['xi_changed_vs_n4000'].max()}.",
        "",
    ]
    if len(tsum):
        L += [
            "Bundle build cost (run_bundle_model on the recorded bundle: fit + 6 GWs x every player, then",
            "xp_sims.parquet written with zstd as the bundle does; mean of repeats; OMP threads = 2 to",
            "approximate",
            "the GitHub runner):",
            "",
            md_table(tsum, 2),
            "",
        ]
    L += [
        "## 3. Error decomposition by minutes class",
        "",
        "Identical rows for both predictors. Class = the largest per-fixture minutes of the GW (0 / 1-59",
        "/ 60+).",
        "mse_contrib = SSE of the class / all rows (adds up to the MSE); share_of_reduction = the",
        "class's part",
        "of the model's MSE reduction over the baseline (adds up to 1).",
        "",
        md_table(t3["error_decomposition"]),
        "",
        "Points given 60+ minutes (players who played 60+ in some fixture): model_cond60 = the model's mean",
        "simulated points over its simulations with 60+ minutes (fallback: given any minutes, count in",
        "model_cond_fallbacks; no simulation with any minutes: the fold's position mean); base_p90 = the",
        "baseline's points per 90 (last 5 GWs with minutes, no",
        "availability factor) x fixtures, position rate without minutes. rmse_*_xp = the unconditional",
        "predictions on the same rows, for reference.",
        "",
        md_table(t3["error_given_60"]),
        "",
        "## 4. P(play) and P(60+): reliability and Brier score",
        "",
        "The displayed probabilities (the model's simulated p_play / p_60; outcome = played / 60+ in some",
        "fixture of the GW). Comparators per fixture combined over a double GW as independent: persistence",
        "(0.9 / 0.05 / 0.05 on the last GW's bin) and 5-GW frequency (add-0.5). Reliability bins are",
        "fixed-width probability deciles.",
        "",
        md_table(t4["prob_brier"], 4),
        "",
        md_table(t4["prob_reliability"]),
        "",
        "## 5. Cold start (GW1-3 of each season)",
        "",
        md_table(t5),
        "",
        "## 6. Multiweek windows: paired per-origin differences",
        "",
        "Per origin t (0..32) the RMSE of the 3- and 6-GW sums (players with a fixture in the window), model",
        "minus naive (horizon-1 xP x fixtures) and model minus baseline, block bootstrap over consecutive",
        "origins (block 4, within a season). The windows of neighbouring origins OVERLAP (by 2 / 5 GWs), so",
        "consecutive differences are strongly dependent; block 4 absorbs part of that, the CI is still",
        "optimistic for the 6-GW window.",
        "",
        md_table(t6["multiweek_paired"]),
        "",
    ]
    path = out / f"{stamp}_v02_decisions_summary.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summarise", default=None, help="stamp of an earlier run to re-summarise")
    ap.add_argument("--archive", default="data/archive/all_gws.parquet")
    ap.add_argument("--out", default="data/backtests")
    ap.add_argument("--bundle", default=None, help="recorded bundle directory for the N timing")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    if args.summarise:
        if args.bundle:  # (re)time the bundle build for an earlier run
            timing = bundle_timing(args.bundle, args.archive, out)
            timing.to_csv(out / f"{args.summarise}_v02_decisions_bundle_timing.csv", index=False)
        print(f"summary: {summarise(out, args.summarise, None)}")
        return 0
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    pre = out / f"{stamp}_v02_decisions"
    df = pd.read_parquet(args.archive)
    t0 = time.perf_counter()
    tasks: list[tuple[str, tuple]] = [("captain", fold) for fold in CAPTAIN_FOLDS]
    for season in TEST_SEASONS:
        tasks += [("horizon", (season, o)) for o in range(0, 38 - HORIZON_MAX + 1)]
    for season in [*TEST_SEASONS, CURRENT]:
        gws = sorted(int(g) for g in df.loc[df["season"] == season, "gw"].unique())
        tasks += [("fold", (season, g)) for g in gws]
    frames = run_tasks(tasks, args.archive, args.workers)
    if "failed" in frames:
        pd.concat(frames["failed"]).to_csv(f"{pre}_failed_tasks.csv", index=False)
    for name, key in (
        ("players", ["season", "gw", "element"]),
        ("top2", ["season", "gw"]),
        ("horizon", ["season", "origin", "h", "element"]),
        ("captain", ["season", "gw", "n_sims", "seed"]),
    ):
        f = pd.concat(frames[name], ignore_index=True).sort_values(key, kind="stable")
        f.to_parquet(f"{pre}_raw_{name}.parquet", index=False)
    runtime = time.perf_counter() - t0
    if args.bundle:
        timing = bundle_timing(args.bundle, args.archive, out)
        timing.to_csv(f"{pre}_bundle_timing.csv", index=False)
    path = summarise(out, stamp, runtime)
    meta = {"stamp": stamp, "runtime_s": runtime, "bundle": args.bundle}
    Path(f"{pre}_run.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"\nrun {stamp} in {runtime / 60:.1f} min; summary: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
