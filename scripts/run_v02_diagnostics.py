"""v0.2 diagnostics and fixes requested by the independent review (RETROSPECTIVE; archive = end-of-GW data).

Two phases, so that every modelling CHOICE is made on 2023-24 only:

  uv run python -I scripts/run_v02_diagnostics.py --phase decide
      season 2023-24 (folds trained on 2022-23 + 2023-24): the newcomer-fallback candidates and the upper-tail
      fix candidates, plus the tail attribution. Writes <ts>_v02_decide_*.csv. The choices are then written
      into src/fpl_rocket/model/config.py (newcomer_fallback, bonus_sim) BY HAND.
  uv run python -I scripts/run_v02_diagnostics.py --phase report --decide-stamp <ts of the decide run>
      2024-25, 2025-26 (76 folds) and 2026-27 (current season, reported separately) with the chosen options
      untouched: paired significance, minutes comparators, goalkeeper diagnostic, newcomers before / after,
      horizon backtest (origins 0..32), interval diagnostics, ablations, simulation stability, tail fix
      effect. Writes <ts>_v02_*.csv and one <ts>_v02_summary.md (decide tables included).
  --summarise <ts>   rebuild the report from the raw frames of an earlier report run (no refitting).

Folds run in parallel worker processes (--workers, --threads OpenMP threads each). Set PYTHONUTF8=1 on
Windows.
"""

from __future__ import annotations

import argparse
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

DECIDE_SEASON = "2023-24"
TEST_SEASONS = ["2024-25", "2025-26"]
CURRENT = "2026-27"
HORIZON_MAX = 6
STABILITY_FOLDS = [("2024-25", 20), ("2025-26", 10), ("2025-26", 30)]
STABILITY_SETTINGS = [(1000, 1), (1000, 2), (4000, 1)]
POS = {0: "all", 1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
V01 = {"newcomer_fallback": "none", "bonus_sim": "round", "defcon_fallback": False}

_DF: pd.DataFrame | None = None


# ============================================================================================ workers
def _init(archive: str) -> None:
    global _DF
    _DF = pd.read_parquet(archive)


def _task(kind: str, args: tuple):
    import warnings

    warnings.filterwarnings("ignore")
    from fpl_rocket.model.config import BACKTEST_CONFIG
    from fpl_rocket.model.diagnostics import run_fold, run_horizon_origin, run_stability

    t0 = time.perf_counter()
    if kind == "fold":
        season, gw, variants = args
        out = run_fold(_DF, season, gw, variants)
    elif kind == "horizon":
        season, origin = args
        cfg = replace(BACKTEST_CONFIG, horizons=tuple(range(1, HORIZON_MAX + 1)))
        out = {"horizon": run_horizon_origin(_DF, season, origin, cfg, HORIZON_MAX)}
    elif kind == "stability":
        season, gw = args
        out = {"stability": run_stability(_DF, season, gw, BACKTEST_CONFIG, STABILITY_SETTINGS)}
    else:
        raise ValueError(kind)
    return kind, args[:2], out, time.perf_counter() - t0


def run_tasks(tasks: list[tuple[str, tuple]], archive: str, workers: int) -> dict[str, list[pd.DataFrame]]:
    frames: dict[str, list[pd.DataFrame]] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(archive,)) as ex:
        futures = [ex.submit(_task, k, a) for k, a in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            kind, key, out, secs = fut.result()
            for name, f in out.items():
                frames.setdefault(name, []).append(f)
            print(f"  [{i}/{len(tasks)}] {kind} {key} {secs:.0f} s", flush=True)
    return frames


def gws_of(df: pd.DataFrame, season: str) -> list[int]:
    return sorted(int(g) for g in df.loc[df["season"] == season, "gw"].unique())


# ============================================================================================ helpers
def md_table(df: pd.DataFrame, nd: int = 3) -> str:
    def fmt(v):
        if isinstance(v, float | np.floating):
            return "n/a" if np.isnan(v) else f"{v:.{nd}f}"
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def rmse(e: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")


def group_error(p: pd.DataFrame, cols: list[str], by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in p.groupby(by) if by else [((), p)]:
        r = dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True))
        r["n"] = len(g)
        r["mean_actual"] = float(g["actual"].mean())
        for c in cols:
            e = g[c].fillna(0.0) - g["actual"]
            r[f"rmse_{c}"] = rmse(e)
            r[f"mae_{c}"] = float(e.abs().mean())
            r[f"bias_{c}"] = float(e.mean())
        rows.append(r)
    return pd.DataFrame(rows)


def tail_frame(p: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Players with the collected interval / component columns of `variant` renamed to the plain prefix."""
    pre = f"{variant}__"
    cols = {c: c[len(pre) :] for c in p.columns if c.startswith(pre)}
    return p.rename(columns=cols)


# ============================================================================================ decide phase
def decide_variants():
    from fpl_rocket.model.diagnostics import Variant

    return [
        Variant("v01", V01, collect=True),
        Variant("fb_withhold", {**V01, "newcomer_fallback": "withhold"}),
        Variant("fb_prior", {**V01, "newcomer_fallback": "prior"}),
        Variant("fb_shrink", {**V01, "newcomer_fallback": "shrink"}),
        Variant("tail_poisson", {**V01, "bonus_sim": "poisson"}, collect=True),
        Variant("tail_linked", {**V01, "bonus_sim": "linked"}, collect=True),
    ]


def decide_tables(players: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from fpl_rocket.model.diagnostics import exceedance_table, pinball_table, tail_attribution

    out = {}
    fb_cols = ["xp_v01", "xp_fb_withhold", "xp_fb_prior", "xp_fb_shrink", "pred_base"]
    low = players[players["low_history"]]
    nf = group_error(low, fb_cols, [])
    nf.insert(0, "subgroup", "low_history")
    nf_all = group_error(players, fb_cols, [])
    nf_all.insert(0, "subgroup", "all players")
    by_hist = group_error(low.assign(hist=np.where(low["n_hist"] == 0, "0 GWs", "1-2 GWs")), fb_cols, ["hist"])
    by_hist.insert(0, "subgroup", "low_history by n_hist")
    out["newcomers"] = pd.concat([nf, by_hist, nf_all], ignore_index=True)
    share = players.groupby("gw")["low_history"].mean()
    out["newcomer_share"] = pd.DataFrame(
        {"season": [DECIDE_SEASON], "mean": [share.mean()], "min": [share.min()], "max": [share.max()]}
    )
    rows, pins = [], []
    for v in ("v01", "tail_poisson", "tail_linked"):
        f = tail_frame(players, v)
        ex = exceedance_table(f, "", [])
        ex.insert(0, "variant", v)
        e = f[f"xp_{v}"] - f["actual"]
        ex["rmse"] = rmse(e)
        played = exceedance_table(f[f["minutes"] > 0], "", [])
        ex["above_p80_played"] = played["above_p80"].iloc[0]
        rows.append(ex)
        pin = pinball_table(f, "", [])
        pin.insert(0, "variant", v)
        pins.append(pin)
    out["tail_candidates"] = pd.concat(rows, ignore_index=True)
    out["tail_pinball"] = pd.concat(pins, ignore_index=True)
    out["tail_attribution"] = tail_attribution(tail_frame(players, "v01"), "")
    m = metrics.groupby("predictor")[["rmse", "mae", "spearman_played", "xi_points"]].mean().reset_index()
    out["fold_means"] = m
    return out


# ============================================================================================ report phase
def report_variants():
    from fpl_rocket.model.config import BACKTEST_CONFIG as C
    from fpl_rocket.model.diagnostics import Variant

    v02 = {"newcomer_fallback": C.newcomer_fallback, "bonus_sim": C.bonus_sim, "defcon_fallback": True}
    vs = [Variant("v01", V01, collect=True), Variant("v02", v02, collect=True)]
    vs.append(Variant("v02_notail", {**v02, "bonus_sim": "round"}, collect=True))
    vs.append(Variant("v02_nofb", {**v02, "newcomer_fallback": "none"}))
    vs += [
        Variant("abl_noshrink", v02, refit="noshrink"),
        Variant("abl_naivegoals", v02, refit="naivegoals"),
        Variant("abl_persistence", v02, refit="persistence"),
        Variant("abl_noprev", v02, refit="noprev"),
    ]
    return vs


def paired_tables(metrics: pd.DataFrame) -> pd.DataFrame:
    from fpl_rocket.model.diagnostics import paired_comparison

    rows = []
    for label, seasons in (
        ("2024-25 + 2025-26 (76 GWs)", TEST_SEASONS),
        ("2024-25", ["2024-25"]),
        ("2025-26", ["2025-26"]),
        ("2026-27 (current, 6 GWs)", [CURRENT]),
    ):
        m = metrics[metrics["season"].isin(seasons)]
        for a, b in (("v02", "baseline"), ("v02", "v01")):
            t = paired_comparison(m, a, b)
            t.insert(0, "folds", label)
            rows.append(t)
    return pd.concat(rows, ignore_index=True)


def minutes_tables(lines: pd.DataFrame) -> pd.DataFrame:
    from fpl_rocket.model.diagnostics import log_loss_probs

    rows = []
    lines = lines.assign(rotation=lines["starts_last_5"].between(1, 4))
    for season, s in [*lines.groupby("season"), ("2024-25 + 2025-26", lines[lines["season"].isin(TEST_SEASONS)])]:
        for group, g in (("all", s), ("rotation_risk", s[s["rotation"]]), ("low_history", s[s["low_history"]])):
            if g.empty:
                continue
            b = g["bin"].to_numpy()
            freq = np.bincount(b, minlength=3) / len(b)
            rows.append(
                {
                    "season": season,
                    "group": group,
                    "n": len(g),
                    "hgb": log_loss_probs(g[["hgb0", "hgb1", "hgb2"]].to_numpy(), b),
                    "persistence": log_loss_probs(g[["pers0", "pers1", "pers2"]].to_numpy(), b),
                    "freq_5gw": log_loss_probs(g[["freq0", "freq1", "freq2"]].to_numpy(), b),
                    "constant_in_sample": log_loss_probs(np.tile(freq, (len(g), 1)), b),
                }
            )
    return pd.DataFrame(rows)


def gk_tables(players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from fpl_rocket.model.diagnostics import gk_diagnostic

    per_gw = gk_diagnostic(players.assign(p_play=players["v02__p_play"]), "xp_v02")
    cols = [c for c in per_gw.columns if c not in ("season", "gw")]
    s = per_gw.assign(group=np.where(per_gw["season"] == CURRENT, CURRENT, "2024-25 + 2025-26"))
    summary = s.groupby("group")[cols].mean().reset_index()
    return per_gw, summary


def newcomer_tables(players: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cols = ["xp_v01", "xp_v02_nofb", "xp_v02", "pred_base"]
    low = players[players["low_history"]]
    out = {
        "error": group_error(low, cols, ["season"]),
        "error_by_hist": group_error(
            low.assign(hist=np.where(low["n_hist"] == 0, "0 GWs", "1-2 GWs")), cols, ["season", "hist"]
        ),
        "error_all": group_error(players, cols, ["season"]),
    }
    share = players.groupby(["season", "gw"])["low_history"].mean().reset_index()
    out["share"] = share.groupby("season")["low_history"].agg(["mean", "min", "max"]).reset_index()
    out["share_per_gw"] = share
    return out


def horizon_tables(h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from fpl_rocket.backtest import spearman

    rows = []
    for (season, hh), g in [
        *h.groupby(["season", "h"]),
        *((("pooled", k), x) for k, x in h.groupby("h")),
    ]:
        r = {"season": season, "h": int(hh), "n": len(g)}
        for c in ("model", "naive", "base"):
            e = g[c] - g["actual"]
            r[f"rmse_{c}"] = rmse(e)
            r[f"mae_{c}"] = float(e.abs().mean())
            sp = [
                spearman(x.loc[x["minutes"] > 0, c], x.loc[x["minutes"] > 0, "actual"])
                for _, x in g.groupby(["season", "origin"])
            ]
            r[f"sp_played_{c}"] = float(np.nanmean(sp))
        rows.append(r)
    per_h = pd.DataFrame(rows)
    sums = []
    for window in (1, 3, 6):
        w = h[h["h"] <= window]
        agg = w.groupby(["season", "origin", "element"])[["model", "naive", "base", "actual", "minutes"]].sum()
        agg = agg.reset_index()
        for season, g in [*agg.groupby("season"), ("pooled", agg)]:
            r = {"season": season, "window_gws": window, "n": len(g)}
            for c in ("model", "naive", "base"):
                r[f"rmse_{c}"] = rmse(g[c] - g["actual"])
                sp = [
                    spearman(x.loc[x["minutes"] > 0, c], x.loc[x["minutes"] > 0, "actual"])
                    for _, x in g.groupby(["season", "origin"])
                ]
                r[f"sp_played_{c}"] = float(np.nanmean(sp))
            sums.append(r)
    return per_h, pd.DataFrame(sums)


def interval_tables(players: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from fpl_rocket.model.diagnostics import exceedance_table, pinball_table, play_group

    out = {}
    test = players[players["season"].isin(TEST_SEASONS)]
    for v in ("v02", "v01", "v02_notail"):
        f = tail_frame(test, v)
        f = f.assign(play_group=play_group(f["p_play"]), played=np.where(f["minutes"] > 0, "played", "0 min"))
        ex_all = exceedance_table(f, "", ["season"])
        ex_all.insert(1, "group", "ALL players")
        ex_grp = exceedance_table(f, "", ["season", "play_group"]).rename(columns={"play_group": "group"})
        ex_grp["group"] = "P(play) " + ex_grp["group"]
        ex_pos = exceedance_table(f.assign(pos=f["element_type"].map(POS)), "", ["season", "pos"])
        ex_pos = ex_pos.rename(columns={"pos": "group"})
        ex_played = exceedance_table(f[f["minutes"] > 0], "", ["season"])
        ex_played.insert(1, "group", "played (post-hoc, v0.1 table)")
        out[f"exceedance_{v}"] = pd.concat([ex_all, ex_grp, ex_pos, ex_played], ignore_index=True)
        pin_all = pinball_table(f, "", ["season"])
        pin_all.insert(1, "group", "ALL players")
        pin_grp = pinball_table(f, "", ["season", "play_group"]).rename(columns={"play_group": "group"})
        pin_grp["group"] = "P(play) " + pin_grp["group"]
        out[f"pinball_{v}"] = pd.concat([pin_all, pin_grp], ignore_index=True)
    return out


def ablation_tables(metrics: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    from fpl_rocket.model.diagnostics import block_bootstrap_ci

    rows = []
    for season in TEST_SEASONS:
        m = metrics[metrics["season"] == season]
        p = players[players["season"] == season]
        wide = m.pivot_table(index="gw", columns="predictor", values=["rmse", "spearman_played"])
        for v in ("abl_noshrink", "abl_naivegoals", "abl_persistence", "abl_noprev"):
            ok = wide["rmse"][v].notna()
            pp = p[p["gw"].isin(wide.index[ok])]
            r_v02 = rmse(pp["xp_v02"].fillna(0) - pp["actual"])
            r_v = rmse(pp[f"xp_{v}"].fillna(0) - pp["actual"])
            d_rmse = (wide["rmse"][v] - wide["rmse"]["v02"])[ok]
            d_sp = (wide["spearman_played"][v] - wide["spearman_played"]["v02"])[ok]
            _, lo, hi = block_bootstrap_ci([d_rmse.to_numpy()])
            _, slo, shi = block_bootstrap_ci([d_sp.to_numpy()])
            rows.append(
                {
                    "season": season,
                    "ablation": v,
                    "gws": int(ok.sum()),
                    "rmse_v02": r_v02,
                    "rmse_ablation": r_v,
                    "delta_rmse_pooled": r_v - r_v02,
                    "delta_rmse_per_gw": float(d_rmse.mean()),
                    "delta_rmse_ci": f"[{lo:+.3f}, {hi:+.3f}]",
                    "sp_played_v02": float(wide["spearman_played"]["v02"][ok].mean()),
                    "sp_played_ablation": float(wide["spearman_played"][v][ok].mean()),
                    "delta_sp_played": float(d_sp.mean()),
                    "delta_sp_ci": f"[{slo:+.3f}, {shi:+.3f}]",
                }
            )
    return pd.DataFrame(rows)


def stability_table(st: pd.DataFrame) -> pd.DataFrame:
    from fpl_rocket.model.diagnostics import stability_compare

    rows = []
    for (season, gw), g in st.groupby(["season", "gw"]):
        for a, b in (("n1000_s1", "n1000_s2"), ("n1000_s1", "n4000_s1"), ("n1000_s2", "n4000_s1")):
            rows.append({"season": season, "gw": gw, "a": a, "b": b, **stability_compare(g, a, b)})
    return pd.DataFrame(rows)


def tail_effect_tables(players: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    from fpl_rocket.model.diagnostics import exceedance_table, pinball_table

    rows = []
    for season in [*TEST_SEASONS, CURRENT]:
        p = players[players["season"] == season]
        for v in ("v02_notail", "v02"):
            f = tail_frame(p, v)
            ex = exceedance_table(f, "", [])
            pin = pinball_table(f, "", [])
            rows.append(
                {
                    "season": season,
                    "variant": v,
                    "rmse": rmse(f[f"xp_{v}"] - f["actual"]),
                    "mae": float((f[f"xp_{v}"] - f["actual"]).abs().mean()),
                    "below_p20": ex["below_p20"].iloc[0],
                    "above_p80": ex["above_p80"].iloc[0],
                    "model_above_p80": ex["model_above_p80"].iloc[0],
                    "above_p80_played": exceedance_table(f[f["minutes"] > 0], "", [])["above_p80"].iloc[0],
                    "pinball_p80": pin["model_p80"].iloc[0],
                    "pinball_p50": pin["model_p50"].iloc[0],
                    "pinball_p20": pin["model_p20"].iloc[0],
                }
            )
    return pd.DataFrame(rows)


# ============================================================================================ main
def write_csvs(prefix: str, tables: dict[str, pd.DataFrame]) -> None:
    for name, t in tables.items():
        t.to_csv(f"{prefix}_{name}.csv", index=False)


def summarise_report(out: Path, stamp: str, decide_stamp: str | None, runtime_s: float | None) -> Path:
    from fpl_rocket.model.config import BACKTEST_CONFIG as C
    from fpl_rocket.model.config import MODEL_BASE_VERSION
    from fpl_rocket.model.diagnostics import tail_attribution
    from fpl_rocket.model.pipeline import git_sha_short

    pre = out / f"{stamp}_v02"
    players = pd.read_parquet(f"{pre}_raw_players.parquet")
    metrics = pd.read_parquet(f"{pre}_raw_metrics.parquet")
    lines = pd.read_parquet(f"{pre}_raw_lines.parquet")
    horizon = pd.read_parquet(f"{pre}_raw_horizon.parquet")
    stab = pd.read_parquet(f"{pre}_raw_stability.parquet")

    paired = paired_tables(metrics)
    mins = minutes_tables(lines)
    gk_gw, gk_sum = gk_tables(players)
    newc = newcomer_tables(players)
    h_tab, h_sums = horizon_tables(horizon)
    iv = interval_tables(players)
    abl = ablation_tables(metrics, players)
    stb = stability_table(stab)
    tail = tail_effect_tables(players, metrics)
    attrib = pd.concat(
        [
            tail_attribution(tail_frame(players[players["season"] == s], "v02_notail"), "").assign(season=s)
            for s in TEST_SEASONS
        ],
        ignore_index=True,
    )
    tables = {
        "paired": paired,
        "minutes_comparators": mins,
        "gk_per_gw": gk_gw,
        "gk_summary": gk_sum,
        "newcomers_error": newc["error"],
        "newcomers_error_by_hist": newc["error_by_hist"],
        "newcomers_error_all_players": newc["error_all"],
        "newcomers_share": newc["share"],
        "newcomers_share_per_gw": newc["share_per_gw"],
        "horizon": h_tab,
        "horizon_sums": h_sums,
        "ablations": abl,
        "stability": stb,
        "tail_effect": tail,
        "tail_attribution_test": attrib,
        **iv,
    }
    write_csvs(str(pre), tables)
    season_means = metrics.groupby(["season", "predictor"])[["rmse", "mae", "spearman_played", "xi_points"]]
    write_csvs(str(pre), {"fold_means": season_means.mean().reset_index()})

    L: list[str] = []
    L += [
        f"# xP model {MODEL_BASE_VERSION}-{git_sha_short()}: diagnostics and fixes after the independent review"
        f" ({stamp})",
        "",
        "RETROSPECTIVE track: rolling origin on the vaastav archive (end-of-GW data; player pool, positions,",
        "club per fixture and the fixture list as played are known after the fact; no historical status /",
        "chance_of_playing). Every number below uses that caveat.",
        "",
        "**Choice protocol.** Every modelling choice was made on season 2023-24 only (folds trained on 2022-23 +",
        f"2023-24; decide run {decide_stamp or 'n/a'}), then frozen in src/fpl_rocket/model/config.py and",
        "reported on 2024-25 / 2025-26 (76 folds) and 2026-27 GW1-6 (current season, separately). Chosen:",
        f"`newcomer_fallback = {C.newcomer_fallback}`, `bonus_sim = {C.bonus_sim}`. The DefCon prior is a",
        "documented constant (2025-26 position rates), not a tuned choice.",
        "",
        "Variants: v01 = v0.1 behaviour; v02 = all v0.2 options; v02_notail = v02 with the v0.1 bonus rounding;",
        "v02_nofb = v02 without the newcomer fallback; abl_* = v02 minus one component (section 7).",
        "",
    ]
    if runtime_s is not None:
        L += [f"Runtime of this report run: {runtime_s / 60:.1f} min (parallel workers).", ""]

    if decide_stamp:
        dpre = out / f"{decide_stamp}_v02_decide"
        L += ["## 0. Decisions on 2023-24 (decide run)", ""]
        for name, title in (
            ("newcomers", "Newcomer fallback candidates (low_history subgroup; withhold scored as xP 0)"),
            ("newcomer_share", "Share of players flagged low_history per GW"),
            ("tail_attribution", "Where the misses above p80 come from (v0.1, ALL players)"),
            ("tail_candidates", "Upper-tail fix candidates (ALL players)"),
            ("tail_pinball", "Pinball loss of the candidates (ALL players)"),
        ):
            path = Path(f"{dpre}_{name}.csv")
            if path.exists():
                L += [f"### {title}", "", md_table(pd.read_csv(path)), ""]

    L += [
        "## 1. Paired significance (per-GW differences a - b)",
        "",
        "Mean difference over GWs with a 95% CI from a moving-block bootstrap over consecutive GWs (block 4,",
        "2000 resamples, blocks within a season), and a two-sided sign test (a_better_gws of non_tied_gws).",
        "xi_points = actual points of the best legal XI + captain by prediction (unmatched decision metric; the",
        "GW average cancels in the difference).",
        "",
        md_table(paired),
        "",
        "## 2. Minutes: classifier vs persistence vs 5-GW frequency (log-loss, 3 classes, lower is better)",
        "",
        "persistence = bin of the last GW (0.9 / 0.05 / 0.05); freq_5gw = add-0.5 bin frequencies over the last 5",
        "GWs (training base rates without history); constant_in_sample = the evaluated rows' own class",
        "frequencies (an in-sample bound for any constant). rotation_risk = 1-4 starts in the last 5 GWs.",
        "",
        md_table(mins),
        "",
        "## 3. Goalkeeper Spearman diagnostic (v02; per-GW means)",
        "",
        md_table(gk_sum),
        "",
        "## 4. Newcomers (low_history: < 3 GWs of history in the training window)",
        "",
        "Share flagged per GW:",
        "",
        md_table(newc["share"]),
        "",
        "Error on the low_history subgroup (before = v02_nofb; after = v02; v01 and the baseline for reference):",
        "",
        md_table(newc["error"]),
        "",
        md_table(newc["error_by_hist"]),
        "",
        "All players (the fallback must not hurt the rest):",
        "",
        md_table(newc["error_all"]),
        "",
        "## 5. Horizon backtest (origins 0..32; t+1 .. t+6 with the fixtures listed for those GWs)",
        "",
        "model = v02 trained on horizons 1-6; naive = the horizon-1 xP per fixture x fixtures in that GW;",
        "base = form per 90 x availability x fixtures. Fixtures as played (postponements known after the fact).",
        "",
        md_table(h_tab),
        "",
        "Sums over the first 1 / 3 / 6 GWs (per origin, players with a fixture in the window):",
        "",
        md_table(h_sums),
        "",
        "## 6. Interval diagnostics (v02)",
        "",
        "Observed share of actual points below p20 / above p80 next to the model-implied share (the mean of",
        "the model's own P(X < p20) and P(X > p80) from its simulations). Points are integers and the quantiles",
        "are inclusive integer quantiles, so a calibrated model shows LESS than 20% outside each side; the",
        "model-implied columns are the honest nominal values. Groups by the pre-kickoff P(play).",
        "",
        md_table(iv["exceedance_v02"]),
        "",
        "Pinball loss (lower is better) at 0.2 / 0.5 / 0.8: model vs the naive per-position empirical",
        "quantiles of the training window:",
        "",
        md_table(iv["pinball_v02"]),
        "",
        "## 7. Ablations (v02 minus one component; 2024-25 and 2025-26)",
        "",
        "abl_noshrink = shares without shrinkage (pseudo-minutes 0); abl_naivegoals = league-average goals",
        "instead of Dixon-Coles; abl_persistence = persistence minutes instead of the classifier; abl_noprev =",
        "no previous-season rows (GW1 cannot be predicted: compared on GW2-38). delta = ablation - v02;",
        "positive delta_rmse / negative delta_sp = the component helps.",
        "",
        md_table(abl),
        "",
        "## 8. Simulation stability (same fitted fold, different seeds / N)",
        "",
        md_table(stb),
        "",
        "## 9. Upper tail fix: effect on the test seasons",
        "",
        md_table(tail),
        "",
        "Attribution of the misses above p80 on the test seasons (v02_notail), for reference:",
        "",
        md_table(attrib),
        "",
        "## 10. DefCon fallback",
        "",
        "When the training window holds no DefCon counts for a position and the target season scores DefCon",
        "(2025-26 GW1), the per-90 prior is ModelConfig.defcon_prior90 = "
        f"{ {k: v for k, v in C.defcon_prior90.items()} } (2025-26 position rates; documented one-fold",
        "look-ahead for 2025-26 GW1). Seasons before 2025-26 get no DefCon (not scored then).",
        "",
        "## Per-season fold means",
        "",
        md_table(season_means.mean().reset_index()),
        "",
    ]
    path = out / f"{stamp}_v02_summary.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["decide", "report"], default=None)
    ap.add_argument("--summarise", default=None, help="stamp of a report run to re-summarise")
    ap.add_argument("--decide-stamp", default=None)
    ap.add_argument("--archive", default="data/archive/all_gws.parquet")
    ap.add_argument("--out", default="data/backtests")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--horizon-seasons", nargs="+", default=TEST_SEASONS)
    ap.add_argument("--horizon-step", type=int, default=1, help="use every k-th origin")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarise:
        path = summarise_report(out, args.summarise, args.decide_stamp, None)
        print(f"summary: {path}")
        return 0
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    df = pd.read_parquet(args.archive)
    t0 = time.perf_counter()
    if args.phase == "decide":
        vs = decide_variants()
        tasks = [("fold", (DECIDE_SEASON, g, vs)) for g in gws_of(df, DECIDE_SEASON)]
        frames = run_tasks(tasks, args.archive, args.workers)
        players = pd.concat(frames["players"], ignore_index=True).sort_values(["gw", "element"])
        metrics = pd.concat(frames["metrics"], ignore_index=True).sort_values(["gw", "predictor"])
        pre = out / f"{stamp}_v02_decide"
        players.to_parquet(f"{pre}_raw_players.parquet", index=False)
        metrics.to_csv(f"{pre}_metrics.csv", index=False)
        tables = decide_tables(players, metrics)
        write_csvs(str(pre), tables)
        pd.set_option("display.width", 250)
        for name, t in tables.items():
            print(f"\n== {name}\n{t.round(4).to_string(index=False)}")
        print(f"\ndecide run {stamp} in {(time.perf_counter() - t0) / 60:.1f} min")
        return 0
    vs = report_variants()
    tasks: list[tuple[str, tuple]] = []
    for season in args.horizon_seasons:
        for origin in range(0, 38 - HORIZON_MAX + 1, args.horizon_step):
            tasks.append(("horizon", (season, origin)))
    for season, gw in STABILITY_FOLDS:
        tasks.append(("stability", (season, gw)))
    for season in [*TEST_SEASONS, CURRENT]:
        tasks += [("fold", (season, g, vs)) for g in gws_of(df, season)]
    frames = run_tasks(tasks, args.archive, args.workers)
    pre = out / f"{stamp}_v02"
    for name, key in (
        ("players", ["season", "gw", "element"]),
        ("metrics", ["season", "gw", "predictor"]),
        ("lines", ["season", "gw", "element", "fixture"]),
        ("horizon", ["season", "origin", "h", "element"]),
        ("stability", ["season", "gw", "element"]),
    ):
        f = pd.concat(frames[name], ignore_index=True).sort_values(key, kind="stable")
        f.to_parquet(f"{pre}_raw_{name}.parquet", index=False)
    runtime = time.perf_counter() - t0
    path = summarise_report(out, stamp, args.decide_stamp, runtime)
    print(f"\nreport run {stamp} in {runtime / 60:.1f} min; summary: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
