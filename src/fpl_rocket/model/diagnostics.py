"""v0.2 diagnostics requested by the independent review (scripts/run_v02_diagnostics.py drives them).

Everything here is RETROSPECTIVE (archive = end-of-GW data). Pieces:

* statistics: moving-block bootstrap CI of a mean over consecutive GWs (blocks never cross seasons), sign
  test;
  paired_comparison() of two predictors' per-GW metrics.
* minutes comparators: persistence (the bin of the last GW, smoothed hi / lo) and 5-GW frequency (add-alpha);
  multi-class log-loss.
* goalkeeper Spearman diagnostic (ties of the zero group, played-only, contender set, collapse test).
* interval diagnostics: exceedance below p20 / above p80 by group with the model-implied rates (the share of
  the model's OWN simulations outside its integer quantiles: the honest nominal for a discrete
  distribution), pinball loss, the naive per-position empirical-quantile predictor.
* tail attribution: actual minus expected points per scoring component for the misses above p80.
* run_fold(): one rolling-origin fold - fit once, predict every requested variant (prediction-time options
  via XPModel.with_config, or a component refit for the ablations), baseline, comparators, components.
* run_horizon_origin(): from origin t (last finished GW) predict t+1 .. t+6 with the fixtures of those GWs.
* run_stability(): one fold predicted with different seeds / simulation counts.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest import (
    PLAYER_TYPES,
    Fold,
    FormPer90Baseline,
    point_metrics,
    previous_season,
    spearman,
    xi_points,
)
from .bonus import BonusModel
from .config import BACKTEST_CONFIG, ModelConfig
from .features import low_history, minutes_bin
from .pipeline import XPModel
from .rates import PlayerRates
from .simulate import COMPONENTS, component_points
from .team_goals import DixonColes, results_from_rows

QUANTILES = (0.2, 0.5, 0.8)


# ============================================================================================ statistics
def block_bootstrap_ci(
    groups: list[np.ndarray] | list[pd.Series],
    block: int = 4,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(mean, lo, hi) of the pooled mean of several consecutive series (one per season), 95% percentile CI
    from a moving-block bootstrap: each series is rebuilt from blocks of `block` consecutive values drawn
    with replacement (blocks never cross series), truncated to its length. NaNs are dropped first."""
    xs = [np.asarray(g, dtype=float) for g in groups]
    xs = [x[~np.isnan(x)] for x in xs]
    xs = [x for x in xs if len(x)]
    if not xs:
        return (math.nan, math.nan, math.nan)
    total = sum(len(x) for x in xs)
    mean = float(np.concatenate(xs).sum() / total)
    rng = np.random.default_rng(seed)
    sums = np.zeros(n_boot)
    for x in xs:
        n = len(x)
        b = min(block, n)
        nb = -(-n // b)
        starts = rng.integers(0, n - b + 1, size=(n_boot, nb))
        idx = (starts[:, :, None] + np.arange(b)).reshape(n_boot, -1)[:, :n]
        sums += x[idx].sum(axis=1)
    stats = sums / total
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return mean, float(lo), float(hi)


def sign_test(diff: np.ndarray | pd.Series, lower_is_better: bool) -> tuple[int, int, float]:
    """(GWs where the first predictor is better, GWs without a tie, two-sided binomial p-value)."""
    from scipy.stats import binomtest

    d = np.asarray(diff, dtype=float)
    d = d[~np.isnan(d) & (d != 0)]
    if not len(d):
        return 0, 0, math.nan
    k = int((d < 0).sum() if lower_is_better else (d > 0).sum())
    return k, len(d), float(binomtest(k, len(d), 0.5).pvalue)


PAIRED_METRICS = (("rmse", True), ("mae", True), ("spearman_played", False), ("xi_points", False))


def paired_comparison(
    folds: pd.DataFrame,
    a: str,
    b: str,
    *,
    block: int = 4,
    n_boot: int = 2000,
    seed: int = 0,
    metrics: tuple[tuple[str, bool], ...] = PAIRED_METRICS,
) -> pd.DataFrame:
    """folds: one row per (season, gw, predictor) with the metric columns. Per metric: mean of a - b over
    the GWs (pooled over the seasons given), block-bootstrap 95% CI, and the sign test (a better)."""
    rows = []
    for metric, lower in metrics:
        wide = folds.pivot_table(index=["season", "gw"], columns="predictor", values=metric)
        if a not in wide or b not in wide:
            continue
        diff = (wide[a] - wide[b]).sort_index()
        groups = [g.to_numpy() for _, g in diff.groupby(level="season")]
        mean, lo, hi = block_bootstrap_ci(groups, block, n_boot, seed)
        k, n, p = sign_test(diff.to_numpy(), lower)
        rows.append(
            {
                "metric": metric,
                "a": a,
                "b": b,
                "gws": int(diff.notna().sum()),
                "mean_a": float(wide[a].mean()),
                "mean_b": float(wide[b].mean()),
                "mean_diff": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "a_better_gws": k,
                "non_tied_gws": n,
                "sign_p": p,
                "better": "lower" if lower else "higher",
            }
        )
    return pd.DataFrame(rows)


# ============================================================================================ minutes
def persistence_probs(rows: pd.DataFrame, hi: float = 0.9, lo: float = 0.05) -> np.ndarray:
    """P(0 / 1-59 / 60+) = the bin of the player's last GW with a row (60+ if any fixture had 60+, else
    1-59 if any minutes, else 0; no history = 0), smoothed hi / lo / lo."""
    b = rows[["bin0_1", "bin1_1", "bin2_1"]].astype(float).fillna(0.0).to_numpy()
    last = np.where(b[:, 2] > 0, 2, np.where(b[:, 1] > 0, 1, 0))
    p = np.full((len(rows), 3), lo)
    p[np.arange(len(rows)), last] = hi
    return p / p.sum(axis=1, keepdims=True)


def frequency_probs(rows: pd.DataFrame, base: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """P(bin) = (fixtures in the bin over the last 5 GWs + alpha) / (fixtures + 3 alpha); without history the
    training base rates `base`."""
    c = rows[["bin0_5", "bin1_5", "bin2_5"]].astype(float).fillna(0.0).to_numpy()
    n = c.sum(axis=1)
    p = (c + alpha) / (n[:, None] + 3 * alpha)
    p[n == 0] = np.asarray(base, dtype=float)
    return p


def log_loss_probs(p: np.ndarray, bins: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, None)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(p[np.arange(len(p)), np.asarray(bins, dtype=int)])))


# ============================================================================================ goalkeepers
def gk_diagnostic(players: pd.DataFrame, model_col: str, base_col: str = "pred_base") -> pd.DataFrame:
    """Per (season, gw) over goalkeepers: tied zero groups, Spearman (all, played, contender set) for the
    model and the baseline, and the collapse test (model predictions of keepers with 0 minutes in the last 3
    GWs set to one common value, 0)."""
    out = []
    gk = players[players["element_type"] == 1]
    for (season, gw), g in gk.groupby(["season", "gw"]):
        played = g["minutes"] > 0
        contender = g["started_last"].astype(bool) | (g["p_play"] >= 0.5)
        collapsed = g[model_col].where(~g["zero_last3"].astype(bool), 0.0)
        out.append(
            {
                "season": season,
                "gw": gw,
                "n_gk": len(g),
                "actual_zero": int((g["actual"] == 0).sum()),
                "zero_minutes": int((~played).sum()),
                "base_pred_zero": int((g[base_col] == 0).sum()),
                "model_distinct_preds": int(g[model_col].round(6).nunique()),
                "zero_last3": int(g["zero_last3"].sum()),
                "sp_all_model": spearman(g[model_col], g["actual"]),
                "sp_all_base": spearman(g[base_col], g["actual"]),
                "sp_played_model": spearman(g.loc[played, model_col], g.loc[played, "actual"]),
                "sp_played_base": spearman(g.loc[played, base_col], g.loc[played, "actual"]),
                "n_contender": int(contender.sum()),
                "sp_contender_model": spearman(g.loc[contender, model_col], g.loc[contender, "actual"]),
                "sp_contender_base": spearman(g.loc[contender, base_col], g.loc[contender, "actual"]),
                "sp_all_model_collapsed": spearman(collapsed, g["actual"]),
            }
        )
    return pd.DataFrame(out)


# ============================================================================================ intervals
def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    d = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def naive_quantiles(train: pd.DataFrame) -> pd.DataFrame:
    """Empirical 20 / 50 / 80% quantiles of GW points per position over every player-GW of the training
    window (the naive interval predictor; inverted-CDF, i.e. observed integers)."""
    t = train[train["element_type"].isin(PLAYER_TYPES)]
    g = t.groupby(["season", "gw", "element"], as_index=False).agg(
        element_type=("element_type", "first"), points=("total_points", "sum")
    )
    rows = []
    for et, x in g.groupby("element_type"):
        q = np.quantile(x["points"].astype(float), QUANTILES, method="inverted_cdf")
        rows.append({"element_type": int(et), "nq20": q[0], "nq50": q[1], "nq80": q[2]})
    return pd.DataFrame(rows)


def play_group(p_play: pd.Series) -> pd.Series:
    """Pre-kickoff groups by the model's P(play): '>=0.9', '0.5-0.9', '<0.5'."""
    p = p_play.to_numpy(float)
    return pd.Series(np.where(p >= 0.9, ">=0.9", np.where(p >= 0.5, "0.5-0.9", "<0.5")), index=p_play.index)


def exceedance_table(frame: pd.DataFrame, prefix: str, by: list[str]) -> pd.DataFrame:
    """Observed share below p20 / inside / above p80 and the model-implied shares (mean of the model's own
    P(X < p20), P(X > p80)) per group. frame columns: actual, {prefix}p20, {prefix}p80, {prefix}p_below,
    {prefix}p_above."""
    rows = []
    for key, g in frame.groupby(by) if by else [((), frame)]:
        y = g["actual"]
        below = y < g[f"{prefix}p20"]
        above = y > g[f"{prefix}p80"]
        rows.append(
            {
                **dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True)),
                "n": len(g),
                "below_p20": float(below.mean()),
                "inside": float((~below & ~above).mean()),
                "above_p80": float(above.mean()),
                "model_below_p20": float(g[f"{prefix}p_below"].mean()),
                "model_above_p80": float(g[f"{prefix}p_above"].mean()),
            }
        )
    return pd.DataFrame(rows)


def pinball_table(frame: pd.DataFrame, prefix: str, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in frame.groupby(by) if by else [((), frame)]:
        r = dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True))
        r["n"] = len(g)
        for tau, col in zip(QUANTILES, ("p20", "p50", "p80"), strict=True):
            r[f"model_{col}"] = pinball(g["actual"], g[f"{prefix}{col}"], tau)
            r[f"naive_{col}"] = pinball(g["actual"], g[f"n{col.replace('p', 'q')}"], tau)
        rows.append(r)
    return pd.DataFrame(rows)


# ============================================================================================ components
def actual_components(target: pd.DataFrame, season: str = "2026-27") -> pd.DataFrame:
    """Actual points per scoring component (A_<component>, A_other = total - components: penalties, own
    goals) per element, summed over the GW's fixtures."""
    t = target[target["element_type"].isin(PLAYER_TYPES)]

    def col(c: str) -> np.ndarray:
        return pd.to_numeric(t[c], errors="coerce").fillna(0).to_numpy(float) if c in t else np.zeros(len(t))

    comp = component_points(
        t["element_type"].astype(int).to_numpy(),
        col("minutes"),
        col("goals_scored"),
        col("assists"),
        col("clean_sheets"),
        col("goals_conceded"),
        col("saves"),
        col("yellow_cards"),
        col("red_cards"),
        col("bonus"),
        col("defensive_contribution"),
        season,
    )
    f = pd.DataFrame({f"A_{k}": v for k, v in comp.items()})
    f["element"] = t["element"].to_numpy()
    f["total"] = col("total_points")
    g = f.groupby("element").sum()
    g["A_other"] = g["total"] - g[[f"A_{k}" for k in COMPONENTS]].sum(axis=1)
    return g.drop(columns="total").reset_index()


def tail_attribution(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """For the misses above p80 (actual > {prefix}p80): mean actual - expected points per component, its
    share of the summed positive surplus, and how often it is the largest surplus."""
    miss = frame[frame["actual"] > frame[f"{prefix}p80"]]
    comps = list(COMPONENTS)
    if miss.empty:
        return pd.DataFrame()
    surplus = pd.DataFrame({c: miss[f"A_{c}"] - miss[f"{prefix}E_{c}"] for c in comps})
    pos = surplus.clip(lower=0)
    largest = surplus.idxmax(axis=1).value_counts(normalize=True)
    return pd.DataFrame(
        {
            "component": comps,
            "misses": len(miss),
            "mean_actual": [float(miss[f"A_{c}"].mean()) for c in comps],
            "mean_expected": [float(miss[f"{prefix}E_{c}"].mean()) for c in comps],
            "mean_surplus": [float(surplus[c].mean()) for c in comps],
            "share_of_positive_surplus": [
                float(pos[c].sum() / max(pos.to_numpy().sum(), 1e-12)) for c in comps
            ],
            "largest_surplus_share": [float(largest.get(c, 0.0)) for c in comps],
        }
    )


class SimCollector:
    """XPModel.predict on_gw hook: per element the model-implied P(X < p20), P(X > p80) and the expected
    points per component (E_<component>)."""

    def __init__(self, season: str, components: bool = True) -> None:
        self.season = season
        self.components = components
        self.frames: list[pd.DataFrame] = []

    def __call__(self, rows: pd.DataFrame, sim: dict, pts: np.ndarray) -> None:
        elements = rows["element"].to_numpy()
        order = np.argsort(elements, kind="stable")
        uniq, starts = np.unique(elements[order], return_index=True)
        q = np.quantile(pts, [0.2, 0.8], axis=1, method="inverted_cdf")
        f = pd.DataFrame(
            {
                "element": uniq,
                "p_below": (pts < q[0][:, None]).mean(axis=1),
                "p_above": (pts > q[1][:, None]).mean(axis=1),
            }
        )
        if self.components:
            et = np.broadcast_to(rows["element_type"].to_numpy(int)[:, None], sim["minutes"].shape)
            comp = component_points(
                et,
                sim["minutes"],
                sim["goals"],
                sim["assists"],
                sim["cs_points"],
                sim["goals_conceded"],
                sim["saves"],
                sim["yellow_cards"],
                sim["red_cards"],
                sim["bonus"],
                sim["defensive_contribution"],
                "2026-27",
            )
            for k, v in comp.items():
                f[f"E_{k}"] = np.add.reduceat(v.mean(axis=1)[order], starts)
        self.frames.append(f)

    def frame(self) -> pd.DataFrame:
        return pd.concat(self.frames, ignore_index=True) if self.frames else pd.DataFrame()


# ============================================================================================ variants
@dataclass(frozen=True)
class Variant:
    """A prediction of one fitted fold: `changes` = XPModel.with_config options; `refit` = an ablation that
    refits one component: noshrink (share pseudo-minutes 0), naivegoals (league-average goals instead of
    Dixon-Coles), persistence (persistence minutes instead of the classifier), noprev (no previous-season
    rows; minutes warm-up off). `collect` = record intervals and components (SimCollector)."""

    name: str
    changes: dict = field(default_factory=dict)
    refit: str | None = None
    collect: bool = False


class PersistenceMinutes:
    """Minutes model replacing the classifier's P(0 / 1-59 / 60+) by persistence_probs (expected minutes
    within the bins and the availability factor from the wrapped model)."""

    def __init__(self, base) -> None:
        self.base = base
        self.n_train = base.n_train

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        out = self.base.predict(rows)
        p = persistence_probs(rows)
        play = p[:, 1] + p[:, 2]
        cap = np.minimum(play, out["availability"].to_numpy(float))
        scale = np.divide(cap, play, out=np.zeros_like(play), where=play > 0)
        out["p1"], out["p2"] = p[:, 1] * scale, p[:, 2] * scale
        out["p0"] = 1.0 - out["p1"] - out["p2"]
        return out


def naive_goals_model(model: XPModel) -> DixonColes:
    """A DixonColes object with league-average goals: lambda_home = training mean home goals, lambda_away =
    mean away goals for every fixture, rho = 0 (independent Poisson)."""
    dc = DixonColes(model.cfg)
    dc.teams = list(model.dc.teams)
    dc.intercept = math.log(max(model.naive.away_mean, 1e-6))
    dc.home = math.log(max(model.naive.home_mean, 1e-6) / max(model.naive.away_mean, 1e-6))
    dc.rho, dc.attack, dc.defence, dc.prior = 0.0, {}, {}, {}
    dc.n_matches, dc.converged = model.dc.n_matches, True
    return dc


def derive(model: XPModel, variant: Variant, train: pd.DataFrame, fit_kwargs: dict) -> XPModel | None:
    m = model.with_config(**variant.changes)
    h = m.history
    if variant.refit == "noshrink":
        m.cfg = dataclasses.replace(m.cfg, share_pseudo_minutes=0.0)
        m.rates = PlayerRates(m.cfg).fit(h, m.season)
        m.bonus = BonusModel(m.cfg)
        m._fit_bonus(h, results_from_rows(h))
    elif variant.refit == "naivegoals":
        m.dc = naive_goals_model(model)
        m.bonus = BonusModel(m.cfg)
        m._fit_bonus(h, results_from_rows(h))
    elif variant.refit == "persistence":
        m.minutes = PersistenceMinutes(model.minutes)
    elif variant.refit == "noprev":
        cur = train[train["season"] == fit_kwargs["season"]]
        if cur.empty:
            return None
        cfg = dataclasses.replace(model.cfg, warmup_gws=0, **variant.changes)
        m = XPModel(cfg).fit(cur, **fit_kwargs)
    elif variant.refit is not None:
        raise ValueError(f"unknown refit {variant.refit!r}")
    return m


# ============================================================================================ one fold
def make_fold(df: pd.DataFrame, season: str, gw: int, include_previous: bool = True) -> Fold:
    s = df[df["season"] == season]
    prev = df[df["season"] == previous_season(season)] if include_previous else df.iloc[0:0]
    train = pd.concat([prev, s[s["gw"] < gw]], ignore_index=True)
    return Fold(season, gw, train, s[s["gw"] == gw].reset_index(drop=True))


def _fit_kwargs(fold: Fold, fixtures: pd.DataFrame) -> dict:
    season = fold.season
    ref = pd.to_datetime(fixtures["kickoff_time"], utc=True).min()
    train = fold.train
    teams = set(train.loc[train["season"] == season, "team"].dropna().astype(str))
    teams |= set(fixtures["team"].dropna().astype(str)) | set(fixtures["opponent"].dropna().astype(str))
    return {"season": season, "ref_time": ref, "season_teams": sorted(teams)}


def fold_metrics(frame: pd.DataFrame, pred_col: str) -> dict[str, float]:
    pred = frame[pred_col].fillna(0.0)
    m = point_metrics(pred, frame["actual"])
    played = frame["minutes"] > 0
    m["spearman_played"] = spearman(pred[played], frame.loc[played, "actual"])
    xi, _ = xi_points(frame.assign(pred=pred))
    m["xi_points"] = xi
    return m


def run_fold(
    df: pd.DataFrame,
    season: str,
    gw: int,
    variants: list[Variant],
    cfg: ModelConfig = BACKTEST_CONFIG,
    n_sims: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Fit once on the fold (previous season + earlier GWs) and predict every variant. Returns
    players (one row per player with a fixture: actual, minutes, components, baseline, xp_<variant>, and for
    collected variants <variant>__p20 / p50 / p80 / p_play / p_below / p_above / E_<component>),
    lines (per player-fixture: classifier and comparator minutes probabilities, actual bin) and
    metrics (one row per predictor)."""
    fold = make_fold(df, season, gw)
    tp = fold.target_players()
    tp = tp[tp["element_type"].isin(PLAYER_TYPES)].reset_index(drop=True)
    fixtures = fold.target_fixtures()
    kwargs = _fit_kwargs(fold, fixtures)
    model = XPModel(cfg).fit(fold.train, **kwargs)
    targets = fixtures.assign(horizon=1)
    seed = cfg.seed + int(gw)

    players = tp.copy()
    players["actual"] = tp["element"].map(fold.actuals()).fillna(0.0).astype(float)
    players["minutes"] = (
        tp["element"].map(fold.target.groupby("element")["minutes"].sum()).fillna(0).astype(float)
    )
    n_hist, low = low_history(tp, fold.train, cfg.low_history_gws)
    players["n_hist"], players["low_history"] = n_hist.to_numpy(), low.to_numpy(bool)
    players["value"] = tp["element"].map(fold.target.groupby("element")["value"].first()).astype(float)
    base = FormPer90Baseline()
    base.fit(fold.train)
    players["pred_base"] = base.predict(tp, gw).to_numpy(float)
    players = players.merge(actual_components(fold.target), on="element", how="left")
    players = players.merge(naive_quantiles(fold.train), on="element_type", how="left")

    # classifier lines (v0.1 minutes) and the comparators
    raw = model.with_config(newcomer_fallback="none")
    rows, _ = raw.lines(targets)
    act = fold.target.groupby(["element", "fixture"])["minutes"].sum()
    rows["actual_minutes"] = [
        float(act.get((e, f), 0.0)) for e, f in zip(rows["element"], rows["fixture"], strict=True)
    ]
    train_bins = minutes_bin(pd.to_numeric(fold.train["minutes"], errors="coerce").fillna(0))
    base_rates = np.bincount(train_bins, minlength=3) / max(len(train_bins), 1)
    pers = persistence_probs(rows)
    freq = frequency_probs(rows, base_rates)
    lines = pd.DataFrame(
        {
            "season": season,
            "gw": gw,
            "element": rows["element"].to_numpy(),
            "fixture": rows["fixture"].to_numpy(),
            "element_type": rows["element_type"].to_numpy(),
            "starts_last_5": rows["starts_last_5"].to_numpy(),
            "low_history": rows["low_history"].to_numpy(bool),
            "bin": minutes_bin(rows["actual_minutes"]),
            **{f"hgb{k}": rows[f"p{k}"].to_numpy() for k in range(3)},
            **{f"pers{k}": pers[:, k] for k in range(3)},
            **{f"freq{k}": freq[:, k] for k in range(3)},
        }
    )
    per_el = rows.groupby("element").agg(
        started_last=("starts_pf_1", lambda s: bool((s.fillna(0) > 0).any())),
        zero_last3=("min_pf_3", lambda s: bool((s.fillna(0) == 0).all())),
    )
    players = players.merge(per_el, left_on="element", right_index=True, how="left")

    metrics = [{"predictor": "baseline", **fold_metrics(players.assign(p=players["pred_base"]), "p")}]
    for v in variants:
        m = derive(model, v, fold.train, kwargs)
        if m is None:
            players[f"xp_{v.name}"] = np.nan
            continue
        col = SimCollector(season) if v.collect else None
        out = m.predict(targets, n_sims=n_sims, sims_frame=False, seed=seed, on_gw=col)
        xp = out.xp.set_index("element")
        players[f"xp_{v.name}"] = players["element"].map(xp["xp"]).astype(float)
        if v.collect:
            extra = xp[["p20", "p50", "p80", "p_play"]].join(col.frame().set_index("element"))
            extra.columns = [f"{v.name}__{c}" for c in extra.columns]
            players = players.merge(extra, left_on="element", right_index=True, how="left")
        metrics.append({"predictor": v.name, **fold_metrics(players, f"xp_{v.name}")})
    met = pd.DataFrame(metrics).assign(season=season, gw=gw)
    players = players.assign(season=season, gw=gw)
    return {"players": players, "lines": lines, "metrics": met}


# ============================================================================================ horizons
def run_horizon_origin(
    df: pd.DataFrame, season: str, origin: int, cfg: ModelConfig, max_h: int = 6, n_sims: int | None = None
) -> pd.DataFrame:
    """From origin t (= last finished GW; 0 = pre-season) predict GWs t+1 .. t+max_h with the fixtures the
    archive lists for them. One row per (player with a fixture, GW): h, model, naive (the horizon-1 xP per
    fixture x fixtures in that GW; a player blank at t+1 uses his first predicted GW), baseline (form per 90
    x availability x fixtures), actual, minutes."""
    s = df[df["season"] == season]
    prev = df[df["season"] == previous_season(season)]
    train = pd.concat([prev, s[s["gw"] <= origin]], ignore_index=True)
    gws = [g for g in range(origin + 1, origin + max_h + 1) if (s["gw"] == g).any()]
    if not gws or train.empty:
        return pd.DataFrame()
    future = s[s["gw"].isin(gws)]
    first = Fold(season, gws[0], train, future[future["gw"] == gws[0]])
    fixtures_all = pd.concat(
        [Fold(season, g, train, future[future["gw"] == g]).target_fixtures() for g in gws], ignore_index=True
    )
    kwargs = _fit_kwargs(first, fixtures_all[fixtures_all["gw"] == gws[0]])
    model = XPModel(cfg).fit(train, **kwargs)
    targets = fixtures_all.assign(horizon=fixtures_all["gw"].astype(int) - origin)
    out = model.predict(targets, n_sims=n_sims, sims_frame=False, seed=cfg.seed + 1000 * origin)
    base = FormPer90Baseline()
    base.fit(train)
    frames = []
    for g in gws:
        fold = Fold(season, g, train, future[future["gw"] == g])
        tp = fold.target_players()
        tp = tp[tp["element_type"].isin(PLAYER_TYPES)].reset_index(drop=True)
        xp = out.xp[out.xp["gw"] == g].set_index("element")
        frames.append(
            tp.assign(
                origin=origin,
                h=g - origin,
                model=tp["element"].map(xp["xp"]).fillna(0.0).astype(float).to_numpy(),
                base=base.predict(tp, g).to_numpy(float),
                actual=tp["element"].map(fold.actuals()).fillna(0.0).astype(float).to_numpy(),
                minutes=tp["element"]
                .map(fold.target.groupby("element")["minutes"].sum())
                .fillna(0)
                .to_numpy(),
            )
        )
    f = pd.concat(frames, ignore_index=True)
    per_fx = (f["model"] / f["n_fixtures"].clip(lower=1)).where(f["n_fixtures"] > 0)
    first_h = f.assign(pf=per_fx).dropna(subset=["pf"]).sort_values("h").groupby("element")["pf"].first()
    h1 = f[f["h"] == f["h"].min()].set_index("element")
    h1_pf = (h1["model"] / h1["n_fixtures"].clip(lower=1)).reindex(first_h.index)
    naive_pf = h1_pf.fillna(first_h)
    f["naive"] = f["element"].map(naive_pf).fillna(0.0) * f["n_fixtures"]
    return f.assign(season=season)


# ============================================================================================ stability
def run_stability(
    df: pd.DataFrame, season: str, gw: int, cfg: ModelConfig, settings: list[tuple[int, int]], top: int = 50
) -> pd.DataFrame:
    """One fold predicted with (n_sims, seed) settings; xP per element per setting (column n{N}_s{seed})."""
    fold = make_fold(df, season, gw)
    fixtures = fold.target_fixtures()
    model = XPModel(cfg).fit(fold.train, **_fit_kwargs(fold, fixtures))
    targets = fixtures.assign(horizon=1)
    cols = {}
    for n, seed in settings:
        out = model.predict(targets, n_sims=n, sims_frame=False, seed=seed)
        cols[f"n{n}_s{seed}"] = out.xp.set_index("element")["xp"]
    return pd.DataFrame(cols).assign(season=season, gw=gw).reset_index()


def stability_compare(frame: pd.DataFrame, a: str, b: str, top: int = 50) -> dict[str, float]:
    d = (frame[a] - frame[b]).abs()
    top_a = set(frame.nlargest(top, a)["element"])
    top_b = set(frame.nlargest(top, b)["element"])
    return {
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "top_changed": len(top_a - top_b),
    }
