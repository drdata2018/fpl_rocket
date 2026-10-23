"""Decision-level diagnostics of the frozen v0.2 model (step 3c; scripts/run_decision_diagnostics.py drives
them). RETROSPECTIVE (archive = end-of-GW data). Measurements only: nothing here changes the model.

* captain_decomposition(): the XI decision of one GW split into the XI points without the captain bonus and
  the bonus itself, plus the captain's actual rank among the GW's players who played.
* top_candidates(): mean prediction vs mean actual of the top-k players (calibration of the top).
* error_decomposition(): squared error of two predictors on identical rows split by an outcome group (minutes
  class) and the share of the MSE reduction each group carries.
* reliability_table() / brier(): probability calibration (P(play), P(60+)) and its comparators
  (combine_fixture_probs over a double GW).
* run_decision_fold(): one rolling-origin fold with the conditional xP given 60+ minutes, the baseline's
  points per 90, the comparator probabilities and the top-2 captain gap with its paired Monte Carlo SE.
* run_captain_seeds(): one fold simulated with several (N, seed) settings (captain / XI stability).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..backtest import PLAYER_TYPES, FormPer90Baseline, best_xi, gw_history
from .config import BACKTEST_CONFIG, ModelConfig
from .diagnostics import _fit_kwargs, frequency_probs, make_fold, persistence_probs
from .features import minutes_bin
from .pipeline import XPModel

MINUTES_CLASSES = ("0 min", "1-59 min", "60+ min")


# ============================================================================================ captaincy
def captain_decomposition(frame: pd.DataFrame, pred_col: str = "pred") -> dict[str, float]:
    """frame: one row per player with a fixture (element, element_type, team_id, actual, minutes, pred_col).
    XI = backtest.best_xi by pred_col, captain = its highest prediction (exactly backtest.xi_points; no
    vice-captain). Returns xi_no_cap (actual points of the XI), cap_bonus (the captain's actual points = the
    doubling), total (= xi_points), the captain and whether he played, and his actual rank among the GW's
    players who played: cap_rank = 1 + players who played with MORE points (ties in his favour), cap_pct =
    mid-rank percentile among them (1 = best; a captain who did not play is placed by his 0 points),
    cap_top10 / cap_top50 = cap_rank <= 10 / 50."""
    f = frame.assign(pred=frame[pred_col].fillna(0.0))
    xi = best_xi(f, "pred")
    cap = xi.sort_values("pred", ascending=False, kind="stable").iloc[0]
    played = f.loc[f["minutes"] > 0, "actual"].to_numpy(float)
    a = float(cap["actual"])
    above, below = int((played > a).sum()), int((played < a).sum())
    tied = len(played) - above - below
    rank = 1 + above
    return {
        "xi_no_cap": float(xi["actual"].sum()),
        "cap_bonus": a,
        "total": float(xi["actual"].sum()) + a,
        "captain": int(cap["element"]),
        "cap_pred": float(cap["pred"]),
        "cap_played": bool(cap["minutes"] > 0),
        "n_played": len(played),
        "cap_rank": rank,
        "cap_pct": (below + 0.5 * tied) / len(played) if len(played) else math.nan,
        "cap_top10": rank <= 10,
        "cap_top50": rank <= 50,
    }


def top_candidates(frame: pd.DataFrame, pred_col: str, ks: tuple[int, ...] = (1, 3, 5)) -> pd.DataFrame:
    """The top-k players of one GW by pred_col (stable order): mean prediction, mean actual, share who
    played. One row per k."""
    r = frame.assign(pred=frame[pred_col].fillna(0.0)).sort_values("pred", ascending=False, kind="stable")
    rows = []
    for k in ks:
        t = r.head(k)
        rows.append(
            {
                "k": k,
                "pred_mean": float(t["pred"].mean()),
                "actual_mean": float(t["actual"].mean()),
                "played_share": float((t["minutes"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


# ============================================================================================ errors
def minutes_class(max_bin: pd.Series | np.ndarray) -> np.ndarray:
    """Outcome class per player-GW from the largest per-fixture minutes bin: '0 min', '1-59 min' or
    '60+ min' (60+ in some fixture; a double GW with 2 x 45 minutes is '1-59 min')."""
    return np.array(MINUTES_CLASSES, dtype=object)[np.asarray(max_bin, dtype=int)]


def error_decomposition(frame: pd.DataFrame, a: str, b: str, group_col: str = "mclass") -> pd.DataFrame:
    """Squared error of predictors a and b on IDENTICAL rows (frame: actual, a, b, group_col). Per group and
    'all': rows, share of rows, RMSE within the group, the group's contribution to the overall MSE
    (SSE_group / N, so contributions add up to the MSE), its contribution to the MSE reduction (b - a) and
    its share of the total reduction (shares add up to 1; negative = a is worse than b in that group)."""
    y = frame["actual"].to_numpy(float)
    ea = frame[a].fillna(0.0).to_numpy(float) - y
    eb = frame[b].fillna(0.0).to_numpy(float) - y
    n = len(frame)
    total_red = float(np.sum(eb**2) - np.sum(ea**2)) / n if n else math.nan
    groups = [(k, v) for k, v in frame.groupby(group_col, sort=True).indices.items()]
    rows = []
    for key, idx in [*groups, ("all", np.arange(n))]:
        sa, sb = float(np.sum(ea[idx] ** 2)), float(np.sum(eb[idx] ** 2))
        red = (sb - sa) / n
        rows.append(
            {
                group_col: key,
                "n": len(idx),
                "row_share": len(idx) / n,
                f"rmse_{a}": math.sqrt(sa / len(idx)) if len(idx) else math.nan,
                f"rmse_{b}": math.sqrt(sb / len(idx)) if len(idx) else math.nan,
                f"mse_contrib_{a}": sa / n,
                f"mse_contrib_{b}": sb / n,
                "mse_reduction": red,
                "share_of_reduction": red / total_red if total_red else math.nan,
            }
        )
    return pd.DataFrame(rows)


# ============================================================================================ probabilities
def reliability_table(p: np.ndarray | pd.Series, y: np.ndarray | pd.Series, bins: int = 10) -> pd.DataFrame:
    """Fixed-width bins of the predicted probability ([0, 0.1), ..., [0.9, 1.0]): count, mean prediction,
    observed rate, gap (observed - predicted)."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    y = np.asarray(y, dtype=float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for k in range(bins):
        m = b == k
        rows.append(
            {
                "bin": f"{k / bins:.1f}-{(k + 1) / bins:.1f}",
                "n": int(m.sum()),
                "pred_mean": float(p[m].mean()) if m.any() else math.nan,
                "observed": float(y[m].mean()) if m.any() else math.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["gap"] = out["observed"] - out["pred_mean"]
    return out


def brier(p: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2)) if len(p) else math.nan


def combine_fixture_probs(lines: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Per element from per-fixture bin probabilities {prefix}0 / {prefix}1 / {prefix}2 (fixtures
    independent): {prefix}_play = 1 - prod P(0), {prefix}_60 = 1 - prod (1 - P(60+))."""
    logs = pd.DataFrame(
        {
            "element": lines["element"].to_numpy(),
            "l0": np.log(np.clip(lines[f"{prefix}0"].to_numpy(float), 1e-300, 1.0)),
            "l2": np.log(np.clip(1.0 - lines[f"{prefix}2"].to_numpy(float), 1e-300, 1.0)),
        }
    )
    g = logs.groupby("element")[["l0", "l2"]].sum()
    return pd.DataFrame(
        {f"{prefix}_play": 1.0 - np.exp(g["l0"]), f"{prefix}_60": 1.0 - np.exp(g["l2"])}, index=g.index
    )


# ============================================================================================ baseline rate
def baseline_p90(train: pd.DataFrame, form_window: int = 5) -> pd.Series:
    """The baseline's points per 90 (its 'given he plays' rate): ratio of sums over the player's last
    `form_window` GWs with minutes (the same window as FormPer90Baseline), WITHOUT the availability factor
    and the zero-recent rule. Indexed by the cross-season key (code, else -element)."""
    h = gw_history(train)
    h["minutes"] = h["minutes"].fillna(0).astype(float)
    h["points"] = h["points"].fillna(0).astype(float)
    played = h[h["minutes"] > 0]
    last = played[played.groupby("pkey").cumcount(ascending=False) < form_window]
    s = last.groupby("pkey")[["points", "minutes"]].sum()
    return (s["points"] / s["minutes"] * 90.0).rename("p90")


def position_p90(train: pd.DataFrame) -> pd.Series:
    """Points per 90 per position over the training rows with minutes (fallback for players without)."""
    m = pd.to_numeric(train["minutes"], errors="coerce").fillna(0)
    t = train.assign(m=m)[m > 0]
    g = t.groupby("element_type")[["total_points", "m"]].sum()
    return g["total_points"] / g["m"] * 90.0


# ============================================================================================ simulations
class ConditionalCollector:
    """XPModel.predict on_gw hook (one GW per predict): per element the conditional xP given 60+ minutes in
    some fixture (cond60) and given any minutes (condplay) from the joint simulations, the number of
    simulations behind cond60, and the points array (elements sorted) for paired Monte Carlo errors."""

    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []
        self.elements: np.ndarray | None = None
        self.pts: np.ndarray | None = None

    def __call__(self, rows: pd.DataFrame, sim: dict, pts: np.ndarray) -> None:
        elements = rows["element"].to_numpy()
        order = np.argsort(elements, kind="stable")
        uniq, starts = np.unique(elements[order], return_index=True)
        m = sim["minutes"][order]
        any60 = np.add.reduceat((m >= 60).astype(np.int64), starts, axis=0) > 0
        anyp = np.add.reduceat((m > 0).astype(np.int64), starts, axis=0) > 0
        n60, npl = any60.sum(axis=1), anyp.sum(axis=1)
        cond60 = np.where(n60 > 0, (pts * any60).sum(axis=1) / np.maximum(n60, 1), np.nan)
        condp = np.where(npl > 0, (pts * anyp).sum(axis=1) / np.maximum(npl, 1), np.nan)
        self.frames.append(
            pd.DataFrame({"element": uniq, "cond60": cond60, "condplay": condp, "n_sims60": n60})
        )
        self.elements, self.pts = uniq, pts

    def frame(self) -> pd.DataFrame:
        return pd.concat(self.frames, ignore_index=True) if self.frames else pd.DataFrame()

    def paired(self, a: int, b: int) -> tuple[float, float]:
        """(xP a - xP b, Monte Carlo SE of that difference from the paired (common-draw) simulations) for
        the last GW collected."""
        assert self.elements is not None and self.pts is not None, "collect a GW first"
        ia, ib = int(np.searchsorted(self.elements, a)), int(np.searchsorted(self.elements, b))
        d = self.pts[ia].astype(float) - self.pts[ib].astype(float)
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def top2_gap(xp: pd.DataFrame, col: ConditionalCollector) -> dict[str, float]:
    """The two highest xP of the GW (captain candidates), their gap and its paired Monte Carlo SE."""
    t = xp.sort_values("xp", ascending=False, kind="stable").head(2)
    a, b = int(t["element"].iloc[0]), int(t["element"].iloc[1])
    gap, se = col.paired(a, b)
    return {
        "top1": a,
        "top2": b,
        "xp_top1": float(t["xp"].iloc[0]),
        "xp_top2": float(t["xp"].iloc[1]),
        "gap": gap,
        "gap_se": se,
    }


# ============================================================================================ folds
def run_decision_fold(
    df: pd.DataFrame, season: str, gw: int, cfg: ModelConfig = BACKTEST_CONFIG, n_sims: int | None = None
) -> dict[str, pd.DataFrame]:
    """One rolling-origin fold of the frozen model (seed cfg.seed + gw: the predictions of the v0.2 report).
    players: one row per player with a fixture: actual, minutes, max_bin (largest per-fixture minutes bin),
    xp / p_play / p_60 (the displayed values), cond60 / condplay / n_sims60, pred_base (form per 90 x
    availability), base_cond (baseline points per 90 x fixtures; position rate without minutes history),
    pers_play / pers_60 / freq_play / freq_60 (comparators, fixtures combined). top2: the captain
    candidates' gap and its Monte Carlo SE."""
    fold = make_fold(df, season, gw)
    tp = fold.target_players()
    tp = tp[tp["element_type"].isin(PLAYER_TYPES)].reset_index(drop=True)
    fixtures = fold.target_fixtures()
    model = XPModel(cfg).fit(fold.train, **_fit_kwargs(fold, fixtures))
    col = ConditionalCollector()
    out = model.predict(
        fixtures.assign(horizon=1), n_sims=n_sims, sims_frame=False, seed=cfg.seed + int(gw), on_gw=col
    )
    xp = out.xp.set_index("element")
    p = tp.copy()
    p["actual"] = tp["element"].map(fold.actuals()).fillna(0.0).astype(float)
    p["minutes"] = tp["element"].map(fold.target.groupby("element")["minutes"].sum()).fillna(0).astype(float)
    t = fold.target.assign(m=pd.to_numeric(fold.target["minutes"], errors="coerce").fillna(0))
    per_fx = t.groupby(["element", "fixture"])["m"].sum()
    max_bin = pd.Series(minutes_bin(per_fx.to_numpy()), index=per_fx.index).groupby(level=0).max()
    p["max_bin"] = tp["element"].map(max_bin).fillna(0).astype(int)
    for c in ("xp", "p_play", "p_60"):
        p[c] = p["element"].map(xp[c]).astype(float)
    p = p.merge(col.frame(), on="element", how="left")
    base = FormPer90Baseline()
    base.fit(fold.train)
    p["pred_base"] = base.predict(tp, gw).to_numpy(float)
    key = tp["code"].fillna(-tp["element"]).astype("int64")
    p90 = key.map(baseline_p90(fold.train)).fillna(tp["element_type"].map(position_p90(fold.train)))
    p["base_cond"] = (p90.fillna(0.0) * tp["n_fixtures"].astype(float)).to_numpy()
    rows = out.lines
    train_bins = minutes_bin(pd.to_numeric(fold.train["minutes"], errors="coerce").fillna(0))
    base_rates = np.bincount(train_bins, minlength=3) / max(len(train_bins), 1)
    pers, freq = persistence_probs(rows), frequency_probs(rows, base_rates)
    comp = pd.DataFrame(
        {
            "element": rows["element"].to_numpy(),
            **{f"pers{k}": pers[:, k] for k in range(3)},
            **{f"freq{k}": freq[:, k] for k in range(3)},
        }
    )
    probs = combine_fixture_probs(comp, "pers").join(combine_fixture_probs(comp, "freq"))
    p = p.merge(probs, left_on="element", right_index=True, how="left")
    top2 = pd.DataFrame([{"season": season, "gw": gw, **top2_gap(out.xp, col)}])
    return {"players": p.assign(season=season, gw=gw), "top2": top2}


def run_captain_seeds(
    df: pd.DataFrame, season: str, gw: int, cfg: ModelConfig, settings: list[tuple[int, int]]
) -> pd.DataFrame:
    """One fold fitted once, simulated with each (n_sims, seed). Reference captain pair (A, B) = the top-2
    xP of the FIRST setting. One row per setting: captain (backtest.xi_points rule), top-1 / top-2 and their
    gap with its paired Monte Carlo SE, xP(A) - xP(B) with its SE, and the XI members."""
    fold = make_fold(df, season, gw)
    tp = fold.target_players()
    tp = tp[tp["element_type"].isin(PLAYER_TYPES)].reset_index(drop=True)
    fixtures = fold.target_fixtures()
    model = XPModel(cfg).fit(fold.train, **_fit_kwargs(fold, fixtures))
    targets = fixtures.assign(horizon=1)
    rows: list[dict] = []
    ref: tuple[int, int] | None = None
    for n, seed in settings:
        col = ConditionalCollector()
        out = model.predict(targets, n_sims=n, sims_frame=False, seed=seed, on_gw=col)
        f = tp.assign(pred=tp["element"].map(out.xp.set_index("element")["xp"]).fillna(0.0))
        xi = best_xi(f, "pred")
        cap = int(xi.sort_values("pred", ascending=False, kind="stable").iloc[0]["element"])
        t2 = top2_gap(out.xp, col)
        if ref is None:
            ref = (int(t2["top1"]), int(t2["top2"]))
        d, se = col.paired(*ref)
        rows.append(
            {
                "season": season,
                "gw": gw,
                "n_sims": n,
                "seed": seed,
                "captain": cap,
                "top1": t2["top1"],
                "top2": t2["top2"],
                "gap_top2": t2["gap"],
                "gap_top2_se": t2["gap_se"],
                "ref_a": ref[0],
                "ref_b": ref[1],
                "ref_diff": d,
                "ref_diff_se": se,
                "xi": " ".join(str(e) for e in sorted(int(x) for x in xi["element"])),
            }
        )
    return pd.DataFrame(rows)


def xi_changed(a: str, b: str) -> int:
    """Number of XI members of a not in b (XI given as space-separated element ids)."""
    return len(set(a.split()) - set(b.split()))
