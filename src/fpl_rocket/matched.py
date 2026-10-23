"""Matched-policy backtest (docs/PLAN_v2.md section 4, "matched policy comparison") - RETROSPECTIVE.

Every policy starts each season from the SAME synthetic squad and state, and is carried forward week by week
with real outcomes under identical rules:

* start (GW1): the xP-optimal legal 15 under 100.0 by the BASELINE predictor's GW1 prediction (so neither
  policy is favoured by its own start), bank = 100.0 - its cost, 1 free transfer entering GW2, no chips
  (chips are never played in this backtest);
* MODEL policy: the planner's multi-GW objective (horizon 3, decay 0.85, at most 2 transfers per GW, hits
  allowed when the objective says so) on the model predictor's retrospective predictions (refit per GW,
  fpl_rocket.backtest.PREDICTORS["model"]);
* BASELINE policy: the same planner on the baseline predictor (form per 90 x availability);
* HOLD policy: never transfers; best XI + captain by the baseline each GW.
* every policy: players with fewer than 3 GWs of history before the GW (backtest.low_history, the rule behind
  the live `low_history` flag) are never captain / vice and never bought; owned ones may stay in the XI.

Predictions for later GWs of the horizon: the model predictor predicts the target GW only (horizon 1), so a
later GW uses the same per-fixture xP x that GW's fixture count (a blank scores 0, a double twice). The
baseline is per fixture by construction, so both policies are treated alike. A player without a fixture in
the target GW keeps his last per-fixture prediction.

Prices: the archive `value` of each GW is the buy price; selling prices follow the rules engine
(selling_price(purchase, value)) on purchase prices tracked in the simulation. Only players with a fixture in
GW t can be bought in GW t (a limitation: a player blank in GW t cannot be bought that week). A squad player
without a row in GW t scores 0 and keeps his last known value.

Realised points: actual GW points of the chosen XI with FPL's automatic substitutions (bench order by xP,
reserve GK first), armband to the vice when the captain did not play, minus 4 per hit.

Reported per season and policy: net points (after hits), transfers, hits, max drawdown of the cumulative
(net - GW average) series against fpl_rocket.backtest.gw_average_proxy, and the paired per-GW difference
MODEL minus BASELINE with a moving-block bootstrap CI (block 4, 2000 resamples). This is the first MATCHED
comparison; the inputs are end-of-GW archive data (see the retrospective-track caveats).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .backtest import LOW_HISTORY_GWS, PLAYER_TYPES, PREDICTORS, Predictor, gw_average_proxy, rolling_origin
from .model.features import low_history
from .planner import (
    BATCH_TIME_LIMIT_S,
    MULTI_GW,
    ONE_GW_WITH_TRANSFERS,
    Candidate,
    Owned,
    PlannerInput,
    PlanSettings,
    bench_order,
    best_xi,
    captain_pair,
    realised_points,
    solve,
)
from .rules.season_2026_27 import SQUAD, TRANSFERS, selling_price

POLICIES = ("model", "baseline", "hold")
HORIZON = 3
MAX_TRANSFERS = 2
DECAY = 0.85
BLOCK = 4
RESAMPLES = 2000
LABEL = (
    "MATCHED policy comparison (first one), RETROSPECTIVE: same start squad, budget, free transfers and "
    "chip state for every policy; archive end-of-GW data"
)


# ============================================================================================ data
@dataclass
class SeasonData:
    season: str
    gws: list[int]
    # per GW: one row per element with a fixture: element_type, team_id, value, points, minutes, n_fixtures
    per_gw: dict[int, pd.DataFrame]
    targets: dict[int, pd.DataFrame]  # archive rows of each GW (for the GW average proxy)


def season_data(df: pd.DataFrame, season: str) -> SeasonData:
    s = df[(df["season"] == season) & (df["element_type"].isin(PLAYER_TYPES))]
    per_gw, targets = {}, {}
    for gw, g in s.groupby("gw", sort=True):
        agg = g.groupby("element", sort=True).agg(
            element_type=("element_type", "first"),
            team_id=("team_id", "first"),
            value=("value", "first"),
            points=("total_points", "sum"),
            minutes=("minutes", "sum"),
            n_fixtures=("fixture", "nunique"),
        )
        agg["value"] = pd.to_numeric(agg["value"], errors="coerce")
        per_gw[int(gw)] = agg.dropna(subset=["value"]).astype(
            {"element_type": int, "team_id": int, "value": int, "points": int, "minutes": int}
        )
        targets[int(gw)] = g
    return SeasonData(season, sorted(per_gw), per_gw, targets)


def per_fixture_predictions(
    df: pd.DataFrame, season: str, predictor: Predictor, log: Callable[[str], None] | None = None
) -> dict[int, pd.Series]:
    """{gw: per-fixture xP by element} from rolling-origin folds (fit on everything before the GW)."""
    out: dict[int, pd.Series] = {}
    for fold in rolling_origin(df, [season]):
        tp = fold.target_players()
        tp = tp[tp["element_type"].isin(PLAYER_TYPES)].reset_index(drop=True)
        if tp.empty:
            continue
        t0 = time.perf_counter()
        predictor.fit(fold.train)
        pred = predictor.predict(tp, fold.target_gw, fixtures=fold.target_fixtures()).reset_index(drop=True)
        per_fix = pred.to_numpy(float) / tp["n_fixtures"].clip(lower=1).to_numpy(float)
        out[fold.target_gw] = pd.Series(per_fix, index=tp["element"].astype(int).to_numpy())
        if log:
            log(f"{predictor.name} {season} GW{fold.target_gw}: {time.perf_counter() - t0:.1f} s")
    return out


def low_history_sets(df: pd.DataFrame, season: str) -> dict[int, frozenset[int]]:
    """{gw: players with fewer than LOW_HISTORY_GWS GWs of history before the GW} (backtest.low_history rule,
    the same for every predictor): never captain nor bought, in every policy."""
    out: dict[int, frozenset[int]] = {}
    for fold in rolling_origin(df, [season]):
        tp = fold.target_players()
        tp = tp[tp["element_type"].isin(PLAYER_TYPES)].reset_index(drop=True)
        _, low = low_history(tp, fold.train, LOW_HISTORY_GWS)
        out[fold.target_gw] = frozenset(int(e) for e in tp.loc[low.to_numpy(bool), "element"])
    return out


def horizon_xp(
    data: SeasonData, per_fixture: dict[int, pd.Series], t: int, horizon: int, memory: dict[int, float]
) -> tuple[list[int], dict[tuple[int, int], float]]:
    """(GWs t..t+H-1 of the season, {(element, gw): xP}) = per-fixture xP at t x fixtures in each GW.
    `memory` carries each player's last per-fixture prediction (updated in place with GW t's)."""
    pf = per_fixture.get(t)
    if pf is not None:
        memory.update({int(e): float(v) for e, v in pf.items()})
    gws = [g for g in data.gws if g >= t][:horizon]
    xp = {}
    for g in gws:
        nfix = data.per_gw[g]["n_fixtures"]
        for e, n in nfix.items():
            if int(e) in memory:
                xp[int(e), g] = memory[int(e)] * int(n)
    return gws, xp


# ============================================================================================ policies
@dataclass
class PolicyState:
    squad: dict[int, int]  # element -> purchase price (tenths)
    bank: int
    free_transfers: int
    rows: list[dict[str, Any]] = field(default_factory=list)


def _pool(info: pd.DataFrame) -> dict[int, Candidate]:
    return {
        int(e): Candidate(int(e), int(r.element_type), int(r.team_id), int(r.value))
        for e, r in zip(info.index, info.itertuples(index=False), strict=True)
    }


def initial_squad(
    data: SeasonData, xp1: dict[tuple[int, int], float], low: frozenset[int] = frozenset()
) -> tuple[dict[int, int], int]:
    """The xP-optimal legal 15 under the budget at GW1 prices (planner, empty squad, wildcard week)."""
    g = data.gws[0]
    inp = PlannerInput(
        squad=[],
        bank=SQUAD.budget,
        free_transfers=1,
        gws=[g],
        xp=xp1,
        pool=_pool(data.per_gw[g]),
        chips_available=[{"name": "wildcard", "start_event": g, "stop_event": g}],
        low_history=low,
    )
    settings = PlanSettings(
        objective=ONE_GW_WITH_TRANSFERS,
        chip="wildcard",
        max_transfers=None,
        pool_top=((1, 10), (2, 30), (3, 30), (4, 20)),
        pool_cheapest=5,
        time_limit=BATCH_TIME_LIMIT_S,
    )
    plan = solve(inp, settings)
    if plan.status != "optimal":
        raise RuntimeError(f"{data.season}: no initial squad ({plan.reason})")
    prices = data.per_gw[g]["value"]
    squad = {int(e): int(prices[e]) for e in plan.first().squad}
    return squad, SQUAD.budget - sum(squad.values())


def _owned(state: PolicyState, info: pd.DataFrame, last: dict[int, dict[str, int]]) -> list[Owned]:
    out = []
    for e, buy in sorted(state.squad.items()):
        meta = last[e]
        out.append(Owned(e, meta["element_type"], meta["team_id"], selling_price(buy, meta["value"])))
    return out


def run_season(
    df: pd.DataFrame,
    season: str,
    predictors: dict[str, Predictor],
    *,
    horizon: int = HORIZON,
    max_transfers: int = MAX_TRANSFERS,
    decay: float = DECAY,
    time_limit: float = BATCH_TIME_LIMIT_S,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """One row per (policy, GW) for one season. predictors: {"model": ..., "baseline": ...}."""
    data = season_data(df, season)
    preds = {name: per_fixture_predictions(df, season, p, log) for name, p in predictors.items()}
    first = data.gws[0]
    _, xp1 = horizon_xp(data, preds["baseline"], first, 1, {})
    lows = low_history_sets(df, season)
    start, bank = initial_squad(data, xp1, lows.get(first, frozenset()))
    if log:
        log(f"{season}: start squad cost {sum(start.values()) / 10:.1f}, bank {bank / 10:.1f}")
    states = {p: PolicyState(dict(start), bank, 1) for p in POLICIES}
    memories: dict[str, dict[int, float]] = {name: {} for name in preds}
    last: dict[int, dict[str, int]] = {}
    rows = []
    for t in data.gws:
        info = data.per_gw[t]
        for e, r in zip(info.index, info.itertuples(index=False), strict=True):
            last[int(e)] = {"element_type": r.element_type, "team_id": r.team_id, "value": r.value}
        horizons = {name: horizon_xp(data, preds[name], t, horizon, memories[name]) for name in preds}
        low = lows.get(t, frozenset())
        avg = gw_average_proxy(data.targets[t])
        pts = info["points"].to_dict()
        mins = info["minutes"].to_dict()
        for policy in POLICIES:
            st = states[policy]
            source = "baseline" if policy == "hold" else policy
            gws, xp = horizons[source]
            et = {e: last[e]["element_type"] for e in st.squad}
            score = {e: xp.get((e, t), 0.0) for e in st.squad}
            t_in: list[int] = []
            t_out: list[int] = []
            status, solve_s, limited = "hold", 0.0, False
            if policy != "hold" and t != first:
                inp = PlannerInput(
                    squad=_owned(st, info, last),
                    bank=st.bank,
                    free_transfers=st.free_transfers,
                    gws=gws,
                    xp=xp,
                    pool=_pool(info),
                    low_history=low,
                )
                plan = solve(
                    inp,
                    PlanSettings(
                        objective=MULTI_GW,
                        horizon=horizon,
                        decay=decay,
                        max_transfers=max_transfers,
                        allow_hits=True,
                        time_limit=time_limit,
                    ),
                )
                if plan.status == "solver_error":  # never scored as a silent hold
                    raise RuntimeError(f"{season} GW{t} {policy}: {plan.reason}")
                gp = plan.first()
                status, solve_s, limited = plan.status, plan.solve_seconds, plan.time_limited
                t_in, t_out = gp.transfers_in, gp.transfers_out
                sell = {o.element: o.selling_price for o in inp.squad}
                for e in t_out:
                    st.bank += sell[e]
                    del st.squad[e]
                for e in t_in:
                    st.bank -= int(info.at[e, "value"])
                    st.squad[e] = int(info.at[e, "value"])
                et = {e: last[e]["element_type"] for e in st.squad}
                score = {e: xp.get((e, t), 0.0) for e in st.squad}
                xi, cap, vice, bench = gp.xi, gp.captain, gp.vice, gp.bench
            else:
                squad = sorted(st.squad)
                xi = best_xi(squad, et, score)
                cap, vice = captain_pair(xi, score, low)
                bench = bench_order([e for e in squad if e not in xi], et, score)
            n = len(t_in)
            hits = 0 if t == first else max(0, n - st.free_transfers)
            res = realised_points(xi, bench, cap, vice, pts, mins, et)
            net = res["points"] - TRANSFERS.hit_cost * hits
            ft_in = st.free_transfers
            rows.append(
                {
                    "season": season,
                    "gw": t,
                    "policy": policy,
                    "points": res["points"],
                    "hits": hits,
                    "net": net,
                    "transfers": n,
                    "captain": cap,
                    "captain_points": int(pts.get(cap, 0)),
                    "armband": res["armband"],
                    "auto_subs": len(res["subs"]),
                    "xi_xp": float(sum(score.get(e, 0.0) for e in xi) + score.get(cap, 0.0)),
                    "gw_average": avg,
                    "bank": st.bank,
                    "squad_value": int(sum(last[e]["value"] for e in st.squad)),
                    "free_transfers": ft_in,
                    "squad": ",".join(str(e) for e in sorted(st.squad)),
                    "status": status,
                    "solve_s": round(solve_s, 2),
                    "time_limited": limited,
                }
            )
            st.free_transfers = (
                TRANSFERS.free_per_gw
                if t == first
                else min(TRANSFERS.max_bank, ft_in - min(n, ft_in) + TRANSFERS.free_per_gw)
            )
        if log:
            line = ", ".join(f"{r['policy']} {r['net']}" for r in rows[-len(POLICIES) :])
            log(f"{season} GW{t}: {line} (avg {avg:.1f})")
    return pd.DataFrame(rows)


# ============================================================================================ statistics
def max_drawdown(excess: np.ndarray) -> float:
    """Largest peak-to-trough fall of the cumulative sum (0 if it never falls)."""
    cum = np.cumsum(excess)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float(np.max(peak - cum)) if len(cum) else 0.0


def block_bootstrap_mean(
    x: np.ndarray, block: int = BLOCK, resamples: int = RESAMPLES, seed: int = 0
) -> np.ndarray:
    """Moving-block bootstrap (circular blocks of length `block`) of the mean of x: `resamples` means."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return np.array([np.nan])
    rng = np.random.default_rng(seed)
    b = max(1, min(block, n))
    k = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=(resamples, k))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % n
    sample = x[idx.reshape(resamples, -1)[:, :n]]
    return sample.mean(axis=1)


def summarise(
    rows: pd.DataFrame, block: int = BLOCK, resamples: int = RESAMPLES
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per season x policy summary, paired MODEL - BASELINE per season and pooled with bootstrap CIs)."""
    out = []
    for (season, policy), g in rows.groupby(["season", "policy"], sort=True):
        g = g.sort_values("gw")
        excess = (g["net"] - g["gw_average"]).to_numpy(float)
        out.append(
            {
                "season": season,
                "policy": policy,
                "gws": len(g),
                "net_points": int(g["net"].sum()),
                "gross_points": int(g["points"].sum()),
                "transfers": int(g["transfers"].sum()),
                "hits": int(g["hits"].sum()),
                "hit_points": int(4 * g["hits"].sum()),
                "gw_average_total": float(g["gw_average"].sum()),
                "net_minus_average": float(excess.sum()),
                "max_drawdown_vs_average": max_drawdown(excess),
                "fallback_plans": int((g["status"] == "fallback").sum()),
                "time_limited_plans": int(g["time_limited"].sum()) if "time_limited" in g else 0,
            }
        )
    summary = pd.DataFrame(out)
    paired = []
    boots = []
    for i, (season, g) in enumerate(rows.groupby("season", sort=True)):
        w = g.pivot(index="gw", columns="policy", values="net").sort_index()
        d = (w["model"] - w["baseline"]).to_numpy(float)
        bs = block_bootstrap_mean(d, block, resamples, seed=i)
        boots.append((len(d), bs))
        paired.append(_paired_row(season, d, bs))
    if len(boots) > 1:
        n_total = sum(n for n, _ in boots)
        pooled = sum(n * bs for n, bs in boots) / n_total
        d_all = np.concatenate(
            [
                (g.pivot(index="gw", columns="policy", values="net").sort_index())
                .pipe(lambda w: w["model"] - w["baseline"])
                .to_numpy(float)
                for _, g in rows.groupby("season", sort=True)
            ]
        )
        paired.append(_paired_row("all", d_all, pooled))
    return summary, pd.DataFrame(paired)


def _paired_row(season: str, d: np.ndarray, bs: np.ndarray) -> dict[str, Any]:
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {
        "season": season,
        "gws": len(d),
        "mean_diff_per_gw": float(np.mean(d)) if len(d) else float("nan"),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "total_diff": float(np.sum(d)),
        "total_ci_low": float(lo * len(d)),
        "total_ci_high": float(hi * len(d)),
        "gws_model_ahead": int((d > 0).sum()),
        "gws_baseline_ahead": int((d < 0).sum()),
    }


def default_predictors() -> dict[str, Predictor]:
    return {"model": PREDICTORS["model"](), "baseline": PREDICTORS["baseline"]()}


def run(
    df: pd.DataFrame,
    seasons: list[str],
    predictors: dict[str, Predictor] | None = None,
    log: Callable[[str], None] | None = None,
    **kw: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(per-GW rows, per season x policy summary, paired MODEL - BASELINE) for `seasons`."""
    predictors = predictors or default_predictors()
    rows = pd.concat([run_season(df, s, predictors, log=log, **kw) for s in seasons], ignore_index=True)
    summary, paired = summarise(rows)
    return rows, summary, paired
