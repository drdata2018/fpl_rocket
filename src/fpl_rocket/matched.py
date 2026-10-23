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

Options (step 3g): policies=RESTRAINED replays the prospective virtual managers' rule
(fpl_rocket.trajectories: MODEL_RESTRAINED / FORM_RESTRAINED = <= 1 transfer per GW, never a hit, vs HOLD);
start="most_owned" uses their published start rule (the most-owned legal 15 at GW1); fixtures_as_known gives
the planner the schedule as known at each deadline (fpl_rocket.fixtures_known) instead of the as-played one
(scoring is unchanged); season_inputs caches the per-season predictions so such replays reuse the same
refits. summarise reports paired differences for the set's pairs with a moving-block bootstrap within seasons
and a sign test.
"""

from __future__ import annotations

import math
import pickle
import time
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
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
    df: pd.DataFrame,
    season: str,
    predictor: Predictor,
    log: Callable[[str], None] | None = None,
    gws: Iterable[int] | None = None,
) -> dict[int, pd.Series]:
    """{gw: per-fixture xP by element} from rolling-origin folds (fit on everything before the GW); `gws`
    restricts the folds (parallel runs)."""
    out: dict[int, pd.Series] = {}
    only = None if gws is None else {int(g) for g in gws}
    for fold in rolling_origin(df, [season]):
        if only is not None and fold.target_gw not in only:
            continue
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


_WORKER_DF: dict[str, pd.DataFrame] = {}


def _predict_chunk(archive: str, season: str, name: str, gws: list[int]) -> dict[int, pd.Series]:
    """Process-pool worker: per-fixture predictions of predictor `name` for `gws` (archive read once per
    process)."""
    if archive not in _WORKER_DF:
        _WORKER_DF[archive] = pd.read_parquet(archive)
    return per_fixture_predictions(_WORKER_DF[archive], season, PREDICTORS[name](), gws=gws)


def season_inputs(
    df: pd.DataFrame,
    season: str,
    *,
    names: tuple[str, ...] = ("model", "baseline"),
    cache_dir: str | Path | None = None,
    archive: str | Path | None = None,
    workers: int = 1,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """{"preds": {name: {gw: per-fixture xP}}, "lows": {gw: low_history set}} for one season.

    With `cache_dir`, read from / written to <cache_dir>/<season>.pkl (a run's predictions are reused by the
    replays that differ only in policies or fixture knowledge: same refits, same numbers). With `workers` > 1
    and `archive` (the parquet `df` was read from), the per-GW refits run in a process pool (one GW per task;
    each fold is independent, so the predictions are identical to a serial run)."""
    path = Path(cache_dir) / f"{season}.pkl" if cache_dir else None
    if path is not None and path.exists():
        with path.open("rb") as fh:
            cached = pickle.load(fh)  # noqa: S301 - our own cache file
        if set(names) <= set(cached["preds"]):
            if log:
                log(f"{season}: predictions from cache {path}")
            return cached
    gws = sorted(int(g) for g in df.loc[df["season"] == season, "gw"].dropna().unique())
    preds: dict[str, dict[int, pd.Series]] = {}
    for name in names:
        t0 = time.perf_counter()
        if workers > 1 and archive is not None:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_predict_chunk, str(archive), season, name, [g]) for g in gws]
                preds[name] = {}
                for f in futs:
                    preds[name].update(f.result())
        else:
            preds[name] = per_fixture_predictions(df, season, PREDICTORS[name](), log)
        if log:
            log(f"{season}: {name} predictions in {time.perf_counter() - t0:.0f} s")
    out = {"preds": preds, "lows": low_history_sets(df, season)}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(out, fh)
    return out


def horizon_xp(
    data: SeasonData,
    per_fixture: dict[int, pd.Series],
    t: int,
    horizon: int,
    memory: dict[int, float],
    counts: dict[int, dict[int, int]] | None = None,
) -> tuple[list[int], dict[tuple[int, int], float]]:
    """(GWs t..t+H-1 of the season, {(element, gw): xP}) = per-fixture xP at t x fixtures in each GW.
    `memory` carries each player's last per-fixture prediction (updated in place with GW t's).
    `counts` ({gw: {element: fixtures}}): the schedule the planner knows (fixtures as known at the GW t
    deadline); None = as played (the archive's n_fixtures)."""
    pf = per_fixture.get(t)
    if pf is not None:
        memory.update({int(e): float(v) for e, v in pf.items()})
    gws = [g for g in data.gws if g >= t][:horizon]
    xp = {}
    for g in gws:
        nfix = data.per_gw[g]["n_fixtures"].to_dict() if counts is None else counts.get(g, {})
        for e, n in nfix.items():
            if int(e) in memory and int(n) > 0:
                xp[int(e), g] = memory[int(e)] * int(n)
    return gws, xp


# ================================================================================== fixtures as known
def known_counts(
    schedule: pd.DataFrame, gws: list[int], team_of: dict[int, int]
) -> dict[int, dict[int, int]]:
    """{gw: {element: fixtures}} from a schedule as known at a deadline (columns event, team_h, team_a);
    players map to clubs by `team_of` (their club at the deadline)."""
    s = schedule.dropna(subset=["event"])
    per_team: dict[int, Counter] = {}
    for g in gws:
        rows = s[s["event"].astype(int) == g]
        per_team[g] = Counter(rows["team_h"].astype(int)) + Counter(rows["team_a"].astype(int))
    return {g: {e: per_team[g].get(t, 0) for e, t in team_of.items()} for g in gws}


# ============================================================================================ policies
@dataclass(frozen=True)
class PolicySpec:
    """A replayed policy: the predictions it plans with and its transfer rule (hold = never transfers)."""

    name: str
    source: str  # "model" | "baseline"
    max_transfers: int = MAX_TRANSFERS
    allow_hits: bool = True
    hold: bool = False


AGGRESSIVE: tuple[PolicySpec, ...] = (
    PolicySpec("model", "model"),
    PolicySpec("baseline", "baseline"),
    PolicySpec("hold", "baseline", 0, False, hold=True),
)
# the prospective trajectories' restrained rule (fpl_rocket.trajectories): <= 1 transfer per GW, never a hit
RESTRAINED: tuple[PolicySpec, ...] = (
    PolicySpec("model_restrained", "model", 1, False),
    PolicySpec("form_restrained", "baseline", 1, False),
    PolicySpec("hold", "baseline", 0, False, hold=True),
)
POLICY_SETS = {"aggressive": AGGRESSIVE, "restrained": RESTRAINED}
PAIRS = {
    "aggressive": (("model", "baseline"),),
    "restrained": (
        ("model_restrained", "form_restrained"),
        ("model_restrained", "hold"),
        ("form_restrained", "hold"),
    ),
}
STARTS = ("baseline_xp", "most_owned")


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
        pool=_pool(data.per_gw[g][["element_type", "team_id", "value"]]),
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


def most_owned_initial_squad(data: SeasonData) -> tuple[dict[int, int], int]:
    """The prospective trajectories' published start rule (trajectories.START_RULE) at GW1: the most-owned
    legal 15 under 100.0 by the archive's GW1 `selected` (managers owning the player: the same ranking as
    selected_by_percent) and GW1 values, greedy within positions, the club limit and the budget."""
    from .trajectories import most_owned_squad

    g = data.gws[0]
    info = data.per_gw[g]
    sel = data.targets[g].groupby("element")["selected"].max()
    players = pd.DataFrame(
        {
            "element": info.index.astype(int),
            "element_type": info["element_type"].astype(int).to_numpy(),
            "team": info["team_id"].astype(int).to_numpy(),
            "price": info["value"].astype(int).to_numpy(),
            "ownership": sel.reindex(info.index).fillna(0).astype(float).to_numpy(),
        }
    )
    squad = {int(e): int(info.at[e, "value"]) for e in most_owned_squad(players)}
    return squad, SQUAD.budget - sum(squad.values())


def _owned(state: PolicyState, last: dict[int, dict[str, int]]) -> list[Owned]:
    out = []
    for e, buy in sorted(state.squad.items()):
        meta = last[e]
        out.append(Owned(e, meta["element_type"], meta["team_id"], selling_price(buy, meta["value"])))
    return out


def run_season(
    df: pd.DataFrame,
    season: str,
    predictors: dict[str, Predictor] | None = None,
    *,
    policies: tuple[PolicySpec, ...] = AGGRESSIVE,
    start: str = "baseline_xp",
    inputs: dict[str, Any] | None = None,
    fixtures_as_known: dict[int, pd.DataFrame] | None = None,
    horizon: int = HORIZON,
    max_transfers: int | None = None,
    decay: float = DECAY,
    time_limit: float = BATCH_TIME_LIMIT_S,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """One row per (policy, GW) for one season.

    predictors: {"model": ..., "baseline": ...} (refit per GW), or `inputs` = season_inputs() (cached
    predictions and low-history sets). start: "baseline_xp" (the xP-optimal 15 by the baseline's GW1
    prediction) | "most_owned" (the trajectories' published rule). fixtures_as_known: {gw t: the schedule as
    known at the GW t deadline (event, team_h, team_a)}: the planner's fixture counts for GWs t..t+H-1 and the
    GW t pool come from it (a player whose club plays in GW t as known but not as played can be bought at his
    last price and scores 0); scoring always uses the actual results. max_transfers overrides every
    non-hold policy's limit."""
    if max_transfers is not None:
        policies = tuple(
            p if p.hold else PolicySpec(p.name, p.source, max_transfers, p.allow_hits) for p in policies
        )
    data = season_data(df, season)
    if inputs is None:
        predictors = predictors or default_predictors()
        inputs = {
            "preds": {name: per_fixture_predictions(df, season, p, log) for name, p in predictors.items()},
            "lows": low_history_sets(df, season),
        }
    preds, lows = inputs["preds"], inputs["lows"]
    first = data.gws[0]
    if start == "baseline_xp":
        _, xp1 = horizon_xp(data, preds["baseline"], first, 1, {})
        start_squad, bank = initial_squad(data, xp1, lows.get(first, frozenset()))
    elif start == "most_owned":
        start_squad, bank = most_owned_initial_squad(data)
    else:
        raise ValueError(f"unknown start {start!r}; one of {STARTS}")
    if log:
        log(f"{season}: start {start} cost {sum(start_squad.values()) / 10:.1f}, bank {bank / 10:.1f}")
    states = {p.name: PolicyState(dict(start_squad), bank, 1) for p in policies}
    sources = sorted({p.source for p in policies})
    memories: dict[str, dict[int, float]] = {name: {} for name in sources}
    last: dict[int, dict[str, int]] = {}
    rows = []
    for t in data.gws:
        info = data.per_gw[t]
        for e, r in zip(info.index, info.itertuples(index=False), strict=True):
            last[int(e)] = {"element_type": r.element_type, "team_id": r.team_id, "value": r.value}
        counts = None
        pool_frame = info[["element_type", "team_id", "value"]]
        if fixtures_as_known is not None:
            gws_h = [g for g in data.gws if g >= t][:horizon]
            counts = known_counts(fixtures_as_known[t], gws_h, {e: m["team_id"] for e, m in last.items()})
            blank = [e for e, n in counts[t].items() if n > 0 and e not in info.index]
            if blank:
                extra = pd.DataFrame.from_dict({e: last[e] for e in blank}, orient="index")
                pool_frame = pd.concat([pool_frame, extra[["element_type", "team_id", "value"]]])
        price = {int(e): int(v) for e, v in pool_frame["value"].items()}
        horizons = {
            name: horizon_xp(data, preds[name], t, horizon, memories[name], counts) for name in sources
        }
        low = lows.get(t, frozenset())
        avg = gw_average_proxy(data.targets[t])
        pts = info["points"].to_dict()
        mins = info["minutes"].to_dict()
        for spec in policies:
            st = states[spec.name]
            gws, xp = horizons[spec.source]
            et = {e: last[e]["element_type"] for e in st.squad}
            score = {e: xp.get((e, t), 0.0) for e in st.squad}
            t_in: list[int] = []
            t_out: list[int] = []
            status, solve_s, limited = "hold", 0.0, False
            if not spec.hold and t != first:
                inp = PlannerInput(
                    squad=_owned(st, last),
                    bank=st.bank,
                    free_transfers=st.free_transfers,
                    gws=gws,
                    xp=xp,
                    pool=_pool(pool_frame),
                    low_history=low,
                )
                plan = solve(
                    inp,
                    PlanSettings(
                        objective=MULTI_GW,
                        horizon=horizon,
                        decay=decay,
                        max_transfers=spec.max_transfers,
                        allow_hits=spec.allow_hits,
                        time_limit=time_limit,
                    ),
                )
                if plan.status == "solver_error":  # never scored as a silent hold
                    raise RuntimeError(f"{season} GW{t} {spec.name}: {plan.reason}")
                gp = plan.first()
                status, solve_s, limited = plan.status, plan.solve_seconds, plan.time_limited
                t_in, t_out = gp.transfers_in, gp.transfers_out
                sell = {o.element: o.selling_price for o in inp.squad}
                for e in t_out:
                    st.bank += sell[e]
                    del st.squad[e]
                for e in t_in:
                    st.bank -= price[e]
                    st.squad[e] = price[e]
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
                    "policy": spec.name,
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
            line = ", ".join(f"{r['policy']} {r['net']}" for r in rows[-len(policies) :])
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


def sign_test(d: np.ndarray) -> float:
    """Two-sided exact binomial sign test of the GW differences (ties dropped)."""
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    n = pos + neg
    if n == 0:
        return float("nan")
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def default_pairs(rows: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    present = set(rows["policy"])
    for pairs in PAIRS.values():
        if all(a in present and b in present for a, b in pairs):
            return pairs
    return ()


def summarise(
    rows: pd.DataFrame,
    block: int = BLOCK,
    resamples: int = RESAMPLES,
    pairs: tuple[tuple[str, str], ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per season x policy summary, paired A - B per season and pooled with bootstrap CIs and sign tests).

    Pairs default to MODEL - BASELINE (aggressive set) or the three restrained pairs. Bootstrap: circular
    blocks within each season (seed = season index); pooled = the GW-weighted mean of the per-season
    resamples."""
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
    for a, b in pairs if pairs is not None else default_pairs(rows):
        boots, diffs = [], []
        for i, (season, g) in enumerate(rows.groupby("season", sort=True)):
            w = g.pivot(index="gw", columns="policy", values="net").sort_index()
            d = (w[a] - w[b]).to_numpy(float)
            bs = block_bootstrap_mean(d, block, resamples, seed=i)
            boots.append((len(d), bs))
            diffs.append(d)
            paired.append(_paired_row(season, a, b, d, bs))
        if len(boots) > 1:
            n_total = sum(n for n, _ in boots)
            pooled = sum(n * bs for n, bs in boots) / n_total
            paired.append(_paired_row("all", a, b, np.concatenate(diffs), pooled))
    return summary, pd.DataFrame(paired)


def _paired_row(season: str, a: str, b: str, d: np.ndarray, bs: np.ndarray) -> dict[str, Any]:
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {
        "season": season,
        "a": a,
        "b": b,
        "gws": len(d),
        "mean_diff_per_gw": float(np.mean(d)) if len(d) else float("nan"),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "total_diff": float(np.sum(d)),
        "total_ci_low": float(lo * len(d)),
        "total_ci_high": float(hi * len(d)),
        "gws_a_ahead": int((d > 0).sum()),
        "gws_b_ahead": int((d < 0).sum()),
        "sign_test_p": sign_test(d),
    }


def default_predictors() -> dict[str, Predictor]:
    return {"model": PREDICTORS["model"](), "baseline": PREDICTORS["baseline"]()}


def run(
    df: pd.DataFrame,
    seasons: list[str],
    predictors: dict[str, Predictor] | None = None,
    log: Callable[[str], None] | None = None,
    *,
    inputs: dict[str, dict[str, Any]] | None = None,
    fixtures_as_known: dict[str, dict[int, pd.DataFrame]] | None = None,
    block: int = BLOCK,
    resamples: int = RESAMPLES,
    **kw: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(per-GW rows, per season x policy summary, paired differences) for `seasons`. inputs: {season:
    season_inputs()} (cached predictions); fixtures_as_known: {season: {gw: schedule as known}}."""
    parts = []
    for s in seasons:
        season_kw = dict(kw)
        if inputs is not None and s in inputs:
            season_kw["inputs"] = inputs[s]
        elif predictors is not None:
            season_kw["predictors"] = predictors
        if fixtures_as_known is not None:
            season_kw["fixtures_as_known"] = fixtures_as_known[s]
        parts.append(run_season(df, s, log=log, **season_kw))
    rows = pd.concat(parts, ignore_index=True)
    summary, paired = summarise(rows, block, resamples)
    return rows, summary, paired
