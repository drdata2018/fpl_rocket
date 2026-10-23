"""The legal planner: XI, captaincy, transfers, free-transfer banking, hits and chips as one integer
program (PuLP + CBC). docs/PLAN_v2.md section 4.

Objectives (PlanSettings.objective):
    ONE_GW_NO_TRANSFERS    XI + captain + vice from the current squad for the next GW (bench ordered by xP
                           outside the program). The only plan that may be shown publicly.
    ONE_GW_WITH_TRANSFERS  the same plus up to `max_transfers` transfers, hits allowed.
    MULTI_GW               horizon H (default 5): squad/XI/captain/transfers per GW, FT banking (cap from the
                           rules), 4-point hits, decay weights w_i = decay**i (0.7 / 0.85 / 1.0).
Chips (PlanSettings.chip): "wildcard" / "freehit" in the FIRST GW of the horizon, "bboost" / "3xc" in
PlanSettings.chip_gw (chip_scenarios picks the GW where the chip adds most to the plan without chips, then
re-plans with the chip there). chip_scenarios() solves with and without each
available chip and reports the conditional gain inside the evaluated window (no recommendation).

Legality (every GW of the plan): squad of 15 with 2/5/5/3 per position, at most 3 per club (a squad that
already breaks the club limit because players changed clubs must be restored by the plan's transfers, or the
plan is infeasible with the reason), XI formation from the rules (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD),
exactly one captain and one vice in the XI, bank >= 0 with outgoing players at their SELLING price and
incoming players at now_cost. Hits = transfers beyond the free balance x hit_cost; the free balance
evolves as f' = min(cap, f - used + 1); in a wildcard / free-hit week transfers are free and the balance
follows the rules' chip_week_policy.

Players flagged low_history (unreliable predictions) are never captain or vice and never transfer-in
candidates; owned ones may stay in the XI (the plan says so).

Solver: HiGHS through PuLP (in-process; CBC selectable, see make_solver), 1 thread, started from the hold
plan. Time limit = PlanSettings.time_limit: TIME_LIMIT_S (10 s) for interactive app solves, BATCH_TIME_LIMIT_S
(60 s) for the registry writer and the matched backtest (a 10 s limit under machine load produced fallbacks
that moved a whole backtest season by ~90 points). HiGHS keeps ONE process-wide task scheduler: once anything
in the process has run HiGHS with another thread count, every later threads=1 run fails at once. So the
scheduler is reset before every HiGHS run and runs are serialised (one at a time per process). A plan is used
when proven optimal, or when the time limit is reached with a plan within ACCEPT_GAP (1%) of the proven bound
(the plan says so; Plan.time_limited). Infeasible, or the time limit with a larger gap, returns the FALLBACK
plan = hold the squad + best XI by xP, with the reason. A solver that fails to run returns
status "solver_error" (never a silent hold): the squad is held for display only and the reason says why.
Deterministic given inputs (sorted candidates and variables, single thread, fixed seed), except
that a time-limited solve can depend on machine speed.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

import pulp

from ..rules.season_2026_27 import DEF, FWD, GK, MID, POSITIONS, SQUAD, TRANSFERS, SquadRules, TransferRules
from .lineup import FORMATION, XI_SIZE, bench_order, best_xi, captain_pair, sort_xi

ONE_GW_NO_TRANSFERS = "one_gw_no_transfers"
ONE_GW_WITH_TRANSFERS = "one_gw_with_transfers"
MULTI_GW = "multi_gw"
OBJECTIVES = (ONE_GW_NO_TRANSFERS, ONE_GW_WITH_TRANSFERS, MULTI_GW)
OBJECTIVE_LABELS = {
    ONE_GW_NO_TRANSFERS: "Next GW, no transfers",
    ONE_GW_WITH_TRANSFERS: "Next GW, with transfers",
    MULTI_GW: "Multi-GW horizon",
}
DECAY_OPTIONS = (0.7, 0.85, 1.0)
DEFAULT_DECAY = 0.85
DEFAULT_HORIZON = 5
TIME_LIMIT_S = 10.0  # interactive (app) solves
BATCH_TIME_LIMIT_S = 60.0  # registry_write.py and the matched backtest (scripts/run_matched.py)
CHIPS = ("wildcard", "freehit", "bboost", "3xc")
CHIP_LABELS = {
    "wildcard": "Wildcard",
    "freehit": "Free Hit",
    "bboost": "Bench Boost",
    "3xc": "Triple Captain",
}
FIRST_GW_CHIPS = ("wildcard", "freehit")
VICE_WEIGHT = 1e-3  # picks the best non-captain as vice without moving the XI
# one HiGHS run at a time per process: the scheduler reset before each run is process-wide (see the docstring)
MAX_CONCURRENT_SOLVES = 1
# a plan found at the time limit is accepted when within this relative gap of the proven bound
ACCEPT_GAP = 0.01
# stopping rule: a plan proven within 0.5% of the bound is final (a 5-GW objective of ~190 points: < 1 point);
# makes multi-GW solves finish well inside the time limit, and deterministic
MIP_GAP = 0.005
SOLVER = "highs"  # "highs" (default) | "cbc"; see make_solver
_SOLVER_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_SOLVES)


# ============================================================================================ inputs
@dataclass(frozen=True)
class Owned:
    """A squad player with the price the manager would get for him."""

    element: int
    element_type: int
    team: int
    selling_price: int  # tenths


@dataclass(frozen=True)
class Candidate:
    """A player that can be bought (price = now_cost)."""

    element: int
    element_type: int
    team: int
    buy: int  # tenths


@dataclass
class PlannerInput:
    squad: list[Owned]
    bank: int  # tenths
    free_transfers: int  # entering the first GW (the LOW end of the planning state's range)
    gws: list[int]  # GWs with predictions, ascending; the first is the next deadline
    xp: dict[tuple[int, int], float]  # (element, gw) -> mean xP
    pool: dict[int, Candidate]  # buyable players (incl. owned ones: their now_cost)
    low_history: frozenset[int] = frozenset()
    chips_available: list[dict[str, Any]] = field(default_factory=list)  # name, start_event, stop_event
    names: dict[int, str] = field(default_factory=dict)
    squad_rules: SquadRules = SQUAD
    transfer_rules: TransferRules = TRANSFERS
    formation: dict[int, tuple[int, int]] = field(default_factory=lambda: dict(FORMATION))
    notes: list[str] = field(default_factory=list)  # e.g. "user-edited: bank 1.5 (reconstructed 0.5)"

    def name(self, e: int) -> str:
        return self.names.get(e, str(e))

    def element_type(self) -> dict[int, int]:
        out = {c.element: c.element_type for c in self.pool.values()}
        out.update({o.element: o.element_type for o in self.squad})
        return out

    def team(self) -> dict[int, int]:
        out = {c.element: c.team for c in self.pool.values()}
        out.update({o.element: o.team for o in self.squad})
        return out

    def chip_gws(self, chip: str, gws: list[int]) -> list[int]:
        """GWs of `gws` inside an available window of `chip`."""
        wins = [
            (int(c["start_event"]), int(c["stop_event"])) for c in self.chips_available if c["name"] == chip
        ]
        return [g for g in gws if any(a <= g <= b for a, b in wins)]

    def chip_window(self, chip: str, gws: list[int]) -> str:
        wins = [
            f"GW{c['start_event']}-GW{c['stop_event']}"
            for c in self.chips_available
            if c["name"] == chip and int(c["stop_event"]) >= gws[0] and int(c["start_event"]) <= gws[-1]
        ]
        return ", ".join(wins) or "-"


@dataclass(frozen=True)
class PlanSettings:
    objective: str = MULTI_GW
    horizon: int = DEFAULT_HORIZON
    decay: float = DEFAULT_DECAY
    max_transfers: int | None = 2  # per GW outside chip weeks; None = no limit
    allow_hits: bool = True
    bench_weight: float = 0.1  # value of bench xP in the objective (squad depth)
    chip: str | None = None
    chip_gw: int | None = None  # bench boost / triple captain: the GW (None = first allowed GW)
    time_limit: float = TIME_LIMIT_S
    pool_top: tuple[tuple[int, int], ...] = ((GK, 6), (DEF, 18), (MID, 18), (FWD, 12))
    pool_cheapest: int = 3
    solver: str = SOLVER
    mip_gap: float = MIP_GAP  # the solver stops once a plan is proven within this relative gap

    def horizon_gws(self, inp: PlannerInput) -> list[int]:
        n = 1 if self.objective in (ONE_GW_NO_TRANSFERS, ONE_GW_WITH_TRANSFERS) else max(1, self.horizon)
        return list(inp.gws[:n])

    @property
    def transfers_on(self) -> bool:
        return self.objective != ONE_GW_NO_TRANSFERS

    def describe(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "horizon": self.horizon if self.objective == MULTI_GW else 1,
            "decay": self.decay,
            "time_limit": self.time_limit,
            "max_transfers": self.max_transfers if self.transfers_on else 0,
            "allow_hits": self.allow_hits and self.transfers_on,
            "chip": self.chip,
            "chip_gw": self.chip_gw,
        }


# ============================================================================================ outputs
@dataclass
class GWPlan:
    gw: int
    squad: list[int]
    xi: list[int]
    captain: int
    vice: int
    bench: list[int]  # ordered: reserve GK first, then outfield by xP
    transfers_out: list[int]
    transfers_in: list[int]
    free_transfers: int  # entering this GW
    hits: int
    hit_cost: int
    bank_after: int  # tenths
    chip: str | None
    expected_points: float  # XI + armband (+ bench under bench boost) - hit cost


@dataclass
class Plan:
    objective: str
    settings: dict[str, Any]
    status: str  # "optimal" | "fallback" | "solver_error"
    reason: str
    gws: list[GWPlan]
    objective_value: float
    solve_seconds: float
    explanation: list[str]
    warnings: list[str] = field(default_factory=list)
    time_limited: bool = False  # the solver stopped at its time limit (accepted within the gap, or fallback)

    @property
    def expected_total(self) -> float:
        return float(sum(g.expected_points for g in self.gws))

    @property
    def hits_total(self) -> int:
        return int(sum(g.hits for g in self.gws))

    @property
    def transfers_total(self) -> int:
        return int(sum(len(g.transfers_in) for g in self.gws))

    def first(self) -> GWPlan:
        return self.gws[0]


@dataclass
class ChipScenario:
    chip: str
    window: str  # the chip's window(s) overlapping the evaluated GWs
    evaluated: str  # "GW9-GW13"
    gw: int | None  # where the chip is played in the scenario
    base_points: float  # expected points of the plan without the chip (net of hits, undiscounted)
    chip_points: float  # same, with the chip
    gain: float  # conditional gain inside the evaluated window
    status: str  # "solved" | "not available" | "fallback" | "solver_error"
    reason: str = ""
    plan: Plan | None = None


# ============================================================================================ helpers
def _fmt(v: float) -> str:
    return f"{v:.1f}"


def _price(t: int) -> str:
    return f"{t / 10:.1f}"


def club_excess(inp: PlannerInput) -> dict[int, int]:
    """{team: players above the club limit} of the current squad."""
    counts = Counter(o.team for o in inp.squad)
    limit = inp.squad_rules.max_per_team
    return {t: n - limit for t, n in counts.items() if n > limit}


def composition_problems(inp: PlannerInput) -> list[str]:
    counts = Counter(o.element_type for o in inp.squad)
    out = []
    if len(inp.squad) != inp.squad_rules.size:
        out.append(f"squad has {len(inp.squad)} players, expected {inp.squad_rules.size}")
    for et, need in inp.squad_rules.per_position.items():
        if counts.get(et, 0) != need:
            out.append(f"{counts.get(et, 0)} {POSITIONS[et]} in the squad, expected {need}")
    return out


def candidates(inp: PlannerInput, settings: PlanSettings, gws: list[int]) -> list[int]:
    """Owned players + the best `pool_top` per position by decayed horizon xP + the `pool_cheapest` cheapest
    per position (budget fillers). low_history players are never candidates to buy."""
    owned = {o.element for o in inp.squad}
    if not settings.transfers_on:
        return sorted(owned)
    weights = {g: settings.decay**i for i, g in enumerate(gws)}
    score = {
        e: sum(w * inp.xp.get((e, g), 0.0) for g, w in weights.items())
        for e in inp.pool
        if e not in inp.low_history
    }
    chosen = set(owned)
    tops = dict(settings.pool_top)
    for et in (GK, DEF, MID, FWD):
        pos = [e for e in score if inp.pool[e].element_type == et]
        pos.sort(key=lambda e: (-score[e], e))
        chosen.update(pos[: tops.get(et, 20)])
        cheap = sorted(pos, key=lambda e: (inp.pool[e].buy, -score[e], e))
        chosen.update(cheap[: settings.pool_cheapest])
    return sorted(chosen)


def make_solver(name: str, time_limit: float, warm: bool = False, gap: float | None = None) -> pulp.LpSolver:
    """HiGHS (in-process, highspy) by default; the bundled CBC binary on request.

    CBC 2.10.3 (the binary bundled with PuLP) was found to hang indefinitely, ignoring its time limit, on
    3-5 GW plans that HiGHS solves in about a second (2026-10-23, same MPS file), so HiGHS is the default;
    its time limit is enforced in-process."""
    if name == "cbc":
        return pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=1, warmStart=warm, gapRel=gap)
    solver = _HiGHSWarm(msg=False, timeLimit=time_limit, threads=1, gapRel=gap, random_seed=0)
    solver.warm = warm
    return solver


class _HiGHSWarm(pulp.HiGHS):  # type: ignore[misc, valid-type]
    """PuLP's in-process HiGHS, optionally started from the variables' current values (a feasible plan, e.g.
    hold + best XI): the solver then always has an incumbent to improve on."""

    warm = False

    run_status: Any = None

    def callSolver(self, lp: pulp.LpProblem) -> None:  # noqa: N802 - PuLP's name
        import highspy

        if self.warm:
            start = highspy.HighsSolution()
            start.col_value = [float(v.varValue or 0.0) for v in lp.variables()]
            start.value_valid = True
            lp.solverModel.setSolution(start)
        # HiGHS's task scheduler is process-wide: a run with another thread count anywhere in the process
        # (another library, a default-threads solve) makes every later threads=1 run fail immediately.
        reset_highs_scheduler()
        self.run_status = lp.solverModel.run()


def reset_highs_scheduler() -> None:
    """Reset HiGHS's process-wide task scheduler (callers hold _SOLVER_SLOTS: no other run is active)."""
    import highspy

    highspy.Highs.resetGlobalScheduler(True)


def _highs_failed(solver: pulp.LpSolver) -> str | None:
    """Why the last HiGHS run failed (None when it ran)."""
    status = getattr(solver, "run_status", None)
    if status is None:
        return None
    import highspy

    return None if status != highspy.HighsStatus.kError else f"HiGHS run returned {status}"


def _solve_status(
    prob: pulp.LpProblem,
    time_limit: float,
    solver: str = SOLVER,
    warm: bool = False,
    gap: float | None = None,
) -> tuple[str, str]:
    """Run the solver (one at a time per process). ('optimal', '') or (kind, human reason); kind is one of
    optimal | infeasible | timeout | solver_error | error."""
    s = make_solver(solver, time_limit, warm, gap)
    try:
        with _SOLVER_SLOTS:
            prob.solve(s)
    except Exception as exc:  # noqa: BLE001 - surfaced as an explicit solver error, never a silent hold
        return "solver_error", f"the solver failed to run ({type(exc).__name__}: {exc})"
    failed = _highs_failed(s)
    if failed:
        return "solver_error", f"the solver failed to run ({failed})"
    if prob.status == pulp.LpStatusOptimal and prob.sol_status == pulp.LpSolutionOptimal:
        g = mip_gap(prob)
        return "optimal", "" if not g or g < 1e-4 else f"plan proven within {g:.2%} of the bound"
    if prob.status == pulp.LpStatusInfeasible or prob.sol_status == pulp.LpSolutionInfeasible:
        return "infeasible", "no legal, affordable plan satisfies the constraints"
    if prob.sol_status == pulp.LpSolutionIntegerFeasible:
        gap = mip_gap(prob)
        if gap is not None and gap <= ACCEPT_GAP:
            return "optimal", f"{TIME_LIMIT_REACHED}; best plan found is within {gap:.2%} of the proven bound"
        txt = "" if gap is None else f" (best plan found was {gap:.1%} from the bound)"
        return "timeout", f"the solver reached its {time_limit:g} s time limit before proving a plan{txt}"
    if prob.status == pulp.LpStatusNotSolved:
        return "timeout", f"the solver reached its {time_limit:g} s time limit without a plan"
    return "error", f"solver status {pulp.LpStatus.get(prob.status, prob.status)}"


def mip_gap(prob: pulp.LpProblem) -> float | None:
    """Relative MIP gap of the last HiGHS solve (None for other solvers)."""
    try:
        return float(prob.solverModel.getInfo().mip_gap)
    except Exception:  # noqa: BLE001 - CBC or no solver model
        return None


TIME_LIMIT_REACHED = "time limit reached"

# module-level hook so tests can simulate a solver timeout without a large problem
SOLVE = _solve_status


# ============================================================================================ fallback
def hold_plan(inp: PlannerInput, settings: PlanSettings, gws: list[int], reason: str, t0: float) -> Plan:
    """Hold the squad, best XI by xP each GW (no solver)."""
    et = inp.element_type()
    squad = [o.element for o in inp.squad]
    plans = []
    ft = min(inp.free_transfers, inp.transfer_rules.max_bank)
    for g in gws:
        score = {e: inp.xp.get((e, g), 0.0) for e in squad}
        xi = best_xi(squad, et, score, inp.formation)
        cap, vice = captain_pair(xi, score, inp.low_history)
        bench = bench_order([e for e in squad if e not in xi], et, score)
        pts = sum(score[e] for e in xi) + score[cap]
        plans.append(GWPlan(g, sorted(squad), xi, cap, vice, bench, [], [], ft, 0, 0, inp.bank, None, pts))
        ft = min(inp.transfer_rules.max_bank, ft + inp.transfer_rules.free_per_gw)
    plan = Plan(
        settings.objective,
        settings.describe(),
        "fallback",
        reason,
        plans,
        float(sum(p.expected_points * settings.decay**i for i, p in enumerate(plans))),
        time.perf_counter() - t0,
        [],
    )
    plan.explanation = [f"Fallback plan (hold the squad + best XI by xP): {reason}.", *explain(inp, plan)]
    plan.warnings = _warnings(inp, plan)
    return plan


# ============================================================================================ solve
def _precheck(inp: PlannerInput, settings: PlanSettings, gws: list[int]) -> str | None:
    problems = composition_problems(inp)
    if problems and not settings.transfers_on:
        return "; ".join(problems)
    chip = settings.chip
    if chip is not None:
        if chip not in CHIPS:
            return f"unknown chip {chip!r}"
        if not settings.transfers_on and chip in FIRST_GW_CHIPS:
            return f"{CHIP_LABELS[chip]} needs a plan with transfers"
        allowed = inp.chip_gws(chip, gws[:1] if chip in FIRST_GW_CHIPS else gws)
        if not allowed:
            span = f"GW{gws[0]}" if chip in FIRST_GW_CHIPS or len(gws) == 1 else f"GW{gws[0]}-GW{gws[-1]}"
            return f"{CHIP_LABELS[chip]} is not available in {span}"
    excess = club_excess(inp)
    if excess and settings.transfers_on and chip not in FIRST_GW_CHIPS:
        need = sum(excess.values())
        free = min(inp.free_transfers, inp.transfer_rules.max_bank)
        cap = settings.max_transfers if settings.allow_hits else free
        if settings.max_transfers is not None:
            cap = min(cap, settings.max_transfers)
        if cap is not None and need > cap:
            clubs = ", ".join(f"team {t} (+{n})" for t, n in sorted(excess.items()))
            return (
                f"the squad breaks the {inp.squad_rules.max_per_team}-per-club limit ({clubs}); restoring it "
                f"needs at least {need} transfer(s) in GW{gws[0]} but this plan allows {cap}"
            )
    return None


def solve(inp: PlannerInput, settings: PlanSettings | None = None) -> Plan:
    """Plan for the next GW(s) under `settings`; never raises for an infeasible state (fallback plan)."""
    settings = settings or PlanSettings()
    t0 = time.perf_counter()
    gws = settings.horizon_gws(inp)
    if not gws:
        raise ValueError("no gameweeks with predictions to plan")
    problem = _precheck(inp, settings, gws)
    if problem:
        return hold_plan(inp, settings, gws, problem, t0)
    m = _Model(inp, settings, gws)
    warm = m.warm_start()
    gap = settings.mip_gap if settings.objective == MULTI_GW else None
    kind, reason = SOLVE(m.prob, settings.time_limit, settings.solver, warm, gap)
    if kind == "solver_error":
        plan = hold_plan(inp, settings, gws, reason, t0)
        plan.status = "solver_error"
        plan.explanation = [
            f"SOLVER ERROR, no plan was solved: {reason}. The squad is shown held for display only.",
            *plan.explanation[1:],
        ]
        return plan
    if kind != "optimal":
        if kind == "infeasible" and club_excess(inp):
            reason += " (the squad breaks the club limit and must be restored)"
        plan = hold_plan(inp, settings, gws, reason, t0)
        plan.time_limited = kind == "timeout"
        return plan
    plan = m.extract(time.perf_counter() - t0)
    plan.reason = reason
    plan.time_limited = reason.startswith(TIME_LIMIT_REACHED)
    plan.explanation = explain(inp, plan) + ([f"Solver: {reason}."] if reason else [])
    plan.warnings = _warnings(inp, plan)
    return plan


class _Model:
    """The integer program for one (input, settings): built in sorted order (deterministic)."""

    def __init__(self, inp: PlannerInput, settings: PlanSettings, gws: list[int]) -> None:
        self.inp, self.settings, self.gws = inp, settings, gws
        self.owned = {o.element: o for o in inp.squad}
        self.P = candidates(inp, settings, gws)
        self.et, self.team = inp.element_type(), inp.team()
        chip = settings.chip
        self.fh_gw = gws[0] if chip == "freehit" else None
        self.wc_gw = gws[0] if chip == "wildcard" else None
        self.tc_gw = self._chip_gw("3xc") if chip == "3xc" else None
        self.bb_gw = self._chip_gw("bboost") if chip == "bboost" else None
        self.w = {g: settings.decay**i for i, g in enumerate(gws)}
        self.xpv = {(p, g): float(inp.xp.get((p, g), 0.0)) for p in self.P for g in gws}
        self.s: dict[tuple[int, int], Any] = {}
        self.x: dict[tuple[int, int], Any] = {}
        self.c: dict[tuple[int, int], Any] = {}
        self.v: dict[tuple[int, int], Any] = {}
        self.tin: dict[tuple[int, int], Any] = {}
        self.tout: dict[tuple[int, int], Any] = {}
        self.bank: dict[int, Any] = {}
        self.h: dict[int, Any] = {}
        self.u: dict[int, Any] = {}
        self.f: dict[int, Any] = {}
        self.n: dict[int, Any] = {}
        self.prob = pulp.LpProblem("fpl_plan", pulp.LpMaximize)
        self.obj: list[Any] = []
        self.captainable = [p for p in self.P if p not in inp.low_history]
        prev_s: dict[int, Any] = {p: int(p in self.owned) for p in self.P}
        prev_bank: Any = inp.bank
        for g in gws:
            prev_s, prev_bank = self._squad(g, prev_s, prev_bank)
            self._lineup(g)
        if settings.transfers_on:
            self._free_transfers()
        self.prob += pulp.lpSum(self.obj)

    def _chip_gw(self, chip: str) -> int | None:
        allowed = self.inp.chip_gws(chip, self.gws)
        want = self.settings.chip_gw
        if want is not None:
            return want if want in allowed else None
        return allowed[0] if allowed else None

    # ------------------------------------------------------------------ squad flow, budget, composition
    def _squad(self, g: int, prev_s: dict[int, Any], prev_bank: Any) -> tuple[dict[int, Any], Any]:
        inp, prob, P = self.inp, self.prob, self.P
        B = pulp.LpBinary
        if not self.settings.transfers_on:
            for p in P:
                self.s[p, g] = 1
            return prev_s, prev_bank
        buy = {p: inp.pool[p].buy for p in P if p in inp.pool}
        if g == self.fh_gw:  # free hit: any legal squad within the selling value; reverts afterwards
            price = {p: self.owned[p].selling_price if p in self.owned else buy.get(p) for p in P}
            for p in P:
                self.s[p, g] = self.prob.add_variable(f"s_{p}_{g}", cat=B) if price[p] is not None else 0
            total = inp.bank + sum(o.selling_price for o in inp.squad)
            prob += pulp.lpSum(price[p] * self.s[p, g] for p in P if price[p] is not None) <= total
            self.n[g] = 0
            self._composition(g)
            return prev_s, prev_bank
        sell = {p: self.owned[p].selling_price if p in self.owned else buy.get(p, 0) for p in P}
        for p in P:
            can_buy = p in buy and p not in inp.low_history
            self.tin[p, g] = self.prob.add_variable(f"in_{p}_{g}", cat=B) if can_buy else 0
            self.tout[p, g] = self.prob.add_variable(f"out_{p}_{g}", cat=B)
            self.s[p, g] = self.prob.add_variable(f"s_{p}_{g}", cat=B)
            prob += self.s[p, g] == prev_s[p] + self.tin[p, g] - self.tout[p, g]
            # valid tightening of the LP relaxation: sell only an owned player, buy only a non-owned one
            prob += self.tout[p, g] <= prev_s[p]
            if can_buy:
                prob += self.tin[p, g] <= 1 - prev_s[p]
        self.bank[g] = self.prob.add_variable(f"bank_{g}", lowBound=0)
        prob += self.bank[g] == (
            prev_bank
            + pulp.lpSum(sell[p] * self.tout[p, g] for p in P)
            - pulp.lpSum(buy[p] * self.tin[p, g] for p in P if p in buy)
        )
        self.n[g] = pulp.lpSum(self.tin[p, g] for p in P)
        self._composition(g)
        return {p: self.s[p, g] for p in P}, self.bank[g]

    def _composition(self, g: int) -> None:
        rules = self.inp.squad_rules
        for et, need in rules.per_position.items():
            self.prob += pulp.lpSum(self.s[p, g] for p in self.P if self.et[p] == et) == need
        for t in sorted({self.team[p] for p in self.P}):
            members = [p for p in self.P if self.team[p] == t]
            if len(members) > rules.max_per_team:
                self.prob += pulp.lpSum(self.s[p, g] for p in members) <= rules.max_per_team

    # ------------------------------------------------------------------ XI, captain, vice
    def _lineup(self, g: int) -> None:
        prob, P, xp, w = self.prob, self.P, self.xpv, self.w[g]
        for p in P:
            self.x[p, g] = self.prob.add_variable(f"x_{p}_{g}", cat=pulp.LpBinary)
            prob += self.x[p, g] <= self.s[p, g]
        prob += pulp.lpSum(self.x[p, g] for p in P) == XI_SIZE
        for et, (lo, hi) in self.inp.formation.items():
            members = [self.x[p, g] for p in P if self.et[p] == et]
            prob += pulp.lpSum(members) >= lo
            prob += pulp.lpSum(members) <= hi
        # the vice is a decision variable of the GW being planned now; later GWs get the best non-captain
        # outside the program (it does not change their expected points)
        with_vice = g == self.gws[0]
        for p in self.captainable:
            self.c[p, g] = self.prob.add_variable(f"c_{p}_{g}", cat=pulp.LpBinary)
            if with_vice:
                self.v[p, g] = self.prob.add_variable(f"v_{p}_{g}", cat=pulp.LpBinary)
                prob += self.c[p, g] + self.v[p, g] <= self.x[p, g]
            else:
                prob += self.c[p, g] <= self.x[p, g]
        prob += pulp.lpSum(self.c[p, g] for p in self.captainable) == 1
        if with_vice:
            prob += pulp.lpSum(self.v[p, g] for p in self.captainable) == 1
        # bench boost week: all 15 score (x weight 0, s weight 1); triple captain week: armband x3
        bw = 1.0 if g == self.bb_gw else self.settings.bench_weight
        armband = 2.0 if g == self.tc_gw else 1.0
        self.obj.append(
            w * pulp.lpSum(xp[p, g] * ((1 - bw) * self.x[p, g] + bw * self.s[p, g]) for p in P)
            + w * armband * pulp.lpSum(xp[p, g] * self.c[p, g] for p in self.captainable)
            + w * VICE_WEIGHT * pulp.lpSum(xp[p, g] * v for (p, gg), v in self.v.items() if gg == g)
        )

    # ------------------------------------------------------------------ free transfers and hits
    def _free_transfers(self) -> None:
        rules, prob, gws = self.inp.transfer_rules, self.prob, self.gws
        cap = rules.max_bank
        for g in gws:
            self.f[g] = self.prob.add_variable(f"ft_{g}", lowBound=0, upBound=cap, cat=pulp.LpInteger)
        prob += self.f[gws[0]] == min(max(0, self.inp.free_transfers), cap)
        for i, g in enumerate(gws):
            chip_week = g in (self.fh_gw, self.wc_gw)
            if chip_week:
                self.u[g], self.h[g] = 0, 0
            else:
                self.u[g] = self.prob.add_variable(f"used_{g}", lowBound=0, upBound=cap, cat=pulp.LpInteger)
                self.h[g] = self.prob.add_variable(f"hits_{g}", lowBound=0, cat=pulp.LpInteger)
                prob += self.u[g] <= self.f[g]
                prob += self.h[g] == self.n[g] - self.u[g]
                if not self.settings.allow_hits:
                    prob += self.h[g] == 0
                if self.settings.max_transfers is not None:
                    prob += self.n[g] <= self.settings.max_transfers
                self.obj.append(-self.w[g] * rules.hit_cost * self.h[g])
            if i + 1 < len(gws):
                nxt = self.f[gws[i + 1]]
                if not chip_week:
                    prob += nxt <= self.f[g] - self.u[g] + rules.free_per_gw
                elif rules.chip_week_policy == "carry":
                    prob += nxt <= self.f[g]
                elif rules.chip_week_policy == "carry_accrue":
                    prob += nxt <= self.f[g] + rules.free_per_gw
                else:  # reset_one
                    prob += nxt <= rules.free_per_gw

    # ------------------------------------------------------------------ warm start
    def warm_start(self) -> bool:
        """Set every variable to the hold plan (owned squad, best XI by xP, no transfers): feasible unless the
        squad breaks the club limit or its composition, in which case no start is given (False)."""
        if self.settings.transfers_on and (club_excess(self.inp) or composition_problems(self.inp)):
            return False
        rules = self.inp.transfer_rules
        squad = sorted(self.owned)
        ft = min(max(0, self.inp.free_transfers), rules.max_bank)
        for g in self.gws:
            score = {p: self.xpv[p, g] for p in squad}
            xi = best_xi(squad, self.et, score, self.inp.formation)
            cap, vice = captain_pair(xi, score, self.inp.low_history)
            values = ((self.s, set(squad)), (self.x, set(xi)), (self.c, {cap}), (self.v, {vice}))
            for p in self.P:
                for d, members in values:
                    var = d.get((p, g))
                    if isinstance(var, pulp.LpVariable):
                        var.setInitialValue(int(p in members))
                for d in (self.tin, self.tout):
                    var = d.get((p, g))
                    if isinstance(var, pulp.LpVariable):
                        var.setInitialValue(0)
            if g in self.bank:
                self.bank[g].setInitialValue(self.inp.bank)
            if g in self.f:
                self.f[g].setInitialValue(ft)
                for d in (self.u, self.h):
                    if isinstance(d.get(g), pulp.LpVariable):
                        d[g].setInitialValue(0)
                chip_week = g in (self.fh_gw, self.wc_gw)
                if not chip_week or rules.chip_week_policy == "carry_accrue":
                    ft = min(rules.max_bank, ft + rules.free_per_gw)
                elif rules.chip_week_policy == "reset_one":
                    ft = min(ft, rules.free_per_gw)
        return True

    # ------------------------------------------------------------------ solution
    @staticmethod
    def _val(v: Any) -> float:
        return float(v) if isinstance(v, int | float) else float(pulp.value(v) or 0.0)

    def _chosen(self, d: dict[tuple[int, int], Any], g: int) -> list[int]:
        return sorted(p for (p, gg), v in d.items() if gg == g and self._val(v) > 0.5)

    def extract(self, seconds: float) -> Plan:
        inp, rules = self.inp, self.inp.transfer_rules
        out = []
        # free transfers / hits are replayed from the chosen transfers (the program only bounds the balance)
        ft_state = min(max(0, inp.free_transfers), rules.max_bank)
        for g in self.gws:
            squad = sorted(p for p in self.P if self._val(self.s[p, g]) > 0.5)
            score = {p: self.xpv[p, g] for p in self.P}
            xi = sort_xi([p for p in squad if self._val(self.x[p, g]) > 0.5], self.et, score)
            cap = self._chosen(self.c, g)[0]
            vices = self._chosen(self.v, g)
            vice = vices[0] if vices else captain_pair([p for p in xi if p != cap], score, inp.low_history)[0]
            bench = bench_order([p for p in squad if p not in xi], self.et, score)
            t_in, t_out = self._chosen(self.tin, g), self._chosen(self.tout, g)
            chip_week = g in (self.fh_gw, self.wc_gw)
            ft = ft_state
            hits = 0 if chip_week else max(0, len(t_in) - ft)
            if not chip_week:
                ft_state = min(rules.max_bank, ft - min(len(t_in), ft) + rules.free_per_gw)
            elif rules.chip_week_policy == "carry_accrue":
                ft_state = min(rules.max_bank, ft + rules.free_per_gw)
            elif rules.chip_week_policy == "reset_one":
                ft_state = rules.free_per_gw
            chip, mult = None, 2
            if g == self.fh_gw:
                chip = "freehit"
            elif g == self.wc_gw:
                chip = "wildcard"
            elif g == self.tc_gw:
                chip, mult = "3xc", 3
            elif g == self.bb_gw:
                chip = "bboost"
            pts = sum(score[p] for p in xi) + (mult - 1) * score[cap] - rules.hit_cost * hits
            if chip == "bboost":
                pts += sum(score[p] for p in bench)
            bank = round(self._val(self.bank[g])) if g in self.bank else inp.bank
            out.append(
                GWPlan(
                    gw=g,
                    squad=squad,
                    xi=xi,
                    captain=cap,
                    vice=vice,
                    bench=bench,
                    transfers_out=t_out,
                    transfers_in=t_in,
                    free_transfers=ft,
                    hits=hits,
                    hit_cost=rules.hit_cost * hits,
                    bank_after=bank,
                    chip=chip,
                    expected_points=float(pts),
                )
            )
        return Plan(
            self.settings.objective,
            self.settings.describe(),
            "optimal",
            "",
            out,
            float(pulp.value(self.prob.objective) or 0.0),
            seconds,
            [],
        )


# ============================================================================================ explanation
def _warnings(inp: PlannerInput, plan: Plan) -> list[str]:
    out = []
    first = plan.first()
    for e in first.xi:
        if e in inp.low_history:
            out.append(
                f"{inp.name(e)}: insufficient history - kept in the XI because he is owned; his xP is "
                "unreliable and he is not considered for captain"
            )
    missing = [e for e in first.squad if all((e, g.gw) not in inp.xp for g in plan.gws)]
    if missing:
        out.append("No prediction for " + ", ".join(inp.name(e) for e in missing) + " (counted as 0 xP)")
    return out


def explain(inp: PlannerInput, plan: Plan) -> list[str]:
    """Human-readable reasons: captaincy margin, bench order, transfers, chips, club-limit restoration."""
    lines = list(inp.notes)
    first = plan.first()
    g = first.gw
    et = inp.element_type()
    xp = {e: inp.xp.get((e, g), 0.0) for e in set(first.squad) | set(first.xi)}

    def label(e: int) -> str:
        return "insufficient history" if e in inp.low_history else f"{_fmt(xp.get(e, 0.0))} xP"

    lines.append(
        f"Captain {inp.name(first.captain)}: {label(first.captain)} vs {inp.name(first.vice)} "
        f"{label(first.vice)} (vice) in GW{g}."
    )
    others = [e for e in first.xi if e not in (first.captain, first.vice) and e not in inp.low_history]
    if others:
        nxt = max(others, key=lambda e: (xp.get(e, 0.0), -e))
        lines.append(f"Next best armband option: {inp.name(nxt)} {label(nxt)}.")
    if first.bench:
        order = ", ".join(f"{i + 1}. {inp.name(e)} ({label(e)})" for i, e in enumerate(first.bench))
        lines.append(f"Bench order (by xP, reserve goalkeeper first): {order}.")
        bench_out = [e for e in first.bench if et[e] != GK]
        xi_out = [e for e in first.xi if et[e] != GK]
        if bench_out and xi_out:
            b = max(bench_out, key=lambda e: (xp.get(e, 0.0), -e))
            w = min(xi_out, key=lambda e: (xp.get(e, 0.0), e))
            lines.append(
                f"Lowest outfield starter {inp.name(w)} ({label(w)}) vs best outfield bench player "
                f"{inp.name(b)} ({label(b)}); formation limits decide ties."
            )
    if club_excess(inp) and any(gp.transfers_out for gp in plan.gws):
        lines.append(
            "The squad broke the club limit (players changed clubs): the plan's transfers restore it."
        )
    owned_sell = {o.element: o.selling_price for o in inp.squad}
    for gp in plan.gws:
        if gp.transfers_in or gp.transfers_out:
            outs = ", ".join(
                f"{inp.name(e)} (sell {_price(owned_sell.get(e, inp.pool[e].buy if e in inp.pool else 0))})"
                for e in gp.transfers_out
            )
            ins = ", ".join(f"{inp.name(e)} (buy {_price(inp.pool[e].buy)})" for e in gp.transfers_in)
            hit = f"; {gp.hits} hit(s) = -{gp.hit_cost}" if gp.hits else "; no hit"
            lines.append(
                f"GW{gp.gw} transfers: out {outs or '-'}; in {ins or '-'}{hit}; "
                f"free transfers entering GW{gp.gw}: {gp.free_transfers}; bank after {_price(gp.bank_after)}."
            )
        if gp.chip:
            lines.append(f"GW{gp.gw}: {CHIP_LABELS[gp.chip]} played in this scenario.")
    if len(plan.gws) > 1:
        per = ", ".join(f"GW{gp.gw} {_fmt(gp.expected_points)}" for gp in plan.gws)
        lines.append(
            f"Expected points per GW (net of hits; decay {plan.settings.get('decay')} weights the "
            f"objective): {per}."
        )
    else:
        lines.append(f"Expected points GW{g}: {_fmt(first.expected_points)} (XI + captain, net of hits).")
    return lines


# ============================================================================================ chips
def _chip_value(chip: str, plan: Plan, gw: int, inp: PlannerInput) -> float:
    """Extra expected points of bench boost / triple captain in `gw` on top of `plan` (squads unchanged)."""
    gp = next((p for p in plan.gws if p.gw == gw), None)
    if gp is None:
        return 0.0
    if chip == "bboost":
        return float(sum(inp.xp.get((e, gw), 0.0) for e in gp.bench))
    return float(inp.xp.get((gp.captain, gw), 0.0))


def chip_scenarios(inp: PlannerInput, settings: PlanSettings) -> tuple[Plan, list[ChipScenario]]:
    """The plan without chips and, for each available chip, the plan with it: the conditional gain inside the
    evaluated window (expected points net of hits, undiscounted). Wildcard / free hit are evaluated in the
    first GW only; bench boost / triple captain in the best GW of (window x horizon)."""
    base_settings = replace(settings, chip=None)
    base = solve(inp, base_settings)
    gws = base_settings.horizon_gws(inp)
    evaluated = f"GW{gws[0]}-GW{gws[-1]}" if len(gws) > 1 else f"GW{gws[0]}"
    names = sorted({c["name"] for c in inp.chips_available if c["name"] in CHIPS}, key=CHIPS.index)
    out = []
    for chip in names:
        window = inp.chip_window(chip, gws)
        allowed = inp.chip_gws(chip, gws[:1] if chip in FIRST_GW_CHIPS else gws)
        if not allowed:
            where = f"GW{gws[0]}" if chip in FIRST_GW_CHIPS else evaluated
            out.append(
                ChipScenario(
                    chip,
                    window,
                    evaluated,
                    None,
                    base.expected_total,
                    base.expected_total,
                    0.0,
                    "not available",
                    f"no {CHIP_LABELS[chip]} window covers {where}",
                )
            )
            continue
        chip_gw = None
        if chip not in FIRST_GW_CHIPS:  # the GW where the chip adds most to the plan without chips
            chip_gw = max(allowed, key=lambda g: (_chip_value(chip, base, g, inp), -g))
        plan = solve(inp, replace(settings, chip=chip, chip_gw=chip_gw))
        used = next((gp.gw for gp in plan.gws if gp.chip == chip), None)
        ok = plan.status == "optimal" and base.status == "optimal"
        out.append(
            ChipScenario(
                chip,
                window,
                evaluated,
                used,
                base.expected_total,
                plan.expected_total,
                plan.expected_total - base.expected_total if ok else float("nan"),  # no gain from a fallback
                "solved"
                if ok
                else "solver_error"
                if "solver_error" in (plan.status, base.status)
                else "fallback",
                plan.reason or base.reason,
                plan,
            )
        )
    return base, out
