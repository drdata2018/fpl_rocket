"""Planner: legality on synthetic leagues (budget via selling prices, 3 per club, formation, captain and vice
in the XI, hits and FT banking), club-limit restoration or an infeasible reason, timeout fallback,
determinism, chip scenarios, low-history exclusions, solver-free line-ups (auto-subs) and the app adapter
with user edits."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace

import pandas as pd
import pulp
import pytest

from fpl_rocket.planner import (
    FORMATION,
    MULTI_GW,
    ONE_GW_NO_TRANSFERS,
    ONE_GW_WITH_TRANSFERS,
    Owned,
    PlanSettings,
    StateEdits,
    auto_subs,
    best_xi,
    chip_scenarios,
    core,
    from_planning_state,
    realised_points,
    solve,
)
from fpl_rocket.state import FreeTransfers, PlanningState, SquadPlayer
from planner_synth import cheap_squad, league, planner_input

ALL_CHIPS = [
    {"name": n, "start_event": 1, "stop_event": 19} for n in ("wildcard", "freehit", "bboost", "3xc")
]


# ------------------------------------------------------------------------------------------- solvers
@pytest.mark.parametrize("solver", ["cbc", "highs"])
def test_bundled_solvers_solve_a_tiny_problem(solver):
    """PuLP's bundled CBC binary (linux/win/osx) and in-process HiGHS both run on this platform."""
    prob = pulp.LpProblem("tiny", pulp.LpMaximize)
    x = prob.add_variable("x", cat=pulp.LpBinary)
    y = prob.add_variable("y", cat=pulp.LpBinary)
    prob += 3 * x + 2 * y
    prob += x + y <= 1
    prob.solve(core.make_solver(solver, 5))
    assert pulp.LpStatus[prob.status] == "Optimal"
    assert (x.value(), y.value()) == (1, 0)


# ------------------------------------------------------------------------------------------- legality
def assert_legal(inp, plan, *, club_limit: bool = True):
    et = inp.element_type()
    team = inp.team()
    sell = {o.element: o.selling_price for o in inp.squad}
    bank = inp.bank
    held = {o.element for o in inp.squad}
    ft = inp.free_transfers
    for gp in plan.gws:
        assert len(gp.squad) == 15 and set(gp.xi) <= set(gp.squad) and len(gp.xi) == 11
        assert Counter(et[e] for e in gp.squad) == Counter({1: 2, 2: 5, 3: 5, 4: 3})
        if club_limit:
            assert max(Counter(team[e] for e in gp.squad).values()) <= 3
        xi_types = Counter(et[e] for e in gp.xi)
        for t, (lo, hi) in FORMATION.items():
            assert lo <= xi_types.get(t, 0) <= hi
        assert gp.captain in gp.xi and gp.vice in gp.xi and gp.captain != gp.vice
        assert sorted(gp.bench) == sorted(set(gp.squad) - set(gp.xi)) and et[gp.bench[0]] == 1
        if gp.chip == "freehit":
            continue
        assert set(gp.transfers_out) <= held and not set(gp.transfers_in) & held
        for e in gp.transfers_out:
            bank += sell.get(e, inp.pool[e].buy)
        for e in gp.transfers_in:
            bank -= inp.pool[e].buy
            sell[e] = inp.pool[e].buy
        assert bank == gp.bank_after and bank >= 0
        held = (held - set(gp.transfers_out)) | set(gp.transfers_in)
        assert held == set(gp.squad)
        n = len(gp.transfers_in)
        assert gp.free_transfers == ft
        if gp.chip == "wildcard":
            assert gp.hits == 0
        else:
            assert gp.hits == max(0, n - ft)
            ft = min(5, ft - min(n, ft) + 1)


@pytest.mark.parametrize("objective", [ONE_GW_NO_TRANSFERS, ONE_GW_WITH_TRANSFERS, MULTI_GW])
def test_plans_are_legal_and_affordable(objective):
    inp = planner_input(bank=20)
    plan = solve(inp, PlanSettings(objective=objective, horizon=3))
    assert plan.status == "optimal", plan.reason
    assert_legal(inp, plan)
    if objective == ONE_GW_NO_TRANSFERS:
        assert plan.transfers_total == 0 and len(plan.gws) == 1
    if objective != ONE_GW_NO_TRANSFERS:
        assert plan.transfers_total >= 1  # the cheapest squad is worth upgrading
    assert plan.explanation and plan.explanation[0].startswith("Captain ")


def test_budget_uses_selling_prices_for_outgoing_players():
    pool, _ = league()
    base = cheap_squad(pool)
    # every owned player sells for 1.0 less than now_cost: upgrades must be paid from selling prices
    squad = [Owned(o.element, o.element_type, o.team, o.selling_price - 10) for o in base]
    inp = planner_input(squad=squad, bank=0)
    plan = solve(inp, PlanSettings(objective=ONE_GW_WITH_TRANSFERS, max_transfers=2))
    assert_legal(inp, plan)
    gp = plan.first()
    spent = sum(inp.pool[e].buy for e in gp.transfers_in)
    got = sum(o.selling_price for o in squad if o.element in gp.transfers_out)
    assert spent <= got and gp.bank_after == got - spent


def test_one_gw_no_transfers_matches_exact_best_xi():
    inp = planner_input()
    plan = solve(inp, PlanSettings(objective=ONE_GW_NO_TRANSFERS))
    squad = [o.element for o in inp.squad]
    score = {e: inp.xp[e, 1] for e in squad}
    best = best_xi(squad, inp.element_type(), score)
    assert sorted(plan.first().xi) == sorted(best)
    xi_scores = sorted((score[e] for e in plan.first().xi), reverse=True)
    assert score[plan.first().captain] == xi_scores[0] and score[plan.first().vice] == xi_scores[1]


def test_hits_taken_only_when_worth_it_and_ft_banked():
    inp = planner_input(free_transfers=1)
    plan = solve(inp, PlanSettings(objective=MULTI_GW, horizon=3, max_transfers=2))
    assert_legal(inp, plan)
    no_hits = solve(inp, PlanSettings(objective=MULTI_GW, horizon=3, max_transfers=2, allow_hits=False))
    assert no_hits.hits_total == 0
    assert plan.objective_value >= no_hits.objective_value - 1e-6
    # banking: with transfers forbidden in GW1 by a zero cap, the balance entering GW2 is 2
    held = solve(replace(inp, free_transfers=1), PlanSettings(objective=MULTI_GW, horizon=2, max_transfers=0))
    assert [g.free_transfers for g in held.gws] == [1, 2]


def test_club_limit_restored_or_infeasible_with_reason():
    pool, _ = league()
    squad = cheap_squad(pool, avoid_clubs=1)
    club1 = [c for c in pool.values() if c.team == 1 and c.element_type == 3][:4]
    mids = [o for o in squad if o.element_type == 3][:4]
    squad = [o for o in squad if o not in mids] + [Owned(c.element, 3, 1, c.buy) for c in club1]
    assert Counter(o.team for o in squad)[1] == 4
    inp = planner_input(squad=squad, bank=0)
    plan = solve(inp, PlanSettings(objective=MULTI_GW, horizon=2, max_transfers=2))
    assert plan.status == "optimal"
    assert_legal(inp, plan)  # 3 per club in EVERY GW, so GW1 already sells one of them
    assert any("club limit" in line for line in plan.explanation)
    # a plan that cannot transfer enough: infeasible with the reason, hold + best XI
    stuck = solve(
        replace(inp, free_transfers=0), PlanSettings(objective=MULTI_GW, horizon=2, allow_hits=False)
    )
    assert stuck.status == "fallback"
    assert "per-club limit" in stuck.reason and "team 1 (+1)" in stuck.reason
    assert stuck.transfers_total == 0
    # holding (objective a) keeps the squad as FPL does
    held = solve(inp, PlanSettings(objective=ONE_GW_NO_TRANSFERS))
    assert held.status == "optimal" and held.transfers_total == 0


def test_timeout_falls_back_to_hold_and_best_xi(monkeypatch):
    monkeypatch.setattr(core, "SOLVE", lambda *a, **k: ("timeout", "the solver reached its 10 s time limit"))
    inp = planner_input()
    plan = solve(inp, PlanSettings(objective=MULTI_GW, horizon=3))
    assert plan.status == "fallback" and "time limit" in plan.reason
    assert plan.transfers_total == 0 and len(plan.gws) == 3
    assert plan.explanation[0].startswith("Fallback plan (hold the squad + best XI by xP)")
    assert_legal(inp, plan)


def test_infeasible_budget_falls_back():
    pool, _ = league()
    squad = cheap_squad(pool)
    inp = planner_input(squad=squad[:14], bank=0)  # 14 players: composition cannot be fielded without a buy
    plan = solve(inp, PlanSettings(objective=ONE_GW_NO_TRANSFERS))
    assert plan.status == "fallback" and "expected 15" in plan.reason


def test_deterministic_given_inputs():
    inp = planner_input()
    settings = PlanSettings(objective=MULTI_GW, horizon=3)
    runs = []
    for _ in range(2):
        p = asdict(solve(inp, settings))
        p.pop("solve_seconds")
        runs.append(p)
    assert runs[0] == runs[1]


def test_low_history_never_captain_nor_bought():
    inp = planner_input()
    owned = [o.element for o in inp.squad]
    score = {e: inp.xp[e, 1] for e in owned}
    star = max(owned, key=lambda e: score[e])
    best_buy = max((e for e in inp.pool if e not in owned), key=lambda e: inp.xp[e, 1])
    inp = replace(inp, low_history=frozenset({star, best_buy}))
    for obj in (ONE_GW_NO_TRANSFERS, ONE_GW_WITH_TRANSFERS, MULTI_GW):
        plan = solve(inp, PlanSettings(objective=obj, horizon=2))
        for gp in plan.gws:
            assert star not in (gp.captain, gp.vice)
            assert best_buy not in gp.transfers_in
    plan = solve(inp, PlanSettings(objective=ONE_GW_NO_TRANSFERS))
    if star in plan.first().xi:
        assert any("insufficient history" in w for w in plan.warnings)


def test_chip_scenarios_report_conditional_gains():
    inp = planner_input(chips=ALL_CHIPS)
    settings = PlanSettings(objective=MULTI_GW, horizon=3)
    base, scenarios = chip_scenarios(inp, settings)
    by = {s.chip: s for s in scenarios}
    assert set(by) == {"wildcard", "freehit", "bboost", "3xc"}
    for s in scenarios:
        assert s.status == "solved", (s.chip, s.reason)
        assert s.evaluated == "GW1-GW3"
        assert s.gain >= -1e-6  # never worse than the plan without the chip (within solver tolerance)
        assert_legal(inp, s.plan)
    assert by["wildcard"].gw == 1 and by["freehit"].gw == 1
    tc = by["3xc"]
    cap_xp = inp.xp[next(g for g in base.gws if g.gw == tc.gw).captain, tc.gw]
    assert tc.gain >= cap_xp - 0.5  # at least (about) the armband on the plan without chips
    assert by["bboost"].plan.gws[[g.gw for g in base.gws].index(by["bboost"].gw)].chip == "bboost"
    # outside the window: not available, no solve
    late = replace(inp, gws=[20, 21], xp={(e, g + 19): v for (e, g), v in inp.xp.items()})
    _, sc = chip_scenarios(late, PlanSettings(objective=MULTI_GW, horizon=2))
    assert all(s.status == "not available" and "window" in s.reason for s in sc)


def test_free_hit_squad_reverts():
    inp = planner_input(chips=ALL_CHIPS, bank=150)
    plan = solve(inp, PlanSettings(objective=MULTI_GW, horizon=2, chip="freehit", max_transfers=0))
    assert plan.status == "optimal"
    fh, after = plan.gws
    assert fh.chip == "freehit" and set(fh.squad) != {o.element for o in inp.squad}
    assert set(after.squad) == {o.element for o in inp.squad}  # max_transfers 0 after the free hit
    assert after.free_transfers == inp.free_transfers  # chip week: balance carried, no +1


# ------------------------------------------------------------------------------------------- line-ups
def test_auto_subs_respect_bench_order_and_formation():
    et = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 1, 13: 3, 14: 2, 15: 4}
    xi = list(range(1, 12))  # 1 GK, 3 DEF, 5 MID, 2 FWD
    bench = [12, 13, 14, 15]
    minutes = {e: 90 for e in et}
    minutes[2] = 0  # a DEF misses: 3 DEF -> the first bench MID would leave 2 DEF, so the bench DEF comes on
    final, subs = auto_subs(xi, bench, minutes, et)
    assert subs == [(2, 14)] and 14 in final
    minutes[1] = 0  # the GK misses: only the reserve GK may replace him
    final, subs = auto_subs(xi, bench, minutes, et)
    assert (1, 12) in subs
    pts = {e: 2 for e in et} | {5: 10}
    res = realised_points(xi, bench, captain=5, vice=6, points=pts, minutes=minutes, element_type=et)
    assert res["armband"] == 5 and res["points"] == 10 * 2 + 10 + 10  # 10 others x2 + C 10, doubled
    minutes[5] = 0
    res = realised_points(xi, bench, captain=5, vice=6, points=pts, minutes=minutes, element_type=et)
    assert res["armband"] == 6
    tc = realised_points(xi, bench, 6, 7, pts, {e: 90 for e in et}, et, chip="3xc")
    assert tc["points"] == sum(pts[e] for e in xi) + 2 * pts[6]
    bb = realised_points(xi, bench, 6, 7, pts, {e: 90 for e in et}, et, chip="bboost")
    assert bb["points"] == sum(pts.values()) + pts[6]


# ------------------------------------------------------------------------------------------- app adapter
def _state(squad: list[Owned], bank: int = 5) -> PlanningState:
    return PlanningState(
        entry_id=1,
        squad=[
            SquadPlayer(
                o.element, o.element_type, o.team, o.selling_price, o.selling_price, o.selling_price, "t"
            )
            for o in squad
        ],
        bank=bank,
        team_value=1000,
        market_value=1000,
        selling_value=1000,
        free_transfers=FreeTransfers(1, 2, exact=False),
        chips_used=[],
        chips_available=ALL_CHIPS,
        last_published_gw=0,
        upcoming_gw=1,
        transfers_since_publication=0,
        as_of={},
    )


def test_from_planning_state_uses_low_ft_and_stamps_user_edits():
    pool, xp = league()
    squad = cheap_squad(pool)
    players = pd.DataFrame(
        [
            {
                "id": c.element,
                "name": f"P{c.element}",
                "team": f"T{c.team}",
                "team_id": c.team,
                "element_type": c.element_type,
                "price": c.buy / 10,
            }
            for c in pool.values()
        ]
    )
    xp_df = pd.DataFrame([{"element": e, "gw": g, "xp": v} for (e, g), v in xp.items()])
    xp_df["low_history"] = xp_df["element"] == squad[0].element
    state = _state(squad)
    inp = from_planning_state(state, players, xp_df)
    assert inp.free_transfers == 1 and inp.bank == 5 and inp.gws == [1, 2, 3, 4, 5]
    assert inp.low_history == frozenset({squad[0].element})
    assert any("low end" in n for n in inp.notes)
    swap_out = next(o for o in squad if o.element_type == 4)
    swap_in = next(
        c for c in pool.values() if c.element_type == 4 and c.element not in {o.element for o in squad}
    )
    wanted = tuple(o.element for o in squad if o != swap_out) + (swap_in.element,)
    edited = from_planning_state(
        state, players, xp_df, edits=StateEdits(bank=20, free_transfers=2, squad=wanted)
    )
    assert edited.bank == 20 and edited.free_transfers == 2
    assert any(n.startswith("User-edited squad") for n in edited.notes)
    assert any(n.startswith("User-edited bank") for n in edited.notes)
    plan = solve(edited, PlanSettings(objective=ONE_GW_NO_TRANSFERS))
    assert plan.explanation[0].startswith("User-edited")
    # missing low_history column = all False
    assert from_planning_state(state, players, xp_df.drop(columns="low_history")).low_history == frozenset()


# ------------------------------------------------------------------------------- HiGHS scheduler, limits
_SCHEDULER_SCRIPT = """
import sys
sys.path[:0] = [{src!r}, {tests!r}]
import pulp
import fpl_rocket.planner.core as core
from fpl_rocket.planner import MULTI_GW, PlanSettings, solve
from planner_synth import planner_input

if {disable_reset!r}:
    core.reset_highs_scheduler = lambda: None
# something else in the process runs HiGHS with the default thread count first
p = pulp.LpProblem("other", pulp.LpMaximize)
x = pulp.LpVariable("x", 0, 10, cat="Integer")
p += x
p += x <= 5
p.solve(pulp.HiGHS(msg=False))
assert pulp.LpStatus[p.status] == "Optimal"
for _ in range(2):
    plan = solve(planner_input(), PlanSettings(objective=MULTI_GW, horizon=3))
    print(plan.status, "|", plan.reason, "|", plan.explanation[0])
"""


def _run_scheduler_script(disable_reset: bool) -> list[str]:
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = _SCHEDULER_SCRIPT.format(
        src=str(root / "src"), tests=str(root / "tests"), disable_reset=disable_reset
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()


def test_planner_solves_after_another_highs_run_in_the_process():
    # a fresh process: HiGHS's task scheduler is process-wide (a default-threads run made every later
    # threads=1 planner run fail and silently return the hold plan as "time limit reached")
    lines = _run_scheduler_script(disable_reset=False)
    assert len(lines) == 2 and all(ln.startswith("optimal |") for ln in lines), lines


def test_a_solver_failure_is_an_explicit_error_never_a_silent_hold():
    lines = _run_scheduler_script(disable_reset=True)  # the bug, without the fix
    assert len(lines) == 2 and all(ln.startswith("solver_error |") for ln in lines), lines
    assert all("time limit" not in ln.split("|")[1] and "SOLVER ERROR" in ln for ln in lines), lines


def test_solver_error_status_and_time_limited_flag(monkeypatch):
    inp = planner_input()
    monkeypatch.setattr(core, "SOLVE", lambda *a, **k: ("solver_error", "the solver failed to run (x)"))
    plan = solve(inp, PlanSettings(objective=MULTI_GW, horizon=2))
    assert plan.status == "solver_error" and not plan.time_limited
    assert plan.explanation[0].startswith("SOLVER ERROR, no plan was solved")
    _, scenarios = chip_scenarios(planner_input(chips=ALL_CHIPS), PlanSettings(objective=MULTI_GW, horizon=2))
    assert scenarios and {sc.status for sc in scenarios} <= {"solver_error", "not available"}
    assert all(sc.gain != sc.gain for sc in scenarios)  # no gain from an unsolved plan
    monkeypatch.setattr(core, "SOLVE", lambda *a, **k: ("timeout", "the solver reached its 10 s time limit"))
    assert solve(inp, PlanSettings(objective=MULTI_GW, horizon=2)).time_limited
    monkeypatch.undo()
    plan = solve(inp, PlanSettings(objective=MULTI_GW, horizon=2))
    assert plan.status == "optimal" and not plan.time_limited


def test_time_limits_interactive_10_batch_60():
    import inspect

    from fpl_rocket import matched, registry
    from fpl_rocket.planner import BATCH_TIME_LIMIT_S, TIME_LIMIT_S

    assert TIME_LIMIT_S == 10.0 and BATCH_TIME_LIMIT_S == 60.0
    assert PlanSettings().time_limit == TIME_LIMIT_S  # the app's default
    assert inspect.signature(matched.run_season).parameters["time_limit"].default == BATCH_TIME_LIMIT_S
    assert inspect.signature(registry.build_registry).parameters["time_limit"].default is None  # -> 60
    assert core.MAX_CONCURRENT_SOLVES == 1
