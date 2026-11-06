"""The four blind managers (each complete: GW1 squad, transfers, line-up, captain, vice, bench order, chips).
Fixed settings, documented here, NOT tuned. No policy ever takes a hit; transfers <= free transfers.

MINIMAL_MODEL    predictions: the v0.2.1 model fitted at T with the as-known availability cap
                 (forecast.model_xp); planner: multi-GW objective, horizon 3, decay 0.85, restrained rule
                 (transfers <= free transfers, never a hit), HiGHS 60 s; line-up / captain / vice / bench from
                 the plan's first GW (best legal XI, top-xP captain, second vice, reserve GK first).
FORM_RESTRAINED  the same planner on the frozen form-per-90 baseline (0 for players flagged unavailable at T).
HOLD             GW1 squad = FORM_RESTRAINED's GW1 squad (baseline); never a permanent transfer; best XI and
                 captain by the baseline each GW. Plays free hit / bench boost / triple captain by the shared
                 chip rule, never the wildcard (a wildcard is a permanent transfer).
LAGGED_CROWD     GW1: the most-owned legal 15 by selected_by_percent at T (trajectories.most_owned_squad; no
                 lagged table exists before GW1). Later: target = the most-owned legal squad (same greedy
                 rule, budget = bank + selling value) by the LAGGED top-10k sample ownership
                 (crowd_lagged_players, tier "10000": picks of GW t-1 as published by T); free transfers
                 only, each one a same-position swap from the least-owned owned player outside the target
                 to the most-owned target player not owned that keeps the budget and the club limit; no
                 lagged table for the GW -> hold. Line-up:
                 the most-owned legal XI of its squad by that table, captain = its most-captained XI player
                 (else most-owned), vice = the next; bench reserve GK first, then by ownership. Without a
                 lagged table the ranking is selected_by_percent at T. Players flagged u / n at T are never in
                 a target. Chips: the shared rule with the baseline's xP; a wildcard moves to the full target,
                 a free hit to the target among players with a fixture in GW t.

GW1 (all planner policies): the planner with an unlimited-transfer first week (empty squad, budget 100.0,
horizon 3, restrained after). Wildcard / free hit weeks and GW1 use a wider candidate pool (top 10 GK / 30 DEF
/ 30 MID / 20 FWD + 5 cheapest per position, matched.initial_squad's pool). The 2022-23 GW17 unlimited-
transfer week is planned like a wildcard week (no chip is played). Bench boost / triple captain are played on
the plan made without the chip (the chip does not change the transfers).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from ..planner import (
    MULTI_GW,
    ONE_GW_WITH_TRANSFERS,
    Candidate,
    Owned,
    PlannerInput,
    PlanSettings,
    solve,
)
from ..planner.lineup import bench_order, best_xi, captain_pair
from ..rules.common import SeasonRules, SquadRules, selling_price
from ..trajectories import most_owned_squad
from . import chips as chip_rule
from .engine import ManagerState, Package
from .forecast import Forecasts

WIDE_POOL = ((1, 10), (2, 30), (3, 30), (4, 20))
WIDE_CHEAPEST = 5
LAGGED_TIER = "10000"


class PolicyError(RuntimeError):
    """A policy could not produce a package (e.g. the solver failed to run): never a silent hold."""


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 3
    decay: float = 0.85
    time_limit: float = 60.0
    restrained: str = "transfers <= free transfers, never a hit"


PLANNER = PlannerConfig()


def _pool(fc: Forecasts) -> dict[int, Candidate]:
    p = fc.players[fc.players["buyable"]]
    return {
        int(e): Candidate(int(e), int(t), int(c), int(v))
        for e, t, c, v in zip(p["element"], p["element_type"], p["team"], p["now_cost"], strict=True)
    }


def _meta(state: ManagerState, fc: Forecasts) -> dict[int, dict[str, int]]:
    m = dict(state.meta)
    for e, t, c, v in zip(
        fc.players["element"],
        fc.players["element_type"],
        fc.players["team"],
        fc.players["now_cost"],
        strict=True,
    ):
        m[int(e)] = {"element_type": int(t), "team": int(c), "cost": int(v)}
    return m


def _owned(state: ManagerState, fc: Forecasts, rules: SeasonRules) -> list[Owned]:
    m = _meta(state, fc)
    return [
        Owned(
            e, m[e]["element_type"], m[e]["team"], selling_price(p, m[e]["cost"], rules.transfers.sell_on_fee)
        )
        for e, p in sorted(state.squad.items())
    ]


def _xp_gw(xp: dict[tuple[int, int], float], gw: int) -> dict[int, float]:
    return {e: v for (e, g), v in xp.items() if g == gw}


class Policy:
    name = "policy"
    version = "1"
    source = "baseline"  # predictor of the chip triggers
    allowed_chips: tuple[str, ...] = chip_rule.PRIORITY

    def settings(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "chip_rule": chip_rule.CHIP_RULE.describe(),
            "allowed_chips": self.allowed_chips,
        }

    def decide(self, fc: Forecasts, state: ManagerState, rules: SeasonRules, inputs: Any) -> Package:
        raise NotImplementedError

    # ------------------------------------------------------------------ shared pieces
    def planner_input(
        self, fc: Forecasts, state: ManagerState, rules: SeasonRules, gws: list[int], chip: str | None
    ) -> PlannerInput:
        empty = not state.squad
        return PlannerInput(
            squad=[] if empty else _owned(state, fc, rules),
            bank=rules.squad.budget if empty else state.bank,
            free_transfers=max(1, state.free_transfers) if empty else state.free_transfers,
            gws=gws,
            xp=fc.xp(self.source),
            pool=_pool(fc),
            low_history=fc.low,
            chips_available=[{"name": chip, "start_event": gws[0], "stop_event": gws[0]}] if chip else [],
            squad_rules=rules.squad,
            transfer_rules=rules.transfers,
            formation=dict(rules.formation),
        )

    def wildcard_gain(self, fc: Forecasts, state: ManagerState, rules: SeasonRules) -> tuple[bool, float]:
        """Best legal squad's xP over the next wc_horizon GWs minus the current squad's (both held)."""
        cfg = chip_rule.CHIP_RULE
        gws = fc.gws[: cfg.wc_horizon]
        xp = fc.xp(self.source)
        et = {e: m["element_type"] for e, m in _meta(state, fc).items()}
        squad = sorted(state.squad)
        cur = 0.0
        for g in gws:
            score = {e: xp.get((e, g), 0.0) for e in squad}
            xi = best_xi(squad, et, score, rules.formation)
            cap, _ = captain_pair(xi, score, fc.low)
            cur += sum(score[e] for e in xi) + score[cap]
        inp = self.planner_input(fc, state, rules, gws, "wildcard")
        plan = solve(
            inp,
            PlanSettings(
                objective=MULTI_GW,
                horizon=len(gws),
                decay=1.0,
                max_transfers=0,
                allow_hits=False,
                bench_weight=0.0,
                chip="wildcard",
                time_limit=PLANNER.time_limit,
                pool_top=WIDE_POOL,
                pool_cheapest=WIDE_CHEAPEST,
            ),
        )
        if plan.status != "optimal":
            self._events.append(
                {"event": "wildcard_trigger_not_evaluated", "status": plan.status, "reason": plan.reason}
            )
            return False, float("nan")
        gain = plan.expected_total - cur
        return gain >= cfg.wc_gain, gain

    _events: list[dict[str, Any]]

    def chip_triggers(
        self,
        fc: Forecasts,
        state: ManagerState,
        rules: SeasonRules,
        bench: list[int],
        captain: int,
    ) -> dict:
        cfg = chip_rule.CHIP_RULE
        xp = _xp_gw(fc.xp(self.source), fc.gw)
        blanks = fc.blanks(sorted(state.squad), fc.gw)
        return {
            "wildcard": lambda: self.wildcard_gain(fc, state, rules),
            "freehit": lambda: (blanks >= cfg.fh_blank_players, float(blanks)),
            "bboost": lambda: (
                sum(xp.get(e, 0.0) for e in bench) >= cfg.bb_bench_xp,
                sum(xp.get(e, 0.0) for e in bench),
            ),
            "3xc": lambda: (xp.get(captain, 0.0) >= cfg.tc_captain_xp, xp.get(captain, 0.0)),
        }


def _package_from_plan(fc: Forecasts, state: ManagerState, rules: SeasonRules, plan, chip, xp) -> Package:
    gp = plan.first()
    # transfers = the difference between the plan's GW squad and the owned squad (the planner models a
    # free-hit week without transfer variables, so its transfer lists are empty there)
    owned = set(state.squad)
    t_in = sorted(set(gp.squad) - owned)
    t_out = sorted(owned - set(gp.squad))
    n = len(t_in) if state.squad else 0
    hits = rules.hits(fc.gw, state.free_transfers, n, chip)
    scores = {e: xp.get((e, fc.gw), 0.0) for e in set(gp.squad) | owned}
    return Package(
        fc.season,
        fc.gw,
        t_in,
        t_out,
        list(gp.xi),
        list(gp.bench),
        int(gp.captain),
        int(gp.vice),
        chip,
        hits,
        scores,
        {
            "status": plan.status,
            "reason": plan.reason,
            "seconds": round(plan.solve_seconds, 2),
            "time_limited": plan.time_limited,
            "fallback": plan.status != "optimal",
        },
    )


class PlannerPolicy(Policy):
    """MINIMAL_MODEL / FORM_RESTRAINED."""

    def __init__(self, name: str, source: str) -> None:
        self.name, self.source = name, source
        self._events = []

    def settings(self) -> dict[str, Any]:
        return {
            **super().settings(),
            "planner": asdict(PLANNER),
            "objective": MULTI_GW,
            "wide_pool": WIDE_POOL,
        }

    def _solve(self, fc, state, rules, chip_for_planner: str | None, wide: bool):
        gws = fc.gws[: PLANNER.horizon]
        inp = self.planner_input(fc, state, rules, gws, chip_for_planner)
        settings = PlanSettings(
            objective=MULTI_GW,
            horizon=PLANNER.horizon,
            decay=PLANNER.decay,
            max_transfers=None,
            allow_hits=False,
            chip=chip_for_planner,
            time_limit=PLANNER.time_limit,
            **({"pool_top": WIDE_POOL, "pool_cheapest": WIDE_CHEAPEST} if wide else {}),
        )
        try:
            plan = solve(inp, settings)
        except (IndexError, ValueError) as exc:  # e.g. no buyable squad at all: never a silent hold
            raise PolicyError(f"{fc.season} GW{fc.gw} {self.name}: planner failed ({exc!r})") from exc
        if plan.status == "solver_error":
            raise PolicyError(f"{fc.season} GW{fc.gw} {self.name}: {plan.reason}")
        if plan.status != "optimal":
            if not state.squad:
                raise PolicyError(f"{fc.season} GW{fc.gw} {self.name}: no GW1 squad ({plan.reason})")
            self._events.append({"event": "solver_fallback", "chip": chip_for_planner, "reason": plan.reason})
        return plan

    def decide(self, fc: Forecasts, state: ManagerState, rules: SeasonRules, inputs: Any) -> Package:
        self._events = []
        xp = fc.xp(self.source)
        unlimited = not state.squad or fc.gw in rules.special.unlimited_free_transfer_gws
        base = self._solve(fc, state, rules, "wildcard" if unlimited else None, wide=unlimited)
        gp = base.first()
        triggers = self.chip_triggers(fc, state, rules, gp.bench, gp.captain)
        if not state.squad:  # GW1: only bench boost / triple captain can apply
            triggers = {k: v for k, v in triggers.items() if k in ("bboost", "3xc")}
        chip, record = chip_rule.decide(
            rules, state.chips, fc.gw, state.last_fh_gw, triggers, self.allowed_chips
        )
        plan = base
        if chip in ("wildcard", "freehit"):
            chip_plan = self._solve(fc, state, rules, chip, wide=True)
            if chip_plan.status == "optimal" or record.get("reason") == "last eligible GW":
                plan = chip_plan
            else:
                self._events.append({"event": "chip_skipped_fallback", "chip": chip})
                chip = None
        pkg = _package_from_plan(fc, state, rules, plan, chip, xp)
        if pkg.hits:
            raise PolicyError(f"{fc.season} GW{fc.gw} {self.name}: the plan takes {pkg.hits} hit(s)")
        pkg.notes = [{"chip_rule": record}, *self._events]
        if base is not plan:
            pkg.solver["base_status"] = base.status
        return pkg


class HoldPolicy(Policy):
    name = "hold"
    source = "baseline"
    allowed_chips = ("freehit", "bboost", "3xc")

    def __init__(self) -> None:
        self._events = []
        self._start = PlannerPolicy("form_restrained", "baseline")

    def decide(self, fc: Forecasts, state: ManagerState, rules: SeasonRules, inputs: Any) -> Package:
        self._events = []
        xp = fc.xp(self.source)
        if not state.squad:
            pkg = self._start.decide(fc, state, rules, inputs)
            pkg.notes.append({"hold": "GW1 squad = FORM_RESTRAINED's GW1 squad"})
            return pkg
        m = _meta(state, fc)
        et = {e: m[e]["element_type"] for e in state.squad}
        squad = sorted(state.squad)
        score = {e: xp.get((e, fc.gw), 0.0) for e in squad}
        xi = best_xi(squad, et, score, rules.formation)
        cap, vice = captain_pair(xi, score, fc.low)
        bench = bench_order([e for e in squad if e not in xi], et, score)
        chip, record = chip_rule.decide(
            rules,
            state.chips,
            fc.gw,
            state.last_fh_gw,
            self.chip_triggers(fc, state, rules, bench, cap),
            self.allowed_chips,
        )
        if chip == "freehit":
            inp = self.planner_input(fc, state, rules, fc.gws[:1], "freehit")
            plan = solve(
                inp,
                PlanSettings(
                    objective=ONE_GW_WITH_TRANSFERS,
                    max_transfers=None,
                    chip="freehit",
                    time_limit=PLANNER.time_limit,
                    pool_top=WIDE_POOL,
                    pool_cheapest=WIDE_CHEAPEST,
                ),
            )
            if plan.status == "solver_error":
                raise PolicyError(f"{fc.season} GW{fc.gw} hold: {plan.reason}")
            if plan.status == "optimal" or record.get("reason") == "last eligible GW":
                pkg = _package_from_plan(fc, state, rules, plan, "freehit", xp)
                pkg.notes = [{"chip_rule": record}, *self._events]
                if plan.status != "optimal":
                    pkg.notes.append({"event": "solver_fallback", "chip": "freehit", "reason": plan.reason})
                return pkg
            self._events.append({"event": "chip_skipped_fallback", "chip": "freehit"})
            chip = None
        scores = {e: xp.get((e, fc.gw), 0.0) for e in squad}
        return Package(
            fc.season,
            fc.gw,
            [],
            [],
            xi,
            bench,
            cap,
            vice,
            chip,
            0,
            scores,
            {"status": "no_solver"},
            [{"chip_rule": record}, *self._events],
        )


class LaggedCrowdPolicy(Policy):
    name = "lagged_crowd"
    source = "baseline"

    def __init__(self) -> None:
        self._events = []

    def settings(self) -> dict[str, Any]:
        return {
            **super().settings(),
            "tier": LAGGED_TIER,
            "rule": "free transfers toward the most-owned legal squad",
        }

    @staticmethod
    def _table(inputs: Any) -> pd.DataFrame:
        t = inputs.asknown.crowd_lagged_players
        if t is None or t.empty:
            return pd.DataFrame()
        return t[t["tier"].astype(str) == LAGGED_TIER]

    def _target(
        self, fc, own: dict[int, float], budget: int, squad_rules: SquadRules, only_playing: bool
    ) -> list[int]:
        p = fc.players
        sel = p["buyable"] & ~p["status"].isin(("u", "n"))
        if only_playing:
            sel &= p["element"].map(lambda e: fc.n_fixtures.get((int(e), fc.gw), 0) > 0)
        table = pd.DataFrame(
            {
                "element": p["element"].astype(int),
                "element_type": p["element_type"].astype(int),
                "team": p["team"].astype(int),
                "price": p["now_cost"].astype(int),
                "ownership": p["element"].map(lambda e: own.get(int(e), 0.0)).astype(float),
                "selectable": sel.to_numpy(bool),
            }
        )
        rules = SquadRules(
            squad_rules.size, dict(squad_rules.per_position), squad_rules.max_per_team, int(budget)
        )
        return most_owned_squad(table, rules)

    def _lineup(self, squad, et, own, capt, formation):
        xi = best_xi(squad, et, own, formation)
        ranked = sorted(xi, key=lambda e: (-capt.get(e, 0.0), -own.get(e, 0.0), e))
        if capt and max(capt.get(e, 0.0) for e in xi) > 0:
            cap, vice = ranked[0], ranked[1]
        else:
            cap, vice = captain_pair(xi, own)
        bench = bench_order([e for e in squad if e not in xi], et, own)
        return xi, bench, cap, vice

    def decide(self, fc: Forecasts, state: ManagerState, rules: SeasonRules, inputs: Any) -> Package:
        self._events = []
        m = _meta(state, fc)
        table = self._table(inputs)
        if len(table):
            own = {int(e): float(v) for e, v in zip(table["element"], table["ownership"], strict=True)}
            capt = {int(e): float(v) for e, v in zip(table["element"], table["captaincy_share"], strict=True)}
            ranking = "lagged top-10k ownership (source GW " + str(int(table["source_gw"].iloc[0])) + ")"
        else:
            own = {
                int(e): float(v) for e, v in zip(fc.players["element"], fc.players["ownership"], strict=True)
            }
            capt = {}
            ranking = "selected_by_percent at T (no lagged table)"
        price = fc.price()
        if not state.squad:
            target = self._target(fc, own, rules.squad.budget, rules.squad, False)
            new = sorted(target)
            et = {e: m[e]["element_type"] for e in new}
            xi, bench, cap, vice = self._lineup(new, et, own, capt, rules.formation)
            chip, record = self._chips(fc, state, rules, bench, cap, gw1=True)
            return Package(
                fc.season,
                fc.gw,
                new,
                [],
                xi,
                bench,
                cap,
                vice,
                chip,
                0,
                own_scores(own, new),
                {"status": "no_solver"},
                [{"ranking": ranking, "chip_rule": record}],
            )
        squad = sorted(state.squad)
        sell = {o.element: o.selling_price for o in _owned(state, fc, rules)}
        budget = state.bank + sum(sell.values())
        et_all = {e: v["element_type"] for e, v in m.items()}
        # line-up of the current squad first (chip triggers use the line-up without a chip)
        xi0, bench0, cap0, _ = self._lineup(squad, {e: et_all[e] for e in squad}, own, capt, rules.formation)
        chip, record = self._chips(fc, state, rules, bench0, cap0, gw1=False, own=own)
        unlimited = chip in ("wildcard", "freehit") or fc.gw in rules.special.unlimited_free_transfer_gws
        t_in: list[int] = []
        t_out: list[int] = []
        if len(table) or unlimited:
            target = set(self._target(fc, own, budget, rules.squad, chip == "freehit"))
            if unlimited:
                t_out = sorted(set(squad) - target)
                t_in = sorted(target - set(squad))
            else:
                t_in, t_out = self._moves(
                    squad, target, own, sell, price, et_all, m, state.bank, rules, state.free_transfers
                )
        new = sorted((set(squad) - set(t_out)) | set(t_in))
        xi, bench, cap, vice = self._lineup(new, {e: et_all[e] for e in new}, own, capt, rules.formation)
        hits = rules.hits(fc.gw, state.free_transfers, len(t_in), chip)
        if hits:
            raise PolicyError(f"{fc.season} GW{fc.gw} lagged_crowd: {hits} hit(s)")
        return Package(
            fc.season,
            fc.gw,
            t_in,
            t_out,
            xi,
            bench,
            cap,
            vice,
            chip,
            0,
            own_scores(own, set(new) | set(squad)),
            {"status": "no_solver"},
            [{"ranking": ranking, "chip_rule": record, "lagged_table": bool(len(table))}, *self._events],
        )

    def _chips(self, fc, state, rules, bench, cap, gw1: bool, own=None):
        trig = self.chip_triggers(fc, state, rules, bench, cap)
        if gw1:
            trig = {k: v for k, v in trig.items() if k in ("bboost", "3xc")}
        else:
            trig["wildcard"] = lambda: self.target_gain(fc, state, rules, own or {})
        return chip_rule.decide(rules, state.chips, fc.gw, state.last_fh_gw, trig, self.allowed_chips)

    def target_gain(self, fc, state, rules, own: dict[int, float]) -> tuple[bool, float]:
        """Wildcard trigger: baseline xP over the next wc_horizon GWs of the full most-owned target (what this
        policy's wildcard builds) minus the current squad's (best XI + captain per GW, both held)."""
        cfg = chip_rule.CHIP_RULE
        sell = {o.element: o.selling_price for o in _owned(state, fc, rules)}
        target = self._target(fc, own, state.bank + sum(sell.values()), rules.squad, False)
        xp = fc.xp(self.source)
        et = {e: m["element_type"] for e, m in _meta(state, fc).items()}

        def value(squad: list[int]) -> float:
            v = 0.0
            for g in fc.gws[: cfg.wc_horizon]:
                score = {e: xp.get((e, g), 0.0) for e in squad}
                xi = best_xi(squad, et, score, rules.formation)
                cap, _ = captain_pair(xi, score, fc.low)
                v += sum(score[e] for e in xi) + score[cap]
            return v

        gain = value(sorted(target)) - value(sorted(state.squad))
        return gain >= cfg.wc_gain, gain

    @staticmethod
    def _moves(squad, target, own, sell, price, et, meta, bank, rules, free) -> tuple[list[int], list[int]]:
        cur = set(squad)
        t_in: list[int] = []
        t_out: list[int] = []
        limit = rules.squad.max_per_team
        for _ in range(max(0, int(free))):
            ins = sorted(target - cur, key=lambda e: (-own.get(e, 0.0), price.get(e, 0), e))
            outs = sorted(cur - target - set(t_in), key=lambda e: (own.get(e, 0.0), e))
            done = False
            for e_in in ins:
                for e_out in outs:
                    if et[e_out] != et[e_in]:
                        continue
                    trial = (cur - {e_out}) | {e_in}
                    clubs = Counter(meta[e]["team"] for e in trial)
                    before = Counter(meta[e]["team"] for e in cur)
                    if any(c > max(limit, before.get(k, 0)) for k, c in clubs.items()):
                        continue
                    new_bank = bank + sell.get(e_out, 0) - price[e_in]
                    if new_bank < 0:
                        continue
                    bank = new_bank
                    t_in.append(e_in)
                    t_out.append(e_out)
                    cur = trial
                    sell[e_in] = price[e_in]
                    done = True
                    break
                if done:
                    break
            if not done:
                break
        return t_in, t_out


def own_scores(own: dict[int, float], elements) -> dict[int, float]:
    return {int(e): float(own.get(int(e), 0.0)) for e in elements}


def make_policies() -> dict[str, Policy]:
    return {
        "minimal_model": PlannerPolicy("minimal_model", "model"),
        "form_restrained": PlannerPolicy("form_restrained", "baseline"),
        "hold": HoldPolicy(),
        "lagged_crowd": LaggedCrowdPolicy(),
    }


POLICY_NAMES = ("minimal_model", "form_restrained", "hold", "lagged_crowd")
SOURCES = {
    "minimal_model": "model",
    "form_restrained": "baseline",
    "hold": "baseline",
    "lagged_crowd": "baseline",
}
