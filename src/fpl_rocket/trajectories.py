"""Prospective policy trajectories: four virtual managers carried forward from GW9 (step 3g).

Why: the retrospective matched backtest cannot settle whether the planner helps (end-of-GW archive data, a
fixture leak, test seasons used up). The decisive evidence is PROSPECTIVE: squads chosen before each deadline,
recorded before it, scored after the gameweek.

The four managers (TRAJECTORIES), all from the SAME start squad, chips never used:
    model_aggressive   planner objective (c): horizon 3, decay 0.85, <= 2 transfers per GW, hits allowed when
                       the objective says so, on the model's xP (xp.parquet as issued)
    model_restrained   the same predictions and horizon under the RESTRAINED rule (RESTRAINED_RULE, step 3h):
                       transfers this GW <= the free transfers available this GW, never a hit
    form_restrained    the restrained rule on the frozen baseline (form per 90 x availability: the per-fixture
                       prediction behind registry_xp's `baseline_xp`, times the fixture count of each GW)
    hold               never transfers; best XI + captain by the baseline each GW

RESTRAINED rule (free_only): the transfers made in a GW never exceed the free transfers available entering it
(banked by the rules engine: +1 per GW, capped at TransferRules.max_bank = 5; the balance after a wildcard /
free-hit week follows the rules' chip_week_policy, never reached because the trajectories play no chips), and
a hit is never taken. A restrained manager may therefore bank a transfer when nothing is worth doing and spend
two or more later, like a disciplined human. There is no minimum-gain threshold: the planner uses a free
transfer whenever its 3-GW decayed objective gains anything at all (a known simplification; a human might keep
the transfer for a marginal gain). In step 3g the rule was "<= 1 transfer per GW" (matched.RESTRAINED_RULES
keeps it selectable as one_per_gw for the retrospective replay).
Every manager: players flagged low_history are never captain / vice and never bought (planner rule); players
the game marks unselectable (can_select False or removed) are never bought.

START (START_RULE, at the first pre-deadline run with gw >= REGISTRY_START_GW that finds no earlier state):
the most-owned legal 15 under 100.0 at that deadline's prices. Players are ranked by selected_by_percent
(highest first; ties: cheaper first, then lower id) and taken greedily while (1) their position has an open
slot (2 GKP, 5 DEF, 5 MID, 3 FWD), (2) their club has fewer than 3 picks and (3) the money left after buying
them still pays for the cheapest legal completion of the open slots (cheapest selectable players per open
slot, at most 3 per club). Neither predictor is used. Purchase price = now_cost, bank = 100.0 - cost, no
transfers in the start GW, 1 free transfer entering the next GW.

WRITE (scripts/trajectories_write.py, bundle workflow, pre-deadline bundle only): the previous state is read
from the latest EARLIER deadline-gw<k> release (never a bundle-* release); GWs k+1..N-1 without a state are
recorded as status=missing (squad unchanged, the last line-up carried, free transfers accrue). The new state
(squads with purchase prices, bank, free transfers, transfer log, line-ups, provisional points from the
bundle's finished-GW live stats) and the GW N choices (XI, captain, vice, bench order, transfers) are written
to trajectories_state.json / trajectories_plans.json and published in deadline-gw<N> BEFORE the deadline.

SCORE (scripts/registry_score.py, registry_score.yml, once the GW is finished and data-checked): realised
points with automatic substitutions and the armband rule, hits deducted, cumulative totals, versus the
official GW average (events.average_entry_score), appended to registry/trajectories_scores.csv on spike-log.
Weeks with status missing / solver_error / fallback are scored on the rolled-forward squad and flagged.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import RELEASES_URL
from .registry import REGISTRY_START_GW, deadline_tag
from .rules.season_2026_27 import SQUAD, SquadRules, TransferRules, selling_price

JSON = dict[str, Any]

SCHEMA = 1
STATE_FILE, PLANS_FILE = "trajectories_state.json", "trajectories_plans.json"
FILES = (STATE_FILE, PLANS_FILE)
SCORES_FILE = "registry/trajectories_scores.csv"  # path on the spike-log branch
SCORES_URL = "https://raw.githubusercontent.com/drdata2018/fpl_rocket/spike-log/" + SCORES_FILE
HORIZON = 3
DECAY = 0.85
START_RULE = (
    "most-owned legal 15 under 100.0 at the start deadline's prices: players ranked by selected_by_percent "
    "(ties: cheaper, then lower id), taken greedily while the position has an open slot (2/5/5/3), the club "
    "has fewer than 3 picks and the money left pays for the cheapest legal completion of the open slots; "
    "selectable players only (can_select, not removed); purchase price = now_cost"
)


RESTRAINED_RULE = (
    "transfers this GW <= the free transfers available (banked by the rules engine, cap 5), never a hit; "
    "no minimum gain: a free transfer is used whenever the horizon objective gains anything"
)


@dataclass(frozen=True)
class TrajectoryPolicy:
    name: str
    label: str
    source: str  # "model" | "baseline": the predictions the policy plans with
    max_transfers: int | None  # per GW; None = no fixed limit (with allow_hits False: the free transfers)
    allow_hits: bool
    hold: bool = False

    @property
    def restrained(self) -> bool:
        return not self.hold and self.max_transfers is None and not self.allow_hits

    def transfer_limit(self, free_transfers: int) -> int:
        """Most transfers this policy may make in a GW entered with `free_transfers`."""
        if self.hold:
            return 0
        if self.allow_hits:
            return int(self.max_transfers) if self.max_transfers is not None else 10**6
        free = max(0, int(free_transfers))
        return free if self.max_transfers is None else min(free, int(self.max_transfers))

    def describe(self) -> JSON:
        out = asdict(self)
        if not self.hold:
            out.update(objective="multi_gw", horizon=HORIZON, decay=DECAY, chips="never")
        if self.restrained:
            out["rule"] = RESTRAINED_RULE
        return out


TRAJECTORIES: tuple[TrajectoryPolicy, ...] = (
    TrajectoryPolicy("model_aggressive", "Model, aggressive", "model", 2, True),
    TrajectoryPolicy("model_restrained", "Model, restrained", "model", None, False),
    TrajectoryPolicy("form_restrained", "Form, restrained", "baseline", None, False),
    TrajectoryPolicy("hold", "Hold (no transfers)", "baseline", 0, False, hold=True),
)
BY_NAME = {p.name: p for p in TRAJECTORIES}
FLAGGED_STATUSES = ("missing", "solver_error", "fallback")
SCORE_COLUMNS = (
    "gw",
    "policy",
    "label",
    "status",  # ok | fallback | solver_error | missing | not_started
    "flagged",  # True when the GW's line-up was not a solved plan (missing / solver_error / fallback)
    "reason",
    "scored_at",
    "bundle_id",
    "points",  # gross: automatic substitutions + armband
    "hits",
    "hit_cost",
    "net",
    "transfers",
    "cumulative_net",
    "gw_average",
    "net_minus_average",
    "cumulative_vs_average",
    "transfers_total",
    "hits_total",
    "captain",
    "vice",
    "armband",
    "auto_subs",
    "xi",
    "bench",
    "transfers_in",
    "transfers_out",
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ============================================================================================ start squad
def players_table(bootstrap: Mapping[str, Any]) -> pd.DataFrame:
    """element, element_type, team, price (tenths), ownership (%), selectable from bootstrap elements."""
    rows = []
    for p in bootstrap.get("elements", []) or []:
        et = int(p.get("element_type") or 0)
        if et not in (1, 2, 3, 4):
            continue
        rows.append(
            {
                "element": int(p["id"]),
                "element_type": et,
                "team": int(p.get("team") or 0),
                "price": int(p.get("now_cost") or 0),
                "ownership": float(p.get("selected_by_percent") or 0.0),
                "selectable": bool(p.get("can_select", True)) and not bool(p.get("removed", False)),
            }
        )
    return pd.DataFrame(rows, columns=["element", "element_type", "team", "price", "ownership", "selectable"])


def _cheapest_completion(
    players: pd.DataFrame, open_slots: Mapping[int, int], clubs: Counter, taken: set[int], max_per_team: int
) -> int | None:
    """Cost of the cheapest way to fill `open_slots` (respecting the club limit greedily), None if none."""
    clubs = Counter(clubs)
    total = 0
    for et in sorted(open_slots):
        need = open_slots[et]
        if need <= 0:
            continue
        pos = players[(players["element_type"] == et) & ~players["element"].isin(taken)]
        for r in pos.sort_values(["price", "element"]).itertuples(index=False):
            if need == 0:
                break
            if clubs[r.team] >= max_per_team:
                continue
            clubs[r.team] += 1
            total += int(r.price)
            need -= 1
        if need:
            return None
    return total


def most_owned_squad(players: pd.DataFrame, rules: SquadRules = SQUAD) -> list[int]:
    """START_RULE on a players table (element, element_type, team, price, ownership[, selectable])."""
    pool = players[players["selectable"]] if "selectable" in players else players
    ranked = pool.sort_values(["ownership", "price", "element"], ascending=[False, True, True])
    open_slots = dict(rules.per_position)
    clubs: Counter = Counter()
    chosen: list[int] = []
    spent = 0
    for r in ranked.itertuples(index=False):
        if open_slots.get(int(r.element_type), 0) <= 0 or clubs[int(r.team)] >= rules.max_per_team:
            continue
        slots = dict(open_slots)
        slots[int(r.element_type)] -= 1
        trial_clubs = Counter(clubs)
        trial_clubs[int(r.team)] += 1
        rest = _cheapest_completion(
            pool, slots, trial_clubs, set(chosen) | {int(r.element)}, rules.max_per_team
        )
        if rest is None or spent + int(r.price) + rest > rules.budget:
            continue
        chosen.append(int(r.element))
        spent += int(r.price)
        open_slots, clubs = slots, trial_clubs
        if not any(open_slots.values()):
            break
    if len(chosen) != rules.size:
        raise ValueError(f"no legal most-owned squad under {rules.budget / 10:.1f} ({len(chosen)} players)")
    return chosen


# ============================================================================================ predictions
def prediction_tables(
    xp: pd.DataFrame, baseline_per_fixture: Mapping[int, float], gws: list[int]
) -> dict[str, dict[tuple[int, int], float]]:
    """{"model": {(element, gw): xP}, "baseline": {(element, gw): per-fixture baseline x n_fixtures}} for
    `gws` from the issued xp table (one row per element and GW, with n_fixtures)."""
    x = xp[xp["gw"].isin(gws)]
    model = {
        (int(e), int(g)): float(v) for e, g, v in zip(x["element"], x["gw"], x["xp"].fillna(0.0), strict=True)
    }
    nfix = x["n_fixtures"] if "n_fixtures" in x else pd.Series(1, index=x.index)
    base = {
        (int(e), int(g)): float(baseline_per_fixture.get(int(e), 0.0)) * float(n)
        for e, g, n in zip(x["element"], x["gw"], nfix.fillna(0), strict=True)
    }
    return {"model": model, "baseline": base}


def low_history(xp: pd.DataFrame) -> frozenset[int]:
    if "low_history" not in xp:
        return frozenset()
    flag = xp["low_history"].fillna(False).astype(bool)
    return frozenset(int(e) for e in xp.loc[flag, "element"].unique())


# ============================================================================================ state
def _lineup(xi: list[int], bench: list[int], captain: int, vice: int) -> JSON:
    return {
        "xi": [int(e) for e in xi],
        "bench": [int(e) for e in bench],
        "captain": int(captain),
        "vice": int(vice),
    }


def _best_lineup(
    squad: list[int], et: Mapping[int, int], score: Mapping[int, float], low: frozenset[int], formation: Any
) -> JSON:
    from .planner.lineup import bench_order, best_xi, captain_pair

    xi = best_xi(squad, et, score, formation)
    cap, vice = captain_pair(xi, score, low)
    return _lineup(xi, bench_order([e for e in squad if e not in xi], et, score), cap, vice)


def start_state(
    bootstrap: Mapping[str, Any],
    gw: int,
    *,
    bundle_id: str,
    squad_rules: SquadRules = SQUAD,
) -> JSON:
    """The trajectories' common start (START_RULE) before any line-up is chosen."""
    players = players_table(bootstrap)
    squad = most_owned_squad(players, squad_rules)
    info = players.set_index("element")
    cost = int(sum(int(info.at[e, "price"]) for e in squad))
    members = [
        {
            "element": int(e),
            "element_type": int(info.at[e, "element_type"]),
            "team": int(info.at[e, "team"]),
            "purchase_price": int(info.at[e, "price"]),
        }
        for e in squad
    ]
    trajectories = {
        p.name: {
            "label": p.label,
            "policy": p.describe(),
            "squad": [dict(m) for m in members],
            "bank": squad_rules.budget - cost,
            "free_transfers_next": 0,
            "transfers_total": 0,
            "hits_total": 0,
            "transfer_log": [],
            "lineups": {},
            "points": {},
            "cumulative_points": 0,
            "last_plan": None,
        }
        for p in TRAJECTORIES
    }
    return {
        "schema": SCHEMA,
        "kind": "trajectories_state",
        "gw": None,
        "created_gw": int(gw),
        "created_bundle_id": bundle_id,
        "start_rule": START_RULE,
        "start_squad": [int(e) for e in squad],
        "start_cost": cost,
        "start_bank": squad_rules.budget - cost,
        "trajectories": trajectories,
    }


def _roll_missing(state: JSON, upto: int, transfer_rules: TransferRules) -> list[int]:
    """Weeks after state["gw"] and before `upto` without a recorded state: squad unchanged, last line-up
    carried, free transfers accrue (no transfer made). Returns the missing GWs."""
    last = state.get("gw")
    if last is None:
        return []
    missing = list(range(int(last) + 1, int(upto)))
    for t in state["trajectories"].values():
        for g in missing:
            prev = t["lineups"].get(str(g - 1)) or {}
            t["lineups"][str(g)] = {
                **{k: prev.get(k) for k in ("xi", "bench", "captain", "vice")},
                "status": "missing",
                "reason": "no deadline state for this GW: squad and line-up carried forward",
                "transfers_in": [],
                "transfers_out": [],
                "hits": 0,
                "hit_cost": 0,
            }
            t["free_transfers_next"] = min(
                transfer_rules.max_bank, int(t["free_transfers_next"]) + transfer_rules.free_per_gw
            )
    return missing


@dataclass
class StepInputs:
    """Everything a deadline step needs (all from the pre-deadline bundle)."""

    gw: int
    bootstrap: Mapping[str, Any]
    xp: pd.DataFrame  # issued xp table (element, gw, xp, n_fixtures, low_history)
    baseline_per_fixture: Mapping[int, float]
    bundle_id: str
    deadline: str | None = None
    live: pd.DataFrame | None = None  # finished-GW live stats (gw, element, total_points, minutes)
    time_limit: float | None = None
    created_at: str | None = None


def inputs_from_bundle(
    bundle_dir: str | Path,
    archive: str | Path | None = None,
    *,
    gw: int | None = None,
    time_limit: float | None = None,
    baseline: Mapping[int, float] | None = None,
) -> StepInputs:
    """StepInputs from a bundle directory (sha256-checked): bootstrap and finished-GW live stats, the issued
    xp table, and the frozen baseline's per-fixture xP fitted on the bundle's current season + the archived
    previous season (when `archive` exists), exactly as registry_write.py does (`baseline` given: used as is,
    for tests). Needs the model group."""
    from . import views
    from .backtest import previous_season
    from .bundle import APP_TABLES, app_shared, read_manifest, verified_file
    from .model.pipeline import bundle_history_rows
    from .registry import baseline_per_fixture
    from .rules import CURRENT_SEASON

    bdir = Path(bundle_dir)
    manifest = read_manifest(bdir)
    if "xp.parquet" not in {f["name"] for f in manifest.get("files", [])}:
        raise ValueError(f"bundle {manifest['bundle_id']} has no xp table ({manifest.get('model_error')})")
    raw: dict[str, bytes] = {}
    names = [f"{t}.parquet" for t in APP_TABLES] + ["element_history.parquet", "xp.parquet"]
    for name in names:
        raw[name] = (bdir / name).read_bytes()
        verified_file(manifest, name, raw[name])
    shared = app_shared(manifest, raw, views.LIVE_STATS)
    tables = {
        t: pd.read_parquet(bdir / f"{t}.parquet")
        for t in ("players", "teams", "events", "fixtures", "element_history")
    }
    if baseline is None:
        history = bundle_history_rows(tables, CURRENT_SEASON)
        if archive is not None and Path(archive).exists():
            arch = pd.read_parquet(archive)
            history = pd.concat(
                [arch[arch["season"] == previous_season(CURRENT_SEASON)], history], ignore_index=True
            )
        players = pd.DataFrame(
            {
                "element": tables["players"]["id"].astype(int),
                "code": tables["players"]["code"].astype("Int64"),
            }
        )
        baseline = baseline_per_fixture(history, players).to_dict()
    return StepInputs(
        gw=int(gw or manifest.get("next_event") or 0),
        bootstrap=shared.bootstrap,
        xp=pd.read_parquet(bdir / "xp.parquet"),
        baseline_per_fixture=dict(baseline),
        bundle_id=str(manifest["bundle_id"]),
        deadline=manifest.get("next_deadline_time"),
        live=shared.live,
        time_limit=time_limit,
    )


def step(
    previous: JSON | None, inp: StepInputs, log: Callable[[str], None] | None = None
) -> tuple[JSON, JSON]:
    """(new state, plans) for deadline GW inp.gw from the previous state (None = create the start)."""
    from .planner import BATCH_TIME_LIMIT_S, MULTI_GW, Candidate, Owned, PlannerInput, PlanSettings, solve
    from .planner.lineup import formation_from_bootstrap
    from .rules import get_rules

    gw = int(inp.gw)
    rules = get_rules()
    transfer_rules = rules.transfer_rules_from_bootstrap(dict(inp.bootstrap))
    formation = formation_from_bootstrap(inp.bootstrap)
    limit = float(inp.time_limit or BATCH_TIME_LIMIT_S)
    players = players_table(inp.bootstrap)
    info = players.set_index("element")
    if previous is None:
        state = start_state(inp.bootstrap, gw, bundle_id=inp.bundle_id)
        missing: list[int] = []
        created = True
    else:
        state = json.loads(json.dumps(previous))  # deep copy
        if state.get("gw") is not None and int(state["gw"]) >= gw:
            raise ValueError(f"previous state is for GW{state['gw']}, not before GW{gw}")
        missing = _roll_missing(state, gw, transfer_rules)
        created = False
    gws = sorted(int(g) for g in inp.xp["gw"].unique() if int(g) >= gw)[:HORIZON]
    if not gws or gws[0] != gw:
        raise ValueError(f"the xp table has no predictions for GW{gw}")
    preds = prediction_tables(inp.xp, inp.baseline_per_fixture, gws)
    low = low_history(inp.xp)
    pool = {
        int(r.element): Candidate(int(r.element), int(r.element_type), int(r.team), int(r.price))
        for r in players.itertuples(index=False)
        if r.selectable
    }
    plans: dict[str, JSON] = {}
    for policy in TRAJECTORIES:
        t = state["trajectories"][policy.name]
        t["policy"] = policy.describe()  # the rule in force at this deadline
        xp = preds[policy.source]
        squad_ids = [int(m["element"]) for m in t["squad"]]
        # current club / position from the bootstrap (a player can change club mid-season)
        for m in t["squad"]:
            if m["element"] in info.index:
                m["team"] = int(info.at[m["element"], "team"])
                m["element_type"] = int(info.at[m["element"], "element_type"])
        et = {int(m["element"]): int(m["element_type"]) for m in t["squad"]}
        score = {e: xp.get((e, gw), 0.0) for e in squad_ids}
        ft = 0 if created else int(t["free_transfers_next"])
        record: JSON = {
            "gw": gw,
            "policy": policy.name,
            "status": "ok",
            "reason": "",
            "free_transfers": ft,
            "transfers_in": [],
            "transfers_out": [],
            "buy": {},
            "sell": {},
            "hits": 0,
            "hit_cost": 0,
            "solve_seconds": 0.0,
            "time_limited": False,
        }
        if created or policy.hold:
            lineup = _best_lineup(squad_ids, et, score, low, formation)
            record["status"] = "start" if created else "ok"
            record["reason"] = "start squad: no transfers" if created else "hold: never transfers"
            record["expected_points"] = round(
                sum(score.get(e, 0.0) for e in lineup["xi"]) + score.get(lineup["captain"], 0.0), 3
            )
        else:
            owned = [
                Owned(
                    int(m["element"]),
                    int(m["element_type"]),
                    int(m["team"]),
                    selling_price(
                        int(m["purchase_price"]),
                        int(info.at[m["element"], "price"])
                        if m["element"] in info.index
                        else int(m["purchase_price"]),
                        transfer_rules.sell_on_fee,
                    ),
                )
                for m in t["squad"]
            ]
            pin = PlannerInput(
                squad=owned,
                bank=int(t["bank"]),
                free_transfers=ft,
                gws=gws,
                xp=xp,
                pool=pool,
                low_history=low,
                chips_available=[],
                transfer_rules=transfer_rules,
                formation=formation,
            )
            settings = PlanSettings(
                objective=MULTI_GW,
                horizon=HORIZON,
                decay=DECAY,
                max_transfers=policy.max_transfers,
                allow_hits=policy.allow_hits,
                time_limit=limit,
            )
            plan = solve(pin, settings)
            gp = plan.first()
            record.update(
                status="ok" if plan.status == "optimal" else plan.status,
                reason=plan.reason,
                solve_seconds=round(float(plan.solve_seconds), 2),
                time_limited=bool(plan.time_limited),
                expected_points=round(float(gp.expected_points), 3),
                expected_total=round(float(plan.expected_total), 3),
                path=[
                    {
                        "gw": g.gw,
                        "transfers_in": g.transfers_in,
                        "transfers_out": g.transfers_out,
                        "hits": g.hits,
                    }
                    for g in plan.gws
                ],
            )
            if plan.status == "optimal":
                sell = {o.element: o.selling_price for o in owned}
                record["transfers_out"] = [int(e) for e in gp.transfers_out]
                record["transfers_in"] = [int(e) for e in gp.transfers_in]
                record["sell"] = {str(e): int(sell[e]) for e in gp.transfers_out}
                record["buy"] = {str(e): int(pool[e].buy) for e in gp.transfers_in}
                record["hits"] = int(gp.hits)
                record["hit_cost"] = int(gp.hit_cost)
                if not policy.allow_hits and gp.hits:
                    raise AssertionError(f"{policy.name}: a hit in a no-hit policy")
                if len(gp.transfers_in) > policy.transfer_limit(ft):
                    raise AssertionError(f"{policy.name}: {len(gp.transfers_in)} transfers with {ft} free")
                t["bank"] = int(t["bank"]) + sum(record["sell"].values()) - sum(record["buy"].values())
                if t["bank"] != int(gp.bank_after):
                    raise AssertionError(f"{policy.name}: bank {t['bank']} != plan {gp.bank_after}")
                t["squad"] = [m for m in t["squad"] if int(m["element"]) not in set(gp.transfers_out)] + [
                    {
                        "element": int(e),
                        "element_type": int(pool[e].element_type),
                        "team": int(pool[e].team),
                        "purchase_price": int(pool[e].buy),
                    }
                    for e in gp.transfers_in
                ]
                t["squad"].sort(key=lambda m: int(m["element"]))
                lineup = _lineup(gp.xi, gp.bench, gp.captain, gp.vice)
            else:  # fallback / solver_error: the squad is held; the planner's hold line-up by xP
                lineup = _lineup(gp.xi, gp.bench, gp.captain, gp.vice)
            if record["transfers_in"] or record["transfers_out"]:
                t["transfer_log"].append(
                    {
                        "gw": gw,
                        "out": [
                            {"element": int(e), "sell": record["sell"][str(e)]}
                            for e in record["transfers_out"]
                        ],
                        "in": [
                            {"element": int(e), "buy": record["buy"][str(e)]} for e in record["transfers_in"]
                        ],
                        "hits": record["hits"],
                    }
                )
        n = len(record["transfers_in"])
        t["transfers_total"] = int(t["transfers_total"]) + n
        t["hits_total"] = int(t["hits_total"]) + int(record["hits"])
        t["free_transfers_next"] = (
            transfer_rules.free_per_gw
            if created
            else min(transfer_rules.max_bank, ft - min(n, ft) + transfer_rules.free_per_gw)
        )
        record.update(lineup)
        record["squad"] = sorted(int(m["element"]) for m in t["squad"])
        record["bank_after"] = int(t["bank"])
        t["lineups"][str(gw)] = {
            **lineup,
            "status": record["status"],
            "reason": record["reason"],
            "transfers_in": record["transfers_in"],
            "transfers_out": record["transfers_out"],
            "hits": record["hits"],
            "hit_cost": record["hit_cost"],
        }
        t["last_plan"] = {
            k: record.get(k)
            for k in ("gw", "status", "reason", "solve_seconds", "time_limited", "expected_points")
        }
        _validate(t, transfer_rules)
        plans[policy.name] = record
        if log:
            log(
                f"GW{gw} {policy.name}: {record['status']} in {record['solve_seconds']:.1f} s, "
                f"in {record['transfers_in']} out {record['transfers_out']} hits {record['hits']}, "
                f"captain {record['captain']}, bank {t['bank'] / 10:.1f}, FT next {t['free_transfers_next']}"
            )
    _provisional_points(state, inp.live, inp.bootstrap)
    state.update(
        gw=gw,
        bundle_id=inp.bundle_id,
        deadline=inp.deadline,
        created_at=inp.created_at or _now(),
        previous_gw=None if created else (previous or {}).get("gw"),
        missing_gws=missing,
        solver_time_limit_s=limit,
        horizon=HORIZON,
        decay=DECAY,
    )
    plans_doc = {
        "schema": SCHEMA,
        "kind": "trajectories_plans",
        "gw": gw,
        "bundle_id": inp.bundle_id,
        "deadline": inp.deadline,
        "created_at": state["created_at"],
        "created_gw": state["created_gw"],
        "missing_gws": missing,
        "policies": {p.name: p.describe() for p in TRAJECTORIES},
        "trajectories": plans,
    }
    state["plans_sha256"] = hashlib.sha256(_dumps(plans_doc)).hexdigest()
    return state, plans_doc


def _validate(t: JSON, transfer_rules: TransferRules, rules: SquadRules = SQUAD) -> None:
    squad = t["squad"]
    if len(squad) != rules.size or len({m["element"] for m in squad}) != rules.size:
        raise AssertionError(f"squad of {len(squad)} players")
    if Counter(int(m["element_type"]) for m in squad) != Counter(rules.per_position):
        raise AssertionError("squad composition")
    if int(t["bank"]) < 0:
        raise AssertionError("negative bank")
    if not 0 <= int(t["free_transfers_next"]) <= transfer_rules.max_bank:
        raise AssertionError("free transfers out of range")


def _provisional_points(state: JSON, live: pd.DataFrame | None, bootstrap: Mapping[str, Any]) -> None:
    """Realised points (net of hits) of every recorded line-up whose GW is in the bundle's finished-GW live
    stats; provisional (the scores CSV is authoritative)."""
    if live is None or live.empty:
        return
    from .planner.lineup import realised_points

    et = {int(p["id"]): int(p["element_type"]) for p in bootstrap.get("elements", [])}
    for t in state["trajectories"].values():
        for g, lu in t["lineups"].items():
            rows = live[live["gw"] == int(g)]
            if rows.empty or not lu.get("xi"):
                continue
            pts = dict(zip(rows["element"].astype(int), rows["total_points"].astype(int), strict=True))
            mins = dict(zip(rows["element"].astype(int), rows["minutes"].astype(int), strict=True))
            types = {e: et.get(e, 3) for e in [*lu["xi"], *lu["bench"]]}
            res = realised_points(lu["xi"], lu["bench"], lu["captain"], lu["vice"], pts, mins, types)
            t["points"][g] = int(res["points"]) - int(lu.get("hit_cost") or 0)
        t["cumulative_points"] = int(sum(t["points"].values()))
    state["points_note"] = (
        "provisional, from the bundle's finished-GW live stats; "
        "registry/trajectories_scores.csv is authoritative"
    )


def _dumps(doc: JSON) -> bytes:
    return json.dumps(doc, indent=1, sort_keys=True, default=str).encode("utf-8")


def write_files(out_dir: str | Path, state: JSON, plans: JSON) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / PLANS_FILE).write_bytes(_dumps(plans))
    (out / STATE_FILE).write_bytes(_dumps(state))
    return [out / STATE_FILE, out / PLANS_FILE]


# ============================================================================================ releases
class SourceError(RuntimeError):
    """A deadline release could not be read for a reason other than 'not there' (never start afresh)."""


def fetch(source: str, gw: int, name: str) -> JSON | None:
    """One trajectories file of deadline-gw<gw> (None when the release or the file does not exist)."""
    if not source.startswith(("http://", "https://")):
        path = Path(source) / deadline_tag(gw) / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    url = f"{source.rstrip('/')}/download/{deadline_tag(gw)}/{name}"
    try:
        resp = requests.get(url, timeout=(5.0, 60.0), headers={"User-Agent": "fpl-rocket-trajectories"})
    except requests.RequestException as exc:
        raise SourceError(f"{url}: {type(exc).__name__}: {exc}") from exc
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise SourceError(f"{url}: HTTP {resp.status_code}")
    return json.loads(resp.content)


def latest_state(source: str, before_gw: int, start: int = REGISTRY_START_GW) -> tuple[int, JSON] | None:
    """(k, state) of the latest deadline-gw<k> release with k < before_gw that holds a trajectories state;
    None when no earlier release has one. Any read error other than 'not found' raises SourceError, so a
    network failure can never restart the trajectories."""
    for k in range(int(before_gw) - 1, int(start) - 1, -1):
        doc = fetch(source, k, STATE_FILE)
        if doc is not None:
            if int(doc.get("gw", -1)) != k:
                raise SourceError(f"{deadline_tag(k)}/{STATE_FILE} says GW{doc.get('gw')}")
            return k, doc
    return None


# ============================================================================================ score
def _ids(v: Iterable[int] | None) -> str:
    return ",".join(str(int(e)) for e in (v or []))


def score_gw(
    gw: int,
    lineups: Mapping[str, JSON] | None,
    live: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    previous_rows: pd.DataFrame | None = None,
    *,
    bundle_id: str | None = None,
    missing_reason: str = "",
) -> list[JSON]:
    """One row per trajectory for GW `gw`. lineups: {policy: line-up record (xi, bench, captain, vice,
    status, hits, hit_cost, transfers_in, transfers_out)}; None = no trajectory exists yet (rows with
    status not_started). Cumulative columns continue `previous_rows` (the CSV so far)."""
    from .planner.lineup import realised_points

    stats = {int(e["id"]): e.get("stats", {}) for e in live.get("elements", [])}
    pts = {e: int(s.get("total_points", 0) or 0) for e, s in stats.items()}
    mins = {e: int(s.get("minutes", 0) or 0) for e, s in stats.items()}
    et = {int(p["id"]): int(p["element_type"]) for p in bootstrap.get("elements", [])}
    ev = next((e for e in bootstrap.get("events", []) if int(e.get("id", -1)) == int(gw)), {})
    avg = ev.get("average_entry_score")
    avg = float(avg) if avg is not None else float("nan")
    prev = previous_rows if previous_rows is not None else pd.DataFrame(columns=list(SCORE_COLUMNS))
    rows = []
    for p in TRAJECTORIES:
        before = prev[(prev["policy"] == p.name) & (pd.to_numeric(prev["gw"], errors="coerce") < gw)]
        cum = float(pd.to_numeric(before["net"], errors="coerce").fillna(0).sum())
        cum_avg = float(pd.to_numeric(before["net_minus_average"], errors="coerce").fillna(0).sum())
        t_total = int(pd.to_numeric(before["transfers"], errors="coerce").fillna(0).sum())
        h_total = int(pd.to_numeric(before["hits"], errors="coerce").fillna(0).sum())
        row: JSON = {
            "gw": int(gw),
            "policy": p.name,
            "label": p.label,
            "scored_at": _now(),
            "bundle_id": bundle_id,
            "gw_average": avg,
        }
        lu = (lineups or {}).get(p.name)
        if not lu or not lu.get("xi"):
            row.update(
                status="not_started" if lineups is None else "missing",
                flagged=True,
                reason=missing_reason or "no line-up recorded",
                cumulative_net=cum,
                cumulative_vs_average=cum_avg,
                transfers_total=t_total,
                hits_total=h_total,
            )
            rows.append(row)
            continue
        types = {e: et.get(e, 3) for e in [*lu["xi"], *lu["bench"]]}
        res = realised_points(lu["xi"], lu["bench"], lu["captain"], lu["vice"], pts, mins, types)
        hits, hit_cost = int(lu.get("hits") or 0), int(lu.get("hit_cost") or 0)
        net = int(res["points"]) - hit_cost
        n = len(lu.get("transfers_in") or [])
        status = str(lu.get("status") or "ok")
        if status == "start":
            status = "ok"
        row.update(
            status=status,
            flagged=status in FLAGGED_STATUSES,
            reason=lu.get("reason") or "",
            points=int(res["points"]),
            hits=hits,
            hit_cost=hit_cost,
            net=net,
            transfers=n,
            cumulative_net=cum + net,
            net_minus_average=net - avg if avg == avg else float("nan"),
            cumulative_vs_average=cum_avg + (net - avg if avg == avg else 0.0),
            transfers_total=t_total + n,
            hits_total=h_total + hits,
            captain=int(lu["captain"]),
            vice=int(lu["vice"]),
            armband=res["armband"],
            auto_subs=len(res["subs"]),
            xi=_ids(lu["xi"]),
            bench=_ids(lu["bench"]),
            transfers_in=_ids(lu.get("transfers_in")),
            transfers_out=_ids(lu.get("transfers_out")),
        )
        rows.append(row)
    return rows


def lineups_for(
    gw: int, source: str, start: int = REGISTRY_START_GW
) -> tuple[dict[str, JSON] | None, str, str | None]:
    """(line-ups per policy, reason, bundle id) to score GW `gw`: the deadline-gw<gw> plans when recorded;
    otherwise the latest earlier state rolled forward (status missing); None when no trajectory exists."""
    plans = fetch(source, gw, PLANS_FILE)
    if plans is not None:
        out = {}
        for name, rec in plans.get("trajectories", {}).items():
            out[name] = {
                k: rec.get(k)
                for k in (
                    "xi",
                    "bench",
                    "captain",
                    "vice",
                    "status",
                    "reason",
                    "hits",
                    "hit_cost",
                    "transfers_in",
                    "transfers_out",
                )
            }
        return out, "", plans.get("bundle_id")
    found = latest_state(source, gw, start)
    if found is None:
        return None, "the trajectories have not started (no state at or before this GW)", None
    k, state = found
    out = {}
    for name, t in state["trajectories"].items():
        last = t["lineups"].get(str(k)) or {}
        out[name] = {
            **{f: last.get(f) for f in ("xi", "bench", "captain", "vice")},
            "status": "missing",
            "reason": f"no deadline-gw{gw} trajectories: GW{k} squad and line-up carried forward",
            "hits": 0,
            "hit_cost": 0,
            "transfers_in": [],
            "transfers_out": [],
        }
    return out, f"carried from GW{k}", state.get("bundle_id")


def scores_frame(rows: Iterable[JSON]) -> pd.DataFrame:
    return pd.DataFrame(list(rows)).reindex(columns=list(SCORE_COLUMNS))


def read_scores(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists() or not path.stat().st_size:
        return scores_frame([])
    return pd.read_csv(path).reindex(columns=list(SCORE_COLUMNS))


def append_scores(csv_path: str | Path, rows: list[JSON]) -> pd.DataFrame:
    path = Path(csv_path)
    old = read_scores(path)
    new = scores_frame(rows)
    out = new if old.empty else pd.concat([old, new], ignore_index=True)
    order = {p.name: i for i, p in enumerate(TRAJECTORIES)}
    out = out.assign(_o=out["policy"].map(order)).sort_values(["gw", "_o"], kind="stable").drop(columns="_o")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def scored_gws(csv_path: str | Path) -> set[int]:
    df = read_scores(csv_path)
    return set(int(g) for g in pd.to_numeric(df["gw"], errors="coerce").dropna().unique())


def score_and_append(
    gw: int, csv_path: str | Path, client: Any, source: str = RELEASES_URL, start: int = REGISTRY_START_GW
) -> list[JSON]:
    """Score GW `gw` for every trajectory and append to the CSV (a missing week still gets rows)."""
    lineups, reason, bundle_id = lineups_for(gw, source, start)
    bootstrap = client.bootstrap()
    live = client.event_live(gw) if lineups is not None else {"elements": []}
    rows = score_gw(
        gw, lineups, live, bootstrap, read_scores(csv_path), bundle_id=bundle_id, missing_reason=reason
    )
    append_scores(csv_path, rows)
    return rows


# ============================================================================================ app
def summary(df: pd.DataFrame | None) -> JSON | None:
    """For the app: {gw, table (latest GW per policy), cumulative (GW x policy, ascending GW)} or None."""
    if df is None or df.empty or "gw" not in df:
        return None
    d = df.copy()
    d["gw"] = pd.to_numeric(d["gw"], errors="coerce")
    d = d.dropna(subset=["gw"])
    if d.empty:
        return None
    d["gw"] = d["gw"].astype(int)
    gw = int(d["gw"].max())
    labels = {p.name: p.label for p in TRAJECTORIES}
    order = [p.name for p in TRAJECTORIES]
    last = d[d["gw"] == gw].set_index("policy").reindex(order)
    table = pd.DataFrame(
        {
            "Policy": [labels[n] for n in order],
            "Points this GW": pd.to_numeric(last["net"], errors="coerce").to_numpy(),
            "Cumulative": pd.to_numeric(last["cumulative_net"], errors="coerce").to_numpy(),
            "Transfers (total)": pd.to_numeric(last["transfers_total"], errors="coerce").to_numpy(),
            "Hits (total)": pd.to_numeric(last["hits_total"], errors="coerce").to_numpy(),
            "vs average (cumulative)": pd.to_numeric(
                last["cumulative_vs_average"], errors="coerce"
            ).to_numpy(),
            "Status": last["status"].fillna("").to_numpy(),
        }
    )
    cum = (
        d.assign(cumulative_net=pd.to_numeric(d["cumulative_net"], errors="coerce"))
        .pivot_table(index="gw", columns="policy", values="cumulative_net", aggfunc="last")
        .reindex(columns=[n for n in order if n in set(d["policy"])])
        .sort_index()
        .rename(columns=labels)
    )
    cum.index.name = "GW"
    flagged = d[(d["gw"] == gw) & d["flagged"].astype(str).str.lower().isin(("true", "1"))]
    return {
        "gw": gw,
        "table": table,
        "cumulative": cum,
        "latest": last.reset_index(),
        "flagged": [labels.get(p, p) for p in flagged["policy"]],
    }


def parse_ids(v: Any) -> list[int]:
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan"):
        return []
    return [int(float(x)) for x in str(v).split(",") if x.strip()]
