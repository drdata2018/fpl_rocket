"""Planner inputs from the app's objects: PlanningState (+ session-only user edits), players frame, xp table.

The xp table is the bundle's xp.parquet (one row per element and GW). Column `low_history` (newcomers with
unreliable predictions) may be absent: treated as all False. Free transfers use the LOW end of the planning
state's range unless the user edits it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..rules import get_rules
from .core import Candidate, Owned, PlannerInput
from .lineup import FORMATION, formation_from_bootstrap


@dataclass(frozen=True)
class StateEdits:
    """Session-only corrections of the planning state (None = keep the reconstructed value)."""

    bank: int | None = None  # tenths
    free_transfers: int | None = None
    squad: tuple[int, ...] | None = None  # the corrected 15 element ids

    def is_empty(self) -> bool:
        return self.bank is None and self.free_transfers is None and self.squad is None


def low_history_set(xp: pd.DataFrame | None) -> frozenset[int]:
    if xp is None or xp.empty or "low_history" not in xp:
        return frozenset()
    flag = xp["low_history"].fillna(False).astype(bool)
    return frozenset(int(e) for e in xp.loc[flag, "element"].unique())


def xp_lookup(xp: pd.DataFrame, gws: list[int] | None = None) -> dict[tuple[int, int], float]:
    x = xp if gws is None else xp[xp["gw"].isin(gws)]
    return {
        (int(e), int(g)): float(v) for e, g, v in zip(x["element"], x["gw"], x["xp"].fillna(0.0), strict=True)
    }


def planning_gws(xp: pd.DataFrame, upcoming_gw: int) -> list[int]:
    return sorted(int(g) for g in xp["gw"].unique() if int(g) >= int(upcoming_gw))


def _names(players: pd.DataFrame) -> dict[int, str]:
    return {
        int(i): f"{n} ({t})" for i, n, t in zip(players["id"], players["name"], players["team"], strict=True)
    }


def from_planning_state(
    state: Any,
    players: pd.DataFrame,
    xp: pd.DataFrame,
    *,
    bootstrap: dict[str, Any] | None = None,
    edits: StateEdits | None = None,
) -> PlannerInput:
    """PlannerInput for the state's upcoming GW. `players` = views.players_frame (price in millions)."""
    edits = edits or StateEdits()
    rules = get_rules()
    transfer_rules = rules.transfer_rules_from_bootstrap(bootstrap) if bootstrap else rules.TRANSFERS
    idx = players.set_index("id")
    pool = {
        int(i): Candidate(int(i), int(r["element_type"]), int(r["team_id"]), round(float(r["price"]) * 10))
        for i, r in idx.iterrows()
        if int(r["element_type"]) in (1, 2, 3, 4)
    }
    owned = {sp.element: Owned(sp.element, sp.element_type, sp.team, sp.selling_price) for sp in state.squad}
    names = _names(players)
    notes: list[str] = []
    squad = list(owned.values())
    if edits.squad is not None:
        wanted = [int(e) for e in edits.squad]
        added = [e for e in wanted if e not in owned]
        removed = [e for e in owned if e not in wanted]
        squad = [
            owned[e] if e in owned else Owned(e, pool[e].element_type, pool[e].team, pool[e].buy)
            for e in wanted
            if e in owned or e in pool
        ]
        if added or removed:
            notes.append(
                "User-edited squad: "
                + (f"added {', '.join(names.get(e, str(e)) for e in added)}" if added else "")
                + ("; " if added and removed else "")
                + (f"removed {', '.join(names.get(e, str(e)) for e in removed)}" if removed else "")
                + (" (added players valued at their current price)" if added else "")
                + "."
            )
    bank = state.bank
    if edits.bank is not None and edits.bank != state.bank:
        notes.append(f"User-edited bank: {edits.bank / 10:.1f} (reconstructed {state.bank / 10:.1f}).")
        bank = edits.bank
    ft = state.free_transfers.lo
    if edits.free_transfers is not None and edits.free_transfers != ft:
        notes.append(
            f"User-edited free transfers: {edits.free_transfers} (reconstructed {state.free_transfers})."
        )
        ft = edits.free_transfers
    elif not state.free_transfers.exact:
        notes.append(f"Free transfers {state.free_transfers} (a range): the plan uses the low end, {ft}.")
    gws = planning_gws(xp, state.upcoming_gw)
    return PlannerInput(
        squad=squad,
        bank=int(bank),
        free_transfers=int(ft),
        gws=gws,
        xp=xp_lookup(xp, gws),
        pool=pool,
        low_history=low_history_set(xp),
        chips_available=list(state.chips_available),
        names=names,
        transfer_rules=transfer_rules,
        formation=formation_from_bootstrap(bootstrap) if bootstrap else dict(FORMATION),
        notes=notes,
    )


def plan_key(inp: PlannerInput) -> tuple:
    """Hashable digest of everything a plan depends on (for caching by planning-state hash)."""
    return (
        tuple(sorted((o.element, o.selling_price) for o in inp.squad)),
        inp.bank,
        inp.free_transfers,
        tuple(inp.gws),
        tuple(sorted((c["name"], int(c["start_event"]), int(c["stop_event"])) for c in inp.chips_available)),
        tuple(inp.notes),
    )
