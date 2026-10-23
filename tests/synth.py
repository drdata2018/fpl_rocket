"""Synthetic FPL world for state-transition tests (no network)."""

from __future__ import annotations

from typing import Any

N_TEAMS = 20
# per team: 1 GK, 2 DEF, 2 MID, 1 FWD -> ids team*6 + 1..6
LAYOUT = (1, 2, 2, 3, 3, 4)
BASE_PRICE = 50


def pid(team: int, slot: int) -> int:
    """Player id for team (0-based) and slot (0..5 in LAYOUT)."""
    return team * 6 + slot + 1


def bootstrap(upcoming: int, prices: dict[int, int] | None = None) -> dict[str, Any]:
    prices = prices or {}
    elements = [
        {"id": pid(t, s), "element_type": et, "team": t + 1, "now_cost": prices.get(pid(t, s), BASE_PRICE)}
        for t in range(N_TEAMS)
        for s, et in enumerate(LAYOUT)
    ]
    events = [
        {"id": g, "deadline_time": f"2026-W{g:02d}", "is_next": g == upcoming, "finished": g < upcoming}
        for g in range(1, 39)
    ]
    chips = []
    for name, first_start in (("wildcard", 2), ("freehit", 2), ("bboost", 1), ("3xc", 1)):
        chips.append({"name": name, "start_event": first_start, "stop_event": 19, "chip_type": "x"})
        chips.append({"name": name, "start_event": 20, "stop_event": 38, "chip_type": "x"})
    return {
        "elements": elements,
        "events": events,
        "chips": chips,
        "game_settings": {"max_extra_free_transfers": 4, "transfers_cap": 20, "transfers_sell_on_fee": 0.5},
    }


def initial_squad() -> list[int]:
    """Legal 15: GKs from teams 0-1, DEFs 2-6, MIDs 7-11, FWDs 12-14 (one per club)."""
    return (
        [pid(t, 0) for t in (0, 1)]
        + [pid(t, 1) for t in range(2, 7)]
        + [pid(t, 3) for t in range(7, 12)]
        + [pid(t, 5) for t in range(12, 15)]
    )


def picks(ids: list[int], gw: int, bank: int = 0, chip: str | None = None) -> dict[str, Any]:
    return {
        "active_chip": chip,
        "entry_history": {"event": gw, "bank": bank},
        "picks": [{"element": e, "position": i + 1, "multiplier": 1} for i, e in enumerate(ids)],
    }


def row(gw: int, n: int = 0, cost: int = 0, bank: int = 0, value: int = 1000) -> dict[str, Any]:
    return {"event": gw, "event_transfers": n, "event_transfers_cost": cost, "bank": bank, "value": value}


def transfer(
    gw: int, out_id: int, in_id: int, out_cost: int = BASE_PRICE, in_cost: int = BASE_PRICE, seq: int = 0
) -> dict[str, Any]:
    return {
        "element_out": out_id,
        "element_in": in_id,
        "element_out_cost": out_cost,
        "element_in_cost": in_cost,
        "event": gw,
        "entry": 1,
        "time": f"2026-W{gw:02d}-{seq:03d}",
    }


def entry(started: int = 1, entry_id: int = 1) -> dict[str, Any]:
    return {"id": entry_id, "started_event": started}


def history(rows: list[dict[str, Any]], chips: list[tuple[str, int]] = ()) -> dict[str, Any]:
    return {"current": rows, "past": [], "chips": [{"name": n, "event": g, "time": ""} for n, g in chips]}


def initial_values(ids: list[int], price: int = BASE_PRICE) -> dict[int, int]:
    return dict.fromkeys(ids, price)
