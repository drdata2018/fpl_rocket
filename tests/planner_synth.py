"""Synthetic planner inputs: a deterministic league of 20 clubs (2 GK, 5 DEF, 5 MID, 3 FWD each) with prices
and per-GW xP, plus a legal starting squad."""

from __future__ import annotations

import random

from fpl_rocket.planner import Candidate, Owned, PlannerInput

CLUBS = 20
SHAPE = {1: 2, 2: 5, 3: 5, 4: 3}
BASE_PRICE = {1: 45, 2: 45, 3: 55, 4: 60}


def league(seed: int = 7, gws: tuple[int, ...] = (1, 2, 3, 4, 5)) -> tuple[dict[int, Candidate], dict]:
    """(pool, xp): xp grows with price plus noise, varies by GW."""
    rng = random.Random(seed)
    pool: dict[int, Candidate] = {}
    xp: dict[tuple[int, int], float] = {}
    pid = 1
    for club in range(1, CLUBS + 1):
        for et, n in SHAPE.items():
            for _ in range(n):
                price = BASE_PRICE[et] + rng.randint(0, 70 if et > 1 else 15)
                pool[pid] = Candidate(pid, et, club, price)
                quality = (price - BASE_PRICE[et]) / 20 + rng.random()
                for g in gws:
                    xp[pid, g] = round(max(0.0, 1.0 + quality + rng.gauss(0, 0.6)), 3)
                pid += 1
    return pool, xp


def cheap_squad(pool: dict[int, Candidate], avoid_clubs: int = 0) -> list[Owned]:
    """The cheapest legal 15 (<= 3 per club), selling price = buy price."""
    out: list[Owned] = []
    per_club: dict[int, int] = {}
    for et, need in {1: 2, 2: 5, 3: 5, 4: 3}.items():
        pos = sorted((c for c in pool.values() if c.element_type == et), key=lambda c: (c.buy, c.element))
        for c in pos:
            if sum(1 for o in out if o.element_type == et) >= need:
                break
            if per_club.get(c.team, 0) >= 3 or c.team <= avoid_clubs:
                continue
            out.append(Owned(c.element, et, c.team, c.buy))
            per_club[c.team] = per_club.get(c.team, 0) + 1
    return out


def planner_input(
    *,
    seed: int = 7,
    bank: int = 50,
    free_transfers: int = 1,
    squad: list[Owned] | None = None,
    gws: tuple[int, ...] = (1, 2, 3, 4, 5),
    chips: list[dict] | None = None,
    low_history: frozenset[int] = frozenset(),
) -> PlannerInput:
    pool, xp = league(seed, gws)
    return PlannerInput(
        squad=squad if squad is not None else cheap_squad(pool),
        bank=bank,
        free_transfers=free_transfers,
        gws=list(gws),
        xp=xp,
        pool=pool,
        low_history=low_history,
        chips_available=chips or [],
        names={e: f"P{e}" for e in pool},
    )
