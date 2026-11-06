"""The chip rule shared by every policy (prespecified; thresholds are fixed defaults, NOT tuned).

At each GW t, with the chips the manager holds (as known at T):
1. "Last eligible GW": the held, supported chips are scheduled backwards so that every one of them can still
   be played, one chip per GW, inside its window: chips in order of window end (latest first) and, for the
   same end, reverse priority (triple captain, bench boost, free hit, wildcard) each take the latest free GW
   of their window that is >= t, is not a void GW, allows the chip (no wildcard in a blocked GW) and keeps
   free hits apart (no free hit in the GW right after another free hit, or after the one played at t-1). A
   chip whose scheduled GW is t is FORCED now (otherwise it would be wasted). A chip with no feasible GW is
   lost.
2. Otherwise the first triggered chip in priority order (wildcard, free hit, bench boost, triple captain):
       wildcard   best legal squad's expected points over the next `wc_horizon` (5) GWs minus the current
                  squad's (both: best XI + captain per GW, the squad held, undiscounted) >= wc_gain (12);
                  "best legal squad" = the squad the policy's own wildcard would build: the planner's 5-GW
                  optimum (MINIMAL_MODEL, FORM_RESTRAINED) or the most-owned target (LAGGED_CROWD), valued
                  with the policy's trigger predictor
       free hit   at least fh_blank_players (4) of the 15 have no fixture in GW t as known at T
       bench boost  expected points of the bench (the policy's own line-up without a chip) >= bb_bench_xp (12)
       triple captain  the captain's expected points >= tc_captain_xp (9)
   using the policy's own predictor (HOLD and LAGGED_CROWD: the form baseline).
One chip per GW. Chips are never played in a void GW (2022-23 GW7: they would be reinstated).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

from ..rules.common import ChipWindow, SeasonRules

PRIORITY = ("wildcard", "freehit", "bboost", "3xc")


@dataclass(frozen=True)
class ChipRuleConfig:
    bb_bench_xp: float = 12.0
    tc_captain_xp: float = 9.0
    fh_blank_players: int = 4
    wc_gain: float = 12.0
    wc_horizon: int = 5

    def describe(self) -> dict:
        return asdict(self)


CHIP_RULE = ChipRuleConfig()


def schedule(
    rules: SeasonRules, chips: list[ChipWindow], gw: int, last_fh_gw: int | None
) -> dict[str, int | None]:
    """{chip key: the latest GW it can still be played in so that all held chips fit} (rule 1)."""
    held = [c for c in chips if c.supported and c.stop_event >= gw and c.name in PRIORITY]
    held.sort(key=lambda c: (-c.stop_event, -PRIORITY.index(c.name), -c.number))
    taken: dict[int, str] = {}
    out: dict[str, int | None] = {}
    for c in held:
        slot = None
        for g in range(c.stop_event, max(c.start_event, gw) - 1, -1):
            if g in taken or g in rules.special.void_gws or not rules.chip_allowed(c.name, g):
                continue
            if c.name == "freehit" and rules.special.fh_not_consecutive:
                if taken.get(g - 1) == "freehit" or taken.get(g + 1) == "freehit":
                    continue
                if last_fh_gw is not None and g == last_fh_gw + 1:
                    continue
            slot = g
            break
        out[c.key] = slot
        if slot is not None:
            taken[slot] = c.name
    return out


def available(rules: SeasonRules, chips: list[ChipWindow], gw: int, last_fh_gw: int | None) -> list[str]:
    """Chip names playable in `gw` (held, supported, window, allowed, free-hit spacing, not a void GW)."""
    if gw in rules.special.void_gws:
        return []
    out = []
    for name in PRIORITY:
        if not any(c.name == name and c.supported and c.covers(gw) for c in chips):
            continue
        if not rules.chip_allowed(name, gw):
            continue
        if name == "freehit" and rules.special.fh_not_consecutive and last_fh_gw == gw - 1:
            continue
        out.append(name)
    return out


def decide(
    rules: SeasonRules,
    chips: list[ChipWindow],
    gw: int,
    last_fh_gw: int | None,
    triggers: dict[str, Callable[[], tuple[bool, float]]],
    allowed: tuple[str, ...] = PRIORITY,
) -> tuple[str | None, dict]:
    """(chip to play or None, a record of the evaluation). `triggers[name]()` -> (fires, value); evaluated
    lazily in priority order. `allowed` restricts the chips a policy may play (HOLD: no wildcard)."""
    can = [c for c in available(rules, chips, gw, last_fh_gw) if c in allowed]
    sched = schedule(rules, [c for c in chips if c.name in allowed], gw, last_fh_gw)
    forced = [k for k, g in sched.items() if g == gw]
    record: dict = {"available": can, "forced": forced, "values": {}}
    for key in forced:
        name = key.split("#")[0]
        if name in can:
            record["reason"] = "last eligible GW"
            return name, record
    for name in PRIORITY:
        if name not in can or name not in triggers:
            continue
        fires, value = triggers[name]()
        record["values"][name] = round(float(value), 3)
        if fires:
            record["reason"] = "trigger"
            return name, record
    return None, record
