"""The season-independent part of the FPL rules engine: the data types every season module fills in, and the
functions that apply them (scoring, selling price, the free-transfer state machine, chip windows).

Each season module (season_2021_22 ... season_2026_27) exposes the SAME interface:
    SEASON, RULES_VERSION, SCORING, TRANSFERS, SQUAD, FORMATION, CHIPS, SPECIAL, RULES (a SeasonRules),
    SOURCES, KNOWN_UNCERTAIN, points_from_stats, points_from_live, fixture_stats_from_explain,
    selling_price, chip_windows, transfer_rules_from_bootstrap.

Free-transfer state machine (SeasonRules.next_free_transfers; the balance ENTERING a gameweek):
    GW1 (the first deadline): unlimited free changes; 1 free transfer entering GW2.
    normal week:   f' = min(cap, f - min(n, f) + free_per_gw); transfers beyond f cost hit_cost each.
    wildcard / free-hit week: transfers free; f' by TransferRules.chip_week_policy:
        "reset_one" (before 2024-25: banked transfers are lost, 1 the following week)
        "carry"     (from 2024-25: the banked balance is kept, no +1 for the chip week)
    unlimited-free-transfer week (2022-23 GW17, World Cup break): transfers free; f' = min(cap, f + 1) when
        no transfer was made, else 1 (SPECIAL; see season_2022_23 for the source and the uncertainty).
    top-up (2025-26 GW16, AFCON): the balance entering that GW is raised to the given value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

GK, DEF, MID, FWD = 1, 2, 3, 4
POSITIONS = {GK: "GKP", DEF: "DEF", MID: "MID", FWD: "FWD"}
FORMATION: dict[int, tuple[int, int]] = {GK: (1, 1), DEF: (3, 5), MID: (2, 5), FWD: (1, 3)}

CHIP_NAMES = ("wildcard", "freehit", "bboost", "3xc", "manager")
FREE_TRANSFER_CHIPS = frozenset({"wildcard", "freehit"})


@dataclass(frozen=True)
class ScoringTable:
    """Points per event, keyed by element_type where position-dependent."""

    appearance_short: int = 1  # 1..59 minutes
    appearance_long: int = 2  # >= long_minutes
    long_minutes: int = 60
    goal: dict[int, int] = field(default_factory=lambda: {GK: 10, DEF: 6, MID: 5, FWD: 4})
    assist: int = 3
    clean_sheet: dict[int, int] = field(default_factory=lambda: {GK: 4, DEF: 4, MID: 1, FWD: 0})
    saves_per_point: int = 3
    save_point_positions: frozenset[int] = frozenset({GK})
    penalty_save: int = 5
    penalty_miss: int = -2
    goals_conceded_per_point: int = 2
    goals_conceded_point: int = -1
    goals_conceded_positions: frozenset[int] = frozenset({GK, DEF})
    yellow: int = -1
    red: int = -3
    own_goal: int = -2
    defcon_points: int = 2
    # threshold applied to the live stat `defensive_contribution` (raw count); None = not eligible
    defcon_threshold: dict[int, int | None] = field(
        default_factory=lambda: {GK: None, DEF: 10, MID: 12, FWD: 12}
    )
    # raw stats summed into the count, documented for model features
    defcon_components: dict[int, tuple[str, ...]] = field(
        default_factory=lambda: {
            GK: (),
            DEF: ("clearances_blocks_interceptions", "tackles"),
            MID: ("clearances_blocks_interceptions", "tackles", "recoveries"),
            FWD: ("clearances_blocks_interceptions", "tackles", "recoveries"),
        }
    )


NO_DEFCON: dict[int, int | None] = {GK: None, DEF: None, MID: None, FWD: None}


@dataclass(frozen=True)
class TransferRules:
    """Free-transfer state machine parameters."""

    free_per_gw: int = 1
    max_bank: int = 5  # 1 + game_settings.max_extra_free_transfers (4)
    hit_cost: int = 4
    transfers_cap: int = 20  # game_settings.transfers_cap (most transfers in one GW outside chip weeks)
    chip_weeks_free: frozenset[str] = FREE_TRANSFER_CHIPS
    # balance entering the GW after a WC/FH week: "carry" | "carry_accrue" | "reset_one" (module docstring)
    chip_week_policy: str = "carry"
    reverting_chips: frozenset[str] = frozenset({"freehit"})
    sell_on_fee: float = 0.5  # game_settings.transfers_sell_on_fee


@dataclass(frozen=True)
class SquadRules:
    """Squad composition limits."""

    size: int = 15
    per_position: dict[int, int] = field(default_factory=lambda: {GK: 2, DEF: 5, MID: 5, FWD: 3})
    max_per_team: int = 3
    budget: int = 1000  # tenths


@dataclass(frozen=True)
class ChipWindow:
    """One chip a manager holds: usable in GWs start_event..stop_event (inclusive). `announced_at` (UTC ISO)
    = when the chip became known (a chip added mid-season is invisible to a policy before that time);
    `supported` False = the replay never plays it (recorded as unsupported)."""

    name: str
    number: int
    start_event: int
    stop_event: int
    announced_at: str | None = None
    supported: bool = True
    source: str = ""

    @property
    def key(self) -> str:
        return f"{self.name}#{self.number}@{self.start_event}-{self.stop_event}"

    def covers(self, gw: int) -> bool:
        return self.start_event <= gw <= self.stop_event

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "start_event": self.start_event, "stop_event": self.stop_event}


@dataclass(frozen=True)
class SpecialEvents:
    """Season-specific events that change the state machine."""

    # every transfer free in these GWs (2022-23 GW17); no wildcard may be played there
    unlimited_free_transfer_gws: frozenset[int] = frozenset()
    wildcard_blocked_gws: frozenset[int] = frozenset()
    # every fixture of the GW postponed (2022-23 GW7): 0 points; FH / BB / TC played there are reinstated
    void_gws: frozenset[int] = frozenset()
    # {gw: balance}: free transfers entering `gw` raised to `balance` (2025-26 GW16, AFCON)
    ft_topup: dict[int, int] = field(default_factory=dict)
    fh_not_consecutive: bool = True  # a free hit cannot follow a free hit in the next GW
    one_chip_per_gw: bool = True


@dataclass(frozen=True)
class SeasonRules:
    season: str
    version: str
    scoring: ScoringTable
    transfers: TransferRules
    squad: SquadRules
    chips: tuple[ChipWindow, ...]
    special: SpecialEvents = field(default_factory=SpecialEvents)
    formation: dict[int, tuple[int, int]] = field(default_factory=lambda: dict(FORMATION))
    n_gws: int = 38
    sources: dict[str, str] = field(default_factory=dict)
    known_uncertain: tuple[str, ...] = ()

    # ------------------------------------------------------------------ chips
    def chips_known_at(self, t: pd.Timestamp | None) -> tuple[ChipWindow, ...]:
        """Chips a manager holds as known at time `t` (all when t is None)."""
        if t is None:
            return self.chips
        t = pd.Timestamp(t)
        return tuple(c for c in self.chips if c.announced_at is None or pd.Timestamp(c.announced_at) <= t)

    def chip_allowed(self, chip: str, gw: int) -> bool:
        blocked = self.special.wildcard_blocked_gws | self.special.unlimited_free_transfer_gws
        return not (chip == "wildcard" and gw in blocked)

    # ------------------------------------------------------------------ transfers
    def free_transfer_week(self, gw: int, chip: str | None) -> bool:
        """Every transfer of this GW is free (first deadline, WC / FH, unlimited-transfer week)."""
        return (
            gw == 1
            or (chip in self.transfers.chip_weeks_free)
            or gw in self.special.unlimited_free_transfer_gws
        )

    def hits(self, gw: int, free_in: int, n_transfers: int, chip: str | None) -> int:
        if self.free_transfer_week(gw, chip):
            return 0
        return max(0, int(n_transfers) - int(free_in))

    def next_free_transfers(self, gw: int, free_in: int, n_transfers: int, chip: str | None) -> int:
        """Free transfers entering gw + 1 (module docstring)."""
        tr = self.transfers
        cap = tr.max_bank
        f = min(int(free_in), cap)
        if gw == 1:
            nxt = tr.free_per_gw
        elif chip in tr.chip_weeks_free:
            if tr.chip_week_policy == "carry":
                nxt = f
            elif tr.chip_week_policy == "carry_accrue":
                nxt = min(cap, f + tr.free_per_gw)
            else:
                nxt = tr.free_per_gw
        elif gw in self.special.unlimited_free_transfer_gws:
            nxt = min(cap, f + tr.free_per_gw) if n_transfers == 0 else tr.free_per_gw
        else:
            nxt = min(cap, f - min(int(n_transfers), f) + tr.free_per_gw)
        top = self.special.ft_topup.get(gw + 1)
        if top is not None:
            nxt = max(nxt, min(cap, int(top)))
        return int(nxt)

    def points_from_stats(self, stats: dict[str, Any], element_type: int) -> dict[str, int]:
        return points_from_stats(stats, element_type, self.scoring)


# ============================================================================================ scoring
def _i(stats: dict[str, Any], key: str) -> int:
    v = stats.get(key, 0)
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0 if pd.isna(v) else int(v)


def points_from_stats(stats: dict[str, Any], element_type: int, table: ScoringTable) -> dict[str, int]:
    """Score ONE fixture's stats for a player of `element_type`; returns a breakdown with 'total'.

    `stats` uses the live-endpoint identifiers (minutes, goals_scored, assists, clean_sheets, saves,
    penalties_saved, penalties_missed, goals_conceded, yellow_cards, red_cards, own_goals, bonus,
    defensive_contribution). For a double gameweek use points_from_live (fixture by fixture).
    """
    minutes = _i(stats, "minutes")
    if minutes <= 0:
        # a card shown to an unused substitute still counts (archive reconciliation, step 4b: the only
        # mismatches in 2021-22..2025-26 were bookings of 0-minute players scored -1 / -3)
        cards = {
            "yellow_cards": _i(stats, "yellow_cards") * table.yellow,
            "red_cards": _i(stats, "red_cards") * table.red,
        }
        out0 = {k: v for k, v in cards.items() if v}
        out0["total"] = sum(out0.values())
        return out0
    long_game = minutes >= table.long_minutes
    out: dict[str, int] = {
        "minutes": table.appearance_long if long_game else table.appearance_short,
        "goals_scored": _i(stats, "goals_scored") * table.goal[element_type],
        "assists": _i(stats, "assists") * table.assist,
        "clean_sheets": (_i(stats, "clean_sheets") if long_game else 0) * table.clean_sheet[element_type],
        "penalties_saved": _i(stats, "penalties_saved") * table.penalty_save,
        "penalties_missed": _i(stats, "penalties_missed") * table.penalty_miss,
        "yellow_cards": _i(stats, "yellow_cards") * table.yellow,
        "red_cards": _i(stats, "red_cards") * table.red,
        "own_goals": _i(stats, "own_goals") * table.own_goal,
        "bonus": _i(stats, "bonus"),
    }
    if element_type in table.save_point_positions:
        out["saves"] = _i(stats, "saves") // table.saves_per_point
    if element_type in table.goals_conceded_positions:
        conceded = _i(stats, "goals_conceded") // table.goals_conceded_per_point
        out["goals_conceded"] = conceded * table.goals_conceded_point
    threshold = table.defcon_threshold[element_type]
    if threshold is not None and _i(stats, "defensive_contribution") >= threshold:
        out["defensive_contribution"] = table.defcon_points
    out = {k: v for k, v in out.items() if v}
    out["total"] = sum(out.values())
    return out


def fixture_stats_from_explain(explain_fixture: dict[str, Any]) -> dict[str, int]:
    """Per-fixture raw stat VALUES from one `explain` entry (its points are ignored).

    `explain` lists only stats that scored; an omitted stat is below its scoring threshold, so treating
    it as 0 cannot change the computed points.
    """
    return {s["identifier"]: int(s["value"]) for s in explain_fixture.get("stats", [])}


def points_from_live(live_element: dict[str, Any], element_type: int, table: ScoringTable) -> dict[str, int]:
    """Score one element of event/{gw}/live, fixture by fixture when it played more than once."""
    explain = live_element.get("explain") or []
    if len(explain) <= 1:
        return points_from_stats(live_element["stats"], element_type, table)
    total: dict[str, int] = {}
    for fx in explain:
        for k, v in points_from_stats(fixture_stats_from_explain(fx), element_type, table).items():
            total[k] = total.get(k, 0) + v
    return total


def selling_price(purchase: int, current: int, sell_on_fee: float = 0.5) -> int:
    """Selling price in tenths: purchase + floor(fee * gain) when the price rose, else the current price."""
    if current > purchase:
        return purchase + int((current - purchase) * sell_on_fee)
    return current


def chip_windows(bootstrap: dict[str, Any]) -> list[dict[str, Any]]:
    """Chip windows from bootstrap['chips'] (two sets per half-season): name, start/stop event, type."""
    keys = ("id", "name", "number", "start_event", "stop_event", "chip_type")
    return [{k: c.get(k) for k in keys} for c in bootstrap.get("chips", []) or []]


def transfer_rules_from_bootstrap(bootstrap: dict[str, Any], default: TransferRules) -> TransferRules:
    """TransferRules with the values discoverable from game_settings overriding `default`."""
    gs = bootstrap.get("game_settings", {})
    return TransferRules(
        free_per_gw=default.free_per_gw,
        max_bank=1 + int(gs.get("max_extra_free_transfers", default.max_bank - 1)),
        hit_cost=default.hit_cost,
        transfers_cap=int(gs.get("transfers_cap", default.transfers_cap)),
        chip_weeks_free=default.chip_weeks_free,
        chip_week_policy=default.chip_week_policy,
        reverting_chips=default.reverting_chips,
        sell_on_fee=float(gs.get("transfers_sell_on_fee", default.sell_on_fee)),
    )


def season_functions(scoring: ScoringTable, transfers: TransferRules):
    """The per-season function interface bound to that season's tables."""

    def _points_from_stats(stats: dict[str, Any], element_type: int, table: ScoringTable = scoring):
        return points_from_stats(stats, element_type, table)

    def _points_from_live(live_element: dict[str, Any], element_type: int, table: ScoringTable = scoring):
        return points_from_live(live_element, element_type, table)

    def _selling_price(purchase: int, current: int, sell_on_fee: float = transfers.sell_on_fee) -> int:
        return selling_price(purchase, current, sell_on_fee)

    def _transfer_rules_from_bootstrap(bootstrap: dict[str, Any]) -> TransferRules:
        return transfer_rules_from_bootstrap(bootstrap, transfers)

    return _points_from_stats, _points_from_live, _selling_price, _transfer_rules_from_bootstrap
