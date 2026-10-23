"""FPL game rules for the 2026-27 season, encoded as data.

The FPL API publishes no scoring table, so the table lives here and is verified by recomputing every
player's GW points from `event/{gw}/live` stats (scripts/reconcile_points.py, tests/test_rules.py).

Scoring (per fixture; a double gameweek is scored fixture by fixture and summed):
- Appearance: 1 point for 1-59 minutes, 2 points for 60+ minutes, nothing at all if minutes == 0.
- Goal: GK/DEF 6, MID 5, FWD 4.  Assist: 3.
- Clean sheet (requires 60+ minutes): GK/DEF 4, MID 1, FWD 0.
- Every 3 saves: 1 (GK).  Penalty saved: 5.  Penalty missed: -2.
- Every 2 goals conceded: -1 (GK/DEF).  Yellow: -1.  Red: -3.  Own goal: -2.
- Bonus: as awarded (0-3); it is its own stat and is part of total_points.
- Defensive contribution (DefCon), 2 points max per fixture, never for GK:
    DEF:     clearances_blocks_interceptions + tackles              >= 10 -> 2
    MID/FWD: clearances_blocks_interceptions + tackles + recoveries >= 12 -> 2
  The live stat `defensive_contribution` holds that RAW COUNT for the player's position (verified on GW7,
  2026-10-23: a DEF with CBI 10 + tackles 2 shows 12 and scores 2; a MID showing 11 scores 0), NOT the
  awarded points, so the threshold is applied to it here.

Transfers:
- Unlimited free changes before a manager's first deadline (GW1, or started_event for late entrants).
- +1 free transfer per GW after that, bank capped at 5 (= 1 + game_settings.max_extra_free_transfers).
- Each transfer beyond the balance costs 4 points.
- Wildcard / free-hit weeks: transfers are free and do not consume the balance, but the chip week does NOT
  earn the usual +1: the balance is carried over unchanged. Verified 2026-10-23 on 300 sampled managers
  (scripts/ft_hypotheses.py): "carry" 0 contradictions, "carry +1" 7, "reset to 1" 54. Example: 1 FT going
  into a GW3 wildcard, then 5 transfers in GW4 cost 16 (=> 1 FT, not 2).
- Free hit: the squad reverts after the GW.
- Selling price = purchase + floor((current - purchase) / 2) when current > purchase, else current (tenths).

Chips: windows (two sets per half-season) are read from bootstrap['chips'], never hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SEASON = "2026-27"
RULES_VERSION = "2026-27.1"

GK, DEF, MID, FWD = 1, 2, 3, 4
POSITIONS = {GK: "GKP", DEF: "DEF", MID: "MID", FWD: "FWD"}


@dataclass(frozen=True)
class ScoringTable:
    """Points per event, keyed by element_type where position-dependent."""

    appearance_short: int = 1  # 1..59 minutes
    appearance_long: int = 2  # >= long_minutes
    long_minutes: int = 60
    goal: dict[int, int] = field(default_factory=lambda: {GK: 6, DEF: 6, MID: 5, FWD: 4})
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


@dataclass(frozen=True)
class TransferRules:
    """Free-transfer state machine parameters."""

    free_per_gw: int = 1
    max_bank: int = 5  # 1 + game_settings.max_extra_free_transfers (4)
    hit_cost: int = 4
    transfers_cap: int = 20  # game_settings.transfers_cap
    chip_weeks_free: frozenset[str] = frozenset({"wildcard", "freehit"})
    # balance entering the GW after a WC/FH week:
    #   "carry"        -> same balance as entering the chip week (no +1)       <- observed 2026-27
    #   "carry_accrue" -> balance + 1 (capped)
    #   "reset_one"    -> 1 (pre-2024 rule)
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


SCORING = ScoringTable()
TRANSFERS = TransferRules()
SQUAD = SquadRules()


def _i(stats: dict[str, Any], key: str) -> int:
    return int(stats.get(key, 0) or 0)


def points_from_stats(
    stats: dict[str, Any], element_type: int, table: ScoringTable = SCORING
) -> dict[str, int]:
    """Score ONE fixture's stats for a player of `element_type`; returns a breakdown with 'total'.

    `stats` uses the live-endpoint identifiers (minutes, goals_scored, assists, clean_sheets, saves,
    penalties_saved, penalties_missed, goals_conceded, yellow_cards, red_cards, own_goals, bonus,
    defensive_contribution). For a double gameweek use points_from_live (fixture by fixture).
    """
    minutes = _i(stats, "minutes")
    if minutes <= 0:
        return {"total": 0}
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


def points_from_live(
    live_element: dict[str, Any], element_type: int, table: ScoringTable = SCORING
) -> dict[str, int]:
    """Score one element of event/{gw}/live, fixture by fixture when it played more than once."""
    explain = live_element.get("explain") or []
    if len(explain) <= 1:
        return points_from_stats(live_element["stats"], element_type, table)
    total: dict[str, int] = {}
    for fx in explain:
        for k, v in points_from_stats(fixture_stats_from_explain(fx), element_type, table).items():
            total[k] = total.get(k, 0) + v
    return total


def selling_price(purchase: int, current: int, sell_on_fee: float = TRANSFERS.sell_on_fee) -> int:
    """Selling price in tenths: purchase + floor(fee * gain) when the price rose, else the current price."""
    if current > purchase:
        return purchase + int((current - purchase) * sell_on_fee)
    return current


def chip_windows(bootstrap: dict[str, Any]) -> list[dict[str, Any]]:
    """Chip windows from bootstrap['chips'] (two sets per half-season): name, start/stop event, type."""
    keys = ("id", "name", "number", "start_event", "stop_event", "chip_type")
    return [{k: c.get(k) for k in keys} for c in bootstrap.get("chips", [])]


def transfer_rules_from_bootstrap(bootstrap: dict[str, Any]) -> TransferRules:
    """TransferRules with the values discoverable from game_settings overriding the defaults."""
    gs = bootstrap.get("game_settings", {})
    return TransferRules(
        max_bank=1 + int(gs.get("max_extra_free_transfers", TRANSFERS.max_bank - 1)),
        transfers_cap=int(gs.get("transfers_cap", TRANSFERS.transfers_cap)),
        sell_on_fee=float(gs.get("transfers_sell_on_fee", TRANSFERS.sell_on_fee)),
    )
