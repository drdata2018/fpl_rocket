"""FPL game rules for the 2026-27 season, encoded as data.

The FPL API publishes no scoring table, so the table lives here and is verified by recomputing every
player's GW points from `event/{gw}/live` stats (scripts/reconcile_points.py, tests/test_rules.py).

Scoring (per fixture; a double gameweek is scored fixture by fixture and summed):
- Appearance: 1 point for 1-59 minutes, 2 points for 60+ minutes, nothing at all if minutes == 0.
- Goal: GK 10, DEF 6, MID 5, FWD 4 (GK 10 from 2024-25: game_config.scoring.goals_scored.GKP = 10 in every
  2026-27 bootstrap; https://premierleague.com/news/4059044 - corrected in step 4b, it was 6).  Assist: 3.
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

Chips: windows (two sets per half-season) are read from bootstrap['chips'] for the live app (chip_windows);
the replay uses CHIPS below = the same windows as recorded in data/asknown/rules/2026-27.json.
"""

from __future__ import annotations

from .common import (
    DEF,
    FORMATION,
    FWD,
    GK,
    MID,
    POSITIONS,
    ChipWindow,
    ScoringTable,
    SeasonRules,
    SpecialEvents,
    SquadRules,
    TransferRules,
    chip_windows,
    fixture_stats_from_explain,
    season_functions,
)

__all__ = [
    "DEF",
    "FWD",
    "GK",
    "MID",
    "POSITIONS",
    "ScoringTable",
    "SquadRules",
    "TransferRules",
    "chip_windows",
    "fixture_stats_from_explain",
]

SEASON = "2026-27"
RULES_VERSION = "2026-27.2"

SCORING = ScoringTable()
TRANSFERS = TransferRules()
SQUAD = SquadRules()
_API = "data/asknown/rules/2026-27.json (bootstrap chips)"
CHIPS = (
    ChipWindow("wildcard", 1, 2, 19, source=_API),
    ChipWindow("wildcard", 2, 20, 38, source=_API),
    ChipWindow("freehit", 1, 2, 19, source=_API),
    ChipWindow("freehit", 2, 20, 38, source=_API),
    ChipWindow("bboost", 1, 1, 19, source=_API),
    ChipWindow("bboost", 2, 20, 38, source=_API),
    ChipWindow("3xc", 1, 1, 19, source=_API),
    ChipWindow("3xc", 2, 20, 38, source=_API),
)
SPECIAL = SpecialEvents()
SOURCES = {
    "scoring": "event/{gw}/live reconciliation (tests/test_rules.py, scripts/reconcile_points.py)",
    "gk_goal_10": "https://premierleague.com/news/4059044 (2024-25 change, unchanged since)",
    "chips": _API,
    "chip_week_policy carry": "scripts/ft_hypotheses.py on 300 sampled managers (2026-10-23)",
}
KNOWN_UNCERTAIN: tuple[str, ...] = ()
RULES = SeasonRules(
    SEASON,
    RULES_VERSION,
    SCORING,
    TRANSFERS,
    SQUAD,
    CHIPS,
    SPECIAL,
    dict(FORMATION),
    sources=SOURCES,
    known_uncertain=KNOWN_UNCERTAIN,
)
points_from_stats, points_from_live, selling_price, transfer_rules_from_bootstrap = season_functions(
    SCORING, TRANSFERS
)
