"""FPL game rules for the 2025-26 season, encoded as data (fpl_rocket.rules.common holds the shared types).

Scoring: goalkeeper goal 10; defensive contributions from this season (DEF CBI + tackles >= 10 -> 2, MID/FWD
CBI + tackles + recoveries >= 12 -> 2, never GK; bootstrap game_config.scoring.defensive_contribution).
Transfers: up to 5 banked, kept through a wildcard / free hit ("carry").
Chips: two of every chip, one set per half-season (bootstrap): wildcard GW2-19 / GW20-38, free hit GW2-19 /
GW20-38, bench boost GW1-19 / GW20-38, triple captain GW1-19 / GW20-38.
AFCON top-up: "Your total number of free transfers is topped up to the maximum possible number of five
immediately after the Gameweek 15 deadline" -> 5 free transfers entering GW16
(https://www.premierleague.com/en/news/4461660, https://www.premierleague.com/en/news/4362102).
"""

from __future__ import annotations

from .common import (
    FORMATION,
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

__all__ = ["chip_windows", "fixture_stats_from_explain"]

SEASON = "2025-26"
RULES_VERSION = "2025-26.1"

TRANSFERS = TransferRules(max_bank=5, chip_week_policy="carry", transfers_cap=20)
SCORING = ScoringTable()
SQUAD = SquadRules()
_API = "data/asknown/rules/2025-26.json (bootstrap chips)"
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
SPECIAL = SpecialEvents(ft_topup={16: 5})
SOURCES = {
    "chips": _API,
    "AFCON top-up to 5 entering GW16": "https://www.premierleague.com/en/news/4461660",
    "AFCON extra transfers announced pre-season": "https://www.premierleague.com/en/news/4362102",
    "defensive contributions": _API,
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
