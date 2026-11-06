"""FPL game rules for the 2024-25 season, encoded as data (fpl_rocket.rules.common holds the shared types).

Scoring: goalkeeper goal 10 from this season (https://premierleague.com/news/4059044; bootstrap
game_config.scoring.goals_scored.GKP = 10 from GW16), no defensive contributions.
Transfers: up to 5 free transfers banked (game_settings.max_extra_free_transfers = 4 from GW1, rules
evolution) and banked transfers are kept through a wildcard / free hit, with no +1 for the chip week
("carry"; https://www.premierleague.com/news/4059225; the +1 question was verified on 2026-27 managers).
Chips: bootstrap chip definitions exist from GW16 (rules/evolution.csv): wildcard GW2-19 and GW20-38, free hit
GW2-38, bench boost and triple captain GW1-38; the first wildcard "had to be played before the Gameweek 19
deadline" (https://www.premierleague.com/news/4209923), so the GW16 windows are used for GW1-15 too.
Assistant manager chip GW24-38 (added between GW22 and GW23,
https://premierleague.com/fpl-chips/assistant-manager-explained): UNSUPPORTED - never played by any replay
policy, recorded as such.
"""

from __future__ import annotations

from .common import (
    FORMATION,
    NO_DEFCON,
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

SEASON = "2024-25"
RULES_VERSION = "2024-25.1"

TRANSFERS = TransferRules(max_bank=5, chip_week_policy="carry", transfers_cap=20)
SCORING = ScoringTable(defcon_threshold=dict(NO_DEFCON))
SQUAD = SquadRules()
_API = "data/asknown/rules/2024-25.json + rules/evolution.csv (bootstrap chips from GW16)"
_AM = "https://premierleague.com/fpl-chips/assistant-manager-explained"
CHIPS = (
    ChipWindow("wildcard", 1, 2, 19, source=_API),
    ChipWindow("wildcard", 2, 20, 38, source=_API),
    ChipWindow("freehit", 1, 2, 38, source=_API),
    ChipWindow("bboost", 1, 1, 38, source=_API),
    ChipWindow("3xc", 1, 1, 38, source=_API),
    ChipWindow("manager", 1, 24, 38, announced_at="2024-12-11T00:00:00Z", supported=False, source=_AM),
)
SPECIAL = SpecialEvents()
SOURCES = {
    "chips": _API,
    "first wildcard until GW19": "https://www.premierleague.com/news/4209923",
    "five free transfers, kept through chips": "https://www.premierleague.com/news/4059225",
    "gk goal 10": "https://premierleague.com/news/4059044",
    "assistant manager (unsupported)": _AM,
}
KNOWN_UNCERTAIN = (
    "2024-25 GW1-15 chip windows: taken from the first bootstrap that lists chips (GW16); the GW19 wildcard "
    "cut-off is confirmed by the Premier League, the other windows are not separately sourced.",
    "2024-25 assistant manager chip: unsupported (never played; a manager could have gained points from it).",
)
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
