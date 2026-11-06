"""FPL game rules for the 2023-24 season, encoded as data (fpl_rocket.rules.common holds the shared types).

Scoring as 2021-22 (goalkeeper goal 6, no defensive contributions).
Transfers: max 2 banked, WC / FH reset the balance to 1 (changed only from 2024-25).
Chips: "The first Wildcard will be available from the start of the season until the Gameweek 20 deadline on
Saturday 30 December"; the second "is immediately available ... after Saturday 30 December 11:00 GMT" ->
wildcard #1 GW2-GW20, #2 GW21-GW38 (https://www.premierleague.com/news/3841904,
https://www.premierleague.com/news/3568527); free hit GW2-GW38, bench boost and triple captain GW1-GW38.
"""

from __future__ import annotations

from .common import (
    FORMATION,
    GK,
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

SEASON = "2023-24"
RULES_VERSION = "2023-24.1"

# max 2 free transfers banked; a wildcard / free hit loses the banked balance (1 the following GW):
# "any saved free transfers will be lost ... the usual 1 free transfer the following Gameweek"
# https://allaboutfpl.com/2021/07/wildcard-chip-explained-fpl-guide/ ; changed only from 2024-25
# ("Starting from 2024/25, you will keep all the free transfers accumulated before activating a chip",
# https://www.fantasyfootballscout.co.uk/2023/06/01/how-many-fpl-transfers-does-each-manager-get)
TRANSFERS = TransferRules(max_bank=2, chip_week_policy="reset_one", transfers_cap=20)
SCORING = ScoringTable(goal={GK: 6, 2: 6, 3: 5, 4: 4}, defcon_threshold=dict(NO_DEFCON))
SQUAD = SquadRules()
_WC = "https://www.premierleague.com/news/3841904"
CHIPS = (
    ChipWindow("wildcard", 1, 2, 20, source=_WC),
    ChipWindow("wildcard", 2, 21, 38, source=_WC),
    ChipWindow("freehit", 1, 2, 38),
    ChipWindow("bboost", 1, 1, 38),
    ChipWindow("3xc", 1, 1, 38),
)
SPECIAL = SpecialEvents()
SOURCES = {
    "wildcard windows": _WC,
    "season launch": "https://www.premierleague.com/news/3568527",
    "free transfers (max 2, lost after WC/FH)": "https://allaboutfpl.com/2021/07/wildcard-chip-explained-fpl-guide/",
}
KNOWN_UNCERTAIN = ("2023-24 Free Hit window GW2-GW38 (standard rule; no 2023-24-specific source found).",)
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
