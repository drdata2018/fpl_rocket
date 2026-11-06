"""FPL game rules for the 2022-23 season, encoded as data (fpl_rocket.rules.common holds the shared types).

Scoring as 2021-22 (goalkeeper goal 6, no defensive contributions).
Transfers: max 2 banked, WC / FH reset the balance to 1.
World Cup break (https://www.premierleague.com/news/2667633, https://premierleague.com/news/2668285):
    "unlimited free transfers between ... the Gameweek 16 deadline and before the Gameweek 17 deadline" ->
    GW17 is an unlimited-free-transfer week; "you won't be able to play your Wildcard" in it; "If you still
    have an available Wildcard, you must play that ahead of GW16 deadline" -> wildcard #1 GW2-GW16;
    "The second Wildcard will be available after Monday 26 December 13:30, ahead of Gameweek 18" ->
    wildcard #2 GW18-GW38. Bench Boost, Free Hit and Triple Captain may be played in GW17.
    Free transfers entering GW18: "two transfers going into GW18 only if you don't make any transfers
    during the unlimited transfers period" (community reading, FFS 2022-07-12) -> min(2, f + 1) without
    transfers, else 1 (KNOWN_UNCERTAIN).
GW7 (8-10 September 2022, death of Queen Elizabeth II): every fixture postponed, deadline unchanged, zero
    points; Free Hit / Bench Boost / Triple Captain played for it were reinstated, a Wildcard was not
    cancelled; free transfers "business as usual"
    (https://www.fantasyfootballscout.co.uk/2022/09/09/gameweek-7-fixtures-postponed) -> void GW.
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

SEASON = "2022-23"
RULES_VERSION = "2022-23.1"

# max 2 free transfers banked; a wildcard / free hit loses the banked balance (1 the following GW):
# "any saved free transfers will be lost ... the usual 1 free transfer the following Gameweek"
# https://allaboutfpl.com/2021/07/wildcard-chip-explained-fpl-guide/ ; changed only from 2024-25
# ("Starting from 2024/25, you will keep all the free transfers accumulated before activating a chip",
# https://www.fantasyfootballscout.co.uk/2023/06/01/how-many-fpl-transfers-does-each-manager-get)
TRANSFERS = TransferRules(max_bank=2, chip_week_policy="reset_one", transfers_cap=20)
SCORING = ScoringTable(goal={GK: 6, 2: 6, 3: 5, 4: 4}, defcon_threshold=dict(NO_DEFCON))
SQUAD = SquadRules()
_WC = "https://www.premierleague.com/news/2667633"
CHIPS = (
    ChipWindow("wildcard", 1, 2, 16, source=_WC),
    ChipWindow("wildcard", 2, 18, 38, source=_WC),
    ChipWindow("freehit", 1, 2, 38),
    ChipWindow("bboost", 1, 1, 38),
    ChipWindow("3xc", 1, 1, 38),
)
SPECIAL = SpecialEvents(
    unlimited_free_transfer_gws=frozenset({17}),
    wildcard_blocked_gws=frozenset({17}),
    void_gws=frozenset({7}),
)
SOURCES = {
    "world cup break (GW17 unlimited, wildcard windows)": _WC,
    "world cup break FAQ": "https://premierleague.com/news/2668285",
    "GW7 postponed": "https://www.fantasyfootballscout.co.uk/2022/09/09/gameweek-7-fixtures-postponed",
    "free transfers (max 2, lost after WC/FH)": "https://allaboutfpl.com/2021/07/wildcard-chip-explained-fpl-guide/",
}
KNOWN_UNCERTAIN = (
    "2022-23 GW17 (unlimited transfers): free transfers entering GW18 = min(2, f+1) if no transfer was made "
    "in GW17, else 1 (community reading, not an official statement).",
    "2022-23 GW7 (void): a Free Hit played there is reinstated and its squad discarded; free transfers "
    "accrue as usual (the official statement did not spell out free transfers).",
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
