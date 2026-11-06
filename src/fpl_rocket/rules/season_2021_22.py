"""FPL game rules for the 2021-22 season, encoded as data (fpl_rocket.rules.common holds the shared types).

Scoring: goalkeeper goal 6 (raised to 10 only from 2024-25, https://premierleague.com/news/4059044), no
defensive contributions (from 2025-26). Reconciled against the archive (scripts/reconcile_seasons.py).
Transfers: at most 2 free transfers banked; a wildcard / free hit resets the balance to 1 the next GW.
game_settings.transfers_cap changed 1000 -> 20 between GW23 and GW24 (rules/evolution.csv): encoded as 20 (no
policy ever makes more than 5 transfers outside a chip week, so the earlier 1000 cannot bind).
Chips (bootstrap had no chip definitions before 2024-25 GW16, so they come from documentation):
    wildcard #1 GW2-GW20: "The first Wildcard can be activated at any time up until 13:30 ... on 28
        December" (= the GW20 deadline, deadlines.parquet), https://www.premierleague.com/news/2207891
    wildcard #2 GW21-GW38: "The second Wildcard, which becomes available from Gameweek 21",
        https://www.premierleague.com/news/2207891
    free hit #1 GW2-GW38 (never in GW1: the first deadline is unlimited anyway)
    free hit #2 GW20-GW38: the extra Free Hit given on 22 December 2021 (COVID postponements), usable "from
        Gameweek 20 onwards"; an unused first Free Hit is kept; Free Hits cannot be played in consecutive
        GWs. https://www.fantasyfootballscout.co.uk/2021/12/22/fpl-managers-to-be-given-extra-free-hit-chip-from-gameweek-20
        (announced_at 2021-12-22: invisible to a policy before that time)
    bench boost GW1-GW38, triple captain GW1-GW38; one chip per GW.
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

SEASON = "2021-22"
RULES_VERSION = "2021-22.1"

# max 2 free transfers banked; a wildcard / free hit loses the banked balance (1 the following GW):
# "any saved free transfers will be lost ... the usual 1 free transfer the following Gameweek"
# https://allaboutfpl.com/2021/07/wildcard-chip-explained-fpl-guide/ ; changed only from 2024-25
# ("Starting from 2024/25, you will keep all the free transfers accumulated before activating a chip",
# https://www.fantasyfootballscout.co.uk/2023/06/01/how-many-fpl-transfers-does-each-manager-get)
TRANSFERS = TransferRules(max_bank=2, chip_week_policy="reset_one", transfers_cap=20)
SCORING = ScoringTable(goal={GK: 6, 2: 6, 3: 5, 4: 4}, defcon_threshold=dict(NO_DEFCON))
SQUAD = SquadRules()
_PL = "https://www.premierleague.com/news/2207891"
_FH2 = (
    "https://www.fantasyfootballscout.co.uk/2021/12/22/"
    "fpl-managers-to-be-given-extra-free-hit-chip-from-gameweek-20"
)
CHIPS = (
    ChipWindow("wildcard", 1, 2, 20, source=_PL),
    ChipWindow("wildcard", 2, 21, 38, source=_PL),
    ChipWindow("freehit", 1, 2, 38, source="FPL standard: not in GW1"),
    ChipWindow("freehit", 2, 20, 38, announced_at="2021-12-22T00:00:00Z", source=_FH2),
    ChipWindow("bboost", 1, 1, 38),
    ChipWindow("3xc", 1, 1, 38),
)
SPECIAL = SpecialEvents()
SOURCES = {
    "wildcard windows": _PL,
    "extra free hit": _FH2,
    "free transfers (max 2, lost after WC/FH)": "https://allaboutfpl.com/2021/07/wildcard-chip-explained-fpl-guide/",
    "gk goal 6 before 2024-25": "https://premierleague.com/news/4059044",
}
KNOWN_UNCERTAIN = (
    "2021-22 extra Free Hit: announcement time taken as 2021-12-22 00:00 UTC (article date); its window end "
    "is taken as GW38 (no source states an end).",
    "2021-22 Free Hit #1 window GW2-GW38 (standard rule; no 2021-22-specific source found).",
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
