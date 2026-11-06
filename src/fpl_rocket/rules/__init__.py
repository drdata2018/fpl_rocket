"""Versioned FPL rules, one module per season with the same interface (fpl_rocket.rules.common).
Use get_rules(season) rather than importing a season module directly; season_rules(season) returns its
SeasonRules (the state machine used by the blind replay, fpl_rocket.replay)."""

from __future__ import annotations

from types import ModuleType

from . import season_2021_22, season_2022_23, season_2023_24, season_2024_25, season_2025_26, season_2026_27
from .common import SeasonRules

CURRENT_SEASON = "2026-27"
_SEASONS: dict[str, ModuleType] = {
    m.SEASON: m
    for m in (season_2021_22, season_2022_23, season_2023_24, season_2024_25, season_2025_26, season_2026_27)
}
SEASONS = tuple(sorted(_SEASONS))


def get_rules(season: str = CURRENT_SEASON) -> ModuleType:
    """Return the rules module for a season like '2026-27' (also accepts '2026_27')."""
    key = season.replace("_", "-")
    try:
        return _SEASONS[key]
    except KeyError as exc:
        raise ValueError(f"no rules encoded for season {season!r}; known: {sorted(_SEASONS)}") from exc


def season_rules(season: str) -> SeasonRules:
    return get_rules(season).RULES
