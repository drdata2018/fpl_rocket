"""Versioned FPL rules. Use get_rules(season) rather than importing a season module directly."""

from __future__ import annotations

from types import ModuleType

from . import season_2026_27

CURRENT_SEASON = "2026-27"
_SEASONS: dict[str, ModuleType] = {"2026-27": season_2026_27}


def get_rules(season: str = CURRENT_SEASON) -> ModuleType:
    """Return the rules module for a season like '2026-27' (also accepts '2026_27')."""
    key = season.replace("_", "-")
    try:
        return _SEASONS[key]
    except KeyError as exc:
        raise ValueError(f"no rules encoded for season {season!r}; known: {sorted(_SEASONS)}") from exc
