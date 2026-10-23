"""Normalise the community FPL archive (github.com/vaastav/Fantasy-Premier-League) to one stable schema.

Input per season (downloaded by scripts/load_archive.py into data/archive/<season>/): gws/merged_gw.csv,
players_raw.csv, teams.csv, fixtures.csv. The files are UNTRUSTED data: they are parsed with pandas.read_csv
only (selected columns, no eval of any field; the Python-repr `stats` column of fixtures.csv is never read).

Output: one row per (season, gw, element, fixture) with ARCHIVE_COLUMNS. Ids are season-local except `code`
(players_raw.code), the player's stable id across seasons. Columns a season lacks are null:
    defensive_contribution, clearances_blocks_interceptions, recoveries, tackles: only from 2025-26
    expected_goals / expected_assists / expected_goal_involvements / expected_goals_conceded, starts:
        from 2022-23 (every season loaded here has them)
element_type 5 = assistant manager (2024-25 only); model code filters element_type 1-4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26", "2026-27")
FILES = ("gws/merged_gw.csv", "players_raw.csv", "teams.csv", "fixtures.csv")
POSITION_TYPES = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4, "AM": 5}

INT_STATS = (
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
)
FLOAT_STATS = (
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
)
DEFCON_STATS = ("defensive_contribution", "clearances_blocks_interceptions", "recoveries", "tackles")
MARKET = ("value", "selected", "transfers_in", "transfers_out", "total_points")

ARCHIVE_COLUMNS = (
    "season",
    "gw",
    "element",
    "code",
    "player_name",
    "element_type",
    "team",
    "team_id",
    "opponent",
    "opponent_id",
    "was_home",
    *INT_STATS,
    *FLOAT_STATS,
    *DEFCON_STATS,
    *MARKET,
    "kickoff_time",
    "fixture",
)
NULLABLE_INT = "Int64"


def _read_csv(path: Path, usecols=None) -> pd.DataFrame:
    """pandas-only parse of an untrusted CSV (utf-8, falling back to latin-1 for old files)."""
    kw = {"usecols": usecols, "low_memory": False}
    try:
        return pd.read_csv(path, encoding="utf-8", **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kw)


def _bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})


def normalise_season(
    season: str,
    merged: pd.DataFrame,
    players_raw: pd.DataFrame,
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """One season's merged_gw rows in the ARCHIVE_COLUMNS schema (pure: frames in, frame out)."""
    m = merged.copy()
    out = pd.DataFrame(index=m.index)
    out["season"] = season
    gw = m["GW"] if "GW" in m else m["round"]
    out["gw"] = pd.to_numeric(gw, errors="coerce").astype(NULLABLE_INT)
    out["element"] = pd.to_numeric(m["element"], errors="coerce").astype(NULLABLE_INT)
    out["fixture"] = pd.to_numeric(m["fixture"], errors="coerce").astype(NULLABLE_INT)

    pr = players_raw.assign(id=pd.to_numeric(players_raw["id"], errors="coerce")).set_index("id")
    out["code"] = out["element"].map(pd.to_numeric(pr["code"], errors="coerce")).astype(NULLABLE_INT)
    raw_type = out["element"].map(pd.to_numeric(pr["element_type"], errors="coerce"))
    pos_type = m["position"].map(POSITION_TYPES) if "position" in m else np.nan
    out["element_type"] = raw_type.fillna(pos_type).astype(NULLABLE_INT)
    out["player_name"] = m["name"].astype("string") if "name" in m else pd.NA

    names = dict(zip(pd.to_numeric(teams["id"]), teams["name"], strict=True))
    by_name = {v: k for k, v in names.items()}
    out["was_home"] = _bool(m["was_home"]).astype("boolean")
    fx = fixtures.assign(id=pd.to_numeric(fixtures["id"], errors="coerce")).set_index("id")
    team_h = out["fixture"].map(pd.to_numeric(fx["team_h"], errors="coerce"))
    team_a = out["fixture"].map(pd.to_numeric(fx["team_a"], errors="coerce"))
    home = out["was_home"].fillna(False).astype(bool)
    team_id = pd.Series(np.where(home, team_h, team_a), index=out.index)
    if "team" in m:  # fixture lookup first (robust); the merged file's team name fills any gap
        team_id = team_id.fillna(m["team"].map(by_name))
    out["team_id"] = pd.to_numeric(team_id, errors="coerce").astype(NULLABLE_INT)
    out["team"] = out["team_id"].map(names).astype("string")
    out["opponent_id"] = pd.to_numeric(m["opponent_team"], errors="coerce").astype(NULLABLE_INT)
    out["opponent"] = out["opponent_id"].map(names).astype("string")

    for c in (*INT_STATS, *DEFCON_STATS, *MARKET):
        out[c] = pd.to_numeric(m[c], errors="coerce").astype(NULLABLE_INT) if c in m else pd.NA
        out[c] = out[c].astype(NULLABLE_INT)
    for c in FLOAT_STATS:
        out[c] = pd.to_numeric(m[c], errors="coerce").astype("Float64") if c in m else pd.NA
        out[c] = out[c].astype("Float64")
    out["kickoff_time"] = pd.to_datetime(m["kickoff_time"], utc=True, errors="coerce")
    out = out[list(ARCHIVE_COLUMNS)]
    out = out.dropna(subset=["gw", "element"]).drop_duplicates(["season", "gw", "element", "fixture"])
    return out.sort_values(["gw", "element", "fixture"], kind="stable").reset_index(drop=True)


def load_season_dir(season: str, season_dir: Path) -> pd.DataFrame:
    d = Path(season_dir)
    return normalise_season(
        season,
        _read_csv(d / "merged_gw.csv"),
        _read_csv(d / "players_raw.csv", usecols=["id", "code", "element_type"]),
        _read_csv(d / "teams.csv", usecols=["id", "name"]),
        _read_csv(d / "fixtures.csv", usecols=["id", "event", "team_h", "team_a"]),
    )


def null_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """{season: columns entirely null in that season}."""
    return {
        str(season): [c for c in ARCHIVE_COLUMNS if g[c].isna().all()]
        for season, g in df.groupby("season", sort=True)
    }
