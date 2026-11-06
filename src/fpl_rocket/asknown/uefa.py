"""European fixtures of English clubs from the UEFA match API (keyless):
https://match.uefa.com/v5/matches?competitionId=<1|14|2019>&seasonYear=<YYYY>&limit=..&offset=..
(1 = Champions League, 14 = Europa League, 2019 = Conference League; seasonYear 2026 = season 2025-26).

What this is and is not as-known: the API returns the fixtures AS CURRENTLY RECORDED (final kickoffs,
results).
For a decision at T only matches with kickoff < T may be used (rest days, fatigue). Future European dates at T
are NOT reconstructable from this source (a date could have moved after T, and the draw for later rounds may
not have been made at T): a policy needing them must use the schedule as known (not available historically
from
any keyless source found; prospectively, snapshot the API at each T). Untrusted data: json only.
"""

from __future__ import annotations

import json
import re

import pandas as pd

URL = "https://match.uefa.com/v5/matches"
COMPETITIONS = {1: "UCL", 14: "UEL", 2019: "UECL"}
SEASON_YEARS = (2022, 2023, 2024, 2025, 2026, 2027)
PAGE = 100

# UEFA internationalName -> FPL team name (where they differ)
ALIASES = {
    "manchester city": "man city",
    "man city": "man city",
    "manchester united": "man utd",
    "man united": "man utd",
    "tottenham hotspur": "spurs",
    "tottenham": "spurs",
    "newcastle united": "newcastle",
    "newcastle": "newcastle",
    "west ham united": "west ham",
    "west ham": "west ham",
    "brighton & hove albion": "brighton",
    "brighton": "brighton",
    "nottingham forest": "nott'm forest",
    "nottm forest": "nott'm forest",
    "leicester city": "leicester",
    "wolverhampton wanderers": "wolves",
    "wolverhampton": "wolves",
    "crystal palace": "crystal palace",
    "aston villa": "aston villa",
    "afc bournemouth": "bournemouth",
    "leeds united": "leeds",
}


def season_of_year(season_year: int) -> str:
    y = season_year - 1
    return f"{y}-{season_year % 100:02d}"


def parse_page(raw: bytes | str, competition_id: int) -> pd.DataFrame:
    data = json.loads(raw)
    rows = []
    for m in data if isinstance(data, list) else []:
        ko = (m.get("kickOffTime") or {}).get("dateTime")
        for side in ("homeTeam", "awayTeam"):
            t = m.get(side) or {}
            o = m.get("awayTeam" if side == "homeTeam" else "homeTeam") or {}
            rows.append(
                {
                    "match_id": str(m.get("id")),
                    "competition_id": competition_id,
                    "competition": COMPETITIONS.get(competition_id, str(competition_id)),
                    "season_year": int(m.get("seasonYear")) if m.get("seasonYear") else None,
                    "phase": m.get("competitionPhase"),
                    "status": m.get("status"),
                    "kickoff_utc": ko,
                    "club": t.get("internationalName"),
                    "club_id": t.get("id"),
                    "country": t.get("countryCode"),
                    "is_home": side == "homeTeam",
                    "placeholder": bool(t.get("isPlaceHolder")),
                    "opponent": o.get("internationalName"),
                    "opponent_country": o.get("countryCode"),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
    return df


def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    return ALIASES.get(s, s)


def map_english(df: pd.DataFrame, fpl_teams: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Keep English clubs (country ENG) and attach the FPL team id of the season the match belongs to."""
    eng = df[df["country"] == "ENG"].copy()
    eng["season"] = eng["season_year"].map(season_of_year)
    team = []
    for season, club in zip(eng["season"], eng["club"], strict=True):
        t = fpl_teams.get(season)
        tid = None
        if t is not None:
            lut = {}
            for i, n, s in zip(t["team"], t["name"], t["short_name"], strict=True):
                lut[_norm(n)] = int(i)
                lut[str(s).lower()] = int(i)
            tid = lut.get(_norm(club))
        team.append(tid)
    eng["fpl_team"] = pd.array(team, dtype="Int64")
    eng["known_at"] = eng["kickoff_utc"]
    return eng.drop_duplicates(["match_id", "club_id"]).sort_values("kickoff_utc").reset_index(drop=True)
