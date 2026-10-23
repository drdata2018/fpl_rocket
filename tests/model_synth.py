"""Synthetic league for xP-model tests: archive-schema rows for two seasons and bundle-like tables.

20 clubs per season (3 relegated / 3 promoted between seasons), 15 players per club (2 GK, 5 DEF, 5 MID,
3 FWD). Per club the regulars (1 GK, 4 DEF, 4 MID, 2 FWD) start and play 90, one MID comes on for ~25
minutes, the rest never play. Team goals are Poisson with known attack strengths and a home advantage;
goals are spread over the regulars in proportion to fixed xG shares. Points are scored by the rules engine.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from fpl_rocket.archive import ARCHIVE_COLUMNS
from fpl_rocket.rules import get_rules

LAYOUT = (1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4)  # element_type per squad slot
REGULAR = (0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13)  # starters (1 GK, 4 DEF, 4 MID, 2 FWD)
SUB = 11  # comes on for ~25 minutes
XG_SHARE = {12: 0.30, 13: 0.22, 7: 0.10, 8: 0.10, 9: 0.08, 10: 0.08, 2: 0.03, 3: 0.03, 4: 0.03, 5: 0.03}
SEASON1, SEASON2 = "2025-26", "2026-27"


def clubs(season: str) -> list[str]:
    base = [f"T{i:02d}" for i in range(17)]
    return base + (["T17", "T18", "T19"] if season == SEASON1 else ["P00", "P01", "P02"])


def strengths() -> dict[str, float]:
    s = {f"T{i:02d}": 0.5 - i * 0.05 for i in range(20)}  # T00 strongest
    s.update({"P00": -0.4, "P01": -0.45, "P02": -0.5})
    return s


def world(n_gws: int = 12, seed: int = 0) -> pd.DataFrame:
    """Archive-schema rows for SEASON1 (n_gws GWs) and SEASON2 (n_gws GWs)."""
    rng = np.random.default_rng(seed)
    rules = get_rules()
    st = strengths()
    rows = []
    for s_idx, season in enumerate((SEASON1, SEASON2)):
        teams = clubs(season)
        team_id = {t: i + 1 for i, t in enumerate(teams)}
        start = datetime(2025 + s_idx, 8, 15, 15, 0, tzinfo=UTC)
        fixture_id = 0
        for gw in range(1, n_gws + 1):
            order = rng.permutation(teams)
            kickoff = start + timedelta(days=7 * (gw - 1))
            for k in range(10):
                home, away = order[2 * k], order[2 * k + 1]
                fixture_id += 1
                lam = np.exp(0.25 + st[home] - st[away] * 0.5)
                mu = np.exp(0.0 + st[away] - st[home] * 0.5)
                goals = {home: rng.poisson(lam), away: rng.poisson(mu)}
                xg = {home: lam, away: mu}
                for team, opp, is_home in ((home, away, True), (away, home, False)):
                    shares = np.array([XG_SHARE.get(i, 0.0) for i in range(15)])
                    scorers = rng.multinomial(goals[team], np.r_[shares, 1 - shares.sum()])[:15]
                    assisters = rng.multinomial(goals[team], np.r_[shares, 1 - shares.sum()])[:15]
                    for slot, et in enumerate(LAYOUT):
                        code = 10_000 + sorted(st).index(team) * 100 + slot
                        element = team_id[team] * 100 + slot
                        minutes = 90 if slot in REGULAR else (int(rng.integers(15, 35)) if slot == SUB else 0)
                        conceded = goals[opp] if minutes else 0
                        stats = {
                            "minutes": minutes,
                            "goals_scored": int(scorers[slot]) if minutes else 0,
                            "assists": int(assisters[slot]) if minutes else 0,
                            "clean_sheets": int(minutes >= 60 and goals[opp] == 0),
                            "goals_conceded": conceded,
                            "saves": int(rng.poisson(3)) if (et == 1 and minutes) else 0,
                            "yellow_cards": int(rng.random() < 0.1) if minutes else 0,
                            "red_cards": 0,
                            "bonus": int(min(3, 3 * scorers[slot])) if minutes else 0,
                            "defensive_contribution": (int(rng.poisson(8)) if minutes and et > 1 else 0),
                        }
                        dc = stats["defensive_contribution"] if season == SEASON2 else None
                        pts = rules.points_from_stats({**stats, "defensive_contribution": dc or 0}, et)[
                            "total"
                        ]
                        rows.append(
                            {
                                "season": season,
                                "gw": gw,
                                "element": element,
                                "code": code,
                                "player_name": f"{team}-{slot}",
                                "element_type": et,
                                "team": team,
                                "team_id": team_id[team],
                                "opponent": opp,
                                "opponent_id": team_id[opp],
                                "was_home": is_home,
                                **{k: v for k, v in stats.items() if k != "defensive_contribution"},
                                "starts": int(slot in REGULAR),
                                "own_goals": 0,
                                "penalties_saved": 0,
                                "penalties_missed": 0,
                                "bps": 0,
                                "expected_goals": XG_SHARE.get(slot, 0.0) * xg[team] * minutes / 90,
                                "expected_assists": XG_SHARE.get(slot, 0.0) * xg[team] * 0.8 * minutes / 90,
                                "defensive_contribution": dc,
                                "total_points": pts,
                                "value": 50,
                                "selected": 1000,
                                "kickoff_time": pd.Timestamp(kickoff),
                                "fixture": fixture_id,
                            }
                        )
    df = pd.DataFrame(rows)
    for c in ARCHIVE_COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df[list(ARCHIVE_COLUMNS)]


def bundle_tables(df: pd.DataFrame, next_gw: int, n_upcoming: int = 3) -> dict[str, pd.DataFrame]:
    """Bundle-like tables for SEASON2 with GWs < next_gw finished and n_upcoming GWs of new fixtures (one
    club, the last, has a blank in the first upcoming GW; club 0 has a double)."""
    cur = df[(df["season"] == SEASON2) & (df["gw"] < next_gw)]
    teams = clubs(SEASON2)
    tid = {t: i + 1 for i, t in enumerate(teams)}
    players = (
        df[df["season"] == SEASON2]
        .drop_duplicates("element")
        .assign(
            id=lambda d: d["element"],
            team=lambda d: d["team"].map(tid),
            status=lambda d: np.where(d["element"] % 100 == 2, "i", "a"),
            chance_of_playing_next_round=lambda d: np.where(d["element"] % 100 == 2, 0.0, np.nan),
            penalties_order=lambda d: np.where(d["element"] % 100 == 12, 1.0, np.nan),
            web_name=lambda d: d["player_name"],
            now_cost=50,
        )[
            [
                "id",
                "code",
                "element_type",
                "team",
                "status",
                "chance_of_playing_next_round",
                "penalties_order",
                "web_name",
                "now_cost",
            ]
        ]
        .reset_index(drop=True)
    )
    past = cur.drop_duplicates("fixture")
    fx_rows = []
    for r in past.itertuples(index=False):
        home, away = (r.team, r.opponent) if r.was_home else (r.opponent, r.team)
        fx_rows.append(
            {
                "id": int(r.fixture),
                "event": int(r.gw),
                "team_h": tid[home],
                "team_a": tid[away],
                "finished": True,
                "kickoff_time": r.kickoff_time.isoformat(),
            }
        )
    fid = 1000
    last_kick = pd.Timestamp(cur["kickoff_time"].max())
    for i, gw in enumerate(range(next_gw, next_gw + n_upcoming)):
        order = list(teams)
        pairs = [(order[2 * k], order[2 * k + 1]) for k in range(10)]
        if i == 0:  # first upcoming GW: club 18 has a blank, club 0 a double (second match vs club 19)
            pairs = pairs[:9] + [(order[19], order[0])]
        for home, away in pairs:
            fid += 1
            fx_rows.append(
                {
                    "id": fid,
                    "event": gw,
                    "team_h": tid[home],
                    "team_a": tid[away],
                    "finished": False,
                    "kickoff_time": (last_kick + timedelta(days=7 * (i + 1))).isoformat(),
                }
            )
    events = pd.DataFrame(
        {
            "id": range(1, 39),
            "is_next": [g == next_gw for g in range(1, 39)],
            "finished": [g < next_gw for g in range(1, 39)],
        }
    )
    eh = cur.rename(columns={"gw": "round", "opponent_id": "opponent_team"})
    eh = eh.assign(kickoff_time=eh["kickoff_time"].map(lambda t: t.isoformat()))
    eh = eh[
        [
            "element",
            "round",
            "fixture",
            "opponent_team",
            "was_home",
            "kickoff_time",
            "minutes",
            "starts",
            "goals_scored",
            "assists",
            "clean_sheets",
            "goals_conceded",
            "own_goals",
            "saves",
            "yellow_cards",
            "red_cards",
            "bonus",
            "expected_goals",
            "expected_assists",
            "defensive_contribution",
            "total_points",
            "value",
        ]
    ]
    return {
        "players": players,
        "teams": pd.DataFrame({"id": [tid[t] for t in teams], "name": teams}),
        "fixtures": pd.DataFrame(fx_rows),
        "events": events,
        "element_history": eh.reset_index(drop=True),
    }
