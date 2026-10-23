"""History panel and lagged features for the minutes model (no outcome of the target ever leaks in).

History rows = player-fixture rows in the archive schema (fpl_rocket.archive.ARCHIVE_COLUMNS; the bundle's
current season is converted to it by pipeline.bundle_history_rows). Players are keyed by the cross-season
`code` (pkey), time by t = first year of the season * 100 + gw (so 2025-26 GW38 < 2026-27 GW1).

* player_states(): per (pkey, t) the state AFTER that GW: minutes / starts per fixture over the player's last
  1 / 3 / 5 GWs with a fixture, number of those GWs (n_hist_5), and the season minutes share (minutes this
  season / (90 x team fixtures this season)).
* attach_states(): for target rows (t, horizon h) the latest state with state_t <= t - h, i.e. as known
  h GWs before the target (h = 1: at the target's deadline).
* schedule_features(): per (team, fixture) days since the team's previous match (capped) and the number of
  team fixtures in that GW - from the fixture list, which is known in advance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (1, 3, 5)


def season_t(season: pd.Series | str, gw) -> pd.Series | int:
    if isinstance(season, str):
        return int(season[:4]) * 100 + int(gw)
    return season.astype(str).str[:4].astype("int64") * 100 + pd.Series(gw, index=season.index).astype(
        "int64"
    )


def pkey(rows: pd.DataFrame) -> pd.Series:
    """Cross-season player key: `code`, else minus the season-local element id."""
    return rows["code"].astype("Float64").fillna(-rows["element"].astype("Float64")).astype("int64")


def minutes_bin(minutes: pd.Series | np.ndarray) -> np.ndarray:
    m = np.asarray(minutes, dtype=float)
    return np.where(m <= 0, 0, np.where(m < 60, 1, 2)).astype(int)


def player_states(history: pd.DataFrame) -> pd.DataFrame:
    """One row per (pkey, t) with the lagged-feature state after that GW."""
    h = history.assign(pkey=pkey(history), t=season_t(history["season"], history["gw"]))
    h["minutes"] = h["minutes"].astype("Float64").fillna(0).astype(float)
    h["starts"] = h["starts"].astype("Float64").fillna(0).astype(float)
    g = (
        h.groupby(["pkey", "t"], as_index=False, sort=True)
        .agg(
            season=("season", "first"),
            team=("team", "last"),
            minutes=("minutes", "sum"),
            starts=("starts", "sum"),
            fixtures=("fixture", "nunique"),
        )
        .sort_values(["pkey", "t"], kind="stable")
        .reset_index(drop=True)
    )
    by = g.groupby("pkey", sort=False)
    cums = {c: by[c].cumsum() for c in ("minutes", "starts", "fixtures")}

    def window(col: str, k: int) -> pd.Series:  # sum over the player's last k rows (incl. this one)
        cs = cums[col]
        return cs - cs.groupby(g["pkey"], sort=False).shift(k).fillna(0.0)

    for k in WINDOWS:
        mins, st, fx = window("minutes", k), window("starts", k), window("fixtures", k)
        g[f"min_pf_{k}"] = mins / fx.clip(lower=1)
        g[f"starts_pf_{k}"] = st / fx.clip(lower=1)
        if k == 5:
            g["starts_last_5"] = st
    g["n_hist_5"] = by.cumcount().add(1).clip(upper=5).astype(float)
    # season minutes share: player's minutes this season / (90 x his team's fixtures this season so far)
    g["season_minutes"] = g.groupby(["pkey", "season"], sort=False)["minutes"].cumsum()
    team_fx = team_fixture_counts(history)
    g = g.merge(team_fx, on=["season", "team", "t"], how="left")
    g["season_min_share"] = g["season_minutes"] / (90.0 * g["team_cum_fx"].clip(lower=1))
    return g.drop(columns=["minutes", "starts", "fixtures", "season_minutes", "team_cum_fx"])


def team_fixture_counts(history: pd.DataFrame) -> pd.DataFrame:
    """(season, team, t) -> cumulative number of the team's fixtures in the season up to and incl. t."""
    f = history.dropna(subset=["fixture", "team"])[["season", "gw", "team", "fixture"]].drop_duplicates()
    f = f.assign(t=season_t(f["season"], f["gw"]))
    c = f.groupby(["season", "team", "t"], as_index=False).agg(n=("fixture", "nunique"))
    c = c.sort_values(["season", "team", "t"], kind="stable")
    c["team_cum_fx"] = c.groupby(["season", "team"])["n"].cumsum()
    return c.drop(columns="n")


def attach_states(targets: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """Targets (pkey, t, horizon) + the latest state with state_t <= t - horizon (NaN when none)."""
    tg = targets.assign(lookup_t=(targets["t"] - targets["horizon"]).astype("int64"))
    tg = tg.reset_index(drop=False).rename(columns={"index": "_row"})
    st = states.rename(columns={"t": "state_t", "season": "state_season", "team": "state_team"})
    st = st.sort_values("state_t", kind="stable")
    tg = tg.sort_values("lookup_t", kind="stable")
    out = pd.merge_asof(
        tg,
        st,
        left_on="lookup_t",
        right_on="state_t",
        by="pkey",
        direction="backward",
        allow_exact_matches=True,
    )
    # the season share is only meaningful within the same season
    same = out["state_season"].astype(object) == out["season"].astype(object)
    out.loc[~same, "season_min_share"] = np.nan
    return out.sort_values("_row", kind="stable").set_index("_row").rename_axis(None)


def schedule_features(team_fixtures: pd.DataFrame, cap_days: float) -> pd.DataFrame:
    """team_fixtures: (season, gw, team, fixture, kickoff_time), one row per team-fixture (history and
    upcoming). Returns the same keys + rest_days (days since the team's previous match, capped; the first
    match in the data gets the cap) and team_fx_in_gw."""
    f = team_fixtures.drop_duplicates(["season", "team", "fixture"]).copy()
    f["kickoff_time"] = pd.to_datetime(f["kickoff_time"], utc=True)
    f = f.sort_values(["team", "kickoff_time", "fixture"], kind="stable")
    prev = f.groupby("team", sort=False)["kickoff_time"].shift(1)
    f["rest_days"] = ((f["kickoff_time"] - prev).dt.total_seconds() / 86400.0).clip(upper=cap_days)
    f["rest_days"] = f["rest_days"].fillna(cap_days)
    f["team_fx_in_gw"] = f.groupby(["season", "gw", "team"])["fixture"].transform("nunique").astype(float)
    return f[["season", "gw", "team", "fixture", "rest_days", "team_fx_in_gw"]]
