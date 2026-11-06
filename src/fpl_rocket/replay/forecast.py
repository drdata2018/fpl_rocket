"""Predictions at the issue time T, computed from ReplayInputs only.

Forecasts(inputs, horizon) for GWs t .. t+horizon-1 (season end caps it):
    players     one row per player at T (bootstrap snapshot <= T): element, code, element_type, team (id),
                team_name, now_cost, status, chance (chance_of_playing_next_round), ownership
                (selected_by_percent), buyable (not removed, can_select not False), unavailable (status
                i / s / u / n or chance 0)
    n_fixtures  {(element, gw): fixtures of his club in that GW as known at T} (fixtures_as_known)
    low         players with fewer than 3 GWs of history in the window (model.features.low_history: never
                captain / vice, never bought - the planner's rule, the same set for every policy)
    xp(source)  {(element, gw): expected points}, source "baseline" or "model":
        baseline  frozen form per 90 x availability (backtest.FormPer90Baseline) fitted on the history at T,
                  per fixture x fixtures as known; 0 for every GW of the horizon when the player is flagged
                  unavailable at T (status i / s / u / n or chance_of_playing_next_round == 0)
        model     the v0.2.1 xP model (model.pipeline.XPModel, DEFAULT_CONFIG: the bundle's configuration
                  except n_sims 1000) fitted at T on the same history, DC reference time T, scoring with that
                  season's rules, horizon = gw - t + 1; the availability cap of the live bundle applied from
                  the AS-KNOWN status / chance (flag_steps = gw - t: decays 0.5 per GW after GW t)
Everything is deterministic (sorted inputs, the model's fixed seed + gw).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..backtest import FormPer90Baseline
from ..model.features import low_history
from .inputs import ReplayInputs

UNAVAILABLE_STATUSES = ("i", "s", "u", "n")
LOW_HISTORY_GWS = 3
FORECAST_HORIZON = 5  # the wildcard trigger looks 5 GWs ahead; the planners use the first 3
FORECAST_VERSION = "1"  # part of the cache key: bump when this module changes what it computes


def players_frame(inp: ReplayInputs) -> pd.DataFrame:
    p = inp.asknown.players_at_T
    teams = inp.asknown.teams_at_T
    names = dict(zip(teams["team"].astype(int), teams["name"].astype(str), strict=True))
    out = pd.DataFrame(
        {
            "element": p["element"].astype(int).to_numpy(),
            "code": pd.to_numeric(p["code"], errors="coerce").astype("Int64").to_numpy(),
            "element_type": p["element_type"].astype(int).to_numpy(),
            "team": p["team"].astype(int).to_numpy(),
            "now_cost": pd.to_numeric(p["now_cost"], errors="coerce").astype(int).to_numpy(),
            "status": p["status"].fillna("a").astype(str).to_numpy(),
            "chance": pd.to_numeric(p["chance_of_playing_next_round"], errors="coerce").to_numpy(float),
            "ownership": pd.to_numeric(p["selected_by_percent"], errors="coerce").fillna(0).to_numpy(float),
            "penalties_order": pd.to_numeric(p["penalties_order"], errors="coerce").to_numpy(float),
            "removed": p["removed"].astype("boolean").fillna(False).to_numpy(bool),
            "can_select": p["can_select"].astype("boolean").fillna(True).to_numpy(bool),
        }
    )
    out = out[out["element_type"].isin((1, 2, 3, 4))]
    out["team_name"] = out["team"].map(names)
    out["buyable"] = ~out["removed"] & out["can_select"]
    out["unavailable"] = out["status"].isin(UNAVAILABLE_STATUSES) | (out["chance"] == 0)
    return out.sort_values("element").reset_index(drop=True)


def horizon_gws(inp: ReplayInputs, horizon: int, n_gws: int = 38) -> list[int]:
    return list(range(inp.gw, min(n_gws, inp.gw + horizon - 1) + 1))


def fixtures_frame(inp: ReplayInputs, gws: list[int]) -> pd.DataFrame:
    f = inp.asknown.fixtures_as_known.dropna(subset=["event"])
    f = f[f["event"].astype(int).isin(gws)]
    return (
        f.assign(event=f["event"].astype(int))
        .sort_values(["event", "kickoff", "fixture"])
        .reset_index(drop=True)
    )


@dataclass
class Forecasts:
    season: str
    gw: int
    gws: list[int]
    players: pd.DataFrame
    n_fixtures: dict[tuple[int, int], int]
    low: frozenset[int]
    tables: dict[str, dict[tuple[int, int], float]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def xp(self, source: str) -> dict[tuple[int, int], float]:
        return self.tables[source]

    def element_type(self) -> dict[int, int]:
        return dict(zip(self.players["element"], self.players["element_type"], strict=True))

    def team(self) -> dict[int, int]:
        return dict(zip(self.players["element"], self.players["team"], strict=True))

    def price(self) -> dict[int, int]:
        return dict(zip(self.players["element"], self.players["now_cost"], strict=True))

    def blanks(self, squad: list[int], gw: int) -> int:
        return sum(1 for e in squad if self.n_fixtures.get((e, gw), 0) == 0)


def build(
    inp: ReplayInputs, sources: tuple[str, ...] = ("baseline", "model"), horizon: int = FORECAST_HORIZON
) -> Forecasts:
    players = players_frame(inp)
    gws = horizon_gws(inp, horizon)
    fx = fixtures_frame(inp, gws)
    per_team: dict[tuple[int, int], int] = {}
    for g, h, a in zip(fx["event"], fx["team_h"].astype(int), fx["team_a"].astype(int), strict=True):
        per_team[(h, g)] = per_team.get((h, g), 0) + 1
        per_team[(a, g)] = per_team.get((a, g), 0) + 1
    nfix = {
        (int(e), g): per_team.get((int(t), g), 0)
        for e, t in zip(players["element"], players["team"], strict=True)
        for g in gws
    }
    hist = inp.history
    _, low = low_history(players.assign(code=players["code"]), hist, LOW_HISTORY_GWS)
    low_set = frozenset(int(e) for e in players.loc[low.to_numpy(bool), "element"])
    fc = Forecasts(inp.season, inp.gw, gws, players, nfix, low_set)
    if "baseline" in sources:
        fc.tables["baseline"] = baseline_xp(players, hist, nfix, gws)
    if "model" in sources:
        fc.tables["model"], fc.meta["model"] = model_xp(inp, players, fx, gws)
    return fc


def baseline_xp(
    players: pd.DataFrame, hist: pd.DataFrame, nfix: dict[tuple[int, int], int], gws: list[int]
) -> dict[tuple[int, int], float]:
    b = FormPer90Baseline()
    b.fit(hist)
    key = players["code"].astype("Float64").fillna(-players["element"].astype("Float64")).astype("int64")
    per_fix = key.map(b.xp_per_fixture).fillna(0.0).astype(float).to_numpy()
    per_fix = np.where(players["unavailable"].to_numpy(bool), 0.0, per_fix)
    out = {}
    for e, v in zip(players["element"].astype(int), per_fix, strict=True):
        for g in gws:
            n = nfix.get((e, g), 0)
            if n:
                out[(e, g)] = float(v) * n
    return out


def model_targets(inp: ReplayInputs, players: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    teams = inp.asknown.teams_at_T
    names = dict(zip(teams["team"].astype(int), teams["name"].astype(str), strict=True))
    parts = []
    for side, other, home in (("team_h", "team_a", True), ("team_a", "team_h", False)):
        parts.append(
            pd.DataFrame(
                {
                    "fixture": fx["fixture"].astype(int).to_numpy(),
                    "gw": fx["event"].astype(int).to_numpy(),
                    "kickoff_time": pd.to_datetime(fx["kickoff"], utc=True).to_numpy(),
                    "team_id": fx[side].astype(int).to_numpy(),
                    "opponent_id": fx[other].astype(int).to_numpy(),
                    "was_home": home,
                }
            )
        )
    tf = pd.concat(parts, ignore_index=True)
    pl = players.rename(
        columns={"team": "team_id", "now_cost": "value", "chance": "chance_of_playing_next_round"}
    )
    pl = pl[
        [
            "element",
            "code",
            "element_type",
            "team_id",
            "value",
            "status",
            "chance_of_playing_next_round",
            "penalties_order",
        ]
    ]
    tg = tf.merge(pl, on="team_id", how="inner")
    tg["season"] = inp.season
    tg["team"] = tg["team_id"].map(names)
    tg["opponent"] = tg["opponent_id"].map(names)
    tg["horizon"] = tg["gw"] - inp.gw + 1
    tg["flag_steps"] = tg["gw"] - inp.gw
    return tg.sort_values(["gw", "fixture", "element"]).reset_index(drop=True)


def model_xp(
    inp: ReplayInputs, players: pd.DataFrame, fx: pd.DataFrame, gws: list[int]
) -> tuple[dict[tuple[int, int], float], dict[str, Any]]:
    from ..model.config import DEFAULT_CONFIG
    from ..model.pipeline import XPModel

    targets = model_targets(inp, players, fx)
    if targets.empty:
        return {}, {"note": "no fixtures as known"}
    teams = sorted(inp.asknown.teams_at_T["name"].astype(str))
    model = XPModel(DEFAULT_CONFIG, scoring_season=inp.season).fit(
        inp.history, season=inp.season, ref_time=inp.issue_time, season_teams=teams
    )
    out = model.predict(targets, sims_frame=False, seed=DEFAULT_CONFIG.seed + int(inp.gw))
    x = out.xp
    table = {
        (int(e), int(g)): float(v)
        for e, g, v in zip(x["element"], x["gw"], x["xp"].fillna(0.0), strict=True)
        if int(g) in gws
    }
    meta = {"fitted_on": model.fitted_on, "rows_predicted": len(targets)}
    return table, meta


# ======================================================================================= disk cache
def cache_path(cache_dir: str | Path, season: str, gw: int) -> Path:
    return Path(cache_dir) / season / f"gw{gw:02d}.pkl"


def load_or_build(inp: ReplayInputs, cache_dir: str | Path | None, digest: str) -> Forecasts:
    """Forecasts from <cache_dir>/<season>/gwNN.pkl when its input digest matches, else built (and cached)."""
    if cache_dir is not None:
        p = cache_path(cache_dir, inp.season, inp.gw)
        if p.exists():
            with p.open("rb") as fh:
                cached = pickle.load(fh)  # noqa: S301 - our own cache file
            if cached.get("digest") == digest:
                return cached["forecasts"]
    fc = build(inp)
    if cache_dir is not None:
        p = cache_path(cache_dir, inp.season, inp.gw)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump({"digest": digest, "forecasts": fc}, fh)
    return fc
