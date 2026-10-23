"""The xP model end to end: fit components on history rows, predict player-fixture lines, simulate jointly.

History rows use the archive schema (fpl_rocket.archive.ARCHIVE_COLUMNS). For the live bundle the previous
season comes from the archive and the current season from the bundle (bundle_history_rows); for the
retrospective backtest both come from the archive (ModelPredictor).

Output (XPOutput):
    xp        one row per (element, gw): xp (mean), p20 / p50 / p80, p_play, p_60, p_cs, exp_goals,
              exp_assists, exp_bonus, exp_minutes, n_fixtures (0 = blank GW: everything 0), team,
              element_type,
              horizon (= gw - last finished GW: the age of the minutes features; 1 = the next GW when no
              GW is in progress), penalties_order (bootstrap, live only; not used yet), low_history (bool:
              fewer than ModelConfig.low_history_gws GWs of history in the training window; v0.2)
    sims      (element, gw, sim, points int8): the joint sample (N per element and GW)
    fixtures  two rows per fixture (team view): xg_for, xg_against, p_cs, p_gc_0..6 = the model
              fixture-difficulty grid
    lines     one row per player-fixture: minutes-model probabilities and the component parameters
"""

from __future__ import annotations

import copy
import dataclasses
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ..rules import CURRENT_SEASON, get_rules
from .bonus import BonusModel, bonus_feature_frame, bonus_link_table
from .config import BACKTEST_CONFIG, BUNDLE_CONFIG, DEFAULT_CONFIG, MODEL_BASE_VERSION, ModelConfig
from .features import low_history, player_states
from .minutes import PLAYER_TYPES, MinutesModel, availability_factor, build_feature_rows, team_fixture_list
from .newcomers import NewcomerPrior, price_band, shrink_to_band
from .rates import PlayerRates
from .simulate import aggregate, simulate_gw
from .team_goals import DixonColes, NaiveGoals, results_from_rows, team_fixture_frame

XP_COLUMNS = (
    "element",
    "gw",
    "horizon",
    "team",
    "element_type",
    "n_fixtures",
    "xp",
    "p20",
    "p50",
    "p80",
    "p_play",
    "p_60",
    "p_cs",
    "exp_goals",
    "exp_assists",
    "exp_bonus",
    "exp_minutes",
    "penalties_order",
    "low_history",
)


def git_sha_short(default: str = "nogit") -> str:
    import os

    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:7]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or default
    except (OSError, subprocess.SubprocessError):
        return default


def model_version() -> str:
    return f"{MODEL_BASE_VERSION}-{git_sha_short()}"


@dataclass
class XPOutput:
    xp: pd.DataFrame
    sims: pd.DataFrame | None
    fixtures: pd.DataFrame
    lines: pd.DataFrame
    timings: dict[str, float] = field(default_factory=dict)


def fixture_list(targets: pd.DataFrame) -> pd.DataFrame:
    """Unique fixtures (season, gw, fixture, kickoff_time, home, away) from team-perspective target rows."""
    t = targets.dropna(subset=["fixture"])
    home = np.where(t["was_home"].astype(bool), t["team"], t["opponent"])
    away = np.where(t["was_home"].astype(bool), t["opponent"], t["team"])
    f = pd.DataFrame(
        {
            "season": t["season"].to_numpy(),
            "gw": t["gw"].astype("int64").to_numpy(),
            "fixture": t["fixture"].astype("int64").to_numpy(),
            "kickoff_time": pd.to_datetime(t["kickoff_time"], utc=True).to_numpy(),
            "home": home.astype(str),
            "away": away.astype(str),
        }
    )
    return f.drop_duplicates(["season", "fixture"]).sort_values(["gw", "fixture"]).reset_index(drop=True)


class XPModel:
    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG, scoring_season: str = CURRENT_SEASON) -> None:
        self.cfg = cfg
        self.scoring_season = scoring_season  # rules engine used to score simulated lines
        self.dc = DixonColes(cfg)
        self.naive = NaiveGoals()
        self.minutes = MinutesModel(cfg)
        self.rates = PlayerRates(cfg)
        self.bonus = BonusModel(cfg)
        self.newcomer = NewcomerPrior(cfg)
        self.bonus_link = np.ones((5, 3, 3, 2))
        self.history: pd.DataFrame | None = None
        self.states: pd.DataFrame | None = None
        self.season = ""
        self.timings: dict[str, float] = {}
        self.fitted_on: dict[str, Any] = {}

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        history: pd.DataFrame,
        *,
        season: str,
        ref_time: pd.Timestamp,
        season_teams: list[str] | None = None,
    ) -> XPModel:
        t0 = time.perf_counter()
        h = history[history["element_type"].isin(PLAYER_TYPES)].reset_index(drop=True)
        self.history, self.season = h, season
        results = results_from_rows(h)
        self.dc.fit(results, ref_time=ref_time, season=season, season_teams=season_teams)
        self.naive.fit(results, ref_time=ref_time)
        t1 = time.perf_counter()
        self.states = player_states(h)
        self.minutes.fit(h, team_fixture_list(h), self.states)
        t2 = time.perf_counter()
        self.rates.fit(h, season)
        self._fit_bonus(h, results)
        self.newcomer = NewcomerPrior(self.cfg).fit(h)
        played = h[pd.to_numeric(h["minutes"], errors="coerce").fillna(0) > 0]
        self.bonus_link = bonus_link_table(played, self.cfg.bonus_link_pseudo_rows)
        t3 = time.perf_counter()
        self.timings = {"fit_team_goals_s": t1 - t0, "fit_minutes_s": t2 - t1, "fit_rates_bonus_s": t3 - t2}
        self.fitted_on = {
            "season": season,
            "seasons": sorted(h["season"].astype(str).unique().tolist()),
            "history_rows": len(h),
            "matches": int(self.dc.n_matches),
            "ref_time": pd.Timestamp(ref_time).isoformat(),
            "minutes_training_rows": int(self.minutes.n_train),
            "bonus_training_rows": int(self.bonus.n_train),
            "promoted": self.dc.promoted,
            "dc_converged": self.dc.converged,
            "newcomer_prior_rows": int(self.newcomer.n_newcomer_rows),
            "defcon_positions_with_history": sorted(self.rates.dc_positions),
        }
        return self

    def with_config(self, **changes: Any) -> XPModel:
        """A shallow copy of this FITTED model whose prediction-time options (newcomer_fallback, bonus_sim,
        defcon_fallback, n_sims, seed) differ; fitted components are shared, not refitted."""
        other = copy.copy(self)
        other.cfg = dataclasses.replace(self.cfg, **changes)
        return other

    def _fit_bonus(self, h: pd.DataFrame, results: pd.DataFrame) -> None:
        fx = self.dc.fixture_frame(results[["season", "fixture", "home", "away"]].reset_index(drop=True))
        played = h[pd.to_numeric(h["minutes"], errors="coerce").fillna(0) > 0]
        j = played.merge(fx, on=["season", "fixture"], how="inner")
        home = j["was_home"].astype(bool).to_numpy()
        lam_for = np.where(home, j["lambda_home"], j["lambda_away"])
        lam_against = np.where(home, j["lambda_away"], j["lambda_home"])
        p_cs = np.where(home, j["p_cs_home"], j["p_cs_away"])
        par = self.rates.params(j)
        feats = bonus_feature_frame(
            par["share_g"],
            par["share_a"],
            lam_for,
            lam_against,
            p_cs,
            j["element_type"],
            home,
            pd.to_numeric(j["minutes"], errors="coerce").fillna(0),
            self.rates.assist_rate,
        )
        self.bonus.fit(feats, pd.to_numeric(j["bonus"], errors="coerce").fillna(0))

    # ------------------------------------------------------------------ predict
    def lines(self, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Per player-fixture parameters for the simulation, and the fixture frame."""
        assert self.history is not None and self.states is not None, "fit first"
        cfg = self.cfg
        tg = targets[targets["element_type"].isin(PLAYER_TYPES)].reset_index(drop=True)
        schedule = pd.concat([team_fixture_list(self.history), team_fixture_list(tg)], ignore_index=True)
        rows = build_feature_rows(tg, self.history, schedule, cfg, self.states)
        rows = pd.concat(
            [rows.reset_index(drop=True), self.minutes.predict(rows).reset_index(drop=True)], axis=1
        )
        n_hist, low = low_history(rows, self.history, cfg.low_history_gws)
        rows["n_hist"], rows["low_history"] = n_hist.to_numpy(), low.to_numpy()
        rows["price_band"] = price_band(rows.get("value"), cfg.price_band_edges, len(rows))
        fixtures = self.dc.fixture_frame(fixture_list(tg))
        fx = fixtures.set_index("fixture")
        home = rows["was_home"].astype(bool).to_numpy()
        f = rows["fixture"].astype("int64")
        rows["lam_for"] = np.where(home, f.map(fx["lambda_home"]), f.map(fx["lambda_away"]))
        rows["lam_against"] = np.where(home, f.map(fx["lambda_away"]), f.map(fx["lambda_home"]))
        rows["p_cs_team"] = np.where(home, f.map(fx["p_cs_home"]), f.map(fx["p_cs_away"]))
        par = self.rates.params(rows)
        rows = pd.concat([rows, par], axis=1)
        if cfg.defcon_fallback and self.season >= cfg.defcon_first_season:
            # no DefCon counts for a position in the window (2025-26 GW1): the documented constant prior
            missing = ~rows["element_type"].astype(int).isin(self.rates.dc_positions)
            prior = rows["element_type"].astype(int).map(cfg.defcon_prior90).fillna(0.0)
            rows.loc[missing, "dc90"] = prior[missing]
        if cfg.newcomer_fallback == "prior":
            rows = self.newcomer.apply(rows, self.history, self.rates, self.season)
        is_gk = rows["element_type"].astype(int).to_numpy() == 1
        rows["saves_mu90"] = np.where(
            is_gk, rows["saves90"] * rows["lam_against"] / max(self.rates.league_goals, 0.1), 0.0
        )
        thresholds = get_rules(self.scoring_season).SCORING.defcon_threshold
        rows["dc_threshold"] = [
            float(thresholds.get(int(e))) if thresholds.get(int(e)) is not None else np.nan
            for e in rows["element_type"]
        ]
        for name, mcol in (("bonus_short", "m_short"), ("bonus_long", "m_long")):
            feats = bonus_feature_frame(
                rows["share_g"],
                rows["share_a"],
                rows["lam_for"],
                rows["lam_against"],
                rows["p_cs_team"],
                rows["element_type"],
                home,
                rows[mcol],
                self.rates.assist_rate,
            )
            rows[name] = self.bonus.predict(feats)
        rows["naive_home_mean"] = self.naive.home_mean
        rows["naive_away_mean"] = self.naive.away_mean
        return rows, fixtures

    def predict(
        self,
        targets: pd.DataFrame,
        *,
        n_sims: int | None = None,
        seed: int | None = None,
        sims_frame: bool = True,
        players: pd.DataFrame | None = None,
        on_gw: Callable[[pd.DataFrame, dict, np.ndarray], None] | None = None,
    ) -> XPOutput:
        """targets: player-fixture rows (season, gw, horizon, element, code, element_type, team, opponent,
        was_home, fixture, kickoff_time [, value, status, chance_of_playing_next_round, penalties_order]).
        players (optional): every player to report for every GW (element, code, element_type, team,
        penalties_order); players without a fixture in a GW get a blank row (n_fixtures 0).
        on_gw (diagnostics): called per GW with (player-fixture rows, raw simulation dict, points (E, S))."""
        n = n_sims or self.cfg.n_sims
        rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
        t0 = time.perf_counter()
        rows, fixtures = self.lines(targets)
        t1 = time.perf_counter()
        summaries, sim_frames = [], []
        for gw in sorted(rows["gw"].unique()):
            r = rows[rows["gw"] == gw]
            fxs = fixtures[fixtures["gw"] == gw]
            sim = simulate_gw(
                r,
                fxs,
                n_sims=n,
                rng=rng,
                season=self.scoring_season,
                assist_rate=self.rates.assist_rate,
                max_goals=self.cfg.dc_max_goals,
                max_card_prob=self.cfg.max_card_prob,
                bonus_sim=self.cfg.bonus_sim,
                bonus_link=self.bonus_link,
            )
            summary, pts = aggregate(r, sim)
            if on_gw is not None:
                on_gw(r, sim, pts)
            summary["gw"] = int(gw)
            summaries.append(summary)
            if sims_frame:
                sim_frames.append(
                    pd.DataFrame(
                        {
                            "element": np.repeat(summary["element"].to_numpy(), n).astype("int32"),
                            "gw": np.int16(gw),
                            "sim": np.tile(np.arange(n, dtype=np.int16), len(summary)),
                            "points": np.clip(pts, -128, 127).astype(np.int8).ravel(),
                        }
                    )
                )
        t2 = time.perf_counter()
        xp = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(columns=["element", "gw"])
        meta = rows.groupby(["element", "gw"], as_index=False).agg(
            horizon=("horizon", "first"),
            team=("team", "first"),
            element_type=("element_type", "first"),
            low_history=("low_history", "first"),
            n_hist=("n_hist", "first"),
            price_band=("price_band", "first"),
        )
        if "penalties_order" in rows:
            pen = rows.groupby(["element", "gw"], as_index=False)["penalties_order"].first()
            meta = meta.merge(pen, on=["element", "gw"], how="left")
        xp = xp.merge(meta, on=["element", "gw"], how="left")
        if self.cfg.newcomer_fallback == "shrink" and len(xp):
            xp["xp"] = shrink_to_band(xp, self.cfg)
        elif self.cfg.newcomer_fallback == "withhold" and len(xp):
            xp.loc[xp["low_history"].astype(bool), ["xp", "p20", "p50", "p80"]] = np.nan
        sims = pd.concat(sim_frames, ignore_index=True) if sim_frames else None
        if players is not None and len(xp):
            _, low = low_history(players, self.history, self.cfg.low_history_gws)
            players = players.assign(low_history=low.to_numpy())
            xp, sims = _add_blanks(xp, sims, players, sorted(rows["gw"].unique()), rows, n)
        xp["low_history"] = xp["low_history"].fillna(False).astype(bool)
        if "penalties_order" not in xp:
            xp["penalties_order"] = np.nan
        xp = xp[list(XP_COLUMNS)].sort_values(["gw", "element"], kind="stable").reset_index(drop=True)
        if sims is not None:
            sims = sims.sort_values(["element", "gw", "sim"], kind="stable").reset_index(drop=True)
        timings = {**self.timings, "lines_s": t1 - t0, "simulate_s": t2 - t1}
        return XPOutput(xp, sims, team_fixture_frame(fixtures), rows, timings)


def _add_blanks(
    xp: pd.DataFrame,
    sims: pd.DataFrame | None,
    players: pd.DataFrame,
    gws: list[int],
    rows: pd.DataFrame,
    n: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    horizon = rows.groupby("gw")["horizon"].first().to_dict()
    have = set(zip(xp["element"], xp["gw"], strict=True))
    blanks = []
    pl = players[players["element_type"].isin(PLAYER_TYPES)]
    for gw in gws:
        for p in pl.itertuples(index=False):
            if (p.element, gw) not in have:
                blanks.append(
                    {
                        "element": p.element,
                        "gw": int(gw),
                        "horizon": horizon[gw],
                        "team": p.team,
                        "element_type": p.element_type,
                        "penalties_order": getattr(p, "penalties_order", np.nan),
                        "low_history": bool(getattr(p, "low_history", False)),
                        "n_fixtures": 0,
                    }
                )
    if not blanks:
        return xp, sims
    b = pd.DataFrame(blanks)
    for c in (
        "xp",
        "p20",
        "p50",
        "p80",
        "p_play",
        "p_60",
        "p_cs",
        "exp_goals",
        "exp_assists",
        "exp_bonus",
        "exp_minutes",
    ):
        b[c] = 0.0
    xp = pd.concat([xp, b], ignore_index=True)
    if sims is not None:
        zeros = pd.DataFrame(
            {
                "element": np.repeat(b["element"].to_numpy(), n).astype("int32"),
                "gw": np.repeat(b["gw"].to_numpy(), n).astype("int16"),
                "sim": np.tile(np.arange(n, dtype=np.int16), len(b)),
                "points": np.zeros(len(b) * n, dtype=np.int8),
            }
        )
        sims = pd.concat([sims, zeros], ignore_index=True)
    return xp, sims


# ============================================================================== retrospective backtest
class ModelPredictor:
    """backtest.Predictor for the xP model: fit = all GWs before the target in the season + the previous
    season (rolling_origin's train set); predict = the target GW (horizon 1), WITHOUT live-only features.

    The full fit needs the target fixtures (DC reference time = first kickoff of the target GW, the season's
    clubs for the promoted prior), so fit() keeps the training rows and predict() fits and predicts.
    Details of the last fold (minutes probabilities, intervals, fixture lambdas, timings) are kept in
    `last_details` for the evaluation."""

    name = "model"

    def __init__(self, cfg: ModelConfig = BACKTEST_CONFIG, n_sims: int | None = None) -> None:
        self.cfg = cfg
        self.n_sims = n_sims
        self._train: pd.DataFrame | None = None
        self.last_details: dict[str, pd.DataFrame] = {}

    def fit(self, train: pd.DataFrame) -> None:
        self._train = train

    def predict(
        self, target_players: pd.DataFrame, target_gw: int, fixtures: pd.DataFrame | None = None
    ) -> pd.Series:
        assert self._train is not None and fixtures is not None, (
            "ModelPredictor needs fit() and target fixtures"
        )
        t0 = time.perf_counter()
        season = str(fixtures["season"].iloc[0])
        ref = pd.to_datetime(fixtures["kickoff_time"], utc=True).min()
        train = self._train
        teams = set(train.loc[train["season"] == season, "team"].dropna().astype(str))
        teams |= set(fixtures["team"].dropna().astype(str)) | set(fixtures["opponent"].dropna().astype(str))
        model = XPModel(self.cfg).fit(train, season=season, ref_time=ref, season_teams=sorted(teams))
        t1 = time.perf_counter()
        targets = fixtures.assign(horizon=1)
        out = model.predict(
            targets, n_sims=self.n_sims, sims_frame=False, seed=self.cfg.seed + int(target_gw)
        )
        t2 = time.perf_counter()
        xp = out.xp.set_index("element")
        pred = target_players["element"].map(xp["xp"]).fillna(0.0).astype(float)
        lines = out.lines
        self.last_details = {
            "players": out.xp.assign(season=season),
            "minutes": lines[
                ["element", "fixture", "element_type", "p0", "p1", "p2", "m_short", "m_long", "starts_last_5"]
            ].assign(season=season, gw=int(target_gw)),
            "fixtures": fixture_list(fixtures)
            .merge(
                out.fixtures[out.fixtures["was_home"]][["fixture", "xg_for", "xg_against"]].rename(
                    columns={"xg_for": "lambda_home", "xg_against": "lambda_away"}
                ),
                on="fixture",
            )
            .assign(rho=model.dc.rho, naive_home=model.naive.home_mean, naive_away=model.naive.away_mean),
            "timing": pd.DataFrame(
                [
                    {
                        "season": season,
                        "gw": int(target_gw),
                        "fit_s": t1 - t0,
                        "predict_s": t2 - t1,
                        **out.timings,
                    }
                ]
            ),
        }
        return pred.rename("xp")


# ============================================================================== live bundle
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def bundle_history_rows(tables: dict[str, Any], season: str = CURRENT_SEASON) -> pd.DataFrame:
    """Current-season history in the archive schema from a bundle's element_history + players + teams +
    fixtures (finished fixtures only). The team of each row comes from the fixture (handles transfers)."""
    from ..archive import ARCHIVE_COLUMNS

    eh = _df(tables["element_history"])
    players, teams, fixtures = _df(tables["players"]), _df(tables["teams"]), _df(tables["fixtures"])
    if eh.empty:
        return pd.DataFrame(columns=list(ARCHIVE_COLUMNS))
    names = dict(zip(teams["id"].astype(int), teams["name"].astype(str), strict=True))
    fx = fixtures.dropna(subset=["event"]).set_index(fixtures.dropna(subset=["event"])["id"].astype(int))
    finished = set(fx.index[fx["finished"].fillna(False).astype(bool)])
    eh = eh[eh["fixture"].astype(int).isin(finished)].copy()
    pl = players.set_index(players["id"].astype(int))
    out = pd.DataFrame(index=eh.index)
    out["season"] = season
    out["gw"] = eh["round"].astype("Int64")
    out["element"] = eh["element"].astype("Int64")
    out["code"] = eh["element"].astype(int).map(pl["code"]).astype("Int64")
    out["player_name"] = eh["element"].astype(int).map(pl["web_name"]).astype("string")
    out["element_type"] = eh["element"].astype(int).map(pl["element_type"]).astype("Int64")
    f = eh["fixture"].astype(int)
    home = eh["was_home"].astype(bool)
    team_id = np.where(home, f.map(fx["team_h"]), f.map(fx["team_a"]))
    out["team_id"] = pd.array(team_id, dtype="Int64")
    out["team"] = out["team_id"].map(names).astype("string")
    out["opponent_id"] = eh["opponent_team"].astype("Int64")
    out["opponent"] = out["opponent_id"].map(names).astype("string")
    out["was_home"] = home.astype("boolean")
    for c in ARCHIVE_COLUMNS:
        if c in out:
            continue
        if c == "kickoff_time":
            out[c] = pd.to_datetime(eh["kickoff_time"], utc=True)
        elif c == "fixture":
            out[c] = eh["fixture"].astype("Int64")
        elif c in eh:
            out[c] = _num(eh[c])
        else:
            out[c] = np.nan
    return out[list(ARCHIVE_COLUMNS)].reset_index(drop=True)


def _df(t: Any) -> pd.DataFrame:
    if isinstance(t, pd.DataFrame):
        return t
    from ..bundle import table_records

    return pd.DataFrame(table_records(t))


def bundle_targets(
    tables: dict[str, Any], n_gws: int = 6, season: str = CURRENT_SEASON
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(player-fixture targets for the next n_gws GWs from events.is_next, players frame). Every current
    player of a club with a fixture gets a row. horizon = gw - last finished GW, so while a GW is in
    progress the next GW is horizon 2 (its minutes features are as of the last finished GW)."""
    players, teams, fixtures, events = (_df(tables[k]) for k in ("players", "teams", "fixtures", "events"))
    nxt = events.loc[events["is_next"].fillna(False).astype(bool), "id"]
    if nxt.empty:
        return pd.DataFrame(), pd.DataFrame()
    first = int(nxt.iloc[0])
    names = dict(zip(teams["id"].astype(int), teams["name"].astype(str), strict=True))
    fx = fixtures.dropna(subset=["event"])
    fx = fx[(fx["event"].astype(int) >= first) & (fx["event"].astype(int) < first + n_gws)]
    fx = fx[~fx["finished"].fillna(False).astype(bool)]
    pl = pd.DataFrame(
        {
            "element": players["id"].astype(int),
            "code": players["code"].astype("Int64"),
            "element_type": players["element_type"].astype(int),
            "team_id": players["team"].astype(int),
            "status": players["status"].astype(str) if "status" in players else "a",
            "chance_of_playing_next_round": _num(
                players.get("chance_of_playing_next_round", pd.Series(np.nan, index=players.index))
            ),
            "penalties_order": _num(players.get("penalties_order", pd.Series(np.nan, index=players.index))),
            # price (tenths): ONLY a prior for low-history players (ModelConfig.newcomer_fallback)
            "value": _num(players.get("now_cost", pd.Series(np.nan, index=players.index))),
        }
    )
    pl["team"] = pl["team_id"].map(names)
    rows = []
    for side, other, home in (("team_h", "team_a", True), ("team_a", "team_h", False)):
        part = pd.DataFrame(
            {
                "fixture": fx["id"].astype(int),
                "gw": fx["event"].astype(int),
                "kickoff_time": pd.to_datetime(fx["kickoff_time"], utc=True),
                "team_id": fx[side].astype(int),
                "opponent_id": fx[other].astype(int),
                "was_home": home,
            }
        )
        rows.append(part)
    tf = pd.concat(rows, ignore_index=True)
    tg = tf.merge(pl, on="team_id", how="inner")
    tg["season"] = season
    tg["opponent"] = tg["opponent_id"].map(names)
    done = events.loc[events["finished"].fillna(False).astype(bool), "id"]
    last_finished = int(done.max()) if len(done) else first - 1
    tg["horizon"] = tg["gw"] - last_finished
    # GWs after the one chance_of_playing_next_round refers to (the next event): the live cap's decay steps
    tg["flag_steps"] = tg["gw"] - first
    tg = (
        tg[tg["element_type"].isin(PLAYER_TYPES)]
        .sort_values(["gw", "fixture", "element"])
        .reset_index(drop=True)
    )
    return tg, pl[pl["element_type"].isin(PLAYER_TYPES)]


def run_bundle_model(
    tables: dict[str, Any],
    archive_rows: pd.DataFrame,
    *,
    season: str = CURRENT_SEASON,
    now: datetime | None = None,
    cfg: ModelConfig = BUNDLE_CONFIG,
    n_gws: int = 6,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Fit on archive (previous season) + bundle (current season) and predict the next n_gws GWs.
    Returns ({xp, xp_sims, xp_fixtures[, availability]}, manifest fields).

    With the news stage's `availability` table in `tables` (fpl_rocket.news), its GW+1 cap (cap_with_news, or
    cap_model_only when cfg.news_in_cap is False) is the live availability cap, and the returned availability
    table gets p_play_unconstrained / p_play_model_only / p_play_with_news (see live_p_play)."""
    from .team_goals import previous_season

    t0 = time.perf_counter()
    prev = archive_rows[archive_rows["season"].astype(str) == previous_season(season)]
    current = bundle_history_rows(tables, season)
    history = pd.concat([prev, current], ignore_index=True)
    targets, players = bundle_targets(tables, n_gws, season)
    if targets.empty:
        raise ValueError("no upcoming fixtures (season over?)")
    avail = tables.get("availability")
    avail = _df(avail) if avail is not None else None
    if avail is not None and len(avail):
        caps = avail[["element", "cap_model_only", "cap_with_news"]].astype({"element": "int64"})
        targets = targets.merge(caps, on="element", how="left")
        targets["availability_next"] = targets["cap_with_news" if cfg.news_in_cap else "cap_model_only"]
    teams = sorted(_df(tables["teams"])["name"].astype(str))
    ref = pd.Timestamp(now or datetime.now(UTC))
    model = XPModel(cfg).fit(history, season=season, ref_time=ref, season_teams=teams)
    out = model.predict(targets, players=players)
    elapsed = time.perf_counter() - t0
    version = model_version()
    xp = out.xp.assign(model_version=version)
    fixtures = out.fixtures.assign(model_version=version)
    manifest = {
        "model_version": version,
        "model_fitted_on": {
            **model.fitted_on,
            "previous_season_rows": len(prev),
            "current_season_rows": len(current),
            "latest_gw_with_results": int(current["gw"].max()) if len(current) else None,
            "gws_predicted": sorted(int(g) for g in xp["gw"].unique()),
        },
        "model_features": {
            "minutes": list(cfg.minutes_features),
            "minutes_live_only (availability cap, not trained)": list(cfg.minutes_live_features),
            "bonus": list(cfg.bonus_features),
        },
        "model_hyperparameters": {k: v for k, v in cfg.as_dict().items() if not k.endswith("features")},
        "model_n_sims": cfg.n_sims,
        "model_runtime_s": round(elapsed, 1),
        "model_timings_s": {k: round(v, 2) for k, v in out.timings.items()},
        "model_availability_cap": (
            ("cap_with_news" if cfg.news_in_cap else "cap_model_only") if avail is not None else "fpl_flags"
        ),
    }
    result = {"xp": xp, "xp_sims": out.sims, "xp_fixtures": fixtures}
    if avail is not None and len(avail):
        result["availability"] = live_p_play(avail, out.lines, cfg)
    return result, manifest


def live_p_play(avail: pd.DataFrame, lines: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """availability + GW+1 P(play) of each player's FIRST fixture of the next predicted GW:
    p_play_unconstrained (the minutes classifier alone), p_play_model_only = min(that, factor from
    cap_model_only), p_play_with_news = min(that, factor from cap_with_news); factor = the minutes model's
    availability_factor (decay steps = GWs after the flagged GW). Players without a fixture there keep NaN."""
    out = avail.copy()
    if lines.empty or "p_play_uncapped" not in lines:
        return out
    first_gw = int(lines["gw"].min())
    r = lines[lines["gw"] == first_gw].sort_values(["element", "kickoff_time", "fixture"], kind="stable")
    r = r.drop_duplicates("element", keep="first")
    r = r.drop(columns=[c for c in ("cap_model_only", "cap_with_news") if c in r]).merge(
        avail[["element", "cap_model_only", "cap_with_news"]], on="element", how="left"
    )
    unc = r["p_play_uncapped"].to_numpy(float)
    status, chance = r.get("status"), r.get("chance_of_playing_next_round")
    steps = r.get("flag_steps")
    decay = cfg.availability_decay
    f_model = availability_factor(status, chance, r["horizon"], decay, r["cap_model_only"], steps)
    f_news = availability_factor(status, chance, r["horizon"], decay, r["cap_with_news"], steps)
    vals = pd.DataFrame(
        {
            "element": r["element"].astype("int64").to_numpy(),
            "p_play_unconstrained": unc,
            "p_play_model_only": np.minimum(unc, f_model),
            "p_play_with_news": np.minimum(unc, f_news),
        }
    )
    out = out.drop(columns=["p_play_unconstrained", "p_play_model_only", "p_play_with_news"], errors="ignore")
    out = out.merge(vals, on="element", how="left")
    out["next_event"] = pd.array([first_gw] * len(out), dtype="Int64")
    return out
