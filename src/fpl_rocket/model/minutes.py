"""Minutes model: P(0 / 1-59 / 60+ minutes) per player-fixture and the expected minutes within each bin.

* Classifier: sklearn HistGradientBoostingClassifier on ModelConfig.minutes_features (minutes and starts per
  fixture over the last 1 / 3 / 5 GWs, season minutes share, history length, position, days since the
  team's last match, team fixtures in the GW, horizon). Trained on every player-fixture row of the history
  (element_type 1-4) for each horizon h in ModelConfig.horizons, with features as known h GWs earlier.
* Expected minutes in 1-59: HistGradientBoostingRegressor on the same features (rows with 1-59 minutes),
  clipped to [1, 59]. Expected minutes in 60+: training mean of 60+ rows by position.
* Live-only availability layer (availability_factor): the archive holds no historical status /
  chance_of_playing snapshots, so these cannot be trained features. When present (live bundle only) they cap
  P(play): P(play) <- min(P(play), factor), the 1-59 / 60+ split kept. The retrospective backtest runs
  WITHOUT this layer (no such columns in the archive).
  v0.2.1: when the rows carry `availability_next` (the bundle's availability.parquet GW+1 cap with the
  staleness relief and the rule-based news, fpl_rocket.news; PROSPECTIVE-ONLY), it replaces chance / 100 as
  the GW+1 cap; the per-GW decay and the u / n rule are unchanged. predict() also returns the unconstrained
  P(play) (`p_play_uncapped`) so both caps can be recorded. Fix: the decay now counts GWs after the GW the
  flag refers to (`flag_steps` = gw - next event), not horizon - 1: while a GW was in progress the next GW
  (horizon 2) was decayed once, so a suspended / injured player got P(play) 0.5 for the very GW he misses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .config import DEFAULT_CONFIG, ModelConfig
from .features import attach_states, minutes_bin, pkey, player_states, schedule_features, season_t

PLAYER_TYPES = (1, 2, 3, 4)


def team_fixture_list(rows: pd.DataFrame) -> pd.DataFrame:
    cols = ["season", "gw", "team", "fixture", "kickoff_time"]
    return rows.dropna(subset=["team", "fixture"])[cols].drop_duplicates(["season", "team", "fixture"])


def build_feature_rows(
    targets: pd.DataFrame,
    history: pd.DataFrame,
    schedule: pd.DataFrame,
    cfg: ModelConfig,
    states: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """targets: player-fixture rows (season, gw, team, fixture, element, code, element_type, horizon).
    Returns targets + every minutes feature (and starts_last_5 for the rotation-risk evaluation)."""
    states = player_states(history) if states is None else states
    tg = targets.assign(pkey=pkey(targets), t=season_t(targets["season"], targets["gw"]))
    sched = schedule_features(schedule, cfg.rest_days_cap)
    tg = tg.merge(sched, on=["season", "gw", "team", "fixture"], how="left")
    tg = attach_states(tg, states)
    tg["rest_days"] = tg["rest_days"].fillna(cfg.rest_days_cap)
    tg["team_fx_in_gw"] = tg["team_fx_in_gw"].fillna(1.0)
    return tg


def feature_matrix(rows: pd.DataFrame, cfg: ModelConfig) -> np.ndarray:
    return rows[list(cfg.minutes_features)].astype(float).to_numpy()


def _categorical_mask(cfg: ModelConfig) -> list[bool]:
    return [f == "element_type" for f in cfg.minutes_features]


def availability_factor(
    status: pd.Series | None,
    chance: pd.Series | None,
    horizon: pd.Series,
    decay: float,
    next_cap: pd.Series | None = None,
    steps: pd.Series | None = None,
) -> np.ndarray:
    """Cap on P(play) from live flags: GW+1 = chance_of_playing_next_round / 100 (status i/s without a
    chance -> 0), or `next_cap` where given (fpl_rocket.news availability: staleness + news); the shortfall
    decays by `decay` per GW after the flagged one: `steps` (= gw - the next event, which
    chance_of_playing_next_round refers to; live bundle) or horizon - 1 when absent; status u / n -> 0 at
    every horizon."""
    n = len(horizon)
    h = np.asarray(horizon, dtype=float)
    st = pd.Series(status, dtype=object).fillna("a").to_numpy() if status is not None else np.full(n, "a")
    ch = (
        pd.to_numeric(pd.Series(chance), errors="coerce").to_numpy(float)
        if chance is not None
        else np.full(n, np.nan)
    )
    ch = np.where(np.isnan(ch) & np.isin(st, ["i", "s"]), 0.0, ch)
    a = np.where(np.isnan(ch), 1.0, np.clip(ch, 0, 100) / 100.0)
    if next_cap is not None:
        nc = pd.to_numeric(pd.Series(next_cap), errors="coerce").to_numpy(float)
        a = np.where(np.isnan(nc), a, np.clip(nc, 0.0, 1.0))
    k = h - 1
    if steps is not None:
        st_k = pd.to_numeric(pd.Series(steps), errors="coerce").to_numpy(float)
        k = np.where(np.isnan(st_k), k, st_k)
    factor = 1.0 - (1.0 - a) * decay ** np.clip(k, 0, None)
    return np.where(np.isin(st, ["u", "n"]), 0.0, factor)


class MinutesModel:
    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG) -> None:
        self.cfg = cfg
        self.clf: HistGradientBoostingClassifier | None = None
        self.reg_short: HistGradientBoostingRegressor | None = None
        self.long_minutes: dict[int, float] = {}
        self.short_default = 25.0
        self.n_train = 0

    def training_rows(
        self, history: pd.DataFrame, schedule: pd.DataFrame, states: pd.DataFrame
    ) -> pd.DataFrame:
        rows = history[history["element_type"].isin(PLAYER_TYPES)]
        base = rows[["season", "gw", "team", "fixture", "element", "code", "element_type", "minutes"]]
        earliest = sorted(rows["season"].astype(str).unique())[0] if len(rows) else None
        if earliest is not None:
            base = base[~((base["season"] == earliest) & (base["gw"] <= self.cfg.warmup_gws))]
        parts = [base.assign(horizon=h) for h in self.cfg.horizons]
        tg = pd.concat(parts, ignore_index=True)
        return build_feature_rows(tg, history, schedule, self.cfg, states)

    def fit(self, history: pd.DataFrame, schedule: pd.DataFrame, states: pd.DataFrame) -> MinutesModel:
        cfg = self.cfg
        tr = self.training_rows(history, schedule, states)
        x = feature_matrix(tr, cfg)
        y = minutes_bin(tr["minutes"].astype("Float64").fillna(0))
        self.n_train = len(tr)
        common = {
            "learning_rate": cfg.hgb_learning_rate,
            "max_iter": cfg.hgb_max_iter,
            "max_leaf_nodes": cfg.hgb_max_leaf_nodes,
            "min_samples_leaf": cfg.hgb_min_samples_leaf,
            "l2_regularization": cfg.hgb_l2,
            "early_stopping": False,
            "random_state": 0,
            "categorical_features": _categorical_mask(cfg),
        }
        self.clf = HistGradientBoostingClassifier(**common).fit(x, y)
        mins = tr["minutes"].astype("Float64").fillna(0).astype(float).to_numpy()
        short = y == 1
        if short.sum() >= 50:
            reg = {**common, "max_iter": 100}
            self.reg_short = HistGradientBoostingRegressor(**reg).fit(x[short], mins[short])
        self.short_default = float(mins[short].mean()) if short.any() else 25.0
        long_rows = tr.loc[y == 2]
        overall = float(long_rows["minutes"].astype(float).mean()) if len(long_rows) else 88.0
        by_pos = long_rows.groupby("element_type")["minutes"].mean().astype(float)
        self.long_minutes = {int(k): float(v) for k, v in by_pos.items()}
        self.long_minutes[0] = overall
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """rows: output of build_feature_rows (+ optional status / chance_of_playing_next_round).
        Returns p0, p1, p2, m_short, m_long, availability, p_play_uncapped aligned with rows."""
        assert self.clf is not None, "fit first"
        x = feature_matrix(rows, self.cfg)
        proba = self.clf.predict_proba(x)
        p = np.zeros((len(rows), 3))
        for j, c in enumerate(self.clf.classes_):
            p[:, int(c)] = proba[:, j]
        factor = availability_factor(
            rows.get("status"),
            rows.get("chance_of_playing_next_round"),
            rows["horizon"],
            self.cfg.availability_decay,
            rows.get("availability_next"),
            rows.get("flag_steps"),
        )
        play = p[:, 1] + p[:, 2]
        uncapped = play.copy()
        capped = np.minimum(play, factor)
        scale = np.divide(capped, play, out=np.zeros_like(play), where=play > 0)
        p[:, 1] *= scale
        p[:, 2] *= scale
        p[:, 0] = 1.0 - p[:, 1] - p[:, 2]
        m_short = (
            np.clip(self.reg_short.predict(x), 1, 59)
            if self.reg_short is not None
            else np.full(len(rows), self.short_default)
        )
        et = rows["element_type"].astype(int).to_numpy()
        m_long = np.array([self.long_minutes.get(int(e), self.long_minutes[0]) for e in et])
        return pd.DataFrame(
            {
                "p0": p[:, 0],
                "p1": p[:, 1],
                "p2": p[:, 2],
                "m_short": m_short,
                "m_long": m_long,
                "availability": factor,
                "p_play_uncapped": uncapped,
            },
            index=rows.index,
        )
