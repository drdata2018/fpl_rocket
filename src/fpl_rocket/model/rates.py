"""Player shares of team xG / xA and shrunk per-90 rates (cards, saves, defensive-contribution counts).

Row weights (ModelConfig): rows of the target season's last `recent_gws` (10) GWs weigh 1; every older row
(earlier this season, the previous season) weighs `older_weight` (0.3) - "last ~10 GWs plus previous-season
carry-over at reduced weight".

Per player (cross-season `code`), weighted sums over his rows, shrunk to the position prior (the weighted
league rate of that position) with pseudo-minutes P:
    rate90 = (sum w*x + prior90 * P / 90) / ((sum w*minutes + P) / 90)
    P = 450 for xG, xA, saves, defensive-contribution counts; 900 for yellow and red cards (rarer events).
Shares: share_g = xG90 / team xG per match, share_a = xA90 / team xA per match, the team rates computed with
the same weights over the team's fixtures (a club without rows - promoted, GW1 - gets the league average).
The player's current team is the target row's team, so a transferred player's absolute xG90 is divided by
his NEW team's rate. assist_rate = FPL assists per team goal in the training rows (each goal is assisted
with this probability in the simulation). Penalty takers: no special handling yet (TODO); penalties_order is
passed through to the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, ModelConfig
from .features import pkey

RATE_STATS = {
    "xg": "expected_goals",
    "xa": "expected_assists",
    "yc": "yellow_cards",
    "rc": "red_cards",
    "saves": "saves",
}


def _f(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def row_weights(history: pd.DataFrame, season: str, cfg: ModelConfig) -> pd.Series:
    cur = history["season"].astype(str) == season
    last = int(history.loc[cur, "gw"].max()) if cur.any() else None
    recent = cur & (history["gw"].astype(float) > (last - cfg.recent_gws if last is not None else np.inf))
    return pd.Series(np.where(recent, 1.0, cfg.older_weight), index=history.index)


@dataclass
class PlayerRates:
    cfg: ModelConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    players: pd.DataFrame = field(
        default_factory=pd.DataFrame
    )  # index pkey: xg90 xa90 yc90 rc90 saves90 dc90
    priors: pd.DataFrame = field(default_factory=pd.DataFrame)  # index element_type: same columns
    team_xg: dict[str, float] = field(default_factory=dict)  # per match
    team_xa: dict[str, float] = field(default_factory=dict)
    league_team_xg: float = 1.35
    league_team_xa: float = 1.0
    league_goals: float = 1.4  # mean goals per team per match (saves scaling)
    assist_rate: float = 0.87

    def fit(self, history: pd.DataFrame, season: str) -> PlayerRates:
        cfg = self.cfg
        h = history[history["element_type"].isin((1, 2, 3, 4))].copy()
        h["w"] = row_weights(h, season, cfg)
        h["pkey"] = pkey(h)
        h["mins"] = _f(h["minutes"]).fillna(0.0)
        h["wmin"] = h["w"] * h["mins"]
        for k, col in RATE_STATS.items():
            h[f"w_{k}"] = h["w"] * _f(h[col]).fillna(0.0)
        dc = (
            _f(h["defensive_contribution"])
            if "defensive_contribution" in h
            else pd.Series(np.nan, index=h.index)
        )
        has_dc = dc.notna()
        h["w_dc"] = h["w"] * dc.fillna(0.0)
        h["wmin_dc"] = h["wmin"].where(has_dc, 0.0)
        sums = [f"w_{k}" for k in RATE_STATS] + ["w_dc", "wmin", "wmin_dc"]

        played = h[h["mins"] > 0]
        pos = played.groupby("element_type")[sums].sum()
        pri = pd.DataFrame(index=pos.index)
        for k in RATE_STATS:
            pri[f"{k}90"] = pos[f"w_{k}"] / pos["wmin"].clip(lower=1e-9) * 90
        pri["dc90"] = (pos["w_dc"] / pos["wmin_dc"].where(pos["wmin_dc"] > 0)) * 90
        self.priors = pri.fillna({"dc90": 0.0}).fillna(0.0)

        per = h.groupby("pkey")[sums].sum()
        etype = h.groupby("pkey")["element_type"].last().astype(int)
        out = pd.DataFrame(index=per.index)
        for k in RATE_STATS:
            p = (
                cfg.card_pseudo_minutes
                if k in ("yc", "rc")
                else (cfg.share_pseudo_minutes if k in ("xg", "xa") else cfg.rate_pseudo_minutes)
            )
            prior = etype.map(self.priors[f"{k}90"]).fillna(0.0)
            out[f"{k}90"] = (per[f"w_{k}"] + prior * p / 90) / ((per["wmin"] + p) / 90)
        prior_dc = etype.map(self.priors["dc90"]).fillna(0.0)
        p = cfg.rate_pseudo_minutes
        out["dc90"] = (per["w_dc"] + prior_dc * p / 90) / ((per["wmin_dc"] + p) / 90)
        out["element_type"] = etype
        self.players = out

        # team xG / xA per match, same weights
        h["xg_f"] = _f(h["expected_goals"]).fillna(0.0)
        h["xa_f"] = _f(h["expected_assists"]).fillna(0.0)
        tf = (
            h.dropna(subset=["team", "fixture"])
            .groupby(["season", "fixture", "team"])
            .agg(w=("w", "first"), xg=("xg_f", "sum"), xa=("xa_f", "sum"))
            .reset_index()
        )
        if len(tf):
            agg = (
                tf.assign(wxg=tf["w"] * tf["xg"], wxa=tf["w"] * tf["xa"])
                .groupby("team")[["w", "wxg", "wxa"]]
                .sum()
            )
            self.team_xg = (agg["wxg"] / agg["w"]).to_dict()
            self.team_xa = (agg["wxa"] / agg["w"]).to_dict()
            self.league_team_xg = float((tf["w"] * tf["xg"]).sum() / tf["w"].sum())
            self.league_team_xa = float((tf["w"] * tf["xa"]).sum() / tf["w"].sum())
        goals = _f(h["goals_scored"]).fillna(0).sum() + _f(h["own_goals"]).fillna(0).sum()
        assists = _f(h["assists"]).fillna(0).sum()
        if goals > 0:
            self.assist_rate = float(min(assists / goals, 1.0))
            n_team_matches = h.dropna(subset=["fixture"]).groupby(["season", "fixture", "team"]).ngroups
            self.league_goals = float(goals / max(n_team_matches, 1))
        return self

    def params(self, targets: pd.DataFrame) -> pd.DataFrame:
        """Per target row (pkey via code/element, element_type, team): share_g, share_a and the per-90 rates
        (a player without history gets his position prior)."""
        key = pkey(targets)
        et = targets["element_type"].astype(int)
        out = pd.DataFrame(index=targets.index)
        for c in ("xg90", "xa90", "yc90", "rc90", "saves90", "dc90"):
            val = key.map(self.players[c]) if len(self.players) else pd.Series(np.nan, index=targets.index)
            prior = et.map(self.priors[c]) if c in self.priors else pd.Series(0.0, index=targets.index)
            out[c] = val.fillna(prior).fillna(0.0).astype(float)
        team_xg = targets["team"].map(self.team_xg).fillna(self.league_team_xg).astype(float)
        team_xa = targets["team"].map(self.team_xa).fillna(self.league_team_xa).astype(float)
        out["share_g"] = out["xg90"] / team_xg.clip(lower=0.05)
        out["share_a"] = out["xa90"] / team_xa.clip(lower=0.05)
        return out
