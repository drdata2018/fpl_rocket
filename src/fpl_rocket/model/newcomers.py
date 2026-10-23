"""Low-history players ("newcomers": fewer than ModelConfig.low_history_gws GWs in the training window).

Price (`value`, tenths) is used ONLY here, as a prior for these players, never as a feature for anyone else.

* price_band(): value -> band index over ModelConfig.price_band_edges (0 = cheapest; -1 = unknown price).
* NewcomerPrior.fit(): from the training rows
    - minutes prior per (position, band): the 0 / 1-59 / 60+ frequencies of NEWCOMER rows (the player had
      < low_history_gws GWs in the window before that GW); the earliest season's first `warmup_gws` GWs are
      skipped (everyone is "new" there only because the season before is not loaded); a cell with fewer than
      newcomer_prior_min_rows rows uses the position frequencies;
    - xG / xA per 90 prior per (position, band): sums over every row with minutes / minutes x 90; a cell with
      under 90 x newcomer_prior_min_rows minutes uses the position prior of PlayerRates.
* NewcomerPrior.apply() (newcomer_fallback = "prior"): for low-history rows, P(0 / 1-59 / 60+) = the player's
  own current-season bin counts with the band prior worth newcomer_prior_pseudo_gws fixtures (only the
  prior when he has no current-season row), capped by the live availability factor like the classifier;
  xG / xA per 90 shrunk to the band prior instead of the position prior (same pseudo-minutes).
* shrink_to_band() (newcomer_fallback = "shrink"): xP <- w xP + (1 - w) x (position x band mean xP per
  fixture of established players in the same GW) x fixtures, w = n_hist / low_history_gws.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, ModelConfig
from .features import minutes_bin, pkey, season_t


def price_band(
    value: pd.Series | np.ndarray | None, edges: tuple[int, ...], n: int | None = None
) -> np.ndarray:
    if value is None:
        return np.full(n or 0, -1, dtype=np.int64)
    v = pd.to_numeric(pd.Series(value), errors="coerce").to_numpy(float)
    band = np.searchsorted(np.asarray(edges, dtype=float), v, side="right").astype(np.int64)
    return np.where(np.isnan(v), -1, band)


def gws_before(history: pd.DataFrame) -> pd.Series:
    """Per history row: number of the player's distinct GWs in `history` strictly before that row's GW."""
    k = pkey(history)
    t = season_t(history["season"], history["gw"])
    gw = pd.DataFrame({"pkey": k, "t": t}).drop_duplicates().sort_values(["pkey", "t"])
    gw["n_before"] = gw.groupby("pkey").cumcount()
    out = pd.DataFrame({"pkey": k, "t": t}).merge(gw, on=["pkey", "t"], how="left")
    return pd.Series(out["n_before"].to_numpy(), index=history.index)


@dataclass
class NewcomerPrior:
    cfg: ModelConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    minutes_cell: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    minutes_pos: dict[int, np.ndarray] = field(default_factory=dict)
    rate_cell: dict[tuple[str, int, int], float] = field(default_factory=dict)  # (stat, position, band)
    n_newcomer_rows: int = 0

    def fit(self, history: pd.DataFrame) -> NewcomerPrior:
        cfg = self.cfg
        h = history[history["element_type"].isin((1, 2, 3, 4))]
        if h.empty:
            return self
        before = gws_before(h)
        earliest = sorted(h["season"].astype(str).unique())[0]
        warm = (h["season"].astype(str) == earliest) & (h["gw"].astype(int) <= cfg.warmup_gws)
        new = h[(before < cfg.low_history_gws) & ~warm]
        self.n_newcomer_rows = len(new)
        et_all = h["element_type"].astype(int).to_numpy()
        band_all = price_band(h.get("value"), cfg.price_band_edges, len(h))
        mins_all = pd.to_numeric(h["minutes"], errors="coerce").fillna(0).to_numpy(float)
        # minutes prior from newcomer rows
        if len(new):
            b = minutes_bin(pd.to_numeric(new["minutes"], errors="coerce").fillna(0))
            et = new["element_type"].astype(int).to_numpy()
            band = price_band(new.get("value"), cfg.price_band_edges, len(new))
            frame = pd.DataFrame({"et": et, "band": band, "bin": b})
            for e, g in frame.groupby("et"):
                self.minutes_pos[int(e)] = np.bincount(g["bin"], minlength=3) / len(g)
            for (e, bd), g in frame.groupby(["et", "band"]):
                if len(g) >= cfg.newcomer_prior_min_rows:
                    self.minutes_cell[(int(e), int(bd))] = np.bincount(g["bin"], minlength=3) / len(g)
        # xG / xA per 90 by position x band (all rows with minutes)
        played = mins_all > 0
        frame = pd.DataFrame(
            {
                "et": et_all[played],
                "band": band_all[played],
                "mins": mins_all[played],
                "xg": pd.to_numeric(h["expected_goals"], errors="coerce").fillna(0).to_numpy(float)[played],
                "xa": pd.to_numeric(h["expected_assists"], errors="coerce").fillna(0).to_numpy(float)[played],
            }
        )
        agg = frame.groupby(["et", "band"])[["mins", "xg", "xa"]].sum()
        for (e, bd), r in agg.iterrows():
            if r["mins"] >= 90 * cfg.newcomer_prior_min_rows:
                self.rate_cell[("xg", int(e), int(bd))] = float(r["xg"] / r["mins"] * 90)
                self.rate_cell[("xa", int(e), int(bd))] = float(r["xa"] / r["mins"] * 90)
        return self

    def minutes_prior(self, et: np.ndarray, band: np.ndarray) -> np.ndarray:
        out = np.zeros((len(et), 3))
        for i, (e, b) in enumerate(zip(et, band, strict=True)):
            p = self.minutes_cell.get((int(e), int(b)))
            if p is None:
                p = self.minutes_pos.get(int(e), np.array([0.5, 0.2, 0.3]))
            out[i] = p
        return out

    def rate_prior(
        self, stat: str, et: np.ndarray, band: np.ndarray, position_prior: np.ndarray
    ) -> np.ndarray:
        return np.array(
            [
                self.rate_cell.get((stat, int(e), int(b)), float(pp))
                for e, b, pp in zip(et, band, position_prior, strict=True)
            ]
        )

    def apply(self, rows: pd.DataFrame, history: pd.DataFrame, rates, season: str) -> pd.DataFrame:
        """Replace minutes probabilities and xG / xA shares of low-history rows (newcomer_fallback 'prior').
        rows: XPModel.lines rows (p0 p1 p2 availability xg90 xa90 share_g share_a low_history price_band)."""
        cfg = self.cfg
        rows = rows.copy()
        low = rows["low_history"].to_numpy(bool)
        if not low.any():
            return rows
        sub = rows.loc[low]
        et = sub["element_type"].astype(int).to_numpy()
        band = sub["price_band"].astype(int).to_numpy()
        prior = self.minutes_prior(et, band)
        # the player's own current-season bins (history rows of the target season)
        cur = history[history["season"].astype(str) == season]
        counts = np.zeros((len(sub), 3))
        if len(cur):
            b = minutes_bin(pd.to_numeric(cur["minutes"], errors="coerce").fillna(0))
            c = pd.DataFrame({"pkey": pkey(cur).to_numpy(), "bin": b})
            tab = pd.crosstab(c["pkey"], c["bin"]).reindex(columns=[0, 1, 2], fill_value=0)
            keys = pkey(sub)
            counts = tab.reindex(keys.to_numpy()).fillna(0).to_numpy(float)
        k = cfg.newcomer_prior_pseudo_gws
        p = (counts + k * prior) / (counts.sum(axis=1, keepdims=True) + k)
        play = p[:, 1] + p[:, 2]
        cap = sub["availability"].to_numpy(float) if "availability" in sub else np.ones(len(sub))
        scale = np.divide(np.minimum(play, cap), play, out=np.zeros_like(play), where=play > 0)
        p[:, 1] *= scale
        p[:, 2] *= scale
        p[:, 0] = 1.0 - p[:, 1] - p[:, 2]
        rows.loc[low, ["p0", "p1", "p2"]] = p
        # shares: shrink to the position x band prior
        keys = pkey(sub)
        team_xg = sub["team"].map(rates.team_xg).fillna(rates.league_team_xg).astype(float).clip(lower=0.05)
        team_xa = sub["team"].map(rates.team_xa).fillna(rates.league_team_xa).astype(float).clip(lower=0.05)
        for stat, share, team in (("xg", "share_g", team_xg), ("xa", "share_a", team_xa)):
            pos_prior = sub["element_type"].astype(int).map(rates.priors[f"{stat}90"]).fillna(0.0).to_numpy()
            pri = self.rate_prior(stat, et, band, pos_prior)
            r90 = rates.reshrink(keys, stat, pri, cfg.share_pseudo_minutes)
            rows.loc[low, f"{stat}90"] = r90
            rows.loc[low, share] = r90 / team.to_numpy()
        return rows


def shrink_to_band(xp: pd.DataFrame, cfg: ModelConfig) -> pd.Series:
    """newcomer_fallback 'shrink': xp frame with element_type, price_band, n_fixtures, n_hist, low_history,
    gw, xp -> shrunk xp (established players unchanged)."""
    x = xp.copy()
    per_fx = x["xp"] / x["n_fixtures"].clip(lower=1)
    est = ~x["low_history"].astype(bool) & (x["n_fixtures"] > 0)
    cell = (
        per_fx[est].groupby([x.loc[est, "gw"], x.loc[est, "element_type"], x.loc[est, "price_band"]]).mean()
    )
    pos = per_fx[est].groupby([x.loc[est, "gw"], x.loc[est, "element_type"]]).mean()
    keys = list(zip(x["gw"], x["element_type"], x["price_band"], strict=True))
    band_mean = pd.Series([cell.get(k, np.nan) for k in keys], index=x.index)
    pos_mean = pd.Series([pos.get(k[:2], 0.0) for k in keys], index=x.index)
    target = band_mean.fillna(pos_mean) * x["n_fixtures"]
    w = (x["n_hist"].astype(float) / cfg.low_history_gws).clip(0, 1)
    low = x["low_history"].astype(bool)
    return x["xp"].where(~low, w * x["xp"] + (1 - w) * target)
