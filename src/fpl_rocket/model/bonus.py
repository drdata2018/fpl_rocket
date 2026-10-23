"""Expected bonus conditional on playing: HistGradientBoostingRegressor (Poisson loss) on
(expected goals, expected assists, P(team CS), position, team strength difference, home, 60+ minutes).

The features are the model's own PRE-MATCH expectations, built the same way for training and prediction:
    exp_goals   = share_g * lambda_for * minutes / 90
    exp_assists = share_a * assist_rate * lambda_for * minutes / 90
    p_cs        = Dixon-Coles P(team concedes 0)
    strength_diff = log lambda_for - log lambda_against
with minutes = the row's actual minutes in training and the bin's expected minutes at prediction (the
`long` flag = 60+ separates the two bins). For the training rows lambda / shares come from the same fit
(in-sample within the training window; the target GW is never seen). In the simulation the bonus is NOT
simulated from the simulated goals: each playing line gets this expectation, rounded to an integer without
bias (floor + Bernoulli(fraction)) so the sample stays integer and its mean equals the expectation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import DEFAULT_CONFIG, ModelConfig


def bonus_feature_frame(
    share_g, share_a, lam_for, lam_against, p_cs, element_type, home, minutes, assist_rate: float
) -> pd.DataFrame:
    lam_for = np.asarray(lam_for, float)
    lam_against = np.asarray(lam_against, float)
    m = np.asarray(minutes, float) / 90.0
    return pd.DataFrame(
        {
            "exp_goals": np.asarray(share_g, float) * lam_for * m,
            "exp_assists": np.asarray(share_a, float) * assist_rate * lam_for * m,
            "p_cs": np.asarray(p_cs, float),
            "element_type": np.asarray(element_type, float),
            "strength_diff": np.log(np.clip(lam_for, 1e-6, None)) - np.log(np.clip(lam_against, 1e-6, None)),
            "home": np.asarray(home, float),
            "long": (np.asarray(minutes, float) >= 60).astype(float),
        }
    )


class BonusModel:
    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG) -> None:
        self.cfg = cfg
        self.reg: HistGradientBoostingRegressor | None = None
        self.mean_bonus = 0.2
        self.n_train = 0

    def fit(self, features: pd.DataFrame, bonus: pd.Series | np.ndarray) -> BonusModel:
        cfg = self.cfg
        y = np.asarray(bonus, float)
        x = features[list(cfg.bonus_features)].to_numpy(float)
        self.n_train = len(y)
        self.mean_bonus = float(y.mean()) if len(y) else 0.2
        if len(y) >= 200:
            self.reg = HistGradientBoostingRegressor(
                loss="poisson",
                learning_rate=cfg.bonus_learning_rate,
                max_iter=cfg.bonus_max_iter,
                min_samples_leaf=cfg.bonus_min_samples_leaf,
                categorical_features=[f == "element_type" for f in cfg.bonus_features],
                early_stopping=False,
                random_state=0,
            ).fit(x, y)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.reg is None:
            return np.full(len(features), self.mean_bonus)
        x = features[list(self.cfg.bonus_features)].to_numpy(float)
        return np.clip(self.reg.predict(x), 0.0, 3.0)


def bonus_link_table(played_rows: pd.DataFrame, pseudo_rows: float = 30.0) -> np.ndarray:
    """Multipliers m[element_type, min(goals, 2), min(assists, 2), clean sheet] = mean bonus in that cell /
    mean bonus of the position, over training rows of players who played; each cell is shrunk to the
    position mean with `pseudo_rows` (an empty cell -> 1). Used by bonus_sim = "linked"."""
    table = np.ones((5, 3, 3, 2))
    if played_rows.empty:
        return table
    num = pd.to_numeric
    r = pd.DataFrame(
        {
            "et": num(played_rows["element_type"], errors="coerce").fillna(0).astype(int).clip(0, 4),
            "g": num(played_rows["goals_scored"], errors="coerce").fillna(0).astype(int).clip(0, 2),
            "a": num(played_rows["assists"], errors="coerce").fillna(0).astype(int).clip(0, 2),
            "cs": (
                (num(played_rows["clean_sheets"], errors="coerce").fillna(0) > 0)
                & (num(played_rows["minutes"], errors="coerce").fillna(0) >= 60)
            ).astype(int),
            "bonus": num(played_rows["bonus"], errors="coerce").fillna(0).astype(float),
        }
    )
    pos_mean = r.groupby("et")["bonus"].mean()
    cells = r.groupby(["et", "g", "a", "cs"])["bonus"].agg(["sum", "size"]).reset_index()
    for c in cells.itertuples(index=False):
        mu = float(pos_mean.get(c.et, 0.0))
        if mu <= 0:
            continue
        table[c.et, c.g, c.a, c.cs] = ((c.sum + pseudo_rows * mu) / (c.size + pseudo_rows)) / mu
    return table
