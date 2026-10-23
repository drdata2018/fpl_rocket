"""A-priori hyperparameters of the xP model (docs/PLAN_v2.md section 4). Fixed BEFORE any backtest.

Nothing here was tuned on the test seasons (2024-25, 2025-26, 2026-27). The values are conventional choices
from the football-modelling literature or round numbers chosen for stability; each is documented where it is
used. Changing any value changes MODEL_BASE_VERSION.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

MODEL_BASE_VERSION = "v0.1"


@dataclass(frozen=True)
class ModelConfig:
    # ------------------------------------------------------------------ minutes model
    # Horizons the minutes classifier is trained on (direct multi-horizon: features as of h GWs before the
    # target, `horizon` is a feature; horizon = target GW - last FINISHED GW). The bundle predicts 6 GWs and
    # trains 1..7 (7 = the 6th GW ahead while a GW is in progress); the retrospective backtest scores the
    # next GW only and therefore trains horizon 1 (same features, same hyperparameters).
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    # Rows of the earliest training season in its first `warmup_gws` GWs have no prior history in the training
    # set (the season before is not loaded), unlike any real prediction: dropped from the minutes training.
    warmup_gws: int = 5
    hgb_max_iter: int = 200
    hgb_learning_rate: float = 0.05
    hgb_max_leaf_nodes: int = 31
    hgb_min_samples_leaf: int = 50
    hgb_l2: float = 1.0
    rest_days_cap: float = 30.0  # days since the team's last match, capped (summer break / season start)
    # live-only availability layer (no historical status snapshots exist, so it cannot be a trained feature):
    # GW+1 availability = chance_of_playing_next_round / 100; for GW+h (h >= 2) the shortfall decays by
    # `availability_decay` per extra GW; status 'u' (unavailable / left) and 'n' (not eligible) -> 0 at
    # every horizon.
    availability_decay: float = 0.5

    # ------------------------------------------------------------------ team goals (Dixon-Coles)
    # time decay per day: weight = exp(-xi * days before the reference time); Dixon & Coles (1997) chose
    # 0.0065 per day for half-weekly data (half-life ~107 days, last season's matches weigh ~0.1-0.5)
    dc_xi: float = 0.0065
    dc_max_goals: int = 10  # score matrix 0..10 per side (mass above is < 1e-6 for lambda < 3.5)
    dc_rho_bounds: tuple[float, float] = (-0.2, 0.2)
    dc_sum_to_zero_weight: float = 100.0  # soft identifiability constraint: mean attack = mean defence = 0
    dc_ridge: float = 1e-3  # tiny ridge on every team parameter (keeps unseen teams finite)
    # promoted clubs: Gaussian prior centred on the mean of the previous season's bottom three, with the
    # precision of `dc_promoted_prior_matches` matches (Fisher information of one Poisson match ~ lambda)
    dc_promoted_prior_matches: float = 10.0

    # ------------------------------------------------------------------ player shares and per-90 rates
    recent_gws: int = 10  # rows in the last 10 GWs of the current season weigh 1 ...
    older_weight: float = 0.3  # ... older rows (earlier this season, previous season) weigh 0.3
    share_pseudo_minutes: float = 450.0  # xG / xA per 90 shrunk to the position prior with 450 pseudo-minutes
    rate_pseudo_minutes: float = 450.0  # saves and defensive-contribution counts per 90
    card_pseudo_minutes: float = 900.0  # cards are rarer: stronger shrinkage
    max_card_prob: float = 0.95

    # ------------------------------------------------------------------ bonus
    bonus_max_iter: int = 200
    bonus_learning_rate: float = 0.05
    bonus_min_samples_leaf: int = 100

    # ------------------------------------------------------------------ simulation
    n_sims: int = 1000
    seed: int = 20261023

    # feature names (recorded in the manifest)
    minutes_features: tuple[str, ...] = field(
        default=(
            "min_pf_1",
            "min_pf_3",
            "min_pf_5",
            "starts_pf_1",
            "starts_pf_3",
            "starts_pf_5",
            "season_min_share",
            "n_hist_5",
            "element_type",
            "rest_days",
            "team_fx_in_gw",
            "horizon",
        )
    )
    minutes_live_features: tuple[str, ...] = ("status", "chance_of_playing_next_round")
    bonus_features: tuple[str, ...] = (
        "exp_goals",
        "exp_assists",
        "p_cs",
        "element_type",
        "strength_diff",
        "home",
        "long",
    )

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = ModelConfig()
BACKTEST_CONFIG = ModelConfig(horizons=(1,))
