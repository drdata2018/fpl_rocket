"""A-priori hyperparameters of the xP model (docs/PLAN_v2.md section 4). Fixed BEFORE any backtest.

v0.2.1 adds the PROSPECTIVE-ONLY live availability constants (stale_*, news_*): they change only the live cap,
never the retrospective backtest (no status / news history exists in the archive).

Nothing here was tuned on the test seasons (2024-25, 2025-26, 2026-27). The values are conventional choices
from the football-modelling literature or round numbers chosen for stability; each is documented where it is
used. Changing any value changes MODEL_BASE_VERSION.

v0.2 (independent-review fixes): the low-history fallback (`newcomer_fallback`) and the simulated-bonus mode
(`bonus_sim`) were CHOSEN on season 2023-24 only (folds trained on 2022-23 + 2023-24) and then reported on
2024-25 / 2025-26 untouched (data/backtests/<ts>_v02_summary.md). The DefCon prior (`defcon_prior90`) is a
documented constant, not a tuned value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

MODEL_BASE_VERSION = "v0.2.1"

NEWCOMER_FALLBACKS = ("none", "withhold", "prior", "shrink")
BONUS_SIMS = ("round", "poisson", "linked")


# ---------------------------------------------------------------------------------- news text rules (v0.2.1)
# Rule-based reading of availability news (fpl_rocket.news.rules; no paid service, no LLM). Applied IN ORDER
# to the lower-cased item text (FPL `news`, or the injury table row rendered as "<Reason>. <Condition>.
# Status: <status>. Potential return: <date>. <quote>"); the FIRST match gives (p_play_next, p_60_given_play,
# confidence). Unmatched text -> confidence "low", NO adjustment. Only confidence >= news_min_confidence moves
# the cap. p_play_next: a float; "pct" = the captured percentage / 100; "date" = 0.0 when the captured return
# date is after the next deadline, else no estimate (None, confidence low); None = classification only.
# p_60_given_play is recorded only (availability.parquet), not used by the minutes model.
# PROSPECTIVE-ONLY round numbers (no news history exists to fit them).
# (name, regex on lower-cased text, p_play_next, p_60_given_play, confidence)
NEWS_TEXT_RULES: tuple[tuple[str, str, float | str | None, float | None, str], ...] = (
    # -- injury table (structured Status cell, rendered first in the item text)
    ("table_suspended", r"status: suspended", 0.0, None, "high"),
    ("table_ruled_out", r"status: ruled out", 0.0, None, "high"),
    ("table_pct", r"status: (\d{1,3})\s*%", "pct", 0.7, "medium"),
    # -- left the club / not available for the season
    (
        "left_permanent",
        r"has joined .*\bpermanently\b|has departed the club|left the club|has returned to "
        r"|contract (?:has been )?terminated|has retired",
        0.0,
        None,
        "high",
    ),
    ("loan", r"\bon loan\b|\bloan move\b|has joined .* loan", 0.0, None, "high"),
    ("not_in_squad", r"not included in (?:the )?squad|not registered|not in .{0,20}squad", 0.0, None, "high"),
    # -- suspensions
    ("suspended_until", r"suspended until (\d{1,2} [a-z]{3,9})", "date", None, "high"),
    ("suspended", r"\bsuspended\b|\bsuspension\b|red card", 0.0, None, "medium"),
    # -- FPL's standard injury phrases
    ("pct_chance", r"(\d{1,3})\s*% chance of playing", "pct", 0.7, "medium"),
    ("expected_back", r"expected back (\d{1,2} [a-z]{3,9})", "date", 0.5, "high"),
    ("unknown_return", r"unknown return date", 0.0, None, "high"),
    (
        "ruled_out",
        r"ruled out|out for the season|season-ending|out for (?:several|a few|\d+) (?:weeks|months)",
        0.0,
        None,
        "medium",
    ),
    ("doubtful", r"\bdoubtful\b|\bdoubt\b", 0.5, 0.6, "medium"),
    ("fitness_test", r"late fitness test|being assessed|to be assessed|assessed ahead", 0.5, 0.6, "low"),
    ("lack_of_fitness", r"lack of (?:match )?fitness|match fitness", 0.75, 0.3, "low"),
    ("illness", r"\billness\b|\bill\b|\bunwell\b", 0.5, 0.7, "low"),
    ("knock", r"\bknock\b", 0.75, 0.7, "low"),
    (
        "personal",
        r"personal reasons|compassionate|family reasons|international duty|awaiting club decision",
        None,
        None,
        "low",
    ),
)


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
    # v0.2.1: news recency + rule-based news in the live cap (fpl_rocket.news; PROSPECTIVE-ONLY: no historical
    # status / news snapshots exist, so NONE of these can be backtested; conservative round numbers, to be
    # reviewed against the prospective registry, which scores p_play_model_only and p_play_with_news).
    # Staleness: a 'd' / 'i' flag is STALE when FPL's news_added is older than `stale_after_days` and the
    # status / chance has not changed since (fpl_rocket.news.history). For a stale 'd' flag only, the GW+1
    # shortfall (1 - chance/100) is multiplied by max(stale_min_shortfall, 0.5 ** ((days since news_added -
    # stale_after_days) / stale_half_life_days)), i.e. the cap relaxes toward the unconstrained model P(play):
    # d75 stays 0.75 for 7 days, is 0.82 after 14 days and never exceeds 0.875 (half the shortfall kept). 'i'
    # flags are never relaxed by age alone (long injuries often go weeks without an FPL update).
    stale_after_days: float = 7.0
    stale_half_life_days: float = 14.0
    stale_min_shortfall: float = 0.5
    # News (fpl_rocket.news.availability; text rules in fpl_rocket.news.rules.NEWS_TEXT_RULES): an item with
    # confidence >= news_min_confidence can LOWER the GW+1 cap to min(cap, item p_play); FPL's own text only
    # when it says less than its own chance. It can RAISE the cap only when the FPL flag is stale and the item
    # (injury table) is newer than news_added: cap + news_raise_weight * (p_play - cap). Injury-table rows are
    # confidence low when the table is older than news_table_fresh_days. news_in_cap False = the cap ignores
    # news (both caps are still recorded).
    news_min_confidence: str = "medium"
    news_raise_weight: float = 0.5
    news_table_fresh_days: float = 3.0
    news_in_cap: bool = True

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

    # ------------------------------------------------------------------ low-history players (v0.2)
    # low_history = fewer than `low_history_gws` GWs with a row in the training window (cross-season `code`):
    # new signings, promoted-club players, academy debuts. Flagged in xp.parquet and backtest predictions.
    low_history_gws: int = 3
    # what the model does for them (NEWCOMER_FALLBACKS; chosen on 2023-24 only, see the module docstring):
    #   none     - v0.1 behaviour (position priors, minutes classifier on missing history)
    #   withhold - xP / quantiles NaN (the planner never picks them)
    #   prior    - minutes from the player's current-season bins if he has >= 1 GW, else the position x
    #              price-band prior of earlier newcomers; xG / xA per 90 shrunk to the position x price-band
    #              prior instead of the position prior
    #   shrink   - xP shrunk toward the position x price-band mean xP of established players:
    #              xP <- w xP + (1 - w) band mean, w = n_hist / low_history_gws
    # CHOSEN on 2023-24 only (decide run 20261023T191333Z; 1,017 low-history player-GWs, RMSE on them):
    # none 1.340, withhold 1.613 (scored as xP 0), prior 1.326, shrink 1.362 -> "prior" (all players
    # 1.9426 -> 1.9415).
    newcomer_fallback: str = "prior"
    # price bands in tenths (value < 5.0, 5.0-5.9, 6.0-7.4, 7.5-9.9, >= 10.0). Price is used ONLY as a prior
    # for low-history players, never as a feature for anyone else.
    price_band_edges: tuple[int, ...] = (50, 60, 75, 100)
    newcomer_prior_min_rows: int = 30  # a position x band cell with fewer newcomer rows uses the position
    newcomer_prior_pseudo_gws: float = 1.0  # weight of the prior against the player's own current-season bins

    # ------------------------------------------------------------------ defensive contributions (v0.2)
    # DefCon scoring exists from 2025-26. When the training window has NO DefCon counts for a position
    # (2025-26 GW1: 2024-25 has none) and the target season scores DefCon, the per-90 count prior of that
    # position is this constant: the 2025-26 full-season rate of players who played (sum of counts /
    # minutes x 90; vaastav archive; GK not eligible). Seasons before 2025-26 get no DefCon (not scored).
    # Caveat: for the 2025-26 GW1 backtest fold this constant uses that season's own data (a documented,
    # one-fold look-ahead).
    defcon_first_season: str = "2025-26"
    defcon_prior90: dict[int, float] = field(default_factory=lambda: {1: 0.0, 2: 7.69, 3: 8.58, 4: 4.79})
    defcon_fallback: bool = True

    # ------------------------------------------------------------------ bonus
    bonus_max_iter: int = 200
    bonus_learning_rate: float = 0.05
    bonus_min_samples_leaf: int = 100

    # ------------------------------------------------------------------ simulation
    n_sims: int = 1000
    seed: int = 20261023
    # how the bonus expectation becomes an integer in each simulation (BONUS_SIMS; chosen on 2023-24 only):
    #   round   - v0.1: floor + Bernoulli(fraction), independent of the simulated goals
    #   poisson - min(Poisson(expectation), 3)
    #   linked  - expectation x m(position, goals, assists, clean sheet of THAT simulation) / mean m, then
    #             unbiased rounding; m = mean bonus per cell / position mean in the training rows (shrunk with
    #             `bonus_link_pseudo_rows`), so a simulated goal carries its bonus and the mean is kept
    # CHOSEN on 2023-24 only (decide run 20261023T191333Z, ALL players): share above p80 observed / model-
    # implied: round 9.36% / 9.15%, poisson 9.33% / 9.22%, linked 9.23% / 8.97%; pinball at p20 / p50 / p80:
    # round .2097 / .4098 / .4296, poisson .2087 / .4096 / .4295, linked .2074 / .4059 / .4310; RMSE equal
    # (1.9426 / 1.9425 / 1.9420) -> "poisson" (smallest observed-implied gap, no pinball loss worse).
    # Note: on ALL players the v0.1 upper tail was already near its nominal; the "25% above p80" of v0.1
    # was measured on players who PLAYED (conditioning on an outcome), see the v02 summary.
    bonus_sim: str = "poisson"
    bonus_link_pseudo_rows: float = 30.0

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

    def __post_init__(self) -> None:
        if self.newcomer_fallback not in NEWCOMER_FALLBACKS:
            raise ValueError(f"newcomer_fallback must be one of {NEWCOMER_FALLBACKS}")
        if self.bonus_sim not in BONUS_SIMS:
            raise ValueError(f"bonus_sim must be one of {BONUS_SIMS}")
        if self.news_min_confidence not in ("low", "medium", "high"):
            raise ValueError("news_min_confidence must be low, medium or high")

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = ModelConfig()
BACKTEST_CONFIG = ModelConfig(horizons=(1,))
