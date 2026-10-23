"""Joint Monte Carlo of one gameweek: every component drawn together so no impossible combination occurs.

Per simulation s (N = ModelConfig.n_sims):
1. each fixture's score is drawn from its Dixon-Coles score matrix;
2. each player-fixture's minutes bin (0 / 1-59 / 60+) from the minutes model, minutes = the bin's expected
   minutes (independent across players and fixtures, given the features);
3. each team's goals are allocated MULTINOMIALLY to its players with weights share_g * minutes / 90 plus an
   "unattributed" remainder max(0, 1 - sum) (own goals, unlisted players); when the weights sum above 1 they
   are normalised. Each goal is assisted with probability assist_rate, the assisted goals allocated the same
   way with share_a. Hence a player's goals never exceed his team's goals and the team's assists never
   exceed its goals;
4. clean sheet = minutes >= 60 and the team conceded 0 (the rules engine also requires 60+); goals conceded
   = the team's goals against for 60+ minutes, Binomial(goals against, minutes / 90) below 60;
5. saves (GK) ~ Poisson(saves90 * lambda_opponent / league mean goals * minutes / 90); yellow / red cards ~
   Bernoulli(rate90 * minutes / 90); defensive contribution ~ Bernoulli(P(Poisson(dc90 * minutes / 90) >=
   threshold)) with the threshold of the rules engine, recorded as a count at the threshold or 0;
6. bonus = the bonus model's expectation for the bin, made an integer per ModelConfig.bonus_sim: "round"
   (v0.1: floor + Bernoulli(fraction), independent of the simulated goals), "poisson" (min(Poisson(e), 3))
   or "linked" (e x m(position, simulated goals, assists, clean sheet) / mean of m over the row's playing
   simulations, then unbiased rounding: a simulated goal carries its bonus, the mean is kept);
7. EVERY simulated stat line is scored by the rules engine (rules.points_from_stats, per fixture; a double
   GW is the sum of its fixtures). There is no second scoring implementation: distinct stat lines are scored
   once each by the engine and mapped back.
Penalty saves / misses and own goals are not simulated (0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ..rules import get_rules
from .team_goals import score_matrix

# stat columns in the encoded line, with radix (values are clipped to radix - 1)
LINE_STATS = (
    ("element_type", 5),
    ("minutes", 121),
    ("goals_scored", 16),
    ("assists", 16),
    ("clean_sheets", 2),
    ("goals_conceded", 16),
    ("saves", 41),
    ("yellow_cards", 2),
    ("red_cards", 2),
    ("bonus", 4),
    ("defensive_contribution", 41),
)
ROW_COLUMNS = (
    "element",
    "fixture",
    "was_home",
    "element_type",
    "p0",
    "p1",
    "p2",
    "m_short",
    "m_long",
    "share_g",
    "share_a",
    "yc90",
    "rc90",
    "saves_mu90",
    "dc90",
    "dc_threshold",
    "bonus_short",
    "bonus_long",
)
FIXTURE_COLUMNS = ("fixture", "lambda_home", "lambda_away", "rho")


def score_lines(lines: np.ndarray, season: str) -> np.ndarray:
    """Points of each stat line (rows of LINE_STATS values) from the rules engine, one call per distinct
    line."""
    rules = get_rules(season)
    dims = tuple(d for _, d in LINE_STATS)
    clipped = np.clip(lines, 0, np.array(dims) - 1).astype(np.int64)
    keys = np.ravel_multi_index(tuple(clipped.T), dims)
    uniq, inverse = np.unique(keys, return_inverse=True)
    decoded = np.array(np.unravel_index(uniq, dims)).T
    names = [n for n, _ in LINE_STATS]
    pts = np.empty(len(uniq), dtype=np.int64)
    for i, vals in enumerate(decoded):
        stats = dict(zip(names[1:], (int(v) for v in vals[1:]), strict=True))
        pts[i] = int(rules.points_from_stats(stats, int(vals[0]))["total"])
    return pts[inverse.ravel()]


COMPONENTS = (
    "appearance",
    "goals",
    "assists",
    "clean_sheet",
    "saves",
    "goals_conceded",
    "cards",
    "bonus",
    "defcon",
)


def component_points(
    element_type: np.ndarray,
    minutes: np.ndarray,
    goals: np.ndarray,
    assists: np.ndarray,
    clean_sheets: np.ndarray,
    goals_conceded: np.ndarray,
    saves: np.ndarray,
    yellow: np.ndarray,
    red: np.ndarray,
    bonus: np.ndarray,
    defcon: np.ndarray,
    season: str,
) -> dict[str, np.ndarray]:
    """Points per scoring component (COMPONENTS) of per-fixture stat lines, vectorised from the season's
    ScoringTable (the same table the rules engine applies; tests check the sum equals the engine's total for
    every simulated stat). Used by the diagnostics (where do misses above p80 come from)."""
    t = get_rules(season).SCORING
    et = np.asarray(element_type, dtype=np.int64)
    m = np.asarray(minutes, dtype=float)
    played = m > 0
    long_game = m >= t.long_minutes

    def by_pos(table: dict, default: float = 0.0) -> np.ndarray:
        lut = np.array([float(table.get(k, default) or 0.0) for k in range(5)])
        return lut[et]

    gc_pos = np.isin(et, list(t.goals_conceded_positions))
    sv_pos = np.isin(et, list(t.save_point_positions))
    thr_lut = np.array(
        [np.nan]
        + [np.nan if t.defcon_threshold.get(k) is None else t.defcon_threshold[k] for k in range(1, 5)]
    )
    thr = thr_lut[et]
    dc_hit = ~np.isnan(thr) & (np.asarray(defcon, dtype=float) >= np.nan_to_num(thr, nan=np.inf))
    out = {
        "appearance": np.where(long_game, t.appearance_long, np.where(played, t.appearance_short, 0)),
        "goals": goals * by_pos(t.goal),
        "assists": assists * t.assist,
        "clean_sheet": np.where(long_game, clean_sheets, 0) * by_pos(t.clean_sheet),
        "saves": np.where(sv_pos, np.asarray(saves, dtype=np.int64) // t.saves_per_point, 0),
        "goals_conceded": np.where(
            gc_pos,
            (np.asarray(goals_conceded, dtype=np.int64) // t.goals_conceded_per_point)
            * t.goals_conceded_point,
            0,
        ),
        "cards": yellow * t.yellow + red * t.red,
        "bonus": np.asarray(bonus, dtype=float),
        "defcon": np.where(dc_hit, t.defcon_points, 0),
    }
    return {k: np.where(played, np.asarray(v, dtype=float), 0.0) for k, v in out.items()}


def _allocate(rng: np.random.Generator, n: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Multinomial allocation of n[s] events over players with weights[s, j] (+ an unattributed remainder).
    Returns counts (S, J); each row sums to <= n[s]."""
    total = weights.sum(axis=1, keepdims=True)
    scale = np.maximum(total, 1.0) * (1.0 + 1e-9)
    p = weights / scale
    pvals = np.concatenate([p, np.clip(1.0 - p.sum(axis=1, keepdims=True), 0.0, None)], axis=1)
    pvals = pvals / pvals.sum(axis=1, keepdims=True)
    return rng.multinomial(n, pvals)[:, :-1]


def simulate_gw(
    rows: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    n_sims: int,
    rng: np.random.Generator,
    season: str,
    assist_rate: float,
    max_goals: int = 10,
    max_card_prob: float = 0.95,
    bonus_sim: str = "round",
    bonus_link: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Simulate one GW. rows: one per player-fixture with ROW_COLUMNS; fixtures: FIXTURE_COLUMNS.
    Returns arrays of shape (R, S): points, minutes, goals, assists, bonus, cs_points (0/1), goals_conceded,
    saves, yellow_cards, red_cards, defensive_contribution; and fixture_goals (fixtures, 2, S).
    bonus_link (bonus_sim "linked"): multipliers [element_type, min(goals, 2), min(assists, 2), CS]."""
    S = n_sims
    R = len(rows)
    rows = rows.reset_index(drop=True)
    # 1. scores
    fx_index = {int(f): i for i, f in enumerate(fixtures["fixture"])}
    goals = np.zeros((len(fixtures), 2, S), dtype=np.int64)  # [:, 0] home goals, [:, 1] away goals
    n = max_goals + 1
    for i, f in enumerate(fixtures.itertuples(index=False)):
        m = score_matrix(f.lambda_home, f.lambda_away, f.rho, max_goals).ravel()
        flat = rng.choice(n * n, size=S, p=m / m.sum())
        goals[i, 0], goals[i, 1] = flat // n, flat % n
    fi = rows["fixture"].map(fx_index).to_numpy(int)
    side = np.where(rows["was_home"].astype(bool).to_numpy(), 0, 1)
    goals_for = goals[fi, side]  # (R, S)
    goals_against = goals[fi, 1 - side]

    # 2. minutes
    p0, p1 = rows["p0"].to_numpy(float)[:, None], rows["p1"].to_numpy(float)[:, None]
    u = rng.random((R, S))
    bins = np.where(u < p0, 0, np.where(u < p0 + p1, 1, 2))
    m_short = np.rint(rows["m_short"].to_numpy(float))[:, None]
    m_long = np.rint(rows["m_long"].to_numpy(float))[:, None]
    minutes = np.where(bins == 0, 0.0, np.where(bins == 1, m_short, m_long))
    played = minutes > 0
    frac = minutes / 90.0

    # 3. goals and assists, per team-fixture
    g = np.zeros((R, S), dtype=np.int64)
    a = np.zeros((R, S), dtype=np.int64)
    share_g = rows["share_g"].to_numpy(float)[:, None]
    share_a = rows["share_a"].to_numpy(float)[:, None]
    team_key = fi * 2 + side
    for k in np.unique(team_key):
        idx = np.flatnonzero(team_key == k)
        n_goals = goals_for[idx[0]]
        if not n_goals.any():
            continue
        g[idx] = _allocate(rng, n_goals, (share_g[idx] * frac[idx]).T).T
        n_assisted = rng.binomial(n_goals, assist_rate)
        a[idx] = _allocate(rng, n_assisted, (share_a[idx] * frac[idx]).T).T

    # 4. clean sheets and goals conceded
    long_game = minutes >= 60
    cs = (goals_against == 0) & long_game
    thinned = rng.binomial(goals_against, np.clip(frac, 0.0, 1.0))
    gc = np.where(long_game, goals_against, thinned) * played

    # 5. saves, cards, defensive contribution
    saves = rng.poisson(rows["saves_mu90"].to_numpy(float)[:, None] * frac) * played
    yc = (
        rng.random((R, S)) < np.clip(rows["yc90"].to_numpy(float)[:, None] * frac, 0, max_card_prob)
    ) & played
    rc = (
        rng.random((R, S)) < np.clip(rows["rc90"].to_numpy(float)[:, None] * frac, 0, max_card_prob)
    ) & played
    thr = rows["dc_threshold"].to_numpy(float)
    has_thr = ~np.isnan(thr)
    dc90 = rows["dc90"].to_numpy(float)
    p_dc = np.zeros((R, S))
    if has_thr.any():
        t = np.where(has_thr, thr, 1.0)[:, None]
        p_dc = np.where(has_thr[:, None], poisson.sf(t - 1, dc90[:, None] * frac), 0.0)
    dc_hit = (rng.random((R, S)) < p_dc) & played
    dc = np.where(dc_hit, np.where(has_thr, thr, 0.0)[:, None], 0.0)

    # 6. bonus: expectation for the bin -> integer (ModelConfig.bonus_sim)
    b = np.where(
        bins == 1, rows["bonus_short"].to_numpy(float)[:, None], rows["bonus_long"].to_numpy(float)[:, None]
    )
    b = np.where(played, np.clip(b, 0.0, 3.0), 0.0)
    if bonus_sim not in ("round", "poisson", "linked"):
        raise ValueError(f"unknown bonus_sim {bonus_sim!r}")
    if bonus_sim == "poisson":
        bonus = np.minimum(rng.poisson(b), 3) * played
    else:
        if bonus_sim == "linked":
            link = np.ones((5, 3, 3, 2)) if bonus_link is None else bonus_link
            et_idx = rows["element_type"].to_numpy(int)[:, None]
            m = link[et_idx, np.minimum(g, 2), np.minimum(a, 2), cs.astype(np.int64)]
            n_play = played.sum(axis=1, keepdims=True)
            mean_m = np.where(played, m, 0.0).sum(axis=1, keepdims=True) / np.maximum(n_play, 1)
            b = np.where(played, np.clip(b * m / np.where(mean_m > 0, mean_m, 1.0), 0.0, 3.0), 0.0)
        bonus = np.floor(b) + (rng.random((R, S)) < (b - np.floor(b)))

    # 7. score with the rules engine
    et = np.broadcast_to(rows["element_type"].to_numpy(int)[:, None], (R, S))
    stack = [et, minutes, g, a, cs, gc, saves, yc, rc, bonus, dc]
    lines = np.stack([np.asarray(x).astype(np.int64).ravel() for x in stack], axis=1)
    points = score_lines(lines, season).reshape(R, S)
    return {
        "points": points,
        "minutes": minutes,
        "goals": g,
        "assists": a,
        "bonus": bonus,
        "cs_points": cs.astype(np.int64),
        "goals_conceded": gc,
        "saves": saves,
        "yellow_cards": yc.astype(np.int64),
        "red_cards": rc.astype(np.int64),
        "defensive_contribution": dc,
        "fixture_goals": goals,  # (fixtures, 2, S): home goals, away goals
    }


def aggregate(rows: pd.DataFrame, sim: dict[str, np.ndarray]) -> tuple[pd.DataFrame, np.ndarray]:
    """Sum player-fixture simulations to one row per element (double GWs summed). Returns (summary, points
    array (E, S)) with the summary columns xp, p20, p50, p80, p_play, p_60, p_cs, exp_goals, exp_assists,
    exp_bonus, exp_minutes, n_fixtures."""
    elements = rows["element"].to_numpy()
    order = np.argsort(elements, kind="stable")
    el_sorted = elements[order]
    uniq, starts = np.unique(el_sorted, return_index=True)

    def total(x: np.ndarray) -> np.ndarray:
        return np.add.reduceat(x[order], starts, axis=0)

    pts = total(sim["points"])
    minutes = total(sim["minutes"])
    any_play = total((sim["minutes"] > 0).astype(np.int64)) > 0
    any_60 = total((sim["minutes"] >= 60).astype(np.int64)) > 0
    any_cs = total(sim["cs_points"]) > 0
    q = np.quantile(pts, [0.2, 0.5, 0.8], axis=1, method="inverted_cdf")
    summary = pd.DataFrame(
        {
            "element": uniq,
            "xp": pts.mean(axis=1),
            "p20": q[0],
            "p50": q[1],
            "p80": q[2],
            "p_play": any_play.mean(axis=1),
            "p_60": any_60.mean(axis=1),
            "p_cs": any_cs.mean(axis=1),
            "exp_goals": total(sim["goals"]).mean(axis=1),
            "exp_assists": total(sim["assists"]).mean(axis=1),
            "exp_bonus": total(sim["bonus"]).mean(axis=1),
            "exp_minutes": minutes.mean(axis=1),
            "n_fixtures": np.diff(np.r_[starts, len(el_sorted)]),
        }
    )
    return summary, pts
