"""Team goals: Dixon-Coles bivariate Poisson with time decay and home advantage (docs/PLAN_v2.md section 4).

    log lambda_home = c + home + attack[h] + defence[a]
    log lambda_away = c        + attack[a] + defence[h]
    P(x, y) = tau(x, y) * Poisson(x; lambda_home) * Poisson(y; lambda_away)
    tau(0,0) = 1 - lambda mu rho, tau(0,1) = 1 + lambda rho, tau(1,0) = 1 + mu rho, tau(1,1) = 1 - rho

(`defence` is the goals-conceded effect: higher = leakier.) Fitted by weighted maximum likelihood
(scipy.optimize L-BFGS-B, analytic gradient), weight = exp(-xi * days before the reference time), xi = 0.0065
per day (ModelConfig.dc_xi). Identifiability: soft sum-to-zero on attack and defence. Promoted clubs (in the
target season, absent from the previous season) get a Gaussian prior centred on the mean attack / defence of
the previous season's bottom three, worth ModelConfig.dc_promoted_prior_matches matches.

Results come from the player rows themselves (results_from_rows): a team's goals in a fixture are its
players' goals plus the opponent's own goals (checked against fixtures.csv for 2022-23..2026-27: 0
mismatches).
Teams are keyed by NAME (FPL short-form names are stable across seasons; ids are season-local).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .config import DEFAULT_CONFIG, ModelConfig

GC_BINS = 7  # goals-conceded distribution columns p_gc_0 .. p_gc_6 (6 = six or more)


def previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{(start + 1) % 100:02d}"


def results_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, fixture): gw, kickoff_time, home, away, hg, ag - from player-fixture rows."""
    r = rows.dropna(subset=["fixture", "team", "was_home"])
    r = r.assign(
        was_home=r["was_home"].astype(bool),
        goals_scored=r["goals_scored"].fillna(0).astype("int64"),
        own_goals=r["own_goals"].fillna(0).astype("int64"),
    )
    keys = ["season", "fixture", "team", "was_home"]
    g = r.groupby(keys, as_index=False, sort=False).agg(
        gw=("gw", "first"),
        kickoff_time=("kickoff_time", "first"),
        goals_scored=("goals_scored", "sum"),
        own_goals=("own_goals", "sum"),
    )
    home = g[g["was_home"]].drop(columns="was_home")
    away = g[~g["was_home"]].drop(columns=["was_home", "gw", "kickoff_time"])
    m = home.merge(away, on=["season", "fixture"], suffixes=("_h", "_a"))
    out = pd.DataFrame(
        {
            "season": m["season"].astype(str),
            "fixture": m["fixture"].astype("int64"),
            "gw": m["gw"].astype("int64"),
            "kickoff_time": pd.to_datetime(m["kickoff_time"], utc=True),
            "home": m["team_h"].astype(str),
            "away": m["team_a"].astype(str),
            "hg": (m["goals_scored_h"] + m["own_goals_a"]).astype("int64"),
            "ag": (m["goals_scored_a"] + m["own_goals_h"]).astype("int64"),
        }
    )
    return out.sort_values(["kickoff_time", "fixture"], kind="stable").reset_index(drop=True)


def league_table(results: pd.DataFrame) -> pd.DataFrame:
    """Points (3/1/0), goal difference and goals for per team; sorted best first."""
    h = pd.DataFrame({"team": results["home"], "gf": results["hg"], "ga": results["ag"]})
    a = pd.DataFrame({"team": results["away"], "gf": results["ag"], "ga": results["hg"]})
    t = pd.concat([h, a], ignore_index=True)
    t["pts"] = np.select([t["gf"] > t["ga"], t["gf"] == t["ga"]], [3, 1], 0)
    out = t.groupby("team", as_index=False)[["pts", "gf", "ga"]].sum()
    out["gd"] = out["gf"] - out["ga"]
    return out.sort_values(["pts", "gd", "gf"], ascending=False, kind="stable").reset_index(drop=True)


def tau_matrix(lam: float, mu: float, rho: float, n: int) -> np.ndarray:
    t = np.ones((n, n))
    t[0, 0] = 1 - lam * mu * rho
    t[0, 1] = 1 + lam * rho
    t[1, 0] = 1 + mu * rho
    t[1, 1] = 1 - rho
    return np.clip(t, 1e-10, None)


def poisson_pmf(lam: float, n: int) -> np.ndarray:
    k = np.arange(n)
    return np.exp(k * np.log(max(lam, 1e-12)) - lam - gammaln(k + 1))


def score_matrix(
    lam: float, mu: float, rho: float, max_goals: int = DEFAULT_CONFIG.dc_max_goals
) -> np.ndarray:
    """P[home goals, away goals] for 0..max_goals each, Dixon-Coles corrected and renormalised."""
    n = max_goals + 1
    m = np.outer(poisson_pmf(lam, n), poisson_pmf(mu, n)) * tau_matrix(lam, mu, rho, n)
    return m / m.sum()


def score_log_prob(lam: float, mu: float, rho: float, hg: int, ag: int, max_goals: int = 10) -> float:
    """log P(exact score) under DC (rho) or independent Poisson (rho = 0)."""
    if hg > max_goals or ag > max_goals:  # beyond the matrix: plain Poisson
        return float(np.log(poisson_pmf(lam, hg + 1)[hg]) + np.log(poisson_pmf(mu, ag + 1)[ag]))
    return float(np.log(score_matrix(lam, mu, rho, max_goals)[hg, ag]))


@dataclass
class DixonColes:
    cfg: ModelConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    teams: list[str] = field(default_factory=list)
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    home: float = 0.0
    rho: float = 0.0
    promoted: list[str] = field(default_factory=list)
    bottom_three: list[str] = field(default_factory=list)
    prior: dict[str, float] = field(default_factory=dict)
    n_matches: int = 0
    converged: bool = False

    # ------------------------------------------------------------------ objective
    def objective(self, results: pd.DataFrame, ref_time: pd.Timestamp, season: str, season_teams):
        """(negative penalised log-likelihood with gradient, initial theta, bounds, teams, extras)."""
        cfg = self.cfg
        ref = pd.Timestamp(ref_time)
        res = results[results["kickoff_time"] < ref].reset_index(drop=True)
        prev = res[res["season"] == previous_season(season)]
        teams = sorted(set(res["home"]) | set(res["away"]) | set(season_teams or []))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        promoted: list[str] = []
        bottom: list[str] = []
        if season_teams and not prev.empty:
            prev_teams = set(prev["home"]) | set(prev["away"])
            promoted = sorted(t for t in season_teams if t not in prev_teams)
            bottom = list(league_table(prev)["team"].iloc[-3:])
        h = res["home"].map(idx).to_numpy(int)
        a = res["away"].map(idx).to_numpy(int)
        x = res["hg"].to_numpy(float)
        y = res["ag"].to_numpy(float)
        days = (ref - res["kickoff_time"]).dt.total_seconds().to_numpy(float) / 86400.0
        w = np.exp(-cfg.dc_xi * np.clip(days, 0, None))
        is00, is01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
        is10, is11 = (x == 1) & (y == 0), (x == 1) & (y == 1)
        p_idx = np.array([idx[t] for t in promoted], int)
        b_idx = np.array([idx[t] for t in bottom], int)
        mean_goals = float(np.average(np.r_[x, y], weights=np.r_[w, w])) if len(x) else 1.4
        # promoted prior precision: k matches x Fisher information of a log-rate (~ mean goals per match)
        tau_p = cfg.dc_promoted_prior_matches * mean_goals
        k = cfg.dc_sum_to_zero_weight
        ridge = cfg.dc_ridge

        def nll(theta: np.ndarray) -> tuple[float, np.ndarray]:
            att, dfc = theta[:n], theta[n : 2 * n]
            c, hadv, rho = theta[2 * n], theta[2 * n + 1], theta[2 * n + 2]
            ll_l = c + hadv + att[h] + dfc[a]
            ll_m = c + att[a] + dfc[h]
            lam, mu = np.exp(ll_l), np.exp(ll_m)
            tau = np.ones_like(lam)
            tau = np.where(is00, 1 - lam * mu * rho, tau)
            tau = np.where(is01, 1 + lam * rho, tau)
            tau = np.where(is10, 1 + mu * rho, tau)
            tau = np.where(is11, 1 - rho, tau)
            tau = np.clip(tau, 1e-10, None)
            ll = w * (np.log(tau) + x * ll_l - lam + y * ll_m - mu)
            dl = w * (x - lam + np.where(is00, -lam * mu * rho, np.where(is01, lam * rho, 0.0)) / tau)
            dm = w * (y - mu + np.where(is00, -lam * mu * rho, np.where(is10, mu * rho, 0.0)) / tau)
            drho_num = np.where(
                is00, -lam * mu, np.where(is01, lam, np.where(is10, mu, np.where(is11, -1.0, 0.0)))
            )
            drho = w * drho_num / tau
            g_att = np.bincount(h, dl, n) + np.bincount(a, dm, n)
            g_def = np.bincount(a, dl, n) + np.bincount(h, dm, n)
            f = -float(ll.sum())
            grad = -np.r_[g_att, g_def, dl.sum() + dm.sum(), dl.sum(), drho.sum()]
            f += k * (att.mean() ** 2 + dfc.mean() ** 2) + ridge * float(att @ att + dfc @ dfc)
            grad[:n] += 2 * k * att.mean() / n + 2 * ridge * att
            grad[n : 2 * n] += 2 * k * dfc.mean() / n + 2 * ridge * dfc
            if len(p_idx) and len(b_idx):
                for vec, off in ((att, 0), (dfc, n)):
                    d = vec[p_idx] - vec[b_idx].mean()
                    f += 0.5 * tau_p * float(d @ d)
                    grad[off + p_idx] += tau_p * d
                    grad[off + b_idx] -= tau_p * d.sum() / len(b_idx)
            return f, grad

        theta0 = np.zeros(2 * n + 3)
        theta0[2 * n] = np.log(max(mean_goals, 0.1))
        theta0[2 * n + 1] = 0.25
        bounds = [(-3.0, 3.0)] * (2 * n) + [(-3.0, 3.0), (-1.0, 1.0), cfg.dc_rho_bounds]
        extras: dict[str, Any] = {
            "promoted": promoted,
            "bottom": bottom,
            "b_idx": b_idx,
            "n_matches": len(res),
        }
        return nll, theta0, bounds, teams, extras

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        results: pd.DataFrame,
        *,
        ref_time: pd.Timestamp,
        season: str,
        season_teams: list[str] | None = None,
    ) -> DixonColes:
        """Weighted ML fit on `results` (results_from_rows layout) with kickoff strictly before ref_time.

        season = the target season; season_teams = its clubs (adds clubs without results yet, e.g. promoted
        clubs before their first match)."""
        nll, theta0, bounds, teams, ex = self.objective(results, ref_time, season, season_teams)
        n = len(teams)
        if ex["n_matches"] == 0:
            theta, self.converged = theta0, True
        else:
            sol = minimize(nll, theta0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000})
            theta, self.converged = sol.x, bool(sol.success)
        att, dfc = theta[:n], theta[n : 2 * n]
        self.teams, self.promoted, self.bottom_three = teams, ex["promoted"], ex["bottom"]
        self.n_matches = ex["n_matches"]
        self.attack = dict(zip(teams, att.tolist(), strict=True))
        self.defence = dict(zip(teams, dfc.tolist(), strict=True))
        self.intercept, self.home, self.rho = (
            float(theta[2 * n]),
            float(theta[2 * n + 1]),
            float(theta[2 * n + 2]),
        )
        b_idx = ex["b_idx"]
        self.prior = (
            {"attack": float(att[b_idx].mean()), "defence": float(dfc[b_idx].mean())} if len(b_idx) else {}
        )
        return self

    # ------------------------------------------------------------------ predict
    def _att(self, team: str) -> float:
        return self.attack.get(team, self.prior.get("attack", 0.0))

    def _def(self, team: str) -> float:
        return self.defence.get(team, self.prior.get("defence", 0.0))

    def rates(self, home: str, away: str) -> tuple[float, float]:
        """(lambda_home, lambda_away) for one fixture (an unknown club gets the promoted prior)."""
        lh = self.intercept + self.home + self._att(home) + self._def(away)
        la = self.intercept + self._att(away) + self._def(home)
        return float(np.exp(lh)), float(np.exp(la))

    def fixture_frame(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Per fixture (columns home, away + passthrough): lambda_home, lambda_away, rho, p_cs_home,
        p_cs_away, p_home_gc_0..6 (goals conceded by the home team), p_away_gc_0..6."""
        rows = []
        for home, away in zip(fixtures["home"], fixtures["away"], strict=True):
            lam, mu = self.rates(home, away)
            m = score_matrix(lam, mu, self.rho, self.cfg.dc_max_goals)
            conceded_home = m.sum(axis=0)  # away goals
            conceded_away = m.sum(axis=1)  # home goals
            row = {"lambda_home": lam, "lambda_away": mu, "rho": self.rho}
            row["p_cs_home"], row["p_cs_away"] = float(conceded_home[0]), float(conceded_away[0])
            for k in range(GC_BINS):
                last = k == GC_BINS - 1
                row[f"p_home_gc_{k}"] = float(conceded_home[k:].sum() if last else conceded_home[k])
                row[f"p_away_gc_{k}"] = float(conceded_away[k:].sum() if last else conceded_away[k])
            rows.append(row)
        out = pd.DataFrame(rows, index=fixtures.index)
        return pd.concat([fixtures, out], axis=1)


def team_fixture_frame(fx: pd.DataFrame) -> pd.DataFrame:
    """Two rows per fixture (one per team) - the model fixture-difficulty grid: xg_for, xg_against, p_cs and
    the goals-conceded distribution p_gc_0..6 from that team's point of view."""
    keep = [c for c in ("season", "gw", "fixture", "kickoff_time") if c in fx]
    sides = []
    for side, other, home in (("home", "away", True), ("away", "home", False)):
        part = fx[keep].copy()
        part["team"] = fx[side].to_numpy()
        part["opponent"] = fx[other].to_numpy()
        part["was_home"] = home
        part["xg_for"] = fx[f"lambda_{side}"].to_numpy()
        part["xg_against"] = fx[f"lambda_{other}"].to_numpy()
        part["p_cs"] = fx[f"p_cs_{side}"].to_numpy()
        for k in range(GC_BINS):
            part[f"p_gc_{k}"] = fx[f"p_{side}_gc_{k}"].to_numpy()
        sides.append(part)
    out = pd.concat(sides, ignore_index=True)
    order = [c for c in ("gw", "fixture") if c in out] + ["was_home"]
    return out.sort_values(order, ascending=[True] * (len(order) - 1) + [False], kind="stable").reset_index(
        drop=True
    )


@dataclass
class NaiveGoals:
    """'League-average goals': independent Poisson with the training mean home / away goals (unweighted)."""

    home_mean: float = 1.5
    away_mean: float = 1.2

    def fit(self, results: pd.DataFrame, *, ref_time: pd.Timestamp) -> NaiveGoals:
        r = results[results["kickoff_time"] < pd.Timestamp(ref_time)]
        if len(r):
            self.home_mean, self.away_mean = float(r["hg"].mean()), float(r["ag"].mean())
        return self
