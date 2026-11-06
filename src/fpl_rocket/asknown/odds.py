"""Premier League match odds from football-data.co.uk, normalised, inverted to Poisson goal rates, and
labelled
with whether they were available at the issue time T.

Source files (keyless; the site redirects, so redirects are followed): /mmz4281/<yyzz>/E0.csv per season and
/fixtures.csv (next round, all leagues, pre-match odds only). Column families present (verified 2026-11-06):
    Avg* (market average 1X2, O/U 2.5, Asian handicap) pre-match and closing: every season 2019-20 .. 2026-27
    BFE* (Betfair exchange) pre-match and closing: from 2024-25 only
    PS* (Pinnacle): up to 2025-26, absent in 2026-27
    HxG / AxG: 2026-27 only
Untrusted data: parsed with pandas only.

Collection time (football-data.co.uk notes.txt, verbatim): "Betting odds for weekend games are collected
Friday afternoons, and on Tuesday afternoons for midweek games." Implemented conservatively: a match on
Fri/Sat/Sun/Mon
is collected on the Friday on or before its date, a match on Tue/Wed/Thu on the Tuesday on or before it, at
COLLECT_HOUR_LATEST (18:00 Europe/London, the end of the afternoon). available_at_T = that time <= T of the
gameweek the match is played in. `available_at_T_noon` repeats the test with 12:00 (sensitivity only). Closing
odds are never available at T.

Inversion: the bookmaker margin is removed (proportional by default; Shin kept), then
(lambda_home, lambda_away)
are solved so an independent-Poisson score grid reproduces P(home), P(draw), P(away) and P(over 2.5) in least
squares; `fit_rmse` is the root-mean-square residual over the targets used.
"""

from __future__ import annotations

import io
from collections.abc import Sequence

import numpy as np
import pandas as pd

BASE = "https://www.football-data.co.uk"
SEASON_CODES = {
    "2019-20": "1920",
    "2020-21": "2021",
    "2021-22": "2122",
    "2022-23": "2223",
    "2023-24": "2324",
    "2024-25": "2425",
    "2025-26": "2526",
    "2026-27": "2627",
}
COLLECTION_SENTENCE = (
    "Betting odds for weekend games are collected Friday afternoons, "
    "and on Tuesday afternoons for midweek games."
)
COLLECT_HOUR_LATEST = 18
COLLECT_HOUR_NOON = 12
LONDON = "Europe/London"
MAX_GOALS = 10


def season_url(season: str) -> str:
    return f"{BASE}/mmz4281/{SEASON_CODES[season]}/E0.csv"


FIXTURES_URL = f"{BASE}/fixtures.csv"

# normalised name -> source column. Pre-match then closing ("c_").
COLUMNS = {
    "avg_h": "AvgH", "avg_d": "AvgD", "avg_a": "AvgA",
    "bfe_h": "BFEH", "bfe_d": "BFED", "bfe_a": "BFEA",
    "ps_h": "PSH", "ps_d": "PSD", "ps_a": "PSA",
    "avg_over25": "Avg>2.5", "avg_under25": "Avg<2.5",
    "bfe_over25": "BFE>2.5", "bfe_under25": "BFE<2.5",
    "ah_line": "AHh", "avg_ah_h": "AvgAHH", "avg_ah_a": "AvgAHA", "bfe_ah_h": "BFEAHH", "bfe_ah_a": "BFEAHA",
    "c_avg_h": "AvgCH", "c_avg_d": "AvgCD", "c_avg_a": "AvgCA",
    "c_bfe_h": "BFECH", "c_bfe_d": "BFECD", "c_bfe_a": "BFECA",
    "c_ps_h": "PSCH", "c_ps_d": "PSCD", "c_ps_a": "PSCA",
    "c_avg_over25": "AvgC>2.5", "c_avg_under25": "AvgC<2.5",
    "c_bfe_over25": "BFEC>2.5", "c_bfe_under25": "BFEC<2.5",
    "c_ah_line": "AHCh", "c_avg_ah_h": "AvgCAHH", "c_avg_ah_a": "AvgCAHA",
    "c_bfe_ah_h": "BFECAHH", "c_bfe_ah_a": "BFECAHA",
    "hxg": "HxG", "axg": "AxG",
}  # fmt: skip
RESULT_COLUMNS = {"result_home_goals": "FTHG", "result_away_goals": "FTAG", "result": "FTR"}

# football-data team name -> FPL team name (where they differ)
TEAM_ALIASES = {
    "Man United": "Man Utd",
    "Tottenham": "Spurs",
    "Sheffield United": "Sheffield Utd",
    "Nott'm Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "West Brom": "West Brom",
    "Leeds United": "Leeds",
}


# ======================================================================================== parsing
def parse_csv(raw: bytes | str, season: str | None = None, div: str = "E0") -> pd.DataFrame:
    """One football-data CSV (a season file or fixtures.csv) -> normalised rows of division `div`."""
    text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else raw.lstrip("﻿")
    df = pd.read_csv(io.StringIO(text), dtype=str, on_bad_lines="skip")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    if "Div" in df:
        df = df[df["Div"].astype(str).str.strip() == div]
    df = df.dropna(subset=["HomeTeam", "AwayTeam"]).reset_index(drop=True)
    out = pd.DataFrame(index=df.index)
    out["season"] = season
    date = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    out["date"] = date.dt.strftime("%Y-%m-%d")  # the file's match date (UK)
    time = df["Time"].fillna("15:00") if "Time" in df else pd.Series("15:00", index=df.index)
    local = pd.to_datetime(date.dt.strftime("%Y-%m-%d") + " " + time.str.strip(), errors="coerce")
    out["kickoff_utc"] = local.dt.tz_localize(
        LONDON, ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")
    out["home"] = df["HomeTeam"].str.strip()
    out["away"] = df["AwayTeam"].str.strip()
    for name, col in COLUMNS.items():
        out[name] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    for name, col in RESULT_COLUMNS.items():
        if col in df:
            out[name] = pd.to_numeric(df[col], errors="coerce") if name != "result" else df[col]
        else:
            out[name] = np.nan
    return out


def fpl_team_name(fd_name: str) -> str:
    return TEAM_ALIASES.get(fd_name, fd_name)


def _lookup(lut: dict[str, int], fd_name: str) -> int | None:
    """FPL id by alias / name, then with the suffixes FPL sometimes uses ("Hull City", "Ipswich Town")."""
    base = fpl_team_name(fd_name).lower()
    for cand in (base, f"{base} city", f"{base} town", f"{base} united"):
        if cand in lut:
            return lut[cand]
    return None


def map_teams(odds: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Add team_h / team_a (FPL ids) from a season's FPL teams (columns team, name, short_name)."""
    lut = {}
    for t, n, s in zip(teams["team"], teams["name"], teams.get("short_name", teams["name"]), strict=True):
        lut[str(n).lower()] = int(t)
        lut[str(s).lower()] = int(t)
    out = odds.copy()
    out["team_h"] = out["home"].map(lambda x: _lookup(lut, x)).astype("Int64")
    out["team_a"] = out["away"].map(lambda x: _lookup(lut, x)).astype("Int64")
    return out


def map_fixtures(odds: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """Add fixture id and gw (as played) by the (team_h, team_a) pair, unique within a season."""
    f = final.dropna(subset=["team_h", "team_a"])[["fixture", "event", "team_h", "team_a", "kickoff"]]
    f = f.rename(columns={"event": "gw", "kickoff": "fpl_kickoff_utc"})
    out = odds.drop(columns=[c for c in ("fixture", "gw", "fpl_kickoff_utc") if c in odds]).merge(
        f, on=["team_h", "team_a"], how="left"
    )
    out["fixture"] = out["fixture"].astype("Int64")
    out["gw"] = out["gw"].astype("Int64")
    return out


# =================================================================================== availability
def collection_time(kickoff_utc: pd.Series, hour: int = COLLECT_HOUR_LATEST) -> pd.Series:
    """Stated collection time per match: Fri/Sat/Sun/Mon -> Friday on or before; Tue/Wed/Thu -> Tuesday on
    or before; at `hour` London time."""
    local = pd.to_datetime(kickoff_utc, utc=True).dt.tz_convert(LONDON)
    dow = local.dt.dayofweek  # Mon=0 .. Sun=6
    weekend = dow.isin([4, 5, 6, 0])
    back_fri = (dow - 4) % 7
    back_tue = (dow - 1) % 7
    back = np.where(weekend, back_fri, back_tue)
    day = local.dt.normalize().dt.tz_localize(None) - pd.to_timedelta(back, unit="D")
    at = (day + pd.Timedelta(hours=hour)).dt.tz_localize(LONDON, ambiguous="NaT", nonexistent="shift_forward")
    return at.dt.tz_convert("UTC")


def label_availability(odds: pd.DataFrame, issue_times: pd.Series) -> pd.DataFrame:
    """`issue_times`: T per row (the T of the gameweek the match is played in)."""
    out = odds.copy()
    out["collected_at_assumed"] = collection_time(out["kickoff_utc"], COLLECT_HOUR_LATEST)
    noon = collection_time(out["kickoff_utc"], COLLECT_HOUR_NOON)
    t = pd.to_datetime(issue_times, utc=True)
    out["issue_time_utc"] = t
    out["available_at_T"] = (out["collected_at_assumed"] <= t).fillna(False).astype(bool)
    out["available_at_T_noon"] = (noon <= t).fillna(False).astype(bool)
    out["closing_available_at_T"] = False
    return out


# ===================================================================================== de-margin
def proportional(odds: Sequence[float]) -> np.ndarray:
    q = 1.0 / np.asarray(odds, dtype=float)
    return q / q.sum()


def shin(odds: Sequence[float], tol: float = 1e-12) -> tuple[np.ndarray, float]:
    """Shin's method: probabilities and the insider share z (bisection on sum p = 1)."""
    pi = 1.0 / np.asarray(odds, dtype=float)
    beta = pi.sum()
    if beta <= 1.0:  # no margin: nothing to remove
        return pi / beta, 0.0

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / beta) - z) / (2.0 * (1.0 - z))

    lo, hi = 0.0, 0.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if probs(mid).sum() > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    z = 0.5 * (lo + hi)
    p = probs(z)
    return p / p.sum(), z


METHODS = {"proportional": lambda o: proportional(o), "shin": lambda o: shin(o)[0]}


# ======================================================================================== Poisson
_G = np.arange(MAX_GOALS + 1)
_LOGFACT = np.array([np.sum(np.log(np.arange(1, k + 1))) for k in _G])


def _pois(lam: float) -> np.ndarray:
    p = np.exp(_G * np.log(lam) - lam - _LOGFACT)
    return p


def outcome_probs(lam_h: float, lam_a: float) -> np.ndarray:
    """(P home, P draw, P away, P over 2.5) of independent Poisson goals (grid 0..MAX_GOALS)."""
    grid = np.outer(_pois(lam_h), _pois(lam_a))
    grid /= grid.sum()
    h = np.tril(grid, -1).sum()
    d = np.trace(grid)
    a = np.triu(grid, 1).sum()
    i, j = np.meshgrid(_G, _G, indexing="ij")
    over = grid[(i + j) >= 3].sum()
    return np.array([h, d, a, over])


def fit_lambdas(p_home: float, p_draw: float, p_away: float, p_over: float | None = None):
    """Least-squares (lambda_home, lambda_away) for the 1X2 (and over-2.5 when given) probabilities.
    Returns (lam_h, lam_a, rmse, n_targets)."""
    from scipy.optimize import least_squares

    target = np.array([p_home, p_draw, p_away] + ([p_over] if p_over is not None else []), dtype=float)
    n = len(target)

    def resid(x):
        return outcome_probs(np.exp(x[0]), np.exp(x[1]))[:n] - target

    # start: total goals ~2.7, split by the 1X2 tilt
    tilt = np.clip(p_home - p_away, -0.9, 0.9)
    x0 = np.log([1.35 * (1 + tilt), 1.35 * (1 - tilt)])
    res = least_squares(resid, x0, bounds=(np.log(0.02), np.log(8.0)), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    lam = np.exp(res.x)
    rmse = float(np.sqrt(np.mean(res.fun**2)))
    return float(lam[0]), float(lam[1]), rmse, n


def invert_frame(df: pd.DataFrame, prefix: str = "avg", method: str = "proportional", closing: bool = False):
    """Implied probabilities and lambdas for one odds family (avg / bfe / ps), pre-match or closing."""
    c = "c_" if closing else ""
    tag = f"{c}{prefix}"
    cols_1x2 = [f"{c}{prefix}_h", f"{c}{prefix}_d", f"{c}{prefix}_a"]
    cols_ou = [f"{c}{prefix}_over25", f"{c}{prefix}_under25"]
    demargin = METHODS[method]
    rows = []
    for _, r in df.iterrows():
        o = r[cols_1x2].to_numpy(dtype=float)
        if not np.isfinite(o).all() or (o <= 1.0).any():
            rows.append({})
            continue
        p = demargin(o)
        ou = r[cols_ou].to_numpy(dtype=float) if all(x in df for x in cols_ou) else np.array([np.nan, np.nan])
        p_over = demargin(ou)[0] if np.isfinite(ou).all() and (ou > 1.0).all() else None
        lh, la, rmse, n = fit_lambdas(p[0], p[1], p[2], p_over)
        rows.append(
            {
                f"{tag}_p_h": p[0],
                f"{tag}_p_d": p[1],
                f"{tag}_p_a": p[2],
                f"{tag}_p_over25": np.nan if p_over is None else p_over,
                f"{tag}_margin": float((1.0 / o).sum() - 1.0),
                f"{tag}_lambda_h": lh,
                f"{tag}_lambda_a": la,
                f"{tag}_fit_rmse": rmse,
                f"{tag}_fit_targets": n,
            }
        )
    return pd.DataFrame(rows, index=df.index)


# ============================================================================== live (next round)
def fetch_next_round(
    fetcher=None, raw: bytes | None = None, fetched_at=None, issue_time=None
) -> pd.DataFrame:
    """The live pipeline's odds: download fixtures.csv now (or parse `raw` with its `fetched_at`), keep the
    Premier League, add the implied probabilities / lambdas of the market average, and stamp `fetched_at`
    (UTC). With `issue_time`, available_at_T = fetched_at <= issue_time (a file fetched after T is never used
    for that T). Not wired into the deadline pipeline in this step."""
    from .common import Fetcher, utcnow

    if raw is None:
        fetcher = fetcher or Fetcher()
        raw = fetcher.get(FIXTURES_URL)
        fetched_at = utcnow()
    if fetched_at is None:
        raise ValueError("fetched_at is required with raw content")
    df = parse_csv(raw, season=None)
    out = pd.concat([df, invert_frame(df, "avg")], axis=1) if len(df) else df
    out["fetched_at"] = pd.Timestamp(fetched_at)
    out["known_at"] = out["fetched_at"]
    if issue_time is not None:
        out["issue_time_utc"] = pd.Timestamp(issue_time)
        out["available_at_T"] = out["fetched_at"] <= pd.Timestamp(issue_time)
    return out
