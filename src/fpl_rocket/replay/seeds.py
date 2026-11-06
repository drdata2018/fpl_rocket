"""Seed diagnostic (step 4e): COMPUTATIONAL randomness of MINIMAL_MODEL only. DIAGNOSTICS ONLY - nothing here
changes a default; the frozen reference is baseline-v0.

What the seed controls: XPModel.predict(seed=...) seeds the numpy generator of the Monte Carlo simulation
(model.simulate.simulate_gw: the joint draws per player-fixture behind every expected-points value). The fit
does not read it: Dixon-Coles, rates, the newcomer prior and the bonus link are deterministic, and the two
boosted models (minutes classifier / regressor, bonus regressor) use random_state=0 with early stopping off.
So the fit is fixed and only the simulation varies. Baseline-v0: seed = config seed (20261023) + GW, n_sims
1000 (unchanged here).

Seed index k (k = 0 is baseline-v0): mc_seed(k, gw) = config seed + SEED_STRIDE * k + gw.

seed tables: ONE fit per (season, GW) on exactly the baseline's inputs at T (forecast.model_xp's fit), then
one prediction per seed index; k = 0 must equal the frozen forecast cache (the self-check). Cached by the
forecast input digest in data/replay_inputs/seed_cache/<season>/gwNN.pkl = {"digest", "version", "n_sims",
"tables": {k: {(element, gw): xP}}}: its own directory and payload, so it can never be read as (or
overwrite) the frozen forecast cache; later steps reuse it per seed index.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from . import forecast
from .forecast import Forecasts
from .inputs import INPUTS_DIR, ReplayInputs

SEED_CACHE = INPUTS_DIR / "seed_cache"
SEED_TABLES_VERSION = "1"
SEED_STRIDE = 100_000


def mc_seed(k: int, gw: int) -> int:
    from ..model.config import DEFAULT_CONFIG

    return int(DEFAULT_CONFIG.seed) + SEED_STRIDE * int(k) + int(gw)


def seed_tables_path(cache_dir: str | Path, season: str, gw: int) -> Path:
    return Path(cache_dir) / season / f"gw{gw:02d}.pkl"


def build_seed_tables(inp: ReplayInputs, ks: list[int]) -> dict[int, dict[tuple[int, int], float]]:
    """One fit at T (the baseline's), one capped as-known prediction per seed index."""
    from ..model.config import DEFAULT_CONFIG
    from ..model.pipeline import XPModel
    from .variants import predict_table

    players = forecast.players_frame(inp)
    gws = forecast.horizon_gws(inp, forecast.FORECAST_HORIZON)
    fx = forecast.fixtures_frame(inp, gws)
    teams = sorted(inp.asknown.teams_at_T["name"].astype(str))
    model = XPModel(DEFAULT_CONFIG, scoring_season=inp.season).fit(
        inp.history, season=inp.season, ref_time=inp.issue_time, season_teams=teams
    )
    return {int(k): predict_table(model, inp, players, fx, gws, True, mc_seed(k, inp.gw)) for k in ks}


def load_or_build(
    inp: ReplayInputs,
    digest: str,
    ks: list[int],
    cache_dir: str | Path = SEED_CACHE,
    build: bool = True,
) -> dict[int, dict[tuple[int, int], float]]:
    """{k: table} for every k in `ks`; missing seed indices are added to the cache (one refit)."""
    from ..model.config import DEFAULT_CONFIG

    p = seed_tables_path(cache_dir, inp.season, inp.gw)
    cached: dict[str, Any] = {}
    if p.exists():
        with p.open("rb") as fh:
            cached = pickle.load(fh)  # noqa: S301 - our own cache file
        if cached.get("digest") != digest or cached.get("version") != SEED_TABLES_VERSION:
            cached = {}
    have = dict(cached.get("tables", {}))
    missing = [int(k) for k in ks if int(k) not in have]
    if missing:
        if not build:
            raise FileNotFoundError(f"seed tables {missing} missing for {inp.season} GW{inp.gw} ({p})")
        have.update(build_seed_tables(inp, missing))
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(
                {
                    "digest": digest,
                    "version": SEED_TABLES_VERSION,
                    "n_sims": DEFAULT_CONFIG.n_sims,
                    "stride": SEED_STRIDE,
                    "tables": have,
                },
                fh,
            )
    return {int(k): have[int(k)] for k in ks}


def with_model_table(fc: Forecasts, table: dict[tuple[int, int], float], k: int) -> Forecasts:
    """fc with the model table of seed index k (a copy; the baseline table and everything else unchanged)."""
    import copy

    out = copy.copy(fc)
    out.tables = {**fc.tables, "model": dict(table)}
    out.meta = {**fc.meta, "seed_index": int(k), "mc_seed": mc_seed(k, fc.gw)}
    return out


# ================================================================================ package comparison
def package_changes(base: dict[str, Any], other: dict[str, Any]) -> dict[str, bool]:
    """Which parts of two packages (decision-log `package` dicts) differ: transfers (in / out sets), captain,
    XI / bench (XI set or bench order; vice counted with the captain), chip."""
    tr = sorted(base["transfers_in"]) != sorted(other["transfers_in"]) or sorted(
        base["transfers_out"]
    ) != sorted(other["transfers_out"])
    cap = base["captain"] != other["captain"] or base["vice"] != other["vice"]
    xi = sorted(base["xi"]) != sorted(other["xi"]) or list(base["bench"]) != list(other["bench"])
    chip = base.get("chip") != other.get("chip")
    return {"transfers": tr, "captain": cap, "xi_bench": xi, "chip": chip, "any": tr or cap or xi or chip}


def se_by_count(totals: list[float], counts: tuple[int, ...] = (1, 2, 4, 8)) -> list[dict[str, Any]]:
    """Standard error of a seed-averaged total by the number of seeds m: sd / sqrt(m) with sd the sample sd of
    `totals` (ddof 1), and the empirical sd of the means of disjoint groups of m consecutive totals (when at
    least two groups exist)."""
    import numpy as np

    x = np.asarray(totals, float)
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
    out = []
    for m in counts:
        groups = [x[i : i + m].mean() for i in range(0, len(x) - m + 1, m)]
        emp = float(np.std(groups, ddof=1)) if len(groups) > 1 else float("nan")
        out.append({"m": m, "se": sd / np.sqrt(m), "groups": len(groups), "empirical_sd": emp})
    return out
