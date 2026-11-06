"""Diagnostic replay variants (step 4d attribution and tie-break sensitivity). DIAGNOSTICS ONLY: nothing here
changes a default; the frozen reference is baseline-v0 (replay/baselines/baseline_v0.json).

A Variant switches factors of the blind replay one at a time:
    model_flags=False  the model's availability cap is disabled: the model is predicted with status "a" and no
                       chance_of_playing_next_round (forecast.model_targets otherwise passes the flags at T)
    form_flags=False   the form baseline does not zero players flagged unavailable at T
    chips=False        the planner policies hold every chip (allowed_chips = ())
    imposed_start      GW1 squad = the given 15 (e.g. FORM_RESTRAINED's baseline GW1 squad) for the planner
                       policies; line-up / captain / vice / bench from the policy's own GW1 xP (best legal
                       XI, top-xP captain outside the low-history set, second vice, reserve GK first) and the
                       shared GW1 chip rule (bench boost / triple captain only). A policy whose own GW1 squad
                       already equals the imposed one keeps its own package unchanged.
    fixtures="played"  every forecast uses the fixtures AS PLAYED (the season's last as-known fixture file,
                       whose event = the GW each match was finally played in; data/asknown/fixtures/<season>/
                       gw38): n_fixtures (free-hit trigger), the form baseline's fixture count and the model's
                       targets. The history the model is fitted on is unchanged (as known at T).
    noise_seed         tie-break perturbation: every expected-points value of every table + eps * U(-1, 1),
                       U drawn per (season, GW) from numpy default_rng((noise_seed, season start, gw)) over
                       the sorted keys (eps = 1e-6)
Model tables for model_flags=False / fixtures="played" need a refit at T: build_variant_tables() fits the
model ONCE per (season, GW) on exactly the baseline's inputs and predicts (a) as known with the cap (must
equal the frozen forecast cache: the self-check), (b) as known without the cap, (c) as played with the cap.
They are cached by input digest in data/replay_inputs/variant_cache/<season>/gwNN.pkl.
"""

from __future__ import annotations

import copy
import pickle
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..asknown.reader import DEFAULT_ROOT
from ..planner.lineup import bench_order, best_xi, captain_pair
from ..rules.common import SeasonRules
from . import chips as chip_rule
from . import forecast
from .engine import ManagerState, Package
from .forecast import Forecasts
from .inputs import INPUTS_DIR, ReplayInputs
from .policies import PlannerPolicy, Policy

VARIANT_CACHE = INPUTS_DIR / "variant_cache"
VARIANT_TABLES_VERSION = "1"
NOISE_EPS = 1e-6


@dataclass(frozen=True)
class Variant:
    name: str = "baseline"
    model_flags: bool = True
    form_flags: bool = True
    chips: bool = True
    imposed_start: bool = False
    fixtures: str = "known"  # "known" | "played"
    noise_seed: int | None = None
    noise_eps: float = NOISE_EPS
    note: str = field(default="", compare=False)

    def needs_model_tables(self) -> bool:
        return (not self.model_flags) or self.fixtures == "played"

    def describe(self) -> dict[str, Any]:
        return asdict(self)


BASELINE = Variant()
ATTRIBUTION: tuple[Variant, ...] = (
    BASELINE,
    Variant("model_flags_off", model_flags=False, note="(1) availability cap disabled for the model"),
    Variant("form_flags_off", form_flags=False, note="(1) form baseline keeps flagged players"),
    Variant("chips_off", chips=False, note="(2) no chips, both policies"),
    Variant("imposed_start", imposed_start=True, note="(3) GW1 squad = FORM_RESTRAINED's GW1 squad"),
    Variant("fixtures_played", fixtures="played", note="(4) fixtures as played"),
    Variant(
        "flags_off_chips_off",
        model_flags=False,
        form_flags=False,
        chips=False,
        note="(5) flags off + chips off",
    ),
    Variant(
        "flags_off_imposed",
        model_flags=False,
        form_flags=False,
        imposed_start=True,
        note="(6) flags off + imposed start",
    ),
)
TIEBREAK_SEEDS = (1, 2, 3, 4, 5)


def tiebreak(seed: int) -> Variant:
    return Variant(f"tiebreak_{seed}", noise_seed=seed, note=f"eps {NOISE_EPS} U(-1,1) noise, seed {seed}")


def by_name(name: str) -> Variant:
    for v in ATTRIBUTION:
        if v.name == name:
            return v
    if name.startswith("tiebreak_"):
        return tiebreak(int(name.split("_")[1]))
    raise KeyError(name)


# ================================================================================ model / fixture tables
def played_fixtures(season: str, gws: list[int], asknown_root: str | Path | None = None) -> pd.DataFrame:
    """Fixtures as played: the season's last as-known fixture file (event = the GW finally played in)."""
    root = Path(asknown_root or DEFAULT_ROOT) / "fixtures" / season
    last = max(int(p.stem[2:]) for p in root.glob("gw*.parquet"))
    f = pd.read_parquet(root / f"gw{last}.parquet").dropna(subset=["event"])
    f = f[f["event"].astype(int).isin(gws)]
    return (
        f.assign(event=f["event"].astype(int))
        .sort_values(["event", "kickoff", "fixture"])
        .reset_index(drop=True)
    )


def fixture_counts(players: pd.DataFrame, fx: pd.DataFrame, gws: list[int]) -> dict[tuple[int, int], int]:
    per_team: dict[tuple[int, int], int] = {}
    for g, h, a in zip(fx["event"], fx["team_h"].astype(int), fx["team_a"].astype(int), strict=True):
        per_team[(h, g)] = per_team.get((h, g), 0) + 1
        per_team[(a, g)] = per_team.get((a, g), 0) + 1
    return {
        (int(e), g): per_team.get((int(t), g), 0)
        for e, t in zip(players["element"], players["team"], strict=True)
        for g in gws
    }


def _predict(model, inp: ReplayInputs, players: pd.DataFrame, fx: pd.DataFrame, gws: list[int], flags: bool):
    from ..model.config import DEFAULT_CONFIG

    pl = players if flags else players.assign(status="a", chance=np.nan)
    targets = forecast.model_targets(inp, pl, fx)
    if targets.empty:
        return {}
    out = model.predict(targets, sims_frame=False, seed=DEFAULT_CONFIG.seed + int(inp.gw))
    x = out.xp
    return {
        (int(e), int(g)): float(v)
        for e, g, v in zip(x["element"], x["gw"], x["xp"].fillna(0.0), strict=True)
        if int(g) in gws
    }


def build_variant_tables(inp: ReplayInputs, asknown_root: str | Path | None = None) -> dict[str, Any]:
    """One model fit at T (the baseline's), three predictions (module docstring)."""
    from ..model.config import DEFAULT_CONFIG
    from ..model.pipeline import XPModel

    players = forecast.players_frame(inp)
    gws = forecast.horizon_gws(inp, forecast.FORECAST_HORIZON)
    fx_known = forecast.fixtures_frame(inp, gws)
    fx_played = played_fixtures(inp.season, gws, asknown_root)
    teams = sorted(inp.asknown.teams_at_T["name"].astype(str))
    model = XPModel(DEFAULT_CONFIG, scoring_season=inp.season).fit(
        inp.history, season=inp.season, ref_time=inp.issue_time, season_teams=teams
    )
    return {
        "version": VARIANT_TABLES_VERSION,
        "gws": gws,
        "model_known_capped": _predict(model, inp, players, fx_known, gws, True),
        "model_known_uncapped": _predict(model, inp, players, fx_known, gws, False),
        "model_played_capped": _predict(model, inp, players, fx_played, gws, True),
        "nfix_played": fixture_counts(players, fx_played, gws),
    }


def variant_tables_path(cache_dir: str | Path, season: str, gw: int) -> Path:
    return Path(cache_dir) / season / f"gw{gw:02d}.pkl"


def load_or_build_tables(
    inp: ReplayInputs, digest: str, cache_dir: str | Path = VARIANT_CACHE, build: bool = True
) -> dict[str, Any]:
    p = variant_tables_path(cache_dir, inp.season, inp.gw)
    if p.exists():
        with p.open("rb") as fh:
            cached = pickle.load(fh)  # noqa: S301 - our own cache file
        if cached.get("digest") == digest and cached.get("version") == VARIANT_TABLES_VERSION:
            return cached
    if not build:
        raise FileNotFoundError(f"variant tables missing for {inp.season} GW{inp.gw} ({p})")
    tables = {"digest": digest, **build_variant_tables(inp)}
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        pickle.dump(tables, fh)
    return tables


# ================================================================================ forecasts of a variant
def perturb(table: dict[tuple[int, int], float], seed: int, season: str, gw: int, eps: float):
    keys = sorted(table)
    rng = np.random.default_rng((int(seed), int(season[:4]), int(gw)))
    u = rng.uniform(-1.0, 1.0, size=len(keys))
    return {k: float(table[k]) + eps * float(x) for k, x in zip(keys, u, strict=True)}


def apply(v: Variant, fc: Forecasts, inp: ReplayInputs, tables: dict[str, Any] | None = None) -> Forecasts:
    """The variant's forecasts at T (a copy; `fc` = the frozen baseline forecasts of that (season, GW))."""
    if v == BASELINE or (v.model_flags and v.form_flags and v.fixtures == "known" and v.noise_seed is None):
        return fc
    out = copy.copy(fc)
    out.tables = dict(fc.tables)
    out.meta = {**fc.meta, "variant": v.name}
    played = v.fixtures == "played"
    if v.needs_model_tables():
        if tables is None:
            raise ValueError(f"variant {v.name} needs the variant model tables")
        if played and not v.model_flags:
            raise ValueError("fixtures played + model flags off is not a defined variant")
        out.tables["model"] = dict(tables["model_played_capped" if played else "model_known_uncapped"])
    if played:
        out.n_fixtures = dict(tables["nfix_played"])  # type: ignore[index]
    if played or not v.form_flags:
        players = fc.players if v.form_flags else fc.players.assign(unavailable=False)
        out.tables["baseline"] = forecast.baseline_xp(players, inp.history, out.n_fixtures, fc.gws)
    if v.noise_seed is not None:
        out.tables = {
            k: perturb(t, v.noise_seed, fc.season, fc.gw, v.noise_eps) for k, t in out.tables.items()
        }
    return out


# ================================================================================ policies of a variant
class VariantPlannerPolicy(PlannerPolicy):
    """MINIMAL_MODEL / FORM_RESTRAINED with the variant's chip and GW1 switches (planner untouched)."""

    def __init__(self, name: str, source: str, variant: Variant, imposed: list[int] | None) -> None:
        super().__init__(name, source)
        self.variant = variant
        self.imposed = sorted(int(e) for e in imposed) if imposed else None
        if not variant.chips:
            self.allowed_chips = ()

    def settings(self) -> dict[str, Any]:
        return {**super().settings(), "variant": self.variant.describe(), "imposed_start": self.imposed}

    def decide(self, fc: Forecasts, state: ManagerState, rules: SeasonRules, inputs: Any) -> Package:
        own = super().decide(fc, state, rules, inputs)
        if state.squad or self.imposed is None or sorted(own.transfers_in) == self.imposed:
            return own
        return self._imposed_gw1(fc, state, rules)

    def _imposed_gw1(self, fc: Forecasts, state: ManagerState, rules: SeasonRules) -> Package:
        self._events = []
        xp = fc.xp(self.source)
        et = fc.element_type()
        squad = list(self.imposed or [])
        score = {e: xp.get((e, fc.gw), 0.0) for e in squad}
        xi = best_xi(squad, et, score, rules.formation)
        cap, vice = captain_pair(xi, score, fc.low)
        bench = bench_order([e for e in squad if e not in xi], et, score)
        triggers = self.chip_triggers(fc, state, rules, bench, cap)
        triggers = {k: t for k, t in triggers.items() if k in ("bboost", "3xc")}
        chip, record = chip_rule.decide(
            rules, state.chips, fc.gw, state.last_fh_gw, triggers, self.allowed_chips
        )
        return Package(
            fc.season,
            fc.gw,
            squad,
            [],
            xi,
            bench,
            int(cap),
            int(vice),
            chip,
            0,
            score,
            {"status": "imposed", "fallback": False},
            [{"chip_rule": record}, {"imposed_start": squad}],
        )


def make_variant_policies(v: Variant, imposed: list[int] | None = None) -> dict[str, Policy]:
    if v.imposed_start and not imposed:
        raise ValueError(f"variant {v.name} needs the imposed GW1 squad")
    imp = imposed if v.imposed_start else None
    return {
        "minimal_model": VariantPlannerPolicy("minimal_model", "model", v, imp),
        "form_restrained": VariantPlannerPolicy("form_restrained", "baseline", v, imp),
    }


def with_name(v: Variant, name: str) -> Variant:
    return replace(v, name=name)
