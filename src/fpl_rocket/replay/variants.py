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
    inputs="october"   (step 4e reconciliation) the October matched replay's inputs (matched.py, run_matched
                       --pred-cache data/backtests/pred_cache_3g): its retrospective per-fixture predictions
                       of both predictors x the archive's as-played fixture counts (a player keeps his last
                       per-fixture prediction, matched.horizon_xp), its low-history sets, and its buy pool
                       (players with an archive row in GW t). Defined only with fixtures="played" and both
                       flags off (October had neither); see october_forecast()
    captain_swap       (step 4e) captain and vice = captain_pair(the package's XI, the OTHER planner policy's
                       xP for GW t, the low-history set): MINIMAL_MODEL captains by the form baseline,
                       FORM_RESTRAINED by the model; the triple-captain trigger reads that other xP of the
                       swapped captain (everything else, bench boost included, is the policy's own)
Model tables for model_flags=False / fixtures="played" need a refit at T: build_variant_tables() fits the
model ONCE per (season, GW) on exactly the baseline's inputs and predicts (a) as known with the cap (must
equal the frozen forecast cache: the self-check), (b) as known without the cap, (c) as played with the cap,
(d) as played without the cap (step 4e).
They are cached by input digest in data/replay_inputs/variant_cache/<season>/gwNN.pkl; a cache file that lacks
a table a variant needs is rebuilt (all tables come from the same single fit).
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
from .policies import PlannerPolicy, Policy, _xp_gw

VARIANT_CACHE = INPUTS_DIR / "variant_cache"
VARIANT_TABLES_VERSION = "1"
NOISE_EPS = 1e-6
TABLE_KEYS = (
    "model_known_capped",
    "model_known_uncapped",
    "model_played_capped",
    "model_played_uncapped",
    "nfix_played",
)


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
    inputs: str = "asknown"  # "asknown" | "october"
    captain_swap: bool = False
    # the imposed GW1 squad: "form_gw1" = FORM_RESTRAINED's baseline-v0 GW1 squad (step 4d); "october" = the
    # October matched replay's start (matched.most_owned_initial_squad: the most-owned legal 15 by the
    # archive's GW1 `selected`, run_matched --start most_owned)
    imposed_rule: str = "form_gw1"
    # False: the season's free-transfer top-ups (2025-26 GW16, AFCON) are removed, as in October's replay,
    # which applied the 2026-27 transfer rules to every season (see rules_for)
    ft_topup: bool = True
    note: str = field(default="", compare=False)

    def needs_model_tables(self) -> bool:
        return self.inputs == "asknown" and ((not self.model_flags) or self.fixtures == "played")

    def table_keys(self) -> tuple[str, ...]:
        """The variant-cache tables this variant reads (empty: none)."""
        if not self.needs_model_tables():
            return ()
        played = self.fixtures == "played"
        model = f"model_{'played' if played else 'known'}_{'capped' if self.model_flags else 'uncapped'}"
        return (model, "nfix_played") if played else (model,)

    def changes_forecasts(self) -> bool:
        return not (
            self.model_flags
            and self.form_flags
            and self.fixtures == "known"
            and self.noise_seed is None
            and self.inputs == "asknown"
        )

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
_OCTOBER = {
    "model_flags": False,
    "form_flags": False,
    "chips": False,
    "imposed_start": True,
    "imposed_rule": "october",
}
# step 4e: October's matched conditions in the blind engine, then restored one at a time (a path of single
# replays, not additive contributions); flags_off_chips_off and chips_off (above) continue the path
RECONCILIATION: tuple[Variant, ...] = (
    Variant(
        "oct_inputs",
        fixtures="played",
        inputs="october",
        **_OCTOBER,
        note="October's inputs (predictions, low-history sets, buy pool), flags off, chips off, October's "
        "GW1 squad, fixtures as played",
    ),
    Variant(
        "oct_conditions",
        fixtures="played",
        **_OCTOBER,
        note="blind forecasts at T; flags off, chips off, October's GW1 squad, fixtures as played",
    ),
    Variant(
        "oct_conditions_known",
        **_OCTOBER,
        note="as oct_conditions with the fixtures as known at T",
    ),
)
RECONCILIATION_EXTRA: tuple[Variant, ...] = (
    Variant(
        "oct_inputs_no_topup",
        fixtures="played",
        inputs="october",
        ft_topup=False,
        **_OCTOBER,
        note="oct_inputs without the season's free-transfer top-up (October's transfer rule)",
    ),
)
CAPTAIN_SWAP = Variant(
    "captain_swap",
    captain_swap=True,
    note="captain / vice by the other planner policy's xP on the policy's own XI",
)


def tiebreak(seed: int) -> Variant:
    return Variant(f"tiebreak_{seed}", noise_seed=seed, note=f"eps {NOISE_EPS} U(-1,1) noise, seed {seed}")


def by_name(name: str) -> Variant:
    for v in (*ATTRIBUTION, *RECONCILIATION, *RECONCILIATION_EXTRA, CAPTAIN_SWAP):
        if v.name == name:
            return v
    if name.startswith("tiebreak_"):
        return tiebreak(int(name.split("_")[1]))
    raise KeyError(name)


def rules_for(v: Variant, rules: SeasonRules) -> SeasonRules:
    """The season's rules under the variant (only ft_topup=False changes them: no free-transfer top-up)."""
    if v.ft_topup or not rules.special.ft_topup:
        return rules
    return replace(rules, special=replace(rules.special, ft_topup={}))


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


def predict_table(
    model,
    inp: ReplayInputs,
    players: pd.DataFrame,
    fx: pd.DataFrame,
    gws: list[int],
    flags: bool,
    seed: int | None = None,
) -> dict[tuple[int, int], float]:
    """{(element, gw): xP} of a fitted model; seed None = the baseline's Monte Carlo seed (config seed +
    gw)."""
    from ..model.config import DEFAULT_CONFIG

    pl = players if flags else players.assign(status="a", chance=np.nan)
    targets = forecast.model_targets(inp, pl, fx)
    if targets.empty:
        return {}
    mc_seed = DEFAULT_CONFIG.seed + int(inp.gw) if seed is None else int(seed)
    out = model.predict(targets, sims_frame=False, seed=mc_seed)
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
        "model_known_capped": predict_table(model, inp, players, fx_known, gws, True),
        "model_known_uncapped": predict_table(model, inp, players, fx_known, gws, False),
        "model_played_capped": predict_table(model, inp, players, fx_played, gws, True),
        "model_played_uncapped": predict_table(model, inp, players, fx_played, gws, False),
        "nfix_played": fixture_counts(players, fx_played, gws),
    }


def variant_tables_path(cache_dir: str | Path, season: str, gw: int) -> Path:
    return Path(cache_dir) / season / f"gw{gw:02d}.pkl"


def load_or_build_tables(
    inp: ReplayInputs,
    digest: str,
    cache_dir: str | Path = VARIANT_CACHE,
    build: bool = True,
    need: tuple[str, ...] = TABLE_KEYS,
) -> dict[str, Any]:
    """The cached tables when the digest and version match and every table in `need` is present, else a
    rebuild (one fit, every table) when `build`."""
    p = variant_tables_path(cache_dir, inp.season, inp.gw)
    if p.exists():
        with p.open("rb") as fh:
            cached = pickle.load(fh)  # noqa: S301 - our own cache file
        if (
            cached.get("digest") == digest
            and cached.get("version") == VARIANT_TABLES_VERSION
            and all(k in cached for k in need)
        ):
            return cached
    if not build:
        raise FileNotFoundError(f"variant tables missing for {inp.season} GW{inp.gw} ({p}; need {need})")
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


def october_xp(
    per_fixture: dict[int, pd.Series], nfix: dict[int, dict[int, int]], t: int, gws: list[int]
) -> dict[tuple[int, int], float]:
    """matched.horizon_xp at GW t: every player's latest per-fixture prediction from GWs <= t x his fixtures
    in each GW of `gws` (a blank GW scores 0, a double twice)."""
    memory: dict[int, float] = {}
    for g in sorted(per_fixture):
        if g > t:
            break
        memory.update({int(e): float(v) for e, v in per_fixture[g].items()})
    out = {}
    for g in gws:
        for e, n in nfix.get(g, {}).items():
            if int(e) in memory and int(n) > 0:
                out[(int(e), g)] = memory[int(e)] * int(n)
    return out


def october_inputs(cache: dict[str, Any], archive: pd.DataFrame, season: str) -> dict[str, Any]:
    """The October replay's inputs of a season: `cache` = its prediction cache (matched.season_inputs:
    {"preds": {"model" | "baseline": {gw: per-fixture xP}}, "lows": {gw: set}}), `archive` = all_gws.parquet
    (as-played fixture counts and the buy pool per GW, matched.season_data)."""
    from ..matched import season_data

    data = season_data(archive, season)
    return {
        "preds": cache["preds"],
        "lows": cache["lows"],
        "nfix": {g: {int(e): int(n) for e, n in d["n_fixtures"].items()} for g, d in data.per_gw.items()},
        "pool": {g: frozenset(int(e) for e in d.index) for g, d in data.per_gw.items()},
    }


def october_forecast(fc: Forecasts, october: dict[str, Any]) -> Forecasts:
    """fc with October's tables (both predictors), low-history set, fixture counts and buy pool for GW t."""
    out = copy.copy(fc)
    nfix = october["nfix"]
    out.tables = {
        "model": october_xp(october["preds"]["model"], nfix, fc.gw, fc.gws),
        "baseline": october_xp(october["preds"]["baseline"], nfix, fc.gw, fc.gws),
    }
    out.low = frozenset(october["lows"].get(fc.gw, frozenset()))
    out.n_fixtures = {
        (int(e), g): int(nfix.get(g, {}).get(int(e), 0)) for e in fc.players["element"] for g in fc.gws
    }
    pool = october["pool"].get(fc.gw, frozenset())
    out.players = fc.players.assign(buyable=fc.players["buyable"] & fc.players["element"].isin(pool))
    out.meta = {**fc.meta, "inputs": "october"}
    return out


def apply(
    v: Variant,
    fc: Forecasts,
    inp: ReplayInputs,
    tables: dict[str, Any] | None = None,
    october: dict[str, Any] | None = None,
) -> Forecasts:
    """The variant's forecasts at T (a copy; `fc` = the frozen baseline forecasts of that (season, GW)).
    `october` = october_inputs() of the season (inputs="october" only)."""
    if v == BASELINE or not v.changes_forecasts():
        return fc
    played = v.fixtures == "played"
    if v.inputs == "october":
        if october is None:
            raise ValueError(f"variant {v.name} needs the October inputs")
        if not played or v.model_flags or v.form_flags or v.noise_seed is not None:
            raise ValueError("October inputs are defined with fixtures played, both flags off, no noise")
        out = october_forecast(fc, october)
        out.meta["variant"] = v.name
        return out
    out = copy.copy(fc)
    out.tables = dict(fc.tables)
    out.meta = {**fc.meta, "variant": v.name}
    if v.needs_model_tables():
        if tables is None:
            raise ValueError(f"variant {v.name} needs the variant model tables")
        out.tables["model"] = dict(tables[v.table_keys()[0]])
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
        self.other = "baseline" if source == "model" else "model"
        self._base_xi: list[int] | None = None
        if not variant.chips:
            self.allowed_chips = ()

    def settings(self) -> dict[str, Any]:
        out = {**super().settings(), "variant": self.variant.describe(), "imposed_start": self.imposed}
        if self.variant.captain_swap:
            out["captain_source"] = self.other
        return out

    def decide(self, fc: Forecasts, state: ManagerState, rules: SeasonRules, inputs: Any) -> Package:
        self._base_xi = None
        own = super().decide(fc, state, rules, inputs)
        if not (state.squad or self.imposed is None or sorted(own.transfers_in) == self.imposed):
            own = self._imposed_gw1(fc, state, rules)
        return self._swap_captain(fc, own)

    # ------------------------------------------------------------------ captain swap (step 4e)
    def _swap_scores(self, fc: Forecasts, xi: list[int]) -> dict[int, float]:
        other = _xp_gw(fc.xp(self.other), fc.gw)
        return {int(e): float(other.get(int(e), 0.0)) for e in xi}

    def _solve(self, fc, state, rules, chip_for_planner: str | None, wide: bool):
        plan = super()._solve(fc, state, rules, chip_for_planner, wide)
        if self._base_xi is None:  # the first solve of a GW = the plan the chip triggers read
            self._base_xi = [int(e) for e in plan.first().xi]
        return plan

    def chip_triggers(self, fc, state, rules, bench, captain) -> dict:
        trig = super().chip_triggers(fc, state, rules, bench, captain)
        if self.variant.captain_swap and self._base_xi is not None:
            score = self._swap_scores(fc, self._base_xi)
            cap = captain_pair(self._base_xi, score, fc.low)[0]
            tc = chip_rule.CHIP_RULE.tc_captain_xp
            trig["3xc"] = lambda: (score[cap] >= tc, score[cap])
        return trig

    def _swap_captain(self, fc: Forecasts, pkg: Package) -> Package:
        if not self.variant.captain_swap:
            return pkg
        cap, vice = captain_pair(pkg.xi, self._swap_scores(fc, pkg.xi), fc.low)
        pkg.notes.append(
            {"captain_swap": {"source": self.other, "own": [pkg.captain, pkg.vice], "swapped": [cap, vice]}}
        )
        pkg.captain, pkg.vice = int(cap), int(vice)
        return pkg

    def _imposed_gw1(self, fc: Forecasts, state: ManagerState, rules: SeasonRules) -> Package:
        self._events = []
        xp = fc.xp(self.source)
        et = fc.element_type()
        squad = list(self.imposed or [])
        score = {e: xp.get((e, fc.gw), 0.0) for e in squad}
        xi = best_xi(squad, et, score, rules.formation)
        cap, vice = captain_pair(xi, score, fc.low)
        bench = bench_order([e for e in squad if e not in xi], et, score)
        self._base_xi = [int(e) for e in xi]
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
