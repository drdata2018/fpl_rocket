"""The season loop, the decision log and the built-in leakage test.

run_season(): for each GW of the season (deadline status ok): inputs = load_inputs(season, gw) (asknown.load
+ the points history known at T); the contract check (no row stamped after T: ReplayInputs.violations() must
be empty, else the run stops); forecasts at T (cached on disk by input digest; identical when rebuilt); for
every policy: decide -> engine.step (validate, execute, score). Every decision is written to
<out>/<season>/<policy>.jsonl as two JSON lines: event "decision" (season, gw, T, deadline, input manifest
hashes, policy name / version / settings, seed, the state entering the GW, the package, solver status and
events) BEFORE scoring, then event "score" (the realised score row). A season / policy pair is complete when
<policy>.done exists; an incomplete pair is replayed from GW1 (resumable per season / policy). The state
entering every GW is pickled to <out>/<season>/states.pkl for the leakage test.

leak_test(): for K deadlines per season chosen with a fixed seed, every as-known file and the points history
are copied and every row known after T is corrupted (asknown.reader.corrupt_after, mode "mutate"); the inputs
and forecasts are rebuilt from the copy WITHOUT the cache and every policy decides again from its recorded
state: the package (transfers, XI, bench, captain, vice, chip, hits) must be identical, and so must the input
digests.
"""

from __future__ import annotations

import json
import pickle
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..asknown.reader import DEFAULT_ROOT
from ..model.config import DEFAULT_CONFIG, MODEL_BASE_VERSION
from ..rules import season_rules
from . import engine, forecast
from .inputs import HistoryStore, corrupt_copy, load_inputs
from .policies import POLICY_NAMES, Policy, make_policies

LEAK_K = 3
LEAK_SEED = 20261106


def season_gws(
    season: str, asknown_root: str | Path | None = None, gws: list[int] | None = None
) -> list[int]:
    dl = pd.read_parquet(Path(asknown_root or DEFAULT_ROOT) / "deadlines.parquet")
    ok = sorted(int(g) for g in dl.loc[(dl["season"] == season) & (dl["status"] == "ok"), "gw"])
    return [g for g in ok if gws is None or g in gws]


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple | set | frozenset):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if np.isnan(x) else float(x)
    if isinstance(x, float) and x != x:
        return None
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def _write(fh, rec: dict[str, Any]) -> None:
    fh.write(json.dumps(_jsonable(rec), sort_keys=True) + "\n")
    fh.flush()


def forecast_digest(inp) -> str:
    m = inp.manifest()
    return f"{m['asknown_digest']}-{m['history_digest']}-{MODEL_BASE_VERSION}-f{forecast.FORECAST_VERSION}"


def forecasts_for(inp, cache_dir: str | Path | None, require_cache: bool = False) -> forecast.Forecasts:
    if require_cache:
        p = forecast.cache_path(cache_dir, inp.season, inp.gw) if cache_dir is not None else None
        ok = p is not None and p.exists()
        if ok:
            with p.open("rb") as fh:
                ok = pickle.load(fh).get("digest") == forecast_digest(inp)  # noqa: S301 - our own cache
        if not ok:
            raise RuntimeError(
                f"{inp.season} GW{inp.gw}: forecast cache miss (digest {forecast_digest(inp)})"
            )
    return forecast.load_or_build(inp, cache_dir, forecast_digest(inp))


def run_season(
    season: str,
    out_dir: str | Path,
    store: HistoryStore,
    *,
    policy_names: tuple[str, ...] = POLICY_NAMES,
    asknown_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    gws: list[int] | None = None,
    log: Callable[[str], None] = print,
    require_cache: bool = False,
    policies: dict[str, Policy] | None = None,
    forecast_hook: Callable[[forecast.Forecasts, Any], forecast.Forecasts] | None = None,
    model_seed: Callable[[int], int] | None = None,
    rules_hook: Callable[[Any], Any] | None = None,
) -> pd.DataFrame:
    """`require_cache`: stop when a forecast is not in the cache (baseline verification). `policies` /
    `forecast_hook` (fc, inputs) -> fc: diagnostic variants only (fpl_rocket.replay.variants); the defaults
    are the frozen policies and forecasts. `model_seed` (gw -> seed): the Monte Carlo seed the hook's model
    table was simulated with, recorded in the decision log (seed diagnostic; default the config seed + gw).
    `rules_hook` (rules) -> rules: a diagnostic change of the season's rules (variants only)."""
    rules = season_rules(season)
    if rules_hook is not None:
        rules = rules_hook(rules)
    out = Path(out_dir) / season
    out.mkdir(parents=True, exist_ok=True)
    todo = [n for n in policy_names if not (out / f"{n}.done").exists()]
    rows_path = out / "rows.csv"
    if not todo:
        log(f"{season}: complete")
        return pd.read_csv(rows_path)
    policies = {n: p for n, p in (policies or make_policies()).items() if n in todo}
    states: dict[str, engine.ManagerState | None] = {n: None for n in todo}
    files = {n: (out / f"{n}.jsonl").open("w", encoding="utf-8") for n in todo}
    snapshots: dict[str, dict[int, engine.ManagerState]] = {n: {} for n in todo}
    rows: list[dict[str, Any]] = []
    try:
        for gw in season_gws(season, asknown_root, gws):
            t0 = time.perf_counter()
            inp = load_inputs(season, gw, store, asknown_root)
            viol = inp.violations()
            if viol:
                raise RuntimeError(f"{season} GW{gw}: inputs stamped after T: {viol}")
            manifest = inp.manifest()
            fc = forecasts_for(inp, cache_dir, require_cache)
            if forecast_hook is not None:
                fc = forecast_hook(fc, inp)
            results = store.results(season, gw)
            line = []
            for name in todo:
                pol = policies[name]
                st = states[name] or engine.new_state(rules, inp.issue_time)
                engine.sync_chips(st, rules, inp.issue_time)
                snapshots[name][gw] = st.copy()
                ts = time.perf_counter()
                pkg = pol.decide(fc, st, rules, inp)
                decide_s = time.perf_counter() - ts
                _write(
                    files[name],
                    {
                        "event": "decision",
                        "season": season,
                        "gw": gw,
                        "issue_time": inp.issue_time,
                        "deadline": inp.deadline,
                        "inputs": manifest,
                        "policy": {"name": name, "version": pol.version, "settings": pol.settings()},
                        "model": {
                            "version": MODEL_BASE_VERSION,
                            "seed": DEFAULT_CONFIG.seed + gw if model_seed is None else int(model_seed(gw)),
                        }
                        if pol.source == "model"
                        else None,
                        "rules_version": rules.version,
                        "state_before": st.to_json(),
                        "package": pkg.to_json(),
                        "decide_seconds": round(decide_s, 2),
                    },
                )
                new, row = engine.step(rules, st, fc.players, pkg, results)
                row = {"policy": name, **row, "fallback": bool(pkg.solver.get("fallback", False))}
                row["events"] = ";".join(
                    n.get("event", "") for n in pkg.notes if isinstance(n, dict) and n.get("event")
                )
                _write(
                    files[name], {"event": "score", "season": season, "gw": gw, "policy": name, "row": row}
                )
                states[name] = new
                rows.append(row)
                line.append(f"{name} {row['net']}{'+' + row['chip'] if row['chip'] else ''}")
            log(f"{season} GW{gw} ({time.perf_counter() - t0:.0f} s): " + ", ".join(line))
    finally:
        for fh in files.values():
            fh.close()
    snap_path = out / "states.pkl"
    old = {}
    if snap_path.exists():
        with snap_path.open("rb") as fh:
            old = pickle.load(fh)  # noqa: S301 - our own file
    old.update(snapshots)
    with snap_path.open("wb") as fh:
        pickle.dump(old, fh)
    df = pd.DataFrame(rows)
    if rows_path.exists():
        prev = pd.read_csv(rows_path)
        df = pd.concat([prev[~prev["policy"].isin(todo)], df], ignore_index=True)
    df.to_csv(rows_path, index=False)
    for n in todo:
        (out / f"{n}.done").write_text("ok", encoding="utf-8")
    return df


def recorded_packages(out_dir: str | Path, season: str, policy: str) -> dict[int, dict[str, Any]]:
    out = {}
    with (Path(out_dir) / season / f"{policy}.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["event"] == "decision":
                out[int(rec["gw"])] = rec
    return out


def leak_test(
    season: str,
    out_dir: str | Path,
    store_path: str | Path,
    *,
    k: int = LEAK_K,
    seed: int = LEAK_SEED,
    policy_names: tuple[str, ...] = POLICY_NAMES,
    asknown_root: str | Path | None = None,
    gws: list[int] | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    rules = season_rules(season)
    out = Path(out_dir) / season
    with (out / "states.pkl").open("rb") as fh:
        snaps = pickle.load(fh)  # noqa: S301 - our own file
    all_gws = season_gws(season, asknown_root, gws)
    rng = np.random.default_rng(seed + int(season[:4]))
    chosen = sorted(int(g) for g in rng.choice(all_gws, size=min(k, len(all_gws)), replace=False))
    recorded = {n: recorded_packages(out_dir, season, n) for n in policy_names}
    results = []
    root = Path(asknown_root or DEFAULT_ROOT)
    for gw in chosen:
        t0 = time.perf_counter()
        rec0 = recorded[policy_names[0]][gw]
        cutoff = pd.Timestamp(rec0["issue_time"])
        with tempfile.TemporaryDirectory(prefix="fpl_leak_") as tmp:
            ak, hist, changed = corrupt_copy(root, store_path, season, cutoff, tmp)
            store2 = HistoryStore(hist)
            inp = load_inputs(season, gw, store2, ak)
            viol = inp.violations()
            man = inp.manifest()
            fc = forecast.build(inp)
            policies = make_policies()
            for name in policy_names:
                pol: Policy = policies[name]
                st = snaps[name][gw].copy()
                pkg = pol.decide(fc, st, rules, inp)
                rec = recorded[name][gw]
                want = engine.Package(**{**rec["package"], "scores": {}}).key()
                same = pkg.key() == want
                same_inputs = (
                    man["asknown_digest"] == rec["inputs"]["asknown_digest"]
                    and man["history_digest"] == rec["inputs"]["history_digest"]
                )
                r = {
                    "season": season,
                    "gw": gw,
                    "issue_time": cutoff.isoformat(),
                    "policy": name,
                    "rows_corrupted": changed,
                    "violations": viol,
                    "inputs_identical": same_inputs,
                    "package_identical": same,
                    "passed": same and same_inputs and not viol,
                }
                if not same:
                    r["recorded"], r["rebuilt"] = want, pkg.key()
                results.append(r)
        log(
            f"leak test {season} GW{gw} ({time.perf_counter() - t0:.0f} s): "
            + ", ".join(
                f"{r['policy']} {'PASS' if r['passed'] else 'FAIL'}" for r in results[-len(policy_names) :]
            )
        )
    with (out / "leak_test.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            _write(fh, r)
    return results
