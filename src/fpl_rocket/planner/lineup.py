"""Line-ups without a solver: exact best XI by xP, bench order, and realised points with automatic subs.

Used by the planner (bench order, the solver-free fallback), the matched-policy backtest (realised points)
and the prospective registry (scoring). Formation limits come from bootstrap element_types
(squad_min_play / squad_max_play); FORMATION holds the 2026-27 values as the default.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

from ..rules.season_2026_27 import DEF, FWD, GK, MID

XI_SIZE = 11
FORMATION: dict[int, tuple[int, int]] = {GK: (1, 1), DEF: (3, 5), MID: (2, 5), FWD: (1, 3)}
OUTFIELD = (DEF, MID, FWD)


def formation_from_bootstrap(bootstrap: Mapping[str, Any] | None) -> dict[int, tuple[int, int]]:
    """{element_type: (min, max) in the XI} from bootstrap element_types, defaults where absent."""
    out = dict(FORMATION)
    for et in (bootstrap or {}).get("element_types", []) or []:
        t = int(et.get("id", 0))
        if t in out and et.get("squad_min_play") is not None and et.get("squad_max_play") is not None:
            out[t] = (int(et["squad_min_play"]), int(et["squad_max_play"]))
    return out


def formations(limits: Mapping[int, tuple[int, int]] = FORMATION) -> list[dict[int, int]]:
    """Every legal (GK, DEF, MID, FWD) count of an XI."""
    out = []
    ranges = [range(limits[t][0], limits[t][1] + 1) for t in (GK, DEF, MID, FWD)]
    for counts in product(*ranges):
        if sum(counts) == XI_SIZE:
            out.append(dict(zip((GK, DEF, MID, FWD), counts, strict=True)))
    return out


def _key(score: Mapping[int, float], e: int) -> tuple[float, int]:
    return (-float(score.get(e, 0.0)), e)  # deterministic: best first, then lower element id


def best_xi(
    squad: Sequence[int],
    element_type: Mapping[int, int],
    score: Mapping[int, float],
    limits: Mapping[int, tuple[int, int]] = FORMATION,
) -> list[int]:
    """The XI with the highest total score among every legal formation (exact). When the squad cannot field
    a legal formation (e.g. no goalkeeper), the best 11 regardless of formation."""
    by_type: dict[int, list[int]] = {t: [] for t in (GK, DEF, MID, FWD)}
    for e in squad:
        by_type.setdefault(int(element_type[e]), []).append(e)
    for t in by_type:
        by_type[t].sort(key=lambda e: _key(score, e))
    best: tuple[float, list[int]] | None = None
    for f in formations(limits):
        if any(len(by_type.get(t, [])) < n for t, n in f.items()):
            continue
        xi = [e for t, n in f.items() for e in by_type[t][:n]]
        total = sum(float(score.get(e, 0.0)) for e in xi)
        if best is None or total > best[0] + 1e-12:
            best = (total, xi)
    if best is None:
        return sorted(squad, key=lambda e: _key(score, e))[:XI_SIZE]
    return sort_xi(best[1], element_type, score)


def sort_xi(xi: Sequence[int], element_type: Mapping[int, int], score: Mapping[int, float]) -> list[int]:
    """GK, DEF, MID, FWD; within a position best score first."""
    return sorted(xi, key=lambda e: (int(element_type[e]), *_key(score, e)))


def bench_order(
    bench: Sequence[int], element_type: Mapping[int, int], score: Mapping[int, float]
) -> list[int]:
    """FPL bench: the reserve goalkeeper first, then outfield players by score (highest first)."""
    gks = sorted((e for e in bench if int(element_type[e]) == GK), key=lambda e: _key(score, e))
    out = sorted((e for e in bench if int(element_type[e]) != GK), key=lambda e: _key(score, e))
    return gks + out


def captain_pair(
    xi: Sequence[int], score: Mapping[int, float], excluded: set[int] | frozenset[int] = frozenset()
) -> tuple[int, int]:
    """(captain, vice) = the two best-scoring XI players not excluded (falls back to excluded ones only
    when fewer than two remain)."""
    ranked = sorted(xi, key=lambda e: _key(score, e))
    eligible = [e for e in ranked if e not in excluded]
    pool = eligible + [e for e in ranked if e in excluded]
    return pool[0], pool[1]


def _valid(counts: Counter, limits: Mapping[int, tuple[int, int]]) -> bool:
    return all(limits[t][0] <= counts.get(t, 0) <= limits[t][1] for t in (GK, DEF, MID, FWD))


def auto_subs(
    xi: Sequence[int],
    bench: Sequence[int],
    minutes: Mapping[int, int],
    element_type: Mapping[int, int],
    limits: Mapping[int, tuple[int, int]] = FORMATION,
) -> tuple[list[int], list[tuple[int, int]]]:
    """FPL automatic substitutions: each starter with 0 minutes is replaced by the first bench player (in
    bench order) who played and keeps the formation legal; a goalkeeper only by the reserve goalkeeper.
    Returns (final XI, [(out, in), ...])."""
    final = list(xi)
    used: set[int] = set()
    subs: list[tuple[int, int]] = []
    for starter in list(xi):
        if int(minutes.get(starter, 0)) > 0:
            continue
        for b in bench:
            if b in used or int(minutes.get(b, 0)) <= 0:
                continue
            gk_out, gk_in = int(element_type[starter]) == GK, int(element_type[b]) == GK
            if gk_out != gk_in:
                continue
            trial = [b if e == starter else e for e in final]
            if not _valid(Counter(int(element_type[e]) for e in trial), limits):
                continue
            final, used = trial, used | {b}
            subs.append((starter, b))
            break
    return final, subs


def realised_points(
    xi: Sequence[int],
    bench: Sequence[int],
    captain: int,
    vice: int,
    points: Mapping[int, int],
    minutes: Mapping[int, int],
    element_type: Mapping[int, int],
    *,
    chip: str | None = None,
    limits: Mapping[int, tuple[int, int]] = FORMATION,
) -> dict[str, Any]:
    """GW points of a line-up under FPL's rules: automatic substitutions, the armband passes to the vice when
    the captain did not play (0 minutes), triple captain x3, bench boost counts all 15 (no subs)."""
    if chip == "bboost":
        scorers, subs = list(xi) + list(bench), []
    else:
        scorers, subs = auto_subs(xi, bench, minutes, element_type, limits)
    armband = None
    if int(minutes.get(captain, 0)) > 0:
        armband = captain
    elif int(minutes.get(vice, 0)) > 0:
        armband = vice
    mult = 3 if chip == "3xc" else 2
    base = sum(int(points.get(e, 0)) for e in scorers)
    extra = (mult - 1) * int(points.get(armband, 0)) if armband is not None else 0
    return {"points": base + extra, "armband": armband, "subs": subs, "scorers": scorers}
