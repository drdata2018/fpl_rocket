"""Frozen reference replays ("baselines"): the record of a replay run and its byte-level verification.

A baseline file (replay/baselines/<name>.json) records: the producing commit (and its git tag), every policy
setting (planner horizon / decay / rule, chip thresholds, pools, seeds), the model and forecast versions, the
season rules versions, sha256 of the as-known archive manifests and of the points history, the forecast-cache
digest, and per season / policy: sha256 of the decision log file, of rows.csv, and the CANONICAL digests.

Canonical form (what must be byte-identical when a season is replayed from the caches): per GW the decision
record with its two wall-clock fields removed (`decide_seconds`, `package.solver.seconds`), serialised as
json.dumps(sort_keys=True); then
    packages   sha256 over the canonical PACKAGES (transfers, XI, bench, captain, vice, chip, hits, scores,
               solver status / reason, notes), one line per GW
    decisions  sha256 over the full canonical decision records (inputs manifest, policy settings, model
               seed, state entering the GW, package)
    scores     sha256 over the score rows (json.dumps(sort_keys=True))
The raw log sha256 is an integrity record of the archived file only (its timings differ on every run).
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path
from typing import Any

BASELINES_DIR = Path(__file__).resolve().parent / "baselines"
TIMING_FIELDS = ("decide_seconds",)
SOLVER_TIMING_FIELDS = ("seconds",)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump(x: Any) -> str:
    return json.dumps(x, sort_keys=True)


def canonical_decision(rec: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in rec.items() if k not in TIMING_FIELDS}
    pkg = dict(out.get("package") or {})
    pkg["solver"] = {k: v for k, v in (pkg.get("solver") or {}).items() if k not in SOLVER_TIMING_FIELDS}
    out["package"] = pkg
    return out


def read_log(path: str | Path) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """({gw: decision record}, {gw: score row}) of one season / policy decision log."""
    decisions, scores = {}, {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["event"] == "decision":
                decisions[int(rec["gw"])] = rec
            elif rec["event"] == "score":
                scores[int(rec["gw"])] = rec["row"]
    return decisions, scores


def canonical_lines(path: str | Path) -> dict[str, dict[int, str]]:
    decisions, scores = read_log(path)
    out: dict[str, dict[int, str]] = {"packages": {}, "decisions": {}, "scores": {}}
    for gw in sorted(decisions):
        c = canonical_decision(decisions[gw])
        out["packages"][gw] = _dump(c["package"])
        out["decisions"][gw] = _dump(c)
    for gw in sorted(scores):
        out["scores"][gw] = _dump(scores[gw])
    return out


def _digest(lines: dict[int, str]) -> str:
    h = hashlib.sha256()
    for gw in sorted(lines):
        h.update(lines[gw].encode("utf-8") + b"\n")
    return h.hexdigest()


def log_record(path: str | Path) -> dict[str, Any]:
    c = canonical_lines(path)
    _, scores = read_log(path)
    return {
        "sha256": sha256_file(path),
        "packages": _digest(c["packages"]),
        "decisions": _digest(c["decisions"]),
        "scores": _digest(c["scores"]),
        "gws": len(c["packages"]),
        "total_net": int(sum(int(r["net"]) for r in scores.values())),
    }


def compare_logs(reference: str | Path, candidate: str | Path) -> dict[str, Any]:
    """Per-GW comparison of two decision logs in canonical form."""
    a, b = canonical_lines(reference), canonical_lines(candidate)
    out: dict[str, Any] = {}
    for kind in ("packages", "decisions", "scores"):
        gws = sorted(set(a[kind]) | set(b[kind]))
        bad = [g for g in gws if a[kind].get(g) != b[kind].get(g)]
        out[kind] = {"gws": len(gws), "different": bad}
    out["identical"] = not any(out[k]["different"] for k in ("packages", "decisions", "scores"))
    return out


def dir_digest(paths: Iterable[Path], root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(f"{p.relative_to(root).as_posix()}:{sha256_file(p)}\n".encode())
    return h.hexdigest()


def forecast_cache_record(cache_dir: str | Path, seasons: Iterable[str]) -> dict[str, Any]:
    root = Path(cache_dir)
    per = {}
    every: list[Path] = []
    for s in seasons:
        files = sorted((root / s).glob("gw*.pkl"))
        every += files
        per[s] = {"files": len(files), "digest": dir_digest(files, root)}
    return {"digest": dir_digest(every, root), "per_season": per}


def environment() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("highspy", "scikit-learn", "numpy", "pandas", "scipy", "pyarrow"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "absent"
    return out


def load(name: str) -> dict[str, Any]:
    return json.loads((BASELINES_DIR / f"{name}.json").read_text(encoding="utf-8"))
