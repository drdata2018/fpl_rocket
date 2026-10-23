"""Immutable, versioned data bundles: parquet tables + manifest.json with checksums.

A bundle is written to `<out_dir>/<bundle_id>.partial/` and renamed to `<out_dir>/<bundle_id>/` only after
every file and the manifest are on disk, so a reader never sees a half-written bundle. read_bundle()
re-verifies every checksum and row count and refuses partial or tampered bundles.

Tables:
    players          bootstrap elements (one row per player)
    teams            bootstrap teams
    events           bootstrap events (nested fields JSON-encoded)
    fixtures         all fixtures (stats JSON-encoded)
    gw_live          finished GWs stacked: one row per (gw, element) with every live stat
    gw_live_explain  long per-fixture scoring lines: (gw, element, fixture, identifier, value, points)

TODO(phase A, step 3): element-summary pulls (~700 calls) for per-player history/upcoming fixtures are
deliberately NOT part of this step; add them with rate limiting and their own snapshot time.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .api import FPLClient
from .rules import get_rules

MANIFEST = "manifest.json"
PARTIAL_SUFFIX = ".partial"


class BundleError(RuntimeError):
    """The bundle is missing, partial, or fails validation."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _flatten(records: list[dict[str, Any]]) -> pd.DataFrame:
    """DataFrame with nested values (dict/list) JSON-encoded so parquet types stay simple."""
    rows = [
        {k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict | list) else v for k, v in r.items()}
        for r in records
    ]
    return pd.DataFrame(rows)


def live_tables(gw_live: dict[int, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(gw_live, gw_live_explain) DataFrames from {gw: event/{gw}/live JSON}."""
    stats_rows, explain_rows = [], []
    for gw, live in sorted(gw_live.items()):
        for el in live["elements"]:
            stats_rows.append({"gw": gw, "element": el["id"], **el["stats"]})
            for fx in el.get("explain") or []:
                for s in fx.get("stats", []):
                    explain_rows.append(
                        {
                            "gw": gw,
                            "element": el["id"],
                            "fixture": fx.get("fixture"),
                            "identifier": s["identifier"],
                            "value": s["value"],
                            "points": s["points"],
                        }
                    )
    stats = pd.DataFrame(stats_rows)
    for col in (
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
    ):
        if col in stats:
            stats[col] = pd.to_numeric(stats[col], errors="coerce")
    return stats, pd.DataFrame(explain_rows)


def build_tables(
    client: FPLClient, now: Callable[[], datetime] = _utc_now
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Fetch the shared data and return (tables, source snapshot times)."""
    snaps: dict[str, str] = {}
    bootstrap = client.bootstrap()
    snaps["bootstrap-static"] = now().isoformat(timespec="seconds")
    fixtures = client.fixtures()
    snaps["fixtures"] = now().isoformat(timespec="seconds")
    finished = [e["id"] for e in bootstrap["events"] if e.get("finished")]
    lives = {gw: client.event_live(gw) for gw in finished}
    snaps["event_live"] = now().isoformat(timespec="seconds")
    gw_live, explain = live_tables(lives)
    tables = {
        "players": _flatten(bootstrap["elements"]),
        "teams": _flatten(bootstrap["teams"]),
        "events": _flatten(bootstrap["events"]),
        "fixtures": _flatten(fixtures),
        "gw_live": gw_live,
        "gw_live_explain": explain,
    }
    snaps["finished_gws"] = ",".join(map(str, finished))
    return tables, snaps


def write_tables(
    tables: dict[str, pd.DataFrame],
    out_dir: str | Path,
    snapshots: dict[str, str],
    *,
    bundle_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write tables as one bundle under out_dir; returns the final bundle directory."""
    created = _utc_now()
    bundle_id = bundle_id or created.strftime("%Y%m%dT%H%M%SZ")
    root = Path(out_dir)
    final = root / bundle_id
    if final.exists():
        raise BundleError(f"bundle {final} already exists (bundles are immutable)")
    tmp = root / (bundle_id + PARTIAL_SUFFIX)
    tmp.mkdir(parents=True, exist_ok=False)
    files = []
    for name, df in tables.items():
        path = tmp / f"{name}.parquet"
        df.to_parquet(path, index=False)
        files.append(
            {
                "name": path.name,
                "table": name,
                "rows": len(df),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "bundle_id": bundle_id,
        "created_at": created.isoformat(timespec="seconds"),
        "source_snapshots": snapshots,
        "rules_version": get_rules().RULES_VERSION,
        "model_version": None,
        "files": files,
        **(extra or {}),
    }
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, final)
    return final


def write_bundle(out_dir: str | Path, client: FPLClient | None = None) -> Path:
    """Pull bootstrap, fixtures and every finished GW's live data; write one immutable bundle."""
    tables, snaps = build_tables(client or FPLClient())
    return write_tables(tables, out_dir, snaps)


def read_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    d = Path(bundle_dir)
    if d.name.endswith(PARTIAL_SUFFIX):
        raise BundleError(f"{d} is a partial bundle")
    m = d / MANIFEST
    if not m.exists():
        raise BundleError(f"{d} has no {MANIFEST}: partial or not a bundle")
    return json.loads(m.read_text(encoding="utf-8"))


def read_bundle(bundle_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Validate checksums and row counts, then return {table: DataFrame}. Refuses partial bundles."""
    d = Path(bundle_dir)
    manifest = read_manifest(d)
    out: dict[str, pd.DataFrame] = {}
    for f in manifest["files"]:
        path = d / f["name"]
        if not path.exists():
            raise BundleError(f"{path} listed in manifest but missing")
        if _sha256(path) != f["sha256"]:
            raise BundleError(f"{path} checksum mismatch")
        df = pd.read_parquet(path)
        if len(df) != f["rows"]:
            raise BundleError(f"{path} has {len(df)} rows, manifest says {f['rows']}")
        out[f["table"]] = df
    return out


def latest_bundle(out_dir: str | Path) -> Path | None:
    """Newest complete bundle directory (ignores *.partial), or None."""
    root = Path(out_dir)
    if not root.exists():
        return None
    done = sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and not p.name.endswith(PARTIAL_SUFFIX) and (p / MANIFEST).exists()
    )
    return done[-1] if done else None
