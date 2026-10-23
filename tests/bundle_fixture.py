"""Build a v2 bundle from the recorded app snapshot (tests/fixtures/app_snapshot), fully offline.

The snapshot holds element summaries only for the recorded owner's initial squad, so the element-summary
fan-out is limited to those ids (the production build covers every player).
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

from fpl_rocket.api import FPLClient
from fpl_rocket.bundle import write_bundle

SNAP = Path(__file__).parent / "fixtures" / "app_snapshot"


def snapshot_summary_ids() -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in SNAP.glob("element-summary_*.json"))


def offline_client(cache: Path = SNAP) -> FPLClient:
    return FPLClient(cache, offline=True, backoff=0)


def build_snapshot_bundle(out_dir: Path, now: datetime | None = None) -> Path:
    stamp = now or datetime.now(UTC)
    return write_bundle(
        out_dir,
        offline_client(),
        now=lambda: stamp,
        element_ids=snapshot_summary_ids(),
        min_interval=0,
    )


def per_id_cache(dest: Path) -> Path:
    """Copy of the snapshot with ONLY per-ID payloads (entries, picks, transfers, leagues): with it, any
    shared data (bootstrap, fixtures, live GWs, element summaries) must come from the bundle."""
    dest.mkdir(parents=True, exist_ok=True)
    for p in SNAP.iterdir():
        if p.name.startswith(("entry_", "leagues-")):
            shutil.copy2(p, dest / p.name)
    return dest
