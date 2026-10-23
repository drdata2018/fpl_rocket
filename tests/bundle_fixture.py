"""Build a v2 bundle from the recorded app snapshot (tests/fixtures/app_snapshot), fully offline.

The snapshot holds element summaries only for the recorded owner's initial squad, so the element-summary
fan-out is limited to those ids (the production build covers every player).

with_xp=True adds a small deterministic xp.parquet (the model's table layout, next 3 GWs, every player) via
the bundle's model hook: xP from the player's form and minutes, P(play) from status / minutes, low_history
for players without minutes this season, plus a tiny xp_sims table (the model's simulation layout) that the
app must never download. It exists to exercise the app's xP paths offline, not the model.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_rocket.api import FPLClient
from fpl_rocket.bundle import table_records, write_bundle

SNAP = Path(__file__).parent / "fixtures" / "app_snapshot"


def snapshot_summary_ids() -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in SNAP.glob("element-summary_*.json"))


def offline_client(cache: Path = SNAP) -> FPLClient:
    return FPLClient(cache, offline=True, backoff=0)


XP_GWS = 3
XP_MODEL_VERSION = "v0.1-fixture"


def snapshot_xp_model(tables):
    """Bundle model hook: a deterministic stand-in xp table (see the module docstring)."""
    players = table_records(tables["players"])
    events = table_records(tables["events"])
    nxt = next(int(e["id"]) for e in events if e.get("is_next"))
    rows = []
    for p in players:
        minutes = int(p.get("minutes") or 0)
        form = float(p.get("form") or 0)
        chance = p.get("chance_of_playing_next_round")
        p_play = 0.0 if minutes == 0 else min(1.0, minutes / 630) * (1.0 if chance is None else chance / 100)
        for k in range(XP_GWS):
            rows.append(
                {
                    "element": int(p["id"]),
                    "gw": nxt + k,
                    "horizon": 1 + k,
                    "team": str(p["team"]),
                    "element_type": int(p["element_type"]),
                    "n_fixtures": 1,
                    "xp": round(p_play * (1.0 + form) * (1 - 0.02 * k) + (int(p["id"]) % 7) * 0.01, 3),
                    "p_play": round(p_play, 3),
                    "low_history": minutes == 0,
                    "model_version": XP_MODEL_VERSION,
                }
            )
    xp = pd.DataFrame(rows)
    head = xp.head(20)
    sims = pd.DataFrame(
        {
            "element": np.repeat(head["element"].to_numpy(), 4),
            "gw": np.repeat(head["gw"].to_numpy(), 4).astype("int16"),
            "sim": np.tile(np.arange(4, dtype=np.int16), len(head)),
            "points": np.zeros(4 * len(head), dtype=np.int8),
        }
    )
    return {"xp": xp, "xp_sims": sims}, {"model_version": XP_MODEL_VERSION, "model_n_sims": 4}


def build_snapshot_bundle(out_dir: Path, now: datetime | None = None, *, with_xp: bool = False) -> Path:
    stamp = now or datetime.now(UTC)
    return write_bundle(
        out_dir,
        offline_client(),
        now=lambda: stamp,
        element_ids=snapshot_summary_ids(),
        min_interval=0,
        model=snapshot_xp_model if with_xp else None,
    )


def per_id_cache(dest: Path) -> Path:
    """Copy of the snapshot with ONLY per-ID payloads (entries, picks, transfers, leagues): with it, any
    shared data (bootstrap, fixtures, live GWs, element summaries) must come from the bundle."""
    dest.mkdir(parents=True, exist_ok=True)
    for p in SNAP.iterdir():
        if p.name.startswith(("entry_", "leagues-")):
            shutil.copy2(p, dest / p.name)
    return dest
