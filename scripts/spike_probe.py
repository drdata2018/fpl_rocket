"""Day-1 hosting spike: call four public FPL endpoints and append one JSON line per call.

Usage: python scripts/spike_probe.py --out spike/actions_log.jsonl
Each line: {"ts", "host_hint", "endpoint", "status", "ms", "bytes"[, "error"]}. Always exits 0 (failures are
data). Single attempt per call, no retries: the spike measures raw success rate, latency and 403/429.
streamlit_app.py imports probe_all from here so both hosts measure exactly the same calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # runnable without installing

from fpl_rocket.api import BASE_URL, BROWSER_UA  # noqa: E402

# The probe entry is the owner's public team id (plan section 0: allowed as the probe/placeholder constant).
PROBE_ENTRY = 7630179
PROBE_PATHS = (
    "bootstrap-static/",
    "fixtures/?event=8",
    f"entry/{PROBE_ENTRY}/",
    "event/7/live/",
)


def host_hint() -> str:
    """Coarse description of where the probe runs (no secrets; the hostname is hashed)."""
    if os.environ.get("GITHUB_ACTIONS"):
        return f"github-actions:{os.environ.get('RUNNER_OS', '?')}:{os.environ.get('RUNNER_NAME', '?')}"
    host = hashlib.sha256(socket.gethostname().encode()).hexdigest()[:8]
    return f"{platform.system().lower()}:host-{host}"


def probe_once(path: str, session: requests.Session, timeout: float = 20.0) -> dict[str, Any]:
    """One timed GET; returns {ts, host_hint, endpoint, status, ms, bytes[, error]}."""
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    rec: dict[str, Any] = {"ts": ts, "host_hint": host_hint(), "endpoint": path}
    try:
        resp = session.get(BASE_URL + path, timeout=timeout)
        rec.update(status=resp.status_code, bytes=len(resp.content))
    except requests.RequestException as exc:
        rec.update(status=None, bytes=0, error=type(exc).__name__)
    rec["ms"] = round((time.perf_counter() - t0) * 1000)
    return rec


def probe_all(paths: tuple[str, ...] = PROBE_PATHS) -> list[dict[str, Any]]:
    """Probe every path once with a fresh browser-UA session."""
    with requests.Session() as s:
        s.headers.update({"User-Agent": BROWSER_UA, "Accept": "application/json"})
        return [probe_once(p, s) for p in paths]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="JSONL file to append to")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = probe_all()
    with out.open("a", encoding="utf-8") as f:
        for rec in records:
            line = json.dumps(rec, separators=(",", ":"))
            print(line)
            f.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
