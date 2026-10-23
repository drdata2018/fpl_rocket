"""Build one shared-data bundle. Usage: uv run python scripts/build_bundle.py [--out data/bundles]

Prints the bundle id, size per file and build time. On GitHub Actions ($GITHUB_OUTPUT set) it also writes
the step outputs used by .github/workflows/bundle.yml: path, bundle_id, next_event, pre_deadline.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpl_rocket.api import FPLClient  # noqa: E402
from fpl_rocket.bundle import read_bundle, read_manifest, write_bundle  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/bundles")
    ap.add_argument("--cache", default=None, help="optional raw JSON cache directory")
    args = ap.parse_args(argv)
    t0 = time.monotonic()
    path = write_bundle(args.out, FPLClient(args.cache))
    read_bundle(path)  # verify every checksum and row count
    elapsed = time.monotonic() - t0
    m = read_manifest(path)
    total = sum(f["bytes"] for f in m["files"]) + (Path(path) / "manifest.json").stat().st_size
    print(f"bundle {m['bundle_id']} at {path}: {total / 1024:.0f} KiB, built in {elapsed:.0f} s")
    print(
        f"  next GW{m['next_event']} deadline {m['next_deadline_time']} pre_deadline={m['pre_deadline']};"
        f" element summaries missing {len(m['element_summary_missing'])}/{m['element_summary_requested']}"
    )
    for f in m["files"]:
        print(f"  {f['name']:<26} {f['rows']:>7} rows {f['bytes'] / 1024:>8.1f} KiB")
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as fh:
            fh.write(f"path={Path(path).as_posix()}\n")
            fh.write(f"bundle_id={m['bundle_id']}\n")
            fh.write(f"next_event={m['next_event'] or ''}\n")
            fh.write(f"pre_deadline={'true' if m['pre_deadline'] else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
