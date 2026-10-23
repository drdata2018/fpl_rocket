"""Build one shared-data bundle. Usage: uv run python scripts/build_bundle.py [--out data/bundles]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fpl_rocket.api import FPLClient
from fpl_rocket.bundle import read_bundle, read_manifest, write_bundle


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/bundles")
    ap.add_argument("--cache", default=None, help="optional raw JSON cache directory")
    args = ap.parse_args(argv)
    path = write_bundle(args.out, FPLClient(args.cache))
    read_bundle(path)  # verify
    m = read_manifest(path)
    total = sum(f["bytes"] for f in m["files"]) + (Path(path) / "manifest.json").stat().st_size
    print(f"bundle {m['bundle_id']} at {path}: {total / 1024:.0f} KiB")
    for f in m["files"]:
        print(f"  {f['name']:<24} {f['rows']:>7} rows {f['bytes'] / 1024:>8.1f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
