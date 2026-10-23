"""Build one shared-data bundle.

Usage: uv run python scripts/build_bundle.py [--out data/bundles] [--with-model] [--previous URL|DIR|none]

--previous (default: the GitHub releases of this repo) is the latest published bundle: its element_history
is reused unless it is older than 6 h, a GW has finished since, or there is none
(bundle.element_history_decision).
--with-model (on in .github/workflows/bundle.yml) fits the xP model (fpl_rocket.model; needs the `model`
dependency group and the previous season in --archive, written by scripts/load_archive.py) and adds
xp.parquet, xp_sims.parquet and xp_fixtures.parquet; a model failure is recorded as model_error and the data
bundle still builds.

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
from fpl_rocket.bundle import (  # noqa: E402
    fetch_previous_release,
    read_bundle,
    read_manifest,
    write_bundle,
)
from fpl_rocket.config import RELEASES_URL  # noqa: E402


def model_fn(archive_path: str):
    """The bundle's model hook: fit on the archive's previous season + the bundle's current season."""

    def run(tables):
        import pandas as pd

        from fpl_rocket.model.pipeline import run_bundle_model

        if not Path(archive_path).exists():
            raise FileNotFoundError(f"{archive_path} missing (scripts/load_archive.py --seasons <previous>)")
        return run_bundle_model(tables, pd.read_parquet(archive_path))

    return run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/bundles")
    ap.add_argument("--cache", default=None, help="optional raw JSON cache directory")
    ap.add_argument("--with-model", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--archive", default="data/archive/all_gws.parquet")
    ap.add_argument(
        "--previous", default=RELEASES_URL, help="releases URL or bundle directory; 'none' = always fetch"
    )
    args = ap.parse_args(argv)
    t0 = time.monotonic()
    previous, note = fetch_previous_release(None if args.previous == "none" else args.previous)
    prev_id = previous.manifest["bundle_id"] if previous else None
    print(f"previous release: {prev_id or '-'} ({note})")
    model = model_fn(args.archive) if args.with_model else None
    path = write_bundle(args.out, FPLClient(args.cache), previous=previous, model=model)
    read_bundle(path)  # verify every checksum and row count
    elapsed = time.monotonic() - t0
    m = read_manifest(path)
    total = sum(f["bytes"] for f in m["files"]) + (Path(path) / "manifest.json").stat().st_size
    print(f"bundle {m['bundle_id']} at {path}: {total / 1024:.0f} KiB, built in {elapsed:.0f} s")
    print(
        f"  next GW{m['next_event']} deadline {m['next_deadline_time']} pre_deadline={m['pre_deadline']};"
        f" element_history {m['element_history_source']} ({m['element_history_reason']});"
        f" summaries missing {len(m['element_summary_missing'])}/{m['element_summary_requested']}"
    )
    if args.with_model:
        if m.get("model_error"):
            print(f"  MODEL FAILED: {m['model_error']}")
        else:
            print(f"  model {m['model_version']} in {m['model_runtime_s']} s: {m['model_timings_s']}")
    for f in m["files"]:
        print(f"  {f['name']:<26} {f['rows']:>8} rows {f['bytes'] / 1024:>8.1f} KiB")
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as fh:
            fh.write(f"path={Path(path).as_posix()}\n")
            fh.write(f"bundle_id={m['bundle_id']}\n")
            fh.write(f"next_event={m['next_event'] or ''}\n")
            fh.write(f"pre_deadline={'true' if m['pre_deadline'] else 'false'}\n")
            fh.write(f"model_version={m.get('model_version') or ''}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
