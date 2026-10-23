"""Download the community FPL archive and normalise it to data/archive/all_gws.parquet.

Usage: uv run python -I scripts/load_archive.py [--seasons 2022-23 ... 2026-27] [--out DIR] [--offline]

Source: https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/<season>/ - for each
season gws/merged_gw.csv, players_raw.csv, teams.csv, fixtures.csv are saved to <out>/<season>/ (gitignored)
and recorded in <out>/manifest.json (url, bytes, sha256, download time). The files are untrusted data: they
are only ever parsed by pandas.read_csv (fpl_rocket.archive); nothing in them is executed. --offline
re-normalises the files already on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpl_rocket.archive import FILES, SEASONS, load_season_dir, null_columns  # noqa: E402

BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
MAX_BYTES = 64 * 1024 * 1024  # refuse anything implausibly large


def download(season: str, out: Path, session: requests.Session) -> list[dict]:
    d = out / season
    d.mkdir(parents=True, exist_ok=True)
    records = []
    for rel in FILES:
        url = f"{BASE}/{season}/{rel}"
        resp = session.get(url, timeout=(10, 120))
        resp.raise_for_status()
        body = resp.content
        if len(body) > MAX_BYTES:
            raise RuntimeError(f"{url}: {len(body)} bytes exceeds {MAX_BYTES}")
        target = d / Path(rel).name
        target.write_bytes(body)
        records.append(
            {
                "season": season,
                "file": f"{season}/{target.name}",
                "url": url,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        print(f"  {season}/{target.name}: {len(body) / 1024:.0f} KiB")
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="+", default=list(SEASONS))
    ap.add_argument("--out", default="data/archive")
    ap.add_argument("--offline", action="store_true", help="skip downloads; normalise files on disk")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if not args.offline:
        with requests.Session() as session:
            session.headers["User-Agent"] = "fpl-rocket-archive-loader"
            files = [f for f in manifest.get("files", []) if f["season"] not in args.seasons]
            for season in args.seasons:
                files += download(season, out, session)
        manifest = {"source": BASE, "files": files}

    frames = [load_season_dir(s, out / s) for s in args.seasons]
    df = pd.concat(frames, ignore_index=True)
    target = out / "all_gws.parquet"
    df.to_parquet(target, index=False)
    counts = df.groupby("season").agg(rows=("gw", "size"), gws=("gw", "max"), players=("element", "nunique"))
    nulls = null_columns(df)
    manifest.update(
        normalised={
            "file": target.name,
            "rows": len(df),
            "rows_per_season": {s: int(r) for s, r in counts["rows"].items()},
            "null_columns": nulls,
            "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n{target}: {len(df)} rows, {target.stat().st_size / 1024 / 1024:.1f} MiB")
    print(counts.to_string())
    for s, cols in nulls.items():
        print(f"  {s} null columns: {', '.join(cols) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
