"""Shared plumbing for the as-known archive: polite HTTP with a local cache, read-only git on untrusted
clones, the per-source manifest, season helpers.

Every downloaded file is UNTRUSTED DATA: it is stored under data/raw/<source>/ and parsed with pandas / json /
lzma only; nothing in it is executed. Clones are made with `--filter=blob:none --no-checkout` and only git's
read commands (log, ls-tree, show, cat-file) run against them, with hooks and fsmonitor disabled.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
RETRY_STATUSES = frozenset({403, 429, 500, 502, 503, 504})
MIN_INTERVAL_S = 1.0  # at most one request per second, per process
SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27")
ISSUE_LEAD = pd.Timedelta(hours=2)  # PLAN_v3 section 1: issue time T = deadline - 2 h

GIT = ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "--no-pager"]


def season_start_year(season: str) -> int:
    return int(season[:4])


def season_of(year: int) -> str:
    return f"{year}-{(year + 1) % 100:02d}"


def utcnow() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(UTC))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================================ HTTP
class Fetcher:
    """Polite keyless GET: browser User-Agent, timeouts, >= MIN_INTERVAL_S between requests, retries with
    exponential backoff, redirects followed, and a cache file per URL (a cached file is never refetched unless
    `refresh=True`)."""

    def __init__(self, min_interval: float = MIN_INTERVAL_S, tries: int = 4, backoff: float = 2.0) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": BROWSER_UA, "Accept": "*/*"})
        self.min_interval = min_interval
        self.tries = tries
        self.backoff = backoff
        self._last = 0.0
        self.requests = 0

    def _wait(self) -> None:
        dt = time.monotonic() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.monotonic()

    def get(self, url: str, params: dict[str, Any] | None = None, timeout=(10.0, 60.0)) -> bytes:
        err: Exception | None = None
        for attempt in range(self.tries):
            self._wait()
            self.requests += 1
            try:
                r = self.session.get(url, params=params, timeout=timeout, allow_redirects=True)
                if r.status_code == 200:
                    return r.content
                if r.status_code not in RETRY_STATUSES:
                    raise FetchError(url, r.status_code)
                err = FetchError(url, r.status_code)
            except requests.RequestException as e:  # network error: retry
                err = e
            time.sleep(self.backoff * 2**attempt)
        raise FetchError(url, None, str(err))

    def cached(self, url: str, dest: Path, refresh: bool = False, **kw) -> tuple[bytes, bool]:
        """(content, fetched_now). The content is written to `dest` atomically."""
        dest = Path(dest)
        if dest.exists() and not refresh:
            return dest.read_bytes(), False
        body = self.get(url, **kw)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(body)
        tmp.replace(dest)
        return body, True


class FetchError(RuntimeError):
    def __init__(self, url: str, status: int | None, msg: str = "") -> None:
        self.status = status
        super().__init__(f"GET {url} failed (status={status}) {msg}".strip())


# ============================================================================================= git
def git(repo: Path, *args: str, timeout: int = 600) -> bytes:
    """A read-only git command against an untrusted clone (hooks off). Raises on failure."""
    res = subprocess.run([*GIT, "-C", str(repo), *args], capture_output=True, timeout=timeout, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: {res.stderr.decode(errors='replace').strip()[:500]}")
    return res.stdout


def git_log_paths(repo: Path, ref: str = "HEAD", *pathspec: str) -> pd.DataFrame:
    """Every (commit, commit time, path added or modified) in `ref`'s history, one row per path."""
    out = git(
        repo,
        "log",
        ref,
        "--format=@%H,%cI",
        "--name-only",
        "--diff-filter=AM",
        "--no-renames",
        "--",
        *pathspec,
        timeout=1800,
    ).decode("utf-8", errors="replace")
    rows = []
    sha = at = None
    for line in out.splitlines():
        if line.startswith("@"):
            sha, at = line[1:].split(",", 1)
        elif line.strip() and sha:
            rows.append((sha, at, line.strip()))
    df = pd.DataFrame(rows, columns=["sha", "commit_at", "path"])
    df["commit_at"] = pd.to_datetime(df["commit_at"], utc=True)
    return df


def git_blob(repo: Path, rev_path: str) -> bytes:
    """`git show <sha>:<path>` (the partial clone fetches only that blob)."""
    return git(repo, "show", rev_path)


def git_ls(repo: Path, ref: str = "HEAD", *paths: str) -> pd.DataFrame:
    """(blob, path) of every file under `paths` at `ref`, names only (`-l` would fetch every blob)."""
    out = git(repo, "ls-tree", "-r", "--full-tree", ref, "--", *paths).decode("utf-8", errors="replace")
    rows = []
    for line in out.splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "blob":
            rows.append((parts[2], path))
    return pd.DataFrame(rows, columns=["blob", "path"])


_last_blob_fetch = [0.0]


def cached_blob(
    repo: Path, rev: str, path: str, cache_root: Path, min_interval: float = MIN_INTERVAL_S
) -> bytes:
    """`git show <rev>:<path>` cached at cache_root/<path> (resumable; a cached file is not refetched).
    A fetch of a blob missing from the partial clone is one request to the git host: spaced >= 1 s."""
    dest = Path(cache_root) / path
    if dest.exists():
        return dest.read_bytes()
    dt = time.monotonic() - _last_blob_fetch[0]
    if dt < min_interval:
        time.sleep(min_interval - dt)
    err = None
    for attempt in range(4):
        try:
            body = git_blob(repo, f"{rev}:{path}")
            break
        except (RuntimeError, subprocess.TimeoutExpired) as e:  # transient network failure: back off
            err = e
            time.sleep(2.0 * 2**attempt)
    else:
        raise RuntimeError(f"blob {rev}:{path} not fetched: {err}")
    _last_blob_fetch[0] = time.monotonic()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    return body


# ======================================================================================== manifest
@dataclass
class SourceManifest:
    """What one source contributed: where from, when, content hashes, row counts, coverage gaps."""

    source: str
    urls: list[str] = field(default_factory=list)
    licence: str = ""
    fetched_at: str = ""
    files: dict[str, dict[str, Any]] = field(default_factory=dict)  # raw file -> {sha256, bytes, url}
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)  # output -> {rows, sha256}
    coverage: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add_raw(self, name: str, body: bytes, url: str = "") -> None:
        self.files[name] = {"sha256": sha256_bytes(body), "bytes": len(body), "url": url}

    def add_output(self, path: Path, df: pd.DataFrame, root: Path) -> None:
        self.outputs[str(Path(path).relative_to(root)).replace("\\", "/")] = {
            "rows": len(df),
            "sha256": sha256_file(path),
        }


def write_manifest(root: Path, m: SourceManifest) -> Path:
    m.fetched_at = m.fetched_at or utcnow().isoformat()
    p = Path(root) / "manifest" / f"{m.source}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m.__dict__, indent=1, default=str, sort_keys=False), encoding="utf-8")
    return p


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.part")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def chunked(it: Iterable, n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
