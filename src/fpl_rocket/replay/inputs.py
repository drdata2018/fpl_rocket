"""Inputs at the issue time T, actual results for scoring, and the leakage-test corruption.

A policy receives ONE object per gameweek, ReplayInputs = asknown.load(season, gw) (players, teams, events,
fixtures as known at T, lagged crowd tables, ...) + `history`: the per-GW points / statistics rows (archive
schema, fpl_rocket.archive) of matches with kickoff + 3 h <= T, of the previous and the current season. The
engine passes nothing else.

The points history is one parquet (data/replay_inputs/points_history.parquet, built by build_points_history)
with a `known_at` stamp = kickoff_time + POINTS_LAG (3 h: the match is over and FPL has published its points;
bonus is usually confirmed about an hour after the last match of the day). A rearranged match of an earlier
gameweek played after T is therefore excluded even though its `gw` label is earlier.

Proxies for seasons without FPL expected-goal / start statistics (2020-21, 2021-22; FPL published them from
2022-23): expected_goals := goals_scored, expected_assists := assists, expected_goal_involvements := their
sum, starts := minutes >= 60 (the FPL long-appearance threshold; a fixed round number, not fitted). Applied
when the history is read, to rows where the statistic is null. Documented in the replay report.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..asknown import reader
from ..asknown.reader import AsKnown

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_DIR = ROOT / "data" / "archive"
INPUTS_DIR = ROOT / "data" / "replay_inputs"
HISTORY_FILE = "points_history.parquet"
POINTS_LAG = pd.Timedelta(hours=3)
NEVER = pd.Timestamp("2100-01-01", tz="UTC")


def previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{(start + 1) % 100:02d}"


# ======================================================================================= points history
def build_points_history(archive_dir: str | Path = ARCHIVE_DIR, out_dir: str | Path = INPUTS_DIR) -> Path:
    """all_gws.parquet (+ live_gws.parquet for (season, gw) the community archive lacks) with known_at."""
    archive_dir, out_dir = Path(archive_dir), Path(out_dir)
    df = pd.read_parquet(archive_dir / "all_gws.parquet")
    live_path = archive_dir / "live_gws.parquet"
    if live_path.exists():
        live = pd.read_parquet(live_path)
        have = set(zip(df["season"], df["gw"].astype(int), strict=True))
        keep = [(s, int(g)) not in have for s, g in zip(live["season"], live["gw"], strict=True)]
        df = pd.concat([df, live[keep]], ignore_index=True)
    ko = pd.to_datetime(df["kickoff_time"], utc=True)
    df["known_at"] = (ko + POINTS_LAG).fillna(NEVER)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / HISTORY_FILE
    df.sort_values(["season", "gw", "element", "fixture"], kind="stable").reset_index(drop=True).to_parquet(
        path, index=False
    )
    return path


def apply_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the statistics FPL did not publish before 2022-23 (module docstring)."""
    df = df.copy()
    for col, src in (("expected_goals", "goals_scored"), ("expected_assists", "assists")):
        miss = df[col].isna()
        if miss.any():
            df[col] = df[col].astype("Float64")
            df.loc[miss, col] = pd.to_numeric(df.loc[miss, src], errors="coerce").astype("Float64")
    miss = df["expected_goal_involvements"].isna()
    if miss.any():
        df["expected_goal_involvements"] = df["expected_goal_involvements"].astype("Float64")
        df.loc[miss, "expected_goal_involvements"] = (
            df.loc[miss, "expected_goals"] + df.loc[miss, "expected_assists"]
        )
    miss = df["starts"].isna()
    if miss.any():
        mins = pd.to_numeric(df.loc[miss, "minutes"], errors="coerce").fillna(0)
        df["starts"] = df["starts"].astype("Int64")
        df.loc[miss, "starts"] = (mins >= 60).astype("int64").to_numpy()
    return df


class HistoryStore:
    """The points history, read once; `known_at(season, t)` = rows of the previous and the current season
    with known_at <= t (proxies applied). `results(season, gw)` = the ACTUAL rows of a GW (scoring only,
    never given to a policy)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.df = pd.read_parquet(self.path)
        self.df["known_at"] = pd.to_datetime(self.df["known_at"], utc=True)

    def known_at(self, season: str, t: pd.Timestamp) -> pd.DataFrame:
        d = self.df
        rows = d[d["season"].isin((season, previous_season(season))) & (d["known_at"] <= pd.Timestamp(t))]
        return apply_proxies(rows.reset_index(drop=True))

    def results(self, season: str, gw: int) -> pd.DataFrame:
        """Per element of GW `gw` (as played): points, minutes, fixtures (double GWs summed)."""
        d = self.df
        g = d[(d["season"] == season) & (d["gw"] == gw)]
        out = g.groupby("element", sort=True).agg(
            points=("total_points", "sum"), minutes=("minutes", "sum"), fixtures=("fixture", "nunique")
        )
        return out.astype("int64")

    def digest(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]


# ======================================================================================= inputs at T
@dataclass
class ReplayInputs:
    season: str
    gw: int
    asknown: AsKnown
    history: pd.DataFrame

    @property
    def issue_time(self) -> pd.Timestamp:
        return self.asknown.issue_time_utc

    @property
    def deadline(self) -> pd.Timestamp:
        return self.asknown.deadline_utc

    def history_digest(self) -> str:
        h = hashlib.sha256()
        if len(self.history):
            cols = [c for c in self.history.columns if c != "known_at"]
            h.update(
                pd.util.hash_pandas_object(self.history[cols].astype(str), index=False).to_numpy().tobytes()
            )
        return h.hexdigest()[:16]

    def manifest(self) -> dict:
        return {
            "asknown_digest": self.asknown.digest()[:16],
            "history_digest": self.history_digest(),
            "history_rows": len(self.history),
            "history_last_known_at": (str(self.history["known_at"].max()) if len(self.history) else None),
            "violations": self.violations(),
        }

    def violations(self) -> dict[str, int]:
        """Rows stamped after T (must be empty): the as-known archive's own check + the history."""
        out = dict(self.asknown.violations())
        if len(self.history):
            late = int((pd.to_datetime(self.history["known_at"], utc=True) > self.issue_time).sum())
            if late:
                out["history"] = late
        return out


def load_inputs(
    season: str, gw: int, store: HistoryStore, asknown_root: str | Path | None = None
) -> ReplayInputs:
    a = reader.load(season, gw, asknown_root)
    return ReplayInputs(season, gw, a, store.known_at(season, a.issue_time_utc))


# ======================================================================================= leakage test
ASKNOWN_SHARED = ("deadlines.parquet", "deadline_history.parquet", "european", "crowd")


def corrupt_copy(
    asknown_root: str | Path,
    history_path: str | Path,
    season: str,
    cutoff: pd.Timestamp,
    work: str | Path,
    mode: str = "mutate",
) -> tuple[Path, Path, int]:
    """Copy the as-known files a `season` replay can read and the points history into `work`, then mutate (or
    drop) every row known after `cutoff` (asknown.reader.corrupt_after). Returns (asknown copy, history
    copy, rows changed)."""
    src, work = Path(asknown_root), Path(work)
    if work.exists():
        shutil.rmtree(work)
    subset = work / "subset"
    subset.mkdir(parents=True)
    for name in ASKNOWN_SHARED:
        p = src / name
        if p.is_dir():
            shutil.copytree(p, subset / name)
        elif p.exists():
            shutil.copy2(p, subset / name)
    for kind in ("players", "teams", "events"):
        p = src / "bootstrap" / kind / season
        if p.exists():
            shutil.copytree(p, subset / "bootstrap" / kind / season)
    if (src / "fixtures" / season).exists():
        shutil.copytree(src / "fixtures" / season, subset / "fixtures" / season)
    for sub in ("odds", "minutes"):
        for p in reader._season_le(src, sub, season):
            target = subset / sub / p.name
            if p.is_dir():
                shutil.copytree(p, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
    hist_dir = subset / "_history"
    hist_dir.mkdir()
    shutil.copy2(history_path, hist_dir / HISTORY_FILE)
    out = work / "corrupt"
    changed = reader.corrupt_after(subset, out, cutoff, mode=mode)
    shutil.rmtree(subset)
    return out, out / "_history" / HISTORY_FILE, changed
