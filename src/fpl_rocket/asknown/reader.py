"""The single reader of the as-known archive: `load(season, gw)` returns everything a policy may use at the
issue time T = deadline - 2 h of that gameweek, and nothing else.

Every table in data/asknown/ carries the time each row became public (`known_at`, or `snapshot_ts` for the
bootstrap tables); the reader keeps rows with known_at <= T only. Odds rows have a second stamp,
`known_at_post` (results, closing odds, xG: after the match), and those columns are blanked unless it is <= T.

Contract helpers (PLAN_v3 section 1, leakage test):
    AsKnown.violations()  rows whose time stamp is after T (must be empty)
    AsKnown.digest()      a hash of everything returned
    corrupt_after(src, dst, cutoff)  copies an archive and mutates (or drops) every row known after `cutoff`;
                          load() on the copy must return the same digest for any gw with T <= cutoff.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "asknown"
STAMPS = ("known_at", "snapshot_ts", "version_committed_at")
ODDS_POST_PREFIXES = ("c_", "result", "hxg", "axg", "shin_c_")


def _post_cols(cols) -> list[str]:
    return [c for c in cols if c.startswith(ODDS_POST_PREFIXES)]


@dataclass
class AsKnown:
    season: str
    gw: int
    deadline_utc: pd.Timestamp
    issue_time_utc: pd.Timestamp
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)

    # convenient names
    @property
    def players_at_T(self) -> pd.DataFrame:
        return self.frames["players_at_T"]

    @property
    def teams_at_T(self) -> pd.DataFrame:
        return self.frames["teams_at_T"]

    @property
    def events_at_T(self) -> pd.DataFrame:
        return self.frames["events_at_T"]

    @property
    def fixtures_as_known(self) -> pd.DataFrame:
        return self.frames["fixtures_as_known"]

    @property
    def odds(self) -> pd.DataFrame:
        return self.frames["odds"]

    @property
    def minutes_history(self) -> pd.DataFrame:
        return self.frames["minutes_history"]

    @property
    def team_npxg_history(self) -> pd.DataFrame:
        return self.frames["team_npxg_history"]

    @property
    def european_history(self) -> pd.DataFrame:
        return self.frames["european_history"]

    @property
    def crowd_lagged_players(self) -> pd.DataFrame:
        return self.frames["crowd_lagged_players"]

    @property
    def crowd_lagged_tiers(self) -> pd.DataFrame:
        return self.frames["crowd_lagged_tiers"]

    def violations(self) -> dict[str, int]:
        """{frame: rows stamped after T} (time stamps, kickoffs of history rows, post-match odds columns)."""
        t = self.issue_time_utc
        out = {}
        for name, df in self.frames.items():
            bad = pd.Series(False, index=df.index)
            for c in STAMPS:
                if c in df:
                    bad |= pd.to_datetime(df[c], utc=True) > t
            if name in ("minutes_history", "team_npxg_history", "european_history") and "kickoff_utc" in df:
                bad |= pd.to_datetime(df["kickoff_utc"], utc=True) >= t
            if name == "odds" and "known_at_post" in df:
                late = pd.to_datetime(df["known_at_post"], utc=True) > t
                post = _post_cols(df.columns)
                if post:
                    bad |= late & df[post].notna().any(axis=1)
            if bad.any():
                out[name] = int(bad.sum())
        return out

    def digest(self) -> str:
        h = hashlib.sha256(f"{self.season}|{self.gw}|{self.deadline_utc}|{self.issue_time_utc}".encode())
        for name in sorted(self.frames):
            df = self.frames[name]
            h.update(name.encode())
            h.update("|".join(map(str, df.columns)).encode())
            if len(df):
                h.update(pd.util.hash_pandas_object(df.astype(str), index=False).to_numpy().tobytes())
        return h.hexdigest()


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def _known(df: pd.DataFrame | None, t: pd.Timestamp, stamp: str = "known_at") -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    return df[pd.to_datetime(df[stamp], utc=True) <= t].reset_index(drop=True)


def _season_le(root: Path, sub: str, season: str) -> list[Path]:
    folder = root / sub
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.name[:7] <= season)


def load(season: str, gw: int, root: str | Path | None = None) -> AsKnown:
    root = Path(root) if root is not None else DEFAULT_ROOT
    dl = pd.read_parquet(root / "deadlines.parquet")
    row = dl[(dl["season"] == season) & (dl["gw"] == gw)]
    if row.empty:
        raise KeyError(f"no deadline for {season} GW{gw}")
    r = row.iloc[0]
    t = pd.Timestamp(r["issue_time_utc"])
    if r["status"] != "ok":
        raise LookupError(f"{season} GW{gw}: no as-known state ({r['status']})")
    frames: dict[str, pd.DataFrame] = {}

    for kind in ("players", "teams", "events"):
        df = _read(root / "bootstrap" / kind / season / f"gw{gw}.parquet")
        frames[f"{kind}_at_T"] = _known(df, t, "snapshot_ts")

    frames["fixtures_as_known"] = _known(_read(root / "fixtures" / season / f"gw{gw}.parquet"), t)

    # odds: every row whose pre-match odds were collected by T; post-match columns only when known by T
    odds = []
    for p in _season_le(root, "odds", season):
        if p.suffix == ".parquet" and p.stem[:2].isdigit():
            odds.append(_known(pd.read_parquet(p), t))
    o = pd.concat(odds, ignore_index=True) if odds else pd.DataFrame()
    if len(o):
        late = pd.to_datetime(o["known_at_post"], utc=True) > t
        post = _post_cols(o.columns)
        o[post] = o[post].astype(object)
        o.loc[late, post] = None
        o = o.drop(columns=["closing_available_at_T"], errors="ignore")
    frames["odds"] = o

    pm, npx = [], []
    for d in _season_le(root, "minutes", season):
        pm.append(_known(_read(d / "player_matches.parquet"), t))
        npx.append(_known(_read(d / "team_npxg.parquet"), t))
    frames["minutes_history"] = pd.concat(pm, ignore_index=True) if pm else pd.DataFrame()
    frames["team_npxg_history"] = pd.concat(npx, ignore_index=True) if npx else pd.DataFrame()

    eu = _read(root / "european" / "english_club_matches.parquet")
    frames["european_history"] = (
        pd.DataFrame()
        if eu is None
        else eu[pd.to_datetime(eu["known_at"], utc=True) < t].reset_index(drop=True)  # known_at = kickoff
    )

    for kind in ("players", "tiers"):
        df = _read(root / "crowd" / f"lagged_{kind}.parquet")
        if df is not None:
            df = df[(df["season"] == season) & (df["gw"] == gw)]
        frames[f"crowd_lagged_{kind}"] = _known(df, t)

    return AsKnown(season, gw, pd.Timestamp(r["deadline_utc"]), t, frames)


# ===================================================================================== corruption
def corrupt_after(src: str | Path, dst: str | Path, cutoff: pd.Timestamp, mode: str = "mutate") -> int:
    """Copy archive `src` to `dst`, then for every parquet mutate (mode="mutate") or delete (mode="drop")
    every
    row whose time stamp is after `cutoff`, and mutate odds post-match columns known after it. Returns the
    number of rows changed. Time-stamp columns themselves are left as they are."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    cutoff = pd.Timestamp(cutoff)
    changed = 0
    for p in dst.rglob("*.parquet"):
        df = pd.read_parquet(p)
        stamps = [c for c in STAMPS if c in df]
        if p.name == "deadlines.parquet":
            stamps = ["issue_time_utc"]  # a later GW's row (its T is after the cutoff)
        if not stamps:
            continue
        late = pd.Series(False, index=df.index)
        for c in stamps:
            late |= pd.to_datetime(df[c], utc=True) > cutoff
        post_late = pd.to_datetime(df["known_at_post"], utc=True) > cutoff if "known_at_post" in df else None
        if mode == "drop":
            df = df[~late].reset_index(drop=True)
            changed += int(late.sum())
        else:
            keep = set(stamps) | {"known_at_post", "season", "gw", "issue_time_utc", "status"}
            cols = [c for c in df.columns if c not in keep]
            df = _mutate(df, late, cols)
            changed += int(late.sum())
        if post_late is not None and mode != "drop":
            df = _mutate(df, post_late, _post_cols(df.columns))
        elif post_late is not None:
            pl = pd.to_datetime(df["known_at_post"], utc=True) > cutoff
            df = _mutate(df, pl, _post_cols(df.columns))
        df.to_parquet(p, index=False)
    return changed


def _mutate(df: pd.DataFrame, rows: pd.Series, cols: list[str]) -> pd.DataFrame:
    if not rows.any() or not cols:
        return df
    df = df.copy()
    for c in cols:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            df.loc[rows, c] = ~s[rows].fillna(False).astype(bool)
        elif pd.api.types.is_numeric_dtype(s):
            df.loc[rows, c] = s[rows] * -3 + 7
        elif pd.api.types.is_datetime64_any_dtype(s):
            df.loc[rows, c] = s[rows] - pd.Timedelta(days=400)
        else:
            df[c] = df[c].astype("string")
            df.loc[rows, c] = "CORRUPT"
    return df
