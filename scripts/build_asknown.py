"""Build the AS-KNOWN archive (fpl_rocket.asknown) under data/asknown/ from keyless public sources.

Usage:
    uv run python -I scripts/build_asknown.py <subcommand> [--data DATA] [--seasons ...] [--update]

Subcommands (each resumable and idempotent: downloads are cached under data/raw/<source>/ and reused):
    bootstrap  deadlines + players/teams/events/rules as known at T (github Randdalf/fplcache)
    fixtures   fixture schedule as known at T (github vaastav/Fantasy-Premier-League history)
    odds       football-data.co.uk E0.csv per season -> normalised odds, implied goal rates, available_at_T
    minutes    all-competition player-match minutes/penalties (github olbauday/FPL-Core-Insights)
    crowd      sampled real managers by tier (github sertalpbilal/fpl_optimized, branch webpage)
    uefa       European fixtures of English clubs (match.uefa.com)
    report     print / write the coverage report from what is built
    all        everything above in order

Clones are partial (`--filter=blob:none --no-checkout`) and only git's read commands run against them; every
downloaded file is untrusted data parsed with pandas / json / lzma. `--update` refreshes the clones and the
current-season files; without it nothing already cached is downloaded again.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpl_rocket.asknown import bootstrap as bs  # noqa: E402
from fpl_rocket.asknown import common as cm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CURRENT = cm.SEASONS[-1]


class Ctx:
    def __init__(self, data: Path, seasons: list[str], update: bool) -> None:
        self.data = Path(data)
        self.raw = self.data / "raw"
        self.out = self.data / "asknown"
        self.seasons = seasons
        self.update = update
        self.fetcher = cm.Fetcher()


# ============================================================================================ git
def ensure_clone(ctx: Ctx, url: str, name: str, branch: str | None = None) -> Path:
    dest = ctx.raw / name
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        cmd = [*cm.GIT, "clone", "--filter=blob:none", "--no-checkout"]
        if branch:
            cmd += ["--single-branch", "--branch", branch]
        subprocess.run([*cmd, url, str(dest)], check=True, timeout=3600)
        for k, v in (("gc.auto", "0"), ("maintenance.auto", "false")):
            subprocess.run([*cm.GIT, "-C", str(dest), "config", k, v], check=True)
    elif ctx.update:
        subprocess.run(
            [*cm.GIT, "-C", str(dest), "fetch", "--filter=blob:none", "origin"], check=True, timeout=3600
        )
        ref = branch or cm.git(dest, "symbolic-ref", "--short", "HEAD").decode().strip()
        cm.git(dest, "update-ref", f"refs/heads/{ref}", f"origin/{ref}")
    return dest


# ====================================================================================== bootstrap
class Snapshots:
    """fplcache snapshots by path, parsed on demand from the blob cache (data/raw/fplcache/)."""

    def __init__(self, ctx: Ctx, repo: Path) -> None:
        self.repo = repo
        self.cache_root = ctx.raw / "fplcache"
        self.index = bs.snapshot_index(cm.git_ls(repo, "HEAD", "cache"))
        self.parsed: dict[str, dict] = {}
        self.raw_sha: dict[str, str] = {}

    def load(self, path: str) -> dict:
        if path not in self.parsed:
            raw = cm.cached_blob(self.repo, "HEAD", path, self.cache_root)
            self.raw_sha[path] = cm.sha256_bytes(raw)
            self.parsed[path] = bs.parse_snapshot(raw)
        return self.parsed[path]

    def season_of(self, row) -> str | None:
        return bs.snapshot_season(self.load(row["path"]))

    def first_of_season(self, season: str) -> pd.Series | None:
        """Binary search for the first snapshot whose bootstrap is the new season (game launch)."""
        y = cm.season_start_year(season)
        idx = self.index[
            (self.index["snapshot_ts"] >= pd.Timestamp(f"{y}-05-01", tz="UTC"))
            & (self.index["snapshot_ts"] <= pd.Timestamp(f"{y}-08-31", tz="UTC"))
        ].reset_index(drop=True)
        if idx.empty:
            return None
        lo, hi = 0, len(idx) - 1
        if self.season_of(idx.iloc[hi]) != season:
            return None
        if self.season_of(idx.iloc[lo]) == season:
            return idx.iloc[lo]
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.season_of(idx.iloc[mid]) == season:
                hi = mid
            else:
                lo = mid
        return idx.iloc[hi]

    def end_of_season(self, season: str) -> pd.Series | None:
        """The last snapshot of the season (before the next season's launch); the latest for the current
        one."""
        y = cm.season_start_year(season)
        nxt = self.first_of_season(cm.season_of(y + 1))
        cands = self.index if nxt is None else self.index[self.index["snapshot_ts"] < nxt["snapshot_ts"]]
        for i in range(len(cands) - 1, max(len(cands) - 40, -1), -1):
            row = cands.iloc[i]
            if self.season_of(row) == season:
                return row
        return None


def cmd_bootstrap(ctx: Ctx) -> None:
    repo = ensure_clone(ctx, bs.REPO_URL, "fplcache_meta")
    snaps = Snapshots(ctx, repo)
    now = cm.utcnow()
    man = cm.SourceManifest(source="fplcache", urls=[bs.REPO_URL], licence="Unlicense")
    man.notes.append(
        f"{len(snaps.index)} snapshots in HEAD, {snaps.index['snapshot_ts'].min()} .. "
        f"{snaps.index['snapshot_ts'].max()}; snapshot_ts = path time (UTC, taken after the response)"
    )
    kickoffs = first_kickoffs(ctx)
    dl_rows, hist_frames, snap_rows, rules_rows = [], [], [], []
    rules_by_season: dict[str, dict] = {}
    for season in ctx.seasons:
        end = snaps.end_of_season(season)
        launch = snaps.first_of_season(season)
        if end is None:
            man.gaps.append(f"{season}: no snapshot of the season")
            continue
        final = bs.event_deadlines(snaps.load(end["path"]))

        def deadline_of(row, gw, season=season):
            d = snaps.load(row["path"])
            if bs.snapshot_season(d) != season:
                return None
            return bs.event_deadlines(d).get(gw)

        used = set()
        for gw, d_final in sorted(final.items()):
            future = d_final - cm.ISSUE_LEAD > now
            snap, deadline, iters = (
                (None, d_final, 0)
                if future
                else bs.resolve_issue_snapshot(snaps.index, gw, d_final, deadline_of)
            )
            if snap is not None and snaps.season_of(snap) != season:
                snap = None  # the last snapshot before T is still last season's game: no as-known state
            issue = deadline - cm.ISSUE_LEAD
            row = {
                "season": season,
                "gw": gw,
                "deadline_utc": deadline,
                "issue_time_utc": issue,
                "first_kickoff_utc": kickoffs.get((season, gw)),
                "deadline_final_utc": d_final,
                "deadline_moved_after_T": deadline != d_final,
                "status": "future" if future else ("ok" if snap is not None else "no_snapshot_before_T"),
                "snapshot_path": None if snap is None else snap["path"],
                "snapshot_ts": None if snap is None else snap["snapshot_ts"],
                "lag_minutes": None if snap is None else (issue - snap["snapshot_ts"]).total_seconds() / 60,
                "fixed_point_iterations": iters,
            }
            dl_rows.append(row)
            if snap is None:
                if not future:
                    man.gaps.append(f"{season} GW{gw}: no snapshot at or before T={issue}")
                continue
            used.add(snap["path"])
            d = snaps.load(snap["path"])
            for kind, fn in (
                ("players", bs.players_frame),
                ("teams", bs.teams_frame),
                ("events", bs.events_frame),
            ):
                p = ctx.out / "bootstrap" / kind / season / f"gw{gw}.parquet"
                df = fn(d, season, gw, snap["snapshot_ts"], issue)
                cm.write_parquet(df, p)
                if kind == "players":
                    row["players"] = len(df)
            rec = bs.rules_record(d)
            rules_rows.append({"season": season, "gw": gw, "rules_hash": bs.rules_hash(rec)})
            rules_by_season.setdefault(season, {})[gw] = rec
            snap_rows.append(
                {
                    "season": season,
                    "gw": gw,
                    "path": snap["path"],
                    "blob": snap["blob"],
                    "sha256": snaps.raw_sha[snap["path"]],
                }
            )
        extra = {p for p in (end["path"], None if launch is None else launch["path"]) if p}
        hist = bs.deadline_history({p: snaps.parsed[p] for p in used | extra}, snaps.index, season)
        hist_frames.append(hist)
        print(
            f"{season}: {len(final)} GWs, {len(used)} snapshots used, "
            f"launch {None if launch is None else launch['snapshot_ts']}",
            flush=True,
        )

    dl = pd.DataFrame(dl_rows)
    hist = pd.concat(hist_frames, ignore_index=True)
    summ = bs.summarise_history(hist)
    dl = dl.merge(summ, on=["season", "gw"], how="left")
    for c in (
        "deadline_utc",
        "issue_time_utc",
        "first_kickoff_utc",
        "deadline_final_utc",
        "snapshot_ts",
        "deadline_first_seen_utc",
        "first_seen_snapshot_ts",
    ):
        dl[c] = pd.to_datetime(dl[c], utc=True)
    cm.write_parquet(dl, ctx.out / "deadlines.parquet")
    cm.write_parquet(hist, ctx.out / "deadline_history.parquet")
    man.add_output(ctx.out / "deadlines.parquet", dl, ctx.out)
    man.add_output(ctx.out / "deadline_history.parquet", hist, ctx.out)
    sn = pd.DataFrame(snap_rows)
    cm.write_parquet(sn, ctx.out / "bootstrap" / "snapshots_used.parquet")
    for r in snap_rows:
        man.files[r["path"]] = {
            "sha256": r["sha256"],
            "blob": r["blob"],
            "url": f"{bs.REPO_URL}/blob/main/{r['path']}",
        }
    write_rules(ctx, rules_by_season, pd.DataFrame(rules_rows), man)
    man.coverage = bootstrap_coverage(dl)
    cm.write_manifest(ctx.out, man)
    print(pd.DataFrame(man.coverage).to_string(index=False))


def bootstrap_coverage(dl: pd.DataFrame) -> list[dict]:
    out = []
    for season, g in dl.groupby("season", sort=True):
        ok = g[g["status"] == "ok"]
        out.append(
            {
                "season": season,
                "gameweeks": len(g),
                "covered": len(ok),
                "future": int((g["status"] == "future").sum()),
                "no_snapshot_before_T": ",".join(
                    str(x) for x in g.loc[g["status"] == "no_snapshot_before_T", "gw"]
                ),
                "median_lag_min": round(float(ok["lag_minutes"].median()), 1) if len(ok) else None,
                "max_lag_min": round(float(ok["lag_minutes"].max()), 1) if len(ok) else None,
                "max_lag_gw": int(ok.loc[ok["lag_minutes"].idxmax(), "gw"]) if len(ok) else None,
                "deadline_moved_after_T": int(g["deadline_moved_after_T"].sum()),
                "gws_deadline_changed_in_season": int((g["deadline_changes"].fillna(0) > 0).sum()),
            }
        )
    return out


def write_rules(ctx: Ctx, rules_by_season: dict, rules_rows: pd.DataFrame, man: cm.SourceManifest) -> None:
    folder = ctx.out / "rules"
    folder.mkdir(parents=True, exist_ok=True)
    cm.write_parquet(rules_rows, folder / "rules_hash_by_gw.parquet")
    evo, prev, prev_season = [], None, None
    for season in sorted(rules_by_season):
        by_gw = rules_by_season[season]
        gws = sorted(by_gw)
        first = by_gw[gws[0]]
        within = []
        for a, b in zip(gws, gws[1:], strict=False):
            for k, before, after in bs.diff_rules(by_gw[a], by_gw[b]):
                within.append(
                    {"season": season, "from_gw": a, "to_gw": b, "key": k, "before": before, "after": after}
                )
        (folder / f"{season}.json").write_text(
            json.dumps(
                {"season": season, "as_of_gw": gws[0], "rules": first, "changes_within_season": within},
                indent=1,
                default=str,
            ),
            encoding="utf-8",
        )
        for k, before, after in bs.diff_rules(prev, first) if prev is not None else []:
            evo.append(
                {"from_season": prev_season, "to_season": season, "key": k, "before": before, "after": after}
            )
        evo.extend(
            {
                "from_season": season,
                "to_season": season,
                **{k: w[k] for k in ("key", "before", "after")},
                "within_gw": f"{w['from_gw']}->{w['to_gw']}",
            }
            for w in within
        )
        prev, prev_season = by_gw[gws[-1]], season  # the next season is compared with this season's end state
    ev = pd.DataFrame(evo)
    for c in ("before", "after"):
        if c in ev:
            ev[c] = ev[c].map(lambda v: json.dumps(v, default=str) if not isinstance(v, str) else v)
    ev.to_csv(folder / "evolution.csv", index=False)
    man.notes.append(f"rules evolution rows: {len(ev)} (rules/evolution.csv)")


# ======================================================================================= kickoffs
def first_kickoffs(ctx: Ctx) -> dict[tuple[str, int], pd.Timestamp]:
    """{(season, gw): first kickoff as played} from the final fixture lists (fixtures module)."""
    from fpl_rocket.asknown import fixtures as fx

    out = {}
    for season in ctx.seasons:
        final = final_fixtures(ctx, season)
        if final is None:
            continue
        for gw, k in fx.first_kickoffs(final).items():
            out[(season, gw)] = k
    return out


def final_fixtures(ctx: Ctx, season: str) -> pd.DataFrame | None:
    """The as-played schedule: the community archive's latest fixtures.csv for finished seasons; the live
    FPL fixtures endpoint (keyless) for the current season (the archive is stale for it)."""
    from fpl_rocket.asknown import fixtures as fx

    if season == CURRENT:
        dest = ctx.raw / "fpl_api" / f"fixtures_{season}.json"
        body, _ = ctx.fetcher.cached(
            "https://fantasy.premierleague.com/api/fixtures/", dest, refresh=ctx.update
        )
        return fx.parse_api_fixtures(body)
    repo = ensure_clone(ctx, fx.VAASTAV_URL, "vaastav_meta")
    try:
        raw = cm.cached_blob(repo, "HEAD", f"data/{season}/fixtures.csv", ctx.raw / "vaastav")
    except RuntimeError:
        return None
    return fx.parse_csv_fixtures(raw)


# ===================================================================================== shared reads
def deadlines_table(ctx: Ctx) -> pd.DataFrame:
    p = ctx.out / "deadlines.parquet"
    if not p.exists():
        raise SystemExit("deadlines.parquet missing: run the `bootstrap` subcommand first")
    return pd.read_parquet(p)


def season_teams(ctx: Ctx, season: str) -> pd.DataFrame | None:
    """FPL teams (team, name, short_name) of a season: the as-known bootstrap tables, else the community
    archive's teams.csv (seasons before the snapshot archive)."""
    folder = ctx.out / "bootstrap" / "teams" / season
    files = sorted(folder.glob("gw*.parquet"), key=lambda p: int(p.stem[2:])) if folder.exists() else []
    if files:
        return pd.read_parquet(files[-1])[["team", "name", "short_name"]]
    from fpl_rocket.asknown import fixtures as fx

    repo = ensure_clone(ctx, fx.VAASTAV_URL, "vaastav_meta")
    try:
        raw = cm.cached_blob(repo, "HEAD", f"data/{season}/teams.csv", ctx.raw / "vaastav")
    except RuntimeError:
        return None
    t = pd.read_csv(io.BytesIO(raw), usecols=["id", "name", "short_name"])
    return t.rename(columns={"id": "team"})


# ======================================================================================= fixtures
def cmd_fixtures(ctx: Ctx) -> None:
    from fpl_rocket import fixtures_known as fk
    from fpl_rocket.asknown import fixtures as fx

    repo = ensure_clone(ctx, fx.VAASTAV_URL, "vaastav_meta")
    dl = deadlines_table(ctx)
    man = cm.SourceManifest(source="fixtures_asknown", urls=[fx.VAASTAV_URL], licence="MIT (repository)")
    rows = []
    for season in ctx.seasons:
        path = f"data/{season}/fixtures.csv"
        commits = fk.parse_log(cm.git(repo, "log", "--format=%H,%cI", "HEAD", "--", path).decode())
        final = final_fixtures(ctx, season)
        cache: dict[str, pd.DataFrame] = {}
        for _, d in dl[(dl["season"] == season) & (dl["status"] != "future")].iterrows():
            gw, t = int(d["gw"]), d["issue_time_utc"]
            commit = fx.commit_at_or_before(commits, t)
            snap = None
            if commit is not None:
                if commit.sha not in cache:
                    raw = cm.cached_blob(
                        repo, commit.sha, path, ctx.raw / "vaastav" / "history" / commit.sha[:12]
                    )
                    cache[commit.sha] = fx.parse_csv_fixtures(raw)
                    man.add_raw(
                        f"{commit.sha[:12]}:{path}", raw, f"{fx.VAASTAV_URL}/blob/{commit.sha}/{path}"
                    )
                snap = cache[commit.sha]
                out = snap.assign(
                    season=season,
                    gw=gw,
                    snapshot_sha=commit.sha,
                    snapshot_at=commit.at,
                    issue_time_utc=t,
                    known_at=commit.at,
                )
                cm.write_parquet(out, ctx.out / "fixtures" / season / f"gw{gw}.parquet")
            else:
                man.gaps.append(f"{season} GW{gw}: no fixtures commit at or before T")
            rows.append(fx.as_known_rows(season, gw, t, commit, snap, final))
        print(f"{season}: {len(commits)} commits, {len(cache)} versions used", flush=True)
    cov = pd.DataFrame(rows)
    cov.to_csv(ctx.out / "fixtures" / "coverage.csv", index=False)
    summ = []
    for season, g in cov.groupby("season", sort=True):
        has = g["snapshot_sha"].notna()
        summ.append(
            {
                "season": season,
                "gws": len(g),
                "with_snapshot": int(has.sum()),
                "median_age_days": round(float(g.loc[has, "age_days"].median()), 2) if has.any() else None,
                "max_age_days": round(float(g.loc[has, "age_days"].max()), 2) if has.any() else None,
                "gws_older_than_7d": int((g["age_days"] > 7).sum()),
                "gws_with_gw_t_differences": int((g.get("differ_gw_t", pd.Series(dtype=float)) > 0).sum()),
            }
        )
    man.coverage = summ
    cm.write_manifest(ctx.out, man)
    print(pd.DataFrame(summ).to_string(index=False))


# =========================================================================================== odds
ODDS_SEASONS = ("2019-20", "2020-21", *cm.SEASONS)


def cmd_odds(ctx: Ctx) -> None:
    from fpl_rocket.asknown import fixtures as fx
    from fpl_rocket.asknown import odds as od

    dl = deadlines_table(ctx)
    man = cm.SourceManifest(source="footballdata", urls=[od.BASE], licence="football-data.co.uk (free use)")
    notes = ctx.raw / "footballdata" / "notes.txt"
    body, _ = ctx.fetcher.cached(f"{od.BASE}/notes.txt", notes)
    text = body.decode("utf-8", errors="replace")
    if od.COLLECTION_SENTENCE not in text:
        man.gaps.append("notes.txt no longer contains the collection-time sentence: re-check before use")
    man.notes.append(f"collection time (notes.txt, verbatim): {od.COLLECTION_SENTENCE}")
    man.add_raw("notes.txt", body, f"{od.BASE}/notes.txt")
    cov = []
    for season in [s for s in ODDS_SEASONS if s in ctx.seasons or s in ("2019-20", "2020-21")]:
        url = od.season_url(season)
        body, _ = ctx.fetcher.cached(
            url,
            ctx.raw / "footballdata" / od.SEASON_CODES[season] / "E0.csv",
            refresh=ctx.update and season == CURRENT,
        )
        man.add_raw(f"{od.SEASON_CODES[season]}/E0.csv", body, url)
        df = od.parse_csv(body, season)
        teams = season_teams(ctx, season)
        final = final_fixtures(ctx, season)
        df = od.map_teams(df, teams) if teams is not None else df.assign(team_h=pd.NA, team_a=pd.NA)
        df = od.map_fixtures(df, final) if final is not None else df.assign(fixture=pd.NA, gw=pd.NA)
        # issue time of the GW the match is played in
        d = dl[dl["season"] == season].set_index("gw")
        if len(d):
            t = df["gw"].map(d["issue_time_utc"])
            df["issue_time_source"] = "deadlines"
        else:  # before the snapshot archive: first as-played kickoff - 90 min - 2 h (approximation, flagged)
            fk_ = fx.first_kickoffs(final) if final is not None else {}
            t = df["gw"].map(lambda g, k=fk_: k.get(int(g)) if pd.notna(g) else None)
            t = pd.to_datetime(t, utc=True) - pd.Timedelta(minutes=90) - cm.ISSUE_LEAD
            df["issue_time_source"] = "approx_first_kickoff_minus_90min"
        df = od.label_availability(df, t)
        df["known_at"] = df["collected_at_assumed"]
        df["known_at_post"] = df["kickoff_utc"] + pd.Timedelta(hours=3)  # results / closing / xG
        # the GW this fixture had in the schedule as known at T (when the as-known fixtures are built)
        df["gw_known_at_T"] = pd.array([pd.NA] * len(df), dtype="Int64")
        for g in df["gw"].dropna().unique():
            p = ctx.out / "fixtures" / season / f"gw{int(g)}.parquet"
            if p.exists():
                k = pd.read_parquet(p, columns=["fixture", "event"]).set_index("fixture")["event"]
                sel = df["gw"] == g
                df.loc[sel, "gw_known_at_T"] = df.loc[sel, "fixture"].map(k).astype("Int64")
        inv = [
            od.invert_frame(df, "avg"),
            od.invert_frame(df, "avg", method="shin").add_prefix("shin_"),
            od.invert_frame(df, "avg", closing=True),
        ]
        if df["bfe_h"].notna().any():
            inv.append(od.invert_frame(df, "bfe"))
        df = pd.concat([df, *inv], axis=1)
        out = ctx.out / "odds" / f"{season}.parquet"
        cm.write_parquet(df, out)
        man.add_output(out, df, ctx.out)
        cov.append(
            {
                "season": season,
                "matches": len(df),
                "teams_mapped": round(float((df["team_h"].notna() & df["team_a"].notna()).mean()), 4),
                "fixture_mapped": round(float(df["fixture"].notna().mean()), 4),
                "with_avg_1x2": int(df["avg_h"].notna().sum()),
                "with_bfe_1x2": int(df["bfe_h"].notna().sum()),
                "with_xg": int(df["hxg"].notna().sum()),
                "available_at_T_share": round(float(df["available_at_T"].mean()), 4),
                "available_at_T_noon_share": round(float(df["available_at_T_noon"].mean()), 4),
                "median_fit_rmse": round(float(df["avg_fit_rmse"].median()), 5)
                if "avg_fit_rmse" in df
                else None,
                "max_fit_rmse": round(float(df["avg_fit_rmse"].max()), 5) if "avg_fit_rmse" in df else None,
                "issue_time_source": df["issue_time_source"].iloc[0] if len(df) else None,
            }
        )
        unmapped = sorted(set(df.loc[df["team_h"].isna(), "home"]) | set(df.loc[df["team_a"].isna(), "away"]))
        if unmapped:
            man.gaps.append(f"{season}: unmapped team names {unmapped}")
        print(f"{season}: {len(df)} matches", flush=True)
    # the live next-round file (not wired into the deadline pipeline)
    nxt = od.fetch_next_round(ctx.fetcher)
    cm.write_parquet(nxt, ctx.out / "odds" / "next_round.parquet")
    man.notes.append(
        f"next_round.parquet: {len(nxt)} Premier League rows fetched at {nxt['fetched_at'].max()}"
    )
    man.coverage = cov
    cm.write_manifest(ctx.out, man)
    print(pd.DataFrame(cov).to_string(index=False))


# ======================================================================================== minutes
def cmd_minutes(ctx: Ctx) -> None:
    from fpl_rocket.asknown import minutes as mn

    repo = ensure_clone(ctx, mn.REPO_URL, "coreinsights_meta")
    ls = cm.git_ls(repo, "HEAD", "data")
    man = cm.SourceManifest(
        source="coreinsights", urls=[mn.REPO_URL], licence="free use with attribution (README)"
    )
    man.notes.append(f"repository HEAD {cm.git(repo, 'rev-parse', 'HEAD').decode().strip()}")
    cov = []
    for season in [s for s in ctx.seasons if s in mn.SEASON_DIRS]:
        files = mn.discover(ls["path"].tolist(), season)
        ms, ps = [], []
        for _, f in files.iterrows():
            raw = cm.cached_blob(repo, "HEAD", f["path"], ctx.raw / "coreinsights")
            (ms if f["kind"] == "matches" else ps).append(
                (mn.parse_matches if f["kind"] == "matches" else mn.parse_player_matches)(raw, int(f["gw"]))
            )
        fill = {}
        eu_path = ctx.out / "european" / "english_club_matches.parquet"
        teams = season_teams_full(ctx, season)
        if eu_path.exists() and teams is not None:
            eu = pd.read_parquet(eu_path)
            dls = deadlines_table(ctx)
            fill = mn.fill_kickoffs(
                mn.dedupe_matches(season, ms),
                eu[eu["season"] == season],
                dict(zip(teams["code"].astype(int), teams["team"].astype(int), strict=True)),
                dls[dls["season"] == season],
            )
        m, p = mn.combine(season, ms, ps, kickoff_fill=fill)
        sd = mn.SEASON_DIRS[season]
        ppath = f"data/{sd}/players/players.csv" if season == "2024-25" else f"data/{sd}/players.csv"
        players = pd.read_csv(
            io.BytesIO(cm.cached_blob(repo, "HEAD", ppath, ctx.raw / "coreinsights")),
            encoding="latin-1" if season == "2024-25" else "utf-8",
            encoding_errors="replace",
        )
        fpl = fpl_codes(ctx, season)
        p = (
            mn.join_fpl(p, players, fpl)
            if fpl is not None
            else p.assign(element=pd.NA, fpl_join="no_bootstrap")
        )
        npxg = mn.team_npxg(m)
        for name, df in (("matches", m), ("player_matches", p), ("team_npxg", npxg)):
            out = ctx.out / "minutes" / season / f"{name}.parquet"
            cm.write_parquet(df, out)
            man.add_output(out, df, ctx.out)
        jr = p["fpl_join"].value_counts(normalize=True).round(4).to_dict()
        cov.append(
            {
                "season": season,
                "gw_files": int(files["gw"].nunique()),
                "matches": len(m),
                "matches_with_kickoff": int(m["kickoff_utc"].notna().sum()),
                "kickoff_filled_from_uefa": int((m["kickoff_source"] == "uefa").sum()),
                "by_tournament": m["tournament"].value_counts().to_dict(),
                "player_match_rows": len(p),
                "rows_with_kickoff": round(float(p["kickoff_utc"].notna().mean()), 4),
                "with_start_min": round(float(p["start_min"].notna().mean()), 4),
                "with_npxg_matches": int(m["home_non_penalty_xg"].notna().sum()),
                "fpl_join": jr,
                "fpl_join_rate": round(float(p["fpl_join"].isin(["code", "id_only"]).mean()), 4),
            }
        )
        print(f"{season}: {len(m)} matches, {len(p)} player-match rows", flush=True)
    man.coverage = cov
    cm.write_manifest(ctx.out, man)
    for c in cov:
        print(c)


def season_teams_full(ctx: Ctx, season: str) -> pd.DataFrame | None:
    """(team, code) of the season from the as-known bootstrap tables."""
    folder = ctx.out / "bootstrap" / "teams" / season
    files = sorted(folder.glob("gw*.parquet")) if folder.exists() else []
    return pd.read_parquet(files[-1], columns=["team", "code"]) if files else None


def fpl_codes(ctx: Ctx, season: str) -> pd.DataFrame | None:
    """(element, code) of every player seen in the season's as-known bootstrap tables."""
    folder = ctx.out / "bootstrap" / "players" / season
    files = sorted(folder.glob("gw*.parquet")) if folder.exists() else []
    if not files:
        return None
    return pd.concat([pd.read_parquet(f, columns=["element", "code"]) for f in files]).drop_duplicates()


# ========================================================================================== crowd
def cmd_crowd(ctx: Ctx) -> None:
    from fpl_rocket.asknown import crowd as cr

    repo = ensure_clone(ctx, cr.REPO_URL, "fpl_optimized_meta", branch=cr.BRANCH)
    dl = deadlines_table(ctx)
    log = cm.git_log_paths(repo, "HEAD", "sample")
    log = log[log["path"].str.match(r"^sample/[0-9-]+/\d+/fpl_sampled\.json$")].copy()
    log["season"] = log["path"].str.split("/").str[1]
    log["gw"] = log["path"].str.split("/").str[2].astype(int)
    man = cm.SourceManifest(
        source="fpl_optimized", urls=[f"{cr.REPO_URL}/tree/{cr.BRANCH}/sample"], licence="see repository"
    )
    seasons_present = sorted(log["season"].unique())
    man.notes.append(f"seasons present in sample/: {seasons_present}")
    for s in ctx.seasons:
        if s not in seasons_present and s >= "2023-24":
            man.gaps.append(f"{s}: no sample/{s}/ folder in the webpage branch")
    t_of = dl.set_index(["season", "gw"])["issue_time_utc"]
    all_m, all_p, ver_rows = [], [], []
    for (season, gw), g in log.groupby(["season", "gw"], sort=True):
        g = g.sort_values("commit_at")
        t_next = t_of.get((season, gw + 1))
        first = g.iloc[0]
        lag_ok = g[g["commit_at"] <= t_next] if t_next is not None else g.iloc[0:0]
        picks = {"comparator": first}
        if len(lag_ok):
            picks["lagged"] = lag_ok.iloc[-1]
        ver_rows.append(
            {
                "season": season,
                "gw": gw,
                "versions": len(g),
                "first_commit_at": first["commit_at"],
                "last_commit_at": g["commit_at"].iloc[-1],
                "next_issue_time": t_next,
                "lagged_version_at": picks["lagged"]["commit_at"] if "lagged" in picks else None,
            }
        )
        for role, v in picks.items():
            raw = cached_blob_xz(repo, v["sha"], v["path"], ctx.raw / "fpl_optimized" / v["sha"][:12])
            m, p = cr.parse_sample(raw, season, gw, v["commit_at"])
            all_m.append(m.assign(role=role, version_sha=v["sha"]))
            all_p.append(p.assign(role=role, version_sha=v["sha"]))
        print(f"{season} GW{gw}: {len(g)} versions, roles {list(picks)}", flush=True)
    M = pd.concat(all_m, ignore_index=True)
    P = pd.concat(all_p, ignore_index=True)
    versions = pd.DataFrame(ver_rows)
    folder = ctx.out / "crowd"
    cm.write_parquet(versions, folder / "versions.parquet")
    for role, label in (("comparator", "after_the_fact"), ("lagged", "lagged_source")):
        m, p = M[M["role"] == role], P[P["role"] == role]
        pl, tiers = cr.aggregate(m, p)
        if role == "comparator":
            # AFTER-THE-FACT comparator: picks revealed after the GW deadline - never an input at T
            for name, df in (("managers", m), ("picks", p), ("players", pl), ("tiers", tiers)):
                cm.write_parquet(
                    df.assign(label="after_the_fact_not_known_at_T"), folder / f"{label}_{name}.parquet"
                )
        else:
            vat = m.groupby(["season", "gw"])["version_committed_at"].max().rename("known_at").reset_index()
            for name, df in (("players", pl), ("tiers", tiers)):
                lagged = cr.lag(df).merge(vat.rename(columns={"gw": "source_gw"}), on=["season", "source_gw"])
                lagged["label"] = "lagged_previous_gw_known_at_T"
                cm.write_parquet(lagged, folder / f"lagged_{name}.parquet")
                man.add_output(folder / f"lagged_{name}.parquet", lagged, ctx.out)
    comp = M[M["role"] == "comparator"]
    counts = comp.groupby(["season", "gw", "tier"]).size().unstack("tier")
    counts.to_csv(folder / "managers_per_tier.csv")
    ov = cr.tier_overlap(comp)
    ov.to_csv(folder / "tier_overlap.csv", index=False)
    cov = []
    for season, g in versions.groupby("season"):
        c = counts.loc[season]
        cov.append(
            {
                "season": season,
                "gws": len(g),
                "gw_range": f"{g['gw'].min()}-{g['gw'].max()}",
                "gws_with_lagged_version_before_next_T": int(g["lagged_version_at"].notna().sum()),
                "gws_recommitted": int((g["versions"] > 1).sum()),
                "median_managers_per_tier": {k: float(v) for k, v in c.median().round(0).items()},
                "gws_with_overlap": int((ov["season"] == season).sum()),
            }
        )
    man.coverage = cov
    cm.write_manifest(ctx.out, man)
    for c in cov:
        print(c)


def cached_blob_xz(repo: Path, rev: str, path: str, cache_root: Path) -> bytes:
    """A large JSON blob cached LZMA-compressed (<cache_root>/<path>.xz); returns the raw bytes."""
    import lzma

    dest = Path(cache_root) / (path + ".xz")
    if dest.exists():
        return lzma.decompress(dest.read_bytes())
    tmp_root = Path(cache_root) / "_tmp"
    raw = cm.cached_blob(repo, rev, path, tmp_root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(lzma.compress(raw, preset=6))
    (tmp_root / path).unlink()
    return raw


# =========================================================================================== uefa
def cmd_uefa(ctx: Ctx) -> None:
    from fpl_rocket.asknown import uefa as ue

    man = cm.SourceManifest(source="uefa", urls=[ue.URL], licence="UEFA website data (private use)")
    frames = []
    for year in ue.SEASON_YEARS:
        season = ue.season_of_year(year)
        if season not in ctx.seasons:
            continue
        for comp in ue.COMPETITIONS:
            offset = 0
            while True:
                dest = ctx.raw / "uefa" / f"{comp}_{year}_{offset}.json"
                refresh = ctx.update and season == CURRENT
                body, _ = ctx.fetcher.cached(
                    ue.URL,
                    dest,
                    refresh=refresh,
                    params={
                        "competitionId": comp,
                        "seasonYear": year,
                        "limit": ue.PAGE,
                        "offset": offset,
                        "order": "ASC",
                    },
                )
                man.add_raw(
                    dest.name, body, f"{ue.URL}?competitionId={comp}&seasonYear={year}&offset={offset}"
                )
                page = ue.parse_page(body, comp)
                if page.empty:
                    break
                frames.append(page)
                if page["match_id"].nunique() < ue.PAGE:
                    break
                offset += ue.PAGE
        print(f"{season}: fetched", flush=True)
    allm = pd.concat(frames, ignore_index=True)
    teams = {s: season_teams(ctx, s) for s in ctx.seasons}
    eng = ue.map_english(allm, {k: v for k, v in teams.items() if v is not None})
    out = ctx.out / "european" / "english_club_matches.parquet"
    cm.write_parquet(eng, out)
    man.add_output(out, eng, ctx.out)
    cov = []
    for season, g in eng.groupby("season"):
        cov.append(
            {
                "season": season,
                "matches": int(g["match_id"].nunique()),
                "by_competition": g.groupby("competition")["match_id"].nunique().to_dict(),
                "clubs": sorted(g["club"].unique().tolist()),
                "unmapped_clubs": sorted(g.loc[g["fpl_team"].isna(), "club"].unique().tolist()),
                "first": str(g["kickoff_utc"].min()),
                "last": str(g["kickoff_utc"].max()),
            }
        )
    man.coverage = cov
    man.notes.append(
        "fixtures as currently recorded: only kickoffs before T are as-known; future dates are not"
    )
    cm.write_manifest(ctx.out, man)
    for c in cov:
        print(c)


# ========================================================================================= report
RULE_NOISE = (
    "phases[",
    "game_config.settings.",
    "game_settings.league_",
    "game_settings.cup_",
    "game_settings.percentile_ranks",
    "game_settings.ui_",
    "game_config.rules.league_",
    "game_config.rules.cup_",
    "game_config.rules.percentile_ranks",
    "game_config.rules.ui_",
)


def rules_summary(evo: pd.DataFrame) -> pd.DataFrame:
    """The rule-relevant rows of rules/evolution.csv (monthly phases, UI and league settings dropped)."""
    keep = ~evo["key"].str.startswith(RULE_NOISE)
    cols = [c for c in ("from_season", "to_season", "within_gw", "key", "before", "after") if c in evo]
    out = evo.loc[keep, cols].fillna("")
    for c in ("before", "after"):
        out[c] = out[c].str.slice(0, 60)
    return out


def cmd_report(ctx: Ctx) -> None:
    folder = ctx.out / "manifest"
    lines = []
    for p in sorted(folder.glob("*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        lines.append(f"## {m['source']}  (fetched {m['fetched_at']})")
        cov = pd.DataFrame(m["coverage"])
        if len(cov):
            lines.append(cov.to_string(index=False))
        for g in m["gaps"][:60]:
            lines.append(f"  gap: {g}")
        if len(m["gaps"]) > 60:
            lines.append(f"  ... {len(m['gaps']) - 60} more gaps")
        for n in m["notes"]:
            lines.append(f"  note: {n}")
        lines.append("")
    evo = ctx.out / "rules" / "evolution.csv"
    if evo.exists():
        lines.append("## rules evolution (chips, game_settings, game_config rules/scoring, squad limits)")
        lines.append(rules_summary(pd.read_csv(evo, dtype=str)).to_string(index=False))
        lines.append("")
    size = sum(f.stat().st_size for f in ctx.out.rglob("*") if f.is_file())
    lines.append(f"data/asknown total: {size / 1e6:.1f} MB")
    text = "\n".join(lines)
    (ctx.out / "REPORT.txt").write_text(text, encoding="utf-8")
    pd.set_option("display.width", 250)
    print(text)


# ========================================================================================== main
SUBCOMMANDS = ("bootstrap", "fixtures", "odds", "uefa", "minutes", "crowd", "report")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=(*SUBCOMMANDS, "all"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--seasons", nargs="+", default=list(cm.SEASONS))
    ap.add_argument("--update", action="store_true", help="refresh clones and current-season downloads")
    args = ap.parse_args(argv)
    ctx = Ctx(Path(args.data), args.seasons, args.update)
    cmds = SUBCOMMANDS if args.cmd == "all" else (args.cmd,)
    for c in cmds:
        print(f"== {c}", flush=True)
        globals()[f"cmd_{c}"](ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
