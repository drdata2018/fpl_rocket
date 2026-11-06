"""All-competition player-match minutes, xG/xA and penalties from github.com/olbauday/FPL-Core-Insights.

Layouts (discovered from the repository tree, 2026-11-06):
    2024-25  data/2024-2025/matches/GW<n>/matches.csv,
             data/2024-2025/playermatchstats/GW<n>/playermatchstats.csv
             (Premier League only; kickoff_time has no zone and is UTC), data/2024-2025/players/players.csv
    2025-26  data/2025-2026/By Gameweek/GW<n>/{matches,playermatchstats}.csv  (all competitions of the FPL GW
             window: Premier League, EFL Cup, Champions/Europa/Conference League; `tournament` column),
             mirrored per competition in By Tournament/<name>/GW<n>/; data/2025-2026/players.csv
    2026-27  as 2025-26, plus Community Shield, UEFA Super Cup and pre-season Friendlies (GW0, By Tournament
             only)
`player_id` is the FPL element id of that season and `player_code` the FPL code (players.csv); the join rate
to
the as-known bootstrap is measured, not assumed. The README states `minutes_played` is the source of truth and
`start_min` / `finish_min` are best effort (a fixture without a substitution timeline shows every player
starting at 0), so `started` is derived and flagged as such.

AS-KNOWN caveat: the repository publishes the CURRENT version of past matches (stats can be revised after the
fact); only the match kickoff can be used to filter, so a row is usable at T only if
kickoff_utc + FINISH_BUFFER <= T (the match had certainly finished, extra time and penalties included).
Untrusted data: pandas only.
"""

from __future__ import annotations

import io
import re

import pandas as pd

SOURCE = "coreinsights"
REPO_URL = "https://github.com/olbauday/FPL-Core-Insights"
FINISH_BUFFER = pd.Timedelta(hours=3)
SEASON_DIRS = {"2024-25": "2024-2025", "2025-26": "2025-2026", "2026-27": "2026-2027"}

PLAYER_MATCH_COLUMNS = (
    "player_id",
    "match_id",
    "minutes_played",
    "start_min",
    "finish_min",
    "goals",
    "assists",
    "xg",
    "xa",
    "penalties_scored",
    "penalties_missed",
    "total_shots",
)
MATCH_COLUMNS = (
    "match_id",
    "kickoff_time",
    "gameweek",
    "tournament",
    "finished",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_expected_goals_xg",
    "away_expected_goals_xg",
    "home_non_penalty_xg",
    "away_non_penalty_xg",
    "home_xg_set_play",
    "away_xg_set_play",
)
_GW_RE = re.compile(r"/GW(\d+)/")


def discover(paths: list[str], season: str) -> pd.DataFrame:
    """Repository paths -> (kind, gw, path) of the per-GW match and player-match files of a season."""
    root = f"data/{SEASON_DIRS[season]}/"
    rows = []
    for p in paths:
        if not p.startswith(root):
            continue
        rest = p[len(root) :]
        name = rest.rsplit("/", 1)[-1]
        if name not in ("matches.csv", "playermatchstats.csv"):
            continue
        m = _GW_RE.search("/" + rest)
        if not m:
            continue  # season aggregates: the per-GW files are used
        gw = int(m.group(1))
        if season == "2024-25":
            ok = rest.startswith(("matches/GW", "playermatchstats/GW"))
        else:
            # By Gameweek holds every competition of the GW window; By Tournament adds only GW0 (friendlies)
            ok = rest.startswith("By Gameweek/") or (rest.startswith("By Tournament/") and gw == 0)
        if ok:
            rows.append(("matches" if name == "matches.csv" else "playermatchstats", gw, p))
    return (
        pd.DataFrame(rows, columns=["kind", "gw", "path"])
        .sort_values(["kind", "gw", "path"])
        .reset_index(drop=True)
    )


def _read(raw: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(raw), low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(raw), low_memory=False, encoding="latin-1")


def parse_matches(raw: bytes, gw_dir: int) -> pd.DataFrame:
    df = _read(raw)
    out = pd.DataFrame({c: df.get(c, pd.NA) for c in MATCH_COLUMNS})
    out["tournament"] = out["tournament"].fillna("prem").astype("string")  # 2024-25: Premier League only
    out["kickoff_utc"] = pd.to_datetime(out.pop("kickoff_time"), utc=True, errors="coerce", format="mixed")
    out["gw_dir"] = gw_dir
    for c in (
        "home_expected_goals_xg",
        "away_expected_goals_xg",
        "home_non_penalty_xg",
        "away_non_penalty_xg",
        "home_xg_set_play",
        "away_xg_set_play",
    ):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Float64")
    for c in ("gameweek", "home_score", "away_score"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    out["match_id"] = out["match_id"].astype("string")
    return out


def parse_player_matches(raw: bytes, gw_dir: int) -> pd.DataFrame:
    df = _read(raw)
    out = pd.DataFrame({c: df.get(c, pd.NA) for c in PLAYER_MATCH_COLUMNS})
    for c in (
        "player_id",
        "minutes_played",
        "start_min",
        "finish_min",
        "goals",
        "assists",
        "penalties_scored",
        "penalties_missed",
        "total_shots",
    ):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    for c in ("xg", "xa"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Float64")
    out["match_id"] = out["match_id"].astype("string")
    out["gw_dir"] = gw_dir
    return out


def dedupe_matches(season: str, matches: list[pd.DataFrame]) -> pd.DataFrame:
    """One row per match (a later GW file wins: the newest published version)."""
    m = pd.concat(matches, ignore_index=True).sort_values("gw_dir", kind="stable")
    m = m.drop_duplicates("match_id", keep="last").reset_index(drop=True)
    m.insert(0, "season", season)
    return m


UEFA_COMP = {"champions-league": "UCL", "europa-league": "UEL", "conference-league": "UECL"}


def fill_kickoffs(
    m: pd.DataFrame, eu: pd.DataFrame, team_code_to_id: dict[int, int], deadlines: pd.DataFrame
) -> dict[str, pd.Timestamp]:
    """{match_id: kickoff} for European matches the source lists without a kickoff, from the UEFA table:
    same competition, same English club (FPL team code -> id), same home/away side, the opponent's name
    matching the match_id slug, kickoff within 45 days of the deadline of the GW the source files it under
    (that GW label is loose: a 11 March match was filed under GW31); only unique candidates are used. Draw
    placeholders ("bologna/roma-vs-...") are skipped."""
    d = deadlines.set_index("gw")["deadline_utc"]
    out = {}
    todo = m["kickoff_utc"].isna() & m["tournament"].isin(list(UEFA_COMP)) & ~m["match_id"].str.contains("/")
    for _, r in m[todo].iterrows():
        gw = r["gameweek"] if pd.notna(r["gameweek"]) else r["gw_dir"]
        if pd.isna(gw) or int(gw) not in d.index:
            continue
        lo, hi = d[int(gw)] - pd.Timedelta(days=45), d[int(gw)] + pd.Timedelta(days=45)
        prefix = f"{r['season'][2:4]}-{r['season'][5:]}-{r['tournament']}-" if "season" in r else ""
        slug = (
            str(r["match_id"])[len(prefix) :] if str(r["match_id"]).startswith(prefix) else str(r["match_id"])
        )
        home_slug, _, away_slug = slug.partition("-vs-")
        for side in ("home", "away"):
            code = r[f"{side}_team"]
            if pd.isna(code) or int(code) not in team_code_to_id:
                continue
            keys = _name_keys(away_slug if side == "home" else home_slug)
            c = eu[
                (eu["competition"] == UEFA_COMP[r["tournament"]])
                & (eu["fpl_team"] == team_code_to_id[int(code)])
                & (eu["is_home"] == (side == "home"))
                & eu["kickoff_utc"].between(lo, hi)
            ]
            c = c[c["opponent"].map(lambda o, k=keys: any(x in _ascii(o) for x in k))]
            if c["match_id"].nunique() == 1:
                out[r["match_id"]] = c["kickoff_utc"].iloc[0]
                break
    return out


_STOP = {"fc", "cf", "ac", "sc", "afc", "fk", "sk", "cp", "as", "ss", "rc", "rb", "1", "club"}


def _ascii(s) -> str:
    import unicodedata

    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def _name_keys(slug: str) -> list[str]:
    """Distinctive tokens of a club slug, 5 letters each (atletico/atleti, bayer-leverkusen/leverkusen)."""
    toks = [t for t in _ascii(slug).replace("�", "").split("-") if len(t) >= 4 and t not in _STOP]
    return [t[:5] for t in toks]


def combine(
    season: str,
    matches: list[pd.DataFrame],
    pms: list[pd.DataFrame],
    kickoff_fill: dict[str, pd.Timestamp] | None = None,
):
    """Deduplicate (a later GW file wins: the newest published version) and attach kickoffs (`kickoff_fill`
    supplies kickoffs the source lacks; `kickoff_source` says which). Returns (matches, player_matches)."""
    m = dedupe_matches(season, matches)
    m["kickoff_source"] = pd.Series("coreinsights", index=m.index, dtype="string").where(
        m["kickoff_utc"].notna()
    )
    if kickoff_fill:
        fill = m["match_id"].map(kickoff_fill)
        use = m["kickoff_utc"].isna() & fill.notna()
        m.loc[use, "kickoff_utc"] = pd.to_datetime(fill[use], utc=True)
        m.loc[use, "kickoff_source"] = "uefa"
    m["known_at"] = m["kickoff_utc"] + FINISH_BUFFER
    p = pd.concat(pms, ignore_index=True).sort_values("gw_dir", kind="stable")
    p = p.dropna(subset=["player_id", "match_id"]).drop_duplicates(["player_id", "match_id"], keep="last")
    p = p.merge(
        m[["match_id", "kickoff_utc", "tournament", "gameweek", "home_team", "away_team"]],
        on="match_id",
        how="left",
    )
    p.insert(0, "season", season)
    played = p["minutes_played"].fillna(0) > 0
    p["started"] = (played & (p["start_min"] == 0)).astype("boolean")  # derived; start_min is best effort
    p["known_at"] = p["kickoff_utc"] + FINISH_BUFFER
    return m, p.reset_index(drop=True)


def team_npxg(matches: pd.DataFrame) -> pd.DataFrame:
    """Per (match, side) non-penalty xG for and against."""
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        rows.append(
            pd.DataFrame(
                {
                    "season": matches["season"],
                    "match_id": matches["match_id"],
                    "kickoff_utc": matches["kickoff_utc"],
                    "tournament": matches["tournament"],
                    "team_code": pd.to_numeric(matches[f"{side}_team"], errors="coerce").astype("Int64"),
                    "is_home": side == "home",
                    "xg": matches[f"{side}_expected_goals_xg"],
                    "npxg": matches[f"{side}_non_penalty_xg"],
                    "xg_set_play": matches[f"{side}_xg_set_play"],
                    "npxg_against": matches[f"{opp}_non_penalty_xg"],
                    "known_at": matches["known_at"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def join_fpl(pm: pd.DataFrame, players: pd.DataFrame, fpl: pd.DataFrame) -> pd.DataFrame:
    """Attach the FPL element via player_code (players.csv) -> bootstrap code; `fpl_join` says how it matched:
    'code' (code matches and the element equals player_id), 'code_other_id' (code matches, element differs),
    'id_only' (no code; the player_id is an FPL element), 'none'."""
    codes = players[["player_id", "player_code"]].drop_duplicates("player_id")
    codes = codes.assign(
        player_id=pd.to_numeric(codes["player_id"], errors="coerce").astype("Int64"),
        player_code=pd.to_numeric(codes["player_code"], errors="coerce").astype("Int64"),
    )
    fpl = fpl[["element", "code"]].drop_duplicates("code")
    out = pm.merge(codes, on="player_id", how="left").merge(
        fpl.rename(columns={"code": "player_code"}), on="player_code", how="left"
    )
    known = set(pd.to_numeric(fpl["element"]).dropna().astype(int))
    has = out["element"].notna()
    same = has & (out["element"] == out["player_id"])
    out["fpl_join"] = "none"
    out.loc[has & same, "fpl_join"] = "code"
    out.loc[has & ~same, "fpl_join"] = "code_other_id"
    out.loc[~has & out["player_id"].isin(known), "fpl_join"] = "id_only"
    out["element"] = (
        out["element"].fillna(out["player_id"].where(out["fpl_join"] == "id_only")).astype("Int64")
    )
    return out
