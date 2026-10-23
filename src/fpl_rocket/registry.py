"""Prospective registry (docs/PLAN_v2.md section 4): the real scorecard of the model and the planner.

WRITE (bundle workflow, pre-deadline bundle only; scripts/registry_write.py): next to the bundle files of the
release `deadline-gw<N>` (the scored snapshot: the bundle created 30-90 min before the deadline):
    registry_xp.parquet   the xp table AS ISSUED (+ bundle_id, issued_gw, and `baseline_xp` = the frozen
                          baseline's prediction for GW N: form per 90 x availability)
    registry_plans.json   objective (a) (next GW, no transfers) for a fixed roster of entry ids computed from
                          their public planning state at that moment, with the model xP; plus the baseline's
                          XI / captain for the same squads
    registry_meta.json    deadline, created_at, bundle id/time, model_version, rules_version, exclusions
                          (low_history count, players without xP), roster, sha256 of the two other files

SCORE (scripts/registry_score.py, daily workflow registry_score.yml): once GW N is finished and checked,
download the deadline-gw<N> assets and score: xP RMSE / Spearman vs actual points (all players and players
who played) for the model and the baseline; per roster entry: captain pick points vs the best captain
available in the squad and vs the manager's ACTUAL captain, planned XI points (automatic substitutions,
armband) vs the manager's actual GW points. Rows are appended to registry_scores.csv on the `spike-log`
branch. A week is NEVER skipped: no deadline release, or a release without registry assets, gives a row with
status=missing.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .backtest import spearman
from .config import PLACEHOLDER_TEAM_ID, RELEASES_URL
from .planner.lineup import realised_points

JSON = dict[str, Any]

# fixed roster: the owner's placeholder id (config) and four public managers
REGISTRY_ENTRY_IDS: tuple[int, ...] = (int(PLACEHOLDER_TEAM_ID), 895045, 4182037, 7398185, 9414355)
# the first deadline the registry covers (the first GW after the planner branch is merged): every finished GW
# from here on gets a row, missing weeks included. Update when the branch is merged.
REGISTRY_START_GW = 9
XP_FILE, PLANS_FILE, META_FILE = "registry_xp.parquet", "registry_plans.json", "registry_meta.json"
FILES = (XP_FILE, PLANS_FILE, META_FILE)
SCORES_FILE = "registry/registry_scores.csv"  # path on the spike-log branch
SCORES_URL = "https://raw.githubusercontent.com/drdata2018/fpl_rocket/spike-log/" + SCORES_FILE
BASELINE_NAME = "form per 90 x availability (frozen baseline)"
SCORE_COLUMNS = (
    "gw",
    "kind",  # "xp" (one per GW) | "entry" (one per roster id)
    "entry_id",
    "status",  # ok | missing | error
    "reason",
    "scored_at",
    "bundle_id",
    "model_version",
    "n_players",
    "n_played",
    "rmse_model",
    "rmse_baseline",
    "spearman_model",
    "spearman_baseline",
    "spearman_played_model",
    "spearman_played_baseline",
    "captain_model",
    "captain_model_pts",
    "captain_baseline",
    "captain_baseline_pts",
    "captain_best_pts",
    "captain_manager",
    "captain_manager_pts",
    "xi_model_pts",
    "xi_baseline_pts",
    "manager_pts",
    "manager_chip",
    "squad_overlap",
)


def deadline_tag(gw: int) -> str:
    return f"deadline-gw{int(gw)}"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ============================================================================================ write
def _plan_record(plan: Any) -> JSON:
    gp = plan.first()
    return {
        "status": plan.status,
        "reason": plan.reason,
        "gw": gp.gw,
        "squad": gp.squad,
        "xi": gp.xi,
        "captain": gp.captain,
        "vice": gp.vice,
        "bench": gp.bench,
        "expected_points": round(gp.expected_points, 3),
        "explanation": plan.explanation,
        "warnings": plan.warnings,
    }


def baseline_xp(history_rows: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
    """The frozen baseline's xP per element for the GW of `targets` (player-fixture rows of ONE GW)."""
    from .backtest import FormPer90Baseline

    tp = targets.groupby("element", as_index=False).agg(
        code=("code", "first"), element_type=("element_type", "first"), n_fixtures=("fixture", "nunique")
    )
    base = FormPer90Baseline()
    base.fit(history_rows)
    pred = base.predict(tp, int(targets["gw"].iloc[0]))
    return pd.Series(pred.to_numpy(float), index=tp["element"].astype(int).to_numpy(), name="baseline_xp")


def build_registry(
    *,
    manifest: JSON,
    bootstrap: JSON,
    xp: pd.DataFrame,
    baseline: pd.Series,
    states: dict[int, Any],
    errors: dict[int, str] | None = None,
    created_at: str | None = None,
) -> tuple[pd.DataFrame, JSON, JSON]:
    """(registry_xp, registry_plans, registry_meta) from already-built pieces (pure; tested offline).

    states: {entry_id: PlanningState} for the roster ids that could be built; errors: {entry_id: reason}."""
    from . import views
    from .planner import ONE_GW_NO_TRANSFERS, PlanSettings, from_planning_state, solve

    errors = dict(errors or {})
    gw = int(manifest.get("next_event") or min(int(g) for g in xp["gw"].unique()))
    reg_xp = xp.copy()
    if "low_history" not in reg_xp:
        reg_xp["low_history"] = False
    reg_xp["bundle_id"] = str(manifest["bundle_id"])
    reg_xp["issued_gw"] = gw
    reg_xp["baseline_xp"] = np.where(
        reg_xp["gw"] == gw, reg_xp["element"].map(baseline).fillna(0.0).to_numpy(float), np.nan
    )
    players = views.players_frame(bootstrap)
    base_df = pd.DataFrame({"element": baseline.index, "gw": gw, "xp": baseline.to_numpy(float)})
    entries = []
    for entry_id in REGISTRY_ENTRY_IDS:
        if entry_id not in states:
            entries.append(
                {"entry_id": entry_id, "status": "error", "error": errors.get(entry_id, "no planning state")}
            )
            continue
        st = states[entry_id]
        try:
            settings = PlanSettings(objective=ONE_GW_NO_TRANSFERS)
            model_plan = solve(
                from_planning_state(st, players, xp[xp["gw"] == gw], bootstrap=bootstrap), settings
            )
            base_plan = solve(from_planning_state(st, players, base_df, bootstrap=bootstrap), settings)
        except Exception as exc:  # noqa: BLE001 - one entry must not sink the registry
            entries.append({"entry_id": entry_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            continue
        entries.append(
            {
                "entry_id": entry_id,
                "status": "ok",
                "state": {
                    "upcoming_gw": st.upcoming_gw,
                    "last_published_gw": st.last_published_gw,
                    "transfers_since_publication": st.transfers_since_publication,
                    "bank": st.bank,
                    "free_transfers": str(st.free_transfers),
                    "squad": sorted(p.element for p in st.squad),
                    "warnings": len(st.warnings),
                },
                "model": _plan_record(model_plan),
                "baseline": _plan_record(base_plan),
            }
        )
    plans = {"bundle_id": str(manifest["bundle_id"]), "gw": gw, "entries": entries}
    gw_xp = reg_xp[reg_xp["gw"] == gw]
    meta = {
        "gw": gw,
        "deadline": manifest.get("next_deadline_time"),
        "created_at": created_at or _now(),
        "bundle_id": str(manifest["bundle_id"]),
        "bundle_created_at": manifest.get("created_at"),
        "pre_deadline": bool(manifest.get("pre_deadline")),
        "model_version": manifest.get("model_version"),
        "rules_version": manifest.get("rules_version"),
        "baseline": BASELINE_NAME,
        "roster": list(REGISTRY_ENTRY_IDS),
        "exclusions": {
            "low_history": int(gw_xp["low_history"].fillna(False).astype(bool).sum()),
            "players_without_xp": int(len(bootstrap.get("elements", [])) - gw_xp["element"].nunique()),
            "entries_failed": sum(1 for e in entries if e["status"] != "ok"),
        },
    }
    return reg_xp, plans, meta


def write_files(out_dir: str | Path, reg_xp: pd.DataFrame, plans: JSON, meta: JSON) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    reg_xp.to_parquet(buf, index=False)
    xp_bytes = buf.getvalue()
    plans_bytes = json.dumps(plans, indent=1, default=str).encode("utf-8")
    meta = {**meta, "sha256": {XP_FILE: _sha(xp_bytes), PLANS_FILE: _sha(plans_bytes)}}
    (out / XP_FILE).write_bytes(xp_bytes)
    (out / PLANS_FILE).write_bytes(plans_bytes)
    (out / META_FILE).write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return [out / f for f in FILES]


# ============================================================================================ read
class RegistryMissing(Exception):
    """No usable registry assets for a GW (the week is scored as missing)."""


def _get(source: str, gw: int, name: str) -> bytes:
    if not source.startswith(("http://", "https://")):
        path = Path(source) / deadline_tag(gw) / name
        if not path.exists():
            raise FileNotFoundError(str(path))
        return path.read_bytes()
    url = f"{source.rstrip('/')}/download/{deadline_tag(gw)}/{name}"
    resp = requests.get(url, timeout=(5.0, 60.0), headers={"User-Agent": "fpl-rocket-registry"})
    resp.raise_for_status()
    return resp.content


def read_registry(gw: int, source: str = RELEASES_URL) -> tuple[pd.DataFrame, JSON, JSON]:
    """(registry_xp, plans, meta) of deadline-gw<gw>; raises RegistryMissing with the reason."""
    try:
        meta_raw = _get(source, gw, META_FILE)
    except (FileNotFoundError, requests.HTTPError) as exc:
        code = getattr(getattr(exc, "response", None), "status_code", None)
        try:  # distinguish "no deadline release" from "a release without registry assets"
            _get(source, gw, "manifest.json")
            reason = "deadline release has no registry assets"
        except Exception:  # noqa: BLE001
            reason = "no deadline release (no bundle published in the pre-deadline window)"
        raise RegistryMissing(reason if code in (None, 404) else f"download failed (HTTP {code})") from exc
    meta = json.loads(meta_raw)
    xp_raw, plans_raw = _get(source, gw, XP_FILE), _get(source, gw, PLANS_FILE)
    sums = meta.get("sha256") or {}
    for name, raw in ((XP_FILE, xp_raw), (PLANS_FILE, plans_raw)):
        if sums.get(name) and sums[name] != _sha(raw):
            raise RegistryMissing(f"{name} checksum mismatch")
    return pd.read_parquet(io.BytesIO(xp_raw)), json.loads(plans_raw), meta


# ============================================================================================ score
def missing_row(gw: int, reason: str) -> JSON:
    return {"gw": int(gw), "kind": "xp", "status": "missing", "reason": reason, "scored_at": _now()}


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def score_gw(
    gw: int,
    reg_xp: pd.DataFrame,
    plans: JSON,
    meta: JSON,
    live: JSON,
    picks: dict[int, JSON | None],
    bootstrap: JSON,
) -> list[JSON]:
    """Score one GW (pure). live = event/{gw}/live; picks = {entry_id: entry picks of GW gw}."""
    stats = {int(e["id"]): e.get("stats", {}) for e in live.get("elements", [])}
    pts = {e: int(s.get("total_points", 0) or 0) for e, s in stats.items()}
    mins = {e: int(s.get("minutes", 0) or 0) for e, s in stats.items()}
    et = {int(p["id"]): int(p["element_type"]) for p in bootstrap.get("elements", [])}
    common = {
        "gw": int(gw),
        "scored_at": _now(),
        "bundle_id": meta.get("bundle_id"),
        "model_version": meta.get("model_version"),
    }
    x = reg_xp[(reg_xp["gw"] == gw) & reg_xp["element"].isin(list(pts))].copy()
    x["actual"] = x["element"].map(pts).astype(float)
    x["played"] = x["element"].map(mins).fillna(0) > 0
    rows = []
    xp_row = {
        **common,
        "kind": "xp",
        "status": "ok",
        "reason": "",
        "n_players": len(x),
        "n_played": int(x["played"].sum()),
    }
    for col, tag in (("xp", "model"), ("baseline_xp", "baseline")):
        if col not in x or x[col].isna().all():
            continue
        v = x.dropna(subset=[col])
        p = v[v["played"]]
        xp_row[f"rmse_{tag}"] = _rmse(v[col].to_numpy(float), v["actual"].to_numpy(float))
        xp_row[f"spearman_{tag}"] = spearman(v[col], v["actual"])
        xp_row[f"spearman_played_{tag}"] = spearman(p[col], p["actual"])
    rows.append(xp_row)
    for e in plans.get("entries", []):
        eid = int(e["entry_id"])
        row = {**common, "kind": "entry", "entry_id": eid}
        if e.get("status") != "ok":
            rows.append({**row, "status": "error", "reason": e.get("error", "")})
            continue
        mp, bp = e["model"], e["baseline"]
        squad = mp["squad"]
        row.update(
            status="ok",
            reason="",
            captain_model=mp["captain"],
            captain_model_pts=pts.get(mp["captain"], 0),
            captain_baseline=bp["captain"],
            captain_baseline_pts=pts.get(bp["captain"], 0),
            captain_best_pts=max((pts.get(p, 0) for p in squad), default=0),
            xi_model_pts=realised_points(mp["xi"], mp["bench"], mp["captain"], mp["vice"], pts, mins, et)[
                "points"
            ],
            xi_baseline_pts=realised_points(bp["xi"], bp["bench"], bp["captain"], bp["vice"], pts, mins, et)[
                "points"
            ],
        )
        pk = picks.get(eid)
        if pk:
            cap = next((int(p["element"]) for p in pk.get("picks", []) if p.get("is_captain")), None)
            actual = {int(p["element"]) for p in pk.get("picks", [])}
            row.update(
                captain_manager=cap,
                captain_manager_pts=pts.get(cap, 0) if cap is not None else None,
                manager_pts=(pk.get("entry_history") or {}).get("points"),
                manager_chip=pk.get("active_chip") or "",
                squad_overlap=len(actual & set(squad)),
            )
        else:
            row["reason"] = "manager picks unavailable"
        rows.append(row)
    return rows


def scores_frame(rows: Iterable[JSON]) -> pd.DataFrame:
    return pd.DataFrame(list(rows)).reindex(columns=list(SCORE_COLUMNS))


def append_scores(csv_path: str | Path, rows: list[JSON]) -> pd.DataFrame:
    """Append rows to registry_scores.csv (created with the header when missing); returns the whole table."""
    path = Path(csv_path)
    old = pd.read_csv(path) if path.exists() and path.stat().st_size else scores_frame([])
    new = scores_frame(rows)
    out = new if old.empty else pd.concat([old, new], ignore_index=True)
    out = out.reindex(columns=list(SCORE_COLUMNS)).sort_values(
        ["gw", "kind", "entry_id"], kind="stable", na_position="first"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def scored_gws(csv_path: str | Path) -> set[int]:
    path = Path(csv_path)
    if not path.exists() or not path.stat().st_size:
        return set()
    df = pd.read_csv(path)
    return set(int(g) for g in df["gw"].dropna().unique()) if "gw" in df else set()


def finished_gws(bootstrap: JSON) -> list[int]:
    """GWs whose points are final: finished and data_checked (when the field is present)."""
    return sorted(
        int(e["id"]) for e in bootstrap.get("events", []) if e.get("finished") and e.get("data_checked", True)
    )


def pending_gws(bootstrap: JSON, scored: set[int], start: int = REGISTRY_START_GW) -> list[int]:
    return [g for g in finished_gws(bootstrap) if g >= start and g not in scored]


def score_and_append(gw: int, csv_path: str | Path, client: Any, source: str = RELEASES_URL) -> list[JSON]:
    """Download deadline-gw<gw> assets, score against the FPL API, append to the CSV. A missing week still
    gets a row (status=missing)."""
    try:
        reg_xp, plans, meta = read_registry(gw, source)
    except RegistryMissing as exc:
        rows = [missing_row(gw, str(exc))]
    else:
        bootstrap = client.bootstrap()
        live = client.event_live(gw)
        picks = {}
        for e in plans.get("entries", []):
            try:
                picks[int(e["entry_id"])] = client.entry_picks(int(e["entry_id"]), gw)
            except Exception:  # noqa: BLE001 - scored without the manager comparison
                picks[int(e["entry_id"])] = None
        rows = score_gw(gw, reg_xp, plans, meta, live, picks, bootstrap)
    append_scores(csv_path, rows)
    return rows


# ============================================================================================ app
def latest_scores(df: pd.DataFrame | None) -> JSON | None:
    """Summary of the last scored GW for the app: {gw, status, xp row, entries frame} or None."""
    if df is None or df.empty or "gw" not in df:
        return None
    gw = int(df["gw"].max())
    g = df[df["gw"] == gw]
    xp_rows = g[g["kind"] == "xp"]
    xp_row = xp_rows.iloc[0].to_dict() if len(xp_rows) else {}
    entries = g[g["kind"] == "entry"]
    return {
        "gw": gw,
        "status": xp_row.get("status", "ok"),
        "xp": xp_row,
        "entries": entries.reset_index(drop=True),
    }
