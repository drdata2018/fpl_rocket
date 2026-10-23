"""status_history: FPL availability flags carried from bundle to bundle, and the recency features derived
from it.

status_history (one row per element and observation where something changed):
    element, observed_at (this bundle's build time, UTC), prev_observed_at (the previous bundle's build time:
    the
    change happened in (prev_observed_at, observed_at]; null for "initial"), status,
    chance_of_playing_next_round, chance_of_playing_this_round, news, news_added (FPL's own timestamp, UTC),
    change ("initial" the first time a player is seen, else the changed fields joined by "+": status, chance,
    news, news_added)

Each build reads the previous release's status_history, compares the last row per element with the current
bootstrap and appends a row for every element whose status / chance_of_playing_next_round / news / news_added
differs (or that is new). The table therefore grows with real changes only (hourly bundles; the 48 h release
retention does not matter because every bundle carries the full table forward).

Recency per player (recency_features):
    status_changed_at   last history row whose change includes status or chance; when the history has none
                        (the
                        flag predates the first bundle with status_history) FPL's news_added (source
                        "news_added"), else the first observation (source "first_seen", a lower bound)
    days_since_status_change, days_since_news_added (NaN without news_added)
    status_transitions_last_14d   history rows in the last 14 days whose change includes status (initial rows
                                  excluded)
    stale               status d / i AND news_added older than ModelConfig.stale_after_days AND no status /
                        chance change since then (days_since_status_change > stale_after_days)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

HISTORY_COLUMNS = (
    "element",
    "observed_at",
    "prev_observed_at",
    "status",
    "chance_of_playing_next_round",
    "chance_of_playing_this_round",
    "news",
    "news_added",
    "change",
)
TRACKED = (
    ("status", "status"),
    ("chance_of_playing_next_round", "chance"),
    ("news", "news"),
    ("news_added", "news_added"),
)


def _utc(s: Any) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")


def current_flags(elements: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "element": [int(e["id"]) for e in elements],
            "status": [str(e.get("status") or "a") for e in elements],
            "chance_of_playing_next_round": [e.get("chance_of_playing_next_round") for e in elements],
            "chance_of_playing_this_round": [e.get("chance_of_playing_this_round") for e in elements],
            "news": [str(e.get("news") or "") for e in elements],
            "news_added": [e.get("news_added") for e in elements],
        }
    )
    for c in ("chance_of_playing_next_round", "chance_of_playing_this_round"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["news_added"] = _utc(df["news_added"])
    return df


def empty_history() -> pd.DataFrame:
    df = pd.DataFrame({c: [] for c in HISTORY_COLUMNS})
    return _typed(df)


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=list(HISTORY_COLUMNS))
    df["element"] = df["element"].astype("int64")
    for c in ("observed_at", "prev_observed_at", "news_added"):
        df[c] = _utc(df[c])
    for c in ("chance_of_playing_next_round", "chance_of_playing_this_round"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in ("status", "news", "change"):
        df[c] = df[c].astype(object).where(df[c].notna(), "").astype(str)
    return df


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    both_na = a.isna().to_numpy() & b.isna().to_numpy()
    eq = a.astype(object).to_numpy() == b.astype(object).to_numpy()
    return both_na | eq


def update_status_history(
    previous: pd.DataFrame | None,
    elements: list[dict[str, Any]],
    now: datetime,
    prev_observed_at: datetime | None,
) -> pd.DataFrame:
    """previous status_history (or None) + rows for every element whose flags changed / that is new."""
    hist = _typed(previous.copy()) if previous is not None and len(previous) else empty_history()
    cur = current_flags(elements)
    now_ts = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo else pd.Timestamp(now, tz="UTC")
    prev_ts = pd.Timestamp(prev_observed_at) if prev_observed_at is not None else pd.NaT
    if prev_ts is not pd.NaT and prev_ts.tzinfo is None:
        prev_ts = prev_ts.tz_localize("UTC")
    last = hist.sort_values(["element", "observed_at"], kind="stable").groupby("element").tail(1)
    j = cur.merge(last, on="element", how="left", suffixes=("", "_prev"), indicator=True)
    new = (j["_merge"] == "left_only").to_numpy()
    changes = []
    for i in range(len(j)):
        if new[i]:
            changes.append("initial")
            continue
        parts = [
            label for col, label in TRACKED if not _same(j[col].iloc[[i]], j[f"{col}_prev"].iloc[[i]])[0]
        ]
        changes.append("+".join(parts))
    j["change"] = changes
    add = j[j["change"] != ""].copy()
    if add.empty:
        return hist.sort_values(["element", "observed_at"], kind="stable").reset_index(drop=True)
    add["observed_at"] = now_ts
    add["prev_observed_at"] = np.where(add["change"] == "initial", pd.NaT, prev_ts)
    out = (
        pd.concat([hist, _typed(add[list(HISTORY_COLUMNS)])], ignore_index=True)
        if len(hist)
        else _typed(add[list(HISTORY_COLUMNS)])
    )
    return _typed(out).sort_values(["element", "observed_at"], kind="stable").reset_index(drop=True)


def _lookup(by_element: pd.Series, out: pd.DataFrame) -> pd.Series:
    """by_element (indexed by element) aligned to out's rows (missing -> NaN / NaT; works when empty)."""
    vals = by_element.reindex(out["element"].to_numpy())
    return pd.Series(vals.to_numpy(), index=out.index, dtype=vals.dtype)


def recency_features(
    history: pd.DataFrame, elements: list[dict[str, Any]], now: datetime, cfg
) -> pd.DataFrame:
    """Per current element: status_changed_at (+ source), days_since_status_change, days_since_news_added,
    status_transitions_last_14d, stale (see the module docstring)."""
    cur = current_flags(elements)
    now_ts = pd.Timestamp(now)
    now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts.tz_convert("UTC")
    h = _typed(history.copy()) if len(history) else empty_history()
    ch = h[h["change"].str.contains("status|chance", regex=True)]
    last_change = ch.groupby("element")["observed_at"].max()
    first_seen = h.groupby("element")["observed_at"].min()
    recent = h[(h["observed_at"] >= now_ts - pd.Timedelta(days=14)) & h["change"].str.contains("status")]
    recent = recent[recent["change"] != "initial"]
    transitions = recent.groupby("element").size()

    out = cur[["element", "status", "chance_of_playing_next_round", "news", "news_added"]].copy()
    hist_at = _lookup(last_change, out)
    seen_at = _lookup(first_seen, out)
    na = out["news_added"]
    use_hist = hist_at.notna()
    use_news = ~use_hist & na.notna()
    out["status_changed_at"] = hist_at.where(use_hist, na.where(use_news, seen_at))
    out["status_change_source"] = np.where(
        use_hist, "history", np.where(use_news, "news_added", "first_seen")
    )
    out["status_changed_at"] = _utc(out["status_changed_at"])
    day = pd.Timedelta(days=1)
    out["days_since_status_change"] = ((now_ts - out["status_changed_at"]) / day).astype(float)
    out["days_since_news_added"] = ((now_ts - na) / day).astype(float)
    out["status_transitions_last_14d"] = _lookup(transitions, out).fillna(0).astype("int64")
    lim = float(cfg.stale_after_days)
    out["stale"] = (
        out["status"].isin(["d", "i"])
        & (out["days_since_news_added"] > lim)
        & (out["days_since_status_change"].fillna(np.inf) > lim)
    ).astype(bool)
    return out
