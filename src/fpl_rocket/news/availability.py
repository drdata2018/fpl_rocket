"""availability.parquet: per player, the GW+1 P(play) cap without and with news, and everything behind it.

Caps (GW+1; later GWs decay the shortfall by ModelConfig.availability_decay per GW in the minutes model):
    cap_fpl          FPL flag only: chance_of_playing_next_round / 100; status i / s without a chance -> 0;
                     u / n -> 0; no flag -> 1 (the v0.2 rule)
    cap_model_only   cap_fpl with the staleness relief (stale 'd' flags relax toward 1; ModelConfig.stale_*)
    cap_with_news    cap_model_only combined with the rule-parsed news (combine_news; ModelConfig.news_*)
The model hook fills p_play_model_only / p_play_with_news (GW+1 P(play) of the player's first fixture of the
next
GW: min(unconstrained model P(play), cap)) and p_play_unconstrained, so the registry can score both.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .rules import CONFIDENCE_ORDER

AVAILABILITY_COLUMNS = (
    "element",
    "web_name",
    "team",
    "status",
    "chance_of_playing_next_round",
    "news",
    "news_added",
    "status_changed_at",
    "status_change_source",
    "days_since_status_change",
    "days_since_news_added",
    "status_transitions_last_14d",
    "stale",
    "fpl_text_rule",
    "fpl_text_p_play",
    "fpl_text_confidence",
    "ext_source",
    "ext_status",
    "ext_rule",
    "ext_p_play",
    "ext_p_60_given_play",
    "ext_confidence",
    "ext_published_at",
    "ext_source_updated_at",
    "cap_fpl",
    "cap_model_only",
    "cap_with_news",
    "news_effect",
    "news_effect_source",
    "next_event",
    "p_play_unconstrained",
    "p_play_model_only",
    "p_play_with_news",
)


def fpl_cap(status: pd.Series, chance: pd.Series) -> np.ndarray:
    st = status.fillna("a").astype(str).to_numpy()
    ch = pd.to_numeric(chance, errors="coerce").to_numpy(float)
    ch = np.where(np.isnan(ch) & np.isin(st, ["i", "s"]), 0.0, ch)
    cap = np.where(np.isnan(ch), 1.0, np.clip(ch, 0, 100) / 100.0)
    return np.where(np.isin(st, ["u", "n"]), 0.0, cap)


def stale_relief(
    cap: np.ndarray, status: pd.Series, stale: pd.Series, days_news: pd.Series, cfg
) -> np.ndarray:
    """Stale 'd' flags: shortfall x max(stale_min_shortfall, 0.5 ** ((days - stale_after_days) /
    half_life))."""
    days = pd.to_numeric(days_news, errors="coerce").fillna(0.0).to_numpy(float)
    over = np.clip(days - cfg.stale_after_days, 0, None)
    mult = np.maximum(cfg.stale_min_shortfall, 0.5 ** (over / cfg.stale_half_life_days))
    apply = stale.fillna(False).astype(bool).to_numpy() & (status.astype(str).to_numpy() == "d")
    return np.where(apply, 1.0 - (1.0 - cap) * mult, cap)


def _ts(v: Any) -> pd.Timestamp:
    return pd.to_datetime(v, utc=True, errors="coerce")


def combine_news(
    cap_fpl: float,
    cap_model_only: float,
    stale: bool,
    news_added: Any,
    evidence: list[dict[str, Any]],
    cfg,
) -> tuple[float, str, str | None]:
    """(cap_with_news, effect "none" | "lower" | "raise", source of the deciding item).

    evidence: parsed items of this player: {source, p_play_next, confidence, published_at}.
    LOWER: any item with confidence >= cfg.news_min_confidence and an estimate p -> min(cap, p); an FPL-text
    item counts only when p < cap_fpl (it then says more than its own flag).
    RAISE (only if `stale`): the NEWEST qualifying external item, when newer than news_added and above the
    cap,
    sets cap + cfg.news_raise_weight * (p - cap) and overrides older lowering items."""
    need = CONFIDENCE_ORDER[cfg.news_min_confidence]
    usable = [
        e
        for e in evidence
        if e.get("p_play_next") is not None
        and not pd.isna(e.get("p_play_next"))
        and CONFIDENCE_ORDER.get(str(e.get("confidence")), 0) >= need
    ]
    cap, effect, src = cap_model_only, "none", None
    if stale:
        ext = [e for e in usable if e["source"] != "fpl_news" and pd.notna(_ts(e.get("published_at")))]
        if ext:
            newest = max(ext, key=lambda e: _ts(e["published_at"]))
            added = _ts(news_added)
            p = float(newest["p_play_next"])
            if pd.notna(added) and _ts(newest["published_at"]) > added and p > cap_model_only:
                return (
                    cap_model_only + cfg.news_raise_weight * (p - cap_model_only),
                    "raise",
                    newest["source"],
                )
    for e in usable:
        p = float(e["p_play_next"])
        if e["source"] == "fpl_news" and p >= cap_fpl - 1e-9:
            continue
        if p < cap - 1e-9:
            cap, effect, src = p, "lower", e["source"]
    return cap, effect, src


def build_availability(
    recency: pd.DataFrame,
    elements: list[dict[str, Any]],
    team_names: dict[int, str],
    parsed: pd.DataFrame,
    cfg,
    next_event: int | None,
) -> pd.DataFrame:
    """One row per element (AVAILABILITY_COLUMNS). parsed: matched news items with element, source, rule,
    p_play_next, p_60_given_play, confidence, status_hint, published_at, source_updated_at."""
    el = {int(e["id"]): e for e in elements}
    a = recency.copy()
    a["web_name"] = a["element"].map(lambda i: str(el[int(i)].get("web_name") or ""))
    a["team"] = a["element"].map(lambda i: team_names.get(int(el[int(i)].get("team") or 0)))
    a["cap_fpl"] = fpl_cap(a["status"], a["chance_of_playing_next_round"])
    a["cap_model_only"] = stale_relief(
        a["cap_fpl"].to_numpy(), a["status"], a["stale"], a["days_since_news_added"], cfg
    )
    by_el: dict[int, list[dict[str, Any]]] = {}
    if len(parsed):
        for r in parsed.to_dict("records"):
            by_el.setdefault(int(r["element"]), []).append(r)
    fpl_cols = {"fpl_text_rule": [], "fpl_text_p_play": [], "fpl_text_confidence": []}
    ext_cols: dict[str, list] = {k: [] for k in AVAILABILITY_COLUMNS if k.startswith("ext_")}
    caps, effects, effect_src = [], [], []
    for row in a.itertuples(index=False):
        ev = by_el.get(int(row.element), [])
        fpl = next((e for e in ev if e["source"] == "fpl_news"), None)
        fpl_cols["fpl_text_rule"].append(fpl["rule"] if fpl else None)
        fpl_cols["fpl_text_p_play"].append(fpl["p_play_next"] if fpl else np.nan)
        fpl_cols["fpl_text_confidence"].append(fpl["confidence"] if fpl else None)
        ext = [e for e in ev if e["source"] != "fpl_news"]
        newest = (
            max(
                ext,
                key=lambda e: (
                    _ts(e.get("published_at"))
                    if pd.notna(_ts(e.get("published_at")))
                    else pd.Timestamp.min.tz_localize("UTC")
                ),
            )
            if ext
            else None
        )
        ext_cols["ext_source"].append(newest["source"] if newest else None)
        ext_cols["ext_status"].append(newest.get("status_hint") if newest else None)
        ext_cols["ext_rule"].append(newest["rule"] if newest else None)
        ext_cols["ext_p_play"].append(newest["p_play_next"] if newest else np.nan)
        ext_cols["ext_p_60_given_play"].append(newest.get("p_60_given_play") if newest else np.nan)
        ext_cols["ext_confidence"].append(newest["confidence"] if newest else None)
        ext_cols["ext_published_at"].append(newest.get("published_at") if newest else None)
        ext_cols["ext_source_updated_at"].append(newest.get("source_updated_at") if newest else None)
        cap, effect, src = combine_news(
            float(row.cap_fpl), float(row.cap_model_only), bool(row.stale), row.news_added, ev, cfg
        )
        caps.append(cap)
        effects.append(effect)
        effect_src.append(src)
    for k, v in {**fpl_cols, **ext_cols}.items():
        a[k] = v
    for c in ("fpl_text_p_play", "ext_p_play", "ext_p_60_given_play"):
        a[c] = pd.to_numeric(a[c], errors="coerce").astype(float)
    for c in ("ext_published_at", "ext_source_updated_at"):
        a[c] = _ts(a[c])
    a["cap_with_news"] = np.asarray(caps, dtype=float)
    a["news_effect"] = effects
    a["news_effect_source"] = effect_src
    a["next_event"] = pd.array([next_event] * len(a), dtype="Int64")
    for c in ("p_play_unconstrained", "p_play_model_only", "p_play_with_news"):
        a[c] = np.nan
    return a.reindex(columns=list(AVAILABILITY_COLUMNS)).sort_values("element").reset_index(drop=True)
