"""The bundle's news stage: status_history, availability, news_items, news_unmatched (+ manifest fields).

Runs inside write_bundle (scripts/build_bundle.py --news, on by default) before the model hook, which reads
`availability` for the live P(play) cap and fills its p_play_* columns. A failure is recorded as news_error
and
never blocks the data bundle. No paid service of any kind: the external source is a public page, the text is
read
by the rule table in fpl_rocket.news.rules.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .availability import build_availability
from .fpl_news import FPLNewsSource
from .history import recency_features, update_status_history
from .items import NEWS_ITEM_COLUMNS, NewsItem, Source
from .matching import Matcher
from .premier_injuries import PremierInjuriesSource
from .rules import parse_item, rules_version

log = logging.getLogger(__name__)

NEWS_TABLES = ("status_history", "availability", "news_items", "news_unmatched")
PARSED_COLUMNS = (
    "element",
    "match_method",
    "rule",
    "p_play_next",
    "p_60_given_play",
    "confidence",
    "return_date",
)


def _records(t: Any) -> list[dict[str, Any]]:
    if isinstance(t, pd.DataFrame):
        return t.to_dict("records")
    from ..bundle import table_records

    return table_records(t)


def next_deadline(events: list[dict[str, Any]]) -> tuple[int | None, datetime | None]:
    nxt = next((e for e in events if e.get("is_next")), None)
    if nxt is None or not nxt.get("deadline_time"):
        return None, None
    return int(nxt["id"]), datetime.fromisoformat(str(nxt["deadline_time"]).replace("Z", "+00:00"))


def items_frame(items: list[NewsItem], extra: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [{**it.as_row(), **ex} for it, ex in zip(items, extra, strict=True)]
    cols = [*NEWS_ITEM_COLUMNS, *PARSED_COLUMNS, "unmatched_reason"]
    df = pd.DataFrame(rows, columns=cols)
    df["element"] = pd.array(df["element"], dtype="Int64")
    df["element_hint"] = pd.array(df["element_hint"], dtype="Int64")
    for c in ("chance_hint", "p_play_next", "p_60_given_play"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in cols:
        if df[c].dtype == object:
            df[c] = df[c].astype("string")
    return df


def run_news_stage(
    tables: dict[str, Any],
    previous: Any,
    now: datetime,
    cfg: Any = None,
    sources: Sequence[Source] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """tables: the bundle tables (players, teams, events); previous: bundle.PreviousRelease or None (its
    `extra["status_history"]` and manifest created_at are carried forward)."""
    if cfg is None:
        from ..model.config import DEFAULT_CONFIG

        cfg = DEFAULT_CONFIG
    now = now.astimezone(UTC)
    elements, teams, events = (_records(tables[k]) for k in ("players", "teams", "events"))
    team_names = {int(t["id"]): str(t.get("name")) for t in teams}
    next_event, deadline = next_deadline(events)

    prev_hist = (
        (getattr(previous, "extra", None) or {}).get("status_history") if previous is not None else None
    )
    prev_at = None
    if previous is not None and previous.manifest.get("created_at"):
        prev_at = datetime.fromisoformat(str(previous.manifest["created_at"]).replace("Z", "+00:00"))
    n_before = 0 if prev_hist is None else len(prev_hist)
    history = update_status_history(prev_hist, elements, now, prev_at)
    recency = recency_features(history, elements, now, cfg)

    external = list(sources) if sources is not None else [PremierInjuriesSource(now=lambda: now)]
    all_sources: list[Source] = [FPLNewsSource(elements, teams, now), *external]
    items: list[NewsItem] = []
    fetched = []
    for src in all_sources:
        got = src.fetch()
        items.extend(got)
        fetched.append(
            {
                "source": src.name,
                "status": src.last_status,
                "items": len(got),
                "error": src.last_error,
                "source_updated_at": src.source_updated_at,
            }
        )

    matcher = Matcher(elements, teams)
    extra = []
    for it in items:
        m = matcher.match(it.player_name_raw, it.team_hint, it.element_hint)
        p = parse_item(it, next_deadline=deadline, now=now, table_fresh_days=cfg.news_table_fresh_days)
        extra.append(
            {
                "element": m.element,
                "match_method": m.method,
                "rule": p.rule,
                "p_play_next": p.p_play_next,
                "p_60_given_play": p.p_60_given_play,
                "confidence": p.confidence,
                "return_date": p.return_date,
                "unmatched_reason": m.reason,
            }
        )
    news = items_frame(items, extra)
    matched = news[news["element"].notna()].reset_index(drop=True)
    unmatched = news[news["element"].isna()].reset_index(drop=True)
    parsed = matched.astype(object).where(matched.notna(), None) if len(matched) else matched
    availability = build_availability(recency, elements, team_names, parsed, cfg, next_event)

    fpl_items = news[news["source"] == "fpl_news"]
    coverage = float((fpl_items["rule"] != "unmatched").mean()) if len(fpl_items) else None
    by_source = {
        s: {
            "matched": int(((news["source"] == s) & news["element"].notna()).sum()),
            "unmatched": int(((news["source"] == s) & news["element"].isna()).sum()),
        }
        for s in sorted(set(news["source"].dropna()))
    }
    manifest = {
        "news_sources_fetched": fetched,
        "news_items_matched": int(len(matched)),
        "news_items_unmatched": int(len(unmatched)),
        "news_items_by_source": by_source,
        "news_rules_version": rules_version(),
        "news_fpl_text_rule_coverage": None if coverage is None else round(coverage, 4),
        "news_effects": availability["news_effect"].value_counts().to_dict(),
        "status_history_rows": int(len(history)),
        "status_history_appended": int(len(history) - n_before),
        "status_history_previous_rows": n_before,
        "availability_stale_players": int(availability["stale"].sum()),
    }
    tables_out = {
        "status_history": history,
        "availability": availability,
        "news_items": matched,
        "news_unmatched": unmatched,
    }
    return tables_out, manifest


def news_fn(cfg: Any = None, sources_factory: Callable[[datetime], Sequence[Source]] | None = None):
    """The bundle's news hook: (tables, previous, now) -> (tables, manifest fields)."""

    def run(tables, previous, now):
        sources = sources_factory(now) if sources_factory is not None else None
        return run_news_stage(tables, previous, now, cfg, sources)

    return run
