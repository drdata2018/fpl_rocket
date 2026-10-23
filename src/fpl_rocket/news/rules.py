"""Rule-based reading of availability news text (no paid service, no LLM). The rule table itself,
NEWS_TEXT_RULES, lives in fpl_rocket.model.config next to the cap constants that use it.

Each item's text is matched against NEWS_TEXT_RULES IN ORDER; the first rule that matches decides
(p_play_next, p_60_given_play, confidence). Text that no rule matches gets rule "unmatched", confidence "low"
and
NO adjustment. Only rules with confidence >= ModelConfig.news_min_confidence ("medium") ever move the live
P(play)
cap (fpl_rocket.news.availability). p_60_given_play is RECORDED only (availability.parquet); it does not
enter the
minutes model yet.

p_play_next values:
    float        a fixed probability
    "pct"        the captured percentage / 100 ("75% chance of playing", the injury table's "Status: 75%")
    "date"       a return date ("Expected back 1 Nov", "Suspended until 31 Oct"): 0.0 when the date is after
                 the
                 next deadline, otherwise the rule says the player should be back in time -> no estimate
                 (None, confidence low: the text cannot tell how likely he plays)
    None         no estimate (the rule only classifies the text)

The injury table's rows are rendered as "<Reason>. <Condition>. Status: <status>. Potential return: <date>.
<quote>", so its structured Status is read by the "table_*" rules before any free-text rule can see the quote.
Injury-table rows older than ModelConfig.news_table_fresh_days (table "Last update") are downgraded to
confidence low, and so is a "Ruled Out" row whose potential return date is not after the next deadline (the
row
is probably out of date).

All constants are PROSPECTIVE-ONLY: no historical news snapshots exist, so none of this can be backtested; the
values are conservative round numbers (an injured player who is declared fit is often managed, hence
p_60_given_play < 1).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime

from ..model.config import NEWS_TEXT_RULES
from .items import NewsItem

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}

_COMPILED = tuple((n, re.compile(rx), p, p60, c) for n, rx, p, p60, c in NEWS_TEXT_RULES)
MONTH_NAMES = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
MONTHS = {m: i for i, m in enumerate(MONTH_NAMES, start=1)}


def rules_version() -> str:
    """Short hash of the rule table (recorded in the manifest: a rule change is visible in every bundle)."""
    return hashlib.sha256(repr(NEWS_TEXT_RULES).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class ParsedNews:
    rule: str
    p_play_next: float | None
    p_60_given_play: float | None
    confidence: str
    return_date: str | None = None


def _date_from(text: str, ref: datetime) -> date | None:
    """'1 Nov' -> a date near `ref` (a day-month more than 60 days before ref belongs to the next year)."""
    m = re.match(r"(\d{1,2}) ([a-z]{3})", text)
    if not m or m.group(2) not in MONTHS:
        return None
    try:
        d = date(ref.year, MONTHS[m.group(2)], int(m.group(1)))
    except ValueError:
        return None
    if (ref.date() - d).days > 60:
        d = d.replace(year=ref.year + 1)
    return d


def parse_text(
    text: str,
    *,
    next_deadline: datetime | None,
    now: datetime,
    published_at: datetime | None = None,
    return_date_hint: str | None = None,
) -> ParsedNews:
    """First matching rule of NEWS_TEXT_RULES -> ParsedNews; no match -> rule "unmatched", confidence low."""
    t = " ".join((text or "").lower().split())
    for name, rx, p, p60, conf in _COMPILED:
        m = rx.search(t)
        if not m:
            continue
        ret = None
        if p == "pct":
            value: float | None = min(int(m.group(1)), 100) / 100.0
        elif p == "date":
            d = _date_from(m.group(1), published_at or now)
            ret = d.isoformat() if d else None
            if d is None or next_deadline is None:
                value, conf = None, "low"
            elif d > next_deadline.astimezone(UTC).date():
                value = 0.0
            else:  # back before the next deadline: the text gives no probability
                value, conf = None, "low"
        else:
            value = p  # type: ignore[assignment]
        return ParsedNews(name, value, p60, conf, ret or return_date_hint)
    return ParsedNews("unmatched", None, None, "low", return_date_hint)


def _ts(v: str | None) -> datetime | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def parse_item(
    item: NewsItem, *, next_deadline: datetime | None, now: datetime, table_fresh_days: float
) -> ParsedNews:
    """parse_text plus the injury-table freshness downgrades (see the module docstring)."""
    parsed = parse_text(
        item.text,
        next_deadline=next_deadline,
        now=now,
        published_at=_ts(item.published_at),
        return_date_hint=item.return_date_hint,
    )
    if item.source == "premier_injuries":
        stamp = _ts(item.source_updated_at)
        old = stamp is None or (now - stamp).total_seconds() > table_fresh_days * 86400
        ret = date.fromisoformat(item.return_date_hint) if item.return_date_hint else None
        stale_out = (
            parsed.rule == "table_ruled_out"
            and ret is not None
            and next_deadline is not None
            and ret <= next_deadline.astimezone(UTC).date()
        )
        if old or stale_out:
            parsed = ParsedNews(
                parsed.rule, parsed.p_play_next, parsed.p_60_given_play, "low", parsed.return_date
            )
    return parsed
