"""NewsItem: one piece of availability news about one player, from any source, and the Source protocol.

Every source returns the same shape so the matching and the (optional) LLM stage treat them alike. Web text is
UNTRUSTED data: it is parsed and stored, never executed or followed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

NEWS_ITEM_COLUMNS = (
    "source",
    "player_name_raw",
    "team_hint",
    "text",
    "status_hint",
    "chance_hint",
    "return_date_hint",
    "published_at",
    "source_updated_at",
    "fetched_at",
    "url",
    "element_hint",
)


@dataclass(frozen=True)
class NewsItem:
    """player_name_raw / team_hint as the source writes them; status_hint e.g. "Ruled Out" / "75%" / FPL's
    status letter; chance_hint 0-100 when the source states one; return_date_hint ISO date or None.
    published_at: when the information itself was produced (a dated quote, FPL news_added), ISO UTC or None;
    source_updated_at: when the source last updated the table/page as a whole (None if unknown);
    element_hint: the FPL element id when the source already knows it (the FPL API's own news)."""

    player_name_raw: str
    team_hint: str | None
    text: str
    status_hint: str | None
    chance_hint: float | None
    return_date_hint: str | None
    source: str
    fetched_at: str
    url: str
    published_at: str | None = None
    source_updated_at: str | None = None
    element_hint: int | None = field(default=None)

    def as_row(self) -> dict:
        d = asdict(self)
        return {c: d[c] for c in NEWS_ITEM_COLUMNS}


class Source(Protocol):
    """A news source: one polite fetch per build. fetch() never raises; on failure it returns [] and
    records why in `last_error` (the build writes it to the manifest)."""

    name: str
    last_error: str | None
    last_status: str  # "ok" | "empty" | "error" | "skipped"
    source_updated_at: str | None

    def fetch(self) -> list[NewsItem]: ...
