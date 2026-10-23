"""Source `fpl_news`: the FPL API's own per-player `news` text (bootstrap-static elements), in the NewsItem
shape.

No extra request: the bootstrap the build already fetched is passed in. The element id is known, so these
items
need no name matching (element_hint). published_at = FPL's news_added.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .items import NewsItem

SOURCE = "fpl_news"
URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


class FPLNewsSource:
    name = SOURCE

    def __init__(
        self, elements: list[dict[str, Any]], teams: list[dict[str, Any]], now: datetime | None = None
    ):
        self.elements = elements
        self.team_names = {int(t["id"]): str(t.get("name")) for t in teams}
        self.now = now or datetime.now(UTC)
        self.last_error: str | None = None
        self.last_status = "skipped"
        self.source_updated_at: str | None = None

    def fetch(self) -> list[NewsItem]:
        out = []
        for e in self.elements:
            text = str(e.get("news") or "").strip()
            if not text:
                continue
            chance = e.get("chance_of_playing_next_round")
            out.append(
                NewsItem(
                    player_name_raw=f"{e.get('first_name', '')} {e.get('second_name', '')}".strip(),
                    team_hint=self.team_names.get(int(e.get("team") or 0)),
                    text=text,
                    status_hint=str(e.get("status") or "") or None,
                    chance_hint=float(chance) if chance is not None else None,
                    return_date_hint=None,
                    source=SOURCE,
                    fetched_at=self.now.isoformat(timespec="seconds"),
                    url=URL,
                    published_at=_iso(e.get("news_added")),
                    element_hint=int(e["id"]),
                )
            )
        self.last_status = "ok" if out else "empty"
        return out


def _iso(v: Any) -> str | None:
    if not v:
        return None
    try:
        return (
            datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            .astimezone(UTC)
            .isoformat(timespec="seconds")
        )
    except ValueError:
        return None
