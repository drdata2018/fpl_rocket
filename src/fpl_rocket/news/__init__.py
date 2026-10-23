"""Availability news for the live minutes cap (step 3d): FPL flag recency (status_history), the public injury
table, the FPL API's own news text, conservative name matching, and a rule-based text parser. No paid service,
no LLM. Everything here is PROSPECTIVE-ONLY (no historical snapshots exist to backtest it).

    items.py             NewsItem + Source protocol
    premier_injuries.py  source: premierinjuries.com injury table (one polite fetch per build)
    fpl_news.py          source: bootstrap `news` text
    matching.py          accent / case-insensitive name + club matching; unmatched items are kept, never
                         guessed
    rules.py             NEWS_TEXT_RULES: the documented regex table -> (p_play_next, p_60_given_play,
                         confidence)
    history.py           status_history diffing between bundles, recency features, staleness
    availability.py      GW+1 caps (FPL only, with staleness relief, with news) -> availability.parquet
    stage.py             the bundle hook tying it together
"""

from .items import NewsItem, Source
from .stage import NEWS_TABLES, news_fn, run_news_stage

__all__ = ["NEWS_TABLES", "NewsItem", "Source", "news_fn", "run_news_stage"]
