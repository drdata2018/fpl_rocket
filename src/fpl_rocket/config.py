"""App-wide constants. The ONLY place the owner's team ID appears (as the input placeholder)."""

from __future__ import annotations

# Shown greyed-out in the Team ID box; never used as a default value or as a cache key.
PLACEHOLDER_TEAM_ID = "7630179"

APP_URL = "https://fpl-rocket.streamlit.app/"

SHARED_TTL_S = 600  # bootstrap, fixtures, live GWs, element summaries
TEAM_TTL_S = 300  # per-ID data: entry, history, picks, transfers, planning state
TEAM_CACHE_ENTRIES = 500  # bound on distinct IDs held in the in-process cache

MAX_TEAM_ID = 2**31 - 1
FETCH_WORKERS = 8  # process-wide cap on parallel FPL API calls (polite, bounded)

# Shared-data bundles published by .github/workflows/bundle.yml as GitHub Releases. The app reads them through
# the releases/.../download/ redirect paths (not api.github.com, so no 60/h unauthenticated API quota).
RELEASES_URL = "https://github.com/drdata2018/fpl_rocket/releases"
MANIFEST_TTL_S = 300  # how long a fetched manifest (or a failed fetch) is reused
BUNDLE_MAX_AGE_H = 12  # an older bundle (publication stalled) is ignored: live API fallback
BUNDLE_TIMEOUT_S = (5.0, 20.0)
