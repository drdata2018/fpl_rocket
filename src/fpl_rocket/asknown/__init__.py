"""AS-KNOWN data archive (PLAN_v3 step 1a): every input a replay may use, stored as it was known at the issue
time T = deadline - 2 h of each gameweek. One module per source (bootstrap, fixtures, odds, minutes, crowd,
uefa), shared plumbing in `common`, and one reader: `load(season, gw)` (fpl_rocket.asknown.reader).
Built by scripts/build_asknown.py into data/asknown/ (gitignored); raw downloads in data/raw/<source>/.
"""

from __future__ import annotations

from .reader import AsKnown, corrupt_after, load

__all__ = ["AsKnown", "corrupt_after", "load"]
