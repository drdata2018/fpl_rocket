"""Blind replay engine (docs/PLAN_v3_blind_follow.md, Phase 1b-1c): a deterministic season loop under the
contract - every package is decided from the inputs known at the issue time T = deadline - 2 h only
(fpl_rocket.asknown + per-GW points rows of matches finished by T), validated against that season's rules
(fpl_rocket.rules), executed, then scored with the actual results.

Modules:
    inputs    the inputs at T (as-known archive + points history with known_at = kickoff + 3 h), the actual
              results used for scoring, and the corruption helper of the leakage test
    forecast  predictions at T (frozen form baseline, v0.2.1 model with the as-known availability cap)
    engine    manager state, the decision package, legality checks (illegal packages are rejected loudly),
              execution and scoring (auto-subs, vice-captain, chips, hits)
    chips     the prespecified chip rule shared by every policy (triggers + "last eligible GW")
    policies  MINIMAL_MODEL, FORM_RESTRAINED, HOLD, LAGGED_CROWD
    run       the season loop, the decision log (JSON lines) and the built-in leakage test
    report    season tables, paired block-bootstrap comparisons, within-sample manager percentiles
"""
