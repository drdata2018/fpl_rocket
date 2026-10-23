# FPL Rocket

Fantasy Premier League analytics and transfer planning for **any** team ID, built only on free data.
Live app: https://fpl-rocket.streamlit.app · Spec: [docs/PLAN_v2.md](docs/PLAN_v2.md).

## Using the app

Type any FPL team ID in the sidebar and press **Load** (the ID is the number in the URL of your Points page).
Pages that need a team say so; everything else works without one. The ID lives in your browser session only:
it is never written to the URL unless you open **Copy shareable link**, and nothing about visitors is stored.

| Page | What it shows |
|---|---|
| **Season Overview** | Premier League table, season leaders, last-gameweek review, price changes since the last deadline, availability and news, template squad by ownership, fixture grid (FPL difficulty). |
| **My Team** | Rank and points per gameweek, value, bank and chips, the squad on a pitch, captain hindsight, transfer ledger, classic leagues, and the reconstructed planning state (squad, purchase and selling prices, bank, free-transfer range). |
| **Compare Players** | 2-4 players side by side: season metrics, points per gameweek, the next 5 fixtures; plus the experimental model rows below. |
| **Differentials** | Players under an ownership threshold (position, price and club filters) with form, minutes and the next 5 fixtures; plus the experimental model columns below. |
| **Model & Planner** | Your planning state with confirm / edit (session-only corrections), how the planner is being tested, the four virtual managers (prospective, from GW9) and the prospective registry's latest scores. No plan is shown before its deadline has passed (shadow mode). |

**Experimental (shown, with caveats).** From xP model v0.2.1 (`v0.2.1-<git sha>`), in Compare Players and
Differentials only:

- *xP next GW* - mean expected points for the next gameweek.
- *xP next 3 GWs (sum over listed fixtures)* - the sum over the player's fixtures in the next three
  gameweeks: a blank gameweek adds 0, a double gameweek counts both fixtures. Nothing longer is shown.
- *P(play)* - a band, not a percentage: **nailed** (>= 90%), **likely** (70-90%), **rotation risk**
  (40-70%), **doubtful** (< 40%). The model's estimate; this season's calibration is still being measured.
- Newcomers with too little history show **insufficient history** instead of a number.
- Every table with model numbers carries the model version and the bundle id it came from.

The model was validated by a retrospective backtest only; its prospective record starts at the GW9 deadline.
It is not advice.

**Shadow (computed, recorded, not shown).** The planner (a legal integer program: squad, XI, captain,
transfers, free-transfer banking, hits, chips) runs in shadow mode. Before every deadline its proposals are
recorded for a fixed roster of five public teams and scored after the gameweek; none is shown to users until
that record shows whether they help.

**How the registry scores.** The bundle built 30-90 minutes before each deadline is kept forever as the
scored snapshot (`deadline-gw<N>`), with the xP table as issued and the planner's proposals for the roster:
(a) best XI, captain and vice without transfers, (b) the next gameweek with up to 2 transfers (4-point hits
allowed), (c) a 5-gameweek path (decay 0.85, free-transfer banking). After the gameweek is finished and
checked, a daily job scores it against the real points: xP RMSE and rank correlation against a frozen
baseline (form per 90 x availability); per team, captain points vs the best captain in the squad and the
manager's own, and the planned line-up's points (automatic substitutions, armband, net of hits for (b) and
(c)) vs the baseline and the manager's actual score. A deadline without a snapshot is recorded as
**missing**, never skipped. Scores live in `registry/registry_scores.csv` on the `spike-log` branch; the Model
& Planner page shows the latest.

**Four virtual managers (prospective policy trajectories, from GW9).** The decisive test of the planner: four
virtual FPL managers start from the SAME squad and play every gameweek without chips, each choice recorded
before its deadline. *Model, aggressive* (planner objective (c): 3-GW horizon, decay 0.85, up to 2 transfers,
hits when the objective says so), *Model, restrained* (same xP under the restrained rule
`trajectories.RESTRAINED_RULE`: transfers in a GW <= the free transfers available entering it, banked by the
rules engine at +1 per GW up to the cap of 5, and never a hit; so it may bank a transfer when nothing is worth
doing and spend two or more later; no minimum-gain threshold: a free transfer is used whenever the 3-GW
objective gains anything, a known simplification; no chips, so the post-wildcard carry rule never applies),
*Form, restrained* (the restrained rule on the frozen form-per-90 baseline) and *Hold* (never transfers; best
XI + captain by the baseline). Until step 3h the restrained rule was "<= 1 transfer per GW"
(`run_matched.py --restrained-rule one_per_gw` replays it). Start (`trajectories.START_RULE`, favours
neither predictor): the most-owned legal 15 under 100.0 at the GW9 deadline prices - players ranked by
`selected_by_percent` (ties: cheaper, then lower id) and taken greedily while the position has an open slot
(2/5/5/3), the club has fewer than 3 picks and the money left still pays for the cheapest legal completion;
selectable players only; bank = 100.0 - cost; 1 free transfer entering GW10. Each pre-deadline run reads the
previous state from the latest earlier `deadline-gw<k>` release (never a `bundle-*` one; a read error other
than "not found" aborts instead of restarting) and writes `trajectories_state.json` (squads with purchase
prices, bank, free transfers, transfer log, line-ups, provisional points) and `trajectories_plans.json` (XI,
captain, vice, bench order, transfers of GW N) into `deadline-gw<N>`. A GW without them is recorded as
**missing**: squad and line-up carried forward (free transfers accrue), still scored with automatic
substitutions. The daily scorer appends realised points (automatic substitutions, armband, hits), cumulative
totals and the official GW average (`events.average_entry_score`) to `registry/trajectories_scores.csv` on
`spike-log`; missing / solver_error / fallback weeks are flagged. The Model & Planner page shows the
cumulative chart and table, and a GW's line-ups only after it has been scored.

**Zero-cost rule.** Free public data only (the official FPL API, the public injury table, the community
archive for backtests). No paid services, no API keys, no logins, no cookies; hosting is the Streamlit
Community Cloud free tier, data jobs run on GitHub Actions and are published as GitHub Releases.

## Operator notes

- **Deploy:** pushing `main` redeploys https://fpl-rocket.streamlit.app. Streamlit Community Cloud installs
  `requirements.txt` only (runtime: streamlit, requests, pandas, pyarrow, pulp, highspy; no model-fitting
  libraries) and runs `streamlit_app.py`. The planner import is guarded: if the solver cannot be installed,
  every other page still renders and Model & Planner shows the state and the registry.
- **Secrets: none needed now.** The workflows use only the built-in `GITHUB_TOKEN`; the app reads public
  release assets and the public `spike-log` branch.
- **First release (once):** on GitHub open the repo -> **Actions** -> **bundle** -> **Run workflow** ->
  branch `main`. After 2-3 minutes **Releases** shows `Bundle <id>` marked *Latest*; the app switches to it
  within 5 minutes. Until then every page works on the live API with a fallback notice. Later runs are
  hourly and automatic.
- **Switches** (code constants, changed by a commit): `SHOW_SHADOW_PLANS` in `src/fpl_rocket/config.py`
  (False = no plan is shown; True re-enables the plan UI as built), `REGISTRY_START_GW = 9` in
  `src/fpl_rocket/registry.py`, `BUNDLE_N_SIMS = 4000` in `src/fpl_rocket/model/config.py`, solver time
  limits in `src/fpl_rocket/planner/core.py` (`TIME_LIMIT_S = 10` for app solves, `BATCH_TIME_LIMIT_S = 60`
  for `registry_write.py` and `run_matched.py`; both scripts take `--time-limit`).
- **Solver health:** HiGHS keeps one process-wide scheduler, so the planner resets it before every run and
  runs one solve at a time. A solver that fails to run gives a plan with status `solver_error` (never a
  silent hold); the registry meta lists `plans_hit_time_limit` and `plans_solver_error`, and a solver
  error scores as `status=error`.
- The day-1 hosting probe page (`app_pages/hosting_spike.py`) is kept but not in the navigation.

## Code map

- `src/fpl_rocket/api.py` - FPL API client (browser UA, retries/backoff, optional raw cache).
- `src/fpl_rocket/rules/` - versioned game rules (`season_2026_27.py`): scoring, transfers, selling price.
- `src/fpl_rocket/state.py` - planning-state reconstruction (squad, purchase/selling prices, bank, free-transfer range, chips).
- `src/fpl_rocket/bundle.py` - immutable parquet bundle + `manifest.json` with checksums (element_history,
  round_prices, deadline flags, optional model and news tables).
- `src/fpl_rocket/data.py` - cached data access: shared (latest bundle release, else live API; ttl 600 s) and
  per-ID (ttl 300 s) layers; `APP_ASSETS` = the only bundle files the app downloads (never `xp_sims`).
- `src/fpl_rocket/archive.py` - normaliser for the community archive (vaastav) to one stable schema.
- `src/fpl_rocket/backtest.py` - rolling-origin backtest harness, Predictor protocol, baseline, metrics.
- `src/fpl_rocket/model/` - xP model v0.2.1: `minutes.py` (P(0/1-59/60+) + minutes per bin),
  `team_goals.py` (Dixon-Coles), `rates.py` (xG/xA shares, per-90 rates), `bonus.py`, `newcomers.py`,
  `simulate.py` (joint Monte Carlo scored by the rules engine), `pipeline.py` (fit/predict, bundle hook,
  backtest predictor), `decisions.py`, `evaluate.py`, `config.py` (a-priori hyperparameters). Needs the
  `model` dependency group (scikit-learn, scipy); NOT in `requirements.txt` - the app never fits models.
- `src/fpl_rocket/news/` - availability news for the live P(play) cap: FPL flag recency (`status_history`),
  the public injury table, the FPL API's own news text, name matching, rule-based text parsing.
- `src/fpl_rocket/planner/` - the legal planner (PuLP + HiGHS): `core.py` (objectives (a)-(c), chips),
  `inputs.py` (planning state -> planner input, session edits), `lineup.py` (best XI, bench, auto subs,
  realised points).
- `src/fpl_rocket/registry.py` - prospective registry: writer (deadline snapshot), reader, scorer.
- `src/fpl_rocket/trajectories.py` - the four virtual managers: start rule, deadline step, carry-forward,
  missing weeks, scoring, app summary.
- `src/fpl_rocket/matched.py` - matched-policy backtest: aggressive (MODEL vs BASELINE planner vs HOLD) or
  restrained (MODEL_RESTRAINED vs FORM_RESTRAINED vs HOLD) policy sets, start rules, fixtures as known,
  per-season prediction cache, paired block bootstrap and sign test.
- `src/fpl_rocket/fixtures_known.py` - deadline-known fixtures from the community archive's git history.
- `src/fpl_rocket/views.py` - pure table builders for the pages (offline-testable).
- `src/fpl_rocket/ui.py` - sidebar (team ID), per-feature error guard, formatting, pitch, model captions.
- `src/fpl_rocket/config.py` - placeholder team ID, cache settings, `SHOW_SHADOW_PLANS`.
- `streamlit_app.py` - entry point: navigation + sidebar; pages in `app_pages/`.
- `scripts/` - `build_bundle.py`, `registry_write.py`, `trajectories_write.py`, `registry_score.py`,
  `reconcile_points.py <gw>`, `state_summary.py <ids>`, `acceptance.py`, `record_app_snapshot.py`,
  `load_archive.py`, `run_backtest.py`, `run_matched.py`, `build_fixtures_as_known.py`,
  `run_decision_diagnostics.py`, `spike_probe.py`.

## Run

```bash
uv sync                                  # Python 3.11 env with dev + model groups (default-groups)
uv run streamlit run streamlit_app.py    # the app
uv run pytest -q                         # tests (offline, replay recorded fixtures)
uv run ruff check                        # lint
uv run python scripts/reconcile_points.py 7   # rules vs API total_points for GW7 (expects 0 mismatches)
uv run python scripts/build_bundle.py --out data/bundles --with-model   # one bundle locally (model + news)
```

## Shared-data bundles (GitHub Releases)

`.github/workflows/bundle.yml` runs every hour at :40 (and on demand). It builds one immutable bundle
(bootstrap tables, fixtures with results, live stats of every finished GW, `element_history` for every player,
`round_prices`, and the model tables `xp`, `xp_sims`, `xp_fixtures`) and publishes it as release
`bundle-<bundle_id>`, marked *latest*.

- **Polite API load:** the 668 element-summary calls are made only when the previous release's
  `element_history` is older than 6 h, a GW has finished since it was fetched, or there is no previous release;
  otherwise it is reused and `round_prices` is rebuilt from it plus the bootstrap `now_cost`. The manifest says
  `element_history_source: fetched | reused:<bundle_id>` (the bundle that fetched it), with
  `element_history_fetched_at` and `element_history_reason`.
- **Scored snapshot:** the bundle created in the window 30-90 min before the deadline (manifest
  `pre_deadline`) is also published as `deadline-gw<N>`, kept forever. If GitHub cron delay leaves no bundle in
  that window, the week is recorded as missing.
- **Retention:** `bundle-*` releases older than 48 h are deleted; `deadline-*` never.
- **Model:** the job restores the archived previous season (actions/cache, `load_archive.py` on a miss) and
  runs `build_bundle.py --with-model`: `xp.parquet` (per player and GW for the next 6 GWs: mean xP, p20/p50/p80,
  p_play, p_60, p_cs, expected goals/assists/bonus/minutes), `xp_sims.parquet` (4000 joint simulations per
  player and GW, int8 points), `xp_fixtures.parquet` (per team-fixture expected goals for/against, P(CS),
  goals-conceded distribution = the model fixture-difficulty grid). Manifest: `model_version`
  (`v0.2.1-<git sha>`), `model_n_sims`, `model_fitted_on`, `model_features`, `model_hyperparameters`,
  `model_runtime_s`. A model failure is recorded as `model_error`; the data bundle still publishes. The app
  reads `xp.parquet` only (next GW, next-3-GW sums, P(play) bands); `xp_sims` and `xp_fixtures` are never
  downloaded by the app.
- **Registry:** for the pre-deadline bundle only, after `bundle-<id>` is published, `registry_write.py` adds
  `registry_xp.parquet`, `registry_plans.json` (objectives (a), (b), (c) per roster team) and
  `registry_meta.json` (model version, N, rules version, exclusions) to `deadline-gw<N>`. The step has
  `continue-on-error` and its own timeout: a failure never blocks either release, the week scores as missing.
  Before it, `trajectories_write.py` (its own `continue-on-error` step, 10 min) adds the four virtual managers'
  `trajectories_state.json` / `trajectories_plans.json` (from GW9 on).
  `.github/workflows/registry_score.yml` scores finished GWs daily at 06:00 UTC (registry and virtual managers;
  `registry_score.py --trajectories-csv`).

The app reads `https://github.com/drdata2018/fpl_rocket/releases/latest/download/manifest.json` (cached 300 s),
then the files of that bundle from its own tag, checks every sha256 and serves all shared data from that one
bundle (cached 600 s). The sidebar shows `Shared data as of <created_at> · bundle <bundle_id>`.
If there is no release yet, a checksum fails, the download times out or the bundle is older than 12 h, the app
uses the live FPL API as before and the sidebar shows **live API fallback** with the reason.

## Availability news (live cap, prospective-only)

Every bundle build (`build_bundle.py --news`, on by default; `--no-news` turns it off) adds four tables; no
API key or paid service is involved:

- `status_history` - FPL `status` / `chance_of_playing_next_round` / `news` / `news_added` changes between
  consecutive bundles, carried forward from the previous release (FPL keeps no history of these flags).
- `news_items` / `news_unmatched` - news per player from the FPL API's `news` text and the public injury table
  at premierinjuries.com (one fetch per build, 10 s timeout; on any failure the source is empty and the
  manifest says why). Names are matched accent- and case-insensitively within the player's club; anything not
  matched to exactly one player is kept in `news_unmatched`, never guessed.
- `availability` - per player: days since the last status change and since `news_added`, status transitions
  in the last 14 days, a `stale` flag (d / i with news older than 7 days and no change since), the rule that
  read the news (`NEWS_TEXT_RULES` in `model/config.py`), the GW+1 caps `cap_fpl` / `cap_model_only` /
  `cap_with_news`, and with the model `p_play_model_only` / `p_play_with_news`, so the registry can score both.

Rules (model v0.2.1, constants in `ModelConfig`, all prospective-only - nothing here can be backtested): a stale
`d` flag relaxes toward the model's own P(play) (half the shortfall at most); news with confidence medium or
better can lower the cap; it can raise it only for a stale flag, from a source newer than `news_added`, and
only half-way. The decay of the cap now counts GWs after the GW the FPL flag refers to.
Manifest: `news_sources_fetched`, `news_items_matched` / `news_items_unmatched`, `news_rules_version`,
`news_fpl_text_rule_coverage`, `news_effects`, `status_history_*`, `availability_stale_players`.
Known fragility: premierinjuries.com answers non-browser clients with a Cloudflare challenge (HTTP 403), so the
injury table is usually empty in automated builds; `--injury-html <saved page>` reads a saved copy instead.

## Archive and backtest

```bash
uv run python -I scripts/load_archive.py              # downloads 2022-23 .. 2026-27 to data/archive/ (gitignored)
uv run python -I scripts/run_backtest.py --predictor baseline --seasons 2024-25 2025-26
```

`load_archive.py` saves `merged_gw.csv`, `players_raw.csv`, `teams.csv`, `fixtures.csv` per season with a
manifest (url, sha256), parses them with pandas only (untrusted data, nothing executed) and writes
`data/archive/all_gws.parquet` (one row per season, GW, player, fixture; `code` = cross-season player id).
Null by season: `defensive_contribution`, `clearances_blocks_interceptions`, `recoveries`, `tackles` before
2025-26; expected goals/assists and `starts` exist for every loaded season (2022-23 on).

```bash
uv run python -I scripts/run_matched.py --seasons 2024-25 2025-26 --policies restrained --start most_owned \
    --pred-cache data/backtests/pred_cache --workers 16 --resamples 5000
uv run python -I scripts/build_fixtures_as_known.py --repo data/vaastav_meta --seasons 2024-25 2025-26
uv run python -I scripts/run_matched.py --seasons 2024-25 2025-26 --fixtures-as-known data/archive/fixtures_as_known
```

`run_matched.py` replays planner policies from one start squad (RETROSPECTIVE, matched): `--policies
aggressive` (MODEL vs BASELINE planner, <= 2 transfers, hits) or `restrained` (the virtual managers' rule;
`--restrained-rule free_only` (default: transfers <= the free transfers banked, cap 5, never a hit) or
`one_per_gw` (the step 3g rule, <= 1 transfer per GW, never a hit));
`--start baseline_xp | most_owned`; `--fixtures-as-known` plans with the schedule as known at each deadline
(snapshots from the community archive repository's git history, `build_fixtures_as_known.py`, which also
writes a coverage report); scoring always uses actual results.

`run_backtest.py` writes `data/backtests/<timestamp>_<predictor>.csv` (+ calibration and summary) and prints
the summary. It is a RETROSPECTIVE track (end-of-GW archive data). The decision metric (best legal XI +
captain by prediction, no squad/budget/transfers, minus the GW average) is an UNMATCHED comparison; the GW
average is official for 2026-27 and a proxy (15 x ownership-weighted mean points) for older seasons.

```bash
uv run python -I scripts/run_backtest.py --predictor model --seasons 2024-25 2025-26 2026-27
```

`--predictor model` also runs the baseline on the same folds and writes `<ts>_model_summary.md` (point
metrics by season and position, xP calibration, [p20, p80] coverage, minutes log-loss and calibration incl.
rotation risks, Dixon-Coles vs naive goals log-loss, runtime per fit). The retrospective backtest runs WITHOUT
the live-only minutes inputs (status, chance_of_playing_next_round): no historical snapshots exist.

Baseline "form per 90 x availability" (the bar the Phase B model must beat):

| season | RMSE | MAE | within-GW Spearman (all / played) | best XI + C minus GW avg (mean, sd) |
|---|---|---|---|---|
| 2024-25 | 2.087 | 1.041 | 0.696 / 0.319 | +4.7, 13.2 |
| 2025-26 | 2.121 | 1.043 | 0.719 / 0.276 | -5.1, 12.2 |

xP model v0.1 (retrospective, hyperparameters fixed a priori; 2026-27 = GW1-6, official averages):

| season | RMSE | MAE | within-GW Spearman (all / played) | best XI + C minus GW avg (mean, sd) |
|---|---|---|---|---|
| 2024-25 | 1.926 | 0.989 | 0.704 / 0.389 | +12.5, 14.0 |
| 2025-26 | 1.930 | 0.972 | 0.714 / 0.353 | +5.8, 13.1 |
| 2026-27 | 2.197 | 1.223 | 0.631 / 0.354 | -3.0, 19.0 |
