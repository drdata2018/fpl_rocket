# FPL Rocket

Fantasy Premier League analytics and transfer planning for **any** team ID, built only on free data
(the public FPL API). Spec: [docs/PLAN_v2.md](docs/PLAN_v2.md).

Current stage: Phase A pages (Season Overview, My Team, Compare Players, Differentials v0, Model & Planner
placeholder) on shared-data bundles with a live-API fallback; Phase B step 1 (bundles v2, archive, backtest
harness + baseline); Phase B step 2 (xP model in the hourly bundle + retrospective backtest; the app does not
show xP yet). Live app: https://fpl-rocket.streamlit.app

- `src/fpl_rocket/api.py` - FPL API client (browser UA, retries/backoff, optional raw cache).
- `src/fpl_rocket/rules/` - versioned game rules (`season_2026_27.py`): scoring, transfers, selling price.
- `src/fpl_rocket/state.py` - planning-state reconstruction (squad, purchase/selling prices, bank, free-transfer range, chips).
- `src/fpl_rocket/bundle.py` - immutable parquet bundle + `manifest.json` with checksums (v2 adds element_history,
  round_prices, deadline flags).
- `src/fpl_rocket/data.py` - cached data access: shared (latest bundle release, else live API; ttl 600 s) and
  per-ID (ttl 300 s) layers.
- `src/fpl_rocket/archive.py` - normaliser for the community archive (vaastav) to one stable schema.
- `src/fpl_rocket/backtest.py` - rolling-origin backtest harness, Predictor protocol, baseline, metrics.
- `src/fpl_rocket/model/` - xP model v0.1 (Phase B): `minutes.py` (P(0/1-59/60+) + minutes per bin),
  `team_goals.py` (Dixon-Coles), `rates.py` (xG/xA shares, per-90 rates), `bonus.py`, `simulate.py` (joint
  Monte Carlo scored by the rules engine), `pipeline.py` (fit/predict, bundle hook, backtest predictor),
  `evaluate.py`, `config.py` (a-priori hyperparameters). Needs the `model` dependency group (scikit-learn,
  scipy); NOT in `requirements.txt` - the app never fits models.
- `src/fpl_rocket/views.py` - pure table builders for the pages (offline-testable).
- `src/fpl_rocket/ui.py` - sidebar (team ID), per-feature error guard, formatting, pitch.
- `src/fpl_rocket/config.py` - placeholder team ID and cache settings (the only place the ID appears).
- `streamlit_app.py` - entry point: navigation + sidebar; pages in `app_pages/`.
- `scripts/` - `reconcile_points.py <gw>`, `spike_probe.py --out <file>`, `build_bundle.py`, `state_summary.py <ids>`,
  `acceptance.py` (live acceptance matrix + timings), `record_app_snapshot.py` (offline app-test snapshot),
  `load_archive.py`, `run_backtest.py`.

## Run

```bash
uv sync                                  # Python 3.11 env with dev + model groups (default-groups)
uv run streamlit run streamlit_app.py    # the app
uv run pytest -q                         # tests (offline, replay recorded fixtures)
uv run ruff check                        # lint
uv run python scripts/reconcile_points.py 7   # rules vs API total_points for GW7 (expects 0 mismatches)
```

Streamlit Community Cloud installs from `requirements.txt` (kept minimal) and runs `streamlit_app.py`.

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
  p_play, p_60, p_cs, expected goals/assists/bonus/minutes), `xp_sims.parquet` (1000 joint simulations per
  player and GW, int8 points), `xp_fixtures.parquet` (per team-fixture expected goals for/against, P(CS),
  goals-conceded distribution = the model fixture-difficulty grid). Manifest: `model_version`
  (`v0.1-<git sha>`), `model_fitted_on`, `model_features`, `model_hyperparameters`, `model_runtime_s`. A model
  failure is recorded as `model_error`; the data bundle still publishes. The app does not read xP yet.

The app reads `https://github.com/drdata2018/fpl_rocket/releases/latest/download/manifest.json` (cached 300 s),
then the files of that bundle from its own tag, checks every sha256 and serves all shared data from that one
bundle (cached 600 s). The sidebar shows `Shared data as of <created_at> · bundle <bundle_id>`.
If there is no release yet, a checksum fails, the download times out or the bundle is older than 12 h, the app
uses the live FPL API as before and the sidebar shows **live API fallback** with the reason.

**First release (owner, once):** on GitHub open the repo -> **Actions** -> **bundle** (left list) ->
**Run workflow** -> branch `main` -> **Run workflow**. After about 2-3 minutes **Releases** shows
`Bundle <id>` marked *Latest*; the app switches to it within 5 minutes (or at once after a reboot of the
app). Until then every page works on the live API with the fallback notice. Later runs are automatic.

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
