# FPL Rocket

Fantasy Premier League analytics and transfer planning for **any** team ID, built only on free data
(the public FPL API). Spec: [docs/PLAN_v2.md](docs/PLAN_v2.md).

Current stage: Phase A pages (Season Overview, My Team, Compare Players, Differentials v0, Model & Planner
placeholder) on shared-data bundles with a live-API fallback; Phase B step 1 (bundles v2, archive, backtest
harness + baseline). Live app: https://fpl-rocket.streamlit.app

- `src/fpl_rocket/api.py` - FPL API client (browser UA, retries/backoff, optional raw cache).
- `src/fpl_rocket/rules/` - versioned game rules (`season_2026_27.py`): scoring, transfers, selling price.
- `src/fpl_rocket/state.py` - planning-state reconstruction (squad, purchase/selling prices, bank, free-transfer range, chips).
- `src/fpl_rocket/bundle.py` - immutable parquet bundle + `manifest.json` with checksums (v2 adds element_history,
  round_prices, deadline flags).
- `src/fpl_rocket/data.py` - cached data access: shared (latest bundle release, else live API; ttl 600 s) and
  per-ID (ttl 300 s) layers.
- `src/fpl_rocket/archive.py` - normaliser for the community archive (vaastav) to one stable schema.
- `src/fpl_rocket/backtest.py` - rolling-origin backtest harness, Predictor protocol, baseline, metrics.
- `src/fpl_rocket/views.py` - pure table builders for the pages (offline-testable).
- `src/fpl_rocket/ui.py` - sidebar (team ID), per-feature error guard, formatting, pitch.
- `src/fpl_rocket/config.py` - placeholder team ID and cache settings (the only place the ID appears).
- `streamlit_app.py` - entry point: navigation + sidebar; pages in `app_pages/`.
- `scripts/` - `reconcile_points.py <gw>`, `spike_probe.py --out <file>`, `build_bundle.py`, `state_summary.py <ids>`,
  `acceptance.py` (live acceptance matrix + timings), `record_app_snapshot.py` (offline app-test snapshot),
  `load_archive.py`, `run_backtest.py`.

## Run

```bash
uv sync                                  # Python 3.11 env with dev deps
uv run streamlit run streamlit_app.py    # the app
uv run pytest -q                         # tests (offline, replay recorded fixtures)
uv run ruff check                        # lint
uv run python scripts/reconcile_points.py 7   # rules vs API total_points for GW7 (expects 0 mismatches)
```

Streamlit Community Cloud installs from `requirements.txt` (kept minimal) and runs `streamlit_app.py`.

## Shared-data bundles (GitHub Releases)

`.github/workflows/bundle.yml` runs every 3 hours at :40 (and on demand). It builds one immutable bundle
(bootstrap tables, fixtures with results, live stats of every finished GW, `element_history` for every player,
`round_prices`; about 0.6 MB, about 1 minute) and publishes it as release `bundle-<bundle_id>`, marked
*latest*. The last run before a deadline (deadline less than 3 h away) is also published as `deadline-gw<N>`,
kept forever (the scored snapshot). `bundle-*` releases older than 7 days are deleted; `deadline-*` never.

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

Baseline "form per 90 x availability" (the bar the Phase B model must beat):

| season | RMSE | MAE | within-GW Spearman (all / played) | best XI + C minus GW avg (mean, sd) |
|---|---|---|---|---|
| 2024-25 | 2.087 | 1.041 | 0.696 / 0.319 | +4.7, 13.2 |
| 2025-26 | 2.121 | 1.043 | 0.719 / 0.276 | -5.1, 12.2 |
