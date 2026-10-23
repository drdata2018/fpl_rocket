# FPL Rocket

Fantasy Premier League analytics and transfer planning for **any** team ID, built only on free data
(the public FPL API). Spec: [docs/PLAN_v2.md](docs/PLAN_v2.md).

Current stage: Phase A pages on live data (Season Overview, My Team, Compare Players, Differentials v0,
Model & Planner placeholder). Live app: https://fpl-rocket.streamlit.app

- `src/fpl_rocket/api.py` - FPL API client (browser UA, retries/backoff, optional raw cache).
- `src/fpl_rocket/rules/` - versioned game rules (`season_2026_27.py`): scoring, transfers, selling price.
- `src/fpl_rocket/state.py` - planning-state reconstruction (squad, purchase/selling prices, bank, free-transfer range, chips).
- `src/fpl_rocket/bundle.py` - immutable parquet bundle + `manifest.json` with checksums.
- `src/fpl_rocket/data.py` - cached data access: shared (ttl 600 s) and per-ID (ttl 300 s) layers.
- `src/fpl_rocket/views.py` - pure table builders for the pages (offline-testable).
- `src/fpl_rocket/ui.py` - sidebar (team ID), per-feature error guard, formatting, pitch.
- `src/fpl_rocket/config.py` - placeholder team ID and cache settings (the only place the ID appears).
- `streamlit_app.py` - entry point: navigation + sidebar; pages in `app_pages/`.
- `scripts/` - `reconcile_points.py <gw>`, `spike_probe.py --out <file>`, `build_bundle.py`, `state_summary.py <ids>`,
  `acceptance.py` (live acceptance matrix + timings), `record_app_snapshot.py` (offline app-test snapshot).

## Run

```bash
uv sync                                  # Python 3.11 env with dev deps
uv run streamlit run streamlit_app.py    # the app
uv run pytest -q                         # tests (offline, replay recorded fixtures)
uv run ruff check                        # lint
uv run python scripts/reconcile_points.py 7   # rules vs API total_points for GW7 (expects 0 mismatches)
```

Streamlit Community Cloud installs from `requirements.txt` (kept minimal) and runs `streamlit_app.py`.
