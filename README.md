# FPL Rocket

Fantasy Premier League analytics and transfer planning for **any** team ID, built only on free data
(the public FPL API). Spec: [docs/PLAN_v2.md](docs/PLAN_v2.md).

Current stage: foundation (Phase A step 2) plus the day-1 hosting spike.

- `src/fpl_rocket/api.py` - FPL API client (browser UA, retries/backoff, optional raw cache).
- `src/fpl_rocket/rules/` - versioned game rules (`season_2026_27.py`): scoring, transfers, selling price.
- `src/fpl_rocket/state.py` - planning-state reconstruction (squad, purchase/selling prices, bank, free-transfer range, chips).
- `src/fpl_rocket/bundle.py` - immutable parquet bundle + `manifest.json` with checksums.
- `streamlit_app.py` - hosting-spike page (Streamlit Community Cloud entry point).
- `scripts/` - `reconcile_points.py <gw>`, `spike_probe.py --out <file>`, `build_bundle.py`, `state_summary.py <ids>`.

## Run

```bash
uv sync                                  # Python 3.11 env with dev deps
uv run streamlit run streamlit_app.py    # the app
uv run pytest -q                         # tests (offline, replay recorded fixtures)
uv run ruff check                        # lint
uv run python scripts/reconcile_points.py 7   # rules vs API total_points for GW7 (expects 0 mismatches)
```

Streamlit Community Cloud installs from `requirements.txt` (kept minimal) and runs `streamlit_app.py`.
