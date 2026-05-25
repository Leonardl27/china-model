# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A set of quantitative economic models analyzing China's economy through the analytical
frameworks of Michael Pettis (structural imbalances) and Brad Setser (hidden reserves,
current-account forensics, FX intervention). Pure Python + matplotlib; outputs are PNG
charts, a text report, and a static GitHub Pages site. There is no application server,
database, or test suite.

## Commands

```bash
pip install -r requirements.txt   # setup (Python 3.12 in CI)

python dashboard.py               # full text report + dashboard.png (runs all 4 models)
python shadow_reserves.py         # one model: prints summary + writes shadow_reserves.png
python structural_imbalance.py    # likewise -> structural_imbalance.png
python current_account_forensics.py
python fx_intervention.py
python generate_site.py           # rebuild docs/index.html + copy PNGs into docs/
```

Each model script is runnable standalone (it has its own `__main__`). There are no lint,
format, or test commands configured — running the scripts and inspecting the PNG/console
output is the only verification loop.

Note: every `plot_*` function calls `plt.show()` after `plt.savefig(...)`. Under the default
Agg backend (headless/CI) this is a harmless no-op; the PNG is still written.

## Architecture

Five model/output scripts all sit on top of one shared data layer (`data_fetcher.py`).

### The 3-tier data fallback — the central design idea

`data_fetcher.py` is the only module that touches the network. Every external value flows
through a three-tier fallback so the models always produce output, even offline or without
an API key:

1. **Live API** — FRED (`requests`), World Bank (`wbgapi`), yfinance.
2. **Local JSON cache** — `data_cache/*.json`, with per-source TTLs (FRED 7d, World Bank 30d,
   yfinance 1d). These cache files are committed to the repo.
3. **Hardcoded `DEFAULTS`** — a single dict at the bottom of `data_fetcher.py` holding the
   original model values as last-resort fallback.

The `merge_into_list()` / `merge_into_dict()` helpers overlay whatever live data exists on
top of the hardcoded baseline, keeping the baseline for any year/date the API didn't return.
The public surface is the `fetch_*` functions (e.g. `fetch_official_reserves`,
`fetch_china_consumption_pct`), each of which knows its FRED/WB series ID, unit divisor, and
which `DEFAULTS` key to fall back to. Models call these `fetch_*` functions and never call the
APIs directly.

`FRED_API_KEY` comes from the environment or a `.env` file (loaded via python-dotenv). Without
it, FRED tiers silently degrade to cache/defaults — World Bank and yfinance need no key.

### Per-model structure

Each of the four models (`shadow_reserves`, `structural_imbalance`, `current_account_forensics`,
`fx_intervention`) follows the same shape:

- `build_*_dataset()` → returns a pandas DataFrame. Pulls live-or-fallback series via
  `data_fetcher`, then combines them with **model-specific hardcoded series defined inline in
  the function** (e.g. `state_bank_net_foreign`, `policy_bank_fx`, `bop_goods_surplus`,
  `fair_value_usdcny`). These inline series are Setser/Pettis analytical estimates with no public
  API source, so they live in the model, not in `data_fetcher.DEFAULTS`.
- `plot_*(df)` → builds a matplotlib figure and `savefig`s a `<model>.png`.
- `print_*_summary(df)` → prints a formatted console summary.

`current_account_forensics` also has `build_annual_ca_comparison()`; `structural_imbalance` also
has `build_rebalancing_scenarios()` and a module-level `BENCHMARKS` dict built at import time.

### Aggregation layers

- `dashboard.py` imports all four models' `build_*` functions, adds `build_market_comparison()`
  (more `fetch_*` calls plus inline series like property/youth-unemployment indices that have no
  API), and `compute_pettis_setser_scorecard()` which scores each framework prediction as
  CONFIRMED/PARTIAL/REJECTED. `plot_dashboard()` renders the 4×3 grid `dashboard.png`;
  `print_full_summary()` prints the consolidated report.
- `generate_site.py` re-runs all the `build_*`/scorecard functions, formats the latest-year
  numbers into `docs/index.html` (one big f-string template), and copies the chart PNGs into
  `docs/`. Writes a `.nojekyll` marker for GitHub Pages.

### Automation

`.github/workflows/weekly_update.yml` runs Mondays 08:00 UTC (and on manual dispatch): it runs
the models with the `FRED_API_KEY` secret, regenerates PNGs + `report.txt` + the site, commits
the refreshed `*.png`, `report.txt`, `data_cache/`, and `docs/` back to the repo, then deploys
`docs/` to GitHub Pages. The "Weekly model update: <date>" commits in the history come from this
job.

## Conventions and gotchas

- **Extending the time series is the most common edit and is positional.** Year ranges are
  hardcoded per model (`range(2004, 2026)` in shadow_reserves, `range(2000, 2026)` in structural,
  2018–2025 / quarterly 2019Q1–2025Q2 / monthly 2022-01→2025-06 elsewhere). To add a data point you
  must extend **both** the relevant list in `data_fetcher.DEFAULTS` **and** every model-specific
  inline list — each list is index-aligned with its year/quarter/month axis, so all lists for a
  given model must keep equal length. A length mismatch raises at DataFrame construction.
- All monetary values are in USD billions unless a name says otherwise (`*_T` = trillions,
  `*_pct` / `*_ZS` = percent). `unit_divisor` in the `fetch_*`/`get_fred_*` calls converts API
  units (e.g. millions → billions) — match it to the FRED/WB series' native units when adding one.
- Generated artifacts (`*.png`, `report.txt`, `docs/`, `data_cache/`) are committed to the repo on
  purpose (the Pages site serves them statically). Only `__pycache__/`, `.env`, and `.claude/` are
  gitignored.
- `data_fetcher` degrades silently: failures are logged via the `logging` module and fall through
  to the next tier rather than raising. If numbers look stale, check whether a cache file is being
  read instead of the API (delete the relevant `data_cache/*.json` to force a refetch).
