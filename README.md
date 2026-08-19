# Arb — Interim Migration (LSEG)

Interim replacement for `ICEBREAKER/Arb`, rebuilt against the **LSEG Data
API** (`lseg.data`) instead of ICE Connect (`icepython`), for the period
while ICE API access is unavailable. Spread monitor for KC/RC (Arabica vs
Robusta) and CC/LCC (NY vs London Cocoa).

## Scope — this is the simplified dashboard, not a 1:1 port

Before migrating, the ICE-side dashboard was deliberately trimmed down (per
direct instruction, applied to `ICEBREAKER/Arb/Dashboard/app.py` first, then
carried over here unchanged):

- **Removed:** KPI card row, Roll-adjusted (Rollex) price source option,
  correlation/beta/intercept metric tiles (Return Scatter keeps its chart
  and R² only), FX Contribution bar + GBP/USD spot chart (CC/LCC), Rolling
  Correlation (both places it appeared), Spread Percentile Rank, and the
  entire Advanced Analytics section (ADF/cointegration tests, rolling
  half-life, rolling cointegration, spread distribution).
- **Kept:** Return Scatter (chart + R² only), Spread + bands, Z-score panel,
  Individual Legs, and — KC/RC only — the Ratio chart. CC/LCC ends with one
  fewer section than KC/RC by design; there's no CC/LCC equivalent to the
  Ratio view.
- **Consequence for this migration:** since Rollex is gone from the
  dashboard entirely, this project doesn't need `arb_{KC,RC,CC,LCC}.parquet`
  (the Rollex-price data) or a `sync_rollex.py` step at all — only
  `front_{KC,RC,CC,LCC}.parquet` (1st/2nd month prices) and `fx_gbp.parquet`.
  `statsmodels` was also dropped from `requirements.txt` since it was only
  used by the now-removed cointegration/ADF tests.

## Where the RICs came from

No new discovery needed — the ICE source's `%KC 1!`/`%KC 2!`-style
continuation symbols are the ICE-side equivalent of the `KCc1`/`KCc2`-style
LSEG continuation RICs already proven in the Rollex and Futures migrations.
Same root map: RC → `LRC`, LCC → `LCC` (both ICE Futures Europe/LIFFE
tickers). GBP/USD uses `GBP=` — the standard Reuters "reversed-quote" major
convention, confirmed returning USD-per-GBP directly (~1.35), no inversion
needed.

## What's here

- **`Code/ingest_front_lseg.py`** — px1 (1st nearby) / px2 (2nd nearby) per
  commodity. Same reindex-onto-union-of-dates + interpolate-internal-gaps
  treatment used in Rollex/Futures for the same underlying LSEG continuation-
  series gap issue.
- **`Code/ingest_gbp_lseg.py`** — GBP/USD spot.
- **`Database/front_{KC,RC,CC,LCC}.parquet`**, **`fx_gbp.parquet`** — full
  history from 2014.
- **`Dashboard/app.py`** — the simplified dashboard described above, copied
  verbatim (no API dependency).
- **`Automator/`** — `run.bat` (daily ingest + git push + email),
  `notify.py`.

## Validation

Checked against the ICE archive for the full overlapping history — cleanest
kind of series in this migration set (continuation prices, not thin
per-contract data):

| Commodity | px1 corr | px2 corr | px1 median diff | px2 median diff |
|---|---|---|---|---|
| KC | 0.99973 | 0.99999 | 0.072% | 0.000% |
| RC | 0.99993 | 1.00000 | 0.138% | 0.060% |
| CC | 0.99940 | 1.00000 | 0.000% | 0.000% |
| LCC | 1.00000 | 1.00000 | 0.056% | 0.055% |

## Running it

```bash
python Code/ingest_front_lseg.py           # incremental
python Code/ingest_front_lseg.py --full    # full rebuild from 2014-01-01
python Code/ingest_gbp_lseg.py
streamlit run Dashboard/app.py
```

Requires an authenticated LSEG Workspace/Eikon session on the host running
the ingest scripts.
