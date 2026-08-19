"""
Arb — GBP/USD Ingest (LSEG interim migration)
=================================================
LSEG-API replacement for ICEBREAKER/Arb/Code/ingest_gbp.py (icepython-
based). Saves to Arb/Database/fx_gbp.parquet, same schema.

LSEG RIC "GBP=" follows the standard Reuters convention for GBP (one of
the "reversed-quote" majors, same as EUR/AUD/NZD): it already returns
USD per 1 GBP directly (~1.25-1.40), matching what the ICE source's
GBP_USD column expects — no --invert needed, but the flag is kept for
parity/safety in case that ever isn't true.

Usage:
    python ingest_gbp_lseg.py            # incremental
    python ingest_gbp_lseg.py --full     # full pull from 2014-01-01
    python ingest_gbp_lseg.py --check    # print tail to verify rate direction
    python ingest_gbp_lseg.py --invert   # store 1/rate if LSEG ever returns GBP per USD
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path

import pandas as pd
pd.set_option("future.no_silent_downcasting", True)  # silences a harmless lseg.data internal FutureWarning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

OUT_FILE   = Path(__file__).parent.parent / "Database" / "fx_gbp.parquet"
FULL_START = "2014-01-01"

GBP_RIC = "GBP="


def _fetch(ld, ric: str, start: str, end: str) -> pd.Series:
    try:
        df = ld.get_history(universe=[ric], fields=["MID_PRICE"], start=start, end=end,
                             interval="daily", count=10000)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        s = df.iloc[:, 0].dropna().rename("GBP_USD")
        s.index = pd.to_datetime(s.index).normalize()
        return s
    except Exception as e:
        log.warning("Fetch failed for %s: %s", ric, e)
        return pd.Series(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",   action="store_true", help="Full pull from 2014-01-01")
    parser.add_argument("--check",  action="store_true", help="Print last rows and exit")
    parser.add_argument("--invert", action="store_true", help="Store 1/rate")
    args = parser.parse_args()

    if args.check and OUT_FILE.exists():
        df = pd.read_parquet(OUT_FILE)
        print(df.tail(10).to_string())
        print(f"\nLast date: {df.index.max().date()}   Last GBP_USD: {df['GBP_USD'].iloc[-1]:.4f}")
        print("Expected: ~1.25-1.40 (1 GBP = X USD). If ~0.72, run with --invert.")
        return

    log.info("=" * 60)
    log.info("GBP/USD Ingest (LSEG) | %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    import lseg.data as ld
    ld.open_session()
    log.info("LSEG session opened.")

    try:
        if args.full or not OUT_FILE.exists():
            start = FULL_START
            log.info("Mode: FULL from %s", start)
        else:
            existing = pd.read_parquet(OUT_FILE)
            latest   = existing.index.max()
            start    = (latest - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            log.info("Mode: INCREMENTAL from %s (overlap 5 days)", start)

        end = datetime.date.today().isoformat()

        log.info("Fetching %s", GBP_RIC)
        series = _fetch(ld, GBP_RIC, start, end)

        if series.empty:
            log.error("No data returned for %s.", GBP_RIC)
            sys.exit(1)

        if args.invert:
            series = (1 / series).rename("GBP_USD")
            log.info("--invert applied: storing 1/rate")

        new_df = series.to_frame()
        new_df.index.name = "Date"

        if OUT_FILE.exists() and not args.full:
            old_df = pd.read_parquet(OUT_FILE)
            combined = (pd.concat([old_df, new_df])
                        .loc[~pd.concat([old_df, new_df]).index.duplicated(keep="last")]
                        .sort_index())
        else:
            combined = new_df.sort_index()

        combined = combined[combined.index < pd.Timestamp.today().normalize()]
        combined = combined.dropna()

        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(OUT_FILE, engine="pyarrow")
        log.info("Saved %d rows to %s  (last: %s  rate: %.4f)",
                 len(combined), OUT_FILE.name,
                 combined.index.max().date(), combined["GBP_USD"].iloc[-1])
    finally:
        ld.close_session()

    log.info("=" * 60)


if __name__ == "__main__":
    main()
