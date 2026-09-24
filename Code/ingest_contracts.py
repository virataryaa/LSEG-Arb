"""
Arb — Per-Contract Price Sync
===========================================================
Copies the trimmed per-contract price history (KC/RC/CC/LCC/SB/LSU) out of the
Futures/Database project into this repo's own Database/ folder, so the
Contract Explorer section of app.py can read a single specific vintage
(e.g. KC H26) with real calendar dates instead of only the rolled
front-month series. Futures/Database is refreshed daily by its own
pipeline; this script just re-copies the columns the dashboard needs so
LSEG-Arb stays a self-contained repo that Streamlit Cloud can deploy
without reaching across to another project's folder.

Usage:
    python ingest_contracts.py            # copy latest snapshot
    python ingest_contracts.py --check    # print tail of each file
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

SRC_DIR = Path(__file__).parent.parent.parent / "Futures" / "Database"
OUT_DIR = Path(__file__).parent.parent / "Database"

FILES = ["kc_futures.parquet", "rc_futures.parquet", "cc_futures.parquet", "lcc_futures.parquet",
         "sb_futures.parquet", "lsu_futures.parquet"]
COLS  = ["Date", "month", "year", "FND", "LTD", "settlement"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Print tail of each file and exit")
    args = parser.parse_args()

    if args.check:
        for fname in FILES:
            path = OUT_DIR / fname
            if path.exists():
                df = pd.read_parquet(path)
                print(f"\n=== {fname} ===")
                print(df.tail(5).to_string())
            else:
                print(f"\n=== {fname} ===  [not found]")
        return

    log.info("=" * 60)
    log.info("Per-Contract Sync | %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    failures = 0
    for fname in FILES:
        src = SRC_DIR / fname
        if not src.exists():
            log.error("%-20s source not found: %s", fname, src)
            failures += 1
            continue

        df = pd.read_parquet(src, columns=COLS)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / fname
        df.to_parquet(out, engine="pyarrow")
        log.info("%-20s copied %d rows  (last: %s)", fname, len(df), df["Date"].max().date())

    log.info("Done.")
    log.info("=" * 60)
    if failures:
        log.error("%d file(s) missing from source — check Futures/Database", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
