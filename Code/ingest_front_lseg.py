"""
Arb — Front-Month Price Ingest (LSEG interim migration)
===========================================================
LSEG-API replacement for ICEBREAKER/Arb/Code/ingest_front.py (icepython-
based). Same output schema (px1 = 1st nearby, px2 = 2nd nearby,
back-adjusted continuation), same 4 commodities. The ICE source's
"%KC 1!"/"%KC 2!"-style symbols are the ICE-side equivalent of the
KCc1/KCc2-style continuation RICs already proven in the Rollex/Futures
migrations, so no new RIC discovery was needed here — same root map
(RC->LRC, LCC->LCC) as those two projects.

Units match the raw LSEG quote, same as the ICE source:
    KC  -> c/lbs   (multiply by 22.0462 for $/MT)
    RC  -> $/MT
    CC  -> $/MT
    LCC -> GBP/MT  (multiply by GBP/USD for $/MT)

Usage:
    python ingest_front_lseg.py            # incremental
    python ingest_front_lseg.py --full     # full pull from 2014-01-01
    python ingest_front_lseg.py --check    # print tail of each file
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

OUT_DIR    = Path(__file__).parent.parent / "Database"
FULL_START = "2014-01-01"

# LSEG continuation RICs — same roots proven in the Rollex/Futures migrations.
ROOTS = {
    "KC":  "KC",
    "RC":  "LRC",
    "CC":  "CC",
    "LCC": "LCC",
}


def _fetch_leg(ld, ric: str, start: str, end: str) -> pd.Series:
    try:
        df = ld.get_history(universe=[ric], fields=["TRDPRC_1"], start=start, end=end,
                             interval="daily", count=10000)
        if df is None or df.empty:
            log.warning("No data for %s", ric)
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        s = df.iloc[:, 0].dropna()
        s.index = pd.to_datetime(s.index).normalize()
        return s
    except Exception as e:
        log.warning("Fetch failed for %s: %s", ric, e)
        return pd.Series(dtype=float)


def fetch_pair(ld, name: str, root: str, start: str, end: str) -> pd.DataFrame:
    ric1, ric2 = f"{root}c1", f"{root}c2"
    log.info("%-4s  fetching %s and %s", name, ric1, ric2)
    s1 = _fetch_leg(ld, ric1, start, end)
    s2 = _fetch_leg(ld, ric2, start, end)

    # Same gap-density fix used in the Rollex/Futures migrations: reindex
    # onto the union of dates either leg actually has, interpolate strictly-
    # internal gaps only.
    if not s1.empty or not s2.empty:
        full_idx = sorted(set(s1.index) | set(s2.index))
        s1 = s1.reindex(full_idx).interpolate(method="linear", limit_area="inside")
        s2 = s2.reindex(full_idx).interpolate(method="linear", limit_area="inside")

    df = pd.concat([s1.rename("px1"), s2.rename("px2")], axis=1)
    df.index.name = "Date"
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",  action="store_true", help="Full pull from 2014-01-01")
    parser.add_argument("--check", action="store_true", help="Print tail of each file and exit")
    args = parser.parse_args()

    if args.check:
        for name in ROOTS:
            path = OUT_DIR / f"front_{name}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                print(f"\n=== {name} ===")
                print(df.tail(5).to_string())
            else:
                print(f"\n=== {name} ===  [not found]")
        return

    log.info("=" * 60)
    log.info("Front-Month Ingest (LSEG) | %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    import lseg.data as ld
    ld.open_session()
    log.info("LSEG session opened.")

    end = datetime.date.today().isoformat()
    failures = 0

    try:
        for name, root in ROOTS.items():
            out = OUT_DIR / f"front_{name}.parquet"

            if args.full or not out.exists():
                start = FULL_START
                log.info("%-4s  mode: FULL from %s", name, start)
            else:
                existing = pd.read_parquet(out)
                existing.index = pd.to_datetime(existing.index)
                latest = existing.index.max()
                start  = (latest - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
                log.info("%-4s  mode: INCREMENTAL from %s", name, start)

            new_df = fetch_pair(ld, name, root, start, end)

            if new_df.dropna(how="all").empty:
                log.error("%-4s  no data returned — check RIC", name)
                failures += 1
                continue

            if out.exists() and not args.full:
                old_df = pd.read_parquet(out)
                old_df.index = pd.to_datetime(old_df.index)
                combined = pd.concat([old_df, new_df])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            else:
                combined = new_df.sort_index()

            combined = combined[combined.index < pd.Timestamp.today().normalize()]
            combined = combined.dropna(how="all")

            OUT_DIR.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(out, engine="pyarrow")
            log.info("%-4s  saved %d rows to %s  (last: %s  px1: %.2f  px2: %.2f)",
                     name, len(combined), out.name,
                     combined.index.max().date(),
                     combined["px1"].iloc[-1] if not combined["px1"].isna().all() else float("nan"),
                     combined["px2"].iloc[-1] if not combined["px2"].isna().all() else float("nan"))
    finally:
        ld.close_session()

    log.info("Done.")
    log.info("=" * 60)
    if failures:
        log.error("%d RIC(s) returned no data — check LSEG session or RICs", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
