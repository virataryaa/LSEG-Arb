"""
notify.py — Arb (LSEG interim migration) Automator email summary
Usage: python notify.py <status> <git_status>
  status     : ok | error
  git_status : pushed | skipped | failed
"""

import sys
import datetime
import pandas as pd
from pathlib import Path

TO_EMAIL = "virat.arya@etgworld.com"
DB_DIR   = Path(r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Arb\Database")

KC_FACTOR = 22.0462  # c/lbs -> $/MT

status     = sys.argv[1] if len(sys.argv) > 1 else "ok"
git_status = sys.argv[2] if len(sys.argv) > 2 else "unknown"
run_dt     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
today      = datetime.date.today().strftime("%Y-%m-%d")


def _tail(path: Path, col: str, factor: float = 1.0):
    if not path.exists():
        return "FILE NOT FOUND", "—", "—"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    last_date = df.index.max().date()
    last_val  = df[col].iloc[-1] * factor if col in df.columns else float("nan")
    row_count = len(df)
    return last_date, last_val, row_count


def spread_summary() -> str:
    lines = []

    lines.append("  -- Front-month (LSEG continuation) --")
    for name, factor, unit in [
        ("KC",  KC_FACTOR, "$/MT"),
        ("RC",  1.0,       "$/MT"),
        ("CC",  1.0,       "$/MT"),
        ("LCC", 1.0,       "GBP/MT"),
    ]:
        path = DB_DIR / f"front_{name}.parquet"
        if not path.exists():
            lines.append(f"  {name:<4}  FILE NOT FOUND")
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        last_date = df.index.max().date()
        px1 = df["px1"].iloc[-1] * factor if "px1" in df.columns else float("nan")
        px2 = df["px2"].iloc[-1] * factor if "px2" in df.columns else float("nan")
        rows = len(df)
        lines.append(
            f"  {name:<4}  {rows:>5} rows   last: {last_date}   "
            f"px1: {px1:>9.2f}  px2: {px2:>9.2f} {unit}"
        )

    lines.append("")
    lines.append("  -- FX --")
    last_date, rate, rows = _tail(DB_DIR / "fx_gbp.parquet", "GBP_USD")
    lines.append(f"  GBP/USD  {rows:>5} rows   last: {last_date}   rate: {rate:.4f}")

    return "\n".join(lines)


def send_outlook_email(subject: str, body: str):
    try:
        import win32com.client
        outlook      = win32com.client.Dispatch("Outlook.Application")
        mail         = outlook.CreateItem(0)
        mail.To      = TO_EMAIL
        mail.Subject = subject
        mail.Body    = body
        mail.Send()
        print(f"  Email sent -> {TO_EMAIL}")
    except Exception as e:
        print(f"  Email failed: {e}")


ok  = status == "ok"
tag = "[OK]" if ok else "[ERROR]"
subject = f"{tag} LSEG-Arb — {today}"

git_line = {
    "pushed":  "GitHub  : Pushed successfully",
    "skipped": "GitHub  : No changes — push skipped",
    "failed":  "GitHub  : PUSH FAILED",
}.get(git_status, f"GitHub  : {git_status}")

body = f"""LSEG Arb — Daily Update
Run time : {run_dt}
Status   : {"OK" if ok else "ERROR — ingest failed, check run_log.txt"}
{git_line}

{"=" * 60}
ARB DATA SUMMARY
{"=" * 60}
{spread_summary()}
{"=" * 60}
Log: C:\\Users\\virat.arya\\ETG\\SoftsDatabase - Documents\\Database\\Hardmine\\LSEG\\Arb\\Automator\\run_log.txt
"""

print(body)
send_outlook_email(subject, body)
