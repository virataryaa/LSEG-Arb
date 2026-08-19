@echo off
setlocal EnableDelayedExpansion
set LOG="C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\Interim_Migration\Arb\Automator\run_log.txt"
set INGEST_STATUS=ok
set GIT_STATUS=skipped

:: Prevent Git Credential Manager from showing an interactive dialog in unattended runs.
:: If credentials are cached it pushes silently; if not, it fails immediately instead of hanging.
set GCM_INTERACTIVE=never
set GIT_TERMINAL_PROMPT=0
echo. >> %LOG%
echo ============================= >> %LOG%
echo Run started: %date% %time% >> %LOG%
echo ============================= >> %LOG%

:: No Rollex sync step here — the simplified dashboard (KPI cards, Rollex
:: price source, Advanced Analytics all removed) doesn't use arb_*.parquet
:: at all anymore, only front_*.parquet and fx_gbp.parquet.

:: Step 1 — Front-month and 2nd-month prices from LSEG
echo [1] Running ingest_front_lseg.py... >> %LOG%
python "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\Interim_Migration\Arb\Code\ingest_front_lseg.py" >> %LOG% 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: ingest_front_lseg.py failed >> %LOG%
    set INGEST_STATUS=error
    goto notify
)

:: Step 2 — GBP/USD spot from LSEG
echo [2] Running ingest_gbp_lseg.py... >> %LOG%
python "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\Interim_Migration\Arb\Code\ingest_gbp_lseg.py" >> %LOG% 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: ingest_gbp_lseg.py failed >> %LOG%
    set INGEST_STATUS=error
    goto notify
)

:: Step 3 — Push updated parquets to GitHub
echo [3] Pushing to GitHub... >> %LOG%
cd /d "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\Interim_Migration\Arb"
git add Database\front_KC.parquet Database\front_RC.parquet Database\front_CC.parquet Database\front_LCC.parquet Database\fx_gbp.parquet >> %LOG% 2>&1
git diff --cached --quiet
if %ERRORLEVEL% NEQ 0 (
    git commit -m "Auto update: Arb (LSEG) %date%" >> %LOG% 2>&1
    git push >> %LOG% 2>&1
    if !ERRORLEVEL! NEQ 0 (
        set GIT_STATUS=failed
        echo ERROR: git push failed >> %LOG%
    ) else (
        set GIT_STATUS=pushed
        echo Git push done. >> %LOG%
    )
) else (
    echo No changes to commit. >> %LOG%
    set GIT_STATUS=skipped
)

:notify
echo [4] Sending email notification... >> %LOG%
python "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\Interim_Migration\Arb\Automator\notify.py" %INGEST_STATUS% %GIT_STATUS% >> %LOG% 2>&1

echo Run finished: %date% %time% >> %LOG%
