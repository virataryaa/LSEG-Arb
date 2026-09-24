@echo off
setlocal EnableDelayedExpansion
set LOG="C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Arb\Automator\run_log.txt"
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

:: No Rollex sync step here - the simplified dashboard (KPI cards, Rollex
:: price source, Advanced Analytics all removed) doesn't use arb_*.parquet
:: at all anymore, only front_*.parquet and fx_gbp.parquet.

:: Step 1 - Front-month and 2nd-month prices from LSEG
echo [1] Running ingest_front_lseg.py... >> %LOG%
python "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Arb\Code\ingest_front_lseg.py" >> %LOG% 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: ingest_front_lseg.py failed >> %LOG%
    set INGEST_STATUS=error
    goto notify
)

:: Step 2 - GBP/USD spot from LSEG
echo [2] Running ingest_gbp_lseg.py... >> %LOG%
python "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Arb\Code\ingest_gbp_lseg.py" >> %LOG% 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: ingest_gbp_lseg.py failed >> %LOG%
    set INGEST_STATUS=error
    goto notify
)

:: Step 3 - Per-contract price sync (copies from Futures/Database for the
:: Contract Explorer section - Futures/Database's own pipeline refreshes it)
echo [3] Running ingest_contracts.py... >> %LOG%
python "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Arb\Code\ingest_contracts.py" >> %LOG% 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: ingest_contracts.py failed >> %LOG%
    set INGEST_STATUS=error
    goto notify
)

:: Step 4 - Push updated parquets to GitHub
echo [4] Pushing to GitHub... >> %LOG%
cd /d "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Arb"
git add Database\front_KC.parquet Database\front_RC.parquet Database\front_CC.parquet Database\front_LCC.parquet Database\front_SB.parquet Database\front_LSU.parquet Database\fx_gbp.parquet >> %LOG% 2>&1
git add Database\kc_futures.parquet Database\rc_futures.parquet Database\cc_futures.parquet Database\lcc_futures.parquet Database\sb_futures.parquet Database\lsu_futures.parquet >> %LOG% 2>&1
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
echo [5] Sending email notification... >> %LOG%
python "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\LSEG\Arb\Automator\notify.py" %INGEST_STATUS% %GIT_STATUS% >> %LOG% 2>&1

echo Run finished: %date% %time% >> %LOG%
