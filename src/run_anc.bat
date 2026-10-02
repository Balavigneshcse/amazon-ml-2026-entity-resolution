@echo off
REM Cluster-consistency experiment: (re)build cluster features (skips finished shards), retrain stage 2 with them, tune thresholds.
REM Safe to stop with Ctrl+C and run again: every step resumes from what is already saved.
cd /d "%~dp0"
set "ER_ANC=1"
set "ER_S2=anc"
python -u -W ignore cluster.py train f2 auto all
if errorlevel 1 exit /b 1
python -u -W ignore train.py stage2 f2 auto all
if errorlevel 1 exit /b 1
python -u -W ignore tune.py f2 auto all
