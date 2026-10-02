@echo off
REM STEP 6: small candidate sets. Stage-1 filter p1 >= 0.02 (~5 candidates per business instead of 100); every final
REM model scores ONLY these candidates, and candidate_pairs.tsv lists exactly them. About 15-20 minutes.
cd /d "%~dp0"
set "ER_PMIN=0.02"
set "ER_ANC=1"
set "ER_S2=anc"
echo ===== 1/4 cluster model (France, US) on filtered candidates %time%
python -u -W ignore infer.py f2 t1 all
if errorlevel 1 goto fail
echo ===== 2/4 twin-aware model (India) on filtered candidates %time%
set "ER_TWIN=1"
python -u -W ignore stage2_v3.py infer
if errorlevel 1 goto fail
python -u -W ignore stage3.py infer
if errorlevel 1 goto fail
echo ===== 3/4 write output\final_p020 (same settings as the 0.946 best) %time%
python -u -W ignore multi_variant.py final_p020 default=infer_hits_anc_p020:0.85 India=infer_hits_v4_tw_p020:0.75 US=infer_hits_anc_p020:0.7
if errorlevel 1 goto fail
echo ===== 4/4 build the zip (candidate_pairs.tsv = filtered candidates) + official validation %time%
python -u -W ignore make_package.py final_p020
echo ===== STEP 6 DONE %time%
exit /b 0
:fail
echo ===== STEP 6 STOPPED WITH AN ERROR %time% - copy the lines above and send them
exit /b 1
