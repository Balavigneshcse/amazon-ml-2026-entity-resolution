@echo off
REM STEP 2: stage-2 v3 (cleaned names, name frequency, city conflict) trained at leaderboard-like density.
REM About 1-1.5 hours. Every part resumes from what is saved, so Ctrl+C and re-run is safe.
cd /d "%~dp0"
set "ER_ANC=1"
set "ER_S2=anc"
echo ===== 1/5 record tables %time%
python -u -W ignore stage2_v3.py prep
if errorlevel 1 goto fail
echo ===== 2/5 pair features %time%
python -u -W ignore stage2_v3.py feats
if errorlevel 1 goto fail
echo ===== 3/5 train + validate (compare with the old model) %time%
python -u -W ignore stage2_v3.py train
if errorlevel 1 goto fail
echo ===== 4/5 score the test set %time%
python -u -W ignore stage2_v3.py infer
if errorlevel 1 goto fail
echo ===== 5/5 write output\v3 and output\v3_fr_strict, then validate %time%
set "ER_HITS=infer_hits_v3"
set "ER_TAUJSON=decode_params_all_v3.json"
python -u -W ignore write_variant.py v3 auto auto auto
if errorlevel 1 goto fail
python -u -W ignore write_variant.py v3_fr_strict auto+0.15 auto auto
if errorlevel 1 goto fail
cd /d "%~dp0..\.."
for %%V in (v3 v3_fr_strict) do python -u dataset\student_resource\utils\validate_submission.py --matching business_entity_resolution\output\%%V\matching_results.tsv --test-dir dataset\student_resource\dataset\test
echo ===== STEP 2 DONE %time%
exit /b 0
:fail
echo ===== STEP 2 STOPPED WITH AN ERROR %time% - copy the lines above and send them
exit /b 1
