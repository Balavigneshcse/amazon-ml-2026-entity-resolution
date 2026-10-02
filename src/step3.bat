@echo off
REM STEP 3: stage-2 v3 with twin/orphan negatives up-weighted (the test has ~6-15x more of them than validation).
REM Trains weight 5 and weight 12, scores the test set, writes 3 candidate files. About 30-40 minutes. Resumable.
cd /d "%~dp0"
set "ER_ANC=1"
set "ER_S2=anc"
echo ===== 1/3 train weighted models + test-like validation %time%
python -u -W ignore stage2_v3.py trainw 5,12
if errorlevel 1 goto fail
echo ===== 2/3 score the test set %time%
python -u -W ignore stage2_v3.py inferw 5,12
if errorlevel 1 goto fail
echo ===== 3/3 write files (threshold chosen on the test-like score) + validate %time%
set "ER_HITS=infer_hits_v3"
set "ER_TAUJSON=decode_params_all_v3_tl.json"
python -u -W ignore write_variant.py v3_tl auto+0.1 auto auto
if errorlevel 1 goto fail
set "ER_HITS=infer_hits_v3_ow5"
set "ER_TAUJSON=decode_params_all_v3_ow5_tl.json"
python -u -W ignore write_variant.py v3_ow5 auto+0.1 auto auto
if errorlevel 1 goto fail
set "ER_HITS=infer_hits_v3_ow12"
set "ER_TAUJSON=decode_params_all_v3_ow12_tl.json"
python -u -W ignore write_variant.py v3_ow12 auto+0.1 auto auto
if errorlevel 1 goto fail
cd /d "%~dp0..\.."
for %%V in (v3_tl v3_ow5 v3_ow12) do python -u dataset\student_resource\utils\validate_submission.py --matching business_entity_resolution\output\%%V\matching_results.tsv --test-dir dataset\student_resource\dataset\test
echo ===== STEP 3 DONE %time%
exit /b 0
:fail
echo ===== STEP 3 STOPPED WITH AN ERROR %time% - copy the lines above and send them
exit /b 1
