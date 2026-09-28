@echo off
REM STEP 4: refinement pass (stage 3). About 20-30 minutes. Resumable.
cd /d "C:\Users\Balavignesh K\Documents\Studies\Hackathon\Amazon ML\business_entity_resolution\src"
set "ER_ANC=1"
set "ER_S2=anc"
echo ===== 1/3 train + compare with v3 %time%
python -u -W ignore stage3.py train
if errorlevel 1 goto fail
echo ===== 2/3 score the test set %time%
python -u -W ignore stage3.py infer
if errorlevel 1 goto fail
echo ===== 3/3 write output\v4_tl and validate %time%
set "ER_HITS=infer_hits_v4"
set "ER_TAUJSON=decode_params_all_v4_tl.json"
python -u -W ignore write_variant.py v4_tl auto+0.1 auto auto
if errorlevel 1 goto fail
cd /d "C:\Users\Balavignesh K\Documents\Studies\Hackathon\Amazon ML"
python -u dataset\student_resource\utils\validate_submission.py --matching business_entity_resolution\output\v4_tl\matching_results.tsv --test-dir dataset\student_resource\dataset\test
echo ===== STEP 4 DONE %time%
exit /b 0
:fail
echo ===== STEP 4 STOPPED WITH AN ERROR %time% - copy the lines above and send them
exit /b 1
