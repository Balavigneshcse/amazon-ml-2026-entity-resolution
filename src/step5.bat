@echo off
REM STEP 5: twin-aware models (decoy "twin" businesses: same name + a distinctive extra word, nearby house number).
REM About 45-60 minutes. Resumable. Existing models and files are not touched (new names end in _tw / v5).
cd /d "C:\Users\Balavignesh K\Documents\Studies\Hackathon\Amazon ML\business_entity_resolution\src"
set "ER_ANC=1"
set "ER_S2=anc"
set "ER_TWIN=1"
echo ===== 1/5 noise vocabulary + novel-word features %time%
python -u -W ignore stage2_v3.py twinprep
if errorlevel 1 goto fail
echo ===== 2/5 stage 2 with twin features %time%
python -u -W ignore stage2_v3.py train
if errorlevel 1 goto fail
python -u -W ignore stage2_v3.py infer
if errorlevel 1 goto fail
echo ===== 3/5 stage 3 with twin features %time%
python -u -W ignore stage3.py train
if errorlevel 1 goto fail
echo ===== 4/5 score the test set %time%
python -u -W ignore stage3.py infer
if errorlevel 1 goto fail
echo ===== 5/5 write output\v5 and validate %time%
set "ER_HITS=infer_hits_v4_tw"
set "ER_TAUJSON=decode_params_all_v4_tw_tl.json"
python -u -W ignore write_variant.py v5 auto+0.1 auto auto
if errorlevel 1 goto fail
cd /d "C:\Users\Balavignesh K\Documents\Studies\Hackathon\Amazon ML"
python -u dataset\student_resource\utils\validate_submission.py --matching business_entity_resolution\output\v5\matching_results.tsv --test-dir dataset\student_resource\dataset\test
echo ===== STEP 5 DONE %time%
exit /b 0
:fail
echo ===== STEP 5 STOPPED WITH AN ERROR %time% - copy the lines above and send them
exit /b 1
