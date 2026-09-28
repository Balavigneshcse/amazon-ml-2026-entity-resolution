@echo off
REM STEP 1 (no training): anc model on the test set + leaderboard-like validation + submission file #2.
REM Every part resumes from what is already saved, so Ctrl+C and re-run is safe.
cd /d "C:\Users\Balavignesh K\Documents\Studies\Hackathon\Amazon ML\business_entity_resolution\src"
set "ER_ANC=1"
set "ER_S2=anc"
set "ER_DECODE=d19"

echo ===== 1/4 cluster-consistency features for the test set %time%
python -u -W ignore cluster.py test t1 France,India,US
if errorlevel 1 goto fail

echo ===== 2/4 leaderboard-like validation (19%% of businesses removed) %time%
python -u -W ignore sim_drop.py
if errorlevel 1 goto fail

echo ===== 3/4 score the test set with the anc model and write the submission %time%
python -u -W ignore infer.py f2 t1 all
if errorlevel 1 goto fail

echo ===== 4/4 official validator %time%
cd /d "C:\Users\Balavignesh K\Documents\Studies\Hackathon\Amazon ML"
python -u dataset\student_resource\utils\validate_submission.py --matching business_entity_resolution\output\variant_anc\matching_results.tsv --candidate business_entity_resolution\output\candidate_pairs.tsv --test-dir dataset\student_resource\dataset\test
echo ===== STEP 1 DONE %time%
exit /b 0

:fail
echo ===== STEP 1 STOPPED WITH AN ERROR %time% - copy the lines above and send them
exit /b 1
