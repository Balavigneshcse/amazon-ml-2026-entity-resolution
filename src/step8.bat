@echo off
REM STEP 8 (overnight): second, larger cross-encoder (MiniLM-L-12, 33M params, Apache-2.0) on 2M different training pairs,
REM then combine BOTH transformers with the cluster model. About 3.5 hours. Resumable: just run it again if it stops.
cd /d "C:\Users\Balavignesh K\Documents\Studies\Hackathon\Amazon ML\business_entity_resolution\src"
set "ER_CE_TAG=_l12"
set "ER_CE_BASE=cross-encoder/ms-marco-MiniLM-L-12-v2"
set "ER_CE_N=1000000"
echo ===== 1/5 training pairs %time%
python -u -W ignore gpu_ce.py data
if errorlevel 1 goto fail
echo ===== 2/5 fine-tune the larger model on the GPU %time%
python -u -W ignore gpu_ce.py train
if errorlevel 1 goto fail
echo ===== 3/5 score validation + test candidates %time%
python -u -W ignore gpu_ce.py score
if errorlevel 1 goto fail
echo ===== 4/5 combine both transformers + cluster model, compare on validation %time%
set "ER_CE_TAG="
set "ER_CE_STACK=,_l12"
python -u -W ignore gpu_ce.py stack
if errorlevel 1 goto fail
python -u -W ignore gpu_ce.py write
if errorlevel 1 goto fail
echo ===== 5/5 validate output\ce_all_l12 %time%
cd /d "C:\Users\Balavignesh K\Documents\Studies\Hackathon\Amazon ML"
python -u dataset\student_resource\utils\validate_submission.py --matching business_entity_resolution\output\ce_all_l12\matching_results.tsv --candidate business_entity_resolution\package\LOGIC_MAKERS_submission\output\candidate_pairs.tsv --test-dir dataset\student_resource\dataset\test
echo ===== STEP 8 DONE %time%
exit /b 0
:fail
echo ===== STEP 8 STOPPED WITH AN ERROR %time% - run the same command again to resume
exit /b 1
