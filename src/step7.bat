@echo off
REM STEP 7: GPU cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2, 22M params, Apache-2.0) on the RTX 3050.
REM About 2-3 hours. Every part resumes where it stopped (training checkpoints every 2000 steps, scoring per shard).
cd /d "%~dp0"
echo ===== 1/6 GPU check %time%
python -u -W ignore gpu_ce.py check
if errorlevel 1 goto fail
echo ===== 2/6 training pairs %time%
python -u -W ignore gpu_ce.py data
if errorlevel 1 goto fail
echo ===== 3/6 fine-tune on the GPU %time%
python -u -W ignore gpu_ce.py train
if errorlevel 1 goto fail
echo ===== 4/6 score validation + test candidates %time%
python -u -W ignore gpu_ce.py score
if errorlevel 1 goto fail
echo ===== 5/6 combine with the current model, compare on validation %time%
python -u -W ignore gpu_ce.py stack
if errorlevel 1 goto fail
echo ===== 6/6 write submission files + validate %time%
python -u -W ignore gpu_ce.py write
if errorlevel 1 goto fail
cd /d "%~dp0..\.."
for %%V in (ce_all final_ceUS final_ceIN final_ceFR) do python -u dataset\student_resource\utils\validate_submission.py --matching business_entity_resolution\output\%%V\matching_results.tsv --candidate business_entity_resolution\package\LOGIC_MAKERS_submission\output\candidate_pairs.tsv --test-dir dataset\student_resource\dataset\test
echo ===== STEP 7 DONE %time%
exit /b 0
:fail
echo ===== STEP 7 STOPPED WITH AN ERROR %time% - copy the lines above and send them
exit /b 1
