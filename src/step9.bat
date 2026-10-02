@echo off
rem STEP 9: dense retrieval (GPU bi-encoder) finds true matches the blocking missed -> output\dn_all
rem Resumable: if it stops, run the same command again. Needs ~3.5 GB free GPU memory (close other GPU applications).
cd /d "%~dp0"
set "ER_PMIN=0.02"
echo ===== 1/7 training pairs %time%
python -u -W ignore dense.py data || goto fail
echo ===== 2/7 fine-tune the bi-encoder on the GPU (~35 min) %time%
python -u -W ignore dense.py train || goto fail
echo ===== 3/7 embed all entities and records (~40 min) %time%
python -u -W ignore dense.py embed || goto fail
echo ===== 4/7 nearest-neighbour search + recall report %time%
python -u -W ignore dense.py retrieve || goto fail
echo ===== 5/7 score the new pairs with both cross-encoders %time%
python -u -W ignore dense.py ce || goto fail
echo ===== 6/7 stacker over old + new candidates, validation comparison %time%
python -u -W ignore dense.py stack || goto fail
echo ===== 7/7 write output\dn_all %time%
python -u -W ignore dense.py write || goto fail
echo ===== STEP 9 DONE %time%
exit /b 0
:fail
echo ===== STEP 9 STOPPED WITH AN ERROR %time% - fix the reported error, then run the same command again to resume
exit /b 1
