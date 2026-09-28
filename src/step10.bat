@echo off
rem STEP 10: third cross-encoder (continues from the L-12 model) trained on candidate + dense-retrieval pairs -> output\dn23_all
rem Resumable: if it stops, run the same command again. Keep DaVinci / Docker / WhatsApp closed (GPU memory).
cd /d "C:\Users\Balavignesh K\Documents\Studies\Hackathon\Amazon ML\business_entity_resolution\src"
set "ER_PMIN=0.02"
echo ===== 1/6 training pairs (dense neighbours of training entities) %time%
python -u -W ignore dense.py trainpairs || goto fail
python -u -W ignore dense.py cedata || goto fail
echo ===== 2/6 fine-tune cross-encoder 3 on the GPU (~70 min) %time%
set "ER_CE_TAG=_v3"
set "ER_CE_BASE=C:\Users\Balavignesh K\Documents\Studies\Hackathon\Amazon ML\business_entity_resolution\artifacts\ce_model_l12\final"
python -u -W ignore gpu_ce.py train || goto fail
echo ===== 3/6 score validation + test candidates (~95 min) %time%
python -u -W ignore gpu_ce.py score || goto fail
set "ER_CE_TAG="
set "ER_CE_BASE="
set "ER_DN_CE=,_l12,_v3"
set "ER_DN_V2=1"
echo ===== 4/6 score the dense-retrieval pairs (~60 min) %time%
python -u -W ignore dense.py ce || goto fail
echo ===== 5/6 stacker with three cross-encoders, validation comparison %time%
python -u -W ignore dense.py stack || goto fail
echo ===== 6/6 write output\dn23_all %time%
python -u -W ignore dense.py write || goto fail
echo ===== STEP 10 DONE %time%
exit /b 0
:fail
echo ===== STEP 10 STOPPED WITH AN ERROR %time% - send me the error, then run the same command again to resume
exit /b 1
