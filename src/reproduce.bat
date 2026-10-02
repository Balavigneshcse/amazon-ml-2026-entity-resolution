@echo off
REM Full reproduction from raw data to the submission zip. Resumable: re-run after any interruption.
cd /d "%~dp0"
python -u -W ignore run.py train 1.0 f2 || goto fail
python -u -W ignore train.py stage1 f2 auto all || goto fail
python -u -W ignore cluster.py train f2 auto all || goto fail
set "ER_ANC=1"
set "ER_S2=anc"
python -u -W ignore train.py stage2 f2 auto all || goto fail
python -u -W ignore tune.py f2 auto all || goto fail
python -u -W ignore run.py test t1 || goto fail
python -u -W ignore infer.py f2 t1 all --stage1-only || goto fail
python -u -W ignore cluster.py test t1 auto || goto fail
python -u -W ignore sim_drop.py || goto fail
python -u -W ignore stage2_v3.py prep || goto fail
python -u -W ignore stage2_v3.py feats || goto fail
python -u -W ignore stage2_v3.py train || goto fail
python -u -W ignore stage2_v3.py infer || goto fail
python -u -W ignore stage3.py train || goto fail
python -u -W ignore stage3.py infer || goto fail
set "ER_HITS=infer_hits_v4"
set "ER_TAUJSON=decode_params_all_v4_tl.json"
python -u -W ignore write_variant.py v4_tl auto+0.1 auto auto || goto fail
set "ER_TWIN=1"
python -u -W ignore stage2_v3.py twinprep || goto fail
python -u -W ignore stage2_v3.py train || goto fail
python -u -W ignore stage3.py train || goto fail
REM final inference: every matching model scores only candidates with stage-1 p1 >= 0.02 (~5 per entity)
set "ER_PMIN=0.02"
python -u -W ignore infer.py f2 t1 all || goto fail
python -u -W ignore stage2_v3.py infer || goto fail
python -u -W ignore stage3.py infer || goto fail
python -u -W ignore multi_variant.py final_p020 default=infer_hits_anc_p020:0.85 India=infer_hits_v4_tw_p020:0.75 US=infer_hits_anc_p020:0.7 || goto fail
python -u -W ignore gpu_ce.py data || goto fail
python -u -W ignore gpu_ce.py train || goto fail
python -u -W ignore gpu_ce.py score || goto fail
set "ER_CE_TAG=_l12"
set "ER_CE_BASE=cross-encoder/ms-marco-MiniLM-L-12-v2"
set "ER_CE_N=1000000"
python -u -W ignore gpu_ce.py data || goto fail
python -u -W ignore gpu_ce.py train || goto fail
python -u -W ignore gpu_ce.py score || goto fail
set "ER_CE_TAG="
set "ER_CE_STACK=,_l12"
python -u -W ignore gpu_ce.py stack || goto fail
python -u -W ignore gpu_ce.py write || goto fail
set "ER_CE_STACK="
set "ER_DN_V2=1"
python -u -W ignore dense.py data || goto fail
python -u -W ignore dense.py train || goto fail
python -u -W ignore dense.py embed || goto fail
python -u -W ignore dense.py retrieve || goto fail
python -u -W ignore dense.py ce || goto fail
python -u -W ignore dense.py stack || goto fail
python -u -W ignore multi_variant.py dn2_FR095 default=infer_hits_dn2:0.95 India=infer_hits_dn2:0.7 US=infer_hits_dn2:0.75 || goto fail
call step10.bat || goto fail
python -u -W ignore multi_variant.py final_FRgate default=infer_hits_dn2:0.8:0.95 India=infer_hits_dn23:0.8 US=infer_hits_dn23:0.7 || goto fail
set "ER_DN=1"
python -u -W ignore make_package.py final_FRgate || goto fail
echo ===== REPRODUCTION DONE
exit /b 0
:fail
echo ===== STOPPED WITH AN ERROR
exit /b 1
