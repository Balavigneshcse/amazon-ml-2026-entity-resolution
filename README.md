# Business Entity Resolution — Team LOGIC MAKERS

Pipeline: **normalise → multi-key blocking (100 candidates / entity) → stage 1 LightGBM (filter) → stage 2 LightGBM
(precision features, trained at test-like density) → stage 3 LightGBM (refinement) → one-owner decoding**.
No external data, APIs, geocoding or pretrained weights are used. The only models are LightGBM (MIT licence).

## Environment
* Python 3.12; `pip install -r requirements.txt` (polars, numpy, scikit-learn, lightgbm, rapidfuzz, Unidecode).
* Tested on Windows 11, 12 CPU cores, 15 GB RAM, ~110 GB free disk for intermediates. No GPU needed.
* Data is expected at `../dataset/student_resource/dataset/{train,test}` relative to this folder; override with the
  environment variables `ER_DATA` (dataset folder) and `ER_ART` (folder for intermediate files).
* Every stage writes its results shard by shard and skips finished work, so an interrupted run is resumed by running the
  same command again.

## Source files (`src/`)
| file | role |
|---|---|
| `config.py` | paths, seeds, candidates per entity (`KBLOCK = 100`) |
| `normalize.py` | name/address normalisation: accent folding, transliteration (Unidecode), legal-suffix stripping, abbreviation and state canonicalisation, phonetic squash, consonant skeleton |
| `build_universe.py` | normalised per-country universe for train (with ground truth) and test |
| `blocking.py` | candidate generation: 9 key families, idf-weighted cosine ranking, top-100 per entity (polars hash joins, sharded) |
| `features.py` | 37 name/address similarity features (rapidfuzz, multiprocessing), blocking evidence, competition context |
| `train.py` | stage 1 (5 out-of-fold LightGBM models) and the original stage 2; decoding and the official macro-F0.5 metric |
| `cluster.py` | cluster-consistency features: similarity of a candidate to the entity's confident matches |
| `sim_drop.py` | leaderboard-like universe: 19 % of Source-1 entities removed so their records become unowned (test density) |
| `stage2_v3.py` | stage 2: cleaned-name, name-frequency, locality-conflict and sibling-support features; trained at test-like density |
| `stage3.py` | stage 3: refinement with features computed from the stage-2 scores of each entity's other candidates |
| `infer.py` | stage-1 scoring of the test set (and the original stage 2) |
| `write_variant.py` | decodes saved scores into `matching_results.tsv` with per-country thresholds |
| `score_f05.py` | the official metric (macro F0.5 per Source-1 entity), standard library only |
| `dense.py` | GPU dense retrieval: bi-encoder fine-tuning, exact cosine search, new candidates, stacker over old + new candidates |
| `gpu_ce.py` | GPU cross-encoders (MiniLM-L-6 / L-12, Apache-2.0): training pairs, fine-tuning, scoring, LightGBM stacker, submission files |
| `make_package.py` | builds `candidate_pairs.tsv` and the submission zip |
| `step1.bat` … `step4.bat`, `reproduce.bat` | the commands below, in order |

## Reproduce end to end (from `src/`)
```bat
reproduce.bat
```
which runs:
```bat
python run.py train 1.0 f2 India US                     & rem training universes: normalise, block, features
python train.py stage1 f2 India,US all                  & rem 5 out-of-fold stage-1 models
python cluster.py train f2 India,US all
set ER_ANC=1& set ER_S2=anc
python train.py stage2 f2 India,US all                  & rem original stage 2 (used for comparison / step 1)
python tune.py f2 India,US all
python run.py test t1                                   & rem test universe: normalise, block, features
python infer.py f2 t1 all --stage1-only                 & rem stage-1 scores for the test set
python cluster.py test t1 France,India,US
python sim_drop.py                                      & rem leaderboard-like validation universe
python stage2_v3.py prep & python stage2_v3.py feats & python stage2_v3.py train & python stage2_v3.py infer
python stage3.py train & python stage3.py infer
set ER_HITS=infer_hits_v4& set ER_TAUJSON=decode_params_all_v4_tl.json
python write_variant.py v4_tl auto+0.1 auto auto        & rem validation-tuned variant
set ER_PMIN=0.02& set ER_TWIN=1                         & rem final: stage-1 filter 0.02 (~5 candidates / entity)
python infer.py f2 t1 all & python stage2_v3.py twinprep & python stage2_v3.py infer & python stage3.py infer
python multi_variant.py final_p020 France=infer_hits_anc_p020:0.85 India=infer_hits_v4_tw_p020:0.75 US=infer_hits_anc_p020:0.7
python gpu_ce.py data & python gpu_ce.py train & python gpu_ce.py score   & rem cross-encoder 1 (MiniLM-L-6)
set ER_CE_TAG=_l12& set ER_CE_BASE=cross-encoder/ms-marco-MiniLM-L-12-v2& set ER_CE_N=1000000
python gpu_ce.py data & python gpu_ce.py train & python gpu_ce.py score   & rem cross-encoder 2 (MiniLM-L-12)
set ER_CE_TAG=& set ER_CE_STACK=,_l12
python gpu_ce.py stack & python gpu_ce.py write                           & rem stacker over both -> output/ce_all_l12
python make_package.py ce_all_l12                       & rem LB 0.970 (needs CUDA torch)
set ER_CE_STACK=& set ER_DN_V2=1                        & rem dense retrieval (bi-encoder) + record-competition stacker
python dense.py data & python dense.py train & python dense.py embed & python dense.py retrieve
python dense.py ce & python dense.py stack
python multi_variant.py dn2_FR095 France=infer_hits_dn2:0.95 India=infer_hits_dn2:0.7 US=infer_hits_dn2:0.75
set ER_DN=1& python make_package.py dn2_FR095           & rem LB 0.977655; candidates include dense-retrieval pairs
src\step10.bat                                          & rem third cross-encoder (L-12 continued on candidate + dense pairs)
python multi_variant.py final_FRgate France=infer_hits_dn2:0.8:0.95 India=infer_hits_dn23:0.8 US=infer_hits_dn23:0.7
set ER_DN=1& python make_package.py final_FRgate        & rem final (LB 0.977678)
```
Wall-clock on the machine above: training universes ≈ 3.5 h, stage 1 ≈ 1 h, test universe ≈ 3 h, test stage 1 ≈ 1 h,
everything after that ≈ 1.5 h.

## Notes
* `country` is never a model feature and nothing is hard-coded to US/India; unseen countries (France) use the same path.
  Localities (cities/regions) are learnt from each country's own Source-1 addresses.
* Validation: 5 folds by hash of the Source-1 id; fold 0 is never used for fitting; stage-2/3 inputs are out-of-fold.
* `candidate_pairs.tsv` lists exactly the pairs scored by the final matching models: blocking candidates that pass the
  stage-1 filter (p1 >= 0.02), about 5 per Source-1 entity.
