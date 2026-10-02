# Business Entity Resolution — Team LOGIC MAKERS

Amazon ML Challenge 2026 — public leaderboard macro F0.5 **0.977678**.

Pipeline: **normalise → multi-key blocking (100 candidates / entity) → stage-1 LightGBM filter (≈ 5 / entity) + GPU dense
retrieval (fine-tuned MiniLM bi-encoder, ≈ 3 more / entity) → cluster-consistency LightGBM + fine-tuned MiniLM
cross-encoders → LightGBM stacker → one-owner decoding** (countries without training labels — France in this test set —
use two-threshold decoding).
No external data, APIs or geocoding are used. Pretrained weights: `cross-encoder/ms-marco-MiniLM-L-6-v2`,
`cross-encoder/ms-marco-MiniLM-L-12-v2`, `sentence-transformers/all-MiniLM-L6-v2` (all Apache-2.0, ≤ 33 M parameters),
fine-tuned on the training data only; LightGBM (MIT). Method and results: `Documentation_template.md` (in the submission
zip) / `Documentation.md` (in the repository).

## Environment
* Python 3.12. Install the CUDA build of PyTorch first, then the pinned packages:
  `pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126` and `pip install -r requirements.txt`.
* Tested on Windows 11, 12 CPU cores, 15 GB RAM, RTX 3050 4 GB (GPU needed for the transformer steps), ~150 GB free disk.
* The three pretrained models are downloaded once from the Hugging Face Hub on first use (model weights only).
* Data is expected at `../dataset/student_resource/dataset/{train,test}` relative to this folder; override with the
  environment variables `ER_DATA` (dataset folder) and `ER_ART` (folder for intermediate files, default `artifacts/`).
* Every stage writes its results shard by shard and skips finished work, so an interrupted run is resumed by running the
  same command again.

## Source files (`src/`)
| file | role |
|---|---|
| `config.py` | paths, seeds, candidates per entity (`KBLOCK = 100`), `countries()` — the country list read from the data |
| `normalize.py` | name/address normalisation: accent folding, transliteration (Unidecode), legal-suffix stripping, abbreviation and state canonicalisation, phonetic squash, consonant skeleton |
| `build_universe.py` | normalised per-country universe for train (with ground truth) and test; countries come from the `country` column |
| `run.py` | driver: universe → blocking → features → record-competition statistics, per country |
| `blocking.py` | candidate generation: 9 key families, idf-weighted cosine ranking, top-100 per entity (polars hash joins, sharded) |
| `features.py` | 37 name/address similarity features (rapidfuzz, multiprocessing), blocking evidence, competition context |
| `features_extra.py` | optional extra name/address features (`ER_X=1`; not used in the final run) |
| `train.py` | stage 1 (5 out-of-fold LightGBM models) and the original stage 2; decoding and the official macro-F0.5 metric |
| `cluster.py` | cluster-consistency features: similarity of a candidate to the entity's confident matches |
| `tune.py` | decoding threshold / singleton gate chosen on validation |
| `sim_drop.py` | leaderboard-like validation universe: 19 % of Source-1 entities removed so their records become unowned |
| `stage2_v3.py` | stage 2: cleaned-name, name-frequency, locality-conflict, sibling-support and twin features; trained at test-like density |
| `stage3.py` | stage 3: refinement with features computed from the stage-2 scores of each entity's other candidates |
| `infer.py` | stage-1 (and original stage-2) scoring of the test set |
| `gpu_ce.py` | GPU cross-encoders (MiniLM-L-6 / L-12): training pairs, fine-tuning, scoring, LightGBM stacker |
| `dense.py` | GPU dense retrieval: bi-encoder fine-tuning, exact cosine search, new candidates, stacker over old + new candidates, third cross-encoder data |
| `multi_variant.py` | decodes saved scores into `output/<name>/matching_results.tsv` (per-country threshold, optional singleton gate, `default=` for countries not listed) |
| `make_package.py` | builds `candidate_pairs.tsv` and the submission zip, runs the official validator |
| `score_f05.py` | the official metric (macro F0.5 per Source-1 entity), standard library only |
| `write_variant.py`, `write_hybrid.py`, `twin_rule.py` | decoding experiments used during the challenge (not part of the final run) |
| `reproduce.bat` | the whole pipeline below, in order; `step*.bat`, `run_anc.bat`, `probes.bat` are the same commands split into the steps we ran |

## Reproduce end to end
From `src/`, run `reproduce.bat`. It executes exactly these commands (`auto` = every country found in the data; `default=`
= the setting for every test country that has no training labels):
```bat
python run.py train 1.0 f2                              & rem training universes: normalise, block, features
python train.py stage1 f2 auto all                      & rem 5 out-of-fold stage-1 models
python cluster.py train f2 auto all
set ER_ANC=1& set ER_S2=anc
python train.py stage2 f2 auto all
python tune.py f2 auto all
python run.py test t1                                   & rem test universe: normalise, block, features
python infer.py f2 t1 all --stage1-only                 & rem stage-1 scores for the test set
python cluster.py test t1 auto
python sim_drop.py                                      & rem leaderboard-like validation universe
python stage2_v3.py prep & python stage2_v3.py feats & python stage2_v3.py train & python stage2_v3.py infer
python stage3.py train & python stage3.py infer
set ER_HITS=infer_hits_v4& set ER_TAUJSON=decode_params_all_v4_tl.json
python write_variant.py v4_tl auto+0.1 auto auto
set ER_TWIN=1
python stage2_v3.py twinprep & python stage2_v3.py train & python stage3.py train
set ER_PMIN=0.02                                        & rem matching models only score candidates with p1 >= 0.02
python infer.py f2 t1 all & python stage2_v3.py infer & python stage3.py infer
python multi_variant.py final_p020 default=infer_hits_anc_p020:0.85 India=infer_hits_v4_tw_p020:0.75 US=infer_hits_anc_p020:0.7
python gpu_ce.py data & python gpu_ce.py train & python gpu_ce.py score   & rem cross-encoder 1 (MiniLM-L-6)
set ER_CE_TAG=_l12& set ER_CE_BASE=cross-encoder/ms-marco-MiniLM-L-12-v2& set ER_CE_N=1000000
python gpu_ce.py data & python gpu_ce.py train & python gpu_ce.py score   & rem cross-encoder 2 (MiniLM-L-12)
set ER_CE_TAG=& set ER_CE_STACK=,_l12
python gpu_ce.py stack & python gpu_ce.py write                           & rem stacker over both -> output/ce_all_l12
set ER_CE_STACK=& set ER_DN_V2=1                        & rem dense retrieval + record-competition stacker
python dense.py data & python dense.py train & python dense.py embed & python dense.py retrieve
python dense.py ce & python dense.py stack
python multi_variant.py dn2_FR095 default=infer_hits_dn2:0.95 India=infer_hits_dn2:0.7 US=infer_hits_dn2:0.75
call step10.bat                                         & rem cross-encoder 3 (L-12 continued on candidate + dense pairs)
python multi_variant.py final_FRgate default=infer_hits_dn2:0.8:0.95 India=infer_hits_dn23:0.8 US=infer_hits_dn23:0.7
set ER_DN=1& python make_package.py final_FRgate        & rem final files + zip (LB 0.977678)
```
`step10.bat` runs: `dense.py trainpairs`, `dense.py cedata`, then with `ER_CE_TAG=_v3` and
`ER_CE_BASE=@art/ce_model_l12/final`: `gpu_ce.py train`, `gpu_ce.py score`, then with `ER_CE_TAG=`, `ER_DN_CE=,_l12,_v3`,
`ER_DN_V2=1`: `dense.py ce`, `dense.py stack`, `dense.py write`.

On Linux/macOS run the same Python commands in the same order from `src/`, with `export VAR=value` instead of
`set VAR=value`. The final files are `output/final_FRgate/matching_results.tsv` and the zip in `package/`.

Wall-clock on the machine above: training universes ≈ 3.5 h, stage 1 ≈ 1 h, test universe ≈ 3 h, test stage 1 ≈ 1 h,
stages 2–3 ≈ 1.5 h, cross-encoders 1–2 ≈ 4.5 h, dense retrieval ≈ 4.2 h, cross-encoder 3 ≈ 4 h.

## Notes
* `country` is never a model feature. The list of countries is read from the data (`config.countries`); the matching
  thresholds of the labelled countries (India, US) are tuned on their validation folds, and every country without training
  labels (France) gets the unseen-country setting through `default=`. Localities are learnt from each country's own
  Source-1 addresses; `normalize.py` contains small abbreviation and state-name tables per address convention.
* Validation: 5 folds by hash of the Source-1 id; fold 0 is never used for fitting; stage-2/3 inputs are out-of-fold;
  the stackers are evaluated with 2-fold cross-fitting inside the validation fold.
* `candidate_pairs.tsv` lists exactly the pairs scored by the final matching models: blocking candidates that pass the
  stage-1 filter (p1 >= 0.02) plus the dense-retrieval candidates (`ER_DN=1`), about 8.4 per Source-1 entity.
