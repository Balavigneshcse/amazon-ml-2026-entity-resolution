# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** LOGIC MAKERS  
**Team Members:** Dinesh Karthick R (team leader), Arunkumar G, Deepak Pichaimuthu, Balavignesh K — VSB Engineering College, Karur, Tamil Nadu  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We treat entity resolution as **candidate generation (keys + GPU dense retrieval) + learned filtering + GPU cross-encoders
combined with gradient-boosted pair models + one-owner decoding**. Multi-key blocking keeps 100 candidates per Source-1 entity; a
stage-1 LightGBM filter cuts them to ≈ 5 per entity, and a fine-tuned bi-encoder (dense retrieval) adds ≈ 3 more that the keys
missed (≈ 8 per entity in `candidate_pairs.tsv`). Two small pretrained transformer cross-encoders
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, 22 M parameters, and `ms-marco-MiniLM-L-12-v2`, 33 M parameters, both Apache-2.0),
fine-tuned on a laptop RTX 3050, read both records' "name | address" text jointly; a LightGBM stacker combines their scores
with our feature-based matching model.
Each Source-2/3 record is then given to its single best Source-1 entity above an F0.5-tuned threshold.
Public leaderboard: **0.977678** (without dense retrieval: 0.970; one cross-encoder: 0.9677; feature models alone: 0.9462).
Held-out validation: India 0.987, US 0.987. No external data or lookup services are used; the only pretrained weights are
Apache-2.0 MiniLM models (three cross-encoders, one bi-encoder).

---

## 2. Methodology

### 2.1 Problem Analysis
Training data: 2.2 M Source-1, 5.0 M Source-2, 5.3 M Source-3 records (test: 1.7 M / 4.9 M / 5.1 M, plus France).
* **Partition structure:** every Source-2/3 record belongs to at most one Source-1 entity (7,638,365 matched pairs =
  7,638,365 distinct records); no match crosses countries; 5.6 % of entities are singletons; 26 % of records are unowned.
* **Heavy noise:** only 22 % of true pairs share an identical normalised name and 9 % an identical address. Patterns:
  domain-style names, word transposition, typos, accent noise, legal-suffix variants, Indic-script names (≈ 7 %), missing
  addresses (≈ 4 %), re-typed / truncated house numbers, abbreviations, component re-ordering, DBA names.
* **Source structure (key finding):** each source re-writes an entity's address once and all that source's copies inherit
  it — if a candidate's house number differs from the reference but is *shared by other candidates of the same entity*,
  it is a true match 89–99 % of the time (vs 29–51 % when the differing number is unique).
* **Test shift (found with leaderboard feedback):** the test set has ~5.8 Source-2/3 records per entity vs 4.7 in train,
  i.e. about twice as many unowned records, and 2–4× more "near-twin" distractors (same name, nearby house number, owner
  absent from Source 1). France (unseen in training) over-matched until its threshold was raised.

### 2.2 Solution Strategy
**Approach type:** Blocking + 3-stage LightGBM cascade + partition-aware decoding.  
**Final configuration:** stage-1 filter p1 ≥ 0.02 (≈ 5 candidates per entity) + dense-retrieval candidates (≤ 5 new per
entity above a similarity floor) → GPU cross-encoders + cluster-consistency stage-2 model → LightGBM stacker with dense
similarity and record-competition features. India/US: stacker over three cross-encoders (τ 0.80 India, 0.70 US). France
(unseen in training): stacker over two cross-encoders with two-threshold decoding — an entity is matched only if its best
candidate scores ≥ 0.95, and then its other records need ≥ 0.80.  
**Core innovations:** (1) sibling-support features exploiting the per-source address re-write; (2) a leaderboard-like
validation universe (19 % of entities removed so their records become unowned) used for training stage 2/3 and choosing
thresholds; (3) country-agnostic locality conflict learnt from each country's own addresses; (4) a test-like metric in
which false matches on unowned records are weighted to the test's distractor density.

---

## 3. Candidate Generation (Blocking)
* **Normalisation** (`normalize.py`): NFKD accent folding, ligatures, Indic → Latin transliteration (Unidecode), removal of
  legal suffixes (also their transliterated spellings), domain-name handling, phonetic squash (ph→f, c→k, doubled letters),
  consonant skeletons, address tokenisation with abbreviation expansion and state-name → code canonicalisation.
* **Keys** (`blocking.py`, 9 families, 64-bit hashed): name tokens; skeleton tokens; 5-char name prefix; whole concatenated
  name; name anagram (swapped-word domain names); sorted pairs of name words; rare address tokens; house number × address
  token; house number × name token (last 4 / last 3 digits, robust to truncation).
* **Ranking:** keys held by > 0.05 % of a country's records are dropped; pairs are ranked by idf-weighted cosine of shared
  keys; the top **100 per entity** are kept (≈ 173 M test pairs). Sharded by entity, every shard saved (resumable).
* **Stage-1 filter (learned blocking):** a LightGBM on the pair features keeps pairs with p1 ≥ 0.02 — **≈ 5 candidates per
  Source-1 entity** (France 6.8, India 5.1, US 4.8) instead of 100, a reduction ratio of ≈ 99.99995 % versus all pairs. On
  held-out entities it keeps 99.3–99.6 % of the true pairs that blocking found. Every final model scores only these pairs
  (including the per-record competition statistics), and `candidate_pairs.tsv` lists exactly them.
* **Recall (held-out, full density):** India 96.0 %, US 97.4 % of true pairs are among the 100 candidates.
* **Dense retrieval (GPU, `dense.py`):** the remaining misses are mostly phonetic transliterations ("praaivett limittedd"),
  domain-style names and re-typed addresses. A bi-encoder (`sentence-transformers/all-MiniLM-L6-v2`, 22 M parameters,
  Apache-2.0) is fine-tuned on 1.76 M true training pairs (pairs missed by blocking counted twice) with in-batch negatives
  (symmetric InfoNCE, scale 20, batch 256, other records of the same entity masked). Every entity's normalised
  "name | address" is compared with every record of its country by exact cosine search on the GPU; up to 5 new pairs per
  entity above a similarity floor (keeping 97 % of recoverable true pairs on validation) join the candidates. Candidate
  recall after the filter rises from 95.4 % to 98.7 % (India) and from 97.0 % to 99.1 % (US); the perfect-matcher ceiling of
  macro F0.5 rises from 0.983 / 0.990 to 0.996 / 0.997.

## 4. Matching Model
**Stage 1 — pair similarity (59 features):**
* Name: Levenshtein ratio, token-sort / token-set / partial / WRatio, Jaro-Winkler of the concatenated core, exact core
  equality, token Jaccard / containment, skeleton Jaccard / ratio, lengths, domain flags, first-token and prefix equality.
* Address (NaN if missing): token-set / sort / partial ratios, alpha-token overlap, numeric-token Jaccard, first-number
  equality, last-4 / last-3 digit matches, last-token equality, missing flags.
* Blocking evidence and competition context (score, cosine, rank, per-family weights; strength relative to the entity's
  best candidate and to other entities competing for the same record).
* 5 out-of-fold models (fold 0 never used for fitting), so every downstream score is out-of-fold.

**Stage 2 — precision features (102 features), trained at test-like density:**
* stage-1 score, within-entity rank/gap/sum, cross-entity competition for the record;
* cluster consistency: similarity to the entity's confident matches;
* cleaned-name comparison (country/ID noise tokens removed), idf-weighted coverage of each side's tokens, rarest missing
  token, number of entities / records sharing the exact cleaned name (generic names need more evidence);
* locality agreement: localities are address components that recur ≥ 5 times in a country's Source-1 addresses; features
  count shared / unmatched localities and flag a conflict when each side names a place the other lacks (same region,
  different city) — no gazetteer, works for France;
* sibling support: whether the record's house number / address / cleaned name is shared by other candidates of the entity
  (counts and stage-1-score-weighted sums), and whether the reference's own number is supported.

**Twin detection (India model):** decoy "twin" businesses are the reference name plus one distinctive extra word at a nearby
house number. Words that genuine copies add are learnt from true training pairs (661-word noise vocabulary: center, services,
dba, formerly, …); features count distinctive words a record adds, reference words it lacks, and whether any candidate sharing
the record's house number adds a distinctive word. Leave-one-country-out F0.5 rises from 0.930 to 0.943 with these features.

**GPU cross-encoder (final model):** `cross-encoder/ms-marco-MiniLM-L-6-v2` (6-layer MiniLM, 22 M parameters, Apache-2.0)
fine-tuned as a binary pair classifier: input = `"name | address"` of the Source-1 record and of the candidate (max 96
tokens), BCE loss, AdamW (lr 3e-5, warm-up 500 steps, linear decay), batch 64, fp16, 1 epoch over 1.2 M pairs sampled from
the filtered candidates of the training folds only (≈ 28 min on an RTX 3050 4 GB). It learns transliterations, typos,
acronyms, legal-form and "extra distinctive word" differences directly from text. Inference scores only the ≈ 5 filtered
candidates per entity (≈ 9 M test pairs, ≈ 50 min).
**Second cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-12-v2` (12 layers, 33 M parameters, Apache-2.0), same recipe on
2 M training pairs (1 M per country, 86 min fine-tuning, ≈ 90 min test scoring); its score features are added to the stacker.
**Stacker:** LightGBM on [stage-1 p1, cluster-model p2, each cross-encoder's score and its rank / max / gap / sum within the
entity's candidates, dense similarity / rank / gap, a new-candidate flag, and record-side competition (how many entities'
stage-1 scores compete for the same record, its rank among them and the best competing score)]; fitted on the held-out validation fold with 2-fold cross-fitting by entity for honest evaluation.

**Stage 3 — refinement (114 features):** the same inputs plus features recomputed from the stage-2 scores of the entity's
other candidates (score-weighted sibling support, rank, share, gap). Training rows use out-of-fold stage-2 scores.

**Model type:** LightGBM binary (127 leaves, learning rate 0.1, 300–400 rounds, feature/bagging fraction 0.8); one model for
all countries; `country` is never a feature.  
**Decoding and thresholds:** each record goes to its highest-scoring entity; pairs are kept if the score ≥ τ. τ is chosen on
the leaderboard-like validation with the test-like metric (τ = 0.8); France uses τ + 0.1 because leaderboard feedback showed
it over-matches.

## 5. Results & Error Analysis
Validation = held-out entities at test-like density (19 % of entities removed). "Test-like" additionally weights false
matches on unowned records ×8 to mimic the test's distractor density.

| model | τ | India plain | US plain | India test-like | US test-like |
|---|---|---|---|---|---|
| stage 1 + original stage 2 | 0.70 | 0.9404 | 0.9602 | – | – |
| + precision features (stage 2 v3) | 0.65 | 0.9588 | 0.9695 | – | – |
| stage 2 v3 | 0.80 | 0.9568 | 0.9684 | 0.9505 | 0.9655 |
| + refinement (stage 3) | 0.80 | 0.9585 | 0.9692 | 0.9525 | 0.9665 |
| cluster model + GPU cross-encoder L-6 (stacker) | 0.70 | 0.9721 | 0.9791 | – | – |
| + second cross-encoder L-12 (stacker) | 0.75 | 0.9746 | 0.9804 | – | – |
| + dense-retrieval candidates | 0.70 / 0.75 | 0.9853 | 0.9859 | – | – |
| + record-competition features | 0.70 / 0.75 | 0.9860 | 0.9863 | – | – |
| **+ third cross-encoder — final (India/US)** | 0.80 / 0.70 | **0.9871** | **0.9871** | – | – |

Public leaderboard (macro F0.5): 0.934 (first pipeline) → 0.940 (cluster consistency + test-like threshold) → 0.941
(stricter France threshold) → 0.946 (India from the twin-aware cascade) → 0.946168 (≈ 5 candidates per entity) → 0.967657 (GPU cross-encoder + stacker) → 0.970 (two cross-encoders + stacker) → 0.977 (+ dense retrieval) →
0.977655 (+ record competition, France τ 0.95) → **0.977678 (third cross-encoder for India/US, France two-threshold decoding)**.
Per-country tests on the leaderboard: twin-aware cascade for India +0.005, for US −0.012, for France −0.001.

* **False positives:** near-twin businesses (same name, house number ±1–20, different legal form) whose own entity is
  absent from Source 1; co-located businesses; generic names in the same city.
* **Error balance (final, validation):** precision 99.6 %; the loss is mostly missed matches — records with no address whose
  generic name is shared by several entities (not resolvable from name + address), plus ≈ 1.3 % of true pairs that neither the
  keys nor dense retrieval propose.

* **Lesson:** features that exploit the per-source address re-write raised validation F0.5 but lowered the leaderboard for the
  US, because the test set contains 2–4× more near-twin decoys than the training data. Per-country evaluation on the
  leaderboard was needed to pick the model for each country.

## 6. Conclusion
A blocking + three-stage LightGBM cascade reaches ≈ 0.96 macro-F0.5 on held-out entities using only the provided data and a
laptop CPU. The biggest gains came from understanding the data-generation structure (per-source address re-writes) and from
validating at the test set's distractor density rather than the training density. Remaining headroom is blocking recall
and near-twin distractors.

## Appendix
### A. Code Artefacts
See `code/business_entity_resolution/README.md`. Entry point: `src/reproduce.bat` (runs every stage in order and builds
`output/matching_results.tsv`, `output/candidate_pairs.tsv` and the zip). Key modules: `blocking.py`, `features.py`,
`train.py`, `sim_drop.py`, `stage2_v3.py`, `stage3.py`, `gpu_ce.py`, `dense.py`, `multi_variant.py`, `make_package.py`.

### B. Additional Results
* Ideas tested and rejected: expanding candidates by identical address (only 25–35 % precise); per-rank thresholds (+0.0003);
  up-weighting unowned negatives (equivalent to raising the threshold).
* A third cross-encoder (the L-12 model further fine-tuned on candidate + dense-retrieval pairs) raised validation to 0.9871 /
  0.9871; used everywhere it lowered the leaderboard (0.97714 vs 0.97766) because France predictions moved ten times more than
  India/US, so it is used for India/US only.
* France (no labels) is the main remaining gap: the data generator is identical for India and US (5.6 % of entities without
  matches, 13.0 % / 12.1 % without a Source-2 / Source-3 match), and France at τ 0.85 matches too many would-be singletons while
  τ 0.95 drops members. A two-threshold rule (entity needs a best score ≥ 0.95, its other records ≥ 0.80) reproduces the
  generator's structure for France (`multi_variant.py` supports it as `hits:tau:gate`); it is part of the final submission.
* Per-source structure features in the stacker (+0.0001) and France street-name / house-number rules (mismatches were mostly
  typos or genuinely different businesses) were tested and not used.
* Leave-one-country-out (fit on US, score India) shows ≈ 3 points lower F0.5 on an unseen country, which motivated the
  country-agnostic features and the stricter France threshold.
