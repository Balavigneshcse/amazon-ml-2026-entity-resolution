#!/bin/bash
# Waits for the test build (run.py test t1) to finish, then runs inference and the official validator.
# Usage: bash after_build.sh   (from anywhere)
ROOT="C:/Users/Balavignesh K/Documents/Studies/Hackathon/Amazon ML"
ART="$ROOT/business_entity_resolution/artifacts"
cd "$ROOT/business_entity_resolution/src" || exit 1

# wait until the build produced the last country's context stats, or the build process is gone
until [ -f "$ART/t1/US/rstats.parquet" ]; do
  if ! tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -qi python; then
    sleep 20
    [ -f "$ART/t1/US/rstats.parquet" ] || { echo "BUILD ENDED WITHOUT FINISHING"; exit 2; }
  fi
  sleep 60
done
echo "build finished $(date)"
python -u -W ignore infer.py f2 t1 all || { echo "INFER FAILED $(date)"; exit 3; }
echo "infer finished $(date)"
cd "$ROOT"
python -u dataset/student_resource/utils/validate_submission.py \
  --matching business_entity_resolution/output/matching_results.tsv \
  --candidate business_entity_resolution/output/candidate_pairs.tsv \
  --test-dir dataset/student_resource/dataset/test
echo "validator exit $? $(date)"
