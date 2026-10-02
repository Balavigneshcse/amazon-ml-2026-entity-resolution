@echo off
REM France probes (decode only, ~5 minutes, no training). Writes two extra submission files + validates them.
cd /d "%~dp0"
python -u -W ignore write_variant.py probe_fr_low 0.5 0.7 0.7
if errorlevel 1 goto fail
python -u -W ignore write_variant.py probe_fr_rescue 0.7 0.7 0.7 France
if errorlevel 1 goto fail
cd /d "%~dp0..\.."
for %%V in (probe_fr_low probe_fr_rescue) do (
  echo ===== validating %%V
  python -u dataset\student_resource\utils\validate_submission.py --matching business_entity_resolution\output\%%V\matching_results.tsv --test-dir dataset\student_resource\dataset\test
)
echo ===== PROBES DONE
exit /b 0
:fail
echo ===== PROBES STOPPED WITH AN ERROR
exit /b 1
