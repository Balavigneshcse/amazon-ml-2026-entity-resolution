"""Paths and global parameters. Override with env vars ER_DATA / ER_ART to move the run (e.g. to a cloud box)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ER_DATA", ROOT.parent / "dataset" / "student_resource" / "dataset"))
ART = Path(os.environ.get("ER_ART", ROOT / "artifacts"))
SEED = 42
N_SHARDS = 16          # blocking shards per country (bounds peak memory)
VAL_PCT = 20           # % of sampled Source-1 entities held out for validation
KBLOCK = 100         # candidates kept per Source-1 entity (ranked later by the model)
USE_ANC = os.environ.get("ER_ANC", "0") == "1"                                   # cluster-consistency features in stage 2
S2TAG = ("_" + os.environ["ER_S2"]) if os.environ.get("ER_S2") else ""             # suffix for stage-2 model / validation / thresholds
PMIN = float(os.environ.get("ER_PMIN", "0.005"))   # stage-1 filter: pairs below this never reach the matching models
HSFX = "" if abs(PMIN - 0.005) < 1e-9 else f"_p{int(round(PMIN * 1000)):03d}"   # suffix of score folders for other filters
