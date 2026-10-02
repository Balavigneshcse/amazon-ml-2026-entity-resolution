"""Resumable pipeline driver: universe -> blocking -> features -> context, per country.

    python run.py train 1.0 full India        # build + block + featurise the full-density India training universe
    python run.py test full_test              # same for the whole test set (all countries)

Every stage skips work already on disk, so the command can be interrupted (Ctrl+C) and re-run to continue.
"""
from __future__ import annotations
import sys
from pathlib import Path
import polars as pl
import build_universe, blocking, features
from config import ART, countries as universe_countries


def stages(tag: str, countries: list[str]) -> None:
    """Blocking, recall report (labelled universes only), similarity features and record-competition statistics
    for each country."""
    for c in countries:
        blocking.block_country(tag, c)
        blocking.eval_blocking(tag, c) if (ART / tag / c / "gt.parquet").exists() else None
        features.build_country_feats(tag, c)
        features.build_rstats(tag, c)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train":
        frac, tag, countries = float(sys.argv[2]), sys.argv[3], sys.argv[4:]
        build_universe.build_train(frac, tag, countries or None)
        stages(tag, countries or list(universe_countries(tag)))
    else:
        tag = sys.argv[2]
        build_universe.build_test(tag)
        stages(tag, list(universe_countries(tag)))
