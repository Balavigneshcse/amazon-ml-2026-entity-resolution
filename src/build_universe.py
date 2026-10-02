"""Build a normalised universe per country from the training data (a hash-sampled fraction, or 1.0 = everything)
or from the full test set.

Train universe: a fraction `frac` of Source-1 entities (deterministic hash of the id), ALL of their true
Source-2/3 records, plus the same fraction of the unowned ("distractor") Source-2/3 records. Matches never cross
countries, so everything is split by country. Outputs under artifacts/<tag>/<country>/ : s1_XXX.parquet,
cand_XXX.parquet (normalised, chunked), gt.parquet (train only). Resumable: finished countries are skipped.

Usage:  python build_universe.py train <frac> <tag> [countries...]      |      python build_universe.py test <tag>
"""
from __future__ import annotations
import sys, time
import polars as pl
from config import DATA, ART, SEED, VAL_PCT
from normalize import normalize

RD = dict(separator="\t", quote_char=None, infer_schema_length=0)
CHUNK = 1_500_000


def _scan(split: str, i: int) -> pl.LazyFrame:
    """Lazy reader for <split>_source<i>.tsv with the shared tab-separated read options (RD)."""
    return pl.scan_csv(DATA / split / f"{split}_source{i}.tsv", **RD)


def _h(col: str, seed: int) -> pl.Expr:
    """Deterministic hash bucket 0-9999 of a column (reproducible sampling)."""
    return pl.col(col).hash(seed) % 10000


def _write_chunks(df: pl.DataFrame, d, name: str, country: str, is_s1: bool) -> None:
    """Normalise records in chunks of CHUNK rows and write <name>_XXX.parquet; Source-1 rows get the is_val flag
    (VAL_PCT % of entities, by id hash). Chunks already on disk are skipped."""
    for k, off in enumerate(range(0, df.height, CHUNK)):
        f = d / f"{name}_{k:03d}.parquet"
        if f.exists():
            continue
        n = normalize(df.slice(off, CHUNK), country)
        if is_s1:
            n = n.with_columns((pl.col("id").hash(SEED + 1) % 100 < VAL_PCT).alias("is_val"))
        n.write_parquet(f)


def build_train(frac: float, tag: str, only: list[str] | None = None) -> None:
    """Build the per-country training universes: normalised Source-1 and candidate (Source-2 + Source-3) chunks and
    the true pairs (gt.parquet). frac < 1 samples Source-1 entities (their records plus the same share of
    unowned records); only restricts the countries. Finished countries are marked _DONE and skipped."""
    out = ART / tag
    thr = int(frac * 10000)
    t0 = time.time()
    gt = pl.scan_csv(DATA / "train" / "train_ground_truth.tsv", **RD).with_columns(pl.col("matched_entity_ids").fill_null(""))
    pairs_all = (gt.filter(pl.col("matched_entity_ids") != "")
                 .with_columns(pl.col("matched_entity_ids").str.split(",").alias("r"))
                 .explode("r").select(pl.col("source1_entity_id").alias("s1"), "r"))
    s1_all = _scan("train", 1)
    s1_sel = s1_all if frac >= 1 else s1_all.filter(_h("entity_id", SEED) < thr)
    cand_all = pl.concat([_scan("train", 2), _scan("train", 3)])
    countries = sorted(s1_all.select("country").unique().collect()["country"].to_list())
    for country in countries:
        if only and country not in only:
            continue
        d = out / country
        if (d / "_DONE").exists():
            print(f"[{tag}] {country}: already done, skipping")
            continue
        d.mkdir(parents=True, exist_ok=True)
        s1c = s1_sel.filter(pl.col("country") == country).collect()
        ids = s1c.select(pl.col("entity_id").alias("s1"))
        pairs_c = pairs_all.join(ids.lazy(), on="s1", how="semi").collect()
        cc = cand_all.filter(pl.col("country") == country)
        if frac < 1:
            owned = cc.join(pairs_c.lazy().select(pl.col("r").alias("entity_id")), on="entity_id", how="semi")
            unowned = (cc.join(pairs_all.select(pl.col("r").alias("entity_id")), on="entity_id", how="anti")
                       .filter(_h("entity_id", SEED + 7) < thr))
            cc = pl.concat([owned, unowned])
        cc = cc.collect()
        print(f"[{tag}] {country}: S1={s1c.height:,} cand={cc.height:,} pairs={pairs_c.height:,} ({time.time()-t0:.0f}s)", flush=True)
        pairs_c.write_parquet(d / "gt.parquet")
        _write_chunks(s1c, d, "s1", country, True)
        del s1c
        _write_chunks(cc, d, "cand", country, False)
        (d / "_DONE").write_text("ok")
        print(f"[{tag}] {country}: normalised ({time.time()-t0:.0f}s)", flush=True)


def build_test(tag: str) -> None:
    """Normalise the full test set per country, chunked. Resumable per country."""
    out = ART / tag
    s1 = pl.scan_csv(DATA / "test" / "test_source1.tsv", **RD)
    cand = pl.concat([pl.scan_csv(DATA / "test" / f"test_source{i}.tsv", **RD) for i in (2, 3)])
    countries = sorted(s1.select("country").unique().collect()["country"].to_list())
    print("test countries:", countries, flush=True)
    t0 = time.time()
    for country in countries:
        d = out / country
        if (d / "_DONE").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        for name, lf, is_s1 in (("s1", s1, True), ("cand", cand, False)):
            df = lf.filter(pl.col("country") == country).collect()
            print(f"[test] {country} {name}: {df.height:,} rows ({time.time()-t0:.0f}s)", flush=True)
            _write_chunks(df, d, name, country, is_s1)
            del df
        (d / "_DONE").write_text("ok")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        build_train(float(sys.argv[2]), sys.argv[3], sys.argv[4:] or None)
    else:
        build_test(sys.argv[2] if len(sys.argv) > 2 else "test")
