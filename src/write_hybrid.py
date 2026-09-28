"""Hybrid submission: take each country's matches from a chosen existing output file.

    python write_hybrid.py <name> France=<dir> India=<dir> US=<dir>      (dirs under output/)
"""
import sys
import polars as pl
from config import ART, DATA, ROOT

name, spec = sys.argv[1], dict(a.split("=") for a in sys.argv[2:])
out = ROOT / "output" / name
out.mkdir(parents=True, exist_ok=True)
parts = []
for c, src in spec.items():
    ids = pl.scan_parquet(ART / "t1" / c / "s1_*.parquet").select(pl.col("id").alias("source1_entity_id")).collect()
    m = pl.read_csv(ROOT / "output" / src / "matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0).fill_null("")
    parts.append(m.join(ids, on="source1_entity_id", how="semi"))
    print(f"[hybrid {name}] {c}: from {src}")
res = pl.concat(parts)
s1_all = pl.read_csv(DATA / "test" / "test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0, columns=["entity_id"])
res = s1_all.rename({"entity_id": "source1_entity_id"}).join(res, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
assert res.height == s1_all.height
res.write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")
print(f"[hybrid {name}] wrote {res.height:,} rows, non-empty {(res['matched_entity_ids'] != '').sum():,}")
