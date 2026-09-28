"""Per-country submission from any saved scores: each country picks its own model scores and threshold.

    python multi_variant.py <name> France=<hits>:<tau>[:<gate>] India=<hits>:<tau> US=<hits>:<tau>
hits = folder under artifacts/t1/<country>/ (infer_hits_anc = cluster model, infer_hits_v4_tw = v5 twin-aware, ...)
gate (optional) = an entity gets matches only if its best candidate scores >= gate; its other records need >= tau
"""
import sys
import polars as pl
from config import ART, DATA, ROOT

name, spec = sys.argv[1], dict(a.split("=") for a in sys.argv[2:])
out = ROOT / "output" / name
out.mkdir(parents=True, exist_ok=True)
parts = []
for c, hv in spec.items():
    hits, tau, *g = hv.split(":")
    tau, gate = float(tau), float(g[0]) if g else 0.0
    h = pl.read_parquet(ART / "t1" / c / hits / "shard_*.parquet").filter(pl.col("p2") >= tau)
    h = h.filter(pl.col("p2") == pl.col("p2").max().over("r"))
    h = h.filter(pl.col("p2").max().over("s1") >= gate)
    ns1 = pl.scan_parquet(ART / "t1" / c / "s1_*.parquet").select(pl.len()).collect().item()
    print(f"[{name}] {c}: {hits} tau={tau} gate={gate} -> {h.height / ns1:.2f} pairs per S1", flush=True)
    parts.append(h.group_by("s1").agg(pl.col("r").sort().str.join(",").alias("matched_entity_ids")))
m = pl.concat(parts).rename({"s1": "source1_entity_id"})
s1_all = pl.read_csv(DATA / "test" / "test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0, columns=["entity_id"])
res = (s1_all.rename({"entity_id": "source1_entity_id"}).join(m, on="source1_entity_id", how="left")
       .with_columns(pl.col("matched_entity_ids").fill_null("")))
res.write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")
