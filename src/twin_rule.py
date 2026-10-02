"""Twin rule (post-processing): drop a whole "twin" group.

A decoy twin business = the reference name + one distinctive extra word, at a nearby house number; its records all carry
the twin's number, but noise removes the extra word from some of them. The model rejects the members that show the word,
not their siblings. Rule: for a Source-1 entity, if a candidate's first house number differs from the entity's and some
candidate with that same number adds a distinctive (non-noise) word, drop every candidate carrying that number.

    python twin_rule.py <src_variant> <new_variant> [--tau France=0.85,India=0.75,US=0.75] (uses infer_hits_v4_tw)
"""
from __future__ import annotations
import os, sys, json
os.environ.setdefault("ER_TWIN", "1")
import polars as pl
from config import ART, DATA, ROOT, countries
import stage2_v3 as V

HITS = os.environ.get("ER_HITS", "infer_hits_v4_tw")

if __name__ == "__main__":
    src, name = sys.argv[1], sys.argv[2]
    prm = json.loads((ART / "f2" / "decode_params_all_v4_tw_tl.json").read_text())
    taus = {c: prm["tau"] if c in countries("f2") else min(0.97, prm["tau"] + 0.1) for c in countries("t1")}  # unseen: +0.1
    if "--tau" in sys.argv:
        taus.update({k: float(v) for k, v in (x.split("=") for x in sys.argv[sys.argv.index("--tau") + 1].split(","))})
    out = ROOT / "output" / name
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for c, tau in taus.items():
        d = ART / "t1" / c
        a, b = V.sup_tables(d)
        nov = pl.read_parquet(d / "nov" / "shard_*.parquet")
        h = pl.read_parquet(d / HITS / "shard_*.parquet")
        # twin groups are judged on ALL plausible candidates (not only accepted ones)
        g = nov.join(a, on="s1", how="left").join(b.select("r", "rn1"), on="r", how="left")
        g = g.with_columns(pl.col("nov_r").max().over("s1", "rn1").alias("grp_nov"))
        twins = g.filter(pl.col("rn1").is_not_null() & (pl.col("rn1") != pl.col("sn1")) & (pl.col("grp_nov") > 0)).select("s1", "r")
        acc = h.filter(pl.col("p2") >= tau)
        acc = acc.filter(pl.col("p2") == pl.col("p2").max().over("r"))
        dropped = acc.join(twins, on=["s1", "r"], how="semi").height
        acc = acc.join(twins, on=["s1", "r"], how="anti")
        ns1 = pl.scan_parquet(d / "s1_*.parquet").select(pl.len()).collect().item()
        print(f"[twin rule] {c}: tau={tau:.2f} dropped {dropped:,} twin pairs -> {acc.height:,} pairs ({acc.height / ns1:.2f} per S1)", flush=True)
        parts.append(acc.group_by("s1").agg(pl.col("r").sort().str.join(",").alias("matched_entity_ids")))
    m = pl.concat(parts).rename({"s1": "source1_entity_id"})
    s1_all = pl.read_csv(DATA / "test" / "test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0, columns=["entity_id"])
    res = (s1_all.rename({"entity_id": "source1_entity_id"}).join(m, on="source1_entity_id", how="left")
           .with_columns(pl.col("matched_entity_ids").fill_null("")))
    res.write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")
    print(f"[twin rule] wrote output/{name}: {res.height:,} rows, non-empty {(res['matched_entity_ids'] != '').sum():,}")
