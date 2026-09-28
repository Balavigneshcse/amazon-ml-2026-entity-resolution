"""Decode-only submission variants (no model runs): per-country thresholds and an optional same-name rescue.

    python write_variant.py <name> <tau_France> <tau_India> <tau_US> [rescue_countries]

Reads the stage-2 scores already saved by infer.py (artifacts/t1/<country>/infer_hits_anc, p2 >= 0.30) and writes
output/<name>/matching_results.tsv.  rescue_countries (e.g. "France"): in those countries a pair with p2 >= 0.30 is
also accepted when the normalised names are identical (typical France miss: same name, house number re-typed).
Each record still goes to its single best Source-1 entity.
"""
from __future__ import annotations
import os, sys, json
import polars as pl
from config import ART, DATA, ROOT

RESCUE_MIN = 0.30
HITS = os.environ.get("ER_HITS", "infer_hits_anc")          # which saved stage-2 scores to decode

if __name__ == "__main__":
    name = sys.argv[1]
    # "auto" = the threshold tuned on the leaderboard-like validation (ER_TAUJSON in artifacts/f2)
    #  "auto+0.15" = that threshold plus an offset (France was over-matching on the leaderboard)
    args = sys.argv[2:5]
    auto = json.loads((ART / "f2" / os.environ.get("ER_TAUJSON", "decode_params_all_v3.json")).read_text())["tau"] if any(a.startswith("auto") for a in args) else None
    taus = {c: (min(0.97, auto + float(v[4:] or 0)) if v.startswith("auto") else float(v)) for c, v in zip(("France", "India", "US"), args)}
    rescue = set(sys.argv[5].split(",")) if len(sys.argv) > 5 and sys.argv[5] else set()
    out = ROOT / "output" / name
    out.mkdir(parents=True, exist_ok=True)
    s1_all = pl.read_csv(DATA / "test" / "test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0, columns=["entity_id"])
    parts = []
    for c, tau in taus.items():
        d = ART / "t1" / c
        h = pl.read_parquet(d / HITS / "shard_*.parquet")
        keep = pl.col("p2") >= tau
        if c in rescue:
            n1 = pl.scan_parquet(d / "s1_*.parquet").select(pl.col("id").alias("s1"), pl.col("name_norm").alias("n1")).collect()
            n2 = pl.scan_parquet(d / "cand_*.parquet").select(pl.col("id").alias("r"), pl.col("name_norm").alias("n2")).collect()
            h = h.join(n1, on="s1", how="left").join(n2, on="r", how="left")
            keep = keep | ((pl.col("p2") >= RESCUE_MIN) & (pl.col("n1") == pl.col("n2")) & (pl.col("n1").str.len_chars() > 0))
        h = h.filter(keep).select("s1", "r", "p2")
        h = h.filter(pl.col("p2") == pl.col("p2").max().over("r"))
        ns1 = pl.scan_parquet(d / "s1_*.parquet").select(pl.len()).collect().item()
        print(f"[variant {name}] {c}: tau={tau}{' +same-name rescue' if c in rescue else ''} -> {h.height:,} pairs ({h.height / ns1:.2f} per S1)", flush=True)
        parts.append(h.group_by("s1").agg(pl.col("r").sort().str.join(",").alias("matched_entity_ids")))
    m = pl.concat(parts).rename({"s1": "source1_entity_id"})
    res = (s1_all.rename({"entity_id": "source1_entity_id"}).join(m, on="source1_entity_id", how="left")
           .with_columns(pl.col("matched_entity_ids").fill_null("")))
    res.write_csv(out / "matching_results.tsv", separator="\t", quote_style="never")
    print(f"[variant {name}] wrote {out / 'matching_results.tsv'}: {res.height:,} rows, non-empty {(res['matched_entity_ids'] != '').sum():,}")
