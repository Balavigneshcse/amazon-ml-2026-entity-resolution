"""Test-time inference: blocking output -> stage-1 -> stage-2 -> decoding -> the two submission files.

    python infer.py <train_tag> <test_tag> [mname]

Requires: run.py test <test_tag> (universe, blocks, features, rstats) and train.py stage1/stage2 + tune.py on the
training tag. Resumable per shard; writes output/matching_results.tsv and output/candidate_pairs.tsv.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import polars as pl
import lightgbm as lgb
from config import ART, DATA, ROOT, S2TAG, PMIN, HSFX
from features import add_context
from train import FEATS1, FEATS2, add_stage2, rp_stats, with_x, with_anc

OUT = ROOT / "output"
KEEP_MIN = 0.30      # rows below this p2 can never be selected (tau >= 0.4), so they are dropped early


def _countries(test_tag):
    """Countries of the test universe (one folder per country under artifacts/<tag>)."""
    return sorted(p.name for p in (ART / test_tag).iterdir() if p.is_dir())


def predict_country(train_tag: str, test_tag: str, mname: str, c: str, stage1_only: bool = False) -> None:
    """Test-time scoring of one country: stage-1 p1 for every blocking candidate, record-competition
    statistics, then (unless stage1_only) the stage-2 model on the candidates that pass the filter."""
    d = ART / test_tag / c
    md = ART / train_tag / f"models_{mname}"
    m1 = lgb.Booster(model_file=str(md / "model_s1_k0.txt"))
    rst = pl.read_parquet(d / "rstats.parquet")
    p1d, hitd = d / "infer_p1", d / f"infer_hits{S2TAG}{HSFX}"
    p1d.mkdir(exist_ok=True)
    hitd.mkdir(exist_ok=True)
    shards = sorted((d / "feats").glob("shard_*.parquet"))
    for f in shards:                                   # stage 1
        o = p1d / f.name
        if o.exists():
            continue
        t0 = time.time()
        df = add_context(with_x(pl.read_parquet(f), d, f.name), rst)
        p = m1.predict(df.select(FEATS1).to_numpy(), num_threads=12)
        df.with_columns(pl.Series("p1", p.astype(np.float32))).write_parquet(o)
        print(f"[infer] {c} p1 {f.name} ({time.time()-t0:.0f}s)", flush=True)
    rp_path = d / f"rp_infer{HSFX}.parquet"
    if not rp_path.exists():
        st = (pl.scan_parquet(p1d / "shard_*.parquet").select("r", "p1").filter(pl.col("p1") >= (PMIN if HSFX else 0.0)).group_by("r")
              .agg(pl.col("p1").sort(descending=True).head(2).alias("t"), (pl.col("p1") > 0.5).sum().cast(pl.UInt16).alias("n_gt50_r"),
                   pl.col("p1").sum().alias("p_sum_r")).collect(engine="streaming"))
        st.select("r", "n_gt50_r", "p_sum_r", *[pl.col("t").list.get(i, null_on_oob=True).fill_null(0.0).alias(f"pt{i+1}") for i in range(2)]).write_parquet(rp_path)
    if stage1_only:                                    # stage-1 scores + record competition stats only
        return
    m2 = lgb.Booster(model_file=str(md / f"model_s2{S2TAG}.txt"))
    rp = pl.read_parquet(rp_path)
    for f in shards:                                   # stage 2
        o = hitd / f.name
        if o.exists():
            continue
        t0 = time.time()
        df = pl.read_parquet(p1d / f.name)
        if HSFX:                                        # the matching model only sees filtered candidates
            df = df.filter(pl.col("p1") >= PMIN)
        df = with_anc(add_stage2(df, rp), d, mname, f.name, "infer_anc")
        p2 = m2.predict(df.select(FEATS2).to_numpy(), num_threads=12)
        df.select("s1", "r").with_columns(pl.Series("p2", p2.astype(np.float32))).filter(pl.col("p2") >= KEEP_MIN).write_parquet(o)
        print(f"[infer] {c} p2 {f.name} ({time.time()-t0:.0f}s)", flush=True)


def candidate_lists(test_tag: str, c: str) -> pl.DataFrame:
    """(s1, candidate_entity_ids) from the blocking shards: exactly the pairs fed to the model."""
    d = ART / test_tag / c
    parts = []
    for f in sorted((d / "feats").glob("shard_*.parquet")):
        parts.append(pl.read_parquet(f, columns=["s1", "r"]).group_by("s1").agg(pl.col("r").sort().str.join(",").alias("candidate_entity_ids")))
    return pl.concat(parts)


def write_outputs(test_tag: str, mname: str, train_tag: str) -> None:
    # ER_DECODE=<suffix> picks thresholds tuned on another validation (e.g. d19 = leaderboard-like density)
    """Decode the stage-2 scores (threshold, one owner per record, singleton gate) into matching_results.tsv
    (and candidate_pairs.tsv) for every test country."""
    sfx = ("_" + os.environ["ER_DECODE"]) if os.environ.get("ER_DECODE") else ""
    prm = json.loads((ART / train_tag / f"decode_params_{mname}{S2TAG}{sfx}.json").read_text())
    tau, gate = prm["tau"], prm["gate"]
    print(f"[write] decode params: tau={tau} gate={gate} (decode_params_{mname}{S2TAG}{sfx}.json)", flush=True)
    out_dir = OUT / f"variant{S2TAG}" if S2TAG else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_all = pl.read_csv(DATA / "test" / "test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0, columns=["entity_id"])
    matches, cands = [], []
    for c in _countries(test_tag):
        d = ART / test_tag / c
        h = pl.read_parquet(d / f"infer_hits{S2TAG}" / "shard_*.parquet").filter(pl.col("p2") >= tau)
        h = h.filter(pl.col("p2") == pl.col("p2").max().over("r"))                    # each record -> its best S1
        h = h.filter(pl.col("p2").max().over("s1") >= gate)                            # singleton gate
        matches.append(h.group_by("s1").agg(pl.col("r").sort().str.join(",").alias("matched_entity_ids")))
        if not S2TAG:
            cands.append(candidate_lists(test_tag, c))
        print(f"[write] {c}: matched S1={matches[-1].height:,} pairs={h.height:,}", flush=True)
    todo = [(None, matches, "matched_entity_ids", "matching_results.tsv")]
    if not S2TAG:
        todo.append((None, cands, "candidate_entity_ids", "candidate_pairs.tsv"))
    for name, parts, col, fn in todo:
        m = pl.concat(parts)
        res = (s1_all.rename({"entity_id": "source1_entity_id"}).join(m.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left")
               .with_columns(pl.col(col).fill_null("")))
        res.write_csv(out_dir / fn, separator="\t", quote_style="never")
        print(f"[write] {fn}: {res.height:,} rows, non-empty {(res[col] != '').sum():,}", flush=True)


if __name__ == "__main__":
    train_tag, test_tag = sys.argv[1], sys.argv[2]
    mname = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "all"
    s1_only = "--stage1-only" in sys.argv
    for c in _countries(test_tag):
        predict_country(train_tag, test_tag, mname, c, s1_only)
    if not s1_only and not HSFX:
        write_outputs(test_tag, mname, train_tag)
