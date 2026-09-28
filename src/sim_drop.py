"""Leaderboard-like validation: re-score the validation fold at the test set's orphan density (no training).

The test set has ~5.8 Source-2/3 records per Source-1 entity vs ~4.7 in train, i.e. about twice as many records whose
owner is not in Source 1. We mimic that by removing ER_DROP % (default 19 %) of the Source-1 entities from the training
universe: their records stay in the candidate pool but now have no owner. Blocking lists of the remaining entities are
unchanged; everything that depends on competition between entities is recomputed:
  rstats (per-record competition) -> stage-1 p1 with the existing 5 out-of-fold models -> rp stats -> cluster features
  -> stage-2 (anc model) p2 on the validation fold -> macro F0.5 per threshold, compared with the full-density score.
Writes artifacts/f2/decode_params_all_anc_d<DROP>.json (threshold that maximises the mean F0.5 at test-like density).

    python sim_drop.py            (resumable; per-shard outputs under artifacts/f2/<country>/)
"""
from __future__ import annotations
import os, sys, time, json
os.environ.setdefault("ER_ANC", "1")
os.environ.setdefault("ER_S2", "anc")
import numpy as np
import polars as pl
import lightgbm as lgb
from config import ART, SEED, KBLOCK
from features import add_context
from train import FEATS1, FEATS2, FOLD, add_stage2, with_anc, rp_stats, decode, fscore
import cluster

DROP = int(os.environ.get("ER_DROP", "19"))
TAG, MN = "f2", "all"
SUF = f"d{DROP}"
DROPPED = (pl.col("s1").hash(SEED + 5) % 100) < DROP
TAUS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def rstats_drop(d) -> pl.DataFrame:
    o = d / f"rstats_{SUF}.parquet"
    if o.exists():
        return pl.read_parquet(o)
    st = (pl.scan_parquet(d / "blocks" / "shard_*.parquet").filter(pl.col("rank") <= KBLOCK).filter(~DROPPED).select("r", "cos")
          .group_by("r").agg(pl.col("cos").sort(descending=True).head(3).alias("t"), pl.len().cast(pl.UInt16).alias("n_s1_for_r"))
          .collect(engine="streaming"))
    st = st.select("r", "n_s1_for_r", *[pl.col("t").list.get(i, null_on_oob=True).fill_null(0.0).alias(f"t{i+1}") for i in range(3)])
    st.write_parquet(o)
    return st


def stage1_drop(d, c, rst, models) -> None:
    pdir, vdir = d / f"pred_{SUF}", d / f"val_{SUF}"
    pdir.mkdir(exist_ok=True)
    vdir.mkdir(exist_ok=True)
    gt = pl.read_parquet(d / "gt.parquet").with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
    shards = sorted((d / "feats").glob("shard_*.parquet"))
    for i, f in enumerate(shards):
        if (pdir / f.name).exists() and (vdir / f.name).exists():
            continue
        t0 = time.time()
        df = add_context(pl.read_parquet(f).filter(~DROPPED), rst).with_columns(FOLD)
        X, fo = df.select(FEATS1).to_numpy(), df["fold"].to_numpy()
        p = np.zeros(df.height, dtype=np.float32)
        for k in range(5):
            m = fo == k
            if m.any():
                p[m] = models[k].predict(X[m], num_threads=12)
        df = df.with_columns(pl.Series("p1", p))
        df.filter(pl.col("fold") == 0).drop("fold").join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0)).write_parquet(vdir / f.name)
        df.select("s1", "r", "p1").write_parquet(pdir / f.name)
        print(f"[sim] {c} stage-1 {i + 1}/{len(shards)} ({time.time() - t0:.0f}s)", flush=True)


def stage2_drop(d, c, rp, m2) -> pl.DataFrame:
    o = d / f"val_pred_{MN}_anc_{SUF}.parquet"
    if o.exists():
        return pl.read_parquet(o)
    out = []
    for f in sorted((d / f"val_{SUF}").glob("shard_*.parquet")):
        va = with_anc(add_stage2(pl.read_parquet(f), rp), d, MN, f.name, f"anc_val_{SUF}")
        p2 = m2.predict(va.select(FEATS2).to_numpy(), num_threads=12)
        out.append(va.select("s1", "r", "p1", "y").with_columns(pl.Series("p2", p2.astype(np.float32))))
    va = pl.concat(out)
    va.write_parquet(o)
    return va


def claimed_per_s1(pred: pl.DataFrame, n: int) -> float:
    return pred.height / max(1, n)


if __name__ == "__main__":
    t0 = time.time()
    md = ART / TAG / f"models_{MN}"
    models = {k: lgb.Booster(model_file=str(md / f"model_s1_k{k}.txt")) for k in range(5)}
    m2 = lgb.Booster(model_file=str(md / "model_s2_anc.txt"))
    table = {}
    for c in ("India", "US"):
        d = ART / TAG / c
        rst = rstats_drop(d)
        print(f"[sim] {c}: rstats at {DROP}% dropped S1 ({time.time() - t0:.0f}s)", flush=True)
        stage1_drop(d, c, rst, models)
        rp = rp_stats(d, SUF)                                   # reads pred_d19/, writes rp_d19.parquet
        cluster.build_dir(TAG, c, f"val_{SUF}", f"anc_val_{SUF}")
        va = stage2_drop(d, c, rp, m2)
        s1_ids = (pl.read_parquet(d / "s1_*.parquet", columns=["id", "is_val"]).filter(pl.col("is_val"))
                  .rename({"id": "s1"}).filter(~DROPPED)["s1"])
        gt = pl.read_parquet(d / "gt.parquet").filter(pl.col("s1").is_in(s1_ids.implode()))
        full = pl.read_parquet(d / "val_pred_all_anc.parquet").filter(pl.col("s1").is_in(s1_ids.implode()))
        n = len(s1_ids)
        f_full = fscore(decode(full, "p2", 0.65, 0.0, True), gt, s1_ids)
        print(f"\n== {c}: validation entities kept={n:,}  true pairs={gt.height:,}", flush=True)
        print(f"   full density  (train-like): F0.5@0.65 = {f_full:.4f}   predicted pairs/S1 = "
              f"{claimed_per_s1(decode(full, 'p2', 0.65, 0.0, True), n):.2f}")
        table[c] = {}
        for tau in TAUS:
            pred = decode(va, "p2", tau, 0.0, True)
            table[c][tau] = fscore(pred, gt, s1_ids)
            print(f"   test-like density: F0.5@{tau:.2f} = {table[c][tau]:.4f}   predicted pairs/S1 = {claimed_per_s1(pred, n):.2f}", flush=True)
    mean = {t: sum(table[c][t] for c in table) / len(table) for t in TAUS}
    best = max(mean, key=mean.get)
    print(f"\n[sim] mean F0.5 at test-like density: " + "  ".join(f"{t}:{mean[t]:.4f}" for t in TAUS))
    print(f"[sim] BEST threshold at test-like density = {best}  (mean F0.5 {mean[best]:.4f}; at 0.65: {mean[0.65]:.4f})")
    (ART / TAG / f"decode_params_{MN}_anc_{SUF}.json").write_text(json.dumps(
        dict(tau=best, gate=0.0, assign=True, per_country={c: table[c][best] for c in table}, drop_pct=DROP)))
    print(f"[sim] wrote decode_params_{MN}_anc_{SUF}.json  ({time.time() - t0:.0f}s total)")
