"""Stage 3 (refinement pass): re-score every plausible pair using the stage-2 v3 scores of the entity's OTHER candidates.

Stage-2 scores (p2) are much sharper than stage-1 scores (p1), so the sibling-support signals ("how strongly do the other
candidates that share this record's house number / address / cleaned name look like matches") are recomputed with p2,
plus the record's rank / share within its entity. Training rows get out-of-fold p2 (model fit on the other half of the
training entities); validation and test rows get p2 from the full v3 model, exactly as at inference time.

    python stage3.py train      fit + compare with v3 on the plain and the test-like validation score
    python stage3.py infer      score the test set -> artifacts/t1/<country>/infer_hits_v4
"""
from __future__ import annotations
import os, sys, time, json
os.environ.setdefault("ER_ANC", "1")
os.environ.setdefault("ER_S2", "anc")
import numpy as np
import polars as pl
import lightgbm as lgb
from config import ART, SEED, HSFX
import stage2_v3 as V
from train import decode, fscore, add_stage2, with_anc

MD = ART / V.TRAIN / f"models_{V.MN}"
P2_NAMES = ["p2", "p2_rank_s1", "p2_max_s1", "p2_gap_s1", "p2_2nd_s1", "p2_sum_s1", "p2_n50_s1",
            "p2_num_w", "p2_addr_w", "p2_name_w", "p2_s1num_w", "p2_share"]
FEATS4 = V.FEATS3 + P2_NAMES


def add_p2feats(df: pl.DataFrame, tabs: tuple) -> pl.DataFrame:
    """df: complete plausible candidate lists (s1, r, p2). Adds P2_NAMES (except p2 itself)."""
    a, b = tabs
    df = df.join(a, on="s1", how="left").join(b, on="r", how="left")
    q = pl.col("p2")
    has_n, has_k, has_a = pl.col("rn1").is_not_null(), pl.col("rk").str.len_chars() > 0, pl.col("ra").is_not_null()
    df = df.with_columns(
        q.rank("ordinal", descending=True).over("s1").alias("p2_rank_s1"),
        q.max().over("s1").alias("p2_max_s1"),
        (q - q.max().over("s1")).alias("p2_gap_s1"),
        pl.when(pl.len().over("s1") > 1).then(q.top_k(2).min().over("s1")).otherwise(0.0).alias("p2_2nd_s1"),
        q.sum().over("s1").alias("p2_sum_s1"),
        (q >= 0.5).sum().over("s1").alias("p2_n50_s1"),
        pl.when(has_n).then(q.sum().over("s1", "rn1") - q).alias("p2_num_w"),
        pl.when(has_a).then(q.sum().over("s1", "ra") - q).alias("p2_addr_w"),
        pl.when(has_k).then(q.sum().over("s1", "rk") - q).alias("p2_name_w"),
        pl.when(pl.col("rn1") == pl.col("sn1")).then(q).otherwise(0.0).sum().over("s1").alias("p2_s1num_w"),
        (q / (q.sum().over("s1") + 1e-6)).alias("p2_share"))
    return df.drop("sn1", "rn1", "rk", "ra").with_columns(pl.col(P2_NAMES).cast(pl.Float32))


def _fit(X, y, path, rounds):
    """Resumable LightGBM training (trees added in blocks of 100, saved after each block)."""
    ds = lgb.Dataset(X, label=y, free_raw_data=False)
    booster = lgb.Booster(model_file=str(path)) if path.exists() else None
    done = booster.num_trees() if booster else 0
    while done < rounds:
        booster = lgb.train(V.PARAMS, ds, num_boost_round=min(100, rounds - done), init_model=booster, keep_training_booster=True)
        done = booster.num_trees()
        booster.save_model(str(path))
        print(f"   {path.name}: trees={done}/{rounds}", flush=True)
    return booster


def train() -> None:
    """Train the stage-3 refinement model on features recomputed from out-of-fold stage-2 scores; report plain and
    test-like validation F0.5 and save the threshold."""
    t0 = time.time()
    tr_parts, va_parts, tabs = [], {}, {}
    for c in V.TRAIN_C:
        d = ART / V.TRAIN / c
        n_tr = pl.scan_parquet(d / "s1_*.parquet").select(pl.col("id").alias("s1")).filter(~V.DROPPED & (V.FOLD != 0)).select(pl.len()).collect().item()
        tr_c, va_c = V._frames(c, int(min(1000, 1000 * 150_000 / max(1, n_tr))))
        tr_parts.append(tr_c.with_columns(pl.lit(c).alias("cty")))
        va_parts[c] = va_c
        tabs[c] = V.sup_tables(d)
    tr = pl.concat(tr_parts).with_columns((pl.col("s1").hash(SEED + 9) % 2).alias("half"))
    print(f"[v4] frames ready: train={tr.height:,} ({time.time() - t0:.0f}s)", flush=True)
    # out-of-fold p2 for training rows
    X = tr.select(V.FEATS3).to_numpy()
    y = tr["y"].to_numpy()
    half = tr["half"].to_numpy()
    p2 = np.zeros(tr.height, dtype=np.float32)
    for h in (0, 1):
        m = _fit(X[half == h], y[half == h], MD / f"model_s3_half{h}{V.MSFX}.txt", 300)
        p2[half != h] = m.predict(X[half != h], num_threads=12)
    del X
    tr = tr.with_columns(pl.Series("p2", p2))
    tr = pl.concat([add_p2feats(tr.filter(pl.col("cty") == c), tabs[c]) for c in V.TRAIN_C])
    m4 = _fit(tr.select(FEATS4).to_numpy(), tr["y"].to_numpy(), MD / f"model_s3{V.MSFX}.txt", 400)
    del tr
    m3 = lgb.Booster(model_file=str(MD / f"model_s2_v3{V.MSFX}.txt"))
    res = {"v3": {}, "v4": {}}
    print(f"\n== validation: plain macro F0.5 | test-like F0.5 (orphan false matches x{V.W_EVAL:g})")
    for c in V.TRAIN_C:
        d = ART / V.TRAIN / c
        s1_ids = (pl.read_parquet(d / "s1_*.parquet", columns=["id", "is_val"]).filter(pl.col("is_val")).rename({"id": "s1"}).filter(~V.DROPPED)["s1"])
        gt = pl.read_parquet(d / "gt.parquet").filter(pl.col("s1").is_in(s1_ids.implode()))
        va = va_parts[c].with_columns(pl.Series("p2", m3.predict(va_parts[c].select(V.FEATS3).to_numpy(), num_threads=12).astype(np.float32)))
        va = add_p2feats(va, tabs[c])
        v4 = va.with_columns(pl.Series("p2", m4.predict(va.select(FEATS4).to_numpy(), num_threads=12)))
        for name, frame in (("v3", va), ("v4", v4)):
            for t in V.TAUS:
                res[name].setdefault(t, []).append((fscore(decode(frame, "p2", t, 0.0, True), gt, s1_ids), V.fscore_w(frame, t, gt, s1_ids, V.W_EVAL)))
    for name in ("v3", "v4"):
        best = max(V.TAUS, key=lambda t: sum(x[1] for x in res[name][t]))
        plain = sum(x[0] for x in res[name][best]) / len(V.TRAIN_C)
        tl = sum(x[1] for x in res[name][best]) / len(V.TRAIN_C)
        per = "  ".join(f"{c}: plain {res[name][best][i][0]:.4f} / test-like {res[name][best][i][1]:.4f}" for i, c in enumerate(V.TRAIN_C))
        print(f"   {name}: best tau={best}  plain={plain:.4f}  test-like={tl:.4f}   ({per})", flush=True)
        if name == "v4":
            (ART / V.TRAIN / f"decode_params_all_v4{V.MSFX}_tl.json").write_text(json.dumps(dict(tau=best, gate=0.0, assign=True, plain=plain, test_like=tl)))
    imp = sorted(zip(FEATS4, m4.feature_importance("gain")), key=lambda x: -x[1])[:10]
    print("   top features:", [(n, int(g)) for n, g in imp], f"({time.time() - t0:.0f}s)")


def infer() -> None:
    """Score the filtered test candidates with stage 2 v3 and then the stage-3 refinement model (keeps p2 >= 0.3)."""
    m3 = lgb.Booster(model_file=str(MD / f"model_s2_v3{V.MSFX}.txt"))
    m4 = lgb.Booster(model_file=str(MD / f"model_s3{V.MSFX}.txt"))
    for c in V.TEST_C:
        d = ART / V.TEST / c
        out = d / f"infer_hits_v4{V.MSFX}{HSFX}"
        out.mkdir(exist_ok=True)
        rp = pl.read_parquet(d / f"rp_infer{HSFX}.parquet")
        tabs = V.sup_tables(d)
        for f in sorted((d / "infer_p1").glob("shard_*.parquet")):
            o = out / f.name
            if o.exists():
                continue
            t0 = time.time()
            df = with_anc(add_stage2(pl.read_parquet(f), rp), d, V.MN, f.name, "infer_anc").filter(pl.col("p1") >= V.P_MIN)
            df = V.add_support(df, tabs).join(pl.read_parquet(d / "v3" / f.name), on=["s1", "r"], how="left")
            df = V.add_twin(df, tabs, d, f.name)
            df = df.with_columns(pl.Series("p2", m3.predict(df.select(V.FEATS3).to_numpy(), num_threads=12).astype(np.float32)))
            df = add_p2feats(df, tabs)
            p4 = m4.predict(df.select(FEATS4).to_numpy(), num_threads=12)
            df.select("s1", "r").with_columns(pl.Series("p2", p4.astype(np.float32))).filter(pl.col("p2") >= 0.3).write_parquet(o)
            print(f"[v4 infer] {c} {f.name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    {"train": train, "infer": infer}[sys.argv[1]]()
