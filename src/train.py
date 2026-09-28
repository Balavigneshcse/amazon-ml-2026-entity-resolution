"""Two-stage pair classifier (LightGBM) with out-of-fold scoring, decoding and the official macro-F0.5 metric.

Stage 1: pair features -> p1.  5 folds by hash of the Source-1 id. Fold 0 is the validation fold and is never used to
         fit anything. Model k (k=1..4) is fit on folds not in {0,k} and scores fold k; model 0 is fit on folds 1..4 and
         scores fold 0. So every pair's p1 comes from a model that never saw that entity (no in-sample scores).
Stage 2: p1 + within-entity ranking + competition between S1 entities for the same record -> p2 (fit on folds 1..4).
Decode:  each candidate record goes to its best S1 owner (partition constraint), thresholded for F0.5.

One model is fit across all listed countries (no country feature, so it transfers to unseen countries such as France).
`--train-on` lets you fit on a subset of countries and score the others (leave-one-country-out robustness check).

Usage:  python train.py stage1 <tag> <countries,comma,sep> [mname] [train_on,comma,sep]
        python train.py stage2 <tag> <countries,comma,sep> [mname] [train_on,comma,sep]
All steps are resumable (trees are checkpointed every 100 rounds; prediction is per shard).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import polars as pl
import lightgbm as lgb
from config import ART, SEED, USE_ANC, S2TAG
from features import SIM_NAMES, BLOCK_COLS, CTX, add_context

USE_X = os.environ.get("ER_X", "0") == "1"
if USE_X:
    from features_extra import X_NAMES
FEATS1 = SIM_NAMES + BLOCK_COLS + CTX + (X_NAMES if USE_X else [])


def with_x(df: pl.DataFrame, d, shard: str | None = None) -> pl.DataFrame:
    """Attach the extra features (row-aligned per shard, or joined on (s1, r) for sampled frames)."""
    if not USE_X:
        return df
    if shard is not None:
        x = pl.read_parquet(d / "feats_x" / shard)
        assert x.height == df.height and (x["s1"] == df["s1"]).all() and (x["r"] == df["r"]).all()
        return df.hstack(x.drop("s1", "r"))
    return df.join(pl.scan_parquet(d / "feats_x" / "shard_*.parquet").filter(pl.col("s1").is_in(df["s1"].unique().implode())).collect(),
                   on=["s1", "r"], how="left")
S1_COLS = ["p_rank_s1", "p_max_s1", "p_gap_s1", "p_2nd_s1", "p_sum_s1", "n_gt50_s1"]
R_COLS = ["p_top1_r", "p_margin_r", "p_ratio_r", "n_gt50_r", "p_sum_r"]
FEATS2 = FEATS1 + ["p1"] + S1_COLS + R_COLS
if USE_ANC:
    from cluster import ANC_NAMES
    FEATS2 = FEATS2 + ANC_NAMES


def with_anc(df: pl.DataFrame, d, mname: str, shard: str | None = None, folder: str | None = None) -> pl.DataFrame:
    """Attach cluster-consistency features (NaN for candidates that were not scored against anchors)."""
    if not USE_ANC:
        return df
    folder = folder or f"anc_{mname}"
    if shard is not None:
        a = pl.read_parquet(d / folder / shard)
    else:
        a = pl.scan_parquet(d / folder / "shard_*.parquet").filter(pl.col("s1").is_in(df["s1"].unique().implode())).collect()
    return df.join(a, on=["s1", "r"], how="left")
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=12, verbose=-1, seed=SEED, max_bin=255)
ROUNDS1, ROUNDS2 = 300, 300
FOLD = ((pl.col("s1").hash(SEED + 1) % 100) // 20).cast(pl.UInt8).alias("fold")


def _d(tag, country):
    return ART / tag / country


def _n_s1(d) -> int:
    return pl.scan_parquet(d / "s1_*.parquet").select(pl.len()).collect().item()


def _gt_y(d) -> pl.DataFrame:
    return pl.read_parquet(d / "gt.parquet").with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))


def _sample_filter(folds: list, n_s1_total: int, max_s1: int) -> pl.Expr:
    per_mille = int(min(1000, 1000 * max_s1 / max(1, n_s1_total * len(folds) / 5)))
    return FOLD.is_in(folds) & ((pl.col("s1").hash(SEED + 3) % 1000) < per_mille)


def train_resumable(X: np.ndarray, y: np.ndarray, names: list, path, rounds: int, block: int = 100) -> lgb.Booster:
    ds = lgb.Dataset(X, label=y, feature_name=names, free_raw_data=False)
    booster = lgb.Booster(model_file=str(path)) if path.exists() else None
    done = booster.num_trees() if booster else 0
    while done < rounds:
        booster = lgb.train(PARAMS, ds, num_boost_round=min(block, rounds - done), init_model=booster, keep_training_booster=True)
        done = booster.num_trees()
        booster.save_model(str(path))
        print(f"   {path.name}: trees={done}/{rounds}", flush=True)
    return booster


def _mdir(tag, mname):
    p = ART / tag / f"models_{mname}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _pred_dir(tag, country, mname):
    p = _d(tag, country) / f"pred_{mname}"
    p.mkdir(exist_ok=True)
    return p


# ------------------------------------------------------------------ stage 1
def stage1(tag: str, countries: list, mname: str = "all", train_on: list | None = None, max_s1: int = 90_000) -> None:
    train_on = train_on or countries
    md = _mdir(tag, mname)
    per_c = max_s1 // len(train_on)
    rst = {c: pl.read_parquet(_d(tag, c) / "rstats.parquet") for c in countries}
    gts = {c: _gt_y(_d(tag, c)) for c in countries}
    models = {}
    for k in range(5):
        path = md / f"model_s1_k{k}.txt"
        folds = [f for f in (1, 2, 3, 4) if f != k]
        if path.exists() and lgb.Booster(model_file=str(path)).num_trees() >= ROUNDS1:
            models[k] = lgb.Booster(model_file=str(path))
            continue
        t0 = time.time()
        parts = []
        for c in train_on:
            d = _d(tag, c)
            df = pl.scan_parquet(d / "feats" / "shard_*.parquet").filter(_sample_filter(folds, _n_s1(d), per_c)).collect()
            df = with_x(df, d)
            df = add_context(df, rst[c]).join(gts[c], on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
            parts.append(df.select(FEATS1 + ["y"]))
        df = pl.concat(parts)
        print(f"[stage1:{mname}] model {k}: train rows={df.height:,} pos={df['y'].mean():.4f}", flush=True)
        models[k] = train_resumable(df.select(FEATS1).to_numpy(), df["y"].to_numpy(), FEATS1, path, ROUNDS1)
        print(f"[stage1:{mname}] model {k} done ({time.time()-t0:.0f}s)", flush=True)
        del df, parts
    for c in countries:
        d = _d(tag, c)
        out = _pred_dir(tag, c, mname)
        for f in sorted((d / "feats").glob("shard_*.parquet")):
            o = out / f.name
            if o.exists():
                continue
            t0 = time.time()
            df = add_context(with_x(pl.read_parquet(f), d, f.name), rst[c]).with_columns(FOLD)
            p = np.zeros(df.height, dtype=np.float32)
            fo = df["fold"].to_numpy()
            X = df.select(FEATS1).to_numpy()
            for k in range(5):
                m = fo == k
                if m.any():
                    p[m] = models[k].predict(X[m], num_threads=12)
            df = df.with_columns(pl.Series("p1", p)).join(gts[c], on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
            df.write_parquet(o)
            print(f"[stage1:{mname}] {c} {f.name}: {df.height:,} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[stage1:{mname}] done")


# ------------------------------------------------------------------ stage 2
def rp_stats(d, mname) -> pl.DataFrame:
    p = d / f"rp_{mname}.parquet"
    if p.exists():
        return pl.read_parquet(p)
    st = (pl.scan_parquet(d / f"pred_{mname}" / "shard_*.parquet").select("r", "p1").group_by("r")
          .agg(pl.col("p1").sort(descending=True).head(2).alias("t"), (pl.col("p1") > 0.5).sum().cast(pl.UInt16).alias("n_gt50_r"),
               pl.col("p1").sum().alias("p_sum_r")).collect(engine="streaming"))
    st = st.select("r", "n_gt50_r", "p_sum_r", *[pl.col("t").list.get(i, null_on_oob=True).fill_null(0.0).alias(f"pt{i+1}") for i in range(2)])
    st.write_parquet(p)
    return st


def add_stage2(df: pl.DataFrame, rp: pl.DataFrame) -> pl.DataFrame:
    df = df.join(rp, on="r", how="left")
    comp = pl.when(pl.col("p1") >= pl.col("pt1")).then(pl.col("pt2")).otherwise(pl.col("pt1"))
    return df.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("s1").alias("p_rank_s1"),
        pl.col("p1").max().over("s1").alias("p_max_s1"),
        (pl.col("p1") - pl.col("p1").max().over("s1")).alias("p_gap_s1"),
        pl.col("p1").sort(descending=True).over("s1").get(1).alias("p_2nd_s1"),
        pl.col("p1").sum().over("s1").alias("p_sum_s1"),
        (pl.col("p1") > 0.5).sum().over("s1").alias("n_gt50_s1"),
        pl.col("pt1").alias("p_top1_r"),
        (pl.col("p1") - comp).alias("p_margin_r"),
        (pl.col("p1") / (comp + 1e-6)).alias("p_ratio_r"),
    ).drop("pt1", "pt2")


def stage2(tag: str, countries: list, mname: str = "all", train_on: list | None = None, max_s1: int = 90_000) -> None:
    train_on = train_on or countries
    md = _mdir(tag, mname)
    per_c = max_s1 // len(train_on)
    rp = {c: rp_stats(_d(tag, c), mname) for c in countries}
    path = md / f"model_s2{S2TAG}.txt"
    parts = []
    for c in train_on:
        d = _d(tag, c)
        df = pl.scan_parquet(d / f"pred_{mname}" / "shard_*.parquet").filter(_sample_filter([1, 2, 3, 4], _n_s1(d), per_c)).collect()
        parts.append(with_anc(add_stage2(df, rp[c]), d, mname).select(FEATS2 + ["y"]))
    df = pl.concat(parts)
    print(f"[stage2:{mname}] train rows={df.height:,}", flush=True)
    m2 = train_resumable(df.select(FEATS2).to_numpy(), df["y"].to_numpy(), FEATS2, path, ROUNDS2)
    del df, parts
    from sklearn.metrics import average_precision_score
    for c in countries:
        d = _d(tag, c)
        s1_val = pl.read_parquet(d / "s1_*.parquet", columns=["id", "is_val"]).filter(pl.col("is_val"))["id"]
        gt = pl.read_parquet(d / "gt.parquet")
        gt_val = gt.filter(pl.col("s1").is_in(s1_val.implode()))
        va = pl.scan_parquet(d / f"pred_{mname}" / "shard_*.parquet").filter(FOLD == 0).collect()
        va = with_anc(add_stage2(va, rp[c]), d, mname)
        va = va.with_columns(pl.Series("p2", m2.predict(va.select(FEATS2).to_numpy(), num_threads=12)))
        held_out = "" if c in train_on else "  (COUNTRY NOT SEEN IN TRAINING)"
        print(f"[stage2:{mname}] {c}{held_out}: val pairs={va.height:,} stage1 AP={average_precision_score(va['y'], va['p1']):.5f} "
              f"stage2 AP={average_precision_score(va['y'], va['p2']):.5f} | recall ceiling={va['y'].sum() / gt_val.height:.4f}")
        for pcol in ("p1", "p2"):
            best = sweep(va, gt_val, s1_val, pcol)
            print(f"[stage2:{mname}] {c} {pcol}: BEST validation macro F0.5 = {best[0]:.4f} at (assign, tau, gate) = {best[1]}", flush=True)
        va.select("s1", "r", "p1", "p2", "y").write_parquet(d / f"val_pred_{mname}{S2TAG}.parquet")


# ------------------------------------------------------------------ decoding + metric
def decode(df: pl.DataFrame, pcol: str, tau: float, gate: float, assign: bool = True) -> pl.DataFrame:
    """Predicted (s1, r): p>=tau; each r goes to its best S1 (if assign); an S1 needs max p>=gate."""
    d = df.select("s1", "r", pl.col(pcol).alias("p")).filter(pl.col("p") >= tau)
    if assign:
        d = d.filter(pl.col("p") == pl.col("p").max().over("r"))
    d = d.filter(pl.col("p").max().over("s1") >= gate)
    return d.select("s1", "r")


def fscore(pred: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series) -> float:
    """Macro F0.5 over s1_ids: F = 1.25*tp / (0.25*n_gt + n_pred), 1.0 when both empty."""
    ids = pl.DataFrame({"s1": s1_ids})
    npred = pred.group_by("s1").len().rename({"len": "npred"})
    ngt = gt.group_by("s1").len().rename({"len": "ngt"})
    tp = pred.join(gt, on=["s1", "r"]).group_by("s1").len().rename({"len": "tp"})
    m = ids.join(npred, on="s1", how="left").join(ngt, on="s1", how="left").join(tp, on="s1", how="left").fill_null(0)
    m = m.with_columns(pl.when(pl.col("npred") + pl.col("ngt") == 0).then(1.0)
                       .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("ngt") + pl.col("npred"))).alias("f"))
    return float(m["f"].mean())


def sweep(df: pl.DataFrame, gt: pl.DataFrame, s1_val: pl.Series, pcol: str):
    best = (0.0, None)
    for assign in (False, True):
        for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            for gate in (0.0, 0.6, 0.8, 0.9):
                if gate and gate < tau:
                    continue
                f = fscore(decode(df, pcol, tau, gate, assign), gt, s1_val)
                if f > best[0]:
                    best = (f, (assign, tau, gate))
    return best


if __name__ == "__main__":
    cmd, tag, countries = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    mname = sys.argv[4] if len(sys.argv) > 4 else "all"
    train_on = sys.argv[5].split(",") if len(sys.argv) > 5 else None
    {"stage1": stage1, "stage2": stage2}[cmd](tag, countries, mname, train_on)
