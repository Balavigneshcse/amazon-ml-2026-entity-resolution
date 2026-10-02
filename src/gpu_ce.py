"""GPU cross-encoder: a small pretrained transformer fine-tuned to read both records' text at once.

Base model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22 M parameters, Apache-2.0). Input: "name | address" of the Source-1
record and of the candidate; output: probability that they are the same business. It scores only the candidates that pass
the stage-1 filter (p1 >= 0.02, ~5 per entity), and a small LightGBM "stacker" combines its score with the cluster-model
score. Everything is resumable (training checkpoints, per-shard scoring).

    python gpu_ce.py check      GPU / library check
    python gpu_ce.py data       build training pairs (training folds only, never the validation fold)
    python gpu_ce.py train      fine-tune on the GPU (1 epoch, fp16)
    python gpu_ce.py score      score validation-fold pairs and all test pairs
    python gpu_ce.py stack      fit the combiner, compare with the current model on validation, score the test set
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import polars as pl
from config import ART, SEED, countries

# second model (step 8): ER_CE_TAG=_l12 ER_CE_BASE=cross-encoder/ms-marco-MiniLM-L-12-v2 ER_CE_N=1000000
TAG = os.environ.get("ER_CE_TAG", "")
BASE = os.environ.get("ER_CE_BASE", "cross-encoder/ms-marco-MiniLM-L-6-v2")
if BASE.startswith("@art/"):                     # a model fine-tuned earlier in this run, e.g. @art/ce_model_l12/final
    BASE = str(ART / BASE[len("@art/"):])
N_PER_COUNTRY = int(os.environ.get("ER_CE_N", "600000"))
STACK_TAGS = [t for t in os.environ.get("ER_CE_STACK", "").split(",")] if os.environ.get("ER_CE_STACK") is not None else [""]
SSFX = "".join(STACK_TAGS)                     # suffix of stacker outputs ("" = first model only, "_l12" = both)
CE_DIR = ART / f"ce_model{TAG}"
P_MIN = 0.02
MAXLEN = 96
DROPPED = (pl.col("s1").hash(SEED + 5) % 100) < 19
FOLD = ((pl.col("s1").hash(SEED + 1) % 100) // 20).cast(pl.UInt8)
TRAIN_C, TEST_C = countries("f2"), countries("t1")         # labelled training countries / all test countries
TAUS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def _text(d, ids: pl.Series, which: str) -> pl.DataFrame:
    """Cross-encoder input text 'name | address' (raw fields, at most 200 characters) for the given ids."""
    f = "s1_*.parquet" if which == "s1" else "cand_*.parquet"
    t = (pl.scan_parquet(d / f).select("id", "name", "addr").filter(pl.col("id").is_in(ids.unique().implode())).collect()
         .select(pl.col("id"), (pl.col("name").fill_null("") + " | " + pl.col("addr").fill_null("")).str.slice(0, 200).alias("t")))
    return t


def pairs_text(d, p: pl.DataFrame) -> pl.DataFrame:
    """Attach the Source-1 text (ta) and the candidate text (tb) to (s1, r) pairs, keeping the pair order."""
    a = _text(d, p["s1"], "s1").rename({"id": "s1", "t": "ta"})
    b = _text(d, p["r"], "cand").rename({"id": "r", "t": "tb"})
    return p.join(a, on="s1", how="left", maintain_order="left").join(b, on="r", how="left", maintain_order="left")


# ------------------------------------------------------------------ check / data
def check() -> None:
    import torch, transformers
    print("torch", torch.__version__, "| transformers", transformers.__version__, "| CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), f"| memory {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def data(n_per_country: int = N_PER_COUNTRY) -> None:
    """Training pairs for a cross-encoder: a sample of filtered candidate pairs (p1 >= 0.02) from the training folds
    only (the validation fold is never used), labelled from the ground truth."""
    out = ART / f"ce_train{TAG}.parquet"
    if out.exists():
        print("[ce] training pairs exist, skip")
        return
    parts = []
    for c in TRAIN_C:
        d = ART / "f2" / c
        p = (pl.scan_parquet(d / "pred_d19" / "shard_*.parquet").select("s1", "r", "p1")
             .filter((pl.col("p1") >= P_MIN) & (FOLD != 0)).collect())
        p = p.sample(min(n_per_country, p.height), seed=SEED + (7 if TAG else 0))
        gt = pl.read_parquet(d / "gt.parquet").with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
        p = p.join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
        parts.append(pairs_text(d, p).select("ta", "tb", "y"))
        print(f"[ce] {c}: {p.height:,} training pairs, positive rate {p['y'].mean():.3f}", flush=True)
    pl.concat(parts).sample(fraction=1.0, shuffle=True, seed=SEED).write_parquet(out)


# ------------------------------------------------------------------ train
def train(batch: int = 64, lr: float = 3e-5, epochs: int = 1) -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda"
    df = pl.read_parquet(ART / f"ce_train{TAG}.parquet")
    ta, tb, y = df["ta"].to_list(), df["tb"].to_list(), df["y"].to_numpy().astype(np.float32)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(dev)
    steps_total = epochs * (len(ta) // batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * max(0.0, 1 - s / steps_total))
    scaler = torch.amp.GradScaler("cuda")
    ck = CE_DIR / "ckpt.pt"
    step = 0
    if ck.exists():
        st = torch.load(ck, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        step = st["step"]
        print(f"[ce] resuming at step {step}/{steps_total}", flush=True)
    if step >= steps_total and (CE_DIR / "final").exists():
        print("[ce] already trained"); return
    CE_DIR.mkdir(parents=True, exist_ok=True)
    order = np.concatenate([np.random.default_rng(SEED + e).permutation(len(ta)) for e in range(epochs)])
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, run, step0 = time.time(), 0.0, step
    while step < steps_total:
        idx = order[step * batch:(step + 1) * batch]
        enc = tok([ta[i] for i in idx], [tb[i] for i in idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(**enc).logits.squeeze(-1)
        loss = lossf(logit.float(), torch.from_numpy(y[idx]).to(dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        step += 1
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 200 == 0:
            el = time.time() - t0
            eta = el / max(1, step - step0) * (steps_total - step) / 60
            print(f"[ce] step {step}/{steps_total}  loss {run:.4f}  elapsed {el / 60:.1f} min, remaining ~{eta:.0f} min", flush=True)
        if step % 2000 == 0 or step == steps_total:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step}, ck)
    model.save_pretrained(CE_DIR / "final"); tok.save_pretrained(CE_DIR / "final")
    print(f"[ce] training done ({(time.time() - t0) / 60:.0f} min)")


# ------------------------------------------------------------------ score
def _scorer():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(CE_DIR / "final")
    model = AutoModelForSequenceClassification.from_pretrained(CE_DIR / "final").to("cuda").half().eval()

    def score(ta, tb, bs=512):
        """Probability that each (ta, tb) text pair is the same business (fp16 batches)."""
        out = np.empty(len(ta), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(ta), bs):
                enc = tok(ta[i:i + bs], tb[i:i + bs], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
                out[i:i + bs] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
        return out
    return score


def score_all() -> None:
    """Score the validation-fold candidates and all test candidates with the fine-tuned cross-encoder, shard by
    shard (resumable)."""
    score = _scorer()
    jobs = [(ART / "f2" / c, "val_d19", f"ce_val{TAG}") for c in TRAIN_C] + [(ART / "t1" / c, "infer_p1", f"ce{TAG}") for c in TEST_C]
    for d, src, dst in jobs:
        out = d / dst
        out.mkdir(exist_ok=True)
        shards = sorted((d / src).glob("shard_*.parquet"))
        for i, f in enumerate(shards):
            o = out / f.name
            if o.exists():
                continue
            t0 = time.time()
            p = pl.read_parquet(f, columns=["s1", "r", "p1"]).filter(pl.col("p1") >= P_MIN)
            x = pairs_text(d, p)
            s = score(x["ta"].to_list(), x["tb"].to_list())
            p.select("s1", "r").with_columns(pl.Series(f"ce{TAG}", s)).write_parquet(o)
            print(f"[ce score] {d.parent.name}/{d.name} {i + 1}/{len(shards)}: {p.height:,} pairs ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ stack
S_FEATS = ["p1", "p2c", "p2_rank_s1", "n_cand"] + [f"{k}{t}" for t in STACK_TAGS for k in ("ce", "ce_rank_s1", "ce_max_s1", "ce_gap_s1", "ce_sum_s1")]


def _feats(df: pl.DataFrame) -> pl.DataFrame:
    """Per-entity context for the stacker: rank of the cluster-model score, number of candidates, and each
    cross-encoder score's rank / max / gap / sum within the entity."""
    ex = [pl.col("p2c").rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("p2_rank_s1"),
          pl.len().over("s1").cast(pl.Float32).alias("n_cand")]
    for t in STACK_TAGS:
        c = pl.col(f"ce{t}")
        ex += [c.rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias(f"ce_rank_s1{t}"), c.max().over("s1").alias(f"ce_max_s1{t}"),
               (c - c.max().over("s1")).alias(f"ce_gap_s1{t}"), c.sum().over("s1").alias(f"ce_sum_s1{t}")]
    return df.with_columns(ex)


def _ce_join(df: pl.DataFrame, d, folder: str) -> pl.DataFrame:
    """Join the scores of every stacked cross-encoder (<folder><tag>/shard_*.parquet) onto the pairs."""
    for t in STACK_TAGS:
        df = df.join(pl.read_parquet(d / f"{folder}{t}" / "shard_*.parquet"), on=["s1", "r"], how="left")
    return df


def stack() -> None:
    """Fit the LightGBM stacker (stage-1 p1, cluster-model p2, cross-encoder features) on the validation fold with
    2-fold cross-fitting, compare it with the cluster model, save model and thresholds, score the test candidates."""
    import lightgbm as lgb
    from train import decode, fscore, PARAMS
    frames, evals = [], {}
    for c in TRAIN_C:
        d = ART / "f2" / c
        ce = _ce_join(pl.read_parquet(d / f"ce_val{STACK_TAGS[0]}" / "shard_*.parquet").select("s1", "r"), d, "ce_val")
        base = pl.read_parquet(d / "val_pred_all_anc_d19.parquet")            # cluster model, test-like universe
        x = ce.join(base, on=["s1", "r"], how="left").with_columns(
            pl.when(pl.col("p2") >= 0.3).then(pl.col("p2")).otherwise(0.0).alias("p2c"), pl.col("y").fill_null(0))
        x = _feats(x).with_columns(pl.lit(c).alias("cty"), ((pl.col("s1").hash(SEED + 11) % 2)).alias("half"))
        s1_ids = pl.read_parquet(d / "s1_*.parquet", columns=["id", "is_val"]).filter(pl.col("is_val")).rename({"id": "s1"}).filter(~DROPPED)["s1"]
        evals[c] = (pl.read_parquet(d / "gt.parquet").filter(pl.col("s1").is_in(s1_ids.implode())), s1_ids)
        frames.append(x)
    x = pl.concat(frames)
    # 2-fold cross-fitting inside the validation fold (by entity) -> honest comparison
    pred = np.zeros(x.height, dtype=np.float32)
    h = x["half"].to_numpy()
    X, y = x.select(S_FEATS).to_numpy(), x["y"].to_numpy()
    params = dict(PARAMS, num_leaves=63, min_data_in_leaf=200)
    for k in (0, 1):
        m = lgb.train(params, lgb.Dataset(X[h != k], y[h != k]), num_boost_round=300)
        pred[h == k] = m.predict(X[h == k], num_threads=12)
    x = x.with_columns(pl.Series("pst", pred))
    best = {}
    print("\n== validation (test-like universe), macro F0.5: current cluster model vs + GPU cross-encoder")
    for c in TRAIN_C:
        gt, s1_ids = evals[c]
        xc = x.filter(pl.col("cty") == c)
        f_old = fscore(decode(xc.rename({"p2c": "p"}).select("s1", "r", pl.col("p").alias("p2")), "p2", 0.7, 0.0, True), gt, s1_ids)
        res = {t: fscore(decode(xc.select("s1", "r", pl.col("pst").alias("p2")), "p2", t, 0.0, True), gt, s1_ids) for t in TAUS}
        bt = max(res, key=res.get)
        best[c] = bt
        print(f"   {c}: cluster model (tau .70) {f_old:.4f}  ->  with cross-encoder {res[bt]:.4f} (tau {bt})   [" + " ".join(f"{t}:{v:.4f}" for t, v in res.items()) + "]", flush=True)
    m = lgb.train(params, lgb.Dataset(X, y), num_boost_round=300)
    m.save_model(str(ART / "f2" / "models_all" / f"model_ce_stack{SSFX}.txt"))
    (ART / "f2" / f"decode_params_ce{SSFX}.json").write_text(json.dumps(best))
    for c in TEST_C:
        d = ART / "t1" / c
        base = pl.read_parquet(d / "infer_hits_anc_p020" / "shard_*.parquet")    # cluster model scores (kept if >= 0.3)
        p1 = pl.scan_parquet(d / "infer_p1" / "shard_*.parquet").select("s1", "r", "p1").filter(pl.col("p1") >= P_MIN).collect()
        t = _ce_join(p1, d, "ce").join(base, on=["s1", "r"], how="left").with_columns(
            pl.when(pl.col("p2") >= 0.3).then(pl.col("p2")).otherwise(0.0).alias("p2c"))
        t = _feats(t)
        t = t.select("s1", "r").with_columns(pl.Series("p2", m.predict(t.select(S_FEATS).to_numpy(), num_threads=12).astype(np.float32)))
        out = d / f"infer_hits_ce{SSFX}"
        out.mkdir(exist_ok=True)
        t.filter(pl.col("p2") >= 0.3).write_parquet(out / "shard_000.parquet")
        print(f"[ce stack] test {c}: scored {t.height:,} pairs")


def write() -> None:
    """Write output/ce_all<suffix>: each labelled country uses the threshold tuned on its validation fold; every country
    without training labels (here France) uses the strictest of those + 0.1, as an unseen country over-matched on the
    leaderboard."""
    import subprocess
    tau = json.loads((ART / "f2" / f"decode_params_ce{SSFX}.json").read_text())
    t_unseen = min(0.97, max(tau.values()) + 0.1)
    spec = {c: f"infer_hits_ce{SSFX}:{tau.get(c, t_unseen)}" for c in TEST_C}
    subprocess.run([sys.executable, "-W", "ignore", "multi_variant.py", f"ce_all{SSFX}"] + [f"{k}={v}" for k, v in spec.items()], check=True)


if __name__ == "__main__":
    {"check": check, "data": data, "train": train, "score": score_all, "stack": stack, "write": write}[sys.argv[1]]()
