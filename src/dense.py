"""Dense retrieval on the GPU: recover true matches that the key-based blocking never proposed.

On held-out entities 3-5 % of true pairs are not among the stage-1-filtered candidates (heavily transliterated / phonetic
names such as "praaivett limittedd", domain names, re-typed addresses), capping macro F0.5 at ~0.983 (India) / ~0.990 (US)
even for a perfect matcher. A small bi-encoder (sentence-transformers/all-MiniLM-L6-v2, 22 M parameters, Apache-2.0) is
fine-tuned on true training pairs (in-batch negatives) to embed "name | address"; every Source-1 entity then retrieves its
nearest Source-2/3 records by cosine similarity (exact search on the GPU, same country). The best new pairs (not already
candidates) are scored by both cross-encoders and a stacker decides over old + new candidates together.

    python dense.py data       training pairs (training folds only; hard = true pairs missed by blocking, twice)
    python dense.py train      fine-tune the bi-encoder (1 epoch, fp16)
    python dense.py embed      embed validation-fold and test entities / records
    python dense.py retrieve   exact top-10 neighbours per entity, new-candidate selection, recall report
    python dense.py ce         score the new pairs with both cross-encoders
    python dense.py stack      stacker over old + new candidates, validation comparison, test scores
    python dense.py write      output/dn_all
"""
from __future__ import annotations
import os, sys, time, json
os.environ["ER_CE_TAG"] = ""
CE_ENV = os.environ.get("ER_DN_CE", ",_l12")       # cross-encoders used by the stacker (step 10 adds "_v3")
os.environ["ER_CE_STACK"] = CE_ENV
import numpy as np
import polars as pl
from config import ART, SEED, countries
import gpu_ce as G

BASE = os.environ.get("ER_DN_BASE", "sentence-transformers/all-MiniLM-L6-v2")
DN_DIR = ART / "dn_model"
MAXLEN = 64
P_MIN = 0.02
K = 10                                              # dense neighbours kept per entity
K_NEW = int(os.environ.get("ER_DN_KNEW", "5"))      # at most this many new pairs per entity go to the cross-encoders
KEEP_POS = float(os.environ.get("ER_DN_KEEP", "0.97"))  # similarity floor keeps this share of recoverable true pairs
N_EASY = 700_000
PART = 1_000_000
DROPPED = (pl.col("s1").hash(SEED + 5) % 100) < 19
FOLD = ((pl.col("s1").hash(SEED + 1) % 100) // 20).cast(pl.UInt8)
TRAIN_C, TEST_C = countries("f2"), countries("t1")         # labelled training countries / all test countries
CE_TAGS = tuple(CE_ENV.split(","))
UNIV = [(ART / "f2" / c, "val") for c in TRAIN_C] + [(ART / "t1" / c, "test") for c in TEST_C]


def _ntext(d, which: str) -> pl.LazyFrame:
    """Bi-encoder text of a universe's entities or records: normalised name (lower-cased raw name if empty) | normalised address."""
    f = "s1_*.parquet" if which == "s1" else "cand_*.parquet"
    nm = pl.when(pl.col("name_norm").fill_null("") == "").then(pl.col("name").fill_null("").str.to_lowercase()).otherwise(pl.col("name_norm"))
    return pl.scan_parquet(d / f).select("id", (nm + " | " + pl.col("addr_norm").fill_null("")).str.slice(0, 200).alias("t"))


def _queries(d, kind: str) -> pl.DataFrame:
    """Entities to search for: all Source-1 entities (test) or the validation-fold entities that remain at test-like
    density (val)."""
    q = _ntext(d, "s1")
    if kind == "val":
        v = pl.scan_parquet(d / "s1_*.parquet").select("id", "is_val").filter(pl.col("is_val")).select(pl.col("id").alias("s1")).filter(~DROPPED)
        q = q.join(v.select(pl.col("s1").alias("id")), on="id", how="semi")
    return q.collect().sort("id")


def _old_pairs(d, kind: str) -> pl.DataFrame:
    """Candidates already produced by blocking + the stage-1 filter (validation fold or test)."""
    if kind == "val":
        return pl.read_parquet(d / "ce_val" / "shard_*.parquet", columns=["s1", "r"])
    return pl.scan_parquet(d / "infer_p1" / "shard_*.parquet").filter(pl.col("p1") >= P_MIN).select("s1", "r").collect()


# ------------------------------------------------------------------ data / train
def data() -> None:
    out = ART / "dn_train.parquet"
    if out.exists():
        print("[dn] training pairs exist, skip"); return
    parts = []
    for c in TRAIN_C:
        d = ART / "f2" / c
        gt = pl.read_parquet(d / "gt.parquet").filter((FOLD != 0) & ~DROPPED)
        have = (pl.scan_parquet(d / "pred_d19" / "shard_*.parquet").select("s1", "r", "p1")
                .filter((pl.col("p1") >= P_MIN) & (FOLD != 0)).select("s1", "r").collect())
        hard = gt.join(have, on=["s1", "r"], how="anti")
        easy = gt.join(have, on=["s1", "r"], how="semi")
        easy = easy.sample(min(N_EASY, easy.height), seed=SEED)
        p = pl.concat([hard, hard, easy])
        a = _ntext(d, "s1").join(p.select(pl.col("s1").alias("id")).unique().lazy(), on="id", how="semi").collect().rename({"id": "s1", "t": "ta"})
        b = _ntext(d, "cand").join(p.select(pl.col("r").alias("id")).unique().lazy(), on="id", how="semi").collect().rename({"id": "r", "t": "tb"})
        parts.append(p.join(a, on="s1").join(b, on="r").select("s1", "ta", "tb"))
        print(f"[dn] {c}: {hard.height:,} hard pairs (missed by blocking/filter, used twice) + {easy.height:,} others", flush=True)
    pl.concat(parts).sample(fraction=1.0, shuffle=True, seed=SEED).write_parquet(out)


def _pool(h, mask):
    """Mean pooling over the non-padding tokens, then L2 normalisation."""
    import torch
    m = mask.unsqueeze(-1).to(h.dtype)
    v = (h * m).sum(1) / m.sum(1).clamp(min=1)
    return torch.nn.functional.normalize(v.float(), dim=-1)


def train(batch: int = 256, lr: float = 5e-5, scale: float = 20.0) -> None:
    """Fine-tune the bi-encoder on true training pairs with in-batch negatives (symmetric InfoNCE; other records of the
    same entity are masked), fp16, resumable from checkpoints; saved to artifacts/dn_model/final."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    if (DN_DIR / "final").exists():
        print("[dn] already trained"); return
    dev = "cuda"
    df = pl.read_parquet(ART / "dn_train.parquet")
    ta, tb = df["ta"].to_list(), df["tb"].to_list()
    grp = (df["s1"].hash(SEED) % (1 << 62)).cast(pl.Int64).to_numpy()
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE).to(dev)
    steps_total = len(ta) // batch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * max(0.0, 1 - s / steps_total))
    scaler = torch.amp.GradScaler("cuda")
    DN_DIR.mkdir(parents=True, exist_ok=True)
    ck, step = DN_DIR / "ckpt.pt", 0
    if ck.exists():
        st = torch.load(ck, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        step = st["step"]
        print(f"[dn] resuming at step {step}/{steps_total}", flush=True)
    order = np.random.default_rng(SEED).permutation(len(ta))
    lossf = torch.nn.CrossEntropyLoss()
    model.train()
    t0, run, step0 = time.time(), 0.0, step
    while step < steps_total:
        idx = order[step * batch:(step + 1) * batch]
        ea = tok([ta[i] for i in idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
        eb = tok([tb[i] for i in idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            va = _pool(model(**ea).last_hidden_state, ea["attention_mask"])
            vb = _pool(model(**eb).last_hidden_state, eb["attention_mask"])
        s = (va @ vb.T) * scale
        g = torch.from_numpy(grp[idx]).to(dev)
        same = (g[:, None] == g[None, :]) & ~torch.eye(len(idx), dtype=torch.bool, device=dev)   # other records of the same entity
        s = s.masked_fill(same, -1e4)
        lab = torch.arange(len(idx), device=dev)
        loss = (lossf(s, lab) + lossf(s.T, lab)) / 2
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        step += 1
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 200 == 0:
            el = time.time() - t0
            print(f"[dn] step {step}/{steps_total}  loss {run:.4f}  elapsed {el / 60:.1f} min, remaining ~{el / max(1, step - step0) * (steps_total - step) / 60:.0f} min", flush=True)
        if step % 1000 == 0 or step == steps_total:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step}, ck)
    model.save_pretrained(DN_DIR / "final"); tok.save_pretrained(DN_DIR / "final")
    print(f"[dn] training done ({(time.time() - t0) / 60:.0f} min)")


# ------------------------------------------------------------------ embed / retrieve
def _encoder():
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(DN_DIR / "final")
    model = AutoModel.from_pretrained(DN_DIR / "final").to("cuda").half().eval()
    dim = model.config.hidden_size

    def enc(texts: list[str], bs: int = 1024) -> np.ndarray:
        """Embed texts in length-sorted batches; returns L2-normalised float16 vectors in the input order."""
        out = np.empty((len(texts), dim), dtype=np.float16)
        order = np.argsort(np.fromiter((len(t) for t in texts), dtype=np.int32, count=len(texts)), kind="stable")
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                idx = order[i:i + bs]
                e = tok([texts[j] for j in idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
                out[idx] = _pool(model(**e).last_hidden_state, e["attention_mask"]).half().cpu().numpy()
        return out
    return enc


def embed() -> None:
    """Embed the query entities and all candidate records of every universe (validation fold and test countries);
    records are saved in parts of PART rows (resumable)."""
    enc = _encoder()
    for d, kind in UNIV:
        dn = d / "dn"
        dn.mkdir(exist_ok=True)
        t0 = time.time()
        if not (dn / "q.npy").exists():
            q = _queries(d, kind)
            e = enc(q["t"].to_list())
            q.select("id").write_parquet(dn / "q_ids.parquet"); np.save(dn / "q.npy", e)
            print(f"[dn embed] {d.parent.name}/{d.name}: {q.height:,} entities ({time.time() - t0:.0f}s)", flush=True)
        p = _ntext(d, "cand").collect().sort("id")
        if not (dn / "p_ids.parquet").exists():
            p.select("id").write_parquet(dn / "p_ids.parquet")
        for k, i in enumerate(range(0, p.height, PART)):
            f = dn / f"p_{k:03d}.npy"
            if f.exists():
                continue
            t0 = time.time()
            np.save(f, enc(p["t"][i:i + PART].to_list()))
            print(f"[dn embed] {d.parent.name}/{d.name}: records {min(i + PART, p.height):,}/{p.height:,} ({time.time() - t0:.0f}s)", flush=True)


def _load_p(dn) -> np.ndarray:
    """All record embeddings of a universe, concatenated in p_ids order."""
    return np.concatenate([np.load(f) for f in sorted(dn.glob("p_*.npy"))])


def _search(dn, qn: str = "q") -> pl.DataFrame:
    """Exact top-K cosine search on the GPU: every query entity against every record of its country (chunked to
    fit in 4 GB)."""
    import torch, gc
    gc.collect(); torch.cuda.empty_cache()                 # free cached encoder memory (4 GB GPU)
    q = np.load(dn / f"{qn}.npy")
    Q = torch.from_numpy(q).cuda()
    bv = torch.full((len(q), K), -2.0, device="cuda")
    bi = torch.full((len(q), K), -1, dtype=torch.int64, device="cuda")
    off = 0
    for f in sorted(dn.glob("p_*.npy")):
        P = np.load(f)
        for j in range(0, len(P), 400_000):
            Pc = torch.from_numpy(P[j:j + 400_000]).cuda()
            for i in range(0, len(q), 1024):
                v, ix = (Q[i:i + 1024] @ Pc.T).topk(K, dim=1)
                cv = torch.cat([bv[i:i + 1024], v.float()], 1)
                ci = torch.cat([bi[i:i + 1024], ix + off + j], 1)
                tv, ti = cv.topk(K, dim=1)
                bv[i:i + 1024], bi[i:i + 1024] = tv, ci.gather(1, ti)
            del Pc
        off += len(P)
    qid = pl.read_parquet(dn / f"{qn}_ids.parquet")["id"]
    pid = pl.read_parquet(dn / "p_ids.parquet")["id"]
    bv, bi = bv.cpu().numpy(), bi.cpu().numpy()
    return pl.DataFrame({"s1": qid.gather(np.repeat(np.arange(len(q)), K)), "r": pid.gather(bi.ravel()),
                         "dn_sim": bv.ravel().astype(np.float32), "dn_rank": np.tile(np.arange(1, K + 1, dtype=np.int16), len(q))})


def _pair_sim(dn, pairs: pl.DataFrame) -> pl.DataFrame:
    """Bi-encoder cosine similarity of given (s1, r) pairs (pairs without an embedding are dropped)."""
    q, P = np.load(dn / "q.npy"), _load_p(dn)
    qi = pl.read_parquet(dn / "q_ids.parquet").with_row_index("qi").rename({"id": "s1"})
    pi = pl.read_parquet(dn / "p_ids.parquet").with_row_index("pi").rename({"id": "r"})
    x = pairs.select("s1", "r").join(qi, on="s1", how="left").join(pi, on="r", how="left")
    ok = x.filter(pl.col("qi").is_not_null() & pl.col("pi").is_not_null())
    a, b = ok["qi"].to_numpy(), ok["pi"].to_numpy()
    sim = np.empty(len(a), dtype=np.float32)
    for i in range(0, len(a), 1_000_000):
        sim[i:i + 1_000_000] = (q[a[i:i + 1_000_000]].astype(np.float32) * P[b[i:i + 1_000_000]].astype(np.float32)).sum(1)
    return ok.select("s1", "r").with_columns(pl.Series("dn_sim", sim))


def retrieve() -> None:
    """Nearest-neighbour search for every universe; the similarity floor is chosen on validation (keeps KEEP_POS of
    the true pairs retrieval can add); prints the recall report; writes the new candidate pairs (at most K_NEW
    per entity, not already candidates) to dn_new.parquet."""
    for d, kind in UNIV:
        dn = d / "dn"
        if not (dn / "top.parquet").exists():
            t0 = time.time()
            _search(dn).write_parquet(dn / "top.parquet")
            print(f"[dn retrieve] {d.parent.name}/{d.name}: top-{K} search done ({time.time() - t0:.0f}s)", flush=True)
        if not (dn / "old_sim.parquet").exists():
            _pair_sim(dn, _old_pairs(d, kind)).write_parquet(dn / "old_sim.parquet")
    # similarity floor from validation: keep KEEP_POS of the true pairs that retrieval can add
    newv = []
    for c in TRAIN_C:
        d = ART / "f2" / c
        new = (pl.read_parquet(d / "dn" / "top.parquet").join(_old_pairs(d, "val"), on=["s1", "r"], how="anti")
               .with_columns(pl.col("dn_sim").rank("ordinal", descending=True).over("s1").alias("new_rank")))
        gt = pl.read_parquet(d / "gt.parquet")
        newv.append(new.join(gt.with_columns(pl.lit(1).alias("y")), on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0), pl.lit(c).alias("cty")))
    newv = pl.concat(newv)
    pos = newv.filter((pl.col("y") == 1) & (pl.col("new_rank") <= K_NEW))
    floor = float(pos["dn_sim"].quantile(1 - KEEP_POS))
    (ART / "f2" / "dn_params.json").write_text(json.dumps({"floor": floor, "k_new": K_NEW}))
    print(f"\n== dense retrieval on validation (similarity floor {floor:.3f})")
    for c in TRAIN_C:
        d = ART / "f2" / c
        s1_ids = pl.read_parquet(d / "dn" / "q_ids.parquet")["id"]
        gt = pl.read_parquet(d / "gt.parquet").filter(pl.col("s1").is_in(s1_ids.implode()))
        old = _old_pairs(d, "val")
        miss = gt.join(old, on=["s1", "r"], how="anti").height
        v = newv.filter(pl.col("cty") == c)
        for k in (1, 2, 3, 5, 10):
            s = v.filter((pl.col("new_rank") <= k) & (pl.col("dn_sim") >= floor))
            print(f"   {c}: new pairs rank<={k:2d}: {s.height / len(s1_ids):.2f} per entity, recovers {s['y'].sum():,} of {miss:,} missed true pairs"
                  f"  -> candidate recall {1 - (miss - s['y'].sum()) / gt.height:.4f} (was {1 - miss / gt.height:.4f})", flush=True)
    for d, kind in UNIV:
        new = (pl.read_parquet(d / "dn" / "top.parquet").join(_old_pairs(d, kind), on=["s1", "r"], how="anti")
               .with_columns(pl.col("dn_sim").rank("ordinal", descending=True).over("s1").alias("new_rank"))
               .filter((pl.col("new_rank") <= K_NEW) & (pl.col("dn_sim") >= floor)).sort("s1", "r"))
        new.select("s1", "r", "dn_sim", "dn_rank").write_parquet(d / "dn_new.parquet")
        print(f"[dn retrieve] {d.parent.name}/{d.name}: {new.height:,} new candidate pairs", flush=True)


# ------------------------------------------------------------------ step 10: a third cross-encoder trained on dense-retrieval pairs too
def trainpairs() -> None:
    """Dense neighbours of the training-fold entities (same selection as validation / test) -> hard pairs for cross-encoder 3."""
    prm = json.loads((ART / "f2" / "dn_params.json").read_text())
    enc = None
    for c in TRAIN_C:
        d = ART / "f2" / c
        dn, out = d / "dn", d / "dn_new_tr.parquet"
        if out.exists():
            continue
        if not (dn / "qtr.npy").exists():
            enc = enc or _encoder()
            ids = pl.scan_parquet(d / "s1_*.parquet").select(pl.col("id").alias("s1")).filter((FOLD != 0) & ~DROPPED).select(pl.col("s1").alias("id"))
            q = _ntext(d, "s1").join(ids, on="id", how="semi").collect().sort("id")
            np.save(dn / "qtr.npy", enc(q["t"].to_list())); q.select("id").write_parquet(dn / "qtr_ids.parquet")
            enc = None
            print(f"[dn trainpairs] {c}: embedded {q.height:,} training-fold entities", flush=True)
        t0 = time.time()
        top = _search(dn, "qtr")
        old = (pl.scan_parquet(d / "pred_d19" / "shard_*.parquet").select("s1", "r", "p1")
               .filter((pl.col("p1") >= P_MIN) & (FOLD != 0)).select("s1", "r").collect())
        new = (top.join(old, on=["s1", "r"], how="anti").with_columns(pl.col("dn_sim").rank("ordinal", descending=True).over("s1").alias("new_rank"))
               .filter((pl.col("new_rank") <= prm["k_new"]) & (pl.col("dn_sim") >= prm["floor"])))
        new.select("s1", "r").write_parquet(out)
        print(f"[dn trainpairs] {c}: {new.height:,} dense-retrieval training pairs ({time.time() - t0:.0f}s)", flush=True)


def cedata(n_old: int = 500_000, n_new: int = 300_000) -> None:
    """Training pairs for cross-encoder 3: a sample of filtered candidate pairs plus dense-retrieval pairs of the
    training folds, labelled from the ground truth."""
    out = ART / "ce_train_v3.parquet"
    if out.exists():
        print("[dn cedata] exists, skip"); return
    parts = []
    for c in TRAIN_C:
        d = ART / "f2" / c
        old = (pl.scan_parquet(d / "pred_d19" / "shard_*.parquet").select("s1", "r", "p1")
               .filter((pl.col("p1") >= P_MIN) & (FOLD != 0)).select("s1", "r").collect())
        old = old.sample(min(n_old, old.height), seed=SEED + 13)
        new = pl.read_parquet(d / "dn_new_tr.parquet")
        new = new.sample(min(n_new, new.height), seed=SEED + 14)
        gt = pl.read_parquet(d / "gt.parquet").with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
        p = pl.concat([old, new]).join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
        parts.append(G.pairs_text(d, p).select("ta", "tb", "y"))
        yn = new.join(gt, on=["s1", "r"], how="semi").height
        print(f"[dn cedata] {c}: {old.height:,} candidate pairs + {new.height:,} dense pairs ({yn / max(1, new.height):.3f} true)", flush=True)
    pl.concat(parts).sample(fraction=1.0, shuffle=True, seed=SEED).write_parquet(out)


# ------------------------------------------------------------------ cross-encoders on the new pairs
def ce() -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    for tag in CE_TAGS:
        path = ART / f"ce_model{tag}" / "final"
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to("cuda").half().eval()
        for d, kind in UNIV:
            out = d / f"dn_ce{tag}"
            out.mkdir(exist_ok=True)
            p = pl.read_parquet(d / "dn_new.parquet", columns=["s1", "r"])
            for k, i in enumerate(range(0, p.height, 200_000)):
                o = out / f"shard_{k:03d}.parquet"
                if o.exists():
                    continue
                t0 = time.time()
                x = G.pairs_text(d, p[i:i + 200_000])
                ta, tb = x["ta"].to_list(), x["tb"].to_list()
                s = np.empty(len(ta), dtype=np.float32)
                with torch.no_grad():
                    for j in range(0, len(ta), 512):
                        e = tok(ta[j:j + 512], tb[j:j + 512], truncation=True, max_length=G.MAXLEN, padding=True, return_tensors="pt").to("cuda")
                        s[j:j + 512] = torch.sigmoid(model(**e).logits.squeeze(-1).float()).cpu().numpy()
                x.select("s1", "r").with_columns(pl.Series(f"ce{tag}", s)).write_parquet(o)
                print(f"[dn ce{tag}] {d.parent.name}/{d.name} {min(i + 200_000, p.height):,}/{p.height:,} ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ stacker over old + new candidates
DN_FEATS = ["dn_sim", "dn_rank", "is_new", "dn_gap_s1"]
V2 = os.environ.get("ER_DN_V2") == "1"            # + record-side competition features (how many entities want this record)
RF = ["r_n", "r_rank", "r_sum", "r_other", "r_n_any"] if V2 else []
VS = ("2" if V2 else "") + ("3" if "_v3" in CE_TAGS else "")


def _rside(d, kind: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stage-1 scores of every entity competing for the same record (all entities, not only the validation fold)."""
    p = pl.scan_parquet(d / ("pred_d19" if kind == "val" else "infer_p1") / "shard_*.parquet").select("s1", "r", "p1").filter(pl.col("p1") >= P_MIN)
    if kind == "val":
        p = p.filter(~DROPPED)
    p = p.collect().with_columns(pl.len().over("r").cast(pl.Float32).alias("r_n"),
                                 pl.col("p1").rank("ordinal", descending=True).over("r").cast(pl.Float32).alias("r_rank"),
                                 pl.col("p1").sum().over("r").alias("r_sum"))
    t = p.group_by("r").agg(pl.col("p1").top_k(2).alias("t"), pl.len().cast(pl.Float32).alias("r_n_any"))
    t = t.select("r", "r_n_any", pl.col("t").list.get(0).alias("m1"), pl.col("t").list.get(1, null_on_oob=True).fill_null(0.0).alias("m2"))
    p = p.join(t.select("r", "m1", "m2"), on="r").with_columns(pl.when(pl.col("r_rank") == 1).then(pl.col("m2")).otherwise(pl.col("m1")).alias("r_other"))
    return p.select("s1", "r", "r_n", "r_rank", "r_sum", "r_other"), t.select("r", "r_n_any")


def _union(d, kind: str) -> pl.DataFrame:
    """Stacker input for one universe: old candidates (cross-encoder, stage-1 and cluster-model scores) plus the new
    dense-retrieval pairs, with dense similarity / rank features, record-competition features (V2) and the
    per-entity cross-encoder features."""
    top = pl.read_parquet(d / "dn" / "top.parquet", columns=["s1", "r", "dn_rank"])
    if kind == "val":
        old = G._ce_join(pl.read_parquet(d / "ce_val" / "shard_*.parquet").select("s1", "r"), d, "ce_val")
        old = old.join(pl.read_parquet(d / "val_pred_all_anc_d19.parquet"), on=["s1", "r"], how="left")
        p1 = pl.scan_parquet(d / "val_d19" / "shard_*.parquet").select("s1", "r", "p1")
    else:
        old = G._ce_join(pl.scan_parquet(d / "infer_p1" / "shard_*.parquet").select("s1", "r", "p1").filter(pl.col("p1") >= P_MIN).collect(), d, "ce")
        old = old.join(pl.read_parquet(d / "infer_hits_anc_p020" / "shard_*.parquet").select("s1", "r", "p2"), on=["s1", "r"], how="left")
        p1 = pl.scan_parquet(d / "infer_p1" / "shard_*.parquet").select("s1", "r", "p1")
    old = old.join(pl.read_parquet(d / "dn" / "old_sim.parquet"), on=["s1", "r"], how="left").with_columns(pl.lit(0, dtype=pl.Int8).alias("is_new"))
    new = pl.read_parquet(d / "dn_new.parquet").select("s1", "r", "dn_sim")
    for tag in CE_TAGS:
        new = new.join(pl.read_parquet(d / f"dn_ce{tag}" / "shard_*.parquet"), on=["s1", "r"], how="left")
    new = new.join(p1.join(new.select("s1", "r").lazy(), on=["s1", "r"], how="semi").collect(), on=["s1", "r"], how="left")
    new = new.with_columns(pl.lit(None, dtype=pl.Float32).alias("p2"), pl.lit(1, dtype=pl.Int8).alias("is_new"))
    cols = ["s1", "r", "p1", "p2", "dn_sim", "is_new"] + [f"ce{t}" for t in CE_TAGS]
    x = pl.concat([old.select(cols), new.select(cols)], how="vertical_relaxed")
    x = x.join(top, on=["s1", "r"], how="left").with_columns(
        pl.col("dn_rank").fill_null(K + 1).cast(pl.Float32), pl.col("dn_sim").fill_null(0.0), pl.col("p1").fill_null(0.0),
        pl.when(pl.col("p2") >= 0.3).then(pl.col("p2")).otherwise(0.0).fill_null(0.0).alias("p2c"))
    x = x.with_columns((pl.col("dn_sim") - pl.col("dn_sim").max().over("s1")).alias("dn_gap_s1"))
    if V2:
        pr, rn = _rside(d, kind)
        x = x.join(pr, on=["s1", "r"], how="left").join(rn, on="r", how="left").with_columns([pl.col(f).fill_null(0.0) for f in RF])
    return G._feats(x)


def stack() -> None:
    """Fit the stacker over old + new candidates on the validation fold (2-fold cross-fitting by entity for an honest
    comparison), report macro F0.5 per threshold, save the model and thresholds, score the test candidates."""
    import lightgbm as lgb
    from train import decode, fscore, PARAMS
    frames, evals = [], {}
    for c in TRAIN_C:
        d = ART / "f2" / c
        x = _union(d, "val")
        gt = pl.read_parquet(d / "gt.parquet")
        x = x.join(gt.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["s1", "r"], how="left").with_columns(
            pl.col("y").fill_null(0), pl.lit(c).alias("cty"), (pl.col("s1").hash(SEED + 11) % 2).alias("half"))
        s1_ids = pl.read_parquet(d / "dn" / "q_ids.parquet")["id"]
        evals[c] = (gt.filter(pl.col("s1").is_in(s1_ids.implode())), s1_ids)
        frames.append(x)
    x = pl.concat(frames)
    feats = G.S_FEATS + DN_FEATS + RF
    params = dict(PARAMS, num_leaves=63, min_data_in_leaf=200)
    h, y = x["half"].to_numpy(), x["y"].to_numpy()
    X = x.select(feats).to_numpy()
    pred = np.zeros(x.height, dtype=np.float32)
    for k in (0, 1):
        m = lgb.train(params, lgb.Dataset(X[h != k], y[h != k]), num_boost_round=300)
        pred[h == k] = m.predict(X[h == k], num_threads=12)
    x = x.with_columns(pl.Series("pst", pred))
    best = {}
    print("\n== validation (test-like universe), macro F0.5: two cross-encoders on old candidates vs + dense-retrieval candidates")
    for c in TRAIN_C:
        gt, s1_ids = evals[c]
        xc = x.filter(pl.col("cty") == c)
        orc_old = fscore(xc.filter((pl.col("y") == 1) & (pl.col("is_new") == 0)).select("s1", "r"), gt, s1_ids)
        orc_new = fscore(xc.filter(pl.col("y") == 1).select("s1", "r"), gt, s1_ids)
        res = {t: fscore(decode(xc.select("s1", "r", pl.col("pst").alias("p2")), "p2", t, 0.0, True), gt, s1_ids) for t in G.TAUS}
        res_old = {t: fscore(decode(xc.filter(pl.col("is_new") == 0).select("s1", "r", pl.col("pst").alias("p2")), "p2", t, 0.0, True), gt, s1_ids) for t in G.TAUS}
        bt, bo = max(res, key=res.get), max(res_old, key=res_old.get)
        best[c] = bt
        print(f"   {c}: ceiling {orc_old:.4f} -> {orc_new:.4f} | old candidates only {res_old[bo]:.4f} (tau {bo})  ->  with new candidates {res[bt]:.4f} (tau {bt})   ["
              + " ".join(f"{t}:{v:.4f}" for t, v in res.items()) + "]", flush=True)
    m = lgb.train(params, lgb.Dataset(X, y), num_boost_round=300)
    m.save_model(str(ART / "f2" / "models_all" / f"model_dn{VS}_stack.txt"))
    (ART / "f2" / f"decode_params_dn{VS}.json").write_text(json.dumps(best))
    for c in TEST_C:
        d = ART / "t1" / c
        t = _union(d, "test")
        t = t.select("s1", "r").with_columns(pl.Series("p2", m.predict(t.select(feats).to_numpy(), num_threads=12).astype(np.float32)))
        out = d / f"infer_hits_dn{VS}"
        out.mkdir(exist_ok=True)
        t.filter(pl.col("p2") >= 0.3).write_parquet(out / "shard_000.parquet")
        print(f"[dn stack] test {c}: scored {t.height:,} pairs", flush=True)


def write() -> None:
    """Write output/dn<suffix>_all: labelled countries use their validation-tuned threshold; every country without
    training labels (here France) uses ER_DN_FR (default 0.95, preferred over 0.85 on the leaderboard)."""
    import subprocess
    tau = json.loads((ART / "f2" / f"decode_params_dn{VS}.json").read_text())
    t_unseen = float(os.environ.get("ER_DN_FR", "0.95"))
    h = f"infer_hits_dn{VS}"
    spec = {c: f"{h}:{tau.get(c, t_unseen)}" for c in TEST_C}
    subprocess.run([sys.executable, "-W", "ignore", "multi_variant.py", f"dn{VS}_all"] + [f"{k}={v}" for k, v in spec.items()], check=True)


if __name__ == "__main__":
    {"trainpairs": trainpairs, "cedata": cedata, "data": data, "train": train, "embed": embed, "retrieve": retrieve, "ce": ce, "stack": stack, "write": write}[sys.argv[1]]()
