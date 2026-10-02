"""Stage 2 v3: precision features for plausible pairs + training at test-like density.

New pair features (computed only for pairs with stage-1 p1 >= P_MIN, ~6-9 per Source-1 entity):
  * cleaned-name comparison: country/ID noise tokens ("(France)", "[India]", "(ID: 74226)") removed before comparing
  * name frequency: idf-weighted coverage of each side's name tokens, rarest missing token, how many Source-1 entities
    and candidate records carry exactly the same cleaned name (generic names need stronger evidence)
  * locality agreement: localities (cities / districts / regions) are learnt from the Source-1 addresses of each
    country (address components without digits that recur >= LOC_MIN times) - no gazetteer, works for unseen countries;
    features: shared localities, conflicting localities, best fuzzy locality similarity
Stage 2 is retrained with these features on the leaderboard-like universe (19 % of Source-1 entities removed, see
sim_drop.py) and validated there.

    python stage2_v3.py prep            record tables (cleaned names, localities, name statistics) for train + test
    python stage2_v3.py feats           pair features (train: pred_d19 pairs, test: infer_p1 pairs)
    python stage2_v3.py train           fit + validate + tune; also a leave-one-country-out check (fit on one labelled country, score another)
    python stage2_v3.py infer           score the test set -> artifacts/t1/<c>/infer_hits_v3
Every command is resumable.
"""
from __future__ import annotations
import os, sys, time, json, math, multiprocessing as mp
os.environ.setdefault("ER_ANC", "1")
os.environ.setdefault("ER_S2", "anc")
import numpy as np
import polars as pl
import lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from config import ART, SEED, PMIN, HSFX, countries
from normalize import STATES, _ascii
from features import add_context
from train import FEATS2, FOLD, PARAMS, add_stage2, with_anc, rp_stats, decode, fscore, train_resumable
import cluster

TRAIN, TEST, MN = "f2", "t1", "all"
TRAIN_C, TEST_C = countries(TRAIN), countries(TEST)       # labelled training countries / all test countries
SUF = "d19"
DROPPED = (pl.col("s1").hash(SEED + 5) % 100) < 19
P_MIN = PMIN
LOC_MIN = 5
NOISE = ["france", "frnce", "india", "usa", "id", "cie"]
V3_NAMES = ["cn_eq", "cn_tset", "cn_ratio", "cn_jac", "cn_cov1", "cn_cov2", "cn_miss_idf1", "cn_miss_idf2", "cn_max_idf_match",
            "loc_n1", "loc_n2", "loc_inter", "loc_unm1", "loc_unm2", "loc_conflict",
            "n_s1_same1", "n_s1_same2", "n_c_same2"]
# sibling support: records of one business in one source share the (re-typed) house number / address, so a candidate
# whose number or address is shared by other candidates of the same Source-1 entity is very likely a true match
SUP_NAMES = ["num_eq", "sup_num_r", "sup_num_r_w", "sup_num_s1", "sup_name_r", "sup_addr_r", "sup_addr_r_w"]
# twin detection (ER_TWIN=1): decoy "twin" businesses = same name + one distinctive extra word, at a nearby house number;
# real copies only add words from a small noise vocabulary (center, services, dba, formerly, ...) learnt from true pairs
TWIN = os.environ.get("ER_TWIN", "0") == "1"
TW_NAMES = ["nov_r", "miss_r", "grp_nov_num", "grp_nov_addr", "grp_novn_num", "twin_flag"]
MSFX = "_tw" if TWIN else ""
FEATS3 = FEATS2 + V3_NAMES + SUP_NAMES + (TW_NAMES if TWIN else [])
TAUS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)


# ------------------------------------------------------------------ record tables
def _rec(df: pl.DataFrame, country: str) -> pl.DataFrame:
    """id, ctoks (cleaned name tokens), ckey (sorted tokens), comps (normalised digit-free address components)."""
    st = STATES.get(country, {})
    toks = pl.col("name_norm").str.split(" ").list.eval(pl.element().filter(
        (pl.element().str.len_chars() > 0) & ~pl.element().is_in(NOISE) & ~pl.element().str.contains(r"^\d+$")))
    comp = _ascii(pl.element()).str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()
    comps = pl.col("addr").fill_null("").str.split(",").list.eval(comp).list.eval(
        pl.element().filter((pl.element().str.len_chars() >= 3) & ~pl.element().str.contains(r"\d"))).list.eval(
        pl.element().replace(st)).list.unique()
    akey = pl.when(pl.col("addr_norm").str.len_chars() > 0).then(pl.col("addr_norm").hash(SEED)).otherwise(None)
    out = df.select("id", toks.alias("ctoks"), comps.alias("comps"), pl.col("addr_nums").list.first().alias("num1"), akey.alias("akey"))
    return out.with_columns(pl.col("ctoks").list.sort().list.join(" ").alias("ckey"))


def prep_country(tag: str, c: str) -> None:
    """Per-country record tables for the v3 features: cleaned names, localities (address components frequent
    in Source 1), name frequencies and token idf; training statistics exclude the dropped entities."""
    d = ART / tag / c
    if (d / "v3_cand.parquet").exists():
        return
    t0 = time.time()
    s1 = _rec(pl.read_parquet(d / "s1_*.parquet", columns=["id", "name_norm", "addr", "addr_nums", "addr_norm"]), c)
    if tag == TRAIN:                                           # statistics must reflect the entities that remain
        s1 = s1.filter(~(pl.col("id").hash(SEED + 5) % 100 < 19))
    n = s1.height
    vocab = (s1.select(pl.col("comps").alias("x")).explode("x").drop_nulls().group_by("x").len()
             .filter(pl.col("len") >= LOC_MIN)["x"])
    df = s1.select(pl.col("ctoks").list.unique().alias("t")).explode("t").drop_nulls().group_by("t").len()
    df.with_columns(((n + 1) / (pl.col("len") + 1)).log().alias("idf")).select("t", "idf").write_parquet(d / "v3_idf.parquet")
    s1_same = s1.filter(pl.col("ckey") != "").group_by("ckey").len().rename({"len": "n_s1_same"})
    cand = _rec(pl.read_parquet(d / "cand_*.parquet", columns=["id", "name_norm", "addr", "addr_nums", "addr_norm"]), c)
    c_same = cand.filter(pl.col("ckey") != "").group_by("ckey").len().rename({"len": "n_c_same"})
    for name, t in (("s1", s1), ("cand", cand)):
        t = (t.with_columns(pl.col("comps").list.eval(pl.element().filter(pl.element().is_in(vocab.to_list()))).alias("locs"))
             .drop("comps").join(s1_same, on="ckey", how="left").join(c_same, on="ckey", how="left")
             .with_columns(pl.col("n_s1_same").fill_null(0), pl.col("n_c_same").fill_null(0)))
        t.write_parquet(d / f"v3_{name}.parquet")
    print(f"[v3 prep] {tag}/{c}: S1={n:,} cand={cand.height:,} localities={len(vocab):,} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ pair features
_IDF: dict = {}
_IDF_MAX = 10.0


def _init(idf: dict, idf_max: float):
    """Pool initialiser: share the token idf table with the workers."""
    global _IDF, _IDF_MAX
    _IDF, _IDF_MAX = idf, idf_max


def _match(a: list, b: list) -> list:
    """For each token of a: matched in b exactly or with Jaro-Winkler >= 0.92."""
    sb = set(b)
    res = []
    for x in a:
        if x in sb:
            res.append(True)
            continue
        res.append(any(JaroWinkler.similarity(x, y) >= 0.92 for y in b[:10]))
    return res


def _chunk(c: dict) -> np.ndarray:
    """Pool worker: cleaned-name similarity, idf-weighted token coverage, rarest missing token and locality
    agreement / conflict features for a chunk of pairs."""
    n = len(c["t1"])
    out = np.full((n, 15), np.nan, dtype=np.float32)
    for i in range(n):
        t1, t2, f = c["t1"][i] or [], c["t2"][i] or [], out[i]
        k1, k2 = c["k1"][i] or "", c["k2"][i] or ""
        if t1 and t2:
            f[0] = 1.0 if k1 == k2 else 0.0
            f[1] = fuzz.token_set_ratio(" ".join(t1), " ".join(t2)) / 100
            f[2] = fuzz.ratio(k1, k2) / 100
            s1, s2 = set(t1), set(t2)
            f[3] = len(s1 & s2) / len(s1 | s2)
            m1, m2 = _match(t1, t2), _match(t2, t1)
            w1 = [_IDF.get(x, _IDF_MAX) for x in t1]
            w2 = [_IDF.get(x, _IDF_MAX) for x in t2]
            f[4] = sum(w for w, m in zip(w1, m1) if m) / max(1e-6, sum(w1))
            f[5] = sum(w for w, m in zip(w2, m2) if m) / max(1e-6, sum(w2))
            f[6] = max([w for w, m in zip(w1, m1) if not m], default=0.0)
            f[7] = max([w for w, m in zip(w2, m2) if not m], default=0.0)
            f[8] = max([w for w, m in zip(w1, m1) if m], default=0.0)
        l1, l2 = c["l1"][i] or [], c["l2"][i] or []
        f[9], f[10] = len(l1), len(l2)
        if l1 and l2:
            # a conflict = each side names a place the other side does not (e.g. same region, different city)
            m1 = [any(x == y or JaroWinkler.similarity(x, y) >= 0.9 for y in l2) for x in l1]
            m2 = [any(x == y or JaroWinkler.similarity(x, y) >= 0.9 for y in l1) for x in l2]
            f[11] = sum(m1)
            f[12] = len(m1) - sum(m1)
            f[13] = len(m2) - sum(m2)
            f[14] = 1.0 if (f[12] > 0 and f[13] > 0) else 0.0
    return out


def pair_feats(tag: str, c: str, src: str, pool_size: int | None = None) -> None:
    """src: folder with (s1, r, p1) shards. Writes <country>/v3/shard_XXX.parquet for pairs with p1 >= P_MIN."""
    d = ART / tag / c
    out = d / "v3"
    if (out / "_DONE").exists():
        return
    out.mkdir(exist_ok=True)
    idf = dict(pl.read_parquet(d / "v3_idf.parquet").iter_rows())
    idf_max = max(idf.values()) if idf else 10.0
    r1 = pl.read_parquet(d / "v3_s1.parquet")
    r2 = pl.read_parquet(d / "v3_cand.parquet")
    with mp.Pool(pool_size or max(1, mp.cpu_count() - 1), initializer=_init, initargs=(idf, idf_max)) as pool:
        shards = sorted((d / src).glob("shard_*.parquet"))
        for i, f in enumerate(shards):
            o = out / f.name
            if o.exists():
                continue
            t0 = time.time()
            p = pl.read_parquet(f, columns=["s1", "r", "p1"]).filter(pl.col("p1") >= P_MIN).select("s1", "r")
            j = (p.join(r1.rename({"id": "s1", "ctoks": "t1", "ckey": "k1", "locs": "l1", "n_s1_same": "n_s1_same1", "n_c_same": "_x"}),
                        on="s1", how="left", maintain_order="left")
                 .join(r2.rename({"id": "r", "ctoks": "t2", "ckey": "k2", "locs": "l2", "n_s1_same": "n_s1_same2", "n_c_same": "n_c_same2"}),
                       on="r", how="left", maintain_order="left"))
            cols = {k: j[k].to_list() for k in ("t1", "t2", "k1", "k2", "l1", "l2")}
            step = 25000
            chunks = [{k: v[o_:o_ + step] for k, v in cols.items()} for o_ in range(0, j.height, step)]
            mat = np.vstack(list(pool.imap(_chunk, chunks, chunksize=1))) if chunks else np.zeros((0, 15), np.float32)
            res = j.select("s1", "r", *[pl.col(x).cast(pl.Float32).log1p().alias(x) for x in ("n_s1_same1", "n_s1_same2", "n_c_same2")])
            res.hstack(pl.DataFrame(mat, schema=V3_NAMES[:15])).write_parquet(o)
            print(f"[v3 feats] {tag}/{c} {i + 1}/{len(shards)}: {j.height:,} pairs ({time.time() - t0:.0f}s)", flush=True)
    (out / "_DONE").write_text("ok")


# ------------------------------------------------------------------ twin detection
from rapidfuzz.distance import Levenshtein, OSA
_VOC: set = set()
_VOC2: dict = {}


def _close(t: str, s_: str) -> bool:
    """True when two tokens are near-identical (Jaro-Winkler >= 0.85 or a small edit distance)."""
    return JaroWinkler.similarity(t, s_) >= 0.85 or Levenshtein.distance(t, s_) <= (1 if len(t) <= 4 else 2)


def _novel(t1: list, t2: list, use_voc: bool = True) -> int:
    """Number of distinctive words in t2 that t1 lacks (ignoring short or numeric words and, if use_voc, words of
    the learnt noise vocabulary)."""
    n = 0
    for t in t2 or []:
        if len(t) < 3 or t.isdigit() or any(_close(t, x) for x in (t1 or [])):
            continue
        if use_voc and (t in _VOC or any(OSA.distance(t, v) <= (2 if len(t) >= 8 else 1) for v in _VOC)):
            continue
        n += 1
    return n


def _init_voc(voc: set):
    """Pool initialiser: share the noise vocabulary (words that genuine copies add) with the workers."""
    global _VOC, _VOC2
    _VOC = voc
    _VOC2 = {}
    for w in voc:
        _VOC2.setdefault(w[:2], []).append(w)


def _nov_chunk(c: dict) -> np.ndarray:
    """Pool worker: distinctive words a record adds and reference words it lacks, for a chunk of pairs."""
    out = np.zeros((len(c["t1"]), 2), dtype=np.float32)
    for i, (t1, t2) in enumerate(zip(c["t1"], c["t2"])):
        out[i, 0] = _novel(t1, t2)                      # distinctive words the record adds
        out[i, 1] = _novel(t2, t1, use_voc=False)       # reference words the record lacks
    return out


def build_vocab() -> set:
    """Words that genuine copies add to a name (learnt from true training pairs of both countries)."""
    f = ART / TRAIN / "noise_vocab.txt"
    if f.exists():
        return set(f.read_text(encoding="utf-8").split())
    from collections import Counter
    cnt = Counter()
    for c in TRAIN_C:
        d = ART / TRAIN / c
        x = (pl.read_parquet(d / "gt.parquet").sample(200_000, seed=SEED)
             .join(pl.read_parquet(d / "v3_s1.parquet", columns=["id", "ctoks"]).rename({"id": "s1", "ctoks": "t1"}), on="s1")
             .join(pl.read_parquet(d / "v3_cand.parquet", columns=["id", "ctoks"]).rename({"id": "r", "ctoks": "t2"}), on="r"))
        for t1, t2 in zip(x["t1"].to_list(), x["t2"].to_list()):
            for t in t2 or []:
                if len(t) >= 3 and not t.isdigit() and not any(_close(t, y) for y in (t1 or [])):
                    cnt[t] += 1
    voc = {w for w, n in cnt.items() if n >= 8}
    f.write_text("\n".join(sorted(voc)), encoding="utf-8")
    print(f"[twin] noise vocabulary: {len(voc)} words", flush=True)
    return voc


def nov_feats(tag: str, c: str, src: str, voc: set) -> None:
    """Twin features for one country: distinctive words each candidate adds to / misses from the entity name
    (resumable)."""
    d = ART / tag / c
    out = d / "nov"
    if (out / "_DONE").exists():
        return
    out.mkdir(exist_ok=True)
    r1 = pl.read_parquet(d / "v3_s1.parquet", columns=["id", "ctoks"]).rename({"id": "s1", "ctoks": "t1"})
    r2 = pl.read_parquet(d / "v3_cand.parquet", columns=["id", "ctoks"]).rename({"id": "r", "ctoks": "t2"})
    with mp.Pool(max(1, mp.cpu_count() - 1), initializer=_init_voc, initargs=(voc,)) as pool:
        shards = sorted((d / src).glob("shard_*.parquet"))
        for i, f in enumerate(shards):
            o = out / f.name
            if o.exists():
                continue
            t0 = time.time()
            p = pl.read_parquet(f, columns=["s1", "r", "p1"]).filter(pl.col("p1") >= P_MIN).select("s1", "r")
            j = p.join(r1, on="s1", how="left", maintain_order="left").join(r2, on="r", how="left", maintain_order="left")
            cols = {k: j[k].to_list() for k in ("t1", "t2")}
            chunks = [{k: v[o_:o_ + 25000] for k, v in cols.items()} for o_ in range(0, j.height, 25000)]
            mat = np.vstack(list(pool.imap(_nov_chunk, chunks, chunksize=1))) if chunks else np.zeros((0, 2), np.float32)
            p.hstack(pl.DataFrame(mat, schema=["nov_r", "miss_r"])).write_parquet(o)
            print(f"[twin] {tag}/{c} {i + 1}/{len(shards)}: {j.height:,} pairs ({time.time() - t0:.0f}s)", flush=True)
    (out / "_DONE").write_text("ok")


def add_twin(df: pl.DataFrame, tabs: tuple, d, shard: str) -> pl.DataFrame:
    """Group-level twin evidence: does any candidate sharing this record's house number / address add a distinctive word?"""
    if not TWIN:
        return df
    a, b = tabs
    df = (df.join(pl.read_parquet(d / "nov" / shard), on=["s1", "r"], how="left")
            .join(a, on="s1", how="left").join(b.select("r", "rn1", "ra"), on="r", how="left"))
    has_n, has_a = pl.col("rn1").is_not_null(), pl.col("ra").is_not_null()
    df = df.with_columns(
        pl.when(has_n).then(pl.col("nov_r").max().over("s1", "rn1")).alias("grp_nov_num"),
        pl.when(has_a).then(pl.col("nov_r").max().over("s1", "ra")).alias("grp_nov_addr"),
        pl.when(has_n).then((pl.col("nov_r") > 0).cast(pl.Int32).sum().over("s1", "rn1")).alias("grp_novn_num"))
    df = df.with_columns(((pl.col("rn1") != pl.col("sn1")).fill_null(False) & (pl.col("grp_nov_num").fill_null(0) > 0)).cast(pl.Float32).alias("twin_flag"))
    return df.drop("sn1", "rn1", "ra").with_columns(pl.col(TW_NAMES).cast(pl.Float32))


# ------------------------------------------------------------------ sibling support
def sup_tables(d) -> tuple:
    """first house number / cleaned-name key / address key per record (read from the normalised universe)."""
    akey = pl.when(pl.col("addr_norm").str.len_chars() > 0).then(pl.col("addr_norm").hash(SEED)).otherwise(None)
    a = pl.read_parquet(d / "s1_*.parquet", columns=["id", "addr_nums"]).select(
        pl.col("id").alias("s1"), pl.col("addr_nums").list.first().alias("sn1"))
    b = (pl.read_parquet(d / "cand_*.parquet", columns=["id", "addr_nums", "addr_norm"])
         .select(pl.col("id").alias("r"), pl.col("addr_nums").list.first().alias("rn1"), akey.alias("ra"))
         .join(pl.read_parquet(d / "v3_cand.parquet", columns=["id", "ckey"]).rename({"id": "r", "ckey": "rk"}), on="r", how="left"))
    return a, b


def add_support(df: pl.DataFrame, tabs: tuple) -> pl.DataFrame:
    """df: plausible pairs of complete Source-1 candidate lists (s1, r, p1). Adds SUP_NAMES."""
    a, b = tabs
    df = df.join(a, on="s1", how="left").join(b, on="r", how="left")
    has_n, has_k, has_a = pl.col("rn1").is_not_null(), pl.col("rk").str.len_chars() > 0, pl.col("ra").is_not_null()
    df = df.with_columns(
        pl.when(has_n & pl.col("sn1").is_not_null()).then((pl.col("rn1") == pl.col("sn1")).cast(pl.Float32)).alias("num_eq"),
        pl.when(has_n).then(pl.len().over("s1", "rn1") - 1).alias("sup_num_r"),
        pl.when(has_n).then(pl.col("p1").sum().over("s1", "rn1") - pl.col("p1")).alias("sup_num_r_w"),
        pl.when(has_k).then(pl.len().over("s1", "rk") - 1).alias("sup_name_r"),
        pl.when(has_a).then(pl.len().over("s1", "ra") - 1).alias("sup_addr_r"),
        pl.when(has_a).then(pl.col("p1").sum().over("s1", "ra") - pl.col("p1")).alias("sup_addr_r_w"))
    df = df.with_columns((pl.col("num_eq").fill_null(0).sum().over("s1")).alias("sup_num_s1"))
    return df.drop("sn1", "rn1", "rk", "ra").with_columns(pl.col(SUP_NAMES).cast(pl.Float32))


# ------------------------------------------------------------------ training frames
def _frames(c: str, per_mille: int) -> tuple:
    """Stage-2 rows (p1 >= P_MIN) at test-like density, one pass: (train = sampled folds 1-4, val = fold 0)."""
    d = ART / TRAIN / c
    rst = pl.read_parquet(d / f"rstats_{SUF}.parquet")
    rp = rp_stats(d, SUF)
    gt = pl.read_parquet(d / "gt.parquet").with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
    tabs = sup_tables(d)
    owned = gt.select("r").unique().with_columns(pl.lit(False).alias("orphan"))
    cols = ["s1", "r", "p1", "y", "fold", "orphan"] + [x for x in FEATS3 if x != "p1"]
    tr, va = [], []
    shards = sorted((d / "feats").glob("shard_*.parquet"))
    for i, f in enumerate(shards):
        t0 = time.time()
        df = add_context(pl.read_parquet(f).filter(~DROPPED), rst)
        df = df.join(pl.read_parquet(d / f"pred_{SUF}" / f.name), on=["s1", "r"], how="left")
        df = with_anc(add_stage2(df, rp), d, MN, f.name, f"anc_{SUF}").with_columns(FOLD)
        df = df.filter((pl.col("p1") >= P_MIN) & ((pl.col("fold") == 0) | ((pl.col("s1").hash(SEED + 3) % 1000) < per_mille)))
        df = add_support(df, tabs)
        df = df.join(pl.read_parquet(d / "v3" / f.name), on=["s1", "r"], how="left")
        df = add_twin(df, tabs, d, f.name)
        df = df.join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
        df = df.join(owned, on="r", how="left").with_columns(pl.col("orphan").fill_null(True)).select(cols)
        va.append(df.filter(pl.col("fold") == 0))
        tr.append(df.filter(pl.col("fold") != 0))
        print(f"[v3 train] {c} frame {i + 1}/{len(shards)} ({time.time() - t0:.0f}s)", flush=True)
    return pl.concat(tr), pl.concat(va)


def _eval(c: str, va: pl.DataFrame, label: str) -> dict:
    """Macro F0.5 per threshold on the validation fold at test-like density; prints and returns {tau: F0.5}."""
    d = ART / TRAIN / c
    s1_ids = (pl.read_parquet(d / "s1_*.parquet", columns=["id", "is_val"]).filter(pl.col("is_val"))
              .rename({"id": "s1"}).filter(~DROPPED)["s1"])
    gt = pl.read_parquet(d / "gt.parquet").filter(pl.col("s1").is_in(s1_ids.implode()))
    res = {t: fscore(decode(va, "p2", t, 0.0, True), gt, s1_ids) for t in TAUS}
    print(f"   [{label}] {c}: " + "  ".join(f"{t}:{v:.4f}" for t, v in res.items()), flush=True)
    return res


def train_all() -> None:
    """Train the v3 stage-2 model on the training folds of every labelled country, compare it with the cluster
    model on validation, save the best threshold, and run a leave-one-country-out check."""
    md = ART / TRAIN / f"models_{MN}"
    frames = {}
    for c in TRAIN_C:
        d = ART / TRAIN / c
        n_tr = pl.scan_parquet(d / "s1_*.parquet").select(pl.col("id").alias("s1")).filter(~DROPPED & (FOLD != 0)).select(pl.len()).collect().item()
        pm = int(min(1000, 1000 * 150_000 / max(1, n_tr)))
        t0 = time.time()
        frames[c] = _frames(c, pm)
        print(f"[v3 train] {c}: train rows={frames[c][0].height:,} val rows={frames[c][1].height:,} ({time.time() - t0:.0f}s)", flush=True)
    base = lgb.Booster(model_file=str(md / "model_s2_anc.txt"))
    tr = pl.concat([frames[c][0] for c in TRAIN_C])
    m3 = train_resumable(tr.select(FEATS3).to_numpy(), tr["y"].to_numpy(), FEATS3, md / f"model_s2_v3{MSFX}.txt", 400)
    table = {}
    print("\n== validation at test-like density (19% of businesses removed), macro F0.5 per threshold")
    for c in TRAIN_C:
        va = frames[c][1]
        _eval(c, va.with_columns(pl.Series("p2", base.predict(va.select(FEATS2).to_numpy(), num_threads=12))), "old anc model")
        table[c] = _eval(c, va.with_columns(pl.Series("p2", m3.predict(va.select(FEATS3).to_numpy(), num_threads=12))), "NEW v3 model ")
    mean = {t: sum(table[c][t] for c in TRAIN_C) / len(TRAIN_C) for t in TAUS}
    best = max(mean, key=mean.get)
    print(f"[v3 train] BEST mean F0.5 {mean[best]:.4f} at tau={best}")
    (ART / TRAIN / f"decode_params_{MN}_v3{MSFX}.json").write_text(json.dumps(dict(tau=best, gate=0.0, assign=True, per_country={c: table[c][best] for c in TRAIN_C})))
    # leave-one-country-out: does v3 help a country the model has never seen? (proxy for the unseen test country)
    held, fit = TRAIN_C[0], TRAIN_C[1:]
    if fit:
        print(f"\n== leave-one-country-out: fit on {', '.join(fit)} only, score {held}")
        tr_fit = pl.concat([frames[c][0] for c in fit])
        va = frames[held][1]
        for label, feats in (("without v3 features", FEATS2), ("with v3 features   ", FEATS3)):
            path = md / f"model_s2_loco_{'v3' if feats is FEATS3 else 'base'}{MSFX}.txt"
            m = train_resumable(tr_fit.select(feats).to_numpy(), tr_fit["y"].to_numpy(), feats, path, 400)
            _eval(held, va.with_columns(pl.Series("p2", m.predict(va.select(feats).to_numpy(), num_threads=12))), label)
    imp = sorted(zip(FEATS3, m3.feature_importance("gain")), key=lambda x: -x[1])
    print("\n   new-feature gains:", [(n, int(g)) for n, g in imp if n in V3_NAMES])


W_EVAL = 8.0   # test has ~6-15x more twin-like orphan records per entity than validation


def fscore_w(va: pl.DataFrame, tau: float, gt: pl.DataFrame, s1_ids: pl.Series, w: float) -> float:
    """Macro F0.5 where a false match on an orphan record counts w times (emulates the test's orphan density)."""
    pred = decode(va, "p2", tau, 0.0, True).join(va.select("s1", "r", "orphan"), on=["s1", "r"], how="left")
    pred = pred.join(gt.with_columns(pl.lit(1).alias("t")), on=["s1", "r"], how="left").with_columns(pl.col("t").fill_null(0))
    per = pred.group_by("s1").agg(pl.col("t").sum().alias("tp"),
                                 ((1 - pl.col("t")) * pl.when(pl.col("orphan")).then(w).otherwise(1.0)).sum().alias("fp"))
    m = (pl.DataFrame({"s1": s1_ids}).join(per, on="s1", how="left").join(gt.group_by("s1").len().rename({"len": "ngt"}), on="s1", how="left")
         .fill_null(0).with_columns(pl.when(pl.col("tp") + pl.col("fp") + pl.col("ngt") == 0).then(1.0)
                                    .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("ngt") + pl.col("tp") + pl.col("fp"))).alias("f")))
    return float(m["f"].mean())


def train_w(X, y, wt, names, path, rounds=400):
    """Resumable LightGBM training with sample weights (trees added in blocks of 100, saved after each block)."""
    ds = lgb.Dataset(X, label=y, weight=wt, feature_name=names, free_raw_data=False)
    booster = lgb.Booster(model_file=str(path)) if path.exists() else None
    done = booster.num_trees() if booster else 0
    while done < rounds:
        booster = lgb.train(PARAMS, ds, num_boost_round=min(100, rounds - done), init_model=booster, keep_training_booster=True)
        done = booster.num_trees()
        booster.save_model(str(path))
        print(f"   {path.name}: trees={done}/{rounds}", flush=True)
    return booster


def train_weighted(ws: list) -> None:
    """Stage 2 v3 with orphan negatives up-weighted; picks each model's threshold on the test-like (weighted) metric."""
    md = ART / TRAIN / f"models_{MN}"
    frames = {}
    for c in TRAIN_C:
        d = ART / TRAIN / c
        n_tr = pl.scan_parquet(d / "s1_*.parquet").select(pl.col("id").alias("s1")).filter(~DROPPED & (FOLD != 0)).select(pl.len()).collect().item()
        frames[c] = _frames(c, int(min(1000, 1000 * 150_000 / max(1, n_tr))))
    tr = pl.concat([frames[c][0] for c in TRAIN_C])
    X, y = tr.select(FEATS3).to_numpy(), tr["y"].to_numpy()
    neg_orphan = ((tr["y"] == 0) & tr["orphan"]).to_numpy()
    print(f"[v3w] train rows={tr.height:,}  orphan negatives={neg_orphan.mean():.3f}", flush=True)
    evals = {}
    for c in TRAIN_C:
        d = ART / TRAIN / c
        s1_ids = (pl.read_parquet(d / "s1_*.parquet", columns=["id", "is_val"]).filter(pl.col("is_val")).rename({"id": "s1"}).filter(~DROPPED)["s1"])
        evals[c] = (frames[c][1], pl.read_parquet(d / "gt.parquet").filter(pl.col("s1").is_in(s1_ids.implode())), s1_ids)
    models = {"v3 (weight 1)": lgb.Booster(model_file=str(md / "model_s2_v3.txt"))}
    for w in ws:
        models[f"v3 orphan-weight {w}"] = train_w(X, y, np.where(neg_orphan, float(w), 1.0), FEATS3, md / f"model_s2_v3_ow{w}.txt")
    print(f"\n== validation: plain macro F0.5  |  test-like F0.5 (orphan false matches x{W_EVAL:g})")
    for name, m in models.items():
        per_tau = {}
        for c in TRAIN_C:
            va, gt, s1_ids = evals[c]
            va = va.with_columns(pl.Series("p2", m.predict(va.select(FEATS3).to_numpy(), num_threads=12)))
            for t in TAUS:
                per_tau.setdefault(t, []).append((fscore(decode(va, "p2", t, 0.0, True), gt, s1_ids), fscore_w(va, t, gt, s1_ids, W_EVAL)))
        best = max(TAUS, key=lambda t: sum(x[1] for x in per_tau[t]))
        plain = sum(x[0] for x in per_tau[best]) / len(TRAIN_C)
        tl = sum(x[1] for x in per_tau[best]) / len(TRAIN_C)
        print(f"   {name:22s}: best test-like tau={best}  plain F0.5={plain:.4f}  test-like F0.5={tl:.4f}   "
              + "  ".join(f"{c}:{per_tau[best][i][1]:.4f}" for i, c in enumerate(TRAIN_C)), flush=True)
        tag = "v3" if name.startswith("v3 (") else f"v3_ow{name.split()[-1]}"
        (ART / TRAIN / f"decode_params_{MN}_{tag}_tl.json").write_text(json.dumps(dict(tau=best, gate=0.0, assign=True, plain=plain, test_like=tl)))


def infer_all(model: str = f"model_s2_v3{MSFX}.txt", hits: str = f"infer_hits_v3{MSFX}{HSFX}") -> None:
    """Score the filtered test candidates of every country with the v3 stage-2 model (keeps p2 >= 0.3)."""
    m3 = lgb.Booster(model_file=str(ART / TRAIN / f"models_{MN}" / model))
    for c in TEST_C:
        d = ART / TEST / c
        out = d / hits
        out.mkdir(exist_ok=True)
        rp = pl.read_parquet(d / f"rp_infer{HSFX}.parquet")
        tabs = sup_tables(d)
        for f in sorted((d / "infer_p1").glob("shard_*.parquet")):
            o = out / f.name
            if o.exists():
                continue
            t0 = time.time()
            df = with_anc(add_stage2(pl.read_parquet(f), rp), d, MN, f.name, "infer_anc").filter(pl.col("p1") >= P_MIN)
            df = add_support(df, tabs)
            df = df.join(pl.read_parquet(d / "v3" / f.name), on=["s1", "r"], how="left")
            df = add_twin(df, tabs, d, f.name)
            p2 = m3.predict(df.select(FEATS3).to_numpy(), num_threads=12)
            df.select("s1", "r").with_columns(pl.Series("p2", p2.astype(np.float32))).filter(pl.col("p2") >= 0.3).write_parquet(o)
            print(f"[v3 infer] {c} {f.name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        for c in TRAIN_C:
            prep_country(TRAIN, c)
        for c in TEST_C:
            prep_country(TEST, c)
    elif cmd == "feats":
        for c in TRAIN_C:
            pair_feats(TRAIN, c, f"pred_{SUF}")
        for c in TEST_C:
            pair_feats(TEST, c, "infer_p1")
    elif cmd == "train":
        for c in TRAIN_C:
            cluster.build_dir(TRAIN, c, f"pred_{SUF}", f"anc_{SUF}")
        train_all()
    elif cmd == "infer":
        infer_all()
    elif cmd == "twinprep":
        voc = build_vocab()
        for c in TRAIN_C:
            nov_feats(TRAIN, c, f"pred_{SUF}", voc)
        for c in TEST_C:
            nov_feats(TEST, c, "infer_p1", voc)
    elif cmd == "trainw":
        train_weighted([int(x) for x in sys.argv[2].split(",")])
    elif cmd == "inferw":
        for w in sys.argv[2].split(","):
            infer_all(f"model_s2_v3_ow{w}.txt", f"infer_hits_v3_ow{w}")
