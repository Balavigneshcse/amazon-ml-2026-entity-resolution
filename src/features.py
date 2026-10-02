"""Pair feature computation (name / address similarity + blocking evidence + candidate-context features).

Similarity features are computed with rapidfuzz in a multiprocessing pool (all cores). Output is written per
blocking shard so an interrupted run resumes at the first missing shard. Candidate-context features (competition
between Source-1 entities for the same record) need all shards, so they are computed in a separate pass.
"""
from __future__ import annotations
import sys, time, multiprocessing as mp
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from config import ART, KBLOCK

NAN = float("nan")
TEXT_COLS = ["id", "name_sq", "name_core", "name_skel", "name_toks", "is_domain", "addr_norm", "addr_toks",
             "addr_nums", "addr_alpha", "addr_missing"]
SIM_NAMES = [
    "nm_ratio", "nm_tsort", "nm_tset", "nm_partial", "nm_jw", "nm_core_ratio", "nm_core_eq", "nm_tok_jac",
    "nm_tok_contain", "nm_skel_jac", "nm_skel_ratio", "len_a", "len_b", "ntok_a", "ntok_b", "len_ratio",
    "dom_a", "dom_b", "first_tok_eq", "prefix4_eq", "nm_wr",
    "ad_tset", "ad_tsort", "ad_ratio", "ad_partial", "ad_alpha_jac", "ad_alpha_inter", "ad_alpha_contain",
    "ad_num_jac", "ad_num_first_eq", "ad_num_t4", "ad_num_t3", "ad_num_inter", "ad_last_eq", "ad_miss_b",
    "ad_ntok_b", "ad_ntok_a"]
BLOCK_COLS = ["score", "nk", "cos", "rank", "w_n", "w_s", "w_p", "w_a", "w_h", "w_m", "w_b", "w_c", "ws", "wr"]
CTX = ["cos_rel_s1", "cos_gap_s1", "n_cand", "n_s1_for_r", "cos_top1_r", "cos_margin_r", "cos_ratio_r", "cos_rank_r"]


def _jac(a: set, b: set) -> float:
    """Jaccard similarity of two sets (NaN if either is empty)."""
    if not a or not b:
        return NAN
    return len(a & b) / len(a | b)


def _chunk_feats(c: dict) -> np.ndarray:
    """Pool worker: name and address similarity features (SIM_NAMES) for a chunk of pairs; the address features
    are NaN when either address is missing."""
    n = len(c["nsq_a"])
    out = np.empty((n, len(SIM_NAMES)), dtype=np.float32)
    for i in range(n):
        sa, sb = c["nsq_a"][i], c["nsq_b"][i]
        ca, cb = c["core_a"][i], c["core_b"][i]
        ta, tb = set(c["toks_a"][i]), set(c["toks_b"][i])
        ka, kb = c["skel_a"][i], c["skel_b"][i]
        f = out[i]
        f[0] = fuzz.ratio(sa, sb) / 100
        f[1] = fuzz.token_sort_ratio(sa, sb) / 100
        f[2] = fuzz.token_set_ratio(sa, sb) / 100
        f[3] = fuzz.partial_ratio(ca, cb) / 100 if ca and cb else NAN
        f[4] = JaroWinkler.normalized_similarity(ca, cb)
        f[5] = fuzz.ratio(ca, cb) / 100
        f[6] = 1.0 if ca == cb and ca else 0.0
        f[7] = _jac(ta, tb)
        f[8] = len(ta & tb) / min(len(ta), len(tb)) if ta and tb else NAN
        f[9] = _jac(set(ka), set(kb))
        f[10] = fuzz.ratio("".join(ka), "".join(kb)) / 100
        f[11], f[12] = len(ca), len(cb)
        f[13], f[14] = len(c["toks_a"][i]), len(c["toks_b"][i])
        f[15] = min(len(ca), len(cb)) / max(len(ca), len(cb), 1)
        f[16], f[17] = float(c["dom_a"][i]), float(c["dom_b"][i])
        f[18] = 1.0 if c["toks_a"][i] and c["toks_b"][i] and c["toks_a"][i][0] == c["toks_b"][i][0] else 0.0
        f[19] = 1.0 if ca[:4] == cb[:4] and ca else 0.0
        f[20] = fuzz.WRatio(sa, sb) / 100
        miss_b, miss_a = c["miss_b"][i], c["miss_a"][i]
        f[34] = float(miss_b)
        f[35], f[36] = len(c["atoks_b"][i]), len(c["atoks_a"][i])
        if miss_a or miss_b:
            f[21:34] = NAN
            continue
        aa, ab = c["an_a"][i], c["an_b"][i]
        f[21] = fuzz.token_set_ratio(aa, ab) / 100
        f[22] = fuzz.token_sort_ratio(aa, ab) / 100
        f[23] = fuzz.ratio(aa, ab) / 100
        f[24] = fuzz.partial_ratio(aa, ab) / 100
        la, lb = set(c["alpha_a"][i]), set(c["alpha_b"][i])
        inter = len(la & lb)
        f[25] = _jac(la, lb)
        f[26] = inter
        f[27] = inter / min(len(la), len(lb)) if la and lb else NAN
        na, nb = c["nums_a"][i], c["nums_b"][i]
        sna, snb = set(na), set(nb)
        f[28] = _jac(sna, snb)
        f[29] = 1.0 if na and nb and na[0] == nb[0] else (0.0 if na and nb else NAN)
        t4a, t4b = {x[-4:] for x in na}, {x[-4:] for x in nb}
        t3a, t3b = {x[-3:] for x in na if len(x) >= 3}, {x[-3:] for x in nb if len(x) >= 3}
        f[30] = float(bool(t4a & t4b)) if na and nb else NAN
        f[31] = float(bool(t3a & t3b)) if t3a and t3b else NAN
        f[32] = len(sna & snb)
        f[33] = 1.0 if c["atoks_a"][i] and c["atoks_b"][i] and c["atoks_a"][i][-1] == c["atoks_b"][i][-1] else 0.0
    return out


def _sides(df: pl.DataFrame, sfx: str) -> dict:
    """Column lists of one side of the pairs (a = Source-1 entity, b = candidate record) for the feature workers."""
    return {f"nsq_{sfx}": df["name_sq"].to_list(), f"core_{sfx}": df["name_core"].to_list(),
            f"skel_{sfx}": df["name_skel"].to_list(), f"toks_{sfx}": df["name_toks"].to_list(),
            f"dom_{sfx}": df["is_domain"].to_list(), f"an_{sfx}": df["addr_norm"].to_list(),
            f"atoks_{sfx}": df["addr_toks"].to_list(), f"nums_{sfx}": df["addr_nums"].to_list(),
            f"alpha_{sfx}": df["addr_alpha"].to_list(), f"miss_{sfx}": df["addr_missing"].to_list()}


def compute_feats(pairs: pl.DataFrame, s1: pl.DataFrame, cand: pl.DataFrame, pool, step: int = 20000) -> pl.DataFrame:
    """pairs: s1, r + block columns. Returns pairs + similarity features (same row order)."""
    j = pairs.select("s1", "r").join(s1.rename({"id": "s1"}), on="s1", how="left", maintain_order="left")
    a = _sides(j, "a")
    j = pairs.select("s1", "r").join(cand.rename({"id": "r"}), on="r", how="left", maintain_order="left")
    b = _sides(j, "b")
    del j
    chunks = [{k: v[o:o + step] for k, v in {**a, **b}.items()} for o in range(0, pairs.height, step)]
    mat = np.vstack(list(pool.imap(_chunk_feats, chunks, chunksize=1)))
    return pairs.hstack(pl.DataFrame(mat, schema=SIM_NAMES))


def build_country_feats(tag: str, country: str, kfeat: int = KBLOCK, procs: int | None = None) -> None:
    """Similarity features for the top-kfeat blocking candidates of one country, shard by shard with a
    process pool (resumable)."""
    d = ART / tag / country
    out = d / "feats"
    if (out / "_DONE").exists():
        print(f"[feats] {tag}/{country} done, skip")
        return
    out.mkdir(exist_ok=True)
    shards = sorted((d / "blocks").glob("shard_*.parquet"))
    procs = procs or max(1, mp.cpu_count() - 1)
    with mp.Pool(procs) as pool:
        for f in shards:
            o = out / f.name
            if o.exists():
                continue
            t0 = time.time()
            pairs = pl.read_parquet(f).filter(pl.col("rank") <= kfeat)
            # only load text rows needed by this shard
            s1 = pl.scan_parquet(d / "s1_*.parquet").select(TEXT_COLS).filter(pl.col("id").is_in(pairs["s1"].unique().implode())).collect()
            cand = pl.scan_parquet(d / "cand_*.parquet").select(TEXT_COLS).filter(pl.col("id").is_in(pairs["r"].unique().implode())).collect()
            res = compute_feats(pairs, s1, cand, pool)
            res.write_parquet(o)
            print(f"[feats] {tag}/{country} {f.name}: {res.height:,} pairs ({time.time()-t0:.0f}s)", flush=True)
    (out / "_DONE").write_text("ok")


def build_rstats(tag: str, country: str, kfeat: int = KBLOCK) -> None:
    """Per-candidate-record competition summary (top-3 candidate scores over all S1 entities). Streaming, tiny."""
    d = ART / tag / country
    o = d / "rstats.parquet"
    if o.exists():
        return
    st = (pl.scan_parquet(d / "blocks" / "shard_*.parquet").filter(pl.col("rank") <= kfeat).select("r", "cos")
          .group_by("r").agg(pl.col("cos").sort(descending=True).head(3).alias("t"), pl.len().cast(pl.UInt16).alias("n_s1_for_r"))
          .collect(engine="streaming"))
    st = st.select("r", "n_s1_for_r", *[pl.col("t").list.get(i, null_on_oob=True).fill_null(0.0).alias(f"t{i+1}") for i in range(3)])
    st.write_parquet(o)
    print(f"[rstats] {tag}/{country}: {st.height:,} records")


def add_context(df: pl.DataFrame, rstats: pl.DataFrame) -> pl.DataFrame:
    """Candidate-context features: strength inside the S1's own list and competition from other S1 entities."""
    df = df.join(rstats, on="r", how="left")
    comp = pl.when(pl.col("cos") >= pl.col("t1")).then(pl.col("t2")).otherwise(pl.col("t1"))
    return df.with_columns(
        (pl.col("cos") / pl.col("cos").max().over("s1")).alias("cos_rel_s1"),
        (pl.col("cos") - pl.col("cos").max().over("s1")).alias("cos_gap_s1"),
        pl.len().over("s1").alias("n_cand"),
        pl.col("t1").alias("cos_top1_r"),
        (pl.col("cos") - comp).alias("cos_margin_r"),
        (pl.col("cos") / (comp + 1e-6)).alias("cos_ratio_r"),
        (1 + (pl.col("t1") > pl.col("cos")).cast(pl.UInt8) + (pl.col("t2") > pl.col("cos")).cast(pl.UInt8)
         + (pl.col("t3") > pl.col("cos")).cast(pl.UInt8)).alias("cos_rank_r"),
    ).drop("t1", "t2", "t3")


if __name__ == "__main__":
    tag = sys.argv[1]
    for country in sys.argv[2:]:
        build_country_feats(tag, country)
        build_rstats(tag, country)
