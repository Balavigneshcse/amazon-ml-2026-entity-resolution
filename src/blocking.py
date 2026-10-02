"""Candidate generation (blocking) by weighted multi-key overlap, done with polars hash joins.

Every record emits a set of keys (name tokens, phonetic skeleton tokens, name prefix / whole core, word pairs,
rare address tokens, house-number x address-token and house-number x name-token composites). A Source-1 record
and a Source-2/3 record are candidates if they share a non-hot key; pairs are ranked by an idf-weighted cosine
of the shared keys and the top-K per Source-1 record are kept.

Memory design: records are integer row indices and keys are 64-bit hashes; work is sharded by S1 index (the number
of shards scales with the S1 count) and each shard is written to disk, so an interrupted run resumes at the first
missing shard.
"""
from __future__ import annotations
import os, sys, time, math
import polars as pl
from config import ART, SEED, KBLOCK

FAMS = ["n", "s", "p", "a", "h", "m", "b", "c", "k", "w", "z"]
FAM_ID = {f: i for i, f in enumerate(FAMS)}
CAP_FRAC = 5e-4      # drop keys held by more than this fraction of the country's candidate records
MIN_CAP = 40
COLS = ["name_toks", "name_skel", "name_core", "addr_nums", "addr_alpha"]
NEW_KEYS = os.environ.get("ER_NEWKEYS", "0") == "1"


def _key(fam: str, *parts: pl.Expr) -> pl.Expr:
    """Hash of family-tagged key text; the family is encoded in the low 3 bits so it can be recovered."""
    seq = [pl.lit(fam + "|")]
    for i, p in enumerate(parts):
        seq += ([pl.lit("|")] if i else []) + [p]
    h = pl.concat_str(seq).hash(SEED)
    return (h // 16) * 16 + pl.lit(FAM_ID[fam], dtype=pl.UInt64)


def make_keys(df: pl.DataFrame) -> pl.DataFrame:
    """df has UInt32 'id' (row index) + COLS. Returns unique (id, key:u64)."""
    ge = lambda c, n: pl.col(c).list.eval(pl.element().filter(pl.element().str.len_chars() >= n))
    fr = []
    fr.append(df.select("id", ge("name_toks", 3).alias("k")).explode("k").drop_nulls().select("id", _key("n", pl.col("k")).alias("key")))
    fr.append(df.select("id", ge("name_skel", 3).alias("k")).explode("k").drop_nulls().select("id", _key("s", pl.col("k")).alias("key")))
    fr.append(df.filter(pl.col("name_core").str.len_chars() >= 5).select("id", _key("p", pl.col("name_core").str.slice(0, 5)).alias("key")))
    fr.append(df.filter(pl.col("name_core").str.len_chars() >= 6).select("id", _key("c", pl.col("name_core")).alias("key")))
    fr.append(df.select("id", ge("addr_alpha", 4).alias("k")).explode("k").drop_nulls().select("id", _key("a", pl.col("k")).alias("key")))
    t = df.select("id", ge("name_toks", 3).list.head(6).alias("t")).explode("t").drop_nulls()
    fr.append(t.join(t, on="id").filter(pl.col("t") < pl.col("t_right"))
              .select("id", _key("b", pl.col("t"), pl.col("t_right")).alias("key")))
    num = pl.concat_list([pl.col("addr_nums").list.eval(pl.element().str.tail(4)),
                          pl.col("addr_nums").list.eval(pl.element().filter(pl.element().str.len_chars() >= 4).str.tail(3))]).list.unique()
    nums = df.select("id", num.alias("num"), "addr_alpha", ge("name_toks", 3).alias("nt")).filter(pl.col("num").list.len() > 0)
    fr.append(nums.select("id", "num", "addr_alpha").explode("num").explode("addr_alpha").drop_nulls()
              .select("id", _key("h", pl.col("num"), pl.col("addr_alpha")).alias("key")))
    fr.append(nums.select("id", "num", "nt").explode("num").explode("nt").drop_nulls()
              .select("id", _key("m", pl.col("num"), pl.col("nt")).alias("key")))
    # anagram key: catches domain-style names whose words are swapped ("trustmars.com" vs "Mars Trust")
    fr.append(df.filter(pl.col("name_core").str.len_chars() >= 6)
              .select("id", _key("z", pl.col("name_core").str.split("").list.sort().list.join("")).alias("key")))
    if NEW_KEYS:
        # skeleton pairs (any length: catches short transliterated words) and the whole-name skeleton
        sk = df.select("id", pl.col("name_skel").list.eval(pl.element().filter(pl.element().str.len_chars() >= 1)).list.head(5).alias("t")).explode("t").drop_nulls()
        fr.append(sk.join(sk, on="id").filter(pl.col("t") < pl.col("t_right"))
                  .select("id", _key("k", pl.col("t"), pl.col("t_right")).alias("key")))
        fr.append(df.select("id", pl.col("name_skel").list.join("").alias("w")).filter(pl.col("w").str.len_chars() >= 4)
                  .select("id", _key("w", pl.col("w")).alias("key")))
    return pl.concat(fr).unique()


def _indexed(df: pl.DataFrame) -> pl.DataFrame:
    """Replace the string ids by consecutive integer row numbers (compact join keys)."""
    return df.drop("id").with_row_index("id")


def n_shards_for(n_s1: int) -> int:
    """Number of blocking shards for a country: about 20,000 Source-1 entities per shard, between 4 and 256."""
    return int(min(256, max(4, math.ceil(n_s1 / 20000))))


def block_country(tag: str, country: str, kmax: int = KBLOCK, cap_frac: float = CAP_FRAC, shards=None, out_name: str = "blocks") -> None:
    """Candidate generation for one country: hashed keys of 9 families for entities and records; keys held by more
    than cap_frac of the records are dropped; pairs are ranked by the idf-weighted cosine of their shared keys and
    the top kmax records are kept per Source-1 entity. Written shard by shard to <country>/<out_name>/ (resumable)."""
    d = ART / tag / country
    out = d / out_name
    if (out / "_DONE").exists():
        print(f"[block] {tag}/{country} done, skip")
        return
    out.mkdir(exist_ok=True)
    t0 = time.time()
    cand_ids = pl.read_parquet(d / "cand*.parquet", columns=["id"])["id"]
    ncand = len(cand_ids)
    ck = make_keys(_indexed(pl.read_parquet(d / "cand*.parquet", columns=["id"] + COLS)))
    df_c = ck.group_by("key").agg(pl.len().alias("dfc"))
    cap = max(MIN_CAP, int(cap_frac * ncand))
    df_c = df_c.filter(pl.col("dfc") <= cap).with_columns(
        ((1 + ncand / pl.col("dfc").cast(pl.Float32)).log()).cast(pl.Float32).alias("w"))
    ck = ck.join(df_c, on="key").select(pl.col("id").alias("r"), "key", "w")
    wc = ck.group_by("r").agg(pl.col("w").sum().alias("wr"))
    print(f"[block] {tag}/{country}: cand={ncand:,} cand_keys(kept)={ck.height:,} cap={cap} ({time.time()-t0:.0f}s)", flush=True)
    s1_ids = pl.read_parquet(d / "s1*.parquet", columns=["id"])["id"]
    ns1 = len(s1_ids)
    sk = make_keys(_indexed(pl.read_parquet(d / "s1*.parquet", columns=["id"] + COLS))).rename({"id": "s1"})
    sk = sk.join(df_c.select("key"), on="key")
    dfs = sk.group_by("key").agg(pl.len().alias("dfs")).filter(pl.col("dfs") <= cap)
    sk = sk.join(dfs.select("key"), on="key")
    ws = sk.join(df_c.select("key", "w"), on="key").group_by("s1").agg(pl.col("w").sum().alias("ws"))
    ns = n_shards_for(ns1)
    (out / "_META").write_text(str(ns))
    for k in (range(ns) if shards is None else shards):
        f = out / f"shard_{k:03d}.parquet"
        if f.exists():
            continue
        t1 = time.time()
        j = sk.filter(pl.col("s1") % ns == k).join(ck, on="key")
        fam = pl.col("key") % 16
        agg = j.group_by("s1", "r").agg(
            pl.col("w").sum().alias("score"), pl.len().cast(pl.UInt16).alias("nk"),
            *[pl.when(fam == FAM_ID[fm]).then(pl.col("w")).otherwise(0.0).sum().alias(f"w_{fm}") for fm in FAMS])
        njoin = j.height
        del j
        agg = (agg.join(ws, on="s1").join(wc, on="r")
               .with_columns((pl.col("score") / (pl.col("ws") * pl.col("wr")).sqrt()).alias("cos"))
               .with_columns(pl.col("cos").rank("ordinal", descending=True).over("s1").alias("rank"))
               .filter(pl.col("rank") <= kmax))
        agg = agg.with_columns(s1_ids.gather(agg["s1"]).alias("s1"), cand_ids.gather(agg["r"]).alias("r"))
        agg.write_parquet(f)
        print(f"[block] {tag}/{country} shard {k + 1}/{ns}: joined={njoin:,} kept={agg.height:,} ({time.time()-t1:.0f}s)", flush=True)
    if shards is None:
        (out / "_DONE").write_text("ok")


def eval_blocking(tag: str, country: str) -> None:
    """Print blocking recall@k (share of true pairs among the top-k candidates) for a labelled universe."""
    d = ART / tag / country
    bl = pl.read_parquet(d / "blocks" / "shard_*.parquet", columns=["s1", "r", "rank"])
    gt = pl.read_parquet(d / "gt.parquet")
    n_s1 = pl.read_parquet(d / "s1*.parquet", columns=["id"]).height
    lab = bl.join(gt.with_columns(pl.lit(1).alias("y")), on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
    tot = gt.height
    print(f"== {tag}/{country}: S1={n_s1:,} true pairs={tot:,} | avg cands/S1={bl.height / n_s1:.1f}")
    print("   recall@" + " ".join(f"{kk}:{lab.filter((pl.col('rank') <= kk) & (pl.col('y') == 1)).height / tot:.4f}"
                                  for kk in (5, 10, 20, 40, 60, 100)))
    print(f"   S1 with >=1 cand: {bl['s1'].n_unique() / n_s1:.4f}")


if __name__ == "__main__":
    tag = sys.argv[1]
    for country in sys.argv[2:]:
        block_country(tag, country)
        eval_blocking(tag, country)
