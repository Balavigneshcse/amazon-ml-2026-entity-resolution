"""Cluster-consistency features (stage-2 add-on).

Records that belong to the same Source-1 entity look like each other (they are noisy copies of the same business).
For every candidate with a non-negligible stage-1 score we therefore measure how similar it is to the entity's
already-confident matches ("anchors": up to 3 candidates with p1 >= 0.9). A candidate that is weak against the
reference record but almost identical to a confident match is probably a match too.

    python cluster.py train <tag> <countries | auto> [mname]     # from the out-of-fold p1 of the training universe
    python cluster.py test  <tag> <countries | auto>            # from the test-time p1 (infer_p1)
Output: <country>/anc_<mname>/shard_XXX.parquet or <country>/infer_anc/shard_XXX.parquet with columns (s1, r, ANC_NAMES).
"""
from __future__ import annotations
import sys, time, multiprocessing as mp
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from config import ART, countries as universe_countries

NAN = float("nan")
ANC_NAMES = ["anc_n", "anc_p_max", "anc_name_max", "anc_name_jw_max", "anc_addr_max", "anc_num_eq"]
P_LOW, P_ANCHOR, N_ANCHOR = 0.02, 0.9, 3
TXT = ["id", "name_sq", "name_core", "addr_norm", "addr_missing", "addr_nums"]


def _chunk(c: dict) -> np.ndarray:
    """Pool worker: for (candidate, anchor) pairs, name token-set ratio, core-name Jaro-Winkler, address token-set
    ratio and first-house-number equality."""
    n = len(c["sa"])
    out = np.full((n, 4), NAN, dtype=np.float32)
    for i in range(n):
        out[i, 0] = fuzz.token_set_ratio(c["sa"][i], c["sb"][i]) / 100
        out[i, 1] = JaroWinkler.normalized_similarity(c["ca"][i], c["cb"][i])
        if not (c["ma"][i] or c["mb"][i]):
            out[i, 2] = fuzz.token_set_ratio(c["aa"][i], c["ab"][i]) / 100
            na, nb = c["na"][i], c["nb"][i]
            if na and nb:
                out[i, 3] = 1.0 if na[0] == nb[0] else 0.0
    return out


def build_dir(tag: str, country: str, src_dir: str, out_dir: str, step: int = 50000) -> None:
    """Cluster-consistency features for one country: every candidate with p1 >= P_LOW is compared with the entity's
    anchors (up to N_ANCHOR candidates with p1 >= P_ANCHOR); the best similarity per pair is kept (ANC_NAMES).
    Reads <src_dir>/shard_*.parquet, writes <out_dir>/ (resumable)."""
    d = ART / tag / country
    out = d / out_dir
    if (out / "_DONE").exists():
        print(f"[anc] {tag}/{country}/{out_dir} done, skip")
        return
    out.mkdir(exist_ok=True)
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for f in sorted((d / src_dir).glob("shard_*.parquet")):
            o = out / f.name
            if o.exists():
                continue
            t0 = time.time()
            df = pl.read_parquet(f, columns=["s1", "r", "p1"])
            cand = df.filter(pl.col("p1") >= P_LOW)
            anch = (df.filter(pl.col("p1") >= P_ANCHOR)
                    .with_columns(pl.col("p1").rank("ordinal", descending=True).over("s1").alias("k")).filter(pl.col("k") <= N_ANCHOR)
                    .select("s1", pl.col("r").alias("ra"), pl.col("p1").alias("pa")))
            pr = cand.join(anch, on="s1").filter(pl.col("r") != pl.col("ra"))
            if pr.height == 0:
                pl.DataFrame(schema={"s1": pl.String, "r": pl.String, **{k: pl.Float32 for k in ANC_NAMES}}).write_parquet(o)
                continue
            ids = pl.concat([pr["r"], pr["ra"]]).unique()
            tx = pl.scan_parquet(d / "cand_*.parquet").select(TXT).filter(pl.col("id").is_in(ids.implode())).collect()
            a = pr.join(tx.rename({"id": "r"}), on="r", how="left", maintain_order="left")
            b = pr.join(tx.rename({"id": "ra"}), on="ra", how="left", maintain_order="left")
            lists = dict(sa=a["name_sq"].to_list(), sb=b["name_sq"].to_list(), ca=a["name_core"].to_list(), cb=b["name_core"].to_list(),
                         aa=a["addr_norm"].to_list(), ab=b["addr_norm"].to_list(), ma=a["addr_missing"].to_list(), mb=b["addr_missing"].to_list(),
                         na=a["addr_nums"].to_list(), nb=b["addr_nums"].to_list())
            chunks = [{k: v[o_:o_ + step] for k, v in lists.items()} for o_ in range(0, pr.height, step)]
            mat = np.vstack(list(pool.imap(_chunk, chunks, chunksize=1)))
            pr = pr.hstack(pl.DataFrame(mat, schema=["nm_ts", "nm_jw", "ad_ts", "num_eq"]))
            res = pr.group_by("s1", "r").agg(
                pl.len().cast(pl.Float32).alias("anc_n"), pl.col("pa").max().alias("anc_p_max"),
                pl.col("nm_ts").max().alias("anc_name_max"), pl.col("nm_jw").max().alias("anc_name_jw_max"),
                pl.col("ad_ts").max().alias("anc_addr_max"), pl.col("num_eq").max().alias("anc_num_eq"))
            res.with_columns(pl.col(ANC_NAMES).cast(pl.Float32)).write_parquet(o)
            print(f"[anc] {tag}/{country} {f.name}: {pr.height:,} compares -> {res.height:,} rows ({time.time()-t0:.0f}s)", flush=True)
    (out / "_DONE").write_text("ok")


if __name__ == "__main__":
    mode, tag = sys.argv[1], sys.argv[2]
    countries = list(universe_countries(tag)) if sys.argv[3] == "auto" else sys.argv[3].split(",")
    mname = sys.argv[4] if len(sys.argv) > 4 else "all"
    for c in countries:
        if mode == "train":
            build_dir(tag, c, f"pred_{mname}", f"anc_{mname}")
        else:
            build_dir(tag, c, "infer_p1", "infer_anc")
