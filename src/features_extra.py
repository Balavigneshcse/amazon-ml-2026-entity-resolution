"""Extra pair features (v2): soft token matching, character n-gram overlap, softer house-number comparison.

Computed in a separate pass and stored next to the base features (feats_x/shard_XXX.parquet, same row order as
feats/shard_XXX.parquet), so the base pipeline is untouched. Resumable per shard.

    python features_extra.py <tag> <country> [country ...]
"""
from __future__ import annotations
import sys, time, multiprocessing as mp
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from config import ART
from features import TEXT_COLS, _sides

NAN = float("nan")
X_NAMES = ["nm_tri_jac", "nm_bi_jac", "nm_me_ab", "nm_me_ba", "nm_me_min", "nm_skel_me", "nm_tok_pref3", "nm_len_diff",
           "nm_skel_first_eq",
           "ad_me_ab", "ad_me_ba", "ad_me_min", "ad_num_lev", "ad_num_suffix", "ad_tok_common", "ad_jw", "ad_alpha_sorted_ratio"]


def _grams(s: str, n: int) -> set:
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}


def _jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else NAN


def _me(a: list, b: list) -> float:
    """Monge-Elkan: mean over tokens of a of the best Jaro-Winkler similarity in b."""
    if not a or not b:
        return NAN
    tot = 0.0
    for x in a[:8]:
        best = 0.0
        for y in b[:8]:
            s = JaroWinkler.similarity(x, y)
            if s > best:
                best = s
                if s == 1.0:
                    break
        tot += best
    return tot / len(a[:8])


def _chunk(c: dict) -> np.ndarray:
    n = len(c["nsq_a"])
    out = np.full((n, len(X_NAMES)), NAN, dtype=np.float32)
    for i in range(n):
        f = out[i]
        ca, cb = c["core_a"][i], c["core_b"][i]
        ta, tb = c["toks_a"][i], c["toks_b"][i]
        ka, kb = c["skel_a"][i], c["skel_b"][i]
        f[0] = _jac(_grams(ca, 3), _grams(cb, 3))
        f[1] = _jac(_grams(ca, 2), _grams(cb, 2))
        ab, ba = _me(ta, tb), _me(tb, ta)
        f[2], f[3] = ab, ba
        f[4] = min(ab, ba) if ab == ab and ba == ba else NAN
        f[5] = _me([x for x in ka if x], [x for x in kb if x])
        if ta and tb:
            pb = {t[:3] for t in tb}
            f[6] = sum(1 for t in ta if t[:3] in pb) / len(ta)
        f[7] = abs(len(ca) - len(cb))
        f[8] = 1.0 if ka and kb and ka[0] == kb[0] else 0.0
        if c["miss_a"][i] or c["miss_b"][i]:
            continue
        la, lb = c["alpha_a"][i], c["alpha_b"][i]
        ab, ba = _me(la, lb), _me(lb, la)
        f[9], f[10] = ab, ba
        f[11] = min(ab, ba) if ab == ab and ba == ba else NAN
        na, nb = c["nums_a"][i], c["nums_b"][i]
        if na and nb:
            f[12] = max(Levenshtein.normalized_similarity(x, y) for x in na for y in nb)
            f[13] = float(any(len(x) >= 3 and len(y) >= 3 and (x.endswith(y) or y.endswith(x)) for x in na for y in nb))
        sa, sb = set(c["atoks_a"][i]), set(c["atoks_b"][i])
        f[14] = len(sa & sb) / max(1, min(len(sa), len(sb)))
        f[15] = JaroWinkler.normalized_similarity(c["an_a"][i], c["an_b"][i])
        f[16] = fuzz.ratio(" ".join(sorted(la)), " ".join(sorted(lb))) / 100 if la and lb else NAN
    return out


def build_country_feats_x(tag: str, country: str, procs: int | None = None, limit: int | None = None) -> None:
    """limit: only the first N shards (for quick experiments; the folder is then not marked _DONE)."""
    d = ART / tag / country
    out = d / "feats_x"
    if (out / "_DONE").exists():
        print(f"[featsx] {tag}/{country} done, skip")
        return
    out.mkdir(exist_ok=True)
    shards = sorted((d / "feats").glob("shard_*.parquet"))
    if limit:
        shards = shards[:limit]
    procs = procs or max(1, mp.cpu_count() - 1)
    with mp.Pool(procs) as pool:
        for f in shards:
            o = out / f.name
            if o.exists():
                continue
            t0 = time.time()
            pairs = pl.read_parquet(f, columns=["s1", "r"])
            s1 = pl.scan_parquet(d / "s1_*.parquet").select(TEXT_COLS).filter(pl.col("id").is_in(pairs["s1"].unique().implode())).collect()
            cand = pl.scan_parquet(d / "cand_*.parquet").select(TEXT_COLS).filter(pl.col("id").is_in(pairs["r"].unique().implode())).collect()
            j = pairs.join(s1.rename({"id": "s1"}), on="s1", how="left", maintain_order="left")
            a = _sides(j, "a")
            j = pairs.join(cand.rename({"id": "r"}), on="r", how="left", maintain_order="left")
            b = _sides(j, "b")
            del j
            step = 20000
            chunks = [{k: v[o_:o_ + step] for k, v in {**a, **b}.items()} for o_ in range(0, pairs.height, step)]
            mat = np.vstack(list(pool.imap(_chunk, chunks, chunksize=1)))
            pairs.hstack(pl.DataFrame(mat, schema=X_NAMES)).write_parquet(o)
            print(f"[featsx] {tag}/{country} {f.name}: {pairs.height:,} pairs ({time.time()-t0:.0f}s)", flush=True)
    if not limit:
        (out / "_DONE").write_text("ok")


if __name__ == "__main__":
    for c in sys.argv[2:]:
        build_country_feats_x(sys.argv[1], c)
