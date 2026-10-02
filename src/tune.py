"""Pick decoding thresholds (tau, gate) that maximise macro-F0.5 on the validation fold, averaged over countries.

    python tune.py <train_tag> <countries,comma,sep | auto> [mname]     -> artifacts/<tag>/decode_params_<mname>.json
"""
from __future__ import annotations
import sys, json
import polars as pl
from config import ART, S2TAG, countries as universe_countries
from train import decode, fscore


def tune(tag: str, countries: list, mname: str = "all") -> dict:
    """Choose the decoding threshold and singleton gate that maximise the mean validation macro F0.5 over countries;
    saved to decode_params_<mname>.json."""
    data = {}
    for c in countries:
        d = ART / tag / c
        s1_val = pl.read_parquet(d / "s1_*.parquet", columns=["id", "is_val"]).filter(pl.col("is_val"))["id"]
        gt = pl.read_parquet(d / "gt.parquet").filter(pl.col("s1").is_in(s1_val.implode()))
        data[c] = (pl.read_parquet(d / f"val_pred_{mname}{S2TAG}.parquet"), gt, s1_val)
    best = (0.0, None)
    for tau in (0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        for gate in (0.0, 0.6, 0.7, 0.8, 0.9, 0.95):
            if gate and gate < tau:
                continue
            per = {c: fscore(decode(v, "p2", tau, gate, True), gt, s1) for c, (v, gt, s1) in data.items()}
            avg = sum(per.values()) / len(per)
            if avg > best[0]:
                best = (avg, dict(tau=tau, gate=gate, assign=True, per_country=per))
    print(f"[tune] best mean macro F0.5 = {best[0]:.4f}  params={best[1]}")
    (ART / tag / f"decode_params_{mname}{S2TAG}.json").write_text(json.dumps(best[1]))
    return best[1]


if __name__ == "__main__":
    cs = list(universe_countries(sys.argv[1])) if sys.argv[2] == "auto" else sys.argv[2].split(",")
    tune(sys.argv[1], cs, sys.argv[3] if len(sys.argv) > 3 else "all")
