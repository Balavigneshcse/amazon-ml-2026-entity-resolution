"""Build the final submission zip.

    python make_package.py <final_variant> [--full US,...]

<final_variant> is a folder under output/ holding the chosen matching_results.tsv (e.g. v4_tl).
candidate_pairs.tsv = exactly the pairs the matching models (stage 2 + 3) score: every blocking candidate that passes
the stage-1 filter (p1 >= 0.005). Countries listed with --full were matched with the older model that scores all 100
blocking candidates, so their candidate list is the full top-100 blocking list.

Output: package/<TEAM>_submission.zip with
    output/matching_results.tsv, output/candidate_pairs.tsv,
    code/business_entity_resolution/{src/, README.md, requirements.txt}, Documentation_template.md
"""
from __future__ import annotations
import os, sys, shutil, subprocess, zipfile
from pathlib import Path
import polars as pl
from config import ART, DATA, ROOT, PMIN

TEAM = "LOGIC_MAKERS"
P_MIN = PMIN


def candidates(full: set) -> pl.DataFrame:
    """candidate_pairs.tsv: exactly the pairs the final models score (stage-1-filtered blocking candidates plus
    dense-retrieval pairs when ER_DN=1; countries in full use the whole top-100 list); one row per entity."""
    parts = []
    for c in sorted(p.name for p in (ART / "t1").iterdir() if p.is_dir()):
        d = ART / "t1" / c
        if c in full:
            lf = pl.scan_parquet(d / "feats" / "shard_*.parquet").select("s1", "r")
        else:
            lf = pl.scan_parquet(d / "infer_p1" / "shard_*.parquet").filter(pl.col("p1") >= P_MIN).select("s1", "r")
        if os.environ.get("ER_DN") == "1" and (d / "dn_new.parquet").exists():     # + dense-retrieval candidates (dense.py)
            lf = pl.concat([lf, pl.scan_parquet(d / "dn_new.parquet").select("s1", "r")]).unique()
        g =lf.group_by("s1").agg(pl.col("r").sort().str.join(",").alias("candidate_entity_ids")).collect(engine="streaming")
        print(f"[package] candidates {c}: {'top-100 blocking list' if c in full else 'after stage-1 filter'}, {g.height:,} entities", flush=True)
        parts.append(g)
    m = pl.concat(parts).rename({"s1": "source1_entity_id"})
    s1_all = pl.read_csv(DATA / "test" / "test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0, columns=["entity_id"])
    return (s1_all.rename({"entity_id": "source1_entity_id"}).join(m, on="source1_entity_id", how="left")
            .with_columns(pl.col("candidate_entity_ids").fill_null("")))


def main() -> None:
    """Build package/<TEAM>_submission.zip with output/, code/business_entity_resolution/ and
    Documentation_template.md at the zip root, after running the organisers' validator."""
    variant = sys.argv[1]
    full = set(sys.argv[sys.argv.index("--full") + 1].split(",")) if "--full" in sys.argv else set()
    final = ROOT / "output" / variant / "matching_results.tsv"
    assert final.exists(), f"missing {final}"
    stage = ROOT / "package" / f"{TEAM}_submission"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "output").mkdir(parents=True)
    shutil.copy2(final, stage / "output" / "matching_results.tsv")
    candidates(full).write_csv(stage / "output" / "candidate_pairs.tsv", separator="\t", quote_style="never")
    code = stage / "code" / "business_entity_resolution"
    (code / "src").mkdir(parents=True)
    for f in sorted((ROOT / "src").iterdir()):
        if f.suffix in (".py", ".bat"):
            shutil.copy2(f, code / "src" / f.name)
    for f in ("README.md", "requirements.txt"):
        shutil.copy2(ROOT / f, code / f)
    shutil.copy2(ROOT / "Documentation.md", stage / "Documentation_template.md")
    (ROOT / "package" / "FINAL_VARIANT.txt").write_text(f"matching_results.tsv = output/{variant}; full-candidate countries: {sorted(full)}\n")
    # validate the staged files with the organisers' script (matches must be a subset of candidates)
    val = ROOT.parent / "dataset" / "student_resource" / "utils" / "validate_submission.py"
    subprocess.run([sys.executable, str(val), "--matching", str(stage / "output" / "matching_results.tsv"),
                    "--candidate", str(stage / "output" / "candidate_pairs.tsv"), "--test-dir", str(DATA / "test")], check=False)
    zpath = ROOT / "package" / f"{TEAM}_submission.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(stage))          # output/, code/, Documentation_template.md at the zip root
    print(f"[package] wrote {zpath}  ({zpath.stat().st_size / 1e6:,.0f} MB)")
    for f in sorted(stage.rglob("*")):
        if f.is_file() and f.stat().st_size > 1e6:
            print(f"           {f.relative_to(stage)}: {f.stat().st_size / 1e6:,.0f} MB")


if __name__ == "__main__":
    main()
