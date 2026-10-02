"""Macro F0.5 exactly as defined in the problem statement (Python standard library only).

    python score_f05.py <predictions.tsv> <ground_truth.tsv>

Both files: header "source1_entity_id<TAB>matched_entity_ids", comma-separated IDs, empty = no matches.
Per Source-1 entity:  F0.5 = 1.25 * TP / (0.25 * |truth| + |predicted|)
    (= (1.25 * P * R) / (0.25 * P + R) with P = TP/|predicted|, R = TP/|truth|)
    both lists empty -> 1.0 ; one empty, the other not -> 0.0
Score = average over every Source-1 entity in the ground truth (entities missing from the predictions count as empty).
"""
import csv
import sys

csv.field_size_limit(1 << 30)


def load(path: str) -> dict[str, set[str]]:
    """Read a results TSV into {source1_entity_id: set of matched ids}."""
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(rd)
        for row in rd:
            ids = row[1] if len(row) > 1 else ""
            out[row[0]] = {x.strip() for x in ids.split(",") if x.strip()}
    return out


def f05(pred: set[str], truth: set[str]) -> float:
    """F0.5 of one Source-1 entity (1.0 when both lists are empty)."""
    if not pred and not truth:
        return 1.0
    tp = len(pred & truth)
    return 1.25 * tp / (0.25 * len(truth) + len(pred))


def score(pred: dict[str, set[str]], truth: dict[str, set[str]]) -> float:
    """Macro F0.5 over every Source-1 entity of the ground truth (missing predictions count as empty)."""
    return sum(f05(pred.get(s1, set()), t) for s1, t in truth.items()) / len(truth)


if __name__ == "__main__":
    p, t = load(sys.argv[1]), load(sys.argv[2])
    print(f"macro F0.5 = {score(p, t):.6f}  over {len(t):,} Source-1 entities")
