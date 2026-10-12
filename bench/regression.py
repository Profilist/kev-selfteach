"""Did the long-document delta hurt ordinary decisions? Kev-0.8B 1.0 vs the delta on Kev's public development
partitions (decision-v7, transfer-v4, transfer-v9) and a BABILong set, scored by kev.benchmark (infra/modal/kevbench.py).

Paired by question: the delta's accuracy minus Kev-0.8B 1.0's, with a 95 % bootstrap interval over records.

    uv run modal volume get jev-runs /kevbench data/kevbench/results   # reg-<model>--<suite>
    uv run python -m bench.regression
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "kevbench" / "results"
SUITES = ["decision-v7", "transfer-v4", "transfer-v9", "babilong-mini"]
A, B = "kev-1.0", "labels-32k"


def correct(r):
    k = max(range(len(r["p"])), key=r["p"].__getitem__)
    return int(r["keys"][k] == r["label"] if isinstance(r["label"], str) else k == r["label"])


def rows(model, suite):
    path = RES / f"reg-{model}--{suite}" / "rows.json"
    if not path.exists():   # reads launched by infra/modal/kevbench.py (pipeline) are named <model>--<suite>
        path = RES / f"{model}--{suite}" / "rows.json"
    return {(r["id"], r["question"]): r for r in json.load(open(path))}


def paired(a, b, keys, n=2000, seed=0):
    by_rec = defaultdict(list)
    for k in keys:
        by_rec[k[0]].append(correct(b[k]) - correct(a[k]))
    recs, rng, means = list(by_rec), random.Random(seed), []
    for _ in range(n):
        xs = [x for _ in recs for x in by_rec[rng.choice(recs)]]
        means.append(sum(xs) / len(xs))
    means.sort()
    flat = [x for r in recs for x in by_rec[r]]
    return sum(flat) / len(flat), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main(B=B, suites=SUITES):
    out = {}
    for suite in suites:
        a, b = rows(A, suite), rows(B, suite)
        keys = sorted(set(a) & set(b))
        d, lo, hi = paired(a, b, keys)
        groups = defaultdict(list)
        for k in keys:
            groups[a[k].get("task") or a[k].get("source")].append(k)
        out[suite] = {"questions": len(keys), A: sum(correct(a[k]) for k in keys) / len(keys),
                      B: sum(correct(b[k]) for k in keys) / len(keys), "delta": d, "ci95": [lo, hi],
                      "by_task": {t: {A: sum(correct(a[k]) for k in ks) / len(ks), B: sum(correct(b[k]) for k in ks) / len(ks), "n": len(ks)}
                                  for t, ks in sorted(groups.items())}}
        o = out[suite]
        print(f"{suite:>14}: {A} {o[A]:.3f} | {B} {o[B]:.3f} | paired {100 * d:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}] ({len(keys)} questions)")
    json.dump(out, open(ROOT / "results" / "regression" / f"{B}.json", "w"), indent=1)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:   # e.g. labels-32k-seed1 decision-v7 transfer-v4
        main(sys.argv[1], sys.argv[2:] or SUITES)
    else:
        main()
