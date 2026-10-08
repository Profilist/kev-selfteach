"""Analysis of the all-contracts test (data/dense-contracts-v0): accuracy by review-file length, paired by question with
the same question asked of the contract alone, and by question kind (clause found / clause not found / clause category /
governing law).

    uv run modal volume get jev-runs /kevbench/<name> data/kevbench/results/
    uv run python -m bench.contracts.report
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "kevbench"
BUCKETS = ["alone", "16k", "32k", "64k"]
KEV_READS = {"Kev-0.8B 1.0": RES / "results" / "kev-1.0--dense-contracts-v0" / "rows.json",
             "Kev-0.8B 1.0 + complaint-document delta (ours)": RES / "results" / "delta-v1--dense-contracts-v0" / "rows.json",
             "Kev-0.8B 1.0 + complaint-document delta, seed 1 (ours)": RES / "results" / "delta-v1-s1--dense-contracts-v0" / "rows.json",
             "Kev-0.8B 1.0 self-taught, kev-items (ours)": RES / "results" / "kev-items--dense-contracts-v0" / "rows.json",
             "Kev-0.8B 1.0 self-taught, fresh-32k (ours)": RES / "results" / "fresh-32k--dense-contracts-v0" / "rows.json",
             "Kev-0.8B 1.0 self-taught, fresh-32k-seed1 (ours)": RES / "results" / "fresh-32k-seed1--dense-contracts-v0" / "rows.json",
             "Kev-0.8B 1.0 self-taught, fresh-32k-ndas (ours)": RES / "results" / "fresh-32k-ndas--dense-contracts-v0" / "rows.json"}
HOSTED = {"Jev 1.13 (hosted)": ROOT / "data" / "hosted" / "jev_contracts_rows.jsonl",
          "Clef-flash 9B (Workers AI)": ROOT / "data" / "hosted" / "clef_contracts_rows_clef-flash.jsonl",
          "Clef 27B (Workers AI)": ROOT / "data" / "hosted" / "clef_contracts_rows_clef.jsonl"}


def kev_correct(r):
    k = max(range(len(r["p"])), key=r["p"].__getitem__)
    return int(r["keys"][k] == r["label"] if isinstance(r["label"], str) else k == int(r["label"]))


def boot(by_doc, n=2000, seed=0):
    rng, keys = random.Random(seed), list(by_doc)
    flat = [v for k in keys for v in by_doc[k]]
    means = sorted(sum(x := [v for _ in keys for v in by_doc[rng.choice(keys)]]) / len(x) for _ in range(n))
    return sum(flat) / len(flat), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def analyze(correct):
    """correct: {(bucket, doc, qid): 0/1}."""
    alone = {q: v for (b, d, q), v in correct.items() if b == "alone"}
    res = {}
    for b in BUCKETS:
        items = [(d, q, v) for (bb, d, q), v in correct.items() if bb == b]
        if not items:
            continue
        e = {"questions": len(items), "acc": sum(v for _, _, v in items) / len(items)}
        kinds = defaultdict(list)
        for _, q, v in items:
            kinds[q.split("_", 1)[1]].append(v)
        e["by_kind"] = {k: round(sum(v) / len(v), 3) for k, v in sorted(kinds.items())}
        if b != "alone":
            by_doc = defaultdict(list)
            for d, q, v in items:
                if q in alone:
                    by_doc[d].append(v - alone[q])
            delta, lo, hi = boot(by_doc)
            e["paired_vs_alone"] = {"pairs": sum(map(len, by_doc.values())), "delta": delta, "ci95": [lo, hi], "files": len(by_doc)}
        res[b] = e
    return res


def main():
    out = {}
    for name, path in KEV_READS.items():
        if path.exists():
            out[name] = analyze({(r["id"].split("/")[1], r["id"].split("#")[0], r["question"]): kev_correct(r)
                                 for r in json.load(open(path))})
    for name, path in HOSTED.items():
        if path.exists():
            rows = [json.loads(line) for line in open(path)]
            out[name] = analyze({(r["bucket"], r["id"].split("#")[0], r["question"]): r["correct"] for r in rows if r["correct"] is not None})
    json.dump(out, open(ROOT / "results" / "contracts.json", "w"), indent=1)
    for name, res in out.items():
        print(f"\n{name}")
        for b, e in res.items():
            p = e.get("paired_vs_alone")
            pv = f" | vs alone {100 * p['delta']:+.1f} pp [{100 * p['ci95'][0]:+.1f}, {100 * p['ci95'][1]:+.1f}] ({p['pairs']} pairs, {p['files']} files)" if p else ""
            print(f"  {b:>5}: acc {e['acc']:.3f} ({e['questions']} q){pv} | by kind {e['by_kind']}")


if __name__ == "__main__":
    main()
