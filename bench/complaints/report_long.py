"""All-complaints past 64k (data/dense-complaints-v1): accuracy by length, paired with each complaint read alone.

For each model scored on dense-complaints-v1 (bench/chunked_scorer.py rows):
  acc            accuracy at alone / 32k / 64k / 128k / 256k
  vs alone       paired by question against the same complaint alone, 95 % bootstrap over complaints (128k and 256k have
                 only 8 and 4 documents, too few to resample documents; each complaint appears twice per long length)
  = own alone    share of questions whose top answer matches the model's own answer with the complaint alone
  edge errors    wrong answers naming the product of the document's first or last complaint
and a consistency check: on the 32k and 64k documents, which are v0's documents as single records, the share of
questions with the same top answer as the model's v0 rows (Kev's unpatched scorer, grouped records).

    uv run python -m bench.complaints.report_long
"""
import json
import random
from collections import defaultdict
from pathlib import Path

from .report import READS as V0_READS, load_rows

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "data" / "dense-complaints-v1"
RES = ROOT / "data" / "kevbench"
BUCKETS = ["alone", "32k", "64k", "128k", "256k"]
MODELS = {  # name -> (rows dir in data/kevbench/results, key in bench.complaints.report.READS for the v0 consistency check)
    "Kev-0.8B 1.0": ("kev-1.0--dense-complaints-v1", "Kev-0.8B 1.0"),
    "Self-taught, fresh-32k": ("fresh-32k--dense-complaints-v1", "Kev-0.8B 1.0 self-taught, fresh-32k (ours)"),
    "Self-taught, fresh-64k": ("fresh-64k--dense-complaints-v1", None),
    "Self-taught, fresh-128k": ("fresh-128k--dense-complaints-v1", None),
}


def top(r):
    return max(range(len(r["p"])), key=r["p"].__getitem__)


def boot(diffs_by_complaint, n=2000, seed=0):
    rng, keys = random.Random(seed), list(diffs_by_complaint)
    flat = [v for k in keys for v in diffs_by_complaint[k]]
    means = sorted(sum(x := [v for _ in keys for v in diffs_by_complaint[rng.choice(keys)]]) / len(x) for _ in range(n))
    return sum(flat) / len(flat), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main():
    index = {i["id"]: i["complaints"] for i in json.load(open(SUITE / "index.json"))}
    label = {qid[1:]: q["label"] for line in open(SUITE / "development.jsonl") for qid, q in json.loads(line)["questions"].items()}
    out = {}
    for name, (folder, v0_key) in MODELS.items():
        path = RES / "results" / folder / "rows.json"
        if not path.exists():
            continue
        rows = json.load(open(path))
        cell = defaultdict(dict)   # bucket -> (doc, complaint) -> (pred key, correct)
        for r in rows:
            doc, b = r["id"].split("#")[0], r["id"].split("/")[1]
            pred = r["keys"][top(r)]
            truth = r["label"] if isinstance(r["label"], str) else r["keys"][r["label"]]
            cell[b][(doc, r["question"][1:])] = (pred, int(pred == truth))
        alone = {c: v for (_, c), v in cell["alone"].items()}
        res = {}
        for b in BUCKETS:
            if b not in cell:
                continue
            items = cell[b]
            e = {"questions": len(items), "acc": sum(v[1] for v in items.values()) / len(items),
                 "documents": len({d for d, _ in items})}
            if b != "alone":
                by = defaultdict(list)
                for (_, c), v in items.items():
                    by[c].append(v[1] - alone[c][1])
                d, lo, hi = boot(by)
                e["vs_alone"] = {"delta": d, "ci95": [lo, hi]}
                e["same_as_alone"] = sum(v[0] == alone[c][0] for (_, c), v in items.items()) / len(items)
                wrong = edge = 0
                for (doc, c), v in items.items():
                    if v[1]:
                        continue
                    seq = index[doc]
                    wrong += 1
                    edge += v[0] in ({label[seq[0]], label[seq[-1]]} - {label[c]})
                e["edge_errors"] = {"wrong": wrong, "edge": edge}
            res[b] = e
        if v0_key and v0_key in V0_READS and all(f.exists() for f in V0_READS[v0_key]):
            v0 = load_rows(V0_READS[v0_key])
            same = n = 0
            for (doc, qid), r in v0.items():
                b = doc.split("/")[1]
                if b not in ("32k", "64k"):
                    continue
                key = (doc.replace("dense-complaints-v0/", "dense-complaints-v1/"), qid[1:])
                if key in cell[b]:
                    n += 1
                    same += cell[b][key][0] == r["keys"][top(r)]
            res["consistency_with_v0"] = {"questions": n, "same_top_answer": same / max(n, 1)}
        out[name] = res
    json.dump(out, open(ROOT / "results" / "complaints_long.json", "w"), indent=1)
    for name, res in out.items():
        print(f"\n{name}")
        for b, e in res.items():
            if b == "consistency_with_v0":
                print(f"  32k/64k vs v0's grouped rows: same top answer on {100 * e['same_top_answer']:.1f}% of {e['questions']} questions")
                continue
            line = f"  {b:>5}: acc {e['acc']:.3f} ({e['questions']} q, {e['documents']} docs)"
            if "vs_alone" in e:
                v = e["vs_alone"]; ee = e["edge_errors"]
                line += (f" | vs alone {100 * v['delta']:+.1f} [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]"
                         f" | = alone {100 * e['same_as_alone']:.0f}% | edge errors {ee['edge']}/{ee['wrong']}")
            print(line)


if __name__ == "__main__":
    main()
