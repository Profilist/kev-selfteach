"""Analysis of the all-complaints ("dense") test: accuracy by document length, paired by complaint against the same
complaint read alone, plus where errors come from (a neighbour's product?) and where in the document they happen.

Rows come from kev.benchmark reads of data/dense-complaints-v0, possibly split across runs (a cancelled read's
predictions.jsonl, plus top-up reads whose records may hold one document's questions in groups: ids ending "#qNN").

    uv run python -m bench.complaints.report
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "data" / "dense-complaints-v0"
RES = ROOT / "data" / "kevbench"
READS = {  # model -> files holding its rows (predictions.jsonl lines carry "rows"; rows.json is a list)
    "Kev-0.8B 1.0": [RES / "partial2" / "dense-kev-1.0.predictions.jsonl", RES / "results" / "dense-kev-1.0-tail" / "predictions.jsonl",
                     RES / "results" / "dense-kev-1.0-tail2" / "rows.json"],
    "Kev-0.8B 1.0 + complaint-document delta (ours)": [RES / "results" / "delta-v1--dense-complaints-v0-split" / "rows.json"],
    "Kev-0.8B 1.0 + complaint-document delta, seed 1 (ours)": [RES / "results" / "delta-v1-s1--dense-complaints-v0-split" / "rows.json"],
    "Kev-0.8B 1.0 self-taught, kev-items (ours)": [RES / "results" / "kev-items--dense-complaints-v0-split" / "rows.json"],
    "Kev-0.8B 1.0 self-taught, fresh-32k (ours)": [RES / "results" / "fresh-32k--dense-complaints-v0-split" / "rows.json"],
    "Kev-0.8B 1.0 self-taught, fresh-32k-seed1 (ours)": [RES / "results" / "fresh-32k-seed1--dense-complaints-v0-split-long" / "rows.json"],
    "Kev-0.8B 1.0 self-taught, fresh-32k-ndas (ours)": [RES / "results" / "fresh-32k-ndas--dense-complaints-v0-split-long" / "rows.json"],
    "Kev-0.8B + long-document training (ours)": [RES / "partial2" / "dense-full-long.predictions.jsonl",
                                                 RES / "results" / "dense-full-long-tail" / "predictions.jsonl",
                                                 RES / "results" / "dense-full-long-tail2" / "rows.json"],
}
BUCKETS = ["alone", "8k", "16k", "32k", "64k"]
JEV = ROOT / "data" / "hosted" / "jev_dense_rows.jsonl"


def load_rows(files):
    rows = {}
    for f in files:
        items = json.load(open(f)) if f.suffix == ".json" else [r for line in open(f) if line.strip() for r in json.loads(line)["rows"]]
        for r in items:
            doc = r["id"].split("#")[0]
            rows[(doc, r["question"])] = r
    return rows


def boot(values_by_cluster, n=2000, seed=0):
    """Mean of all values and its 95 % interval, resampling clusters (documents) with replacement."""
    rng = random.Random(seed)
    keys = list(values_by_cluster)
    flat = [v for k in keys for v in values_by_cluster[k]]
    means = []
    for _ in range(n):
        pick = [v for _ in keys for v in values_by_cluster[rng.choice(keys)]]
        means.append(sum(pick) / len(pick))
    means.sort()
    return sum(flat) / len(flat), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main():
    index = json.load(open(SUITE / "index.json"))
    doc_of = {(i["bucket"], c): i["id"] for i in index for c in i["complaints"]}
    order = {i["id"]: i["complaints"] for i in index}
    label = {}
    for line in open(SUITE / "development.jsonl"):
        r = json.loads(line)
        for qid, q in r["questions"].items():
            label[qid[1:]] = q["label"]
    def analyze(pred, correct):
        res = {}
        alone = {c: v for (b, c), v in correct.items() if b == "alone"}
        for b in BUCKETS:
            cs = [c for (bb, c) in correct if bb == b]
            if not cs:
                continue
            entry = {"questions": len(cs), "acc": sum(correct[(b, c)] for c in cs) / len(cs)}
            if b != "alone":
                by_doc = defaultdict(list)
                for c in cs:
                    if c in alone:
                        by_doc[doc_of[(b, c)]].append(correct[(b, c)] - alone[c])
                d, lo, hi = boot(by_doc)
                entry["paired_vs_alone"] = {"pairs": sum(map(len, by_doc.values())), "delta": d, "ci95": [lo, hi], "documents": len(by_doc)}
                # errors that name a neighbour's product (previous or next complaint in the document), against the same
                # complaints' errors when read alone (no neighbours, so any match is chance)
                hit_long = hit_alone = n_long = n_alone = 0
                for doc in {doc_of[(b, c)] for c in cs}:
                    seq = order[doc]
                    for i, c in enumerate(seq):
                        neigh = {label[x] for x in seq[max(0, i - 1):i] + seq[i + 1:i + 2]} - {label[c]}
                        if not neigh:
                            continue
                        if (b, c) in correct and not correct[(b, c)]:
                            n_long += 1; hit_long += pred[(b, c)] in neigh
                        if c in alone and not alone[c]:
                            n_alone += 1; hit_alone += pred[("alone", c)] in neigh
                entry["neighbour_errors"] = {"long": [hit_long, n_long], "alone_same_complaints": [hit_alone, n_alone]}
                pos = defaultdict(list)   # accuracy by position (fifths of the document)
                for doc in {doc_of[(b, c)] for c in cs}:
                    seq = order[doc]
                    for i, c in enumerate(seq):
                        if (b, c) in correct:
                            pos[min(4, 5 * i // len(seq))].append(correct[(b, c)])
                entry["acc_by_fifth"] = [round(sum(v) / len(v), 3) for _, v in sorted(pos.items())]
            res[b] = entry
        return res

    out = {}
    for model, files in READS.items():
        if not all(f.exists() for f in files):
            continue
        rows = load_rows(files)
        pred, correct = {}, {}
        for (doc, qid), r in rows.items():
            bucket = doc.split("/")[1]
            k = max(range(len(r["p"])), key=r["p"].__getitem__)
            pred[(bucket, qid[1:])] = r["keys"][k]
            correct[(bucket, qid[1:])] = int(r["keys"][k] == r["label"] if isinstance(r["label"], str) else k == r["label"])
        out[model] = analyze(pred, correct)
    external = {"Jev 1.13 (hosted; 64k split into two 32k halves)": JEV,
                "Clef-flash 9B (Workers AI)": ROOT / "data" / "hosted" / "clef_dense_rows_clef-flash.jsonl",
                "Clef 27B (Workers AI)": ROOT / "data" / "hosted" / "clef_dense_rows_clef.jsonl"}
    for name, path in external.items():   # hosted models: bench/complaints/jev.py, bench/complaints/clef.py
        if not path.exists():
            continue
        pred, correct = {}, {}
        for line in open(path):
            r = json.loads(line)
            if r["correct"] is None:
                continue
            b = "64k" if r["bucket"] == "64k-split" else r["bucket"]
            pred[(b, r["question"][1:])], correct[(b, r["question"][1:])] = r["pred"], r["correct"]
        out[name] = analyze(pred, correct)
    json.dump(out, open(ROOT / "results" / "complaints.json", "w"), indent=1)
    for model, res in out.items():
        print(f"\n{model}")
        for b, e in res.items():
            line = f"  {b:>5}: acc {e['acc']:.3f} ({e['questions']} questions)"
            if "paired_vs_alone" in e:
                p = e["paired_vs_alone"]
                nl, na = e["neighbour_errors"]["long"], e["neighbour_errors"]["alone_same_complaints"]
                line += (f" | vs alone {100 * p['delta']:+.1f} pp [{100 * p['ci95'][0]:+.1f}, {100 * p['ci95'][1]:+.1f}]"
                         f" | errors naming a neighbour's product {nl[0]}/{nl[1]} vs {na[0]}/{na[1]} alone"
                         f" | by fifth {e['acc_by_fifth']}")
            print(line)


if __name__ == "__main__":
    main()
