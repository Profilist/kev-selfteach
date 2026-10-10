"""Self-taught long context: does a model read an item packed among many others the way it reads that item alone?

On the packed-complaints and packed-contracts tests, for each Kev-family model and document length:
  acc          accuracy against the dataset label
  = teacher    share of questions whose top answer matches Kev-0.8B 1.0's answer with the item alone (the self-taught
               model's training target, measured on test items no model trained on)
  = own alone  share matching the same model's own answer with the item alone
  gap closed   (acc - Kev 1.0's acc) / (Kev 1.0 alone - Kev 1.0's acc) at that length
and the self-taught model minus delta-v1 (the labelled recipe) at each length, paired by question with a 95 % bootstrap
interval over documents.

    uv run python -m selfteach.report
"""
import json
from collections import defaultdict

from bench.contracts.report import KEV_READS as CONTRACT_READS, RES, ROOT, boot, kev_correct
from bench.complaints.report import READS as COMPLAINT_READS, load_rows

KEV, DELTA, SELF = "Kev-0.8B 1.0", "Kev-0.8B 1.0 + complaint-document delta (ours)", "Kev-0.8B 1.0 self-taught, kev-items (ours)"
DELTA_S1 = "Kev-0.8B 1.0 + complaint-document delta, seed 1 (ours)"
SELF_1B = "Kev-0.8B 1.0 self-taught, fresh-32k (ours)"
SELF_1B_S1 = "Kev-0.8B 1.0 self-taught, fresh-32k-seed1 (ours)"
SELF_1C = "Kev-0.8B 1.0 self-taught, fresh-32k-ndas (ours)"


def top(r):
    return max(range(len(r["p"])), key=r["p"].__getitem__)


def complaints():
    out = {}
    for model in (KEV, DELTA, DELTA_S1, SELF, SELF_1B, SELF_1B_S1, SELF_1C):
        files = COMPLAINT_READS[model]
        if not all(f.exists() for f in files):
            continue
        out[model] = {(doc.split("/")[1], qid[1:]): (doc, top(r), int(r["keys"][top(r)] == r["label"] if isinstance(r["label"], str) else top(r) == r["label"]))
                      for (doc, qid), r in load_rows(files).items()}
    return out


def contracts():
    out = {}
    for model in (KEV, DELTA, DELTA_S1, SELF, SELF_1B, SELF_1B_S1, SELF_1C):
        path = CONTRACT_READS[model]
        if path.exists():
            out[model] = {(r["id"].split("/")[1], r["question"]): (r["id"].split("#")[0], top(r), kev_correct(r)) for r in json.load(open(path))}
    return out


def edge_errors():
    """Per model and packed length: wrong answers, and how many name the product of the document's first or last
    complaint (other than the right one): the signature of reading the whole document as one complaint."""
    from bench.complaints.report import SUITE
    order = {i["id"]: i["complaints"] for i in json.load(open(SUITE / "index.json"))}
    label = {qid[1:]: q["label"] for line in open(SUITE / "development.jsonl") for qid, q in json.loads(line)["questions"].items()}
    out = {}
    for model in (KEV, DELTA, DELTA_S1, SELF, SELF_1B, SELF_1B_S1, SELF_1C):
        files = COMPLAINT_READS[model]
        if not all(f.exists() for f in files):
            continue
        cells = defaultdict(lambda: [0, 0])
        for (doc, qid), r in load_rows(files).items():
            bucket = doc.split("/")[1]
            if bucket == "alone":
                continue
            pred, truth = r["keys"][top(r)], (r["label"] if isinstance(r["label"], str) else r["keys"][r["label"]])
            if pred == truth:
                continue
            seq = order[doc.split("#")[0]]
            cells[bucket][0] += 1
            cells[bucket][1] += pred in ({label[seq[0]], label[seq[-1]]} - {truth})
        out[model] = {b: {"wrong": w, "edge": e} for b, (w, e) in sorted(cells.items())}
    return out


def analyze(reads, buckets):
    teacher = {item: v[1] for (b, item), v in reads[KEV].items() if b == "alone"}
    kev = {(b, item): v[2] for (b, item), v in reads[KEV].items()}
    res = {}
    for model, rows in reads.items():
        own = {item: v[1] for (b, item), v in rows.items() if b == "alone"}
        res[model] = {}
        for b in buckets:
            cell = [(item, v) for (bb, item), v in rows.items() if bb == b]
            if not cell:
                continue
            acc = sum(v[2] for _, v in cell) / len(cell)
            e = {"questions": len(cell), "acc": acc,
                 "agree_teacher": sum(v[1] == teacher[i] for i, v in cell if i in teacher) / sum(i in teacher for i, _ in cell),
                 "agree_own_alone": sum(v[1] == own[i] for i, v in cell if i in own) / max(1, sum(i in own for i, _ in cell))}
            same = [i for i, _ in cell if (b, i) in kev and ("alone", i) in kev]   # Kev 1.0's gap on these same questions
            kev_here, kev_alone = (sum(kev[(x, i)] for i in same) / max(1, len(same)) for x in (b, "alone"))
            if b != "alone" and kev_alone > kev_here:
                e["gap_closed"] = (sum(rows[(b, i)][2] for i in same) / len(same) - kev_here) / (kev_alone - kev_here)
            res[model][b] = e
    for a, b, name in ((SELF, DELTA, "kev-items minus delta-v1"), (SELF_1B, DELTA, "fresh-32k minus delta-v1"),
                       (SELF_1B, SELF, "fresh-32k minus kev-items"), (SELF_1B_S1, SELF_1B, "fresh-32k-seed1 minus fresh-32k"),
                       (SELF_1C, SELF_1B, "fresh-32k-ndas minus fresh-32k"), (SELF_1C, DELTA, "fresh-32k-ndas minus delta-v1")):   # paired by question, clustered by document
        if a not in reads or b not in reads:
            continue
        res[name] = {}
        for bucket in buckets:
            by_doc = defaultdict(list)
            for (bb, item), v in reads[a].items():
                if bb == bucket and (bucket, item) in reads[b]:
                    by_doc[v[0]].append(v[2] - reads[b][(bucket, item)][2])
            if by_doc:
                d, lo, hi = boot(by_doc)
                res[name][bucket] = {"delta": d, "ci95": [lo, hi], "pairs": sum(map(len, by_doc.values()))}
    return res


def main():
    out = {"complaints": analyze(complaints(), ["alone", "8k", "16k", "32k", "64k"]),
           "contracts": analyze(contracts(), ["alone", "16k", "32k", "64k"]),
           "complaints_edge_errors": edge_errors()}
    json.dump(out, open(ROOT / "results" / "selfteach.json", "w"), indent=1)
    print("\n=== packed-complaints: wrong answers naming the first or last complaint's product")
    for model, cells in out["complaints_edge_errors"].items():
        print(f"{model}: " + ", ".join(f"{b} {c['edge']}/{c['wrong']} ({c['edge'] / max(c['wrong'], 1):.0%})" for b, c in cells.items()))
    for test in ("complaints", "contracts"):
        res = out[test]
        print(f"\n=== all-{test}")
        for model, cells in res.items():
            print(f"{model}")
            for b, e in cells.items():
                if "delta" in e:
                    print(f"  {b:>5}: {100 * e['delta']:+.1f} pp [{100 * e['ci95'][0]:+.1f}, {100 * e['ci95'][1]:+.1f}] ({e['pairs']} pairs)")
                    continue
                gc = f" | gap closed {100 * e['gap_closed']:.0f}%" if "gap_closed" in e else ""
                print(f"  {b:>5}: acc {e['acc']:.3f} | = teacher {e['agree_teacher']:.3f} | = own alone {e['agree_own_alone']:.3f}{gc} ({e['questions']} q)")


if __name__ == "__main__":
    main()
