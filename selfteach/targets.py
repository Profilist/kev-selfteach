"""Attach the teacher's answers as soft targets: teacher rows (teacher.sh: kev.benchmark on an items-alone suite, raw logits) ->
kev.train records. Each question's target is the teacher's probability for every option with the item read alone, and
its label (which kev.load_records requires) is the teacher's own top answer. No dataset label is used.

    python selfteach/targets.py TEACHER_ROWS.json PACKED.jsonl OUT_TRAIN.jsonl
"""
import json
import sys


def main(rows_path, packed_path, out_path):
    teacher = {(r["id"], r["question"]): r for r in json.load(open(rows_path))}
    n_rec = n_q = missing = 0
    agree = {}   # teacher row source -> [top answer = dataset label, labelled questions]
    with open(out_path, "w") as out:
        for line in open(packed_path):
            r = json.loads(line)
            qs = {}
            for qid, q in r["questions"].items():
                t = teacher.get((q["item"], q["item_qid"]))
                if t is None:
                    missing += 1
                    continue
                target = dict(zip(t["keys"], t["p"]))
                top = max(target, key=target.get)
                label = top if q["type"] == "choice" else (top == "true" if q["type"] == "noul" else int(top))
                qs[qid] = {k: v for k, v in q.items() if k not in ("item", "item_qid")} | {"target": target, "label": label}
                if not t.get("task", "").endswith(("_unmapped", "_unlabelled")):   # placeholder labels (no option / no label)
                    a = agree.setdefault(t.get("source", "?"), [0, 0])
                    a[0] += t["keys"].index(top) == t["label"]; a[1] += 1
            if qs:
                out.write(json.dumps({"state": r["state"], "questions": qs, "_meta": r["_meta"]}) + "\n")
                n_rec += 1; n_q += len(qs)
    hit, n = sum(a[0] for a in agree.values()), sum(a[1] for a in agree.values())
    print(f"wrote {n_rec} records, {n_q} questions with teacher targets; {missing} questions had no teacher row; "
          f"teacher's top answer matches the dataset label on {hit / max(n, 1):.1%} of {n} labelled ones (reported only; "
          + ", ".join(f"{src} {a[0] / a[1]:.1%} of {a[1]}" for src, a in sorted(agree.items())) + ")")


if __name__ == "__main__":
    main(*sys.argv[1:4])
