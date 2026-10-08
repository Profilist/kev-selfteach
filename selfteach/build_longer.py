"""Builds the longer-document rounds' training data: fresh-32k's items, repacked into longer documents.

Run fresh-64k continues fresh-32k on documents of 8k-63k tokens; run fresh-128k continues fresh-64k on 32k-125k.
The items and the teacher's answers are fresh-32k's, unchanged: 5,219 fresh CFPB complaints
(data/selfteach/fresh-32k/alone) and 3,432 decision-v2 train records (data/selfteach/kev-items/alone-records), each
answered alone by Kev-0.8B 1.0. No label is read. Only the packing changes: documents are log-uniform in the given range
(fresh-32k used 1k-31k), with more items asked each. Mix: 60 % complaint documents, 15 % record documents, 25 % items
alone. Then drop records that don't fit with `data/kev/.venv/bin/python selfteach/filter.py <packed.jsonl> 65536`
(131072 for fresh-128k, whose training lifts Kev's 64k trainer cap).

    uv run python -m selfteach.build_longer                                         # fresh-64k
    uv run python -m selfteach.build_longer --records 400 --asked 12 --min_len 32000 --max_len 125000 --seed 1 \
        --out data/selfteach/fresh-128k                                             # fresh-128k
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from .common import DATA, TOKENIZER, ask, pack

COMPLAINTS = DATA / "fresh-32k" / "alone" / "development.jsonl"
RECORDS = DATA / "kev-items" / "alone-records" / "development.jsonl"
OUT = DATA / "fresh-64k"
MIX = {"complaint_docs": 0.60, "record_docs": 0.15, "alone": 0.25}


def main(a):
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    items = []
    for line in open(COMPLAINTS):
        r = json.loads(line)
        items.append({"key": r["_meta"]["id"].split("/", 1)[1], "kind": "complaint", "block": r["state"], "tokens": ntok(r["state"]) + 2,
                      "questions": r["questions"]})
    for line in open(RECORDS):
        r = json.loads(line)
        items.append({"key": r["_meta"]["id"].split("/", 1)[1], "kind": "record", "block": r["state"][0], "tokens": ntok(r["state"][0]["text"]) + 8,
                      "questions": r["questions"]})
    by_kind = {k: [i for i in items if i["kind"] == k] for k in ("complaint", "record")}
    n = {k: int(round(v * a.records)) for k, v in MIX.items()}
    records = [pack(rng, by_kind["complaint"], "complaint", a) for _ in range(n["complaint_docs"])]
    records += [pack(rng, by_kind["record"], "record", a) for _ in range(n["record_docs"])]
    for it in rng.sample(items, n["alone"]):
        state = it["block"] if it["kind"] == "complaint" else [it["block"]]
        records.append({"state": state, "questions": {qid: ask(it, qid, q) for qid, q in it["questions"].items()}, "_meta": {"family": f"{it['kind']}_alone"}})
    rng.shuffle(records)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "packed.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tokens = sum(r["_meta"].get("approx_tokens", 300) for r in records)
    print(f"training: {len(records)} records, {sum(len(r['questions']) for r in records)} questions, ~{tokens / 1e6:.1f}M state tokens; "
          f"{dict(Counter(r['_meta']['family'] for r in records))}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=int, default=800)
    p.add_argument("--asked", type=int, default=24)
    p.add_argument("--min_len", type=int, default=8000)
    p.add_argument("--max_len", type=int, default=63000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(OUT))
    main(p.parse_args())
