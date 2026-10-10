"""Builds the headline model's training data (run fresh-32k; with --seed 1 --out data/selfteach/fresh-32k-seed1, its
second seed): complaints Kev never trained on, so no label reaches the teacher.

The first attempt (build_kev_items.py) used complaints from Kev's documents-v1 train, and the teacher's answers matched
their labels on 92.8 % of training questions, against about 80 % on fresh complaints. Here every complaint item is fresh:
  - CFPB narratives from the same Hub snapshot and shard as the packed-complaints test (davidheineman/
    consumer-finance-complaints-large @ 44cfa17, shard 10), 80-600 tokens like the test;
  - minus Kev's documents-v1 train and the packed-complaints test pool (by complaint id and by normalised text);
  - drawn at random with no product filter: no label is read to select, balance or target them.
Each is asked documents-v1's product question (nine options, verbatim), pointed at the complaint's id (the first
attempt also asked documents-v1's issue question where it had one; its options depend on the product, a label). The dataset's
product is kept only to report the teacher's accuracy; complaints whose product has no matching option get a
placeholder label and the task "cfpb_product_unmapped", and are left out of that report.

The decision-record items, and the teacher's answers to them, are reused unchanged (data/selfteach/kev-items/alone-records),
so only the fresh complaints need a teacher read. Packing, mix and sizes are common.write_packed's (documents 1k-31k tokens).
Writes data/selfteach/fresh-32k/alone/ (fresh complaints alone, for the teacher) and packed.jsonl; then drop over-long
records with `data/kev/.venv/bin/python selfteach/filter.py data/selfteach/fresh-32k/packed.jsonl`.

    uv run python -m selfteach.build_fresh
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from bench.complaints.build import KEV_TRAIN, PRODUCT, SHARD, norm, product_question
from .common import CONTEXT, DATA, HEADERS, ROOT, TOKENIZER, name_it, write_packed

TEST = ROOT / "data" / "packed-complaints-v0"
RECORDS = DATA / "kev-items" / "alone-records" / "development.jsonl"
OUT = DATA / "fresh-32k"


def main(a):
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    instructions, criteria = product_question()

    # ---- exclusions: Kev's documents-v1 train and the packed-complaints test pool
    seen = {hashlib.sha256(norm(json.loads(line)["state"]).encode()).hexdigest() for line in open(KEV_TRAIN)}
    test_ids = set()
    for line in open(TEST / "development.jsonl"):
        r = json.loads(line)
        if r["_meta"]["bucket"] == "alone":
            cid = next(iter(r["questions"]))[1:]
            test_ids.add(cid)
            seen.add(hashlib.sha256(norm(r["state"].split("\n", 1)[1]).encode()).hexdigest())

    # ---- fresh complaint items, in random order with no product filter
    df = pd.read_parquet(SHARD, columns=["complaint_id", "product", "complaint_what_happened"])
    df = df[df.complaint_what_happened.str.strip().str.len() > 0].sample(frac=1.0, random_state=a.seed)
    items, ids, skipped, unmapped = [], set(), 0, 0
    for r in df.itertuples():
        if str(r.complaint_id) in test_ids:
            skipped += 1
            continue
        text = " ".join(r.complaint_what_happened.split())
        key = hashlib.sha256(norm(text).encode()).hexdigest()
        if key in seen:
            skipped += 1
            continue
        if not a.min_tokens <= ntok(text) <= a.max_tokens:
            continue
        seen.add(key)
        cid = str(rng.randint(10_000_000, 99_999_999))
        while cid in ids:
            cid = str(rng.randint(10_000_000, 99_999_999))
        ids.add(cid)
        head, ref = rng.choice(HEADERS)
        block = head.format(id=cid) + "\n" + text
        label = PRODUCT.get(r.product)
        unmapped += label is None
        q = {"type": "choice", "instructions": name_it(instructions, ref.format(id=cid)), "criteria": criteria,
             "label": label or next(iter(criteria)), "src": "cfpb_product" if label else "cfpb_product_unmapped"}
        items.append({"key": f"c{cid}", "kind": "complaint", "block": block, "tokens": ntok(block) + 2, "questions": {"product": q}})
        if len(items) == a.complaints:
            break
    print(f"fresh complaints: {len(items)} ({unmapped} with no matching product option); {skipped} skipped as Kev train or test duplicates")

    # ---- the decision-record items, unchanged (their teacher answers are reused)
    n_complaints = len(items)
    for line in open(RECORDS):
        r = json.loads(line)
        block = r["state"][0]
        items.append({"key": r["_meta"]["id"].split("/", 1)[1], "kind": "record", "block": block,
                      "tokens": ntok(block["text"]) + 8, "questions": r["questions"]})
    print(f"decision records: {len(items) - n_complaints}")

    # ---- the teacher's suite: the fresh complaints alone
    out = Path(a.out)
    alone_dir = out / "alone"
    alone_dir.mkdir(parents=True, exist_ok=True)
    with open(alone_dir / "development.jsonl", "w") as f:
        for it in items[:n_complaints]:
            f.write(json.dumps({"state": it["block"], "questions": it["questions"],
                                "_meta": {"id": f"selfteach-alone/{it['key']}", "source": "selfteach_complaint_fresh", "group_id": it["key"],
                                          "variant": "clean", "split": "development"}}) + "\n")
    manifest = {"version": f"{out.name}-alone", "partitions": ["development"], "locked": [], "eval_only": True, "report_only": True,
                "trainable_sources": [], "eval_only_sources": ["selfteach_complaint_fresh"], "holdout_sources": [], "context": CONTEXT,
                "files": {"development.jsonl": {"sha256": hashlib.sha256((alone_dir / "development.jsonl").read_bytes()).hexdigest(),
                                                "records": n_complaints, "questions": n_complaints}}}
    json.dump(manifest, open(alone_dir / "manifest.json", "w"), indent=1)

    # ---- packed training records
    write_packed(rng, items, a, out / "packed.jsonl")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--complaints", type=int, default=5219)   # as many as the first attempt had
    p.add_argument("--records", type=int, default=2400)
    p.add_argument("--asked", type=int, default=16)
    p.add_argument("--min_len", type=int, default=1000)
    p.add_argument("--max_len", type=int, default=31000)
    p.add_argument("--min_tokens", type=int, default=80)
    p.add_argument("--max_tokens", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(OUT))
    main(p.parse_args())
