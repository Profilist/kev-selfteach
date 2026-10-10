"""Packed-complaints long-document test: every complaint in a long document gets its own question.

A decision model reads the document once and answers each question in its own branch off that read, so asking about every
item costs little more than asking about one. This test measures whether it keeps every part of a long document straight,
not just whether it can find one needle.

Complaints: the public CFPB database (US government work, public domain), via the same Hub snapshot Kev's documents-v1
was built from (davidheineman/consumer-finance-complaints-large @ 44cfa17, shard 10 of 11), narratives of 80-600 tokens,
deduplicated against Kev's public documents-v1 train set. Question: documents-v1's product question with its nine options
verbatim, naming the complaint by its CFPB id. Labels: the consumer's own product choice, mapped to the nine options (no
model involved; "Consumer Loan", "Other financial service" and "Debt or credit management" are skipped as ambiguous).

One pool of complaints, balanced over the nine products, is asked at every length, so accuracy is paired by complaint:
  alone   one complaint per state
  8k-64k  complaints packed into documents whose state holds 80-93 % of the nominal size under the Qwen3.5 tokenizer
          (Kev's longdoc-v1 bucket rule), in a fresh random order per length, so neighbours differ between lengths.
Written as a frozen eval-only suite in Kev's layout (manifest.json + development.jsonl, serving context: 64k-token
states), so `kev.benchmark --suite` scores it like Kev's own suites; index.json maps each record to its bucket.

    uv run python -m bench.complaints.build --out data/packed-complaints-v0
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SHARD = ROOT / "data" / "cfpb" / "train-00010.parquet"
KEV_TRAIN = ROOT / "data" / "cfpb" / "kev_documents_v1_train.jsonl"
TOKENIZER = ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
BUCKETS = {"8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536}
FILL = (0.80, 0.93)
PRODUCT = {
    "Credit reporting, credit repair services, or other personal consumer reports": "credit_reporting",
    "Credit reporting or other personal consumer reports": "credit_reporting",
    "Credit reporting": "credit_reporting",
    "Debt collection": "debt_collection",
    "Mortgage": "mortgage",
    "Checking or savings account": "bank_account",
    "Bank account or service": "bank_account",
    "Credit card or prepaid card": "card",
    "Credit card": "card",
    "Prepaid card": "card",
    "Student loan": "student_loan",
    "Money transfer, virtual currency, or money service": "money_transfer",
    "Money transfers": "money_transfer",
    "Virtual currency": "money_transfer",
    "Vehicle loan or lease": "vehicle_loan",
    "Payday loan, title loan, or personal loan": "personal_loan",
    "Payday loan": "personal_loan",
    "Payday loan, title loan, personal loan, or advance loan": "personal_loan",
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def product_question() -> tuple[str, dict]:
    """documents-v1's product question (instructions and the nine options with their descriptions), verbatim."""
    for line in open(KEV_TRAIN):
        for q in json.loads(line)["questions"].values():
            if q.get("src") == "cfpb_product":
                return q["instructions"], q["criteria"]
    raise SystemExit("no product question in documents-v1 train")


def main(a):
    rng = random.Random(a.seed)
    instructions, criteria = product_question()
    seen = {hashlib.sha256(norm(json.loads(line)["state"]).encode()).hexdigest() for line in open(KEV_TRAIN)}
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])

    df = pd.read_parquet(SHARD, columns=["complaint_id", "product", "complaint_what_happened"])
    df = df[df.complaint_what_happened.str.strip().str.len() > 0]
    df["label"] = df["product"].map(PRODUCT)
    df = df.dropna(subset=["label"]).sample(frac=1.0, random_state=a.seed)

    pool, in_kev = [], 0
    for label in sorted(set(PRODUCT.values())):
        kept = 0
        for r in df[df.label == label].itertuples():
            text = " ".join(r.complaint_what_happened.split())
            key = hashlib.sha256(norm(text).encode()).hexdigest()
            if key in seen:   # in Kev's documents-v1 train, or already in the pool
                in_kev += 1
                continue
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            if not a.min_tokens <= n <= a.max_tokens:
                continue
            seen.add(key)
            pool.append({"id": str(r.complaint_id), "label": label, "text": text})
            kept += 1
            if kept == a.per_product:
                break
    print(f"pool: {len(pool)} complaints ({a.per_product} per product); {in_kev} candidates skipped as duplicates of Kev's train set or the pool")

    def block(c):
        return f"Complaint {c['id']}:\n{c['text']}"

    def question(c):
        return {"type": "choice", "instructions": instructions.replace("this complaint", f"complaint {c['id']}"),
                "criteria": criteria, "label": c["label"], "src": "cfpb_product"}

    def tokens(blocks):
        return len(tok("\n\n".join(blocks), add_special_tokens=False)["input_ids"])

    for c in pool:
        c["tokens"] = tokens([block(c)])
    records, index = [], []
    def record(bucket, d, state_tokens):
        rid = f"packed-complaints-v0/{bucket}/{sum(i['bucket'] == bucket for i in index):04d}"
        index.append({"id": rid, "bucket": bucket, "complaints": [c["id"] for c in d], "state_tokens": state_tokens})
        return {"state": "\n\n".join(block(c) for c in d), "questions": {f"c{c['id']}": question(c) for c in d},
                "_meta": {"id": rid, "source": "packed_cfpb", "group_id": rid, "variant": "clean", "split": "development",
                          "bucket": bucket, "state_tokens": state_tokens, "complaints": len(d)}}

    for c in pool:   # alone: each complaint is its own state
        records.append(record("alone", [c], c["tokens"]))
    for name, nominal in BUCKETS.items():
        lo, hi = int(FILL[0] * nominal), int(FILL[1] * nominal)
        order = pool[:]
        rng.shuffle(order)
        docs, cur, size = [], [], 0   # size: sum of block tokens + one per "\n\n" separator (checked exactly below)
        for c in order:
            if size + c["tokens"] + 1 <= hi - 64:
                cur.append(c)
                size += c["tokens"] + 1
            elif size >= lo:
                docs.append(cur)
                cur, size = [c], c["tokens"]
        if cur and size >= lo:
            docs.append(cur)
        for d in docs:   # exact count under the tokenizer; trim if the estimate was low
            while tokens([block(x) for x in d]) > hi:
                d.pop()
        docs = [d for d in docs if tokens([block(x) for x in d]) >= lo]
        asked = sum(len(d) for d in docs)
        for d in docs:
            records.append(record(name, d, tokens([block(c) for c in d])))
        print(f"{name}: {len(docs)} documents, {asked} of {len(pool)} complaints asked, "
              f"{min(i['state_tokens'] for i in index if i['bucket'] == name):,}-{max(i['state_tokens'] for i in index if i['bucket'] == name):,} state tokens")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "development.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    manifest = {"version": "packed-complaints-v0", "partitions": ["development"], "locked": [],
                "files": {"development.jsonl": {"sha256": hashlib.sha256((out / "development.jsonl").read_bytes()).hexdigest(),
                                                "records": len(records), "questions": sum(len(r["questions"]) for r in records)}},
                "eval_only": True, "report_only": True, "trainable_sources": [], "eval_only_sources": ["packed_cfpb"], "holdout_sources": [],
                "context": {"max_state": 65536, "max_branch": 65536 + 8192, "max_packed": 2 * 65536 + 8192, "truncate": False},
                "source": "davidheineman/consumer-finance-complaints-large@44cfa170a402e254407470275ce05d7dcaccde30 (shard 10)",
                "seed": a.seed, "tokenizer": TOKENIZER, "buckets": BUCKETS, "fill": FILL, "per_product": a.per_product}
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    json.dump(index, open(out / "index.json", "w"))
    print(f"wrote {len(records)} records, {sum(len(r['questions']) for r in records)} questions -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/packed-complaints-v0")
    p.add_argument("--per_product", type=int, default=220)
    p.add_argument("--min_tokens", type=int, default=80)
    p.add_argument("--max_tokens", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
