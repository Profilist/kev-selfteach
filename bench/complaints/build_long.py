"""Packed-complaints test past 64k (data/packed-complaints-v1): the v0 pool asked in 128k and 256k documents.

Same 1,980 complaints, blocks and questions as data/packed-complaints-v0 (taken verbatim from its alone records), so every
answer pairs with the same complaint alone. Buckets:
  alone   v0's alone records, unchanged
  32k/64k v0's 32k and 64k documents (same complaints, same order) as single records: every question in one record.
          Scored with the chunked scorer, these must reproduce v0's grouped rows, which checks the scorer.
  128k    documents filling 80-93 % of 131,072 tokens (~490 complaints), first-fit in a random order
  256k    the same for 262,144 tokens (~980 complaints; Qwen3.5's native window is 262,144)
Each long bucket is packed in two independent random orders, so every complaint is asked twice per bucket with
different neighbours. One record per document, scored by bench/chunked_scorer.py (Kev's scorer, branches chunked).
Also writes data/packed-complaints-v1-check: the first two 64k documents alone, for a fast scorer check.

    uv run python -m bench.complaints.build_long
"""
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "data" / "packed-complaints-v0"
OUT = ROOT / "data" / "packed-complaints-v1"
CHECK = ROOT / "data" / "packed-complaints-v1-check"
TOKENIZER = ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
BUCKETS = {"128k": 131072, "256k": 262144}
FILL = (0.80, 0.93)
ORDERS = 2
CONTEXT = {"max_state": 262144, "max_branch": 262144 + 8192, "max_packed": 4 * 262144, "truncate": False}


def write_suite(path, records, version, note):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "development.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    manifest = {"version": version, "partitions": ["development"], "locked": [], "eval_only": True, "report_only": True,
                "trainable_sources": [], "eval_only_sources": ["packed_cfpb"], "holdout_sources": [], "context": CONTEXT, "note": note,
                "files": {"development.jsonl": {"sha256": hashlib.sha256((path / "development.jsonl").read_bytes()).hexdigest(),
                                                "records": len(records), "questions": sum(len(r["questions"]) for r in records)}}}
    json.dump(manifest, open(path / "manifest.json", "w"), indent=1)


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    count = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    alone, blocks, questions = [], {}, {}
    for line in open(V0 / "development.jsonl"):
        r = json.loads(line)
        if r["_meta"]["bucket"] == "alone":
            alone.append(r)
            (cid,) = [q[1:] for q in r["questions"]]
            blocks[cid], questions[cid] = r["state"], r["questions"]
    ntok = {c: count(b) for c, b in blocks.items()}
    records, index = [r for r in alone], []

    def record(bucket, ids, n_state):
        rid = f"packed-complaints-v1/{bucket}/{sum(i['bucket'] == bucket for i in index):04d}"
        index.append({"id": rid, "bucket": bucket, "complaints": ids, "state_tokens": n_state})
        return {"state": "\n\n".join(blocks[c] for c in ids), "questions": {k: v for c in ids for k, v in questions[c].items()},
                "_meta": {"id": rid, "source": "packed_cfpb", "group_id": rid, "variant": "clean", "split": "development",
                          "bucket": bucket, "state_tokens": n_state, "complaints": len(ids)}}

    for i in json.load(open(V0 / "index.json")):
        if i["bucket"] in ("32k", "64k"):
            records.append(record(i["bucket"], i["complaints"], i["state_tokens"]))
    pool = sorted(blocks)
    for name, T in BUCKETS.items():
        lo, hi = int(FILL[0] * T), int(FILL[1] * T)
        for order in range(ORDERS):
            ids = pool[:]
            random.Random(f"dense-complaints-v1:{name}:{order}").shuffle(ids)   # seed text from the suite's first name, kept so the data reproduces
            docs, cur, size = [], [], 0
            for c in ids:
                if size + ntok[c] + 1 > hi - 64 and cur:
                    docs.append(cur); cur, size = [], 0
                cur.append(c); size += ntok[c] + 1
            docs.append(cur)
            for d in docs:
                n = count("\n\n".join(blocks[c] for c in d))
                while n > hi:
                    d = d[:-1]; n = count("\n\n".join(blocks[c] for c in d))
                if n >= lo:
                    records.append(record(name, d, n))
        got = [i for i in index if i["bucket"] == name]
        print(f"{name}: {len(got)} documents, {sum(len(i['complaints']) for i in got)} questions, "
              f"{min(i['state_tokens'] for i in got):,}-{max(i['state_tokens'] for i in got):,} state tokens, "
              f"{min(len(i['complaints']) for i in got)}-{max(len(i['complaints']) for i in got)} complaints each")
    write_suite(OUT, records, "packed-complaints-v1", "v0's pool and questions; 32k and 64k = v0's documents as single records; 128k and 256k in two orders")
    json.dump(index, open(OUT / "index.json", "w"))
    check = [r for r in records if r["_meta"]["bucket"] == "64k"][:2]
    write_suite(CHECK, check, "packed-complaints-v1-check", "the first two 64k documents of packed-complaints-v1, for the scorer check")
    print(f"wrote {len(records)} records -> {OUT}; scorer check: {len(check)} records, {sum(len(r['questions']) for r in check)} questions -> {CHECK}")


if __name__ == "__main__":
    main()
