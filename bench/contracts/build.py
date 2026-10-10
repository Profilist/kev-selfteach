"""Packed-contracts long-document test: every contract in a review file gets its own questions.

A layout none of our training used: CUAD contracts (CC BY 4.0; Hendrycks, Burns, Chen and Ball, NeurIPS 2021; The Atticus
Project) in Kev's longdoc-v1 review-file format ("===== Contract k of n: <title> ====="). Every contract gets longdoc-v1's
own question set: two clause-presence questions, a clause-category choice and, where CUAD's governing law is a common one,
a jurisdiction choice. The questions come from Kev's builder (scripts/build_longdoc_v1.py @ a0255ce: load_cuad,
cuad_questions, cuad_request, cuad_state) with the lawyers' labels. Only longdoc-v1's development companies are used, so
its locked test stays unread, and contracts from one company never share a file.

One pool of contracts (each at most --max_tokens) is asked at every length, paired by question:
  alone   each contract in a review file of its own
  16k-64k whole contracts packed first-fit, in a fresh random order per length, into review files holding 70-93 % of the
          nominal size and at least two contracts; a long file's questions go in groups that share the file, because
          Kev's scorer batches one record's questions together
Written as a frozen eval-only suite (manifest.json + development.jsonl, 64k serving context) for `kev.benchmark --suite`.

    PYTHONPATH=data/kev data/kev/.venv/bin/python bench/contracts/build.py
"""
import argparse
import hashlib
import importlib.util
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEV = ROOT / "data" / "kev"
BUCKETS = {"16k": 16384, "32k": 32768, "64k": 65536}
FILL = (0.70, 0.93)   # review-file size as a share of the nominal length (longdoc-v1 uses 0.80-0.93; whole contracts pack looser)
GROUP = {"alone": 10**9, "16k": 60, "32k": 30, "64k": 15}


def kev_builder():
    spec = importlib.util.spec_from_file_location("build_longdoc_v1", KEV / "scripts" / "build_longdoc_v1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(a):
    L = kev_builder()
    count = L.Tokens()
    contracts, details = L.load_cuad(L.fetch_cuad(), count)
    part = L.deal(contracts)
    prevalence = {k: sum(k in c["present"] for c in contracts.values()) / len(contracts) for k in details}
    law_counts = Counter(c["law"] for c in contracts.values())
    laws_common = sorted(k for k, v in law_counts.items() if v >= 3 and k and not re.search(r"[;/]", k))
    pool = sorted(t for t, c in contracts.items()
                  if part[t] == "development" and c["tokens"] <= a.max_tokens and c["present"] and c["absent"])
    questions = {}
    for i, t in enumerate(pool):
        qs = L.cuad_request(t, L.cuad_questions(t, contracts[t], details, prevalence, laws_common), details)
        questions[t] = {f"c{i:03d}_{q}": {**d, "src": f"cuad_{q}"} for q, d in qs.items()}
    print(f"pool: {len(pool)} development contracts of at most {a.max_tokens:,} tokens, "
          f"{sum(map(len, questions.values()))} questions ({len(contracts)} CUAD contracts in all)")

    records, index = [], []

    def add(bucket, titles):
        state, _ = L.cuad_state([(t, contracts[t]["text"], False) for t in titles])
        n_state = count(state)
        doc = f"packed-contracts-v0/{bucket}/{sum(i['bucket'] == bucket for i in index):04d}"
        index.append({"id": doc, "bucket": bucket, "contracts": titles, "state_tokens": n_state})
        qids = [q for t in titles for q in questions[t]]
        allq = {q: d for t in titles for q, d in questions[t].items()}
        g = GROUP[bucket]
        for k in range(0, len(qids), g):
            rid = doc if len(qids) <= g else f"{doc}#q{k // g:02d}"
            records.append({"state": state, "questions": {q: allq[q] for q in qids[k:k + g]},
                            "_meta": {"id": rid, "doc_id": doc, "source": "packed_cuad", "group_id": doc, "variant": "clean",
                                      "split": "development", "bucket": bucket, "state_tokens": n_state, "contracts": len(titles)}})
        return n_state

    for t in pool:
        add("alone", [t])
    for name, T in BUCKETS.items():
        lo, hi = int(FILL[0] * T), int(FILL[1] * T)
        rng = random.Random(f"dense-contracts-v0:{name}:{a.seed}")   # seed text from the suite's first name, kept so the data reproduces
        order = pool[:]
        rng.shuffle(order)
        bins = []   # [titles, estimated tokens]; first fit in random order keeps neighbours random
        for t in order:
            n = contracts[t]["tokens"] + 40
            for b in bins:
                if b[1] + n <= hi - 64 and all(contracts[x]["company"] != contracts[t]["company"] for x in b[0]):
                    b[0].append(t); b[1] += n
                    break
            else:
                bins.append([[t], 40 + n])
        docs = [b[0] for b in bins if b[1] >= lo and len(b[0]) >= 2]
        kept = 0
        for d in docs:
            state, _ = L.cuad_state([(t, contracts[t]["text"], False) for t in d])
            while count(state) > hi:   # the estimate ran over: drop the last contract
                d = d[:-1]
                state, _ = L.cuad_state([(t, contracts[t]["text"], False) for t in d])
            if count(state) >= lo and len(d) >= 2:
                add(name, d); kept += len(d)
        sizes = [i["state_tokens"] for i in index if i["bucket"] == name]
        print(f"{name}: {len(sizes)} review files, {kept} of {len(pool)} contracts asked, {min(sizes):,}-{max(sizes):,} state tokens")

    out = ROOT / "data" / "packed-contracts-v0"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "development.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    manifest = {"version": "packed-contracts-v0", "partitions": ["development"], "locked": [], "eval_only": True, "report_only": True,
                "trainable_sources": [], "eval_only_sources": ["packed_cuad"], "holdout_sources": [],
                "context": {"max_state": 65536, "max_branch": 65536 + 8192, "max_packed": 2 * 65536 + 8192, "truncate": False},
                "cuad": L.CUAD, "builder": "Kev scripts/build_longdoc_v1.py @ a0255ce (load_cuad, deal, cuad_questions, cuad_request, cuad_state)",
                "tokenizer": L.TOKENIZER, "window": L.WINDOW, "max_tokens": a.max_tokens, "seed": a.seed,
                "files": {"development.jsonl": {"sha256": hashlib.sha256((out / "development.jsonl").read_bytes()).hexdigest(),
                                                "records": len(records), "questions": sum(len(r["questions"]) for r in records)}}}
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    json.dump(index, open(out / "index.json", "w"))
    print(f"wrote {len(records)} records, {manifest['files']['development.jsonl']['questions']} questions -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_tokens", type=int, default=7500, help="longest contract in the pool (two always fit a 16k file)")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
