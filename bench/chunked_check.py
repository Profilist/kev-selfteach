"""Check the chunked scorer (bench/chunked_scorer.py) against Kev's unpatched scorer on the same documents and checkpoint.

CHUNKED: rows for dense-complaints-v1-check (v0's first two 64k documents, each one record with all its questions).
REFERENCE: Kev's unpatched rows for v0's 64k documents, where each document was split into records of at most 15
questions that each re-read it. Same complaints, questions and model, so every probability should match up to kernel
rounding. Exits non-zero unless every question pairs, the top answers agree on at least 99.5 % and max |dp| <= 5e-3.

    python bench/chunked_check.py CHUNKED_ROWS.json REFERENCE_ROWS.json
"""
import json
import sys


def top(r):
    return max(range(len(r["p"])), key=r["p"].__getitem__)


def main(chunked, reference):
    a = {(r["id"].replace("dense-complaints-v1/", "dense-complaints-v0/").split("#")[0], r["question"]): r for r in json.load(open(chunked))}
    docs = {doc for doc, _ in a}
    b = {(r["id"].split("#")[0], r["question"]): r for r in json.load(open(reference)) if r["id"].split("#")[0] in docs}
    common = sorted(set(a) & set(b))
    dp = max(max(abs(x - y) for x, y in zip(a[k]["p"], b[k]["p"])) for k in common) if common else float("inf")
    agree = sum(top(a[k]) == top(b[k]) for k in common)
    ok = len(common) == len(a) and agree >= 0.995 * len(common) and dp <= 5e-3
    print(f"scorer check: {len(common)}/{len(a)} questions paired, max |dp| {dp:.2e}, top answer agrees on {agree}/{len(common)} "
          f"-> {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
