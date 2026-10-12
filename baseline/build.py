"""Training data for the label-trained baseline (run delta-v1; with --seed 1, delta-v1-s1): Kev-0.8B 1.0 trained with
Kev's own trainer on long documents whose questions carry the real labels. It's the comparison for the self-taught runs.

Mix (2,400 records; the long families' state lengths log-uniform in 1k-31k tokens):
  35 % complaint documents  Kev's public documents-v1 train complaints (CFPB narratives; labels from the consumer's own
                            filing, kept where two open-weight models agreed) packed with IDs into one document, with up to
                            16 of them asked about (their product and issue questions, naming the complaint); headers and
                            separators vary so the model learns to look up the named item, not one layout
  20 % buried records       one Kev decision-v2 train record hidden among others, named by its record ID
  15 % evidence tasks       synthetic long documents with programmatic labels (baseline/synthetic.py)
  30 % replay               15 % documents-v1 complaints alone, as published; 15 % decision-v2 train records
No complaint here is in the packed-complaints test (its pool was deduplicated against documents-v1 train), and no
hosted-model output is used anywhere. Then drop records over the 32k training limit with selfteach/filter.py:

    uv run python -m baseline.build                                                    # delta-v1
    uv run python -m baseline.build --seed 1 --out data/train-delta-v1-s1/train.jsonl  # delta-v1-s1
    data/kev/.venv/bin/python selfteach/filter.py data/train-delta-v1/train.jsonl
"""
import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from selfteach.common import HEADERS, ROOT, TOKENIZER, kev_records, name_it, render

from . import synthetic

DOCS = ROOT / "data" / "cfpb" / "kev_documents_v1_train.jsonl"
SEPARATORS = ["\n\n", "\n\n---\n\n", "\n\n\n"]
MIX = {"complaints": 0.35, "buried": 0.20, "evidence": 0.15, "replay_documents": 0.15, "replay_decision": 0.15}


def complaint_document(rng, docs, target, tok):
    head, ref = rng.choice(HEADERS)
    sep = rng.choice(SEPARATORS)
    blocks, chosen, size, seen_ids = [], [], 0, set()
    while size < target:
        d = rng.choice(docs)
        cid = str(rng.randint(1_000_000, 99_999_999))
        if cid in seen_ids or d in chosen:
            continue
        block = head.format(id=cid) + "\n" + d["state"]
        n = len(tok(block, add_special_tokens=False)["input_ids"]) + 2
        if size + n > target and blocks:
            break
        seen_ids.add(cid)
        blocks.append(block); chosen.append(d); size += n
        d["_cid"] = cid
    asked = rng.sample(range(len(chosen)), min(16, len(chosen)))
    questions = {}
    for i in asked:
        d, r = chosen[i], ref.format(id=chosen[i]["_cid"])
        for key, q in d["questions"].items():
            questions[f"{key}_{i}"] = {**q, "instructions": name_it(q["instructions"], r)}
    return {"state": sep.join(blocks), "questions": questions,
            "_meta": {"family": "complaint_document", "complaints": len(blocks), "asked": len(asked), "state_tokens": size}}


def buried_record(base: dict, length: int, rng: random.Random) -> dict:
    """Hide `base`'s state among other decision-v2 train records until ~length tokens; the questions name the record."""
    pool = kev_records("train")
    rid = f"R-{rng.randint(1000, 9999)}"
    items, used, total = [(rid, render(base["state"]))], {rid}, len(render(base["state"])) / synthetic.CPT["squad"]
    while total < length:
        other = rng.choice(pool)
        if other is base:
            continue
        oid = f"R-{rng.randint(1000, 9999)}"
        if oid in used:
            continue
        used.add(oid)
        text = render(other["state"])
        items.append((oid, text))
        total += len(text) / synthetic.CPT["squad"] + 6
    target = items.pop(0)
    items.insert(rng.randint(0, len(items)), target)
    state = [{"record": i, "text": t} for i, t in items]
    qs = {}
    for qid, q in base["questions"].items():
        q = {k: v for k, v in q.items() if k != "src"}
        qs[qid] = {**q, "instructions": f"Answer about record {rid} only. {render(q.get('instructions'))}"}
    return {"state": state, "questions": qs,
            "_meta": {"family": "buried", "length": length, "records": len(items),
                      "source": base.get("_meta", {}).get("source"), "split": "train"}}


def main(a):
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    docs = [json.loads(line) for line in open(DOCS)]
    docs = [d for d in docs if isinstance(d["state"], str)]
    kev = kev_records("train")

    def length():
        return int(round(a.min_len * (a.max_len / a.min_len) ** rng.random()))

    counts = {k: int(round(v * a.records)) for k, v in MIX.items()}
    records, rep = [], 0
    for _ in range(counts["complaints"]):
        records.append(complaint_document(rng, docs, length(), tok))
    for _ in range(counts["buried"]):
        records.append(buried_record(rng.choice(kev), length(), rng))
    for _ in range(counts["evidence"]):
        rep += 1
        records.append(rng.choice(synthetic.evidence_records(rng.choice(synthetic.TYPES), rng.choice(synthetic.FILLERS),
                                                             length(), rng.random(), rep)))
    for d in rng.sample(docs, counts["replay_documents"]):
        records.append({"state": d["state"], "questions": d["questions"], "_meta": {"family": "documents_alone"}})
    for base in rng.sample(kev, counts["replay_decision"]):
        records.append({**base, "_meta": {**base.get("_meta", {}), "family": "kev_short"}})
    rng.shuffle(records)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps({"state": r["state"], "questions": r["questions"], "_meta": r.get("_meta", {})}) + "\n")
    fam = {}
    for r in records:
        k = r.get("_meta", {}).get("family", "?")
        fam[k] = fam.get(k, 0) + 1
    print(f"wrote {len(records)} records, {sum(len(r['questions']) for r in records)} questions -> {out}; families {fam}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/train-delta-v1/train.jsonl")
    p.add_argument("--records", type=int, default=2400)
    p.add_argument("--min_len", type=int, default=1000)
    p.add_argument("--max_len", type=int, default=31000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
