"""Hosted Jev on the all-complaints test (data/dense-complaints-v0): every question of a document in one request.

Jev ingests the state once per request and answers each question against it independently (docs.typesafe.ai/models.md:
billed per input token, state counted once; 32k tokens for state + the longest question, 64k for the whole request).
  alone, 8k, 16k, 32k   the suite's records as they are (a record over Jev's window comes back refused)
  64k-split             each 64k document split at the complaint boundary nearest its middle; each complaint's question
                        goes to the half that holds it (routing by the id in the question, as a developer would)
Jev outputs are used only as evaluation results here (TypeSafe MCA §2.3(b)).

    uv run python -m bench.complaints.jev --budget 4.28
"""
import argparse
import asyncio
import json
from pathlib import Path

from ..hosted import ROOT, Jev

SUITE = ROOT / "data" / "dense-complaints-v0"
OUT = ROOT / "data" / "hosted" / "jev_dense_rows.jsonl"
PRICE = 0.042 / 1e6   # $ per input token


def requests():
    """(request id, bucket, state, {qid: question without label}, {qid: label})."""
    for line in open(SUITE / "development.jsonl"):
        r = json.loads(line)
        meta, qs = r["_meta"], r["questions"]
        ask = {q: {k: v for k, v in d.items() if k in ("type", "instructions", "criteria")} for q, d in qs.items()}
        labels = {q: d["label"] for q, d in qs.items()}
        if meta["bucket"] != "64k":
            yield meta["id"], meta["bucket"], r["state"], ask, labels
            continue
        blocks = r["state"].split("\n\n")
        total, run, cut = sum(map(len, blocks)), 0, 0
        for i, b in enumerate(blocks):
            run += len(b)
            if run >= total / 2:
                cut = i + 1
                break
        for half, part in (("a", blocks[:cut]), ("b", blocks[cut:])):
            ids = {"c" + b.split(":", 1)[0].removeprefix("Complaint ").strip() for b in part}
            yield (f"{meta['id']}#{half}", "64k-split", "\n\n".join(part),
                   {q: d for q, d in ask.items() if q in ids}, {q: l for q, l in labels.items() if q in ids})


async def main(a):
    jev = Jev(ROOT / "data" / "hosted" / "jev_dense_cache.jsonl", rps=20, tps=90_000, concurrency=24)
    reqs = list(requests())
    spent = {"tokens": 0}
    rows = []

    async def one(rid, bucket, state, ask, labels):
        if spent["tokens"] * PRICE > a.budget:
            return
        rec = await jev.ask(state, ask)
        if rec["status"] == 200:
            spent["tokens"] += rec["body"].get("usage", {}).get("input_tokens", 0)
            for q, ans in rec["body"]["answers"].items():
                probs = ans.get("probabilities", {})
                pred = max(probs, key=probs.get) if probs else None
                rows.append({"id": rid, "bucket": bucket, "question": q, "label": labels[q], "pred": pred,
                             "correct": int(pred == labels[q]), "probabilities": probs})
        else:
            rows.extend({"id": rid, "bucket": bucket, "question": q, "label": labels[q], "pred": None, "correct": None,
                         "status": rec["status"], "error": str(rec["body"])[:200]} for q in ask)

    order = ["8k", "16k", "32k", "64k-split", "alone"]   # most informative first, in case the budget runs out
    for bucket in order:
        await asyncio.gather(*(one(*r) for r in reqs if r[1] == bucket))
        done = [x for x in rows if x["bucket"] == bucket]
        ok = [x for x in done if x["correct"] is not None]
        print(f"{bucket:>10}: {len(ok)}/{len(done)} answered, accuracy {sum(x['correct'] for x in ok) / max(len(ok), 1):.3f}"
              f" | spent so far ${spent['tokens'] * PRICE:.2f}", flush=True)
    await jev.close()
    with open(OUT, "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT}; {jev.stats}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=4.28)
    asyncio.run(main(p.parse_args()))
