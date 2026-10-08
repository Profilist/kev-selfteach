"""Hosted decision models on the all-contracts test (data/dense-contracts-v0): Jev 1.13 (TypeSafe) and Clef / Clef-flash
(Cloudflare Workers AI), with the same question groups as Kev's scorer.

- Yes/no (noul) answers come back as one probability of "yes"; a question counts as correct when that probability falls
  on the labelled side of 0.5. Choice answers are scored by their most probable option.
- Jev reads at most 32k tokens. A review file it refuses as too long is split at contract boundaries into two smaller
  review files (Kev's cuad_state layout, renumbered). Each question goes to the half that holds the contract it names,
  recursively if a half is still too long; such rows carry "split".
- Clef takes up to 64 questions per request, and every review file fits its 65,536-token window. usage.input_tokens is
  checked against what was sent, to catch silent truncation.
Jev outputs are used only as evaluation results (TypeSafe MCA §2.3(b)).

    uv run python -m bench.contracts.hosted --model jev
    uv run python -m bench.contracts.hosted --model clef-flash --budget 1.0
"""
import argparse
import asyncio
import json
import re

from ..hosted import ROOT, Jev

from ..complaints.clef import PRICE as CLEF_PRICE, Clef

SUITE = ROOT / "data" / "dense-contracts-v0"
BLOCK = re.compile(r"===== Contract (\d+) of \d+: (.*?)( \(excerpt; the rest of this contract is not included\))? =====\n(.*?)\n===== End of contract \1 =====", re.S)
PRICE = {"jev": 0.042 / 1e6, **CLEF_PRICE}


def review_file(blocks):
    """Kev's cuad_state layout for [(title, text)]."""
    n = len(blocks)
    parts = [f"Contract review file: {n} commercial contracts filed with the SEC, in no particular order.\n\n"]
    for k, (title, text) in enumerate(blocks, start=1):
        parts.append(f"===== Contract {k} of {n}: {title} =====\n{text}\n===== End of contract {k} =====\n\n")
    return "".join(parts).rstrip() + "\n"


def halves(state, questions):
    """Split a review file in two at the contract boundary nearest its middle; route each question to its contract's half."""
    blocks = [(m.group(2), m.group(4)) for m in BLOCK.finditer(state)]
    total, run, cut = sum(len(t) for _, t in blocks), 0, 1
    for i, (_, t) in enumerate(blocks):
        run += len(t)
        if run >= total / 2:
            cut = max(1, min(i + 1, len(blocks) - 1))
            break
    out = []
    for part in (blocks[:cut], blocks[cut:]):
        titles = {t for t, _ in part}
        qs = {q: d for q, d in questions.items() if re.search(r'contract titled "(.*?)"', d["instructions"]).group(1) in titles}
        if qs:
            out.append((review_file(part), qs))
    return out


def score(ans, q, label):
    if q["type"] == "noul":
        p = ans.get("noul")
        if p is None:
            p = (ans.get("probabilities") or {}).get("true")
        return (None, None) if p is None else (p >= 0.5, int((p >= 0.5) == bool(label)))
    probs = ans.get("probabilities") or {}
    pred = max(probs, key=probs.get) if probs else None
    return pred, int(pred == label)


async def main(a):
    client = Jev(ROOT / "data" / "hosted" / "jev_contracts_cache.jsonl", rps=20, tps=90_000, concurrency=24) if a.model == "jev" else Clef(a.model)
    out = ROOT / "data" / "hosted" / ("jev_contracts_rows.jsonl" if a.model == "jev" else f"clef_contracts_rows_{a.model}.jsonl")
    rows, spent = [], {"tokens": 0}

    async def ask(rid, bucket, state, questions, labels, depth=0):
        if spent["tokens"] * PRICE[a.model] > a.budget:
            return
        send = {q: {k: v for k, v in d.items() if k in ("type", "instructions", "criteria")} for q, d in questions.items()}
        rec = await client.ask(state, send)
        if rec["status"] == 400 and a.model == "jev" and depth < 3 and len(BLOCK.findall(state)) > 1:   # too long for Jev
            await asyncio.gather(*(ask(f"{rid}/h{i}", bucket, s, qs, labels, depth + 1) for i, (s, qs) in enumerate(halves(state, questions))))
            return
        if rec["status"] != 200:
            rows.extend({"id": rid, "bucket": bucket, "question": q, "label": labels[q], "pred": None, "correct": None,
                         "status": rec["status"], "error": str(rec["body"])[:200]} for q in questions)
            return
        spent["tokens"] += rec["body"].get("usage", {}).get("input_tokens", 0)
        for q, ans in rec["body"]["answers"].items():
            pred, ok = score(ans, questions[q], labels[q])
            rows.append({"id": rid, "bucket": bucket, "question": q, "label": labels[q], "pred": pred, "correct": ok,
                         "split": depth > 0, "input_tokens": rec["body"].get("usage", {}).get("input_tokens")})

    records = [json.loads(line) for line in open(SUITE / "development.jsonl")]
    for bucket in ["16k", "32k", "64k", "alone"]:
        await asyncio.gather(*(ask(r["_meta"]["id"], bucket, r["state"], r["questions"], {q: d["label"] for q, d in r["questions"].items()})
                               for r in records if r["_meta"]["bucket"] == bucket))
        ok = [x for x in rows if x["bucket"] == bucket and x["correct"] is not None]
        print(f"{bucket:>6}: {len(ok)} answered ({sum(x.get('split', False) for x in ok)} after splitting), accuracy "
              f"{sum(x['correct'] for x in ok) / max(len(ok), 1):.3f} | spent ${spent['tokens'] * PRICE[a.model]:.2f}", flush=True)
    await (client.close() if a.model == "jev" else client.client.aclose())
    with open(out, "w") as f:
        for x in rows:
            f.write(json.dumps(x, default=str) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["jev", "clef", "clef-flash"], required=True)
    p.add_argument("--budget", type=float, default=1.5)
    asyncio.run(main(p.parse_args()))
