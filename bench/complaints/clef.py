"""Cloudflare's Clef / Clef-flash (Workers AI, Jev-compatible) on the packed-complaints test (data/packed-complaints-v0).

Clef takes 1-64 typed questions per request and silently truncates a state that does not fit its 65,536-token window
(developers.cloudflare.com/workers-ai/models/clef), so long documents are asked in groups small enough that state plus
questions fit: up to 64 questions per request through 16k, 32 at 32k, 16 at 64k. Each response's usage.input_tokens is
compared with the Qwen3.5 token count of what was sent, and any request that looks truncated is flagged.

Credentials come from the environment or .env (CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN) and are never printed.

    uv run python -m bench.complaints.clef --model clef-flash --smoke          # one question, to check access and format
    uv run python -m bench.complaints.clef --model clef-flash --budget 2.0
"""
import argparse
import asyncio
import json
import os
import random
import time

import httpx

from ..hosted import ROOT, Cache, Limiter

SUITE = ROOT / "data" / "packed-complaints-v0"
PRICE = {"clef": 0.24 / 1e6, "clef-flash": 0.09 / 1e6}   # $ per input token, Workers AI list price
GROUP = {"alone": 64, "8k": 64, "16k": 64, "32k": 32, "64k": 16}
RETRY = {408, 409, 429, 500, 502, 503, 504, 529}


def credential(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"{name} is not set (environment or .env)")


class Clef:
    def __init__(self, model: str, rps: float = 10, tps: float = 200_000, concurrency: int = 16):
        self.model = model
        self.url = (f"https://api.cloudflare.com/client/v4/accounts/{credential('CLOUDFLARE_ACCOUNT_ID')}"
                    f"/ai/run/@cf/cloudflare/{model}")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=15.0),
                                        headers={"Authorization": f"Bearer {credential('CLOUDFLARE_API_TOKEN')}"})
        self.cache = Cache(ROOT / "data" / "hosted" / f"clef_complaints_cache_{model}.jsonl")
        self.limiter, self.sem = Limiter(rps, tps), asyncio.Semaphore(concurrency)

    async def ask(self, state: str, questions: dict) -> dict:
        body = {"model": self.model, "state": state, "questions": questions}
        key = Cache.key(body)
        if key in self.cache.data:
            return self.cache.data[key]
        async with self.sem:
            for attempt in range(8):
                await self.limiter.acquire(len(state) / 4)
                t0 = time.perf_counter()
                try:
                    r = await self.client.post(self.url, json=body)
                    status, payload = r.status_code, (r.json() if r.content else {})
                except (httpx.TransportError, json.JSONDecodeError) as e:
                    status, payload = -1, {"error": repr(e)}
                if status == 200 or (status not in RETRY and status != -1):
                    break
                await asyncio.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
        result = payload.get("result", payload) if isinstance(payload, dict) else payload
        rec = {"key": key, "status": status, "body": result, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        if status == 200:
            self.cache.put(rec)
        return rec


def requests(tok):
    """(request id, bucket, state, questions without labels, labels, Qwen tokens sent) per question group."""
    for line in open(SUITE / "development.jsonl"):
        r = json.loads(line)
        meta, qids = r["_meta"], list(r["questions"])
        n = GROUP[meta["bucket"]]
        for k in range(0, len(qids), n):
            group = qids[k:k + n]
            ask = {q: {f: v for f, v in r["questions"][q].items() if f in ("type", "instructions", "criteria")} for q in group}
            sent = meta["state_tokens"] + len(tok(json.dumps(ask), add_special_tokens=False)["input_ids"])
            yield (f"{meta['id']}#g{k // n:02d}", meta["bucket"], r["state"], ask,
                   {q: r["questions"][q]["label"] for q in group}, sent)


async def main(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base", revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
    clef = Clef(a.model)
    reqs = list(requests(tok))
    if a.smoke:
        rid, bucket, state, ask, labels, sent = reqs[0]
        q = next(iter(ask))
        rec = await clef.ask(state, {q: ask[q]})
        ans = (rec["body"] or {}).get("answers", {}).get(q, {}) if rec["status"] == 200 else rec["body"]
        print(f"status {rec['status']} | {a.model} answer for {q}: {json.dumps(ans)[:300]} | label {labels[q]} | usage {(rec['body'] or {}).get('usage') if rec['status'] == 200 else '-'}")
        return
    out = ROOT / "data" / "hosted" / f"clef_complaints_rows_{a.model}.jsonl"
    rows, spent, truncated = [], {"tokens": 0}, []

    async def one(rid, bucket, state, ask, labels, sent):
        if spent["tokens"] * PRICE[a.model] > a.budget:
            return
        rec = await clef.ask(state, ask)
        if rec["status"] != 200:
            rows.extend({"id": rid, "bucket": bucket, "question": q, "label": labels[q], "pred": None, "correct": None,
                         "status": rec["status"], "error": str(rec["body"])[:200]} for q in ask)
            return
        used = rec["body"].get("usage", {}).get("input_tokens", 0)
        spent["tokens"] += used
        if used and used < 0.9 * sent:
            truncated.append((rid, used, sent))
        for q, ans in rec["body"]["answers"].items():
            probs = ans.get("probabilities", {})
            pred = max(probs, key=probs.get) if probs else None
            rows.append({"id": rid, "bucket": bucket, "question": q, "label": labels[q], "pred": pred,
                         "correct": int(pred == labels[q]), "probabilities": probs, "input_tokens": used, "sent_tokens": sent})

    for bucket in ["8k", "16k", "32k", "64k", "alone"]:
        await asyncio.gather(*(one(*r) for r in reqs if r[1] == bucket))
        ok = [x for x in rows if x["bucket"] == bucket and x["correct"] is not None]
        print(f"{bucket:>6}: {len(ok)} answered, accuracy {sum(x['correct'] for x in ok) / max(len(ok), 1):.3f} | "
              f"spent ${spent['tokens'] * PRICE[a.model]:.2f} | possibly truncated requests so far: {len(truncated)}", flush=True)
    await clef.client.aclose()
    with open(out, "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    print(f"wrote {len(rows)} rows -> {out}; truncated: {truncated[:5]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(PRICE), default="clef-flash")
    p.add_argument("--budget", type=float, default=2.0, help="stop sending once this many dollars of input tokens are used")
    p.add_argument("--smoke", action="store_true")
    asyncio.run(main(p.parse_args()))
