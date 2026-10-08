"""Minimal async client for TypeSafe's /v1/systemone with rate limiting, retries and an on-disk cache."""
import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def api_key() -> str:
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("TYPESAFE_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("TYPESAFE_API_KEY not set (env or .env)")


class Limiter:
    """Token buckets for requests/second and (estimated) input tokens/second."""

    def __init__(self, rps: float, tps: float):
        self.rps, self.tps = rps, tps
        self.req, self.tok = 1.0, tps / 4
        self.t = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self, ntok: float):
        ntok = min(ntok, self.tps)
        while True:
            async with self.lock:
                now = time.monotonic()
                dt, self.t = now - self.t, now
                self.req = min(self.rps, self.req + dt * self.rps)
                self.tok = min(self.tps, self.tok + dt * self.tps)
                if self.req >= 1 and self.tok >= ntok:
                    self.req -= 1
                    self.tok -= ntok
                    return
                wait = max((1 - self.req) / self.rps, (ntok - self.tok) / self.tps, 0.01)
            await asyncio.sleep(wait)


class Cache:
    """Append-only JSONL cache keyed by the sha256 of the request body. Final (non-retryable) results only."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists():
            for line in path.open():
                rec = json.loads(line)
                self.data[rec["key"]] = rec
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("a")

    @staticmethod
    def key(body: dict) -> str:
        return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def put(self, rec: dict):
        self.data[rec["key"]] = rec
        self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.fh.flush()


class Jev:
    def __init__(self, cache_path: Path, rps: float = 16, tps: float = 200_000, concurrency: int = 32,
                 chars_per_token: float = 3.3):
        self.cache = Cache(cache_path)
        self.limiter = Limiter(rps, tps)
        self.sem = asyncio.Semaphore(concurrency)
        self.cpt = chars_per_token
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0),
                                        headers={"Authorization": f"Bearer {api_key()}"},
                                        limits=httpx.Limits(max_connections=concurrency))
        self.stats = {"calls": 0, "cached": 0, "retries": 0, "errors": 0, "input_tokens": 0}

    async def close(self):
        await self.client.aclose()
        self.cache.fh.close()

    async def ask(self, state, questions: dict, model: str = MODEL) -> dict:
        """Returns the cache record: {key, status, body (response json or error), latency_ms}."""
        body = {"state": state, "model": model, "questions": questions}
        key = Cache.key(body)
        if key in self.cache.data:
            self.stats["cached"] += 1
            return self.cache.data[key]
        est = len(json.dumps(body)) / self.cpt + 400
        async with self.sem:
            for attempt in range(8):
                await self.limiter.acquire(est)
                t0 = time.perf_counter()
                try:
                    r = await self.client.post(API, json=body)
                    status, payload = r.status_code, (r.json() if r.content else {})
                except (httpx.TransportError, json.JSONDecodeError) as e:
                    status, payload = -1, {"error": repr(e)}
                latency = (time.perf_counter() - t0) * 1000
                if status == 200 or (status not in RETRY_STATUS and status != -1):
                    break
                self.stats["retries"] += 1
                await asyncio.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
        self.stats["calls"] += 1
        rec = {"key": key, "status": status, "body": payload, "latency_ms": round(latency, 1)}
        if status == 200:
            self.stats["input_tokens"] += payload.get("usage", {}).get("input_tokens", 0)
            self.cache.put(rec)
        else:
            self.stats["errors"] += 1
            if status in (400, 422):  # deterministic rejections: cache so reruns don't pay again
                self.cache.put(rec)
        return rec
