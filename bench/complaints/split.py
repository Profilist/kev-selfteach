"""Scoring copies of the packed-complaints suite with each long document's questions in groups that share its state.

Kev's scorer runs all of a record's questions in one batch, and 120-250 questions on a 32k-64k state run out of GPU
memory. Splitting a document's questions into groups (same state, ~1M state-plus-question tokens per batch) fixes that;
every question is still scored on its own, so the answers equal the unsplit suite's.

  packed-complaints-v0-split       every length; documents over 8k split into groups of 60 / 30 / 15 questions at 16k / 32k / 64k
  packed-complaints-v0-split-long  the alone, 32k and 64k records only (8k and 16k run on Kev's slow exact path)
  packed-complaints-v1-64k         the alone, 32k and 64k records of packed-complaints-v1, unchanged: each 64k document as one
                                  record, for bench/chunked_scorer.py, which reads it once (much faster for big models)

    uv run python -m bench.complaints.split
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "packed-complaints-v0"
GROUP = {"16k": 60, "32k": 30, "64k": 15}   # questions per record
NOTES = {"packed-complaints-v0-split": "packed-complaints-v0 with each long document's questions in groups that share its state; every "
                                      "question is scored independently, so answers equal the unsplit suite's",
         "packed-complaints-v0-split-long": "the alone, 32k and 64k records of packed-complaints-v0-split, unchanged (8k and 16k "
                                           "dropped: they run on Kev's slow exact path)",
         "packed-complaints-v1-64k": "the alone, 32k and 64k records of packed-complaints-v1, unchanged (128k and 256k dropped: "
                                    "past the 64k window Kev's models are served at)"}


def split(rec: dict) -> list[dict]:
    meta, qs = rec["_meta"], list(rec["questions"].items())
    size = GROUP.get(meta["bucket"])
    if size is None or len(qs) <= size:
        return [{**rec, "_meta": {**meta, "doc_id": meta["id"]}}]
    return [{"state": rec["state"], "questions": dict(qs[i:i + size]),
             "_meta": {**meta, "id": f"{meta['id']}#q{i // size:02d}", "doc_id": meta["id"]}} for i in range(0, len(qs), size)]


def write(name: str, records: list[dict], src: Path = SRC):
    out = ROOT / "data" / name
    out.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in records)
    (out / "development.jsonl").write_text(text)
    manifest = json.load(open(src / "manifest.json"))
    manifest.update(version=name, note=NOTES[name],
                    files={"development.jsonl": {"sha256": hashlib.sha256(text.encode()).hexdigest(), "records": len(records),
                                                 "questions": sum(len(r["questions"]) for r in records)}})
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)


def main():
    records = [part for line in open(SRC / "development.jsonl") for part in split(json.loads(line))]
    write("packed-complaints-v0-split", records)
    write("packed-complaints-v0-split-long", [r for r in records if r["_meta"]["bucket"] in ("alone", "32k", "64k")])
    v1 = ROOT / "data" / "packed-complaints-v1"
    write("packed-complaints-v1-64k", [r for line in open(v1 / "development.jsonl") if (r := json.loads(line))["_meta"]["bucket"] in ("alone", "32k", "64k")], v1)


if __name__ == "__main__":
    main()
