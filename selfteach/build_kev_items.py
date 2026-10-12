"""Builds the first attempt's training data (run kev-items), and the decision-records teacher suite every run reuses.

Items, each with a fixed id and header style so the teacher answers it once:
  complaints  Kev's public documents-v1 train (CFPB narratives) with their product and issue questions; none is in the
              packed-complaints test pool, which was deduplicated against documents-v1 train. Kev trained on these, and its
              answers turned out to match their labels on 93 % of questions, so this run is not label-free (build_fresh.py is)
  records     Kev's decision-v2 train records with their own questions (ten public datasets), each named by its record id
Writes, under data/selfteach/kev-items/:
  alone/          an eval-only suite with every item alone, for the teacher (`kev.benchmark --suite`, raw logits)
  alone-records/  the decision records of alone/ with each question's src restored: the teacher suite for the records,
                  reused unchanged by every other run
  alone-complaints/  the complaints of alone/ on their own: the teacher suite for this run's complaints
  packed.jsonl    2,400 training records: 45 % complaint documents, 25 % record documents (lengths log-uniform 1k-31k
                  tokens, up to 16 items asked each), 30 % items alone. Each question names its teacher item; targets
                  are attached on the GPU box by selfteach/targets.py.

    uv run python -m selfteach.build_kev_items
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from .common import CONTEXT, DATA, HEADERS, TOKENIZER, kev_records, name_it, render, write_packed

DOCS = Path(__file__).resolve().parents[1] / "data" / "cfpb" / "kev_documents_v1_train.jsonl"
OUT = DATA / "kev-items"


def main(a):
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

    # ---- items: fixed id, header and question wording; the alone record is exactly what the teacher reads
    items = []
    seen_ids = set()
    for line in open(DOCS):
        d = json.loads(line)
        if not isinstance(d["state"], str):
            continue
        cid = str(rng.randint(10_000_000, 99_999_999))
        while cid in seen_ids:
            cid = str(rng.randint(10_000_000, 99_999_999))
        seen_ids.add(cid)
        head, ref = rng.choice(HEADERS)
        block = head.format(id=cid) + "\n" + d["state"]
        qs = {k: {**q, "instructions": name_it(q["instructions"], ref.format(id=cid))} for k, q in d["questions"].items()}
        items.append({"key": f"c{cid}", "kind": "complaint", "block": block, "tokens": ntok(block) + 2, "questions": qs})
    for base in kev_records("train"):
        rid = f"R-{rng.randint(10_000, 99_999)}"
        while rid in seen_ids:
            rid = f"R-{rng.randint(10_000, 99_999)}"
        seen_ids.add(rid)
        text = render(base["state"])
        qs = {k: {**{f: v for f, v in q.items() if f != "src"}, "instructions": f"Answer about record {rid} only. {render(q.get('instructions'))}"}
              for k, q in base["questions"].items()}
        items.append({"key": rid, "kind": "record", "block": {"record": rid, "text": text}, "tokens": ntok(text) + 8, "questions": qs})
    by_kind = {k: [i for i in items if i["kind"] == k] for k in ("complaint", "record")}
    print(f"items: {len(by_kind['complaint'])} complaints, {len(by_kind['record'])} decision records")

    # ---- the teacher's suite: every item alone
    alone_dir = Path(a.out) / "alone"
    alone_dir.mkdir(parents=True, exist_ok=True)
    with open(alone_dir / "development.jsonl", "w") as f:
        for it in items:
            state = it["block"] if it["kind"] == "complaint" else [it["block"]]
            f.write(json.dumps({"state": state, "questions": it["questions"],
                                "_meta": {"id": f"selfteach-alone/{it['key']}", "source": f"selfteach_{it['kind']}", "group_id": it["key"],
                                          "variant": "clean", "split": "development"}}) + "\n")
    manifest = {"version": "kev-items-alone", "partitions": ["development"], "locked": [], "eval_only": True, "report_only": True,
                "trainable_sources": [], "eval_only_sources": ["selfteach_complaint", "selfteach_record"], "holdout_sources": [],
                "context": CONTEXT,
                "files": {"development.jsonl": {"sha256": hashlib.sha256((alone_dir / "development.jsonl").read_bytes()).hexdigest(),
                                                "records": len(items), "questions": sum(len(i["questions"]) for i in items)}}}
    json.dump(manifest, open(alone_dir / "manifest.json", "w"), indent=1)

    # ---- packed training records; each question points at its teacher item
    write_packed(rng, items, a, Path(a.out) / "packed.jsonl")
    alone_records(a.out)
    alone_complaints(a.out)


def alone_records(out):
    """The decision-record half of the teacher suite with each question's `src` restored from decision-v2 train:
    kev.benchmark reads every question's src, and main() strips it from record questions. The records keep their ids
    and text, so packed.jsonl's references are unchanged. Writes <out>/alone-records/, which every later build reuses."""
    recs = kev_records("train")
    lines = open(Path(out) / "alone" / "development.jsonl").read().splitlines()
    rec_lines = [l for l in lines if json.loads(l)["_meta"]["source"] == "selfteach_record"]
    assert len(rec_lines) == len(recs), (len(rec_lines), len(recs))
    out_dir = Path(out) / "alone-records"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "development.jsonl", "w") as f:
        for line, base in zip(rec_lines, recs):
            r = json.loads(line)
            assert set(r["questions"]) == set(base["questions"])
            for qid, q in r["questions"].items():
                q["src"] = base["questions"][qid].get("src") or f"selfteach_record_{q['type']}"
            f.write(json.dumps(r) + "\n")
    m = json.load(open(Path(out) / "alone" / "manifest.json"))
    m.update({"version": "kev-items-alone-records", "eval_only_sources": ["selfteach_record"],
              "files": {"development.jsonl": {"sha256": hashlib.sha256((out_dir / "development.jsonl").read_bytes()).hexdigest(),
                                              "records": len(rec_lines),
                                              "questions": sum(len(json.loads(l)["questions"]) for l in open(out_dir / "development.jsonl"))}}})
    json.dump(m, open(out_dir / "manifest.json", "w"), indent=1)
    print(f"records suite: {len(rec_lines)} records with src restored -> {out_dir}")



def alone_complaints(out):
    """The complaint half of the teacher suite, on its own: alone/ also holds the decision records without their `src`,
    which kev.benchmark needs, so the teacher reads the complaints from here and the records from alone-records/.
    Writes <out>/alone-complaints/."""
    lines = [l for l in open(Path(out) / "alone" / "development.jsonl").read().splitlines()
             if json.loads(l)["_meta"]["source"] == "selfteach_complaint"]
    out_dir = Path(out) / "alone-complaints"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "development.jsonl").write_text("".join(l + "\n" for l in lines))
    m = json.load(open(Path(out) / "alone" / "manifest.json"))
    m.update({"version": "kev-items-alone-complaints", "eval_only_sources": ["selfteach_complaint"],
              "files": {"development.jsonl": {"sha256": hashlib.sha256((out_dir / "development.jsonl").read_bytes()).hexdigest(),
                                              "records": len(lines), "questions": sum(len(json.loads(l)["questions"]) for l in lines)}}})
    json.dump(m, open(out_dir / "manifest.json", "w"), indent=1)
    print(f"complaints suite: {len(lines)} complaints -> {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=int, default=2400)
    p.add_argument("--asked", type=int, default=16)
    p.add_argument("--min_len", type=int, default=1000)
    p.add_argument("--max_len", type=int, default=31000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(OUT))
    main(p.parse_args())
