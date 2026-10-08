"""Shared pieces of the self-taught training data, used by every build_*.py.

  items      a training item is {key, kind ("complaint" | "record"), block, tokens, questions}; the teacher reads its
             block alone, and the student reads it packed among others with the same questions
  packing    pack() builds one training document of one kind, log-uniform in length, with some of its items asked;
             write_packed() writes a training set in MIX proportions; every question names its teacher item (ask())
  Kev        kev_records() reads Kev's decision-v2 train / decision-v7 dev suites from the Kev checkout (data/kev/evals);
             render() flattens a record's state the way Kev does
"""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "data" / "kev" / "evals"
DATA = ROOT / "data" / "selfteach"   # one folder per training set: alone/ (the teacher's suite), packed.jsonl
TOKENIZER = ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
CONTEXT = {"max_state": 65536, "max_branch": 65536 + 8192, "max_packed": 2 * 65536 + 8192, "truncate": False}
MIX = {"complaint_docs": 0.45, "record_docs": 0.25, "alone": 0.30}   # write_packed's default mix
HEADERS = [("Complaint {id}:", "complaint {id}"), ("Complaint #{id}", "complaint #{id}"),
           ("[Complaint {id}]", "complaint {id}"), ("Case {id}:", "case {id}")]


def name_it(instructions: str, ref: str) -> str:
    """Point a documents-v1 question at one complaint of a multi-complaint document."""
    if "this complaint" in instructions:
        return instructions.replace("this complaint", ref)
    return instructions.rstrip("?") + f" in {ref}?"


@lru_cache
def kev_records(split: str) -> tuple[dict, ...]:
    path = EVALS / ("decision-v2/train.jsonl" if split == "train" else "v7/decision-v7/development.jsonl")
    return tuple(json.loads(line) for line in open(path))


def render(v, indent: int = 0) -> str:
    """Flatten str | object | array into the text the model sees (kev/api.py)."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
                     for k, x in v.items())


def ask(it, qid, q):
    return {**{f: v for f, v in q.items() if f not in ("label",)}, "item": f"selfteach-alone/{it['key']}", "item_qid": qid}


def pack(rng, pool, kind, a):
    """One training document of `kind` items, log-uniform in length, with up to a.asked of its items asked."""
    target = int(round(a.min_len * (a.max_len / a.min_len) ** rng.random()))
    chosen, size, keys = [], 0, set()
    while size < target:
        it = rng.choice(pool)
        if it["key"] in keys:
            continue
        if size + it["tokens"] > target and chosen:
            break
        chosen.append(it); keys.add(it["key"]); size += it["tokens"]
    asked = rng.sample(range(len(chosen)), min(a.asked, len(chosen)))
    qs = {f"{qid}_{i}": ask(chosen[i], qid, q) for i in asked for qid, q in chosen[i]["questions"].items()}
    state = "\n\n".join(it["block"] for it in chosen) if kind == "complaint" else [it["block"] for it in chosen]
    return {"state": state, "questions": qs, "_meta": {"family": f"{kind}_document", "items": len(chosen), "asked": len(asked), "approx_tokens": size}}


def write_packed(rng, items, a, path):
    by_kind = {k: [i for i in items if i["kind"] == k] for k in ("complaint", "record")}
    n = {k: int(round(v * a.records)) for k, v in MIX.items()}
    records = [pack(rng, by_kind["complaint"], "complaint", a) for _ in range(n["complaint_docs"])]
    records += [pack(rng, by_kind["record"], "record", a) for _ in range(n["record_docs"])]
    for it in rng.sample(items, n["alone"]):
        state = it["block"] if it["kind"] == "complaint" else [it["block"]]
        records.append({"state": state, "questions": {qid: ask(it, qid, q) for qid, q in it["questions"].items()},
                        "_meta": {"family": f"{it['kind']}_alone"}})
    rng.shuffle(records)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    fam = {}
    for r in records:
        fam[r["_meta"]["family"]] = fam.get(r["_meta"]["family"], 0) + 1
    print(f"training: {len(records)} records, {sum(len(r['questions']) for r in records)} questions; {fam}")
