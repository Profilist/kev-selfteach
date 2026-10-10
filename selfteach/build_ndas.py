"""Builds run fresh-32k-ndas's training data: fresh-32k's items plus fresh contracts (NDAs), each answered alone by the teacher.

fresh-32k (fresh complaints + decision records) fixed packed complaints but carried less to long contracts. This adds
contract items, so training also holds long, multi-question items in Kev's contract review-file layout.

CUAD can't be used: Kev's longdoc-v1 deals every CUAD contract to its development or locked test partition, and our
packed-contracts test reads the development half. The contracts are ContractNLI's train split instead (Koreeda and Manning,
Findings of EMNLP 2021, CC BY 4.0; stanfordnlp.github.io/contract-nli, the zip Kev's breadth-v1 builder pins, sha256
e03fc77b...). Kev's breadth-v1 suite reads only ContractNLI's dev and test splits, and no ContractNLI document is in Kev's
training data.
  - 423 train NDAs, minus any over --max_tokens and any whose word 8-grams are contained in a CUAD contract at >= 0.2
    (Kev's screen_longdoc_v1.py measure: casefold \\w+ 8-grams, grams in more than 50 CUAD contracts ignored as
    boilerplate; Kev counts overlap at 0.5).
  - Each NDA is a contract in longdoc-v1's review-file layout (cuad_state), titled by its file name.
  - Questions, drawn at random without reading any label, each naming the contract by title as longdoc-v1 does:
    - three of ContractNLI's 17 statements, in Kev's breadth-v1 wording (entailment / contradiction / not mentioned);
    - two CUAD clause-presence questions in longdoc-v1's wording, for categories in at least 10 % of CUAD contracts.
  - ContractNLI's labels are kept only to report the teacher's accuracy. The CUAD questions have no label on an NDA, so
    they get a placeholder and the task "cuad_presence_unlabelled".
Mix (--records before the 32k filter): 35 % complaint documents, 20 % record documents, 20 % review files of two or
more NDAs (4k-31k tokens), 25 % items alone. The fresh complaints and decision records, and the teacher's answers to
them, are fresh-32k's.
Writes data/selfteach/fresh-32k-ndas/{alone/ (the NDAs alone, for the teacher), packed.jsonl}; then drop over-long
records with selfteach/filter.py.

    PYTHONPATH=data/kev data/kev/.venv/bin/python selfteach/build_ndas.py
"""
import argparse
import hashlib
import importlib.util
import json
import random
import re
import urllib.parse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEV = ROOT / "data" / "kev"
CNLI = ROOT / "data" / "contractnli" / "contract-nli" / "train.json"
COMPLAINTS = ROOT / "data" / "selfteach" / "fresh-32k" / "alone" / "development.jsonl"   # fresh-32k's fresh complaints
RECORDS = ROOT / "data" / "selfteach" / "kev-items" / "alone-records" / "development.jsonl"
OUT = ROOT / "data" / "selfteach" / "fresh-32k-ndas"
MIX = {"complaint_docs": 0.35, "record_docs": 0.20, "contract_docs": 0.20, "alone": 0.25}
CONTEXT = {"max_state": 65536, "max_branch": 65536 + 8192, "max_packed": 2 * 65536 + 8192, "truncate": False}
CNLI_LABELS = {"Entailment": "entailment", "Contradiction": "contradiction", "NotMentioned": "not_mentioned"}
CNLI_CRITERIA = {"entailment": "The contract entails the statement", "contradiction": "The contract contradicts the statement",
                 "not_mentioned": "The contract does not mention it"}   # Kev's breadth-v1 (scripts/build_breadth_v1.py)
N, COMMON = 8, 50


def kev_builder():
    spec = importlib.util.spec_from_file_location("build_longdoc_v1", KEV / "scripts" / "build_longdoc_v1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grams(text):
    w = re.findall(r"\w+", text.casefold())
    return {hash(" ".join(w[i:i + N])) for i in range(len(w) - N + 1)}


def cuad_containment(ndas, cuad_texts):
    """Each NDA's largest share of 8-grams found in one CUAD contract, ignoring grams in more than COMMON contracts."""
    index = defaultdict(list)
    for key, g in ndas.items():
        for x in g:
            index[x].append(key)
    hits, df = [], defaultdict(int)
    for text in cuad_texts:
        shared = grams(text) & index.keys()
        for x in shared:
            df[x] += 1
        hits.append(shared)
    best = defaultdict(float)
    for shared in hits:
        per = defaultdict(int)
        for x in shared:
            if df[x] <= COMMON:
                for key in index[x]:
                    per[key] += 1
        for key, n in per.items():
            best[key] = max(best[key], n / len(ndas[key]))
    return best


def main(a):
    rng = random.Random(a.seed)
    L = kev_builder()
    count = L.Tokens()
    cuad, details = L.load_cuad(L.fetch_cuad(), count)
    prevalence = {k: sum(k in c["present"] for c in cuad.values()) / len(cuad) for k in details}
    categories = sorted(k for k, v in prevalence.items() if v >= 0.10)

    # ---- fresh contract items from ContractNLI train
    data = json.load(open(CNLI))
    hyps = data["labels"]
    docs = {d["id"]: d for d in data["documents"]}
    containment = cuad_containment({i: grams(d["text"]) for i, d in docs.items()}, [c["text"] for c in cuad.values()])
    items, titles, dropped = [], set(), {"overlap": 0, "length": 0}
    for i in sorted(docs):
        d = docs[i]
        text = d["text"].strip()
        if containment.get(i, 0.0) >= a.max_containment:
            dropped["overlap"] += 1
            continue
        if count(text) > a.max_tokens:
            dropped["length"] += 1
            continue
        title = re.sub(r"\s+", " ", re.sub(r"[_]+", " ", urllib.parse.unquote(d["file_name"]).rsplit(".", 1)[0])).strip()
        if title in titles:
            title = f"{title} ({i})"
        titles.add(title)
        ref = f'Consider only the contract titled "{title}" in this file.'
        q_rng = random.Random(f"selfteach-v1c:{a.seed}:{i}")   # seed string from the run's first name; kept so the data reproduces
        answers = d["annotation_sets"][0]["annotations"]
        qs = {}
        for key in sorted(q_rng.sample(sorted(hyps), 3)):
            qs[key] = {"type": "choice", "src": "contractnli",
                       "instructions": f'{ref} Statement: "{hyps[key]["hypothesis"]}" Does the contract entail, contradict or not mention this statement?',
                       "criteria": CNLI_CRITERIA, "label": CNLI_LABELS[answers[key]["choice"]]}
        for k, cat in enumerate(sorted(q_rng.sample(categories, 2))):
            qs[f"presence_{k}"] = {"type": "noul", "src": "cuad_presence_unlabelled", "instructions": f"{ref} {details[cat]}", "label": False}
        items.append({"key": f"n{i}", "kind": "contract", "title": title, "text": text, "tokens": count(text) + 40, "questions": qs})
    worst = max(containment.values()) if containment else 0.0
    print(f"contracts: {len(items)} ContractNLI train NDAs kept; dropped {dropped['overlap']} for CUAD overlap "
          f"(largest containment {worst:.3f}) and {dropped['length']} over {a.max_tokens:,} tokens")

    # ---- the teacher's suite: the NDAs alone, each in a review file of its own
    def contract_state(chosen):
        return L.cuad_state([(it["title"], it["text"], False) for it in chosen])[0]

    alone_dir = Path(a.out) / "alone"
    alone_dir.mkdir(parents=True, exist_ok=True)
    with open(alone_dir / "development.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps({"state": contract_state([it]), "questions": it["questions"],
                                "_meta": {"id": f"selfteach-alone/{it['key']}", "source": "selfteach_contract", "group_id": it["key"],
                                          "variant": "clean", "split": "development"}}) + "\n")
    manifest = {"version": "fresh-32k-ndas-alone", "partitions": ["development"], "locked": [], "eval_only": True, "report_only": True,
                "trainable_sources": [], "eval_only_sources": ["selfteach_contract"], "holdout_sources": [], "context": CONTEXT,
                "files": {"development.jsonl": {"sha256": hashlib.sha256((alone_dir / "development.jsonl").read_bytes()).hexdigest(),
                                                "records": len(items), "questions": sum(len(i["questions"]) for i in items)}}}
    json.dump(manifest, open(alone_dir / "manifest.json", "w"), indent=1)

    # ---- fresh-32k's fresh complaints and the decision records, as items
    for line in open(COMPLAINTS):
        r = json.loads(line)
        items.append({"key": r["_meta"]["id"].split("/", 1)[1], "kind": "complaint", "block": r["state"],
                      "tokens": count(r["state"]) + 2, "questions": r["questions"]})
    for line in open(RECORDS):
        r = json.loads(line)
        items.append({"key": r["_meta"]["id"].split("/", 1)[1], "kind": "record", "block": r["state"][0],
                      "tokens": count(r["state"][0]["text"]) + 8, "questions": r["questions"]})
    by_kind = {k: [i for i in items if i["kind"] == k] for k in ("complaint", "record", "contract")}
    print("items: " + ", ".join(f"{len(v)} {k}s" for k, v in by_kind.items()))

    # ---- packed training records; each question points at its teacher item
    def ask(it, qid, q):
        return {**{f: v for f, v in q.items() if f != "label"}, "item": f"selfteach-alone/{it['key']}", "item_qid": qid}

    def state_of(kind, chosen):
        if kind == "complaint":
            return "\n\n".join(it["block"] for it in chosen)
        if kind == "record":
            return [it["block"] for it in chosen]
        return contract_state(chosen)

    def pack(kind, min_len, min_items):
        pool = by_kind[kind]
        target = int(round(min_len * (a.max_len / min_len) ** rng.random()))
        chosen, size, keys = [], 0, set()
        while size < target or len(chosen) < min_items:
            it = rng.choice(pool)
            if it["key"] in keys:
                continue
            if size + it["tokens"] > target and len(chosen) >= min_items:
                break
            chosen.append(it); keys.add(it["key"]); size += it["tokens"]
        asked = rng.sample(range(len(chosen)), min(a.asked, len(chosen)))
        qs = {f"{qid}_{i}": ask(chosen[i], qid, q) for i in asked for qid, q in chosen[i]["questions"].items()}
        return {"state": state_of(kind, chosen), "questions": qs,
                "_meta": {"family": f"{kind}_document", "items": len(chosen), "asked": len(asked), "approx_tokens": size}}

    n = {k: int(round(v * a.records)) for k, v in MIX.items()}
    records = [pack("complaint", a.min_len, 1) for _ in range(n["complaint_docs"])]
    records += [pack("record", a.min_len, 1) for _ in range(n["record_docs"])]
    records += [pack("contract", a.min_contract_len, 2) for _ in range(n["contract_docs"])]
    for it in rng.sample(items, n["alone"]):
        records.append({"state": state_of(it["kind"], [it]), "questions": {qid: ask(it, qid, q) for qid, q in it["questions"].items()},
                        "_meta": {"family": f"{it['kind']}_alone"}})
    rng.shuffle(records)
    with open(Path(a.out) / "packed.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    fam = defaultdict(int)
    for r in records:
        fam[r["_meta"]["family"]] += 1
    print(f"training: {len(records)} records, {sum(len(r['questions']) for r in records)} questions; {dict(fam)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=int, default=2400)
    p.add_argument("--asked", type=int, default=16)
    p.add_argument("--min_len", type=int, default=1000)
    p.add_argument("--min_contract_len", type=int, default=4000)
    p.add_argument("--max_len", type=int, default=31000)
    p.add_argument("--max_tokens", type=int, default=7500)
    p.add_argument("--max_containment", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(OUT))
    main(p.parse_args())
