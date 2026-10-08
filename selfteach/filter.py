"""Drop packed self-taught training records that don't fit Kev's 32k training context, in place.

kev.train needs a label on every question and the teacher's labels only exist on the GPU box, so each record is checked
with placeholder labels (the first option, true, level 0); encoding never reads labels. Run with Kev's own environment:

    data/kev/.venv/bin/python selfteach/filter.py data/selfteach/fresh-32k/packed.jsonl
"""
import json
import sys

from kev.data import materialize
import kev.model as M
from kev.model import ContextOverflow, encode, load_tokenizer, training_context

TOKENIZER = ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")


def fits(r, tok, ctx):
    r = json.loads(json.dumps(r))
    for q in r["questions"].values():
        q.pop("item", None); q.pop("item_qid", None)
        q["label"] = next(iter(q["criteria"])) if q["type"] == "choice" else (True if q["type"] == "noul" else 0)
        q.setdefault("src", f"custom_{q['type']}")
    try:
        return len(encode(tok, materialize(r), max_state=ctx["max_state"], max_branch=ctx["max_branch"], strict=True)["ids"]) <= ctx["max_packed"]
    except ContextOverflow:
        return False


def main(path, max_state=32768):
    if max_state > M.MAX_TRAIN_STATE:   # fresh-128k trains past Kev's 64k cap, which train.sh lifts on the pod too
        M.MAX_TRAIN_STATE = max_state
    tok, ctx = load_tokenizer(*TOKENIZER), training_context(max_state)
    lines = open(path).read().splitlines()
    keep = [line for line in lines if fits(json.loads(line), tok, ctx)]
    open(path, "w").write("\n".join(keep) + "\n")
    print(f"kept {len(keep)} training records, dropped {len(lines) - len(keep)} over the {max_state:,}-token training limits")


if __name__ == "__main__":
    main(sys.argv[1], *map(int, sys.argv[2:3]))
