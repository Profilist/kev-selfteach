#!/usr/bin/env bash
# Train: attach the teacher's answers to a packed file as soft targets (selfteach/targets.py), then continue a Kev
# checkpoint on it with Kev's trainer -> $OUT/<name>/ (head.pt + adapter).
#
#   bash infra/runpod/train.sh <name> --packed <packed.jsonl> [--teacher <name>[,<name>...]] \
#        [--init <run in $OUT | Hub id>] [--max_state <tokens>] [--seed <n>]
#
# --teacher lists teacher reads in $OUT (teacher.sh); their rows are pooled. Without --teacher, the file is trained on
# its own labels as it is (the label-trained baseline). --init defaults to released Kev-0.8B,
# --max_state to 32768, --seed to 0. Kev's trainer caps training states at 64k (kev.model.MAX_TRAIN_STATE); a larger
# --max_state lifts that cap in this Kev checkout for this run only (Qwen3.5's own window is 262,144).
source "$(dirname "$0")/env.sh"
name=${1:?usage: train.sh <name> --packed <file> [--teacher <names>] [--init <run>] [--max_state <n>] [--seed <n>]}; shift
packed= teachers= init=$RELEASE max_state=32768 seed=0
while [ $# -gt 0 ]; do
  case $1 in
    --packed) packed=$(abspath "$2");; --teacher) teachers=$2;; --init) init=$2;;
    --max_state) max_state=$2;; --seed) seed=$2;; *) echo "unknown option $1"; exit 2;;
  esac
  shift 2
done
[ -n "$packed" ] || { echo "--packed is required"; exit 2; }
init=$(model "$init")
rows=(); for t in ${teachers//,/ }; do rows+=("$OUT/$t/rows.json"); done
cd "$KEV"

if [ -n "$teachers" ]; then
  stage "targets $name"
  run "$OUT/$name.merge.log" $PY -c "
import json, sys
rows = [r for p in sys.argv[2:] for r in json.load(open(p))]
json.dump(rows, open(sys.argv[1], 'w')); print(len(rows), 'teacher rows')" "$OUT/$name.teacher_rows.json" "${rows[@]}" \
    && run "$OUT/$name.targets.log" $PY "$REPO/selfteach/targets.py" "$OUT/$name.teacher_rows.json" "$packed" "$OUT/$name.train.jsonl" \
    || { stage "FAILED targets $name"; exit 1; }
  data="$OUT/$name.train.jsonl"
else
  data="$packed"   # labels already in the file
fi

if [ "$max_state" -gt 65536 ]; then
  CAP='^MAX_TRAIN_STATE = SERVE_MAX_STATE$'
  if [ -n "${DRY:-}" ]; then echo "lift kev.model.MAX_TRAIN_STATE to $max_state"
  else
    sed -i "s/$CAP/MAX_TRAIN_STATE = $max_state   # lifted by kev-selfteach train.sh/" kev/model.py
    grep -q "^MAX_TRAIN_STATE = $max_state " kev/model.py || { stage "FAILED lifting the training cap"; exit 1; }
    trap 'sed -i "s/^MAX_TRAIN_STATE = $max_state   # lifted by kev-selfteach train.sh$/MAX_TRAIN_STATE = SERVE_MAX_STATE/" kev/model.py' EXIT
  fi
fi

stage "train $name (from $init, states to $max_state, seed $seed)"
run "$OUT/$name.train.log" $PY -m kev.train --data "$data" --out "$OUT/$name" --device cuda \
  --base Qwen/Qwen3.5-0.8B-Base --base_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 --init_from "$init" \
  --max_state "$max_state" --shared_prefix 1 --checkpointing 1 --dtype bf16 --weights_dtype fp32 --lr 5e-5 --accum 4 --batch 1 \
  --epochs 1 --seed "$seed" \
  || { stage "FAILED train $name"; exit 1; }
