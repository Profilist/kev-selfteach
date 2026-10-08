#!/usr/bin/env bash
# Teacher: released Kev-0.8B reads every item of an items-alone suite by itself (raw logits) -> $OUT/<name>/rows.json.
#
#   bash infra/runpod/teacher.sh <suite under $KEV/evals> <name>
source "$(dirname "$0")/env.sh"
[ $# -eq 2 ] || { echo "usage: teacher.sh <suite> <name>"; exit 2; }
suite=$1 name=$2
cd "$KEV"
stage "teacher $name ($suite)"
run "$OUT/$name.log" $PY -m kev.benchmark --run $RELEASE --suite evals/$suite --out "$OUT/$name" --device cuda \
  || { stage "FAILED teacher $name"; exit 1; }
