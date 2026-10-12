#!/usr/bin/env bash
# Teacher: a Kev model (released Kev-0.8B unless --model says otherwise) reads every item of an items-alone suite by
# itself (raw logits) -> $OUT/<name>/rows.json.
#
#   bash infra/runpod/teacher.sh <suite under $KEV/evals> <name> [--model <run in $OUT | Hub id>]
source "$(dirname "$0")/env.sh"
[ $# -eq 2 ] || [ $# -eq 4 ] && [ "${3:---model}" = "--model" ] || { echo "usage: teacher.sh <suite> <name> [--model <run>]"; exit 2; }
suite=$1 name=$2 teacher=$(model "${4:-$RELEASE}")
cd "$KEV"
stage "teacher $name ($suite, $teacher)"
run "$OUT/$name.log" $PY -m kev.benchmark --run "$teacher" --suite evals/$suite --out "$OUT/$name" --device cuda \
  || { stage "FAILED teacher $name"; exit 1; }
