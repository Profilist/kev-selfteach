#!/usr/bin/env bash
# Score: one model on one or more suites -> $OUT/<as>--<suite>/rows.json (the name the reports read from
# data/kevbench/results). Suites past Kev's 64k serving window (dense-complaints-v1*) go through bench/chunked_scorer.py,
# everything else through kev.benchmark.
#
#   bash infra/runpod/score.sh <run in $OUT | Hub id> [--as <name>] <suite under $KEV/evals>...
source "$(dirname "$0")/env.sh"
[ $# -ge 2 ] || { echo "usage: score.sh <run> [--as <name>] <suite>..."; exit 2; }
run_=$(model "$1"); as=$(basename "$1"); shift
[ "${1:-}" = "--as" ] && { as=$2; shift 2; }
cd "$KEV"
failed=0
for suite in "$@"; do
  out="$OUT/$as--$(basename "$suite")"
  case $suite in
    dense-complaints-v1*) scorer=("$PY" "$REPO/bench/chunked_scorer.py");;
    *) scorer=("$PY" -m kev.benchmark);;
  esac
  stage "score $as $suite"
  run "$out.log" "${scorer[@]}" --run "$run_" --suite "evals/$suite" --out "$out" --device cuda \
    || { stage "FAILED score $as $suite"; failed=1; }
done
exit $failed
