# Shared settings for the pod scripts (sourced by teacher.sh, train.sh and score.sh, not run on its own).
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # this repository on the pod
KEV=${KEV:-/workspace/kev}                                  # Kev checkout: .venv from setup.sh, suites under evals/
OUT=${OUT:-/workspace/out}                                  # every teacher read, checkpoint and score lands here
export HF_HOME=${HF_HOME:-/workspace/hf}
export TRITON_CACHE_DIR=$HF_HOME/triton-cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export KEV_TEMPERATURE=1.0   # raw logits: the teacher's full answer distribution, and unscaled scores
RELEASE=jaredpalmer/kev-0.8b
PY=$KEV/.venv/bin/python
mkdir -p "$OUT"

stage() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/STATUS"; }

run() {   # run <log> <command...>; with DRY=1, print the command instead of running it
  local log=$1; shift
  if [ -n "${DRY:-}" ]; then echo "$*"; else "$@" > "$log" 2>&1; fi
}

model() {   # a run name in $OUT, a local path, or a Hub id
  if [ -d "$OUT/$1" ]; then echo "$OUT/$1"; elif [ -e "$1" ]; then (cd "$(dirname "$1")" && echo "$PWD/$(basename "$1")"); else echo "$1"; fi
}

abspath() { case $1 in /*) echo "$1";; *) echo "$PWD/$1";; esac; }
