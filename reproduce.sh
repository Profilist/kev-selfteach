#!/usr/bin/env bash
# Reproduce everything in results/. Run from the repository root, after the README's Setup.
#
#   bash reproduce.sh data             CPU: the tests and every run's training data, under data/
#   bash reproduce.sh gpu [step...]    GPU: teacher answers, training and scoring (every step, or the ones named)
#   bash reproduce.sh hosted           optional, paid APIs: Jev and Clef on both tests (keys in .env)
#   bash reproduce.sh reports          CPU: results/*.json and results/figures/
#
# GPU steps, in order (each needs the ones before it): released teachers kev-items fresh-32k fresh-32k-seed1
# fresh-32k-ndas fresh-64k fresh-128k labels-32k labels-32k-seed1 bigger-kevs kev27b-teacher. One H100 runs all of them
# except bigger-kevs and kev27b-teacher, which need Kev-27B (an H200 or similar). DRY=1 prints the GPU commands instead.
set -euo pipefail
cd "$(dirname "$0")"
KEVPY=data/kev/.venv/bin/python          # Kev's own environment (Setup)
S=data/selfteach B=data/baseline

data() {
  uv run python -m bench.complaints.build --out data/packed-complaints-v0
  uv run python -m bench.complaints.build_long          # 128k / 256k
  uv run python -m bench.complaints.split               # scoring copies that fit Kev's scorer
  PYTHONPATH=data/kev $KEVPY bench/contracts/build.py

  uv run python -m selfteach.build_kev_items
  uv run python -m selfteach.build_fresh
  uv run python -m selfteach.build_fresh --seed 1 --out $S/fresh-32k-seed1
  PYTHONPATH=data/kev $KEVPY selfteach/build_ndas.py
  uv run python -m selfteach.build_longer
  uv run python -m selfteach.build_longer --records 400 --asked 12 --min_len 32000 --max_len 125000 --seed 1 --out $S/fresh-128k
  uv run python -m baseline.build
  uv run python -m baseline.build --seed 1 --out $B/labels-32k-seed1/train.jsonl

  # drop training records that don't fit each run's training context
  for r in kev-items fresh-32k fresh-32k-seed1 fresh-32k-ndas; do $KEVPY selfteach/filter.py $S/$r/packed.jsonl; done
  $KEVPY selfteach/filter.py $S/fresh-64k/packed.jsonl 65536
  $KEVPY selfteach/filter.py $S/fresh-128k/packed.jsonl 131072
  for r in labels-32k labels-32k-seed1; do $KEVPY selfteach/filter.py $B/$r/train.jsonl; done
}

gpu() {
  export KEV=${KEV:-$PWD/data/kev} OUT=${OUT:-$PWD/data/kevbench/runs} RESULTS=${RESULTS:-$PWD/data/kevbench/results}
  [ -d /workspace ] || export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}   # on a pod, env.sh uses /workspace/hf
  local R=infra/runpod
  teacher() { bash $R/teacher.sh "$@"; }
  train() { bash $R/train.sh "$@"; }
  score() { bash $R/score.sh "$@"; }
  stage_suites() {   # copy the suites into Kev's evals/, where its scorer reads them
    local src dst
    for src in data/packed-complaints-v0-split data/packed-complaints-v0-split-long data/packed-complaints-v1 \
               data/packed-complaints-v1-64k data/packed-complaints-v1-check data/packed-contracts-v0 \
               $S/kev-items/alone-complaints $S/kev-items/alone-records $S/fresh-32k/alone $S/fresh-32k-seed1/alone \
               $S/fresh-32k-ndas/alone; do
      case $src in $S/*) dst=$(basename "$(dirname "$src")")-$(basename "$src");; *) dst=$(basename "$src");; esac
      [ -n "${DRY:-}" ] && { echo "copy $src -> $KEV/evals/$dst"; continue; }
      rm -rf "$KEV/evals/$dst" && cp -r "$src" "$KEV/evals/$dst"
    done
  }
  local suites_v0="packed-complaints-v0-split packed-contracts-v0 v7/decision-v7 v4/transfer-v4"
  local suites_v0_long="packed-complaints-v0-split-long packed-contracts-v0 v7/decision-v7 v4/transfer-v4"
  local step
  stage_suites
  for step in ${@:-released teachers kev-items fresh-32k fresh-32k-seed1 fresh-32k-ndas fresh-64k fresh-128k labels-32k labels-32k-seed1 bigger-kevs kev27b-teacher}; do
    case $step in
      released)   # released Kev-0.8B: the reference every run is compared with
        score jaredpalmer/kev-0.8b --as kev-1.0 $suites_v0 packed-complaints-v1;;
      teachers)   # released Kev-0.8B reads every training item alone
        teacher kev-items-alone-records teacher-records
        teacher kev-items-alone-complaints teacher-kev-items
        teacher fresh-32k-alone teacher-fresh
        teacher fresh-32k-seed1-alone teacher-fresh-seed1
        teacher fresh-32k-ndas-alone teacher-ndas;;
      kev-items)
        train kev-items --packed $S/kev-items/packed.jsonl --teacher teacher-kev-items,teacher-records
        score kev-items $suites_v0;;
      fresh-32k)
        train fresh-32k --packed $S/fresh-32k/packed.jsonl --teacher teacher-fresh,teacher-records
        score fresh-32k $suites_v0 packed-complaints-v1
        # check the chunked scorer used past 64k against Kev's own scorer, on two 64k documents
        score fresh-32k --as check packed-complaints-v1-check
        [ -n "${DRY:-}" ] || uv run python bench/chunked_check.py $RESULTS/check--packed-complaints-v1-check/rows.json \
                                                                  $RESULTS/fresh-32k--packed-complaints-v0-split/rows.json;;
      fresh-32k-seed1)
        train fresh-32k-seed1 --packed $S/fresh-32k-seed1/packed.jsonl --teacher teacher-fresh-seed1,teacher-records --seed 1
        score fresh-32k-seed1 $suites_v0_long;;
      fresh-32k-ndas)
        train fresh-32k-ndas --packed $S/fresh-32k-ndas/packed.jsonl --teacher teacher-fresh,teacher-records,teacher-ndas
        score fresh-32k-ndas $suites_v0_long;;
      fresh-64k)
        train fresh-64k --packed $S/fresh-64k/packed.jsonl --teacher teacher-fresh,teacher-records --init fresh-32k --max_state 65536
        score fresh-64k packed-complaints-v1 v7/decision-v7 v4/transfer-v4;;
      fresh-128k)
        train fresh-128k --packed $S/fresh-128k/packed.jsonl --teacher teacher-fresh,teacher-records --init fresh-64k --max_state 131072
        score fresh-128k packed-complaints-v1 v7/decision-v7 v4/transfer-v4;;
      labels-32k)
        train labels-32k --packed $B/labels-32k/train.jsonl
        score labels-32k $suites_v0;;
      labels-32k-seed1)
        train labels-32k-seed1 --packed $B/labels-32k-seed1/train.jsonl --seed 1
        score labels-32k-seed1 $suites_v0;;
      bigger-kevs)   # released Kev-4B, 9B and 27B, alone to 64k
        for m in 4b 9b 27b; do score jaredpalmer/kev-$m --as kev-$m packed-complaints-v1-64k; done;;
      kev27b-teacher)   # Kev-27B as the teacher, on fresh-32k-ndas's documents
        teacher fresh-32k-alone t27-fresh --model jaredpalmer/kev-27b
        teacher kev-items-alone-records t27-records --model jaredpalmer/kev-27b
        teacher fresh-32k-ndas-alone t27-ndas --model jaredpalmer/kev-27b
        train fresh-32k-ndas-kev27b --packed $S/fresh-32k-ndas/packed.jsonl --teacher t27-fresh,t27-records,t27-ndas
        score fresh-32k-ndas-kev27b packed-complaints-v1-64k packed-contracts-v0 v7/decision-v7 v4/transfer-v4;;
      *) echo "unknown step: $step"; exit 2;;
    esac
  done
}

hosted() {
  uv run python -m bench.complaints.jev
  uv run python -m bench.complaints.clef --model clef
  uv run python -m bench.complaints.clef --model clef-flash
  for m in jev clef clef-flash; do uv run python -m bench.contracts.hosted --model $m; done
}

reports() {
  uv run python -m bench.complaints.report
  uv run python -m bench.complaints.report_long
  uv run python -m bench.contracts.report
  uv run python -m selfteach.report
  for r in labels-32k labels-32k-seed1 kev-items fresh-32k fresh-32k-seed1 fresh-32k-ndas fresh-64k fresh-128k fresh-32k-ndas-kev27b; do
    uv run python -m bench.regression $r
  done
  uv run python results/figures.py
}

case ${1:-} in
  data|hosted|reports) "$1";;
  gpu) shift; gpu "$@";;
  *) sed -n '2,12p' "$0"; exit 2;;
esac
