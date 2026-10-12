# GPU runs (one H100 pod)

Everything that needs a GPU is three steps, each run inside Kev's environment on the pod:

| Script | Does | Writes |
|---|---|---|
| `teacher.sh <suite> <name>` | released Kev-0.8B reads every item of an items-alone suite by itself | `$OUT/<name>/rows.json` |
| `train.sh <name> --packed <file> --teacher <names> [--init <run>] [--max_state <n>] [--seed <n>]` | attaches those answers to a packed file as soft targets, then continues a checkpoint with `kev.train` | `$OUT/<name>/` |
| `score.sh <run> [--as <name>] <suite>...` | scores a model with Kev's scorer (`bench/chunked_scorer.py` for the 128k/256k suite) | `$OUT/<name>--<suite>/rows.json` |

`setup.sh` builds Kev's environment once per pod. `self_stop.sh` stops the pod 40 minutes after a run unless
`/workspace/FETCHED` appears, so a forgotten pod can't keep billing. `$OUT` defaults to `/workspace/out` and `$KEV` to
`/workspace/kev`. Set `DRY=1` to print the commands without running them.

## Pod layout

```
/workspace/kev/              Kev @ fe64b12 (data/kev), with these suites copied into evals/:
  evals/<run>-alone/                ← data/selfteach/<run>/alone          the teacher suites: kev-items, fresh-32k,
                                                                          fresh-32k-seed1, fresh-32k-ndas
  evals/kev-items-alone-records/    ← data/selfteach/kev-items/alone-records   the decision records, read once for every run
  evals/packed-complaints-v0-split/  ← data/packed-complaints-v0-split    (python -m bench.complaints.split; also -split-long)
  evals/packed-complaints-v1/        ← data/packed-complaints-v1          (and -v1-check, -v1-64k)
  evals/packed-contracts-v0/         ← data/packed-contracts-v0
/workspace/kev-selfteach/    this repository, plus data/selfteach/<run>/packed.jsonl (filtered locally with selfteach/filter.py)
/workspace/out/              results; copy <model>--<suite>/ back to data/kevbench/results/
```

## Every model in `results/`

```bash
R=/workspace/kev-selfteach/infra/runpod; D=/workspace/kev-selfteach/data/selfteach
bash $R/setup.sh

# Teacher answers: released Kev reads each training item alone. Shared by every model below.
bash $R/teacher.sh kev-items-alone-records teacher-records    # 3,432 decision records (Kev's decision-v2 train)
bash $R/teacher.sh fresh-32k-alone teacher-fresh                # 5,219 fresh CFPB complaints

# The headline model (fresh-32k): documents up to 32k, from released Kev.
bash $R/train.sh fresh-32k --packed $D/fresh-32k/packed.jsonl --teacher teacher-fresh,teacher-records
bash $R/score.sh fresh-32k packed-complaints-v0-split packed-contracts-v0 v7/decision-v7 v4/transfer-v4

# Its second seed (fresh-32k-seed1): a new sample of fresh complaints, seed 1.
bash $R/teacher.sh fresh-32k-seed1-alone teacher-fresh-seed1
bash $R/train.sh fresh-32k-seed1 --packed $D/fresh-32k-seed1/packed.jsonl --teacher teacher-fresh-seed1,teacher-records --seed 1
bash $R/score.sh fresh-32k-seed1 packed-complaints-v0-split-long packed-contracts-v0 v7/decision-v7 v4/transfer-v4

# Plus fresh ContractNLI NDAs (fresh-32k-ndas).
bash $R/teacher.sh fresh-32k-ndas-alone teacher-ndas
bash $R/train.sh fresh-32k-ndas --packed $D/fresh-32k-ndas/packed.jsonl --teacher teacher-fresh,teacher-records,teacher-ndas
bash $R/score.sh fresh-32k-ndas packed-complaints-v0-split-long packed-contracts-v0 v7/decision-v7 v4/transfer-v4

# Longer documents, same teacher answers: round 2 to 64k, round 3 to 128k (fresh-64k, fresh-128k).
bash $R/train.sh fresh-64k --packed $D/fresh-64k/packed.jsonl --teacher teacher-fresh,teacher-records --init fresh-32k --max_state 65536
bash $R/train.sh fresh-128k --packed $D/fresh-128k/packed.jsonl --teacher teacher-fresh,teacher-records --init fresh-64k --max_state 131072
for m in fresh-32k fresh-64k fresh-128k; do bash $R/score.sh $m packed-complaints-v1; done
bash $R/score.sh fresh-64k v7/decision-v7 v4/transfer-v4 && bash $R/score.sh fresh-128k v7/decision-v7 v4/transfer-v4
bash $R/score.sh jaredpalmer/kev-0.8b --as kev-1.0 packed-complaints-v1

# Before trusting any 128k/256k score: the chunked scorer against Kev's unpatched one, on two 64k documents.
bash $R/score.sh fresh-32k --as check packed-complaints-v1-check
/workspace/kev/.venv/bin/python /workspace/kev-selfteach/bench/chunked_check.py \
  /workspace/out/check--packed-complaints-v1-check/rows.json /workspace/kev/evals/packed-complaints-v1-check/reference_rows_fresh-32k.json

# Released bigger Kevs on the packed-complaints test, alone to 64k (Kev-27B needs an H200 or similar).
for m in 4b 9b 27b; do bash $R/score.sh jaredpalmer/kev-$m --as kev-$m packed-complaints-v1-64k; done

# Kev-27B as the teacher (fresh-32k-ndas-kev27b): same documents as fresh-32k-ndas, only the targets change.
T=jaredpalmer/kev-27b
bash $R/teacher.sh fresh-32k-alone t27-fresh --model $T
bash $R/teacher.sh kev-items-alone-records t27-records --model $T
bash $R/teacher.sh fresh-32k-ndas-alone t27-ndas --model $T
bash $R/train.sh fresh-32k-ndas-kev27b --packed $D/fresh-32k-ndas/packed.jsonl --teacher t27-fresh,t27-records,t27-ndas
bash $R/score.sh fresh-32k-ndas-kev27b packed-complaints-v1-64k packed-contracts-v0 v7/decision-v7 v4/transfer-v4

# First attempt (kev-items), on items from Kev's own training sets: its teacher had memorized them, so it isn't label-free.
bash $R/teacher.sh kev-items-alone teacher-kev-items
bash $R/train.sh kev-items --packed $D/kev-items/packed.jsonl --teacher teacher-kev-items
bash $R/score.sh kev-items packed-complaints-v0-split packed-contracts-v0 v7/decision-v7 v4/transfer-v4
```

To leave a pod unattended, chain the steps and end with the self-stop:

```bash
nohup bash -c "bash $R/train.sh ... && bash $R/score.sh ...; bash $R/self_stop.sh" > /workspace/out/run.log 2>&1 &
```

Each training run takes 42–46 minutes on one H100 NVL. Round 3 (128k states) peaks at 80.5 GB.
