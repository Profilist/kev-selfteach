# GPU runs on a rented pod

`bash reproduce.sh gpu` (repository root) runs every GPU step: teacher answers, training and scoring for every model in
`results/`. It is built from three scripts here, each run inside Kev's environment:

| Script | Does | Writes |
|---|---|---|
| `teacher.sh <suite> <name> [--model <run>]` | a Kev model (released Kev-0.8B by default) reads every item of an items-alone suite by itself | `$OUT/<name>/rows.json` |
| `train.sh <name> --packed <file> [--teacher <names>] [--init <run>] [--max_state <n>] [--seed <n>]` | with `--teacher`, attaches those answers as soft targets; without it, trains on the file's own labels; then continues a checkpoint with `kev.train` | `$OUT/<name>/` |
| `score.sh <run> [--as <name>] <suite>...` | scores a model with Kev's scorer (`bench/chunked_scorer.py` for the 128k/256k suite) | `$RESULTS/<name>--<suite>/rows.json` |

`setup.sh` builds Kev's environment once per pod. `self_stop.sh` stops the pod 40 minutes after a run unless
`/workspace/FETCHED` appears, so a forgotten pod can't keep billing.

## On a RunPod pod

One H100 (an H200 for the two Kev-27B steps), with a volume at `/workspace`:

```bash
cd /workspace && git clone https://github.com/Profilist/kev-selfteach.git && cd kev-selfteach
git clone https://github.com/jaredpalmer/kev.git data/kev && git -C data/kev checkout fe64b12
KEV=$PWD/data/kev bash infra/runpod/setup.sh
# copy the CPU step's output here: data/packed-*/, data/selfteach/, data/baseline/ (from `bash reproduce.sh data`)
nohup bash -c "bash reproduce.sh gpu; bash infra/runpod/self_stop.sh $PWD/data/kevbench/runs" > gpu.log 2>&1 &
```

Copy `data/kevbench/results/` (scores) and `data/kevbench/runs/` (checkpoints) back, then `touch /workspace/FETCHED`.
Each training run takes 42–49 minutes on one H100 NVL; the 128k round peaks at 80.5 GB.
