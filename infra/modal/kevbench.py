"""Score checkpoints on Kev's frozen long-document suite with Kev 1.0's own benchmark tool, in the environment of Kev's
published reads (H100, torch 2.8 / CUDA 12.8, transformers 5.17, flash-linear-attention 0.5.2, causal-conv1d 1.7.0).

The suite (evals/longdoc-v1, development partition) is rebuilt locally from public CUAD with Kev's own builder
(scripts/build_longdoc_v1.py @ a0255ce) and matches Kev's manifest byte for byte; kev.suite.load_split re-checks its
sha256 before scoring. Kev's code is the sparse checkout in data/kev (github.com/jaredpalmer/kev @ fe64b12, Apache-2.0).

    uv run modal volume put jev-runs data/longdoc-v1-rebuild/development.jsonl /longdoc-v1/development.jsonl
    uv run modal run infra/modal/kevbench.py --runs-arg kev-1.0=jaredpalmer/kev-0.8b,full-long=/runs/kevbench/ckpt/full-long
    uv run modal run infra/modal/kevbench.py --runs-arg packed-kev-1.0=jaredpalmer/kev-0.8b@packed-complaints-v0   # NAME=RUN@SUITE
    uv run modal volume get jev-runs /kevbench data/kevbench/results

Long reads should be submitted with `modal deploy` + Function.spawn (see `submit` below): work dispatched by a local
`modal run` is cancelled when that local process dies, even with --detach.

KEV_TEMPERATURE=1.0 scores raw logits, as Kev's own read of Kev-0.8B did (runs/r28-08b-r15-longdoc); accuracy and the
validated-context rule do not depend on the temperature.
"""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
KEV = ROOT / "data" / "kev"
CAUSAL_CONV1D = ("https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/"
                 "causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl")

app = modal.App("jev-kevbench")
runs_volume = modal.Volume.from_name("jev-runs")
hf_cache = modal.Volume.from_name("jev-hf-cache")
image = (  # mirrors Kev's modal_app.py image (minus serving and test extras)
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_sync(uv_project_dir=str(KEV), groups=[])
    .uv_pip_install("flash-linear-attention==0.5.2", "triton>=3.7.1")
    .uv_pip_install(CAUSAL_CONV1D, extra_options="--no-deps")
    .env({"HF_HOME": "/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONUNBUFFERED": "1", "TRITON_CACHE_DIR": "/hf/triton-cache", "KEV_TEMPERATURE": "1.0"})
    .add_local_dir(KEV / "kev", "/root/kev")
    .add_local_file(KEV / "evals" / "longdoc-v1" / "manifest.json", "/root/evals/longdoc-v1/manifest.json")
    .add_local_file(ROOT / "data" / "packed-complaints-v0" / "manifest.json", "/root/evals/packed-complaints-v0/manifest.json")
)


@app.function(image=image, gpu="H100", cpu=2, memory=(32768, 131072), timeout=4 * 3600,
              volumes={"/runs": runs_volume, "/hf": hf_cache})
def bench(name: str, run: str, suite: str = "longdoc-v1") -> dict:
    """kev.benchmark for one checkpoint (Hub id or /runs path) on a suite's development partition (its file staged on
    the volume at /runs/<suite>/development.jsonl) -> /runs/kevbench/<name>."""
    import shutil
    import subprocess
    import sys
    suite_dir = Path("/root/evals") / suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    if Path(f"/runs/{suite}/manifest.json").exists():   # suites staged on the volume carry their own manifest
        shutil.copyfile(f"/runs/{suite}/manifest.json", suite_dir / "manifest.json")
    shutil.copyfile(f"/runs/{suite}/development.jsonl", suite_dir / "development.jsonl")
    out = Path("/runs/kevbench") / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not any(out.iterdir()):
        out.rmdir()   # kev.benchmark refuses an existing output directory; an empty one is a failed start

    subprocess.run([sys.executable, "-m", "kev.benchmark", "--run", run, "--suite", str(suite_dir), "--out", str(out),
                    "--device", "cuda"], check=True, cwd="/root")
    runs_volume.commit()
    report = json.loads((out / "report.json").read_text())
    return {"name": name, "acc": report["clean"]["acc"], "n": report["clean"]["n"], "environment": report.get("environment")}


@app.local_entrypoint()
def main(runs_arg: str = "kev-1.0=jaredpalmer/kev-0.8b,full-long=/runs/kevbench/ckpt/full-long"):
    jobs = []
    for spec in runs_arg.split(","):
        name, rest = spec.split("=", 1)
        run, _, suite = rest.partition("@")
        jobs.append((name, run, suite or "longdoc-v1"))
    for result in bench.starmap(jobs):
        print(json.dumps(result))


@app.function(image=image, gpu="H100", cpu=4, memory=(32768, 131072), timeout=8 * 3600,
              volumes={"/runs": runs_volume, "/hf": hf_cache})
def train(name: str, data: str, args: str) -> str:
    """kev.train, Kev's own trainer, on a records file staged on the volume -> /runs/kevtrain/<name>; returns the
    checkpoint directory (the one holding head.pt)."""
    import shlex
    import subprocess
    import sys
    out = Path("/runs/kevtrain") / name
    subprocess.run([sys.executable, "-m", "kev.train", "--data", data, "--out", str(out), "--device", "cuda", *shlex.split(args)],
                   check=True, cwd="/root")
    runs_volume.commit()
    return str(min(out.rglob("head.pt"), key=lambda p: len(p.parts)).parent)


@app.function(image=image, cpu=1, timeout=16 * 3600, volumes={"/runs": runs_volume})
def pipeline(name: str, data: str, args: str, suites: list[str]) -> list[dict]:
    """Train, then score the checkpoint on each suite in parallel; runs entirely on Modal."""
    run = train.remote(name, data, args)
    calls = [bench.spawn(f"{name}--{suite.split('/')[-1]}", run, suite) for suite in suites]
    return [c.get() for c in calls]


def submit(jobs: list[tuple[str, str, str]]) -> list[str]:
    """Spawn reads on the deployed app (uv run modal deploy infra/modal/kevbench.py); they run independently of this process."""
    bench_fn = modal.Function.from_name("jev-kevbench", "bench")
    return [bench_fn.spawn(*job).object_id for job in jobs]
