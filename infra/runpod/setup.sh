#!/usr/bin/env bash
# Kev's environment on the pod (once per pod), as in infra/modal/kevbench.py: Kev's locked deps (uv sync of data/kev @ fe64b12),
# flash-linear-attention 0.5.2 with Triton >= 3.7.1, and causal-conv1d 1.7.0 (cu12, torch 2.8, CPython 3.13).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH" HF_HOME=${HF_HOME:-/workspace/hf}
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd "${KEV:-/workspace/kev}"
uv sync --frozen
uv pip install "flash-linear-attention==0.5.2" "triton>=3.7.1"
uv pip install --no-deps "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
.venv/bin/python -c "import torch, transformers, fla, causal_conv1d; print('ready:', torch.__version__, torch.cuda.get_device_name(0), 'transformers', transformers.__version__, 'fla', fla.__version__)"
