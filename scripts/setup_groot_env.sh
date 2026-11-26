#!/usr/bin/env bash
# Create the pinned GR00T training environment in <repo>/.venv-groot (used by scripts/train.py and
# scripts/policy_server.py when --policy groot). Needs: uv, an NVIDIA driver, internet access.
#
#   make install-groot            # or: bash scripts/setup_groot_env.sh [env_dir]
#
# flash-attn cannot be compiled reliably (it needs an nvcc that matches the CUDA version of the PyTorch wheels, and takes
# hours on a small machine), so the prebuilt wheel matching the pinned torch is installed instead.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${1:-$ROOT/.venv-groot}"
CONSTRAINTS="$ROOT/envs/groot/constraints.txt"
FLASH_ATTN_VERSION="2.8.3"
TORCH_TAG="torch2.7"   # must match the torch pin in constraints.txt
PY_TAG="cp311"         # must match the Python version below

command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/" >&2; exit 1; }
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found, training will fail to decode videos (sudo apt-get install -y ffmpeg)" >&2

uv venv "$ENV_DIR" --python 3.11
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

uv pip install -c "$CONSTRAINTS" torch torchvision ninja "packaging>=24.2,<26.0"

# The prebuilt flash-attn wheel depends on the C++ ABI the installed torch was built with.
ABI="$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")"
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu12${TORCH_TAG}cxx11abi${ABI}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"

uv pip install -c "$CONSTRAINTS" "lerobot[groot]==0.4.4"

python - <<'PY'
import flash_attn, lerobot, torch, torchcodec, transformers
from torchcodec.decoders import VideoDecoder  # fails when torchcodec does not match torch or FFmpeg is missing

print("GR00T environment OK:", {
    "torch": torch.__version__, "flash-attn": flash_attn.__version__, "torchcodec": torchcodec.__version__,
    "transformers": transformers.__version__, "lerobot": getattr(lerobot, "__version__", "?"),
})
PY
