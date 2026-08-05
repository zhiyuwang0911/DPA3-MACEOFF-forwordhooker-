#!/usr/bin/env bash
# Install DPA3-PROBE stack for NVIDIA B200 (Blackwell, sm_100).
#
# Prerequisites on the cluster:
#   - CUDA toolkit >= 12.8 loaded (module load cuda/...)
#   - conda/mamba available
#
# Usage:
#   bash scripts/install_b200.sh
#   conda activate dpa3_probe_b200
#   python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-dpa3_probe_b200}"

# PyTorch CUDA wheel tag. B200 needs >=12.8; use cu128 or cu129.
# Override if your cluster docs recommend another index, e.g.:
#   TORCH_CUDA_INDEX=cu129 bash scripts/install_b200.sh
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-cu128}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA_INDEX}"

echo "==> Checking CUDA"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "WARNING: nvcc not found. Load a CUDA >=12.8 module first, e.g.:"
  echo "  module load cuda/12.8.0"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  echo "CUDA_HOME=${CUDA_HOME}"
elif [[ -n "${CUDA_PATH:-}" ]]; then
  export CUDA_HOME="${CUDA_PATH}"
  echo "CUDA_HOME=${CUDA_HOME} (from CUDA_PATH)"
else
  echo "WARNING: CUDA_HOME unset; deepmd CUDA build may fail."
fi

echo "==> Creating conda env ${ENV_NAME} from environment_b200.yml"
if command -v mamba >/dev/null 2>&1; then
  mamba env create -f "${ROOT}/environment_b200.yml" -n "${ENV_NAME}" || \
    mamba env update -f "${ROOT}/environment_b200.yml" -n "${ENV_NAME}"
else
  conda env create -f "${ROOT}/environment_b200.yml" -n "${ENV_NAME}" || \
    conda env update -f "${ROOT}/environment_b200.yml" -n "${ENV_NAME}"
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "==> Installing PyTorch (${TORCH_CUDA_INDEX})"
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    major, minor = torch.cuda.get_device_capability(0)
    print(f"capability: sm_{major}{minor}")
PY

echo "==> Installing DeePMD-kit with CUDA + PyTorch custom OPs"
export DP_VARIANT=cuda
export DP_ENABLE_PYTORCH=1
# Help CMake find the toolkit if modules set CUDA_HOME
if [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDAToolkit_ROOT="${CUDA_HOME}"
  export CUDA_PATH="${CUDA_HOME}"
fi

# Prefer a release with DPA3 (>=3.1). If the wheel lacks CUDA ops on B200,
# fall back to source install from GitHub.
set +e
pip install "deepmd-kit[torch]>=3.1.0"
DP_PIP_STATUS=$?
set -e

python - <<'PY' || DP_IMPORT_FAIL=1
import deepmd
print("deepmd:", deepmd.__version__)
PY

if [[ "${DP_PIP_STATUS}" -ne 0 || "${DP_IMPORT_FAIL:-0}" -eq 1 ]]; then
  echo "==> Wheel install failed or incomplete; building DeePMD from source"
  TMPDIR_DP="$(mktemp -d)"
  git clone --depth 1 https://github.com/deepmodeling/deepmd-kit.git "${TMPDIR_DP}/deepmd-kit"
  (
    cd "${TMPDIR_DP}/deepmd-kit"
    export DP_VARIANT=cuda
    export DP_ENABLE_PYTORCH=1
    pip install -e ".[torch]" -v
  )
fi

python - <<'PY'
import deepmd, torch
print("OK deepmd", deepmd.__version__, "| torch.cuda", torch.cuda.is_available())
PY

echo ""
echo "Done. Activate with:"
echo "  conda activate ${ENV_NAME}"
echo "Then run e.g.:"
echo "  python save_atomic_dpa3.py --model /path/DPA3-L6.pt --xyz data.xyz --output out.npz --device cuda"
