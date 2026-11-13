#!/usr/bin/env bash
# Run inside the ARM64 NVIDIA PyTorch 25.08 container on EmpireAI beta_test.
# Pyxis mounts this repository at /workspace/Dynamis before invoking this file.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this script in a Slurm allocation with Pyxis; see the launch command." >&2
    exit 1
fi
if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "Expected an ARM64 Beta node/container, got $(uname -m)." >&2
    exit 1
fi
for path in dynamis_state/pyproject.toml dynamis_model/pyproject.toml \
            dynamis_integrator/pyproject.toml dynamis/pyproject.toml; do
    test -f "$path" || { echo "Missing $PROJECT_ROOT/$path" >&2; exit 1; }
done
command -v nvcc >/dev/null || { echo "CUDA development image with nvcc required." >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "GPU driver is not visible in container." >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# NGC 25.08 includes ARM64 Python 3.12, PyTorch 2.8, CUDA 13.0 and NCCL 2.27.7.
python - <<'PY'
import platform
import sys
import torch
assert sys.version_info >= (3, 12), sys.version
assert platform.machine() == "aarch64"
assert torch.version.cuda and torch.version.cuda.startswith("13.0"), torch.version.cuda
assert torch.cuda.is_available(), "No CUDA device visible inside the container"
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
PY

export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
export CUDAToolkit_ROOT="$CUDA_HOME"
export TORCH_CUDA_ARCH_LIST="10.0"
export CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
export CMAKE_GENERATOR=Ninja
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-4}"
export MAX_JOBS="$CMAKE_BUILD_PARALLEL_LEVEL"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

# Keep an editable environment on the mounted filesystem. System site packages
# expose the image's matching PyTorch build; pip installs here persist across jobs.
VENV_DIR="$PROJECT_ROOT/.venv"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python - <<'PY'
import torch
assert torch.version.cuda and torch.version.cuda.startswith("13.0"), (
    "The venv cannot see the container PyTorch/CUDA 13.0 installation. "
    "Use a fresh .venv created by this script inside this container image."
)
PY

python -m pip install --upgrade pip setuptools wheel
python -m pip install scikit-build-core pybind11 cmake ninja pytest scipy

# MACE is used by Dynamis's MACE backend. Install the CUDA 13 ARM64 CuEq
# counterparts to the CUDA 12 packages in the original Delta build script.
python -m pip install 'mace-torch==0.3.16'
if [[ "${INSTALL_CUEQ:-1}" == 1 ]]; then
    python -m pip install \
        'cuequivariance==0.9.1' \
        'cuequivariance-torch==0.9.1' \
        'cuequivariance-ops-cu13==0.9.1' \
        'cuequivariance-ops-torch-cu13==0.9.1'
fi

# Dynamis's current NCCL finder looks for nvidia.nccl/include and lib in a
# Python wheel, rather than the system layout used by the NGC image. Match the
# image's NCCL version to expose those paths without changing repository code.
python -m pip install --no-deps 'nvidia-nccl-cu13==2.27.7'
python dynamis_state/cmake/torch_nccl_paths.py
python - <<'PY'
import json
import subprocess
import sys
import torch
assert torch.version.cuda and torch.version.cuda.startswith("13.0"), (
    "Dependency installation replaced the container's PyTorch build"
)
paths = json.loads(subprocess.check_output(
    [sys.executable, "dynamis_state/cmake/torch_nccl_paths.py"], text=True
))
assert paths["include"] and paths["lib"] and not paths["static"], paths
PY

export CMAKE_ARGS="-DPython_EXECUTABLE=$(command -v python)"
for package in dynamis_state dynamis_integrator dynamis_model dynamis; do
    python -m pip install --no-build-isolation --no-deps -e "./$package"
done

python -m pip check
python - <<'PY'
from importlib.metadata import version
import torch
import dynamis
import dynamis_state.dynamis_state_impl
import dynamis_integrator.dynamis_integrator_impl
import dynamis_model.dynamis_model_impl
assert torch.cuda.is_available()
print("Dynamis native extensions imported successfully")
print("MACE:", version("mace-torch"))
print("Commit and GPU validation can now run in this environment")
PY
python -m pytest -q dynamis_integrator/tests/test_import.py

if [[ "${RUN_DISTRIBUTED_TESTS:-0}" == 1 ]]; then
    if [[ "${SLURM_GPUS_ON_NODE:-0}" -lt 2 ]]; then
        echo "RUN_DISTRIBUTED_TESTS=1 needs at least two allocated GPUs." >&2
        exit 1
    fi
    torchrun --standalone --nproc_per_node=2 -m pytest \
        dynamis/tests/test_simulation_distributed.py -v
fi

echo "Beta Dynamis build passed. Commit: $(git rev-parse --short HEAD)"
