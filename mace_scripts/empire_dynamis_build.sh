#!/usr/bin/env bash
# Run with Pyxis in NVIDIA's ARM64 PyTorch 25.08 container on EmpireAI Beta.
# Expected location in the Beta clone: ~/Dynamis/mace_scripts/empire_dynamis_build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"
RESULTS_DIR="$PROJECT_ROOT/mace_results"
JOB_TAG="${SLURM_JOB_ID:-manual}"
RESULT_RUN_DIR="$RESULTS_DIR/build_$JOB_TAG"
LOG_FILE="$OUTPUT_DIR/empire_dynamis_build_$JOB_TAG.log"
BUILD_PHASE=layout-check

# Validate the mount and repository layout before installing or compiling.
for path in dynamis_state/pyproject.toml dynamis_model/pyproject.toml \
            dynamis_integrator/pyproject.toml dynamis/pyproject.toml; do
    if [[ ! -f "$PROJECT_ROOT/$path" ]]; then
        echo "Missing $PROJECT_ROOT/$path. Mount the repo root at /workspace/Dynamis." >&2
        exit 1
    fi
done
mkdir -p "$OUTPUT_DIR" "$RESULT_RUN_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
cd "$PROJECT_ROOT"

write_summary() {
    local status="$1"
    BUILD_EXIT_CODE="$status" BUILD_PHASE="$BUILD_PHASE" \
    RESULT_RUN_DIR="$RESULT_RUN_DIR" LOG_FILE="$LOG_FILE" \
    PROJECT_ROOT="$PROJECT_ROOT" python - <<'PY'
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

def installed(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None

root = Path(os.environ["PROJECT_ROOT"])
commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                        capture_output=True, text=True)
summary = {
    "status": "passed" if os.environ["BUILD_EXIT_CODE"] == "0" else "failed",
    "exit_code": int(os.environ["BUILD_EXIT_CODE"]),
    "last_phase": os.environ["BUILD_PHASE"],
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "finished_utc": datetime.now(timezone.utc).isoformat(),
    "repository": str(root),
    "commit": commit.stdout.strip() if commit.returncode == 0 else None,
    "architecture": platform.machine(),
    "torch": installed("torch"),
    "mace_torch": installed("mace-torch"),
    "cuequivariance": installed("cuequivariance"),
    "nccl_wheel": installed("nvidia-nccl-cu13"),
    "log": os.environ["LOG_FILE"],
}
out = Path(os.environ["RESULT_RUN_DIR"]) / "build_summary.json"
out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print("Build summary:", out)
PY
}
trap 'status=$?; trap - EXIT; write_summary "$status"; exit "$status"' EXIT

echo "Repository: $PROJECT_ROOT"
echo "Script log: $LOG_FILE"
echo "JSON results: $RESULT_RUN_DIR"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run under Slurm with Pyxis, not on the Beta login node." >&2
    exit 1
fi
if [[ "$(uname -m)" != aarch64 ]]; then
    echo "Expected an ARM64 Beta node/container, got $(uname -m)." >&2
    exit 1
fi
command -v nvcc >/dev/null || { echo "CUDA development image with nvcc required." >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "GPU driver is not visible in container." >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

BUILD_PHASE=container-check
python - <<'PY'
import sys
import torch
assert sys.version_info >= (3, 12), sys.version
assert torch.version.cuda and torch.version.cuda.startswith("13.0"), torch.version.cuda
assert torch.cuda.is_available(), "No CUDA device visible inside the container"
assert torch.cuda.get_device_capability(0) == (10, 0), torch.cuda.get_device_capability(0)
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
export CUDAToolkit_ROOT="$CUDA_HOME"
export TORCH_CUDA_ARCH_LIST=10.0
export CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
export CMAKE_GENERATOR=Ninja
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-4}"
export MAX_JOBS="$CMAKE_BUILD_PARALLEL_LEVEL"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

BUILD_PHASE=dependencies
VENV_DIR="$PROJECT_ROOT/.venv"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    python -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python - <<'PY'
import torch
assert torch.version.cuda and torch.version.cuda.startswith("13.0"), (
    "The venv cannot see the container PyTorch/CUDA 13.0 installation. "
    "Recreate .venv with --system-site-packages inside this image."
)
PY
python -m pip install --upgrade pip setuptools wheel
python -m pip install scikit-build-core pybind11 cmake ninja pytest scipy
python -m pip install 'mace-torch==0.3.16'
if [[ "${INSTALL_CUEQ:-1}" == 1 ]]; then
    python -m pip install \
        'cuequivariance==0.9.1' \
        'cuequivariance-torch==0.9.1' \
        'cuequivariance-ops-cu13==0.9.1' \
        'cuequivariance-ops-torch-cu13==0.9.1'
fi

# Dynamis's NCCL finder expects the Python wheel layout. Version 2.27.7
# matches the NCCL included with NVIDIA's PyTorch 25.08 image.
python -m pip install --no-deps 'nvidia-nccl-cu13==2.27.7'
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
print("NCCL paths:", paths)
PY

BUILD_PHASE=extension-build
export CMAKE_ARGS="-DPython_EXECUTABLE=$(command -v python)"
for package in dynamis_state dynamis_integrator dynamis_model dynamis; do
    python -m pip install --no-build-isolation --no-deps -e "./$package"
done

BUILD_PHASE=validation
python -m pip check
python - <<'PY'
from importlib.metadata import version
import torch
import dynamis
import dynamis_state.dynamis_state_impl
import dynamis_integrator.dynamis_integrator_impl
import dynamis_model.dynamis_model_impl
assert torch.cuda.is_available()
print("Dynamis extensions imported; MACE:", version("mace-torch"))
PY
python -m pytest -q dynamis_integrator/tests/test_import.py

if [[ "${RUN_DISTRIBUTED_TESTS:-0}" == 1 || "${RUN_MACE_TESTS:-0}" == 1 ]]; then
    python - <<'PY'
import torch
assert torch.cuda.device_count() >= 2, "Two-GPU validation needs --gres=gpu:2"
PY
fi
if [[ "${RUN_DISTRIBUTED_TESTS:-0}" == 1 ]]; then
    BUILD_PHASE=distributed-test
    torchrun --standalone --nproc_per_node=2 -m pytest \
        dynamis/tests/test_simulation_distributed.py -v
fi
if [[ "${RUN_MACE_TESTS:-0}" == 1 ]]; then
    BUILD_PHASE=mace-reference
    python dynamis_model/tests/mace/mace_standalone_reference.py \
        --output "$RESULT_RUN_DIR/reference.pt" \
        --output-json "$RESULT_RUN_DIR/reference.json"
    BUILD_PHASE=mace-distributed
    torchrun --standalone --nproc_per_node=2 \
        dynamis_model/tests/mace/test_mace_model.py run-distributed \
        --reference-artifact "$RESULT_RUN_DIR/reference.pt" \
        --strict-parity --strict-graph-interior \
        --output-json "$RESULT_RUN_DIR/distributed.json"
fi

BUILD_PHASE=complete
echo "Beta Dynamis build passed. Commit: $(git rev-parse --short HEAD)"
