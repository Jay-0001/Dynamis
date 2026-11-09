#!/bin/bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:45:00
#SBATCH --job-name=dynamis-t214-mace
#SBATCH --output=/u/jsaravananindira/Dynamis/dynamis_t214_mace_%j.out

set -euo pipefail

# ---------------------------------------------------------------------------
# Cluster toolchain and source checks
# ---------------------------------------------------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the Dynamis repository root}"
ENV_PREFIX="/u/jsaravananindira/.conda/envs/dynamis-py312-t2141-cu132-cueq091"

# Preserve the working default Cray environment, including
# craype-network-ofi. Do not module purge.
module load miniforge3-python
module load gcc-native/14
module load cudatoolkit/26.5_13.2
module load slurm-env/0.1

module list
nvcc --version
nvidia-smi

test -d "${PROJECT_ROOT}/.git"
for cmake in \
    dynamis_state/CMakeLists.txt \
    dynamis_model/CMakeLists.txt \
    dynamis_integrator/CMakeLists.txt
do
    test -f "${PROJECT_ROOT}/${cmake}"
    if grep -Eq '(CXX|CUDA)_STANDARD[[:space:]]+17' \
        "${PROJECT_ROOT}/${cmake}"
    then
        echo "C++17 pin remains in ${cmake}; use the C++20 revision." >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Separate conda environment
# ---------------------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    conda create --yes --prefix "${ENV_PREFIX}" python=3.12 pip
fi
conda activate "${ENV_PREFIX}"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
    torch==2.14.1 \
    --index-url https://download.pytorch.org/whl/cu132
python -m pip install scikit-build-core cmake ninja pytest scipy
python -m pip install mace-torch==0.3.16
python -m pip check

python - <<'PY'
from importlib.metadata import version
import torch

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("Visible GPUs:", torch.cuda.device_count())
print("MACE:", version("mace-torch"))

assert torch.__version__ == "2.14.1+cu132"
assert torch.version.cuda == "13.2"
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 2
assert version("mace-torch") == "0.3.16"
PY

# ---------------------------------------------------------------------------
# Unique scratch copy and native builds
# ---------------------------------------------------------------------------
command -v rsync >/dev/null
SCRATCH_ROOT="$(mktemp -d \
    "${SLURM_TMPDIR:-/tmp}/dynamis-t214-${SLURM_JOB_ID}-XXXXXX")"

rsync -a \
    --exclude=.git \
    --exclude=skbuild \
    --exclude=build \
    --exclude=__pycache__ \
    --exclude='*.egg-info' \
    --exclude='*.so' \
    --exclude='*.pyd' \
    --exclude=tmp \
    "${PROJECT_ROOT}/" "${SCRATCH_ROOT}/"

cd "${SCRATCH_ROOT}"

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export CUDAToolkit_ROOT="${CUDA_HOME}"
export CMAKE_PREFIX_PATH="$(
    python -c 'import torch; print(torch.utils.cmake_prefix_path)'
)"
export CMAKE_GENERATOR=Ninja
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-4}"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-4}"
export CMAKE_ARGS="-DPython_EXECUTABLE=$(command -v python)"

for package in dynamis_state dynamis_model dynamis_integrator dynamis; do
    python -m pip install --no-build-isolation --no-deps "./${package}"
done

python -m pip check
python - <<'PY'
import dynamis
import dynamis_state.dynamis_state_impl
import dynamis_model.dynamis_model_impl
import dynamis_integrator.dynamis_integrator_impl
from mace.calculators import mace_mp

print("Dynamis native modules and MACE imported successfully.")
PY

# ---------------------------------------------------------------------------
# Dynamis regression and ordinary MACE baseline
# ---------------------------------------------------------------------------
torchrun --standalone --nproc_per_node=2 -m pytest \
    dynamis/tests/test_simulation_distributed.py -v

mkdir -p "${SCRATCH_ROOT}/synthetic_run"
cd "${SCRATCH_ROOT}/synthetic_run"
torchrun --standalone --nproc_per_node=2 \
    "${SCRATCH_ROOT}/dynamis/experiments/synthetic/main.py"
cd "${SCRATCH_ROOT}"

RESULT_DIR="${SCRATCH_ROOT}/mace_validation"
mkdir -p "${RESULT_DIR}/e3nn" "${RESULT_DIR}/cueq"

python dynamis_model/tests/mace/mace_standalone_reference.py \
    --builtin-system natural_chain12 \
    --model-family mp --model medium-mpa-0 \
    --output "${RESULT_DIR}/e3nn/reference.pt"

torchrun --standalone --nproc_per_node=2 \
    dynamis_model/tests/mace/test_mace_model.py run-distributed \
    --builtin-system natural_chain12 \
    --model-family mp --model medium-mpa-0 \
    --reference-artifact "${RESULT_DIR}/e3nn/reference.pt" \
    --strict-parity

# ---------------------------------------------------------------------------
# Experimental CUDA 13 cuEquivariance and MACE run
# ---------------------------------------------------------------------------
cueq_failed=0

if ! python -m pip install \
    cuequivariance==0.9.1 \
    cuequivariance-torch==0.9.1 \
    cuequivariance-ops-cu13==0.9.1 \
    cuequivariance-ops-torch-cu13==0.9.1
then
    echo "CuEq package installation failed." >&2
    cueq_failed=1
fi

if (( cueq_failed == 0 )); then
    if ! python -m pip check; then
        echo "CuEq dependency check failed." >&2
        cueq_failed=1
    fi

    if ! python - <<'PY'
from importlib.metadata import version
import torch
import cuequivariance_ops
import cuequivariance_ops_torch

print("Torch after CuEq installation:", torch.__version__)
print("Compiled CuEq CUDA 13 ops imported.")
assert torch.__version__ == "2.14.1+cu132"
assert all(version(name) == "0.9.1" for name in (
    "cuequivariance",
    "cuequivariance-torch",
    "cuequivariance-ops-cu13",
    "cuequivariance-ops-torch-cu13",
))
PY
    then
        echo "Compiled CuEq import or version check failed." >&2
        cueq_failed=1
    fi

    # Attempt the actual MACE conversion and calculation even if the
    # direct compiled-ops import failed; record both results in the log.
    if python dynamis_model/tests/mace/mace_standalone_reference.py \
        --builtin-system natural_chain12 \
        --model-family mp --model medium-mpa-0 \
        --enable-cueq \
        --output "${RESULT_DIR}/cueq/reference.pt"
    then
        if ! torchrun --standalone --nproc_per_node=2 \
            dynamis_model/tests/mace/test_mace_model.py run-distributed \
            --builtin-system natural_chain12 \
            --model-family mp --model medium-mpa-0 \
            --enable-cueq \
            --reference-artifact "${RESULT_DIR}/cueq/reference.pt" \
            --strict-parity
        then
            echo "Distributed MACE CuEq parity failed." >&2
            cueq_failed=1
        fi
    else
        echo "Standalone MACE CuEq run failed." >&2
        cueq_failed=1
    fi
fi

echo "Environment: ${ENV_PREFIX}"
echo "Scratch copy: ${SCRATCH_ROOT}"
echo "Source commit: $(git -C "${PROJECT_ROOT}" rev-parse HEAD)"

if (( cueq_failed != 0 )); then
    echo "Baseline passed; experimental CuEq validation failed." >&2
    exit 1
fi

echo "Dynamis and MACE CuEq validation passed."