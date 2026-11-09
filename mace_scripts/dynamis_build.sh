#!/bin/bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --job-name=dynamis-clean-install
#SBATCH --output=/u/jsaravananindira/dynamis_clean_install_%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
CONDA_ENVS_PATH="/u/jsaravananindira/.conda/envs"
CONDA_ENV_PREFIX="${CONDA_ENVS_PATH}/dynamis-py312-cu129-clean"

module purge
module load miniforge3-python
module load gcc-native/14
module load cuda/12.9
module load slurm-env/0.1

source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH

cd "${PROJECT_ROOT}"

test -d .git
test -f dynamis_state/pyproject.toml
test -f dynamis_model/pyproject.toml
test -f dynamis_integrator/pyproject.toml
test -f dynamis/pyproject.toml

if [[ ! -x "${CONDA_ENV_PREFIX}/bin/python" ]]; then
    conda create \
        --yes \
        --prefix "${CONDA_ENV_PREFIX}" \
        python=3.12 \
        pip
fi

conda activate "${CONDA_ENV_PREFIX}"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu129

python -m pip install \
    scikit-build-core \
    cmake \
    ninja \
    pytest \
    scipy

python -m pip install \
    mace-torch==0.3.16 \
    cuequivariance==0.9.1 \
    cuequivariance-torch==0.9.1 \
    cuequivariance-ops-cu12==0.9.1 \
    cuequivariance-ops-torch-cu12==0.9.1

python - <<'PY'
from importlib.metadata import version
import torch

print("Python/Torch environment:")
print("  Torch:", torch.__version__)
print("  Torch CUDA:", torch.version.cuda)
print("  Visible GPUs:", torch.cuda.device_count())
print("  MACE:", version("mace-torch"))
print("  CuEq:", version("cuequivariance"))

assert torch.__version__ == "2.8.0+cu129"
assert torch.version.cuda == "12.9"
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 2
assert version("mace-torch") == "0.3.16"
assert version("cuequivariance") == "0.9.1"
PY

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export CUDAToolkit_ROOT="${CUDA_HOME}"
export TORCH_CUDA_ARCH_LIST="8.0"
export CMAKE_PREFIX_PATH="$(
    python -c 'import torch; print(torch.utils.cmake_prefix_path)'
)"
export CMAKE_GENERATOR="Ninja"
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK}"
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"
export CMAKE_ARGS="-DPython_EXECUTABLE=$(command -v python)"

rm -rf -- \
    "${PROJECT_ROOT}/dynamis_state/skbuild" \
    "${PROJECT_ROOT}/dynamis_model/skbuild" \
    "${PROJECT_ROOT}/dynamis_integrator/build" \
    "${PROJECT_ROOT}/dynamis/skbuild"

python -m pip install \
    --no-build-isolation \
    --no-deps \
    --force-reinstall \
    -e ./dynamis_state

python -m pip install \
    --no-build-isolation \
    --no-deps \
    --force-reinstall \
    -e ./dynamis_model

python -m pip install \
    --no-build-isolation \
    --no-deps \
    --force-reinstall \
    -e ./dynamis_integrator

python -m pip install \
    --no-build-isolation \
    --no-deps \
    --force-reinstall \
    -e ./dynamis

python -m pip check

python - <<'PY'
import dynamis
import dynamis_integrator.dynamis_integrator_impl
import dynamis_model.dynamis_model_impl
import dynamis_state.dynamis_state_impl

print("All Dynamis packages and native extensions imported successfully.")
PY

torchrun \
    --standalone \
    --nproc_per_node=2 \
    -m pytest \
    dynamis/tests/test_simulation_distributed.py \
    -v

RUN_DIR="${SLURM_TMPDIR:-/tmp}/dynamis_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

torchrun \
    --standalone \
    --nproc_per_node=2 \
    "${PROJECT_ROOT}/dynamis/experiments/synthetic/main.py"

echo "Dynamis installation and validation succeeded."
echo "Environment: ${CONDA_ENV_PREFIX}"
echo "Commit: $(git -C "${PROJECT_ROOT}" rev-parse HEAD)"