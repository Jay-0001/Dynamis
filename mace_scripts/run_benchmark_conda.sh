#!/bin/bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:10:00
#SBATCH --job-name=dynamis-mace-cueq-2g
#SBATCH --output=/u/jsaravananindira/Dynamis/mace_scripts/outputs/dynamis_mace_off_legacy_%j.out

set -euo pipefail

PROJECT_ROOT="/u/jsaravananindira/Dynamis"
CONDA_ENV_PREFIX="/u/jsaravananindira/.conda/envs/dynamis-py312-cu129-clean"
RESULT_DIR="${PROJECT_ROOT}/mace_results/dynamis_mace_legacy_2g_${SLURM_JOB_ID}"

MODEL_FAMILY="${MODEL_FAMILY:-off}"
MODEL_NAME="${MODEL_NAME:-medium}"
XYZ_PATH="${XYZ_PATH:-${PROJECT_ROOT}/mace_data/benchmark_scaling_molecular_15000.xyz}"
HEAD_NAME="${HEAD_NAME:-}"

module purge
module load miniforge3-python
module load gcc-native/14
module load cuda/12.9
module load slurm-env/0.1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"

unset PYTHONPATH PYTHONHOME
unset TORCHDYNAMO_VERBOSE TORCH_LOGS TORCH_TRACE

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"

cd "${PROJECT_ROOT}"

test -f "${XYZ_PATH}"
mkdir -p "${RESULT_DIR}"

REFERENCE_ARTIFACT="${RESULT_DIR}/reference_cueq.pt"
REFERENCE_JSON="${RESULT_DIR}/reference_cueq.json"
SUMMARY_JSON="${RESULT_DIR}/distributed_cueq_2gpu_15000.json"

MODEL_ARGS=(
    --model-family "${MODEL_FAMILY}"
    --model "${MODEL_NAME}"
)

CUEQ_ARGS=(
    --enable-cueq
)

if [[ -n "${HEAD_NAME}" ]]; then
    MODEL_ARGS+=(--head-name "${HEAD_NAME}")
fi

echo "============================================================"
echo "Dynamis MACE/CuEq two-GPU parity benchmark"
echo "Job: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Commit: $(git rev-parse HEAD)"
echo "Environment: ${CONDA_PREFIX}"
echo "Model: ${MODEL_FAMILY}/${MODEL_NAME}"
echo "Structure: ${XYZ_PATH}"
echo "Results: ${RESULT_DIR}"
echo "============================================================"

python - <<'PY'
from importlib.metadata import version

import dynamis
import dynamis_integrator
import dynamis_model
import dynamis_state
import torch

print("Dynamis:", dynamis.__file__)
print("dynamis_state:", dynamis_state.__file__)
print("dynamis_model:", dynamis_model.__file__)
print("dynamis_integrator:", dynamis_integrator.__file__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("Visible GPUs:", torch.cuda.device_count())
print("MACE:", version("mace-torch"))
print("CuEq:", version("cuequivariance"))
print("CuEq Torch:", version("cuequivariance-torch"))
print("CuEq CUDA ops:", version("cuequivariance-ops-cu12"))
print("CuEq Torch ops:", version("cuequivariance-ops-torch-cu12"))

assert torch.__version__ == "2.8.0+cu129"
assert torch.version.cuda == "12.9"
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 2
assert version("mace-torch") == "0.3.16"
assert version("cuequivariance") == "0.9.1"
PY

echo "GPU topology:"
nvidia-smi topo -m

echo "Generating standalone CuEq MACE reference"

python dynamis_model/tests/mace/mace_standalone_reference.py \
    --xyz-path "${XYZ_PATH}" \
    "${MODEL_ARGS[@]}" \
    "${CUEQ_ARGS[@]}" \
    --output "${REFERENCE_ARTIFACT}" \
    --benchmark \
    --benchmark-warmup 10 \
    --benchmark-iterations 50 \
    --benchmark-samples 7 \
    --output-json "${REFERENCE_JSON}"

echo "Running two-rank Dynamis MACE/CuEq benchmark"


torchrun \
    --nnodes=1 \
    --node-rank=0 \
    --nproc-per-node=2 \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    dynamis_model/tests/mace/test_mace_model.py \
    run-distributed \
    --xyz-path "${XYZ_PATH}" \
    "${MODEL_ARGS[@]}" \
    "${CUEQ_ARGS[@]}" \
    --reference-artifact "${REFERENCE_ARTIFACT}" \
    --energy-mode legacy \
    --output-json "${RESULT_DIR}/legacy.json"

echo "============================================================"
echo "Dynamis MACE/CuEq benchmark completed successfully"
echo "Reference artifact: ${REFERENCE_ARTIFACT}"
echo "Reference metrics:  ${REFERENCE_JSON}"
echo "Distributed metrics: ${SUMMARY_JSON}"
echo "============================================================"