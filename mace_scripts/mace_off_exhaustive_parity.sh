#!/bin/bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:25:00
#SBATCH --job-name=dynamis-mace-off-parity
#SBATCH --output=/u/jsaravananindira/Dynamis/mace_scripts/outputs/dynamis_mace_off_parity_%j.out

set -euo pipefail

PROJECT_ROOT="/u/jsaravananindira/Dynamis"
CONDA_ENV_PREFIX="/u/jsaravananindira/.conda/envs/dynamis-py312-cu129-clean"
RESULT_DIR="${PROJECT_ROOT}/mace_results/dynamis_mace_off_parity_${SLURM_JOB_ID}"

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

mkdir -p "${RESULT_DIR}/small" "${RESULT_DIR}/large"

MODEL_ARGS=(
    --model-family "${MODEL_FAMILY}"
    --model "${MODEL_NAME}"
)
CUEQ_ARGS=(--enable-cueq)

if [[ -n "${HEAD_NAME}" ]]; then
    MODEL_ARGS+=(--head-name "${HEAD_NAME}")
fi

echo "============================================================"
echo "Dynamis MACE/CuEq exhaustive parity check"
echo "Job: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Commit: $(git rev-parse HEAD)"
echo "Environment: ${CONDA_PREFIX}"
echo "Model: ${MODEL_FAMILY}/${MODEL_NAME}"
echo "Large structure: ${XYZ_PATH}"
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
assert torch.cuda.device_count() >= 2
assert version("mace-torch") == "0.3.16"
assert version("cuequivariance") == "0.9.1"
PY

echo "GPU topology:"
nvidia-smi topo -m

# The float64 diagnostic uses a small structure. Its float64 calculator/core
# evaluations use e3nn; the ordinary float32 reference still uses CuEq.
echo "Generating small OFF calculator/core float64 diagnostic"
python dynamis_model/tests/mace/mace_standalone_reference.py \
    --builtin-system natural_chain12 \
    "${MODEL_ARGS[@]}" \
    "${CUEQ_ARGS[@]}" \
    --float64-diagnostic \
    --repeatability-runs 5 \
    --output "${RESULT_DIR}/small/reference.pt" \
    --output-json "${RESULT_DIR}/small/reference.json"

echo "Checking small-system stable distributed parity"
torchrun \
    --nnodes=1 \
    --node-rank=0 \
    --nproc-per-node=2 \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    dynamis_model/tests/mace/test_mace_model.py \
    run-distributed \
    --builtin-system natural_chain12 \
    "${MODEL_ARGS[@]}" \
    "${CUEQ_ARGS[@]}" \
    --energy-mode stable \
    --reference-artifact "${RESULT_DIR}/small/reference.pt" \
    --strict-parity \
    --output-json "${RESULT_DIR}/small/distributed_stable.json"

LARGE_REFERENCE="${RESULT_DIR}/large/reference.pt"

echo "Generating 15k-atom calculator/core reference and repeatability diagnostics"
python dynamis_model/tests/mace/mace_standalone_reference.py \
    --xyz-path "${XYZ_PATH}" \
    "${MODEL_ARGS[@]}" \
    "${CUEQ_ARGS[@]}" \
    --repeatability-runs 5 \
    --output "${LARGE_REFERENCE}" \
    --benchmark \
    --benchmark-warmup 10 \
    --benchmark-iterations 50 \
    --benchmark-samples 7 \
    --output-json "${RESULT_DIR}/large/reference.json"

for ENERGY_MODE in legacy stable; do
    echo "Running two-rank ${ENERGY_MODE} distributed parity and benchmark"

    torchrun \
        --nnodes=1 \
        --node-rank=0 \
        --nproc-per-node=2 \
        --master-addr="${MASTER_ADDR}" \
        --master-port="$((MASTER_PORT + 1))" \
        dynamis_model/tests/mace/test_mace_model.py \
        run-distributed \
        --xyz-path "${XYZ_PATH}" \
        "${MODEL_ARGS[@]}" \
        "${CUEQ_ARGS[@]}" \
        --reference-artifact "${LARGE_REFERENCE}" \
        --energy-mode "${ENERGY_MODE}" \
        --benchmark \
        --benchmark-warmup 10 \
        --benchmark-iterations 50 \
        --benchmark-samples 7 \
        --output-json "${RESULT_DIR}/large/distributed_${ENERGY_MODE}.json"
done

echo "============================================================"
echo "Key 15k-atom comparisons (Dynamis energy units)"
echo "============================================================"

RESULT_DIR="${RESULT_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RESULT_DIR"]) / "large"

for mode in ("legacy", "stable"):
    with (root / f"distributed_{mode}.json").open() as file:
        summary = json.load(file)

    parity = summary["parity"]
    print(f"\n{mode.upper()} — {summary['mismatch_classification']}")
    for key in (
        "distributed_energy",
        "calculator_mace_energy",
        "calculator_controlled_per_atom_energy",
        "core_model_mace_energy",
        "core_model_stable_energy",
        "explicit_dynamis_graph_energy",
        "explicit_stable_energy",
        "distributed_minus_calculator_energy",
        "distributed_minus_calculator_controlled_energy",
        "distributed_minus_core_model_stable_energy",
        "distributed_minus_explicit_stable_energy",
        "calculator_legacy_minus_controlled_energy",
        "explicit_legacy_minus_stable_energy",
        "explicit_vs_core_node_energy_mae",
        "calculator_force_mae",
    ):
        if key in parity:
            print(f"{key}: {parity[key]}")

    diagnostics = summary.get("standalone_diagnostics", {})
    print("calculator/core comparison (eV):", diagnostics.get("calculator_core"))
    print("calculator repeatability:", diagnostics.get("calculator_repeatability"))
PY

echo "============================================================"
echo "Exhaustive parity check completed"
echo "Small diagnostic: ${RESULT_DIR}/small/"
echo "Large reference: ${RESULT_DIR}/large/reference.json"
echo "Legacy report:   ${RESULT_DIR}/large/distributed_legacy.json"
echo "Stable report:   ${RESULT_DIR}/large/distributed_stable.json"
echo "============================================================"