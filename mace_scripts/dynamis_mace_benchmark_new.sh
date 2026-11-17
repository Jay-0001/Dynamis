#!/bin/bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:20:00
#SBATCH --job-name=mace-mpa-benchmark
#SBATCH --output=/u/jsaravananindira/Dynamis/mace_scripts/outputs/new_mace_mpa_benchmark_%j.out

set -euo pipefail

PROJECT_ROOT="/u/jsaravananindira/Dynamis"
CONDA_ENV_PREFIX="/u/jsaravananindira/.conda/envs/dynamis-py312-cu129-clean"
RESULT_DIR="${PROJECT_ROOT}/mace_results/mpa_benchmark_${SLURM_JOB_ID}"
SYSTEM_SEED=1234

module purge
module load miniforge3-python gcc-native/14 cuda/12.9 slurm-env/0.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1

cd "${PROJECT_ROOT}"
mkdir -p "${RESULT_DIR}"

python - <<'PY'
import torch
assert torch.cuda.device_count() >= 2, "This test requires two visible CUDA devices."
print("Visible GPUs:", torch.cuda.device_count())
PY

MODEL_ARGS=(--model-family mp --model medium-mpa-0 --enable-cueq)

# Small cross-rank feature-exchange and graph check.
python dynamis_model/tests/mace/mace_standalone_reference.py \
    --system-seed "${SYSTEM_SEED}" --molecule-count 1 \
    "${MODEL_ARGS[@]}" \
    --output "${RESULT_DIR}/halo_reference.pt" \
    --output-json "${RESULT_DIR}/halo_reference.json"

torchrun --standalone --nproc-per-node=2 \
    dynamis_model/tests/mace/test_mace_model.py run-distributed \
    --system-seed "${SYSTEM_SEED}" --molecule-count 1 \
    "${MODEL_ARGS[@]}" \
    --reference-artifact "${RESULT_DIR}/halo_reference.pt" \
    --strict-parity --strict-graph-interior \
    --output-json "${RESULT_DIR}/halo_distributed.json"

# Larger deterministic molecular benchmark: 625 × 12 = 7,500 atoms.
python dynamis_model/tests/mace/mace_standalone_reference.py \
    --system-seed "${SYSTEM_SEED}" --molecule-count 625 \
    "${MODEL_ARGS[@]}" \
    --output "${RESULT_DIR}/benchmark_reference.pt" \
    --output-json "${RESULT_DIR}/benchmark_reference.json" \
    --benchmark --benchmark-warmup 10 \
    --benchmark-iterations 50 --benchmark-samples 7

torchrun --standalone --nproc-per-node=2 \
    dynamis_model/tests/mace/test_mace_model.py run-distributed \
    --system-seed "${SYSTEM_SEED}" --molecule-count 625 \
    "${MODEL_ARGS[@]}" \
    --reference-artifact "${RESULT_DIR}/benchmark_reference.pt" \
    --strict-parity --strict-graph-interior \
    --benchmark --benchmark-warmup 10 \
    --benchmark-iterations 50 --benchmark-samples 7 \
    --output-json "${RESULT_DIR}/benchmark_distributed.json"

echo "Static results: ${RESULT_DIR}"