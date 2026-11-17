#!/bin/bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:15:00
#SBATCH --job-name=mace-mpa-trajectory
#SBATCH --output=/u/jsaravananindira/Dynamis/mace_scripts/outputs/new_mace_mpa_trajectory_%j.out

set -euo pipefail

PROJECT_ROOT="/u/jsaravananindira/Dynamis"
CONDA_ENV_PREFIX="/u/jsaravananindira/.conda/envs/dynamis-py312-cu129-clean"
RESULT_DIR="${PROJECT_ROOT}/mace_results/mpa_trajectory_${SLURM_JOB_ID}"
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
PY

MODEL_ARGS=(
    --system organic
    --system-seed "${SYSTEM_SEED}"
    --model-family mp
    --model medium-mpa-0
    --enable-cueq
)

torchrun --standalone --nproc-per-node=2 \
    dynamis_model/tests/mace/run_distributed_trajectory.py \
    --path "${RESULT_DIR}" \
    "${MODEL_ARGS[@]}"

python dynamis_model/tests/mace/test_md.py \
    --path "${RESULT_DIR}" \
    "${MODEL_ARGS[@]}"

echo "Trajectory results: ${RESULT_DIR}"