#!/bin/bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --job-name=mace-off24-md
#SBATCH --output=/u/jsaravananindira/Dynamis/mace_scripts/outputs/mace_off24_md_%j.out

set -euo pipefail

PROJECT_ROOT="/u/jsaravananindira/Dynamis"
CONDA_ENV_PREFIX="/u/jsaravananindira/.conda/envs/dynamis-py312-cu129-clean"
MODEL="https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off24/MACE-OFF24_medium.model"
RESULT_DIR="${PROJECT_ROOT}/mace_results/off24_md_${SLURM_JOB_ID}"

module purge
module load miniforge3-python gcc-native/14 cuda/12.9 slurm-env/0.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"
unset PYTHONPATH PYTHONHOME TORCHDYNAMO_VERBOSE TORCH_LOGS TORCH_TRACE
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"

cd "${PROJECT_ROOT}"
mkdir -p "${RESULT_DIR}"
python - "${MODEL}" <<'PY'
import sys
import torch
from dynamis_model.nnp.mace import MACEModel, load_raw_mace_model

assert torch.cuda.device_count() >= 2, "Two CUDA devices are required"
model = load_raw_mace_model("off", sys.argv[1], device="cuda")
MACEModel._validate_model_contract(model)
assert float(model.r_max) == 6.0
print("OFF24 checkpoint cached before distributed ranks start")
PY

for backend in e3nn cueq; do
    backend_args=()
    if [[ "${backend}" == cueq ]]; then
        backend_args+=(--enable-cueq)
    fi
    path="${RESULT_DIR}/${backend}"
    echo "OFF24 ${backend}: two-rank 20-step trajectory"
    torchrun --nnodes=1 --node-rank=0 --nproc-per-node=2 \
        --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
        dynamis_model/tests/mace/run_distributed_trajectory.py \
        --path "${path}" --system off24 --model-family off --model "${MODEL}" \
        "${backend_args[@]}"

    echo "OFF24 ${backend}: independent raw-core trajectory comparison"
    python dynamis_model/tests/mace/test_md.py \
        --path "${path}" --system off24 --model-family off --model "${MODEL}" \
        "${backend_args[@]}"
done

echo "OFF24 20-step trajectory parity passed: ${RESULT_DIR}"
