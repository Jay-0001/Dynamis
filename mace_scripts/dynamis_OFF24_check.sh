#!/bin/bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:25:00
#SBATCH --job-name=mace-off24-smoke
#SBATCH --output=/u/jsaravananindira/Dynamis/mace_scripts/outputs/mace_off24_smoke_%j.out

set -euo pipefail

PROJECT_ROOT="/u/jsaravananindira/Dynamis"
CONDA_ENV_PREFIX="/u/jsaravananindira/.conda/envs/dynamis-py312-cu129-clean"
RESULT_DIR="${PROJECT_ROOT}/mace_results/off24_smoke_${SLURM_JOB_ID}"
OFF24_MODEL="https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off24/MACE-OFF24_medium.model"

module purge
module load miniforge3-python gcc-native/14 cuda/12.9 slurm-env/0.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"

unset PYTHONPATH PYTHONHOME TORCHDYNAMO_VERBOSE TORCH_LOGS TORCH_TRACE
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"

cd "${PROJECT_ROOT}"
mkdir -p "${RESULT_DIR}"

python - <<'PY'
from importlib.metadata import version
import torch

assert torch.cuda.device_count() >= 2, "This smoke test requires two visible GPUs"
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Visible GPUs:", torch.cuda.device_count())
print("MACE:", version("mace-torch"))
PY

echo "Inspecting OFF24 checkpoint contract"
python - <<'PY'
from dynamis_model.nnp.mace import MACEModel, load_raw_mace_model

url = "https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off24/MACE-OFF24_medium.model"
model = load_raw_mace_model("off", url, device="cuda")
MACEModel._validate_model_contract(model)

print("Class:", type(model).__name__)
print("Cutoff (Å):", float(model.r_max))
print("Interactions:", len(model.interactions))
print("Products:", len(model.products))
print("Readouts:", len(model.readouts))
print("Heads:", getattr(model, "heads", None))
print("Joint embedding:", getattr(model, "joint_embedding", None) is not None)
print("Embedding readout:", getattr(model, "embedding_readout", None) is not None)
PY

run_backend() {
    local backend="$1"
    local prefix="${RESULT_DIR}/${backend}"
    local -a backend_args=()

    if [[ "${backend}" == "cueq" ]]; then
        backend_args+=(--enable-cueq)
    fi

    echo "Generating ${backend} raw-core reference"
    python dynamis_model/tests/mace/mace_standalone_reference.py \
        --model-family off --model "${OFF24_MODEL}" \
        --system-seed 1234 --molecule-count 1 \
        "${backend_args[@]}" \
        --output "${prefix}_reference.pt" \
        --output-json "${prefix}_reference.json"

    echo "Checking ${backend} two-rank parity"
    torchrun \
        --nnodes=1 \
        --node-rank=0 \
        --nproc-per-node=2 \
        --master-addr="${MASTER_ADDR}" \
        --master-port="${MASTER_PORT}" \
        dynamis_model/tests/mace/test_mace_model.py \
        run-distributed \
        --model-family off --model "${OFF24_MODEL}" \
        --system-seed 1234 --molecule-count 1 \
        "${backend_args[@]}" \
        --reference-artifact "${prefix}_reference.pt" \
        --strict-parity \
        --graph-edge-test \
        --strict-graph-interior \
        --output-json "${prefix}_distributed.json"
}

run_backend e3nn
run_backend cueq

echo "OFF24 smoke results: ${RESULT_DIR}"