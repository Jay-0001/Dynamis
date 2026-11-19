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
#SBATCH --job-name=mace-off24-static
#SBATCH --output=/u/jsaravananindira/Dynamis/mace_scripts/outputs/mace_off24_static_%j.out

set -euo pipefail

PROJECT_ROOT="/u/jsaravananindira/Dynamis"
CONDA_ENV_PREFIX="/u/jsaravananindira/.conda/envs/dynamis-py312-cu129-clean"
MODEL="https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off24/MACE-OFF24_medium.model"
RESULT_DIR="${PROJECT_ROOT}/mace_results/off24_static_${SLURM_JOB_ID}"

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
SMALL_XYZ="${PROJECT_ROOT}/mace_data/off24_bridge.xyz"
LARGE_XYZ="${RESULT_DIR}/off24_bridge_x50.xyz"
python dynamis_model/tests/mace/repeat_off24_xyz.py --copies 50 --output "${LARGE_XYZ}"

python - "${MODEL}" <<'PY'
import sys
import torch
from dynamis_model.nnp.mace import MACEModel, MACEScaleShiftRuntime, load_raw_mace_model

assert torch.cuda.device_count() >= 2, "Two CUDA devices are required"
model = load_raw_mace_model("off", sys.argv[1], device="cuda")
MACEModel._validate_model_contract(model)
runtime = MACEScaleShiftRuntime(model)
assert float(model.r_max) == 6.0
assert len(model.interactions) == len(model.products) == len(model.readouts) == 2
assert runtime._should_exchange_after_product(0)
assert not runtime._should_exchange_after_product(1)
assert getattr(model, "joint_embedding", None) is None
print("OFF24 checkpoint contract: ordinary ScaleShiftMACE, 6 Å, one feature exchange")
PY

run_case() {
    local case_name="$1" xyz_path="$2" backend="$3"
    local prefix="${RESULT_DIR}/${case_name}_${backend}"
    local -a backend_args=() benchmark_args=()
    if [[ "${backend}" == cueq ]]; then
        backend_args+=(--enable-cueq)
    fi
    if [[ "${case_name}" == large ]]; then
        benchmark_args+=(--benchmark --benchmark-warmup 3 --benchmark-iterations 10 --benchmark-samples 5)
    fi

    echo "OFF24 ${case_name}/${backend}: native raw-core reference"
    python dynamis_model/tests/mace/mace_standalone_reference.py \
        --xyz-path "${xyz_path}" --model-family off --model "${MODEL}" \
        "${backend_args[@]}" "${benchmark_args[@]}" \
        --output "${prefix}_reference.pt" --output-json "${prefix}_reference.json"

    echo "OFF24 ${case_name}/${backend}: two-rank adapter parity"
    torchrun --nnodes=1 --node-rank=0 --nproc-per-node=2 \
        --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
        dynamis_model/tests/mace/test_mace_model.py run-distributed \
        --xyz-path "${xyz_path}" --model-family off --model "${MODEL}" \
        "${backend_args[@]}" "${benchmark_args[@]}" \
        --reference-artifact "${prefix}_reference.pt" \
        --strict-parity --graph-edge-test --strict-graph-interior \
        --output-json "${prefix}_distributed.json"
}

for backend in e3nn cueq; do
    run_case small "${SMALL_XYZ}" "${backend}"
    run_case large "${LARGE_XYZ}" "${backend}"
done

python - "${RESULT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted(root.glob("*_distributed.json"))
assert len(files) == 4, f"Expected four parity results, found {len(files)}"
for path in files:
    result = json.loads(path.read_text())
    graph = result["graph"]
    assert result["energy"]["close"] and result["forces"]["close"]
    assert graph["assembled_directed_edges"] == graph["native_directed_edges"]
    assert all(graph[key] == 0 for key in (
        "interior_extra", "interior_missing", "boundary_extra", "boundary_missing", "exterior_extra"
    )), f"Graph mismatch in {path}"
    assert all(rank["owned_atoms"] > 0 for rank in result["ranks"])
    if "small" in path.stem:
        assert all(rank["halo_atoms"] > 0 for rank in result["ranks"])
    print(path.name, "energy error (MD):", result["energy"]["error_md"],
          "force MAE (MD):", result["forces"]["mae_md"],
          "edges:", graph["assembled_directed_edges"])
print("OFF24 static parity passed:", root)
PY
