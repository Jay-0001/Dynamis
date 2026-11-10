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
#SBATCH --job-name=mace-off-graph-audit
#SBATCH --output=/u/jsaravananindira/Dynamis/mace_scripts/outputs/mace_off_graph_%j.out

set -euo pipefail

PROJECT_ROOT="/u/jsaravananindira/Dynamis"
CONDA_ENV_PREFIX="/u/jsaravananindira/.conda/envs/dynamis-py312-cu129-clean"
RESULT_DIR="${PROJECT_ROOT}/mace_results/off_graph_${SLURM_JOB_ID}"

read -r -a BACKENDS <<< "${BACKENDS:-e3nn cueq}"
read -r -a ORGANIC_MODELS <<< "${ORGANIC_MODELS:-small medium large}"
RUN_BOUNDARY="${RUN_BOUNDARY:-1}"
RUN_JITTERED="${RUN_JITTERED:-1}"
RUN_ORGANIC="${RUN_ORGANIC:-1}"

module purge
module load miniforge3-python gcc-native/14 cuda/12.9 slurm-env/0.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1

cd "${PROJECT_ROOT}"
mkdir -p "${RESULT_DIR}"

BOUNDARY_XYZ="${PROJECT_ROOT}/mace_data/off_cutoff_boundary.xyz"
ORGANIC_XYZ="${PROJECT_ROOT}/mace_data/off_organic_irregular.xyz"
JITTERED_XYZ="${RESULT_DIR}/jittered_15000.xyz"

input_files=()
if [[ "${RUN_BOUNDARY}" == 1 ]]; then
    input_files+=("${BOUNDARY_XYZ}")
fi
if [[ "${RUN_ORGANIC}" == 1 ]]; then
    input_files+=("${ORGANIC_XYZ}")
fi
if [[ "${RUN_JITTERED}" == 1 ]]; then
    python dynamis_model/tests/mace/jitter_mace_xyz.py \
        --input "${PROJECT_ROOT}/mace_data/benchmark_scaling_molecular_15000.xyz" \
        --output "${JITTERED_XYZ}" --seed 2026 --amplitude 0.01
    input_files+=("${JITTERED_XYZ}")
fi
if (( ${#input_files[@]} == 0 )); then
    echo "No graph-audit cases selected" >&2
    exit 2
fi

python - "${input_files[@]}" <<'PY'
import sys
from pathlib import Path
import torch

sys.path.insert(0, "dynamis_model/tests/mace")
from mace_standalone_reference import load_xyz

for name in sys.argv[1:]:
    raw = load_xyz(Path(name))
    if any(raw["pbc"]):
        raise ValueError(f"Graph-audit fixture must be nonperiodic: {name}")
    positions = torch.tensor(raw["positions"], dtype=torch.float32)
    cell = torch.tensor(raw["cell"], dtype=torch.float64)
    fractional = positions.double() @ torch.linalg.inv(cell)
    if (positions < 0).any() or (fractional < 0).any() or (fractional >= 1).any():
        raise ValueError(f"Atoms lie outside the nonperiodic cell: {name}")
    print(f"Validated {name}: {len(positions)} atoms inside nonperiodic cell")
PY

failed_cases=()

run_case() {
    local case_name="$1"
    local xyz_path="$2"
    local model_name="$3"
    local backend="$4"
    local prefix="${RESULT_DIR}/${case_name}_${model_name}_${backend}"
    local -a backend_args=()

    if [[ "${backend}" == "cueq" ]]; then
        backend_args+=(--enable-cueq)
    elif [[ "${backend}" != "e3nn" ]]; then
        failed_cases+=("${case_name}/${model_name}/${backend}: unknown backend")
        return 0
    fi

    echo "Running ${case_name}/${model_name}/${backend}"

    if ! python dynamis_model/tests/mace/mace_standalone_reference.py \
        --xyz-path "${xyz_path}" \
        --model-family off --model "${model_name}" \
        "${backend_args[@]}" \
        --output "${prefix}_reference.pt" \
        --output-json "${prefix}_reference.json"; then
        failed_cases+=("${case_name}/${model_name}/${backend}: standalone reference")
        return 0
    fi

    if ! torchrun --standalone --nproc-per-node=2 \
        dynamis_model/tests/mace/test_mace_model.py run-distributed \
        --xyz-path "${xyz_path}" \
        --model-family off --model "${model_name}" \
        "${backend_args[@]}" \
        --energy-mode stable --graph-edge-test \
        --reference-artifact "${prefix}_reference.pt" \
        --output-json "${prefix}_distributed.json"; then
        failed_cases+=("${case_name}/${model_name}/${backend}: distributed audit")
    fi
}

for backend in "${BACKENDS[@]}"; do
    if [[ "${RUN_BOUNDARY}" == 1 ]]; then
        run_case boundary "${BOUNDARY_XYZ}" medium "${backend}"
    fi
    if [[ "${RUN_JITTERED}" == 1 ]]; then
        run_case jittered "${JITTERED_XYZ}" medium "${backend}"
    fi
    if [[ "${RUN_ORGANIC}" == 1 ]]; then
        for model in "${ORGANIC_MODELS[@]}"; do
            run_case organic "${ORGANIC_XYZ}" "${model}" "${backend}"
        done
    fi
done

python - "${RESULT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("\nGraph-audit case summary (Dynamis MD units):")
for path in sorted(root.glob("*_distributed.json")):
    result = json.loads(path.read_text())
    parity = result["parity"]
    audit = result["owned_graph_audit"]
    interior = audit["assembled_vs_native_core_by_band"]["interior"]
    print(
        f"{path.stem}: {result['mismatch_classification']}; "
        f"interior missing={interior['core_only']['count']}, "
        f"extra={interior['explicit_only']['count']}; "
        f"stable energy gap={parity['distributed_minus_core_model_stable_energy']:.6g}; "
        f"force MAE={parity['core_model_force_mae']:.6g}"
    )
PY

if (( ${#failed_cases[@]} > 0 )); then
    printf 'Failed cases:\n'
    printf '  %s\n' "${failed_cases[@]}"
    exit 1
fi

echo "Graph audit completed: ${RESULT_DIR}"