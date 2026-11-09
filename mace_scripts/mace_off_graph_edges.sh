#!/bin/bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --job-name=mace-off-graph-audit
#SBATCH --output=/u/jsaravananindira/Dynamis/mace_scripts/outputs/mace_off_graph_%j.out

set -euo pipefail

PROJECT_ROOT="/u/jsaravananindira/Dynamis"
CONDA_ENV_PREFIX="/u/jsaravananindira/.conda/envs/dynamis-py312-cu129-clean"
RESULT_DIR="${PROJECT_ROOT}/mace_results/off_graph_${SLURM_JOB_ID}"

module purge
module load miniforge3-python gcc-native/14 cuda/12.9 slurm-env/0.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1

cd "${PROJECT_ROOT}"
mkdir -p "${RESULT_DIR}"

BOUNDARY_XYZ="${PROJECT_ROOT}/mace_data/off_cutoff_boundary_shifted.xyz"
ORGANIC_XYZ="${PROJECT_ROOT}/mace_data/off_organic_irregular.xyz"
JITTERED_XYZ="${RESULT_DIR}/jittered_15000.xyz"

python dynamis_model/tests/mace/jitter_mace_xyz.py \
    --input "${PROJECT_ROOT}/mace_data/benchmark_scaling_molecular_15000.xyz" \
    --output "${JITTERED_XYZ}" --seed 2026 --amplitude 0.01

run_case() {
    local case_name="$1"
    local xyz_path="$2"
    local model_name="$3"
    local backend="$4"
    local prefix="${RESULT_DIR}/${case_name}_${model_name}_${backend}"
    local -a backend_args=()

    if [[ "${backend}" == "cueq" ]]; then
        backend_args+=(--enable-cueq)
    fi

    python dynamis_model/tests/mace/mace_standalone_reference.py \
        --xyz-path "${xyz_path}" \
        --model-family off --model "${model_name}" \
        "${backend_args[@]}" \
        --output "${prefix}_reference.pt" \
        --output-json "${prefix}_reference.json"

    torchrun --standalone --nproc-per-node=2 \
        dynamis_model/tests/mace/test_mace_model.py run-distributed \
        --xyz-path "${xyz_path}" \
        --model-family off --model "${model_name}" \
        "${backend_args[@]}" \
        --energy-mode stable \
        --strict-graph-interior \
        --reference-artifact "${prefix}_reference.pt" \
        --output-json "${prefix}_distributed.json"
}

for backend in e3nn cueq; do
    run_case boundary "${BOUNDARY_XYZ}" medium "${backend}"
    run_case jittered "${JITTERED_XYZ}" medium "${backend}"
    for model in small medium large; do
        run_case organic "${ORGANIC_XYZ}" "${model}" "${backend}"
    done
done

echo "Graph audit results: ${RESULT_DIR}"