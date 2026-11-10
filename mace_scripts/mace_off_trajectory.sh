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
#SBATCH --job-name=mace-off-trajectory
#SBATCH --output=/u/jsaravananindira/Dynamis/mace_scripts/outputs/mace_off_trajectory_%j.out

set -euo pipefail

PROJECT_ROOT="/u/jsaravananindira/Dynamis"
CONDA_ENV_PREFIX="/u/jsaravananindira/.conda/envs/dynamis-py312-cu129-clean"
RESULT_DIR="${PROJECT_ROOT}/mace_results/off_trajectory_${SLURM_JOB_ID}"

read -r -a MODELS <<< "${OFF_MODELS:-small medium large}"
read -r -a BACKENDS <<< "${BACKENDS:-e3nn cueq}"

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
assert torch.cuda.device_count() >= 2, "The trajectory gate requires two visible GPUs."
print(f"Visible GPUs: {torch.cuda.device_count()}")
PY

echo "OFF trajectory gate"
echo "Commit: $(git rev-parse HEAD)"
echo "Results: ${RESULT_DIR}"

failed_cases=()
passed_cases=()

run_case() {
    local model="$1"
    local backend="$2"
    local case_name="${model}_${backend}"
    local case_dir="${RESULT_DIR}/${case_name}"
    local -a backend_args=()

    case "${backend}" in
        e3nn) ;;
        cueq) backend_args+=(--enable-cueq) ;;
        *)
            failed_cases+=("${case_name}: unknown backend")
            return
            ;;
    esac

    mkdir -p "${case_dir}"
    echo "===== ${case_name}: two-rank Dynamis trajectory ====="

    if ! torchrun --standalone --nproc-per-node=2 \
        dynamis_model/tests/mace/run_distributed_trajectory.py \
        --path "${case_dir}" \
        --system organic \
        --model-family off \
        --model "${model}" \
        "${backend_args[@]}"; then
        failed_cases+=("${case_name}: distributed trajectory")
        return
    fi

    echo "===== ${case_name}: public calculator comparison ====="

    if ! python dynamis_model/tests/mace/test_md.py \
        --path "${case_dir}" \
        --system organic \
        --model-family off \
        --model "${model}" \
        --reference-mode monolithic \
        "${backend_args[@]}"; then
        failed_cases+=("${case_name}: trajectory comparison")
        return
    fi

    passed_cases+=("${case_name}")
    echo "PASS ${case_name}: ${case_dir}/ref_out/comparison_report.csv"
}

for backend in "${BACKENDS[@]}"; do
    for model in "${MODELS[@]}"; do
        run_case "${model}" "${backend}"
    done
done

printf '\nPassed cases (%d):\n' "${#passed_cases[@]}"
printf '  %s\n' "${passed_cases[@]}"

if (( ${#failed_cases[@]} > 0 )); then
    printf 'Failed cases (%d):\n' "${#failed_cases[@]}"
    printf '  %s\n' "${failed_cases[@]}"
    exit 1
fi

echo "OFF trajectory gate passed: ${RESULT_DIR}"
