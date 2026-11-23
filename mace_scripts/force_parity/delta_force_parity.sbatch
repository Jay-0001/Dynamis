#!/usr/bin/env bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:45:00
#SBATCH --job-name=delta-force-parity
#SBATCH --output=mace_scripts/outputs/delta-force-parity-%j.out
#SBATCH --error=mace_scripts/outputs/delta-force-parity-%j.err
set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:?Submit from the Dynamis root}"
module purge
module load miniforge3-python gcc-native/14 cuda/12.9 slurm-env/0.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${DELTA_CONDA_ENV:-/u/jsaravananindira/.conda/envs/dynamis-py312-cu129-clean}"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export FORCE_RANKS="${FORCE_RANKS:-2}"
export FORCE_OUTPUT_DIR="$project_root/mace_results/force_parity/delta_${SLURM_JOB_ID:?}"
cd "$project_root"
bash mace_scripts/force_parity/run_matrix.sh
