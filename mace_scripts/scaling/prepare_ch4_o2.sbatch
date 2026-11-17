#!/usr/bin/env bash
#SBATCH --job-name=mace-off-xyz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=mace_off_prepare_%j.out
set -euo pipefail

project_root="${DYNAMIS_FORK_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the fork root}}"
cd "$project_root"
python_bin="${PYTHON_BIN:-python}"
"$python_bin" mace_scripts/scaling/prepare_ch4_o2.py
