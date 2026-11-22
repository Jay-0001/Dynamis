#!/usr/bin/env bash
# Two active MPA workers on a four-GPU Beta allocation.
#SBATCH --job-name=mace-mpa-2worker
#SBATCH --partition=beta
#SBATCH --account=su_kchiu_molecular
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=7
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=mace_scripts/output/mace-mpa-2worker-%j.out
#SBATCH --error=mace_scripts/output/mace-mpa-2worker-%j.err
set -euo pipefail
export MACE_MODEL_FAMILY=mp
bash "${SLURM_SUBMIT_DIR:?}/mace_scripts/scaling/run_beta_job.sh" 2
