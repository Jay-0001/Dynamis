#!/usr/bin/env bash
# Static OFF24 reference versus four-rank adapter parity on EmpireAI Beta.
#SBATCH --job-name=mace-off24-parity
#SBATCH --partition=beta
#SBATCH --account=su_kchiu_molecular
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=mace_scripts/output/mace-off24-parity-%j.out
#SBATCH --error=mace_scripts/output/mace-off24-parity-%j.err

set -euo pipefail
project_root="${SLURM_SUBMIT_DIR:?Submit from the Dynamis root}"
srun --ntasks=1 --cpus-per-task=7 \
    --container-image=nvcr.io#nvidia/pytorch:25.08-py3 \
    --container-mounts="$project_root:/workspace/Dynamis" \
    --container-workdir=/workspace/Dynamis \
    bash /workspace/Dynamis/mace_scripts/scaling/check_off24_parity_container.sh
