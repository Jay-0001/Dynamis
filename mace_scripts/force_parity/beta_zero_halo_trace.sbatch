#!/usr/bin/env bash
#SBATCH --job-name=beta-mace-trace
#SBATCH --partition=beta
#SBATCH --account=su_kchiu_molecular
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=mace_scripts/output/beta-mace-trace-%j.out
#SBATCH --error=mace_scripts/output/beta-mace-trace-%j.err
set -euo pipefail
project_root="${SLURM_SUBMIT_DIR:?Submit from the Dynamis root}"
srun --ntasks=1 --cpus-per-task=7 \
    --container-image=nvcr.io#nvidia/pytorch:25.08-py3 \
    --container-mounts="$project_root:/workspace/Dynamis" \
    --container-workdir=/workspace/Dynamis \
    bash /workspace/Dynamis/mace_scripts/force_parity/beta_zero_halo_trace_container.sh
