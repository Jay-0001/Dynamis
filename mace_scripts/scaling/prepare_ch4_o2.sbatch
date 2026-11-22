#!/usr/bin/env bash
# Prepare the shared XYZ on a permitted Beta allocation; GPUs are idle here.
#SBATCH --job-name=mace-off-prepare
#SBATCH --partition=beta
#SBATCH --account=su_kchiu_molecular
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=7
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=mace_scripts/output/mace-off-prepare-%j.out
#SBATCH --error=mace_scripts/output/mace-off-prepare-%j.err

set -euo pipefail
project_root="${SLURM_SUBMIT_DIR:?Submit from the Dynamis repository root}"
test -f "$project_root/mace_scripts/scaling/prepare_ch4_o2.py" || {
    echo "Submit from the Dynamis repository root." >&2
    exit 1
}

srun --ntasks=1 --cpus-per-task=7 \
    --container-image=nvcr.io#nvidia/pytorch:25.08-py3 \
    --container-mounts="$project_root:/workspace/Dynamis" \
    --container-workdir=/workspace/Dynamis \
    bash -lc '
        set -euo pipefail
        cd /workspace/Dynamis
        cache_root=/workspace/Dynamis/mace_scripts/output/cache
        mkdir -p "$cache_root/home" "$cache_root/xdg"
        export HOME="$cache_root/home" XDG_CACHE_HOME="$cache_root/xdg"
        export PYTHONNOUSERSITE=1
        unset PYTHONPATH PYTHONHOME
        source .venv/bin/activate
        python mace_scripts/scaling/prepare_ch4_o2.py --size-k "${SIZE_K:-45}"
    '
