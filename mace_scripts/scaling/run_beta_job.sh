#!/usr/bin/env bash
# Host-side Slurm step launcher shared by the 1-, 2-, and 4-worker jobs.
set -euo pipefail

workers="${1:?Pass the active GPU worker count (1, 2, or 4)}"
case "$workers" in
    1|2|4) ;;
    *) echo "Unsupported worker count: $workers" >&2; exit 2 ;;
esac

project_root="${SLURM_SUBMIT_DIR:?Submit from the Dynamis repository root}"
test -f "$project_root/mace_scripts/scaling/run_mace_off.py" || {
    echo "Submit from the Dynamis repository root." >&2
    exit 1
}

srun --ntasks=1 --cpus-per-task=7 \
    --container-image=nvcr.io#nvidia/pytorch:25.08-py3 \
    --container-mounts="$project_root:/workspace/Dynamis" \
    --container-workdir=/workspace/Dynamis \
    bash /workspace/Dynamis/mace_scripts/scaling/run_beta_container.sh "$workers"
