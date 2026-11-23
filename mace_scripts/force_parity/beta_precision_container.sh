#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Dynamis
cache_root=/workspace/Dynamis/mace_scripts/output/cache
mkdir -p "$cache_root"/{xdg,pip,mpl,triton,cueq,torch}
export XDG_CACHE_HOME="$cache_root/xdg" PIP_CACHE_DIR="$cache_root/pip"
export MPLCONFIGDIR="$cache_root/mpl" TRITON_CACHE_DIR="$cache_root/triton"
export CUEQ_TRITON_CACHE_DIR="$cache_root/cueq" TORCH_HOME="$cache_root/torch"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
source .venv/bin/activate
export FORCE_OUTPUT_DIR="/workspace/Dynamis/mace_results/force_parity/precision_beta_${SLURM_JOB_ID:?}"
export FORCE_MODEL_FAMILY="${FORCE_MODEL_FAMILY:-mp}"
if [[ "$FORCE_MODEL_FAMILY" == off ]]; then
    export FORCE_MODEL="${FORCE_MODEL:-https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off24/MACE-OFF24_medium.model}"
else
    export FORCE_MODEL="${FORCE_MODEL:-medium-mpa-0}"
fi
export FORCE_SEED="${FORCE_SEED:-1234}"
mode="${FORCE_EXPERIMENT:-native}"
case "$mode" in native|adapter) ;; *) echo "FORCE_EXPERIMENT must be native or adapter" >&2; exit 2 ;; esac
bash mace_scripts/force_parity/run_precision.sh "$mode"
