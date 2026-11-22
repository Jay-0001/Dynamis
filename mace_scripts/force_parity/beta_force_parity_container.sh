#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Dynamis
cache_root=/workspace/Dynamis/mace_scripts/output/cache
mkdir -p "$cache_root"/{home,xdg,pip,mpl,triton,cueq,torch}
export HOME="$cache_root/home" XDG_CACHE_HOME="$cache_root/xdg"
export PIP_CACHE_DIR="$cache_root/pip" MPLCONFIGDIR="$cache_root/mpl"
export TRITON_CACHE_DIR="$cache_root/triton" CUEQ_TRITON_CACHE_DIR="$cache_root/cueq"
export TORCH_HOME="$cache_root/torch" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
source .venv/bin/activate
export FORCE_RANKS="${FORCE_RANKS:-2,4}"
export FORCE_OUTPUT_DIR="/workspace/Dynamis/mace_results/force_parity/beta_${SLURM_JOB_ID:?}"
bash mace_scripts/force_parity/run_matrix.sh
