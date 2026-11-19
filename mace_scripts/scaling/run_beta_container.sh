#!/usr/bin/env bash
# Runs inside the Pyxis container with the repository mounted at /workspace/Dynamis.
set -euo pipefail

cd /workspace/Dynamis
workers="${1:?Pass the active GPU worker count}"
case "$workers" in
    1) export CUDA_VISIBLE_DEVICES=0 ;;
    2) export CUDA_VISIBLE_DEVICES=0,1 ;;
    4) ;;  # Preserve Pyxis's four-GPU visibility.
    *) echo "Unsupported worker count: $workers" >&2; exit 2 ;;
esac

cache_root=/workspace/Dynamis/mace_scripts/output/cache
mkdir -p "$cache_root"/{home,xdg,pip,mpl,triton,cueq,torch}
export HOME="$cache_root/home"
export XDG_CACHE_HOME="$cache_root/xdg"
export PIP_CACHE_DIR="$cache_root/pip"
export MPLCONFIGDIR="$cache_root/mpl"
export TRITON_CACHE_DIR="$cache_root/triton"
export CUEQ_TRITON_CACHE_DIR="$cache_root/cueq"
export TORCH_HOME="$cache_root/torch"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export SCALING_ALLOCATED_GPUS=4
unset PYTHONPATH PYTHONHOME

source .venv/bin/activate
python -m pip check
python - "$workers" <<'PY'
import sys
import torch
from dynamis_state import TIMING_ENABLED

expected = int(sys.argv[1])
assert TIMING_ENABLED, "Rebuild dynamis_state with -DDYNAMIS_TIMING=ON"
assert torch.cuda.device_count() == expected, (
    f"Expected {expected} visible GPUs; found {torch.cuda.device_count()}"
)
print(f"Python: {sys.executable}; Torch: {torch.__version__}; "
      f"CUDA: {torch.version.cuda}; active GPUs: {expected}; allocated GPUs: 4", flush=True)
PY

xyz_path="${XYZ_PATH:-/workspace/Dynamis/mace_data/strong_scaling_off/ch4_45k_splited.xyz}"
test -f "$xyz_path" || { echo "Missing XYZ: $xyz_path. Run preparation first." >&2; exit 1; }
test -f "${xyz_path%.xyz}.json" || { echo "Missing XYZ manifest: ${xyz_path%.xyz}.json" >&2; exit 1; }

model="${MACE_OFF_MODEL:-medium}"
backend_args=()
case "${MACE_ENABLE_CUEQ:-0}" in
    0) ;;
    1) backend_args=(--enable-cueq) ;;
    *) echo "MACE_ENABLE_CUEQ must be 0 or 1" >&2; exit 2 ;;
esac

output_dir="/workspace/Dynamis/mace_results/strong_scaling_off/${workers}gpu_${SLURM_JOB_ID:?}"

# Cache the checkpoint once before the ranks load it concurrently.
python - "$model" "${MACE_ENABLE_CUEQ:-0}" <<'PY'
import sys
from dynamis_model.nnp.mace import load_raw_mace_model

load_raw_mace_model("off", sys.argv[1], device="cuda:0", enable_cueq=sys.argv[2] == "1")
print("MACE OFF checkpoint preloaded", flush=True)
PY

python -m torch.distributed.run --standalone --nproc-per-node="$workers" \
    mace_scripts/scaling/run_mace_off.py \
    --xyz "$xyz_path" --output-dir "$output_dir" --expected-gpus "$workers" \
    --steps "${STEPS:-120}" --warmup-steps "${WARMUP_STEPS:-20}" \
    --timestep-fs "${TIMESTEP_FS:-0.1}" --rebalance-interval "${REBALANCE_INTERVAL:-0}" \
    --model "$model" "${backend_args[@]}"

python post_processing/strong_scaling_summary.py "$output_dir"
python post_processing/timing_summary.py "$output_dir" --by-rank > "$output_dir/timing_diagnostics.txt"
echo "Results: $output_dir"
