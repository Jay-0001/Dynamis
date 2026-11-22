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
backend="${OFF24_BACKEND:-e3nn}"
case "$backend" in
    e3nn) backend_args=() ;;
    cueq) backend_args=(--enable-cueq) ;;
    *) echo "OFF24_BACKEND must be e3nn or cueq" >&2; exit 2 ;;
esac
model="${OFF24_MODEL:-https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off24/MACE-OFF24_medium.model}"
xyz="${OFF24_XYZ:-/workspace/Dynamis/mace_data/off24_bridge.xyz}"
test -f "$xyz" || { echo "Missing XYZ: $xyz" >&2; exit 1; }
result="/workspace/Dynamis/mace_results/off24_beta_${backend}_${SLURM_JOB_ID:?}"
mkdir -p "$result"
python dynamis_model/tests/mace/mace_standalone_reference.py \
    --xyz-path "$xyz" --model-family off --model "$model" \
    "${backend_args[@]}" --output "$result/reference.pt" \
    --output-json "$result/reference.json"
python -m torch.distributed.run --standalone --nproc-per-node=4 \
    dynamis_model/tests/mace/test_mace_model.py run-distributed \
    --xyz-path "$xyz" --model-family off --model "$model" \
    "${backend_args[@]}" --reference-artifact "$result/reference.pt" \
    --strict-parity --graph-edge-test --strict-graph-interior \
    --output-json "$result/distributed.json"
echo "OFF24 parity results: $result"
