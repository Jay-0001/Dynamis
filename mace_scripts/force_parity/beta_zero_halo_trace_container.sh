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
export CUDA_VISIBLE_DEVICES=0
output="/workspace/Dynamis/mace_results/force_parity/trace_beta_${SLURM_JOB_ID:?}"
mkdir -p "$output"
family="${FORCE_MODEL_FAMILY:-mp}"
if [[ "$family" == off ]]; then
    model="${FORCE_MODEL:-https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off24/MACE-OFF24_medium.model}"
else
    model="${FORCE_MODEL:-medium-mpa-0}"
fi
export FORCE_MODEL_FAMILY="$family" FORCE_MODEL="$model" FORCE_SEED="${FORCE_SEED:-1234}"
if [[ -n "${FORCE_XYZ_PATH:-}" ]]; then
    input=(--xyz-path "$FORCE_XYZ_PATH")
else
    input=(--system-seed "${FORCE_SEED:-1234}" --molecule-count "${FORCE_MOLECULE_COUNT:-64}")
fi
IFS=, read -r -a backends <<< "${FORCE_BACKENDS:-e3nn,cueq}"
if [[ "${FORCE_PRECISION:-no_tf32}" == no_tf32 ]]; then
    export NVIDIA_TF32_OVERRIDE=0
fi
for backend in "${backends[@]}"; do
    backend_args=()
    case "$backend" in e3nn) ;; cueq) backend_args=(--enable-cueq) ;; *) echo "Bad backend: $backend" >&2; exit 2 ;; esac
    case_dir="$output/$backend"
    mkdir -p "$case_dir"
    python -m torch.distributed.run --standalone --nproc-per-node=1 \
        mace_scripts/force_parity/trace_zero_halo.py \
        "${input[@]}" --model-family "$family" --model "$model" \
        "${backend_args[@]}" --precision-mode "${FORCE_PRECISION:-no_tf32}" \
        --output-dir "$case_dir" > "$case_dir/run.log" 2>&1
    echo "Trace: $case_dir/trace.json"
done
python mace_scripts/force_parity/provenance.py "$output/provenance.json"
