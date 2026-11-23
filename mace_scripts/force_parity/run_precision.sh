#!/usr/bin/env bash
# Run independent native repeats, or matched native/2-rank adapter comparisons.
set -euo pipefail
mode="${1:?native or adapter}"
cd "$(dirname "$0")/../.."
: "${FORCE_OUTPUT_DIR:?}"
mkdir -p "$FORCE_OUTPUT_DIR"
case "${FORCE_MODEL_FAMILY:-mp}" in mp|off) ;; *) echo "Bad FORCE_MODEL_FAMILY" >&2; exit 2 ;; esac
IFS=, read -r -a backends <<< "${FORCE_BACKENDS:-e3nn,cueq}"
IFS=, read -r -a precisions <<< "${FORCE_PRECISIONS:-default,no_tf32}"
for backend in "${backends[@]}"; do
    case "$backend" in e3nn|cueq) ;; *) echo "Bad backend: $backend" >&2; exit 2 ;; esac
done
for precision in "${precisions[@]}"; do
    case "$precision" in default|no_tf32) ;; *) echo "Bad precision: $precision" >&2; exit 2 ;; esac
done
repeats="${FORCE_REPEATS:-5}"
[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || { echo "Bad repeat count: $repeats" >&2; exit 2; }
count="${FORCE_MOLECULE_COUNT:-1}"
[[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo "Bad molecule count: $count" >&2; exit 2; }
if [[ -n "${FORCE_XYZ_PATH:-}" ]]; then
    test -f "$FORCE_XYZ_PATH"
    input=(--xyz-path "$FORCE_XYZ_PATH")
else
    input=(--system-seed "${FORCE_SEED:-1234}" --molecule-count "$count")
fi
export FORCE_MODEL_FAMILY FORCE_MODEL FORCE_SEED
python mace_scripts/force_parity/provenance.py "$FORCE_OUTPUT_DIR/provenance.json"
had_tf32_override=false
if [[ -v NVIDIA_TF32_OVERRIDE ]]; then
    had_tf32_override=true
    original_tf32_override="$NVIDIA_TF32_OVERRIDE"
fi
for precision in "${precisions[@]}"; do
    if [[ "$precision" == no_tf32 ]]; then
        export NVIDIA_TF32_OVERRIDE=0
    elif [[ "$had_tf32_override" == true ]]; then
        export NVIDIA_TF32_OVERRIDE="$original_tf32_override"
    else
        unset NVIDIA_TF32_OVERRIDE
    fi
    for backend in "${backends[@]}"; do
        backend_args=()
        [[ "$backend" != cueq ]] || backend_args=(--enable-cueq)
        case_dir="$FORCE_OUTPUT_DIR/$precision/$backend"
        mkdir -p "$case_dir"
        if [[ "$mode" == native ]]; then
            for ((trial=1; trial<=repeats; trial++)); do
                echo "Native $precision $backend trial $trial/$repeats"
                python dynamis_model/tests/mace/mace_standalone_reference.py \
                    "${input[@]}" --model-family "$FORCE_MODEL_FAMILY" --model "$FORCE_MODEL" \
                    "${backend_args[@]}" --precision-mode "$precision" \
                    --output "$case_dir/trial${trial}.pt" \
                    --output-json "$case_dir/trial${trial}.json" \
                    > "$case_dir/trial${trial}.log" 2>&1
            done
        elif [[ "$mode" == adapter ]]; then
            echo "Matched adapter $precision $backend"
            python dynamis_model/tests/mace/mace_standalone_reference.py \
                "${input[@]}" --model-family "$FORCE_MODEL_FAMILY" --model "$FORCE_MODEL" \
                "${backend_args[@]}" --precision-mode "$precision" \
                --output "$case_dir/reference.pt" --output-json "$case_dir/reference.json" \
                > "$case_dir/reference.log" 2>&1
            python -m torch.distributed.run --standalone --nproc-per-node=2 \
                dynamis_model/tests/mace/test_mace_model.py run-distributed \
                "${input[@]}" --model-family "$FORCE_MODEL_FAMILY" --model "$FORCE_MODEL" \
                "${backend_args[@]}" --precision-mode "$precision" \
                --reference-artifact "$case_dir/reference.pt" --graph-edge-test \
                --output-json "$case_dir/adapter.json" \
                --output-forces "$case_dir/adapter_forces.pt" \
                > "$case_dir/adapter.log" 2>&1
        fi
    done
done
python mace_scripts/force_parity/summarize_precision.py "$FORCE_OUTPUT_DIR" "$mode"
python mace_scripts/force_parity/provenance.py "$FORCE_OUTPUT_DIR/provenance.json"
echo "Precision results: $FORCE_OUTPUT_DIR"
