#!/usr/bin/env bash
# Run matching native-reference and Dynamis adapter cases on one cluster node.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
: "${FORCE_OUTPUT_DIR:?Set FORCE_OUTPUT_DIR to a fresh result directory}"
mkdir -p "$FORCE_OUTPUT_DIR"

family="${FORCE_MODEL_FAMILY:-mp}"
case "$family" in
    mp) model="${FORCE_MODEL:-medium-mpa-0}" ;;
    off) model="${FORCE_MODEL:-https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off24/MACE-OFF24_medium.model}" ;;
    *) echo "FORCE_MODEL_FAMILY must be mp or off" >&2; exit 2 ;;
esac
IFS=, read -r -a backends <<< "${FORCE_BACKENDS:-e3nn,cueq}"
IFS=, read -r -a ranks_list <<< "${FORCE_RANKS:-2}"
IFS=, read -r -a reductions <<< "${FORCE_REDUCTIONS:-legacy,current}"
IFS=, read -r -a counts <<< "${FORCE_COUNTS:-1}"
seed="${FORCE_SEED:-1234}"
xyz="${FORCE_XYZ_PATH:-}"

for backend in "${backends[@]}"; do
    case "$backend" in e3nn|cueq) ;; *) echo "Bad backend: $backend" >&2; exit 2 ;; esac
done
for ranks in "${ranks_list[@]}"; do
    [[ "$ranks" =~ ^[0-9]+$ ]] && (( ranks >= 2 )) || {
        echo "Ranks must be at least 2: $ranks" >&2; exit 2;
    }
    (( ranks <= $(python -c 'import torch; print(torch.cuda.device_count())') )) || {
        echo "Requested $ranks ranks exceed visible GPUs" >&2; exit 2;
    }
done
for reduction in "${reductions[@]}"; do
    case "$reduction" in legacy|current) ;; *) echo "Bad reduction: $reduction" >&2; exit 2 ;; esac
done
for count in "${counts[@]}"; do
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo "Bad molecule count: $count" >&2; exit 2; }
done

export FORCE_MODEL_FAMILY="$family" FORCE_MODEL="$model" FORCE_SEED="$seed"
python mace_scripts/force_parity/provenance.py "$FORCE_OUTPUT_DIR/provenance.json"

if [[ -n "$xyz" ]]; then
    test -f "$xyz" || { echo "Missing XYZ: $xyz" >&2; exit 2; }
    case_names=(xyz)
else
    case_names=("${counts[@]}")
fi

for case_name in "${case_names[@]}"; do
    if [[ "$case_name" == xyz ]]; then
        input_args=(--xyz-path "$xyz")
        case_dir="$FORCE_OUTPUT_DIR/xyz"
    else
        input_args=(--system-seed "$seed" --molecule-count "$case_name")
        case_dir="$FORCE_OUTPUT_DIR/count${case_name}"
    fi
    for backend in "${backends[@]}"; do
        backend_args=()
        [[ "$backend" != cueq ]] || backend_args=(--enable-cueq)
        result="$case_dir/$backend"
        mkdir -p "$result"
        echo "Reference: $case_name $backend"
        python dynamis_model/tests/mace/mace_standalone_reference.py \
            "${input_args[@]}" --model-family "$family" --model "$model" \
            "${backend_args[@]}" --output "$result/reference.pt" \
            --output-json "$result/reference.json" > "$result/reference.log" 2>&1
        for ranks in "${ranks_list[@]}"; do
            for reduction in "${reductions[@]}"; do
                run_dir="$result/${ranks}rank_$reduction"
                mkdir -p "$run_dir"
                echo "Adapter: $case_name $backend ${ranks}rank $reduction"
                python -m torch.distributed.run --standalone --nproc-per-node="$ranks" \
                    dynamis_model/tests/mace/test_mace_model.py run-distributed \
                    "${input_args[@]}" --model-family "$family" --model "$model" \
                    "${backend_args[@]}" --reference-artifact "$result/reference.pt" \
                    --adapter-reduction "$reduction" --graph-edge-test \
                    --output-json "$run_dir/distributed.json" \
                    --output-forces "$run_dir/forces.pt" > "$run_dir/run.log" 2>&1
            done
        done
    done
done

python mace_scripts/force_parity/provenance.py "$FORCE_OUTPUT_DIR/provenance.json"
python mace_scripts/force_parity/summarize.py "$FORCE_OUTPUT_DIR"
echo "Force parity results: $FORCE_OUTPUT_DIR"
