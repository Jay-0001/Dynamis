#!/usr/bin/env bash
# Submit from the Dynamis repository root after creating mace_scripts/output.
# Override MOLECULE_COUNT, BENCHMARK_WARMUP, BENCHMARK_ITERATIONS, or
# BENCHMARK_SAMPLES in the submission environment to change the workload.
#SBATCH --job-name=mace-cueq-bench-4gpu
#SBATCH --partition=beta
#SBATCH --account=su_kchiu_molecular
#SBATCH --qos=standard
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=mace_scripts/output/mace-benchmark-4gpu-%j.out
#SBATCH --error=mace_scripts/output/mace-benchmark-4gpu-%j.err

set -euo pipefail

test -f "$SLURM_SUBMIT_DIR/dynamis_model/tests/mace/test_mace_model.py" || {
    echo "Submit from the Dynamis repository root." >&2
    exit 1
}

srun --ntasks=1 --cpus-per-task=7 \
    --container-image=nvcr.io#nvidia/pytorch:25.08-py3 \
    --container-mounts="$SLURM_SUBMIT_DIR:/workspace/Dynamis" \
    --container-workdir=/workspace/Dynamis \
    bash -lc '
        set -euo pipefail

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

        source .venv/bin/activate
        python -m pip check
        python -c "import sys, torch, dynamis_model; print(sys.executable, torch.__version__, torch.cuda.device_count()); assert torch.cuda.device_count() == 4"

        molecule_count="${MOLECULE_COUNT:-64}"
        warmup="${BENCHMARK_WARMUP:-3}"
        iterations="${BENCHMARK_ITERATIONS:-10}"
        samples="${BENCHMARK_SAMPLES:-5}"
        for value in "$molecule_count" "$iterations" "$samples"; do
            [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Molecule count, iterations, and samples must be positive integers." >&2; exit 1; }
        done
        [[ "$warmup" =~ ^[0-9]+$ ]] || { echo "Warmup must be a nonnegative integer." >&2; exit 1; }

        result_dir="mace_results/mace_benchmark_4gpu_${SLURM_JOB_ID}"
        mkdir -p "$result_dir/smoke" "$result_dir/benchmark"
        echo "Result directory: $result_dir"

        # Prove four-rank CuEq parity and graph agreement before timing.
        python dynamis_model/tests/mace/mace_standalone_reference.py \
            --enable-cueq \
            --output "$result_dir/smoke/reference.pt" \
            --output-json "$result_dir/smoke/reference.json"
        python -m torch.distributed.run --standalone --nproc-per-node=4 \
            dynamis_model/tests/mace/test_mace_model.py run-distributed \
            --enable-cueq \
            --reference-artifact "$result_dir/smoke/reference.pt" \
            --strict-parity --strict-graph-interior \
            --output-json "$result_dir/smoke/distributed.json"

        # Use the same generated structure and timing counts in both runs.
        python dynamis_model/tests/mace/mace_standalone_reference.py \
            --enable-cueq --molecule-count "$molecule_count" \
            --benchmark --benchmark-warmup "$warmup" \
            --benchmark-iterations "$iterations" --benchmark-samples "$samples" \
            --output "$result_dir/benchmark/reference.pt" \
            --output-json "$result_dir/benchmark/reference.json"
        python -m torch.distributed.run --standalone --nproc-per-node=4 \
            dynamis_model/tests/mace/test_mace_model.py run-distributed \
            --enable-cueq --molecule-count "$molecule_count" \
            --benchmark --benchmark-warmup "$warmup" \
            --benchmark-iterations "$iterations" --benchmark-samples "$samples" \
            --reference-artifact "$result_dir/benchmark/reference.pt" \
            --strict-parity --strict-graph-interior \
            --output-json "$result_dir/benchmark/distributed.json"

        echo "Four-GPU MACE-CuEq benchmark passed: $result_dir"
    '
