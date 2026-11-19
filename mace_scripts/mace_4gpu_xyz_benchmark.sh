#!/usr/bin/env bash
# Four-rank MPA/CuEq parity and timing on a large nonperiodic XYZ fixture.
# Submit from the Dynamis repository root after creating mace_scripts/output.
# XYZ_FILE defaults to mace_data/benchmark_scaling_molecular_7500.xyz.
# This script requires --xyz-path support in the two MACE test entry points.
#SBATCH --job-name=mace-cueq-xyz-4gpu
#SBATCH --partition=beta
#SBATCH --account=su_kchiu_molecular
#SBATCH --qos=standard
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=mace_scripts/output/mace-xyz-4gpu-%j.out
#SBATCH --error=mace_scripts/output/mace-xyz-4gpu-%j.err

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
        export OMP_NUM_THREADS=1

        source .venv/bin/activate
        python -m pip check
        python -c "import sys, torch, dynamis_model; print(sys.executable, torch.__version__, torch.cuda.device_count()); assert torch.cuda.device_count() == 4"

        xyz_path="${XYZ_FILE:-mace_data/benchmark_scaling_molecular_7500.xyz}"
        [[ -f "$xyz_path" ]] || { echo "Missing XYZ file: $xyz_path" >&2; exit 1; }
        reference_help="$(python dynamis_model/tests/mace/mace_standalone_reference.py --help)"
        [[ "$reference_help" == *--xyz-path* ]] || {
            echo "The Beta clone lacks --xyz-path support; sync the MACE test entry points." >&2
            exit 1
        }
        distributed_help="$(python dynamis_model/tests/mace/test_mace_model.py run-distributed --help)"
        [[ "$distributed_help" == *--xyz-path* ]] || {
            echo "The distributed MACE test lacks --xyz-path support; sync the test files." >&2
            exit 1
        }
        warmup="${BENCHMARK_WARMUP:-2}"
        iterations="${BENCHMARK_ITERATIONS:-5}"
        samples="${BENCHMARK_SAMPLES:-3}"
        for value in "$iterations" "$samples"; do
            [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Iterations and samples must be positive integers." >&2; exit 1; }
        done
        [[ "$warmup" =~ ^[0-9]+$ ]] || { echo "Warmup must be a nonnegative integer." >&2; exit 1; }

        result_dir="mace_results/mace_xyz_4gpu_${SLURM_JOB_ID}"
        mkdir -p "$result_dir"
        echo "XYZ: $xyz_path"
        echo "Results: $result_dir"

        python dynamis_model/tests/mace/mace_standalone_reference.py \
            --xyz-path "$xyz_path" --enable-cueq \
            --benchmark --benchmark-warmup "$warmup" \
            --benchmark-iterations "$iterations" --benchmark-samples "$samples" \
            --output "$result_dir/reference.pt" \
            --output-json "$result_dir/reference.json"

        # The test writes its JSON before enforcing strict parity. A numerical
        # mismatch therefore leaves both timings available and fails the job.
        python -m torch.distributed.run --standalone --nproc-per-node=4 \
            dynamis_model/tests/mace/test_mace_model.py run-distributed \
            --xyz-path "$xyz_path" --enable-cueq \
            --benchmark --benchmark-warmup "$warmup" \
            --benchmark-iterations "$iterations" --benchmark-samples "$samples" \
            --reference-artifact "$result_dir/reference.pt" \
            --strict-parity --strict-graph-interior \
            --output-json "$result_dir/distributed.json"

        echo "Four-GPU large-XYZ CuEq benchmark and strict parity passed: $result_dir"
    '
