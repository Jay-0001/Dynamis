#!/usr/bin/env bash
#SBATCH --job-name=dynamis-cueq-4gpu
#SBATCH --partition=beta
#SBATCH --account=su_kchiu_molecular
#SBATCH --qos=standard
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=mace_scripts/output/cueq-4gpu-%j.out
#SBATCH --error=mace_scripts/output/cueq-4gpu-%j.err

set -euo pipefail

srun --ntasks=1 --cpus-per-task=7 --container-image=nvcr.io#nvidia/pytorch:25.08-py3 \
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

    source .venv/bin/activate
    export PYTHONNOUSERSITE=1

    result_dir="mace_results/cueq_4gpu_${SLURM_JOB_ID}"
    mkdir -p "$result_dir"

    python -m pip check
    python -c "import torch, dynamis, dynamis_state.dynamis_state_impl, dynamis_integrator.dynamis_integrator_impl, dynamis_model.dynamis_model_impl; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.device_count() == 4"

    python dynamis_model/tests/mace/mace_standalone_reference.py \
      --enable-cueq \
      --output "$result_dir/reference.pt" \
      --output-json "$result_dir/reference.json"

    torchrun --standalone --nproc-per-node=4 \
      dynamis_model/tests/mace/test_mace_model.py run-distributed \
      --enable-cueq \
      --reference-artifact "$result_dir/reference.pt" \
      --strict-parity --strict-graph-interior \
      --output-json "$result_dir/distributed.json"

    echo "Four-GPU MACE-CuEq verification passed: $result_dir"
  '
