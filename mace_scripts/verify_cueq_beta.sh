#!/usr/bin/env bash
#SBATCH --job-name=dynamis-cueq-check
#SBATCH --partition=beta_test
#SBATCH --account=su_kchiu_molecular
#SBATCH --qos=test
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=mace_scripts/output/cueq-%j.out
#SBATCH --error=mace_scripts/output/cueq-%j.err

set -euo pipefail

srun --container-image=nvcr.io#nvidia/pytorch:25.08-py3 \
  --container-mounts="$SLURM_SUBMIT_DIR:/workspace/Dynamis" \
  --container-workdir=/workspace/Dynamis \
  bash -lc '
    set -euo pipefail
    source .venv/bin/activate
    export PYTHONNOUSERSITE=1

    result_dir="mace_results/cueq_smoke_${SLURM_JOB_ID}"
    mkdir -p "$result_dir"

    python -m pip check
    python -c "import torch, dynamis, dynamis_state.dynamis_state_impl, dynamis_integrator.dynamis_integrator_impl, dynamis_model.dynamis_model_impl; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.device_count() >= 2"
    python -c "from importlib.metadata import version; print({p: version(p) for p in [\"mace-torch\", \"cuequivariance\", \"cuequivariance-torch\", \"cuequivariance-ops-cu13\", \"cuequivariance-ops-torch-cu13\"]})"

    python dynamis_model/tests/mace/mace_standalone_reference.py \
      --enable-cueq \
      --output "$result_dir/reference.pt" \
      --output-json "$result_dir/reference.json"

    torchrun --standalone --nproc-per-node=2 \
      dynamis_model/tests/mace/test_mace_model.py run-distributed \
      --enable-cueq \
      --reference-artifact "$result_dir/reference.pt" \
      --strict-parity --strict-graph-interior \
      --output-json "$result_dir/distributed.json"

    echo "MACE-CuEq verification passed: $result_dir"
    '	
