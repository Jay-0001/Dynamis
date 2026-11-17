#!/usr/bin/env bash
#SBATCH --job-name=mace-off-1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=mace_off_1gpu_%j.out
set -euo pipefail

project_root="${DYNAMIS_FORK_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the fork root}}"
cd "$project_root"
python_bin="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
xyz_path="${XYZ_PATH:-$project_root/mace_data/strong_scaling_off/ch4_45k_splited.xyz}"
test -f "$xyz_path" || { echo "Missing XYZ: $xyz_path. Run prepare_ch4_o2.py first." >&2; exit 1; }
output_dir="$project_root/mace_results/strong_scaling_off/1gpu_${SLURM_JOB_ID:?}"

"$python_bin" -m torch.distributed.run --standalone --nproc_per_node=1 \
  mace_scripts/scaling/run_mace_off.py \
  --xyz "$xyz_path" --output-dir "$output_dir" --expected-gpus 1 \
  --steps "${STEPS:-120}" --warmup-steps "${WARMUP_STEPS:-20}" \
  --timestep-fs "${TIMESTEP_FS:-0.1}" --rebalance-interval "${REBALANCE_INTERVAL:-0}" \
  --model "${MACE_OFF_MODEL:-medium}"

"$python_bin" post_processing/strong_scaling_summary.py "$output_dir"
"$python_bin" post_processing/timing_summary.py "$output_dir" --by-rank > "$output_dir/timing_diagnostics.txt"
echo "Results: $output_dir"
