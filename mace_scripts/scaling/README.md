# CH4/O2 MACE OFF strong scaling on EmpireAI Beta

These scripts prepare one fixed CH4/O2 system on EmpireAI Beta and run **separate** one-GPU and two-GPU jobs. The Python driver also accepts larger GPU counts when later job scripts are added. The same XYZ, model, step count, and timestep must be used for every GPU count.

## Prepare the input

Submit the CPU preparation job from the fork root on EmpireAI Beta. Its Python environment needs NumPy. The default target is **45,000 atoms**, which gives 5,625 atoms per GPU at eight GPUs. The generator creates 5,000 CH4 and 10,000 O2 molecules in a 107 Å cubic periodic cell, with seed 42 and zero initial momenta. It writes the XYZ and adjacent SHA-256 manifest to `mace_data/strong_scaling_off/` on the cluster's shared filesystem.

```bash
cd /path/to/dynamis_fork
prep_job=$(sbatch --parsable --account=<your-account> --partition=<cpu-partition> mace_scripts/scaling/prepare_ch4_o2.sbatch)
echo "Preparation job: $prep_job"
```

This uses the existing `experiments/exp_datasets/scripts/03_generate_ch4.py` generator. The input is `mace_data/strong_scaling_off/ch4_45k_splited.xyz`; it is generated on the cluster, not shipped in Git. The preparation script refuses to replace an existing XYZ or manifest unless invoked manually with `--overwrite`. Keep the original pair for the full scaling series.

## Submit distinct jobs

Use the EmpireAI Beta allocation and GPU partition names assigned to your account. The active environment must have this fork's Dynamis packages, PyTorch with CUDA, and MACE installed. Build `dynamis_state` with `DYNAMIS_TIMING=ON`; the driver fails immediately when timing is unavailable. Confirm that the build matches the actual GPU architecture of the allocated node. Submit the run jobs with a dependency on successful preparation:

```bash
cd /path/to/dynamis_fork
sbatch --account=<your-account> --partition=<gpu-partition> --dependency=afterok:${prep_job%%;*} mace_scripts/scaling/run_1gpu.sbatch
sbatch --account=<your-account> --partition=<gpu-partition> --dependency=afterok:${prep_job%%;*} mace_scripts/scaling/run_2gpu.sbatch
```

The scripts use `SLURM_SUBMIT_DIR` as the fork root, request one node, and launch one `torchrun` process per allocated GPU. They use the `python` on `PATH`; set `PYTHON_BIN` if needed. Both set `OMP_NUM_THREADS=4` per process by default. Supported environment overrides are `XYZ_PATH`, `STEPS`, `WARMUP_STEPS`, `TIMESTEP_FS`, `REBALANCE_INTERVAL`, `MACE_OFF_MODEL`, and `OMP_NUM_THREADS`. Apply the same values to both jobs. `DYNAMIS_FORK_ROOT` can override the submit directory if jobs must be submitted elsewhere.

## Results and metric

The preparation job writes `mace_off_prepare_<job ID>.out` in the submission directory. Each run job gets its own `mace_results/strong_scaling_off/{1gpu,2gpu}_<Slurm job ID>/` directory. It contains `run_metadata.json`, one `timing_rank<N>.jsonl` per rank, `strong_scaling_summary.json`, and `timing_diagnostics.txt`. The run job's Slurm stdout is `mace_off_<N>gpu_<job ID>.out` in the submission directory. Generated inputs and job outputs are ignored by Git.

`Simulator.run()` writes the existing per-rank `iteration` timers. The driver synchronizes its GPU at the end of every step so the **host** duration includes queued GPU work. The new summary omits the first 20 of 120 steps by default, takes the slowest rank's host duration for each later step, and reports the median of those step durations. It checks that every expected rank and step is present. `timing_diagnostics.txt` comes from the existing `post_processing/timing_summary.py` and is for per-key investigation; the strong-scaling throughput metric comes from `strong_scaling_summary.json`.

The measured region includes MD integration, MACE forward/backward, communication, and any drift-triggered halo rebuild. Initial model loading and simulation setup are excluded. Periodic repartitioning is disabled by default (`REBALANCE_INTERVAL=0`), while the drift watchdog remains active. The synthetic mixture is a scaling workload; the timings do not establish chemical accuracy or stable long-time dynamics.

Compare the two jobs only after checking `xyz_sha256`, atom count, model, step count, and timestep in their metadata. The ideal two-GPU speedup denominator is the one-GPU median ms/step; divide that by the two-GPU median ms/step to get observed speedup. Confirm the 45,000-atom input fits on one GPU before interpreting the scaling series; capacity on EmpireAI Beta has not been measured here.
