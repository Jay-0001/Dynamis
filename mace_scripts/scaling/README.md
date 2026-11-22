# MACE OFF active-worker scaling on EmpireAI Beta

The shared driver now also supports Medium MPA with separate `run_{1,2,4}gpu_mpa.sh` jobs. They use the same cluster-generated XYZ and manifest, and write to `mace_results/strong_scaling_mpa/`. This CH4/O2 workload compares execution cost; it does not validate MPA chemistry. Start with `STEPS=4,WARMUP_STEPS=1` to verify the MPA path, then run the standard 120-step setting. Keep `MACE_ENABLE_CUEQ=0` for the initial MPA E2E runs while the static CuEq force-parity discrepancy remains open.

These jobs compare one, two, and four active GPU workers on the **same** CH4/O2
system. Every job reserves four B200 GPUs on one Beta node to satisfy the
minimum GPU request shown in the Beta login banner. The one- and two-worker
jobs mask the unused GPUs inside the container. They still consume a four-GPU
allocation; the comparison measures worker scaling, not service-unit
efficiency or allocation scaling. Recheck the live policy before submission.

## Runtime and inputs

Submit from the root of the Beta clone (`~/Dynamis`) after installing the
repaired `.venv` with `mace_scripts/empire_dynamis_build.sh`. The jobs use the
ARM64 `nvcr.io#nvidia/pytorch:25.08-py3` image via Pyxis, mount the repository
at `/workspace/Dynamis`, and run `.venv/bin/python`. Runtime caches are placed
in the writable `mace_scripts/output/cache` directory. The installed
`dynamis_state` extension must report `TIMING_ENABLED=True`; its CMake option
defaults to ON, but the binary in `.venv` is checked at runtime.

The preparation job requests four GPUs because the observed Beta partitions
do not include a CPU-only partition. Its GPU time is idle. If you already
generated the XYZ and matching manifest elsewhere, copy both into
`mace_data/strong_scaling_off/` and skip preparation.

The default preparation creates 45,000 atoms (5,000 CH4 and 10,000 O2
molecules) in a periodic 107 Å cube with seed 42 and zero initial momenta.
It writes `ch4_45k_splited.xyz` and `ch4_45k_splited.json`; the runner checks
the manifest hash and atom count. Do not regenerate the input between worker
counts.

## Pilot: small input and short run

Create the log directory **before** calling `sbatch`, from the clone root:

```bash
cd ~/Dynamis
mkdir -p mace_scripts/output mace_results
sinfo -o '%P %a %l %D %G'
sacctmgr show assoc user="$USER" format=Account%30,Partition%20,QOS%60
sbatch --test-only mace_scripts/scaling/run_4gpu.sh
```

An 8k target produces 8,001 atoms because each CH4 plus two O2 molecules
contains nine atoms. Prepare it once, then run four active workers for 20
steps. The `--dependency` avoids launching the GPU run if preparation fails.

```bash
prep_job=$(sbatch --parsable --export=ALL,SIZE_K=8 mace_scripts/scaling/prepare_ch4_o2.sbatch)
sbatch --dependency=afterok:${prep_job%%;*} \
  --export=ALL,XYZ_PATH=/workspace/Dynamis/mace_data/strong_scaling_off/ch4_8k_splited.xyz,STEPS=20,WARMUP_STEPS=5 \
  mace_scripts/scaling/run_4gpu.sh
```

If that XYZ and manifest already exist, omit the preparation job and its
dependency. The preparation script intentionally refuses to overwrite either
file. Inspect `mace_scripts/output/mace-off-4worker-<job>.{out,err}` and the
summary JSON before scheduling the full system.

## Full fixed-input series

The default workload is 45,000 atoms, 120 steps, and 20 warmup steps. One
active GPU must be able to hold the whole system; this capacity has **not**
been established on Beta. Run the four-worker case first and inspect its
output before submitting the smaller worker counts.

```bash
cd ~/Dynamis
prep_job=$(sbatch --parsable mace_scripts/scaling/prepare_ch4_o2.sbatch)
sbatch --dependency=afterok:${prep_job%%;*} mace_scripts/scaling/run_4gpu.sh
```

After the four-worker summary has passed, submit `run_2gpu.sh`; inspect its
memory use and summary before submitting `run_1gpu.sh`. If the 45k XYZ and
manifest already exist, skip preparation and submit `run_4gpu.sh` directly.
All three request `beta`, `su_kchiu_molecular`, `standard`, four GPUs, one
Slurm task, and seven CPUs per task. They set the same value on `srun` to
avoid conflicting Slurm CPU settings. The default limit is four hours and
128 GiB of host memory per run; override these with `sbatch --time` or
`--mem` after the pilot if needed. `OMP_NUM_THREADS` defaults to one for each
Torch worker so four workers fit within the seven-CPU allocation.

The checkpoint is MACE OFF24 medium (the explicit OFF24 model URL), using the e3nn
backend. To compare CuEq instead, submit **all** jobs with
`--export=ALL,MACE_ENABLE_CUEQ=1`. `MACE_OFF_MODEL`, `XYZ_PATH`, `STEPS`,
`WARMUP_STEPS`, `TIMESTEP_FS`, `REBALANCE_INTERVAL`, and `OMP_NUM_THREADS`
are also accepted, but must match across worker counts. Model loading and
simulation setup are outside the timed region.
Set `MACE_OFF_MODEL=medium` explicitly only when comparing against the older
OFF23 checkpoint; record that override because the two models are not
interchangeable scaling cases.

## Outputs and interpretation

Slurm logs are in `mace_scripts/output/`. Each run writes to
`mace_results/strong_scaling_off/<workers>gpu_<Slurm job ID>/`, including
`run_metadata.json`, `timing_rank<N>.jsonl`, `strong_scaling_summary.json`,
and `timing_diagnostics.txt`. Metadata records both the four allocated GPUs
and the active worker count, along with the backend and input hash.

The driver synchronizes the GPU at the end of each MD step. The summary
requires every rank and step, takes the slowest rank's synchronized host
duration for each post-warmup step, and reports their median. This includes
integration, MACE forward/backward, communication, and any halo rebuild. The
input has zero initial momenta and serves as a scaling workload; these timings
do not establish chemical accuracy or stable long-time dynamics. Compare
summaries only after confirming the input hash, atom count, model, backend,
steps, timestep, and warmup match.

An eight-worker series needs a separate multi-node launcher because the
observed Beta nodes expose four B200 GPUs each. These scripts are single-node.
