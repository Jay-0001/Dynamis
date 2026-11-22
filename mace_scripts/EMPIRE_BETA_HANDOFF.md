# EmpireAI Beta / Dynamis handoff (2026-11-22)

## Cluster and allocation

- Login/compute architecture: `aarch64`; Beta GPU nodes report NVIDIA GB200 (B200 GPU), 4 GPUs per node.
- Account: `su_kchiu_molecular`; reported QoS includes `long,priority,standard,test`. Production jobs use `--partition=beta --qos=standard`; the Beta banner stated a four-GPU minimum. Recheck `sinfo` and `sacctmgr` before new submissions.
- Batch jobs here request `--gres=gpu:4 --ntasks=1 --cpus-per-task=7`. Keep 7 in the inner `srun` too; an earlier conflicting 7 vs 4 CPU setting stopped a job. `--test-only` may report 144 processors because a full four-GPU NVL72 node is assigned, despite the per-task CPU request.
- Login node lacks a CUDA driver/toolkit; GPU access is on allocated compute nodes. Pyxis is installed (`srun --help | grep container-image`); the NGC image is `nvcr.io#nvidia/pytorch:25.08-py3` and mounts `~/Dynamis` as `/workspace/Dynamis`.
- Container Python uses repository `.venv` with NVIDIA image system packages. Observed Torch `2.8.0a0+34c6371d24.nv25.08`, CUDA 13.0; `mace-torch` and `cueq` were verified installed. Writable caches live under `mace_scripts/output/cache/` (`HOME`, XDG, pip, Triton, CuEq, Torch). No per-user NVIDIA Container Toolkit installation is needed for Pyxis GPU jobs.

## Important paths

- Local working branch: `W:\Building\Summer Research\dynamis_fork`, branch `scaling-experiments-mace`.
- Beta clone: `~/Dynamis`, container path `/workspace/Dynamis`.
- Batch logs: `mace_scripts/output/` (singular). Older Delta logs may be in `mace_scripts/outputs/` (plural).
- Parity JSON: `mace_results/`; scaling results: `mace_results/strong_scaling_off/<workers>gpu_<jobid>/`.
- Scaling inputs: `mace_data/strong_scaling_off/ch4_8k_splited.xyz` and same-basename `.json` manifest, or 45k counterparts. The driver requires an axis-aligned cubic periodic XYZ and matching manifest SHA-256. `mace_data/off24_bridge.xyz` is for **static parity**, not the end-to-end driver.
- Scripts: `mace_scripts/scaling/run_{1,2,4}gpu.sh`, `run_beta_job.sh`, `run_beta_container.sh`, `run_mace_off.py`, `prepare_ch4_o2.sbatch`, and new `check_off24_parity_{beta,container}.sh`.

## Completed and failed work

- A previous environment repair succeeded; `.venv` supports `mace-torch` and `cueq`. Do not rebuild the environment solely due to the OFF end-to-end traceback.
- MPA CuEq static Beta runs: `mace-benchmark-4gpu-233976` (12 and 768 atoms), `mace-xyz-4gpu-233977` (7,500 atoms). Energy parity passed, force parity failed, graph edge multisets matched. For 768 atoms, all four ranks had **zero halos**, and force max error was `3.2853e-5` MD units; the 7,500-atom max was `6.3246e-5` MD units with about 780 halos/rank. Therefore halo exchange is not the only possible source. The 7,500-atom distributed JSON is `mace_results/mace_xyz_4gpu_233977/distributed.json`.
- Earlier Delta OFF24 smoke run `mace_results/off24_smoke_24080085/` passed 12-atom force parity for e3nn and CuEq (max `1.49e-8` and `3.73e-8` MD units). OFF24 URL: `https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off24/MACE-OFF24_medium.model`. The adapter accepts `--model-family off --model <URL>` and the checkpoint contract was inspected in `mace_scripts/off24_static.sh`. The loader may print “OFF23” while resolving this explicit OFF24 URL; verify the checkpoint path/metadata rather than that generic label.
- End-to-end jobs `mace-off-2worker-235825` and `mace-off-4worker-235826` both reserved four GPUs, activated 2 and 4 workers respectively, loaded the **default OFF23 medium e3nn** checkpoint, checked pip and initialized distributed ranks. Both failed during step 0: TorchScript/e3nn `_w3j_1_2_1` stayed on CPU while inputs were on `cuda:<rank>`. `ChildFailedError` and `srun` exit 1 are downstream symptoms. Partial timing files/trajectory headers are not successful scaling measurements; no summary was produced.
- The scaling driver `run_mace_off.py` now explicitly calls `raw.to(device=env.device, dtype=torch.float32)` before `MACEModel(raw)`. Static tests already performed an explicit device move. The end-to-end launcher and driver now default to the explicit OFF24 medium URL; `MACE_OFF_MODEL=medium` is an override for the historical OFF23 comparison. These changes have passed syntax checking but **have not yet been run on Beta**. Do not label the end-to-end issue resolved until a short rerun completes with `strong_scaling_summary.json`.

## Next commands on Beta (after syncing the changed files)

```bash
cd ~/Dynamis
mkdir -p mace_scripts/output mace_results
sinfo -o '%P %a %l %D %G'
sacctmgr show assoc user="$USER" format=Account%30,Partition%20,QOS%60

# Static OFF24 parity on the 12-atom bridge, first e3nn then CuEq.
sbatch --export=ALL,OFF24_BACKEND=e3nn mace_scripts/scaling/check_off24_parity_beta.sh
sbatch --export=ALL,OFF24_BACKEND=cueq mace_scripts/scaling/check_off24_parity_beta.sh

# Run a short OFF24 end-to-end check with each worker count.
sbatch --export=ALL,XYZ_PATH=/workspace/Dynamis/mace_data/strong_scaling_off/ch4_8k_splited.xyz,STEPS=4,WARMUP_STEPS=1 mace_scripts/scaling/run_2gpu.sh
sbatch --export=ALL,XYZ_PATH=/workspace/Dynamis/mace_data/strong_scaling_off/ch4_8k_splited.xyz,STEPS=4,WARMUP_STEPS=1 mace_scripts/scaling/run_4gpu.sh

# To reproduce the older OFF23 model explicitly, override the default:
# sbatch --export=ALL,MACE_OFF_MODEL=medium,XYZ_PATH=/workspace/Dynamis/mace_data/strong_scaling_off/ch4_8k_splited.xyz,STEPS=4,WARMUP_STEPS=1 mace_scripts/scaling/run_2gpu.sh
```

Each `sbatch` command reserves 4 GPUs, including the 2-worker case. Inspect `mace_scripts/output/mace-off-*-<jobid>.{out,err}` and corresponding `mace_results/strong_scaling_off/*_<jobid>/strong_scaling_summary.json`. Static OFF24 writes `mace_results/off24_beta_<backend>_<jobid>/{reference.json,distributed.json}`. Submit only when Beta policy/queue and project budget permit.

## Open numerical question

`MACEScaleShiftRuntime._reduce_owned_energy` casts the owned float32 node energies to float64 and sums baseline and interaction energies separately. This improves scalar energy accumulation; it does not make the MACE forward/backward float64. It could change the derivative reduction path, so compare a pre-change and current adapter under the **same Beta image, checkpoint, XYZ, backend, partition count, tolerance, and node type**. Also compare e3nn vs CuEq and 2 vs 4 ranks, preserving raw per-atom force differences. A full graph edge multiset match does not establish identical edge order or all intermediate features. GB200-specific floating point differences are plausible but not proven; native reference and adapter on Beta both execute on GB200, so the architecture alone does not explain the discrepancy. Do not relax force thresholds or use production trajectory accuracy claims until isolated.

`git log -S 'owned_e0 = node_e0.reshape(-1)[:num_local].double()' -- dynamis_model/src/dynamis_model/nnp/mace.py` identifies `27c3bca` as the introducing commit. Its parent is a starting point for an isolated comparison, but inspect the full diff and keep both versions in separate checkouts: the commit may change more than accumulation.
