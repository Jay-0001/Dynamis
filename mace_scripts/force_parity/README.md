# Delta–Beta MACE force parity investigation

Run these scripts from the root of a clone of `beta-force-parity-investigation`
that has its Dynamis extensions built in the cluster's Python environment.
The native MACE reference and every adapter variant in a job use the same
model name, generated coordinates, backend, and installed packages. The
current adapter is the default production implementation. `legacy` replaces
only the owned-energy reduction in the parity harness with the pre-OFF
float32 add-then-sum computation from the earlier `main` adapter. Loader
defaults and model-family contract checks are held fixed with an explicit
checkpoint name. This isolates the force-relevant adapter change; it is not a
rebuild of every historical commit.

## First: matching 12-atom MPA runs

Create log directories *before* submission. Beta uses the ARM64 PyTorch
25.08 container and four allocated GB200 GPUs. Delta uses the existing A100
partition and conda environment. The Beta job runs 2 and 4 active ranks; the
Delta job runs 2. Both default to MPA medium, e3nn and CuEq, 12 atoms,
seed 1234, and legacy and current reductions.

```bash
cd ~/Dynamis
mkdir -p mace_scripts/output mace_scripts/outputs mace_results/force_parity
sbatch mace_scripts/force_parity/beta_force_parity.sbatch   # on Beta
sbatch mace_scripts/force_parity/delta_force_parity.sbatch  # on Delta
```

Each job writes `mace_results/force_parity/<cluster>_<jobid>/summary.json`,
`provenance.json`, and a directory for each case/backend/rank/reduction.
`reference.pt` has native per-atom forces; `forces.pt` has aligned native,
distributed, and error tensors. Parity failure is recorded as data, so a job
can finish successfully while its summary reports `force_close: false`.
Runtime failures still stop the job and leave the individual `run.log`.

Copy one completed result directory from each cluster to the same filesystem,
then compare the common 2-rank cases:

```bash
python mace_scripts/force_parity/compare_clusters.py \
  mace_results/force_parity/delta_<jobid> \
  mace_results/force_parity/beta_<jobid> > delta_beta_comparison.json
```

First inspect `same_reference_signature` and cached checkpoint SHA-256. If
the signatures match but native reference energies or forces differ, check
checkpoint bytes and PyTorch/CuEq versions before attributing adapter errors
to partitioning. `provenance.json` records the Git commit and working-tree
status, GPU, packages, Python/CUDA versions, and cached MACE file hashes.
Run a new job after any code or environment change; do not mix artifacts
between runs.

## Next: zero-halo and OFF24 checks

Use 64 copies of the deterministic 12-atom molecule for 768 atoms. The prior
four-rank Beta run had zero halos here. Run the same count on Delta to make
the comparison closer; the matrix records actual halo counts per rank.

```bash
sbatch --export=ALL,FORCE_COUNTS=64 mace_scripts/force_parity/beta_force_parity.sbatch
sbatch --export=ALL,FORCE_COUNTS=64 mace_scripts/force_parity/delta_force_parity.sbatch
```

For the missing static OFF24 comparison, use the bridge fixture and the
explicit OFF24 URL selected by `FORCE_MODEL_FAMILY=off`. The script defaults
to the generated structure, or accepts `FORCE_XYZ_PATH` for a fixed XYZ.
These commands use the generated 12-atom structure on both clusters:

```bash
sbatch --export=ALL,FORCE_MODEL_FAMILY=off mace_scripts/force_parity/beta_force_parity.sbatch
sbatch --export=ALL,FORCE_MODEL_FAMILY=off mace_scripts/force_parity/delta_force_parity.sbatch
```

For the exact bridge XYZ, export `FORCE_XYZ_PATH` in the submitting shell
with the path valid on that cluster, then submit with `sbatch --export=ALL`.
Use the same file bytes on Delta and Beta. To run a single backend, set
`FORCE_BACKENDS=e3nn` or `FORCE_BACKENDS=cueq`. `FORCE_RANKS` and
`FORCE_REDUCTIONS` accept comma-separated values when exported by the shell.
The harness currently requires at least two ranks.

## Reading the result

If e3nn passes and CuEq fails on Beta for the same case, inspect CuEq
conversion and backward kernels. If both fail at two ranks but pass with
`legacy`, inspect the owned-energy reduction. If legacy and current forces
agree yet both fail, follow edge ordering, per-layer features, and gradients
through the adapter. A failure on a zero-halo fixture cannot be explained by
halo exchange alone. Edge multiset equality does not establish identical
edge order or floating-point accumulation. Do not change parity tolerances
while locating the first divergent intermediate.
