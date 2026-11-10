# MACE adapter validation

`MACEModel` runs ordinary `ScaleShiftMACE` checkpoints in float32 and reports a
controlled energy total. It widens each owned atom's baseline and interaction
contribution to float64, sums those contributions separately, then converts the
energy and forces to Dynamis MD units. MPA (`mp/medium-mpa-0`) is the default;
`--model-family off` selects OFF23 medium unless `--model` selects small or large.
The checkpoint supplies its cutoff, species table, interactions, and readouts.

The acceptance reference is the **raw MACE model** on a MACE-native graph. Its
selected-head per-atom baseline and interaction energies are summed in float64.
The public MACE foundation loader is used to obtain the raw checkpoint; these
tests do not construct or evaluate an ASE calculator. The native model's
float32 scalar energy is not the parity reference.

## Static energy, forces, and graph

Run from the repository root with MACE dependencies and two CUDA devices:

```bash
mkdir -p tmp/mace_mp
python dynamis_model/tests/mace/mace_standalone_reference.py \
  --output tmp/mace_mp/reference.pt
torchrun --standalone --nproc-per-node=2 \
  dynamis_model/tests/mace/test_mace_model.py run-distributed \
  --reference-artifact tmp/mace_mp/reference.pt \
  --strict-parity --strict-graph-interior \
  --output-json tmp/mace_mp/distributed.json
```

For OFF23, pass identical model arguments to both commands. Use a distinct
artifact path for each checkpoint and backend:

```bash
mkdir -p tmp/mace_off_medium
python dynamis_model/tests/mace/mace_standalone_reference.py \
  --model-family off --model medium \
  --output tmp/mace_off_medium/reference.pt
torchrun --standalone --nproc-per-node=2 \
  dynamis_model/tests/mace/test_mace_model.py run-distributed \
  --model-family off --model medium \
  --reference-artifact tmp/mace_off_medium/reference.pt \
  --strict-parity --strict-graph-interior \
  --output-json tmp/mace_off_medium/distributed.json
```

Repeat with `--model small` and `--model large`. To validate CuEq, add
`--enable-cueq` to **both** commands. `--xyz-path FILE` selects the same
XYZ/extxyz structure in both commands; otherwise the nonperiodic 12-atom C/H
fixture is used. The OFF23 medium fixture checks that both ranks own and halo
atoms and that its two interactions require one feature exchange. The graph
audit compares assembled owned directed edges with the native graph and fails
on any difference more than `1e-4` Å inside the checkpoint cutoff. Exact cutoff
boundary edges are reported separately.

Static acceptance uses `1e-3` MD energy units absolute error and `1e-5`
absolute plus relative force tolerances. `--benchmark` on both commands adds
lean raw-core and distributed step timing; it does not relax parity.

## Trajectory

The two-rank trajectory is compared with a one-rank raw checkpoint that rebuilds
the native MACE graph every step. The organic fixture has 12 C/H atoms, is
nonperiodic, and runs 20 Verlet steps at 0.1 fs. The comparison checks energy,
atom IDs, positions, velocities, and energy drift.

```bash
torchrun --standalone --nproc-per-node=2 \
  dynamis_model/tests/mace/run_distributed_trajectory.py \
  --path tmp/mace_off_md --system organic --model-family off --model medium
python dynamis_model/tests/mace/test_md.py \
  --path tmp/mace_off_md --system organic --model-family off --model medium
```

Repeat with small and large, each in its own `--path` directory. Add
`--enable-cueq` to both commands for CuEq. The trajectory artifact records atom
ownership by rank; the static halo fixture remains the mandatory cross-rank
feature-exchange check.

## Scope

Only ordinary `ScaleShiftMACE` checkpoints are accepted. Specialized subclasses
such as `PolarMACE` are rejected. The current graph and GPU Direct exchange
implementations use float32; end-to-end float64 execution is outside this
adapter's supported scope. Old diagnostic reference artifacts are incompatible
with the production artifact version and must be regenerated.
