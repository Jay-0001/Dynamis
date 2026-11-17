# MACE adapter validation

`MACEModel` runs ordinary `ScaleShiftMACE` checkpoints in float32 and reports a
controlled energy total. It widens each owned atom's baseline and interaction
contribution to float64, sums those contributions separately, then converts the
energy and forces to Dynamis MD units. MPA (`mp/medium-mpa-0`) is the default;
`--model-family off` selects OFF23 medium unless `--model` selects small or large.
The checkpoint supplies its cutoff, species table, interactions, and readouts.
The OFF24 medium checkpoint is available through the same `off` family with an
explicit checkpoint URL or local path; the OFF23 default is unchanged.

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
`--enable-cueq` to **both** commands. Both commands generate the same
nonperiodic 12-atom C/H molecule in memory. Set the same `--system-seed` and
`--molecule-count` on each command to select another deterministic structure.
For example, `--molecule-count 625` creates 7,500 atoms for a scaling run.
Each molecule stays clear of the 5 and 6 Å cutoff boundaries, and all atoms
lie inside a positive cell with more than 6 Å of margin. The OFF23 medium
single-molecule fixture checks that both ranks own and halo
atoms and that its two interactions require one feature exchange. The graph
audit compares assembled owned directed edges with the native graph and fails
on any difference more than `1e-4` Å inside the checkpoint cutoff. Exact cutoff
boundary edges are reported separately.

### OFF24 medium validation

`data/off24_bridge.xyz` contains two nonperiodic C/H fragments inside a positive
36 Å cell. It includes cross-fragment distances between 5 and 6 Å and beyond
6 Å, with no pair on the 6 Å boundary. The static test compares this fixture
and a deterministic 50-copy, 1,200-atom repeat against native MACE on both
e3nn and CuEq. The large case also records raw-core and distributed timings.
The trajectory test runs 20 Verlet steps on the 24-atom XYZ fixture, comparing
the two-rank adapter with a one-rank raw checkpoint.

From the repository root on a two-GPU Delta allocation, prepare the Slurm log
directory and submit the two independent gates:

```bash
mkdir -p mace_scripts/outputs
sbatch dynamis_model/tests/mace/off24_static.sbatch
sbatch dynamis_model/tests/mace/off24_trajectory.sbatch
```

The scripts load the official OFF24 medium checkpoint with `--model-family off`
and an explicit `--model` URL to `mace_off24/MACE-OFF24_medium.model`.
Each script caches the checkpoint before the two distributed ranks load it.
The native loader currently prints "MACE-OFF23" even when this OFF24
checkpoint is selected; inspect the checkpoint path and reported 6 Å cutoff.
Static acceptance requires the controlled energy and force tolerances below
and exact directed-edge agreement for these boundary-clear fixtures. Trajectory
acceptance uses the existing position, velocity, atom-ID, energy, and drift
checks. The smoke result alone does not establish trajectory support.

Static acceptance uses `1e-3` MD energy units absolute error and `1e-5`
absolute plus relative force tolerances. `--benchmark` on both commands adds
lean raw-core and distributed step timing; it does not relax parity.

## Trajectory

The two-rank trajectory is compared with a one-rank raw checkpoint that rebuilds
the native MACE graph every step. Both runs generate the same organic structure
in memory and pass it to the simulator with nonperiodic graph settings. The
organic fixture has 12 C/H atoms and runs 20 Verlet steps at 0.1 fs. The comparison checks energy,
atom IDs, positions, velocities, and energy drift.

```bash
torchrun --standalone --nproc-per-node=2 \
  dynamis_model/tests/mace/run_distributed_trajectory.py \
  --path tmp/mace_off_md --system organic --model-family off --model medium
python dynamis_model/tests/mace/test_md.py \
  --path tmp/mace_off_md --system organic --model-family off --model medium
```

Repeat with small and large, each in its own `--path` directory. Add
`--enable-cueq` to both commands for CuEq. Pass the same `--system-seed` to
both commands to vary the molecular geometry. The trajectory artifact records atom
ownership by rank; the static halo fixture remains the mandatory cross-rank
feature-exchange check.

## Scope

Only ordinary `ScaleShiftMACE` checkpoints are accepted. Specialized subclasses
such as `PolarMACE` are rejected. The current graph and GPU Direct exchange
implementations use float32; end-to-end float64 execution is outside this
adapter's supported scope. Old diagnostic reference artifacts are incompatible
with the production artifact version and must be regenerated.
