# MACE validation

`MACEModel` is validated with direct Python and `torchrun` commands. Run the
commands below from the `Dynamis` repository root in an environment where the
project packages and the optional MACE dependency are installed.

MPA remains the default model family and checkpoint. A Bash benchmark script
can pass the same model arguments to the standalone reference and distributed
driver:

```bash
MODEL_FAMILY="${MODEL_FAMILY:-mp}"
MODEL_ARGS=(--model-family "${MODEL_FAMILY}")
if [[ -n "${MODEL_NAME:-}" ]]; then
    MODEL_ARGS+=(--model "${MODEL_NAME}")
fi
```

Leave both variables unset for MPA. Set `MODEL_FAMILY=off` to select OFF23
medium. `MODEL_NAME` overrides the family-specific default when set; use
`MODEL_NAME=small` or `MODEL_NAME=large` for the other OFF23 checkpoints, and
give each model a separate reference artifact and JSON output path. Keep
`CUEQ_ARGS` identical in the reference and distributed commands: either an
empty array or `(--enable-cueq)`.

## OFF23 float32 validation

The adapter accepts ordinary `ScaleShiftMACE` checkpoints from `mace_off` and
uses the checkpoint's interaction blocks, readouts, atomic-number table, and
cutoff. Omitting `--model` selects `medium` for `--model-family off` and
`medium-mpa-0` for `--model-family mp`. The Dynamis graph and GPU Direct exchange
currently require float32; the public calculator reference is configured to
float32 as well. OFF23 medium has two interactions and one inter-layer feature
exchange. Specialized subclasses such as `PolarMACE` are rejected.

From PowerShell, generate an independent OFF23 medium reference and compare
distributed energy and forces for the nonperiodic C/H fixture:

```powershell
New-Item -ItemType Directory -Force .\tmp\mace_off_medium | Out-Null
python dynamis_model/tests/mace/mace_standalone_reference.py `
  --builtin-system natural_chain12 --model-family off `
  --output .\tmp\mace_off_medium\reference.pt
torchrun --standalone --nproc_per_node=2 `
  dynamis_model/tests/mace/test_mace_model.py run-distributed `
  --builtin-system natural_chain12 --model-family off `
  --reference-artifact .\tmp\mace_off_medium\reference.pt --strict-parity
```

The OFF strict-parity check requires both the explicit Dynamis-graph reference
and the independent public MACE calculator to match. Repeat the two commands
with `--model small` and `--model large`, keeping a separate reference artifact
for each checkpoint. The medium fixture also checks that two ranks own atoms,
both receive halo atoms, and the two-layer runtime has one feature exchange.

For a nonperiodic molecular trajectory, compare the two-rank adapter with a
one-rank public calculator run using the same 12-atom C/H fixture. This run uses
20 steps at 0.1 fs and does not apply the periodic RDF check:

```powershell
torchrun --standalone --nproc_per_node=2 `
  dynamis_model/tests/mace/run_distributed_trajectory.py `
  --path .\tmp\mace_off_trajectory --system organic `
  --model-family off
python dynamis_model/tests/mace/test_md.py `
  --path .\tmp\mace_off_trajectory --system organic `
  --model-family off --reference-mode monolithic
```

The existing MP/MPA static and trajectory commands below remain regression
checks for the shared runtime.

## Static parity

Generate an independent public-MACE reference artifact, then require the
two-rank Dynamis model to match it:

```bash
RESULT_DIR=./tmp/mace_static
mkdir -p "${RESULT_DIR}"

python dynamis_model/tests/mace/mace_standalone_reference.py \
  --model-family mp --model medium-mpa-0 \
  --output "${RESULT_DIR}/reference.pt"

torchrun --standalone --nproc_per_node=2 \
  dynamis_model/tests/mace/test_mace_model.py \
  run-distributed \
  --model-family mp --model medium-mpa-0 \
  --reference-artifact "${RESULT_DIR}/reference.pt" \
  --strict-parity \
  --output-json "${RESULT_DIR}/mace_summary.json"
```

Use `--xyz-path` to supply an XYZ/extxyz structure. The static driver also
supports `--builtin-system`, `--head-name`, and `--ghost-cutoff-buffer`.

## Static-driver controls

Use `--builtin-system` or `--xyz-path` to select a structure, and keep
`--model-family`, `--model`, and optional `--head-name` identical in both
commands. `--ghost-cutoff-buffer` controls the distributed halo margin.
`--benchmark` with its warmup, iteration, and sample flags records timing
statistics; it does not relax `--strict-parity`. `--time-initialize-buffers`
additionally profiles buffer setup. `--output-json PATH` writes the
corresponding summary.

The reference artifact signature includes positions, cell, PBC flags, and
model identity. Both commands in a reference/distributed pair must therefore
use the same structure and model arguments. Nonperiodic XYZ/extxyz inputs must
provide a valid cell containing their coordinates.

## Calculator/core reduction diagnostics

Regenerate the reference artifact with the current driver (artifact version 4).
It now stores the calculator's full per-atom `results["energies"]` and its
baseline-removed `results["node_energy"]`, their dtypes and float64 sums,
calculator/core per-atom errors, and directed calculator/core edge counts and
identities. The calculator graph is reconstructed through its own batch builder
outside the timed forward. The printed report groups configuration and
precision, energy definitions, matched parity, reduction decomposition,
per-atom and force parity, graph diagnostics, repeatability, and performance.
The JSON retains the original `parity` fields and adds `report_sections` with
named, unit-bearing metrics.

For a small molecular structure, add `--float64-diagnostic` to the standalone
reference command. It loads the original checkpoint in float64 through the
public MACE loader and compares calculator/core legacy scalars with float64
per-atom sums. It uses e3nn even when the main float32 reference uses CuEq.
The default limit is 1000 atoms; `--float64-max-atoms` can raise it explicitly.
Add `--repeatability-runs 5` to collect five untimed calculator evaluations of
the same structure, including scalar and controlled energy spreads and force
differences. These diagnostic evaluations do not enter benchmark timing.

```bash
python dynamis_model/tests/mace/mace_standalone_reference.py \
  --builtin-system natural_chain12 --model-family off --model medium \
  --enable-cueq --float64-diagnostic --repeatability-runs 5 \
  --output ./tmp/mace_off_diagnostics/reference.pt \
  --output-json ./tmp/mace_off_diagnostics/reference.json

torchrun --standalone --nproc-per-node=2 \
  dynamis_model/tests/mace/test_mace_model.py run-distributed \
  --builtin-system natural_chain12 --model-family off --model medium \
  --enable-cueq --energy-mode stable \
  --reference-artifact ./tmp/mace_off_diagnostics/reference.pt \
  --output-json ./tmp/mace_off_diagnostics/distributed.json
```

For the 15,000-atom structure, omit `--float64-diagnostic` or explicitly raise
its atom limit if resources permit. Use the same structure, family, model, head,
and CuEq settings in both commands.

## cuEquivariance

The static reference and distributed drivers both accept `--enable-cueq`.
When selected, the raw MACE model is converted with MACE's
`convert_e3nn_cueq` conversion before either path is timed or evaluated. The
flag is part of the reference-artifact signature, so a CuEq reference can only
be compared with a CuEq distributed run. Omitting the flag retains the e3nn
implementation.

Install mutually compatible versions of all four CuEq packages. The validated
cluster environment currently uses version `0.9.1` for
`cuequivariance`, `cuequivariance-torch`, `cuequivariance-ops-cu12`, and
`cuequivariance-ops-torch-cu12` with Python 3.12, PyTorch 2.8.0+cu129, and
MACE 0.3.16.

Install the validated CUDA 12 combination from the `dynamis_model` package
directory with `python -m pip install -e '.[mace-cueq]'`. The plain
`python -m pip install -e '.[mace]'` extra installs MACE without CuEq.

Generate a CuEq-enabled standalone reference and run the two-GPU benchmark:

```bash
RESULT_DIR=./tmp/mace_cueq_2gpu
XYZ_PATH=dynamis_model/tests/mace/data/benchmark_scaling_molecular_7500.xyz
mkdir -p "${RESULT_DIR}"

python dynamis_model/tests/mace/mace_standalone_reference.py \
  --xyz-path "${XYZ_PATH}" --model-family mp --model medium-mpa-0 \
  --enable-cueq --output "${RESULT_DIR}/reference.pt" \
  --benchmark --benchmark-warmup 10 --benchmark-iterations 50 \
  --benchmark-samples 7 --output-json "${RESULT_DIR}/reference.json"

torchrun --standalone --nproc_per_node=2 \
  dynamis_model/tests/mace/test_mace_model.py run-distributed \
  --xyz-path "${XYZ_PATH}" --model-family mp --model medium-mpa-0 \
  --enable-cueq --reference-artifact "${RESULT_DIR}/reference.pt" \
  --strict-parity --benchmark --benchmark-warmup 10 \
  --benchmark-iterations 50 --benchmark-samples 7 \
  --output-json "${RESULT_DIR}/distributed.json"
```

For a four-GPU strong-scaling run, generate the same standalone reference and
change only the distributed launch to `--nproc_per_node=4`. Keep the XYZ,
model, head, dtype, CuEq setting, warmup, iterations, and sample count fixed
when comparing rank counts. The 7,500-atom fixture is preferred over the
3,750-atom fixture for two-versus-four-GPU scaling because the smaller system
does not provide enough per-rank compute to amortize halo and synchronization
costs.

The distributed parity report also compares the explicit Dynamis graph with
the MACE-native core graph. `edge_comparison` preserves directed edge identities,
periodic shifts, and duplicate counts. Its `explicit_only` and `core_only`
sections report unmatched-edge distances, including how many lie within
1e-4 Angstrom of the checkpoint cutoff. `explicit_vs_core_node_energy_mae`
and `explicit_vs_core_node_energy_max_abs` measure the effect on individual
atomic predictions without comparing the two different total-energy reductions.
Regenerate the standalone reference artifact to populate these fields; older
artifacts remain readable but do not include the edge comparison.

To test the opt-in stable energy reduction, generate a fresh standalone
reference with the same family/model/XYZ arguments, then pass `--energy-mode
stable` to `test_mace_model.py run-distributed`. The original MACE scalar
remains `explicit_dynamis_graph_energy`; `explicit_stable_energy` separately
reduces the same forward's atomic baseline and interaction contributions in
float64. Compare `distributed_minus_explicit_stable_energy` and
`explicit_stable_minus_node_sum` with the legacy
`distributed_minus_explicit_energy` and `explicit_node_sum_minus_model_energy`.
`--strict-parity` uses the controlled reference in stable mode, with an
absolute MD-energy tolerance set by `--stable-energy-atol` (default `1e-3`).
The runtime still computes features and forces in float32; only scalar energy
accumulation is widened. `--energy-mode legacy` remains the default.

## OFF graph edge audit

Use `--graph-edge-test` on the distributed static driver to reconstruct each
rank's owned-destination graph with global atom IDs. The report compares the
assembled directed edges and integer cell shifts with the core, calculator,
and explicit Dynamis graphs. Each comparison separates edges more than
`1e-4` Angstrom inside the checkpoint cutoff, near the cutoff, and outside.
`--strict-graph-interior` enables the audit and asserts that the assembled
graph and native core agree for strictly interior edges. The audit also compares owned per-atom
energies against all three references; existing force parity remains in the
same report. The extra graph build and ID exchange occur outside benchmark
timing. Generate a fresh standalone artifact for every XYZ, checkpoint, and
backend selection.

`data/off_cutoff_boundary.xyz` has probe pairs at 4.999, 5.000, and
5.001 Angstrom plus a bonded fragment that produces nonzero interactions and
forces. `data/off_organic_irregular.xyz` contains two nonperiodic
ethanol conformers. For the large synthetic lattice, create a deterministic
nonperiodic jittered XYZ before generating either reference:

```bash
python dynamis_model/tests/mace/jitter_mace_xyz.py \
  --input mace_data/benchmark_scaling_molecular_15000.xyz \
  --output tmp/off_jittered_15000.xyz --seed 2026 --amplitude 0.01
```

The generator checks float32 coordinates and resamples only atoms involved in
cutoff-boundary pairs until no pair is within `1e-3` Angstrom of 5 Angstrom.
It preserves atom species and
the XYZ cell/PBC comment; it is intended for nonperiodic input. Run the same
static reference and distributed audit with `--model-family off --model
small`, `medium`, and `large`; pass `--enable-cueq` to both commands for the
CuEq variant.

## Distributed trajectory

Run the two-rank MACEModel trajectory, then compare it with a one-rank public
MACE calculator trajectory:

```bash
RESULT_DIR=./tmp/mace_trajectory
mkdir -p "${RESULT_DIR}"

torchrun --standalone --nproc_per_node=2 \
  dynamis_model/tests/mace/run_distributed_trajectory.py \
  --path "${RESULT_DIR}" --model-family mp --model medium-mpa-0

python dynamis_model/tests/mace/test_md.py \
  --path "${RESULT_DIR}" --model-family mp --model medium-mpa-0 \
  --reference-mode monolithic
```

The acceptance trajectory is a deterministic 216-atom periodic diamond-Si
system with 100 steps, a 0.5 fs timestep, and rebalancing every seven steps.
The generated artifact records model identity, structure hash, rank count, and
force-unit convention before numerical comparison begins. These commands use
the default e3nn implementation. To test the CuEq path, append
`--enable-cueq` to both commands:

```bash
RESULT_DIR=./tmp/mace_trajectory_cueq
mkdir -p "${RESULT_DIR}"

torchrun --standalone --nproc_per_node=2 \
  dynamis_model/tests/mace/run_distributed_trajectory.py \
  --path "${RESULT_DIR}" --model-family mp --model medium-mpa-0 \
  --enable-cueq

python dynamis_model/tests/mace/test_md.py \
  --path "${RESULT_DIR}" --model-family mp --model medium-mpa-0 \
  --reference-mode monolithic --enable-cueq
```

The CuEq setting is stored in the trajectory artifact metadata. A reference
command using a different setting is rejected before trajectory comparison.

## Benchmark collection

Benchmarking is opt-in and does not change parity acceptance. Add
`--benchmark --benchmark-warmup 5 --benchmark-iterations 20
--benchmark-samples 7` to both static-parity commands. Use the same model and
structure on both sides of the comparison.

The uninterrupted `distributed_median_ms_per_step` is the primary end-to-end
measurement and uses the slowest rank. The static driver also records a
diagnostic per-rank decomposition into structural/position exchange, graph
construction, MACE forward/feature exchange, and force recovery/backward
exchange. These diagnostic stages synchronize the device after every boundary
and therefore should be used to attribute costs, not substituted for the
uninterrupted result. Workload fields include owned and halo atoms, graph-edge
counts, rows exchanged per peer, feature width, and logical feature payload
bytes.

## Coverage boundary

Static parity anchors `MACEModel` to public MACE calculator and raw-core model
values. Trajectory validation covers distributed partitioning, halo and feature
exchange, rank-local energy aggregation, repartitioning, and Verlet
integration against monolithic MACE.

`MACEModel` returns Dynamis MD units — energy in `amu*A^2/fs^2` and forces in
`amu*A/fs^2` — matching `UMAModel` and `ORBModel`, so it drives `Simulator`
directly with no wrapper. Public MACE predicts eV and eV/Angstrom, so every
reference in this directory is scaled by
`dynamis_model.units.EV_PER_A_TO_AMU_A_PER_FS2` before it is compared. That
constant is defined once and imported by all three backends; do not re-spell
the literal.
