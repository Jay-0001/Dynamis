"""Detached standalone MACE reference writer for MACEModel benchmarks.

This driver avoids Dynamis partitioning and Dynamis graph construction. It
records two monolithic baselines:

1. the public calculator path via ``mace.calculators``
2. a raw ``ScaleShiftMACE`` path with MACE-native graph creation in this file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from dynamis_model.nnp.mace import (
    COMPUTE_DTYPE,
    load_raw_mace_model,
    resolve_mace_model_name,
    resolve_model_head_name,
)

try:
    from ase import Atoms
    from ase.calculators.calculator import all_changes
    from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS
    from mace.calculators import mace_mp, mace_off, mace_omol, mace_polar
    from mace.data import AtomicData, Configuration
    from mace.tools import AtomicNumberTable, torch_tools
    from mace.tools.torch_geometric.batch import Batch

    HAS_STANDALONE_MACE = True
except Exception as exc:  # pragma: no cover - optional dependency
    Atoms = None  # type: ignore[assignment]
    all_changes = None  # type: ignore[assignment]
    ASE_ATOMIC_NUMBERS = {}
    mace_mp = mace_off = mace_omol = mace_polar = None
    AtomicData = Configuration = AtomicNumberTable = Batch = None  # type: ignore[assignment]
    HAS_STANDALONE_MACE = False
    _STANDALONE_IMPORT_ERROR = exc
else:
    _STANDALONE_IMPORT_ERROR = None

ARTIFACT_VERSION = 4
TEST_FILE_DIR = Path(__file__).resolve().parent
ELEMENT_TO_Z = dict(ASE_ATOMIC_NUMBERS)


def _require_standalone_mace() -> None:
    if not HAS_STANDALONE_MACE:
        raise ImportError(
            "Standalone MACE reference generation requires ASE and MACE."
        ) from _STANDALONE_IMPORT_ERROR


def natural_chain12() -> Dict[str, Any]:
    carbon = [
        [2.00, 2.00, 2.00],
        [3.52, 2.12, 2.00],
        [5.01, 1.90, 2.18],
        [6.50, 2.08, 1.95],
    ]
    hydrogen = [
        [1.45, 2.92, 2.20],
        [1.52, 1.08, 1.80],
        [3.52, 3.08, 1.55],
        [3.55, 1.22, 2.62],
        [5.00, 2.82, 2.72],
        [5.02, 0.95, 1.58],
        [7.03, 2.96, 2.20],
        [7.02, 1.18, 1.72],
    ]
    return {
        "name": "natural_chain12",
        "atomic_numbers": [6] * 4 + [1] * 8,
        "positions": carbon + hydrogen,
        "cell": [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]],
        "pbc": [False, False, False],
    }


BUILTIN_SYSTEMS = {
    "natural_chain12": natural_chain12,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--builtin-system", choices=BUILTIN_SYSTEMS, default="natural_chain12"
    )
    parser.add_argument(
        "--xyz-path", type=Path, help="XYZ/extxyz input; overrides --builtin-system."
    )
    parser.add_argument(
        "--model-family", choices=["mp", "off", "omol", "polar"], default="mp"
    )
    parser.add_argument("--model", default=None, help="Checkpoint name (defaults by family).")
    parser.add_argument(
        "--head-name",
        default="",
        help="Optional explicit MACE head name for metadata validation.",
    )
    parser.add_argument(
        "--enable-cueq",
        action="store_true",
        help="Enable cuEquivariance for both the public calculator and raw-core reference paths.",
    )
    parser.add_argument("--ghost-cutoff-buffer", type=float, default=0.5)
    parser.add_argument(
        "--output", type=Path, required=True, help="Reference artifact path."
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Record standalone timing stats."
    )
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    parser.add_argument("--benchmark-iterations", type=int, default=10)
    parser.add_argument("--benchmark-samples", type=int, default=5)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--float64-diagnostic", action="store_true",
                        help="Compare float64 calculator/core scalar and per-atom totals outside benchmark timing.")
    parser.add_argument("--float64-max-atoms", type=int, default=1000)
    parser.add_argument("--repeatability-runs", type=int, default=0,
                        help="Number of untimed repeated calculator evaluations (0 disables).")
    return parser


def resolve_xyz_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [Path.cwd() / path, TEST_FILE_DIR / path]
        candidates.extend(parent / path for parent in TEST_FILE_DIR.parents)
        if len(path.parts) == 1:
            candidates.append(TEST_FILE_DIR / "data" / path)
    attempted: List[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        attempted.append(resolved)
        if resolved.is_file():
            return resolved
    attempted_text = "\n".join(f"  - {candidate}" for candidate in attempted)
    raise FileNotFoundError(
        f"XYZ input does not exist: {path}\n"
        f"working directory: {Path.cwd()}\n"
        f"Attempted paths:\n{attempted_text}"
    )


def load_xyz(path: Path) -> Dict[str, Any]:
    path = resolve_xyz_path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file is incomplete: {path}")
    try:
        num_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"XYZ first line must be an atom count: {path}") from exc
    atom_lines = [line for line in lines[2:] if line.strip()]
    if len(atom_lines) != num_atoms:
        raise ValueError(
            f"XYZ declares {num_atoms} atoms but contains {len(atom_lines)} rows."
        )

    atomic_numbers: List[int] = []
    positions: List[List[float]] = []
    for row_index, line in enumerate(atom_lines, start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(
                f"XYZ atom row {row_index} needs symbol and x/y/z coordinates."
            )
        token = fields[0]
        if token.isdigit():
            atomic_number = int(token)
        elif token in ELEMENT_TO_Z:
            atomic_number = ELEMENT_TO_Z[token]
        else:
            raise ValueError(
                f"Unsupported element symbol {token!r} on XYZ row {row_index}."
            )
        atomic_numbers.append(atomic_number)
        positions.append([float(fields[1]), float(fields[2]), float(fields[3])])

    comment = lines[1]
    lattice_match = re.search(r'Lattice="([^"]+)"', comment)
    pbc_match = re.search(r'pbc="([^"]+)"', comment, flags=re.IGNORECASE)
    if lattice_match:
        lattice = [float(value) for value in lattice_match.group(1).split()]
        if len(lattice) != 9:
            raise ValueError("extxyz Lattice must contain nine values.")
        cell = [lattice[0:3], lattice[3:6], lattice[6:9]]
    else:
        coordinate_tensor = torch.tensor(positions, dtype=COMPUTE_DTYPE)
        span = coordinate_tensor.max(dim=0).values - coordinate_tensor.min(dim=0).values
        lengths = torch.maximum(span + 20.0, torch.full_like(span, 20.0))
        cell = torch.diag(lengths).tolist()
    if pbc_match:
        tokens = pbc_match.group(1).split()
        if len(tokens) != 3:
            raise ValueError("extxyz pbc must contain three values.")
        pbc = [token.lower() in {"t", "true", "1"} for token in tokens]
    else:
        pbc = [False, False, False]
    return {
        "name": path.stem,
        "source_path": str(path.resolve()),
        "atomic_numbers": atomic_numbers,
        "positions": positions,
        "cell": cell,
        "pbc": pbc,
    }


def load_raw_structure(args: argparse.Namespace) -> Dict[str, Any]:
    return (
        load_xyz(args.xyz_path)
        if args.xyz_path
        else BUILTIN_SYSTEMS[args.builtin_system]()
    )


def raw_to_atoms(raw: Dict[str, Any]) -> Atoms:
    _require_standalone_mace()
    return Atoms(
        numbers=raw["atomic_numbers"],
        positions=raw["positions"],
        cell=raw["cell"],
        pbc=raw["pbc"],
    )


def load_mace_calculator(
    model_family: str, model_name: str, device: str, enable_cueq: bool = False,
    head_name: Optional[str] = None, default_dtype: str = "float32",
):
    _require_standalone_mace()
    loaders = {"mp": mace_mp, "off": mace_off, "omol": mace_omol, "polar": mace_polar}
    if model_family not in loaders:
        raise ValueError(f"Unsupported MACE model family: {model_family}")
    kwargs = {"head": head_name} if head_name is not None else {}
    return loaders[model_family](
        model=model_name,
        device=device,
        default_dtype=default_dtype,
        enable_cueq=enable_cueq,
        **kwargs,
    )


def evaluate_with_calculator(
    calculator, atoms: Atoms, collect_precision: bool = False,
    collect_details: bool = False, collect_graph: bool = True,
) -> Dict[str, Any]:
    _require_standalone_mace()
    calculator.calculate(
        atoms, properties=["energy", "forces"], system_changes=all_changes
    )
    energy_result = calculator.results["energy"]
    force_result = calculator.results["forces"]
    result = {
        "energy": float(energy_result),
        "forces": torch.as_tensor(
            force_result, dtype=COMPUTE_DTYPE
        ).cpu(),
    }
    if collect_details:
        full = np.asarray(calculator.results["energies"])
        interaction = np.asarray(calculator.results["node_energy"])
        if full.shape != (len(atoms),) or interaction.shape != full.shape:
            raise ValueError("Calculator per-atom energies have unexpected shape.")
        # MACECalculator copies the full node energy into `energies`, then
        # removes E0 from `node_energy`. Preserve the original NumPy dtypes.
        result.update(
            node_energy_values=torch.from_numpy(full.copy()).cpu(),
            interaction_node_energy_values=torch.from_numpy(interaction.copy()).cpu(),
            node_energy_dtype=str(full.dtype),
            interaction_node_energy_dtype=str(interaction.dtype),
            stable_energy=float(full.astype(np.float64).sum(dtype=np.float64)),
            stable_interaction_energy=float(interaction.astype(np.float64).sum(dtype=np.float64)),
        )
        result["stable_baseline_energy"] = (
            result["stable_energy"] - result["stable_interaction_energy"]
        )
        # Rebuild the calculator's graph through its own batch constructor.
        # This is a diagnostic reconstruction, separate from the timed forward.
        if collect_graph:
            batch = calculator._atoms_to_batch(atoms)
            rows = torch.cat(
                (batch["edge_index"].T.long(), batch["unit_shifts"].round().long()),
                dim=1,
            ).to(dtype=torch.int32, device="cpu")
            result["edge_rows"] = rows
            result["edge_count"] = int(rows.shape[0])
    if collect_precision:
        result["precision"] = {
            "requested_dtype": str(calculator.default_dtype),
            "model_parameter_dtype": str(next(calculator.models[0].parameters()).dtype),
            "energy_result_type": type(energy_result).__name__,
            "energy_float32_roundtrip_delta_ev": (
                float(energy_result) - float(np.float32(energy_result))
            ),
            "force_result_dtype": str(np.asarray(force_result).dtype),
            "full_node_energy_result_dtype": str(
                np.asarray(calculator.results["energies"]).dtype
            ),
            "interaction_node_energy_result_dtype": str(
                np.asarray(calculator.results["node_energy"]).dtype
            ),
        }
    return result


def _align_batch_dtype(
    batch_dict: Dict[str, torch.Tensor], model: torch.nn.Module
) -> Dict[str, torch.Tensor]:
    model_dtype = next(model.parameters()).dtype
    aligned: Dict[str, torch.Tensor] = {}
    for key, value in batch_dict.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            aligned[key] = value.to(dtype=model_dtype)
        else:
            aligned[key] = value
    return aligned


def build_core_model_batch(
    raw: Dict[str, Any],
    model: torch.nn.Module,
    cutoff: float,
    device: torch.device,
    head_name: Optional[str] = None,
) -> Tuple[Dict[str, torch.Tensor], int]:
    _require_standalone_mace()
    resolved_head = resolve_model_head_name(model, head_name)
    config = Configuration(
        atomic_numbers=np.asarray(raw["atomic_numbers"], dtype=np.int64),
        positions=np.asarray(
            raw["positions"],
            dtype=np.float64 if next(model.parameters()).dtype == torch.float64 else np.float32,
        ),
        properties={},
        property_weights={},
        cell=np.asarray(
            raw["cell"],
            dtype=np.float64 if next(model.parameters()).dtype == torch.float64 else np.float32,
        ),
        pbc=tuple(raw["pbc"]),
        weight=1.0,
        config_type="dynamis_mace_model_core_model_reference",
        head=resolved_head,
    )
    model_heads = list(getattr(model, "heads", []))
    z_table = AtomicNumberTable(
        [int(value) for value in model.atomic_numbers.detach().cpu().tolist()]
    )
    with torch_tools.default_dtype(
        "float64" if next(model.parameters()).dtype == torch.float64 else "float32"
    ):
        atomic_data = AtomicData.from_config(
            config=config,
            z_table=z_table,
            cutoff=cutoff,
            heads=model_heads or [resolved_head],
        )
    edge_count = int(atomic_data.edge_index.shape[1])
    batch = Batch.from_data_list([atomic_data]).to(device).to_dict()
    return _align_batch_dtype(batch, model), edge_count


def evaluate_with_core_model(
    raw: Dict[str, Any],
    model: torch.nn.Module,
    cutoff: float,
    device: torch.device,
    head_name: Optional[str] = None,
) -> Dict[str, Any]:
    batch, edge_count = build_core_model_batch(
        raw, model, cutoff, device, head_name=head_name
    )
    out = model(
        batch,
        training=False,
        compute_force=True,
        compute_stress=False,
        compute_hessian=False,
    )
    energy_tensor = out["energy"].detach()
    node_energy_tensor = out["node_energy"].detach().reshape(-1)
    head_index = int(batch["head"].reshape(()).item())
    node_e0 = model.atomic_energies_fn(batch["node_attrs"])[
        :, head_index
    ].detach().double()
    stable_baseline = node_e0.sum()
    stable_interaction = (node_energy_tensor.double() - node_e0).sum()
    stable_energy = stable_baseline + stable_interaction
    # Preserve directed edge identities for comparison with Dynamis's graph.
    # A count alone cannot distinguish cutoff-boundary edges from missing
    # neighbors, reversed edges, or duplicate records.
    edge_rows = torch.cat(
        (batch["edge_index"].T, batch["unit_shifts"].round().long()), dim=1
    ).to(dtype=torch.int32, device="cpu")
    return {
        "energy": float(energy_tensor.cpu().reshape(())),
        "node_energy_sum_float64": float(node_energy_tensor.double().sum().cpu()),
        "node_energy_sum_float32": float(node_energy_tensor.float().sum().cpu()),
        "stable_energy": float(stable_energy.cpu()),
        "stable_baseline_energy": float(stable_baseline.cpu()),
        "stable_interaction_energy": float(stable_interaction.cpu()),
        "node_energy_values": node_energy_tensor.double().cpu(),
        "forces": out["forces"].detach().cpu(),
        "edge_count": edge_count,
        "edge_rows": edge_rows,
        "precision": {
            "model_parameter_dtype": str(next(model.parameters()).dtype),
            "input_position_dtype": str(batch["positions"].dtype),
            "energy_tensor_dtype": str(energy_tensor.dtype),
            "node_energy_tensor_dtype": str(out["node_energy"].dtype),
            "interaction_energy_tensor_dtype": str(out["interaction_energy"].dtype),
            "force_tensor_dtype": str(out["forces"].dtype),
            "energy_float32_roundtrip_delta_ev": (
                float(energy_tensor.cpu().reshape(()))
                - float(energy_tensor.to(torch.float32).cpu().reshape(()))
            ),
        },
    }


def compare_calculator_core(calculator: Dict[str, Any], core: Dict[str, Any]) -> Dict[str, Any]:
    """Compare independently evaluated MACE paths in eV and eV/Angstrom."""
    calc_nodes = calculator["node_energy_values"].double().reshape(-1)
    core_nodes = core["node_energy_values"].double().reshape(-1)
    if calc_nodes.shape != core_nodes.shape:
        raise ValueError("Calculator and core per-atom energy shapes differ.")
    delta = calc_nodes - core_nodes
    calc_edges = calculator["edge_rows"].long()
    core_edges = core["edge_rows"].long()
    from collections import Counter
    calc_counts = Counter(map(tuple, calc_edges.tolist()))
    core_counts = Counter(map(tuple, core_edges.tolist()))
    force_delta = calculator["forces"].double() - core["forces"].double()
    return {
        "calculator_minus_core_legacy_energy_ev": calculator["energy"] - core["energy"],
        "calculator_minus_core_controlled_energy_ev": (
            calculator["stable_energy"] - core["stable_energy"]
        ),
        "calculator_minus_core_node_energy_sum_ev": float(delta.sum()),
        "calculator_vs_core_node_energy_mae_ev": float(delta.abs().mean()),
        "calculator_vs_core_node_energy_max_abs_ev": float(delta.abs().max()),
        "calculator_vs_core_force_mae_ev_per_angstrom": float(force_delta.abs().mean()),
        "calculator_vs_core_force_max_abs_ev_per_angstrom": float(force_delta.abs().max()),
        "calculator_graph_edge_count": int(calc_edges.shape[0]),
        "core_graph_edge_count": int(core_edges.shape[0]),
        "calculator_only_edge_count": sum((calc_counts - core_counts).values()),
        "core_only_edge_count": sum((core_counts - calc_counts).values()),
    }


def repeat_calculator(calculator, atoms: Atoms, runs: int) -> Dict[str, Any]:
    if runs < 2:
        raise ValueError("--repeatability-runs must be 0 or at least 2.")
    records = []
    first_forces = None
    for index in range(runs):
        result = evaluate_with_calculator(
            calculator, atoms, collect_details=True, collect_graph=False
        )
        forces = result["forces"].double()
        if first_forces is None:
            first_forces = forces
        delta = forces - first_forces
        records.append({
            "run": index + 1,
            "calculator_legacy_energy_ev": result["energy"],
            "calculator_controlled_energy_ev": result["stable_energy"],
            "calculator_force_max_abs_ev_per_angstrom": float(forces.abs().max()),
            "force_vs_first_mae_ev_per_angstrom": float(delta.abs().mean()),
            "force_vs_first_max_abs_ev_per_angstrom": float(delta.abs().max()),
        })
    legacy = [item["calculator_legacy_energy_ev"] for item in records]
    controlled = [item["calculator_controlled_energy_ev"] for item in records]
    return {
        "runs": records,
        "calculator_legacy_energy_spread_ev": max(legacy) - min(legacy),
        "calculator_controlled_energy_spread_ev": max(controlled) - min(controlled),
        "calculator_force_max_abs_spread_ev_per_angstrom": max(
            item["force_vs_first_max_abs_ev_per_angstrom"] for item in records
        ),
    }


def run_core_model_forward(
    raw: Dict[str, Any],
    model: torch.nn.Module,
    cutoff: float,
    device: torch.device,
    head_name: Optional[str] = None,
) -> None:
    batch, _ = build_core_model_batch(raw, model, cutoff, device, head_name=head_name)
    model(
        batch,
        training=False,
        compute_force=True,
        compute_stress=False,
        compute_hessian=False,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timing_statistics(seconds_per_step: List[float], num_atoms: int) -> Dict[str, Any]:
    ordered = sorted(seconds_per_step)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered) + 0.999999) - 1))
    median = statistics.median(ordered)
    mean = statistics.mean(ordered)
    stdev = statistics.stdev(ordered) if len(ordered) > 1 else 0.0
    return {
        "samples_ms_per_step": [value * 1.0e3 for value in seconds_per_step],
        "median_ms_per_step": median * 1.0e3,
        "mean_ms_per_step": mean * 1.0e3,
        "min_ms_per_step": ordered[0] * 1.0e3,
        "p95_ms_per_step": ordered[p95_index] * 1.0e3,
        "stdev_ms_per_step": stdev * 1.0e3,
        "coefficient_of_variation_percent": (
            0.0 if mean == 0.0 else 100.0 * stdev / mean
        ),
        "atoms_per_second": 0.0 if median == 0.0 else num_atoms / median,
    }


def benchmark_step_loop(
    step_fn,
    device: torch.device,
    warmup: int,
    iterations: int,
    samples: int,
    num_atoms: int,
) -> Dict[str, Any]:
    if warmup < 0 or iterations <= 0 or samples <= 0:
        raise ValueError(
            "Benchmark warmup must be nonnegative; iterations and samples must be positive."
        )
    for _ in range(warmup):
        step_fn()
    _synchronize(device)

    elapsed_samples: List[float] = []
    for _ in range(samples):
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            step_fn()
        _synchronize(device)
        elapsed_samples.append((time.perf_counter() - start) / iterations)
    return _timing_statistics(elapsed_samples, num_atoms)


def _reference_signature(
    raw: Dict[str, Any],
    model_family: str,
    model_name: str,
    head_name: str,
    enable_cueq: bool,
) -> str:
    payload = {
        "system": raw["name"],
        "source_path": raw.get("source_path", f"builtin://{raw['name']}"),
        "atomic_numbers": raw["atomic_numbers"],
        "positions": raw["positions"],
        "cell": raw["cell"],
        "pbc": raw["pbc"],
        "model_family": model_family,
        "model_name": model_name,
        "head_name": head_name,
        "enable_cueq": enable_cueq,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reference_artifact_metadata(
    raw: Dict[str, Any],
    args: argparse.Namespace,
    cutoff: float,
    head_name: str,
) -> Dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "system_name": raw["name"],
        "source_path": raw.get("source_path", f"builtin://{raw['name']}"),
        "num_atoms": len(raw["atomic_numbers"]),
        "model_family": args.model_family,
        "model_name": args.model,
        "head_name": head_name,
        "enable_cueq": args.enable_cueq,
        "cutoff": cutoff,
        "signature": _reference_signature(
            raw, args.model_family, args.model, head_name, args.enable_cueq
        ),
    }


def generate_reference_artifact(
    raw: Dict[str, Any],
    args: argparse.Namespace,
    output_path: Path,
    benchmark: bool,
) -> Dict[str, Any]:
    _require_standalone_mace()
    args.model = resolve_mace_model_name(args.model_family, args.model)
    if args.repeatability_runs == 1 or args.repeatability_runs < 0:
        raise ValueError("--repeatability-runs must be 0 or at least 2.")
    if args.float64_diagnostic and (
        args.float64_max_atoms <= 0 or len(raw["atomic_numbers"]) > args.float64_max_atoms
    ):
        raise ValueError(
            "Float64 diagnostic exceeds --float64-max-atoms; use a smaller structure "
            "or explicitly raise the limit."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_model = (
        load_raw_mace_model(
            args.model_family,
            args.model,
            str(device),
            enable_cueq=args.enable_cueq,
        )
        .to(
            device=device,
            dtype=COMPUTE_DTYPE,
        )
        .eval()
    )
    head_name = resolve_model_head_name(raw_model, args.head_name or None)
    cutoff = float(raw_model.r_max.detach().cpu())
    calculator = load_mace_calculator(
        args.model_family,
        args.model,
        str(device),
        enable_cueq=args.enable_cueq,
        head_name=head_name,
    )
    atoms = raw_to_atoms(raw)

    calculator_reference = evaluate_with_calculator(
        calculator, atoms, collect_precision=True, collect_details=True
    )
    core_model_reference = evaluate_with_core_model(
        raw, raw_model, cutoff, device, head_name=head_name
    )
    calc_core_energy_delta = (
        calculator_reference["energy"] - core_model_reference["energy"]
    )
    calc_core_force_delta = (
        calculator_reference["forces"] - core_model_reference["forces"]
    )
    calculator_core_comparison = compare_calculator_core(
        calculator_reference, core_model_reference
    )

    float64_diagnostic = None
    if args.float64_diagnostic:
        # Bypass load_raw_mace_model here: it intentionally casts to float32.
        # Load the original checkpoint directly, then retain its float64 values.
        loader = {"mp": mace_mp, "off": mace_off, "omol": mace_omol,
                  "polar": mace_polar}[args.model_family]
        raw_model64 = loader(
            model=args.model, device=str(device), return_raw_model=True
        ).to(device=device, dtype=torch.float64).eval()
        calculator64 = load_mace_calculator(
            args.model_family, args.model, str(device),
            head_name=head_name, default_dtype="float64",
        )
        calc64 = evaluate_with_calculator(
            calculator64, atoms, collect_precision=True, collect_details=True
        )
        core64 = evaluate_with_core_model(
            raw, raw_model64, float(raw_model64.r_max.cpu()), device,
            head_name=head_name,
        )
        float64_diagnostic = {
            "backend": "e3nn (CuEq disabled for float64 diagnostic)",
            "calculator": {
                "legacy_energy_ev": calc64["energy"],
                "controlled_energy_ev": calc64["stable_energy"],
                "legacy_minus_controlled_ev": calc64["energy"] - calc64["stable_energy"],
                "precision": calc64["precision"],
            },
            "core": {
                "legacy_energy_ev": core64["energy"],
                "controlled_energy_ev": core64["stable_energy"],
                "legacy_minus_controlled_ev": core64["energy"] - core64["stable_energy"],
                "precision": core64["precision"],
            },
            "calculator_core": compare_calculator_core(calc64, core64),
        }

    calculator_benchmark = None
    core_model_benchmark = None
    if benchmark:
        calculator_benchmark = {
            "scope": (
                "single-GPU standalone MACE calculator path; includes MACE internal graph construction, "
                "calculator evaluation, energy extraction, and force extraction; excludes input parsing and model loading"
            ),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else str(device)
            ),
            "warmup_steps": args.benchmark_warmup,
            "iterations_per_sample": args.benchmark_iterations,
            "num_samples": args.benchmark_samples,
            "timing_method": "perf_counter with device synchronization",
            "timings": benchmark_step_loop(
                lambda: evaluate_with_calculator(calculator, atoms),
                device,
                args.benchmark_warmup,
                args.benchmark_iterations,
                args.benchmark_samples,
                len(raw["atomic_numbers"]),
            ),
        }
        core_model_benchmark = {
            "scope": (
                "single-GPU raw ScaleShiftMACE path; includes MACE-native Configuration creation, "
                "AtomicData.from_config graph construction, Batch.from_data_list packaging, and raw model forward "
                "with force computation; excludes ASE calculator result packaging, host-side output extraction, and model loading"
            ),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else str(device)
            ),
            "warmup_steps": args.benchmark_warmup,
            "iterations_per_sample": args.benchmark_iterations,
            "num_samples": args.benchmark_samples,
            "timing_method": "perf_counter with device synchronization",
            "timings": benchmark_step_loop(
                lambda: run_core_model_forward(
                    raw, raw_model, cutoff, device, head_name=head_name
                ),
                device,
                args.benchmark_warmup,
                args.benchmark_iterations,
                args.benchmark_samples,
                len(raw["atomic_numbers"]),
            ),
        }

    repeatability = (
        repeat_calculator(calculator, atoms, args.repeatability_runs)
        if args.repeatability_runs else None
    )

    artifact = {
        "metadata": reference_artifact_metadata(raw, args, cutoff, head_name),
        "precision": {
            "calculator": calculator_reference["precision"],
            "core_model": core_model_reference["precision"],
        },
        "reference": {
            "calculator_energy": calculator_reference["energy"],
            "calculator_node_energy_sum_float64": calculator_reference["stable_energy"],
            "calculator_interaction_node_energy_sum_float64": calculator_reference[
                "stable_interaction_energy"
            ],
            "calculator_baseline_node_energy_sum_float64": calculator_reference[
                "stable_baseline_energy"
            ],
            "calculator_node_energy_values": calculator_reference["node_energy_values"],
            "calculator_interaction_node_energy_values": calculator_reference[
                "interaction_node_energy_values"
            ],
            "calculator_graph_edge_count": calculator_reference["edge_count"],
            "calculator_graph_edge_rows": calculator_reference["edge_rows"],
            "calculator_forces": calculator_reference["forces"],
            "core_model_energy": core_model_reference["energy"],
            "core_model_node_energy_sum_float64": core_model_reference[
                "node_energy_sum_float64"
            ],
            "core_model_stable_energy": core_model_reference[
                "stable_energy"
            ],
            "core_model_stable_baseline_energy": core_model_reference[
                "stable_baseline_energy"
            ],
            "core_model_stable_interaction_energy": core_model_reference[
                "stable_interaction_energy"
            ],
            "core_model_node_energy_sum_float32": core_model_reference[
                "node_energy_sum_float32"
            ],
            "core_model_forces": core_model_reference["forces"],
            "core_model_edge_count": core_model_reference["edge_count"],
            "core_model_edge_rows": core_model_reference["edge_rows"],
            "core_model_node_energy_values": core_model_reference[
                "node_energy_values"
            ],
            "calculator_minus_core_model_energy": calc_core_energy_delta,
            "calculator_minus_core_model_force_mae": float(
                calc_core_force_delta.abs().mean()
            ),
            "calculator_minus_core_model_force_max_abs": float(
                calc_core_force_delta.abs().max()
            ),
        },
        "diagnostics": {
            "calculator_core": calculator_core_comparison,
            "float64": float64_diagnostic,
            "calculator_repeatability": repeatability,
        },
        "calculator_benchmark": calculator_benchmark,
        "core_model_benchmark": core_model_benchmark,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    return artifact


def print_summary(artifact: Dict[str, Any]) -> None:
    def section(title: str, values: Dict[str, Any]) -> None:
        print("\n" + "-" * 60 + f"\n{title}\n" + "-" * 60)
        for key, value in values.items():
            print(f"{key}: {value}")

    print("=" * 60)
    print("Standalone MACE Reference Artifact")
    print("=" * 60)
    metadata = artifact["metadata"]
    reference = artifact["reference"]
    diagnostics = artifact["diagnostics"]
    section("Configuration and precision", {
        **{key: metadata[key] for key in (
            "system_name", "source_path", "num_atoms", "model_family",
            "model_name", "head_name", "enable_cueq", "cutoff", "artifact_version"
        )},
        **{f"{path}.{key}": value for path, values in artifact["precision"].items()
           for key, value in values.items()},
    })
    section("Energy definitions (eV)", {
        "calculator_legacy_scalar": reference["calculator_energy"],
        "calculator_controlled_per_atom_sum": reference["calculator_node_energy_sum_float64"],
        "core_legacy_scalar": reference["core_model_energy"],
        "core_controlled_per_atom_sum": reference["core_model_stable_energy"],
    })
    section("Matched parity comparisons (eV)", {
        key: value for key, value in diagnostics["calculator_core"].items()
        if "energy" in key and "node_energy_mae" not in key
        and "node_energy_max_abs" not in key
    })
    section("Reduction decomposition (eV)", {
        "calculator_legacy_minus_controlled": (
            reference["calculator_energy"] - reference["calculator_node_energy_sum_float64"]
        ),
        "core_legacy_minus_controlled": (
            reference["core_model_energy"] - reference["core_model_stable_energy"]
        ),
        "calculator_controlled_baseline": reference["calculator_baseline_node_energy_sum_float64"],
        "calculator_controlled_interaction": reference["calculator_interaction_node_energy_sum_float64"],
        "core_controlled_baseline": reference["core_model_stable_baseline_energy"],
        "core_controlled_interaction": reference["core_model_stable_interaction_energy"],
    })
    section("Per-atom and force parity", {
        key: value for key, value in diagnostics["calculator_core"].items()
        if "mae" in key or "max_abs" in key
    })
    section("Graph diagnostics", {
        key: value for key, value in diagnostics["calculator_core"].items()
        if "edge_count" in key
    })
    if diagnostics["float64"] is not None:
        section("Float64 calculator and core diagnostic (eV)", diagnostics["float64"])
    if diagnostics["calculator_repeatability"] is not None:
        section("Calculator repeatability (outside benchmark timing)",
                diagnostics["calculator_repeatability"])
    if artifact["calculator_benchmark"] is not None:
        calc = artifact["calculator_benchmark"]
        core = artifact["core_model_benchmark"]
        section("Performance benchmark", {
            "calculator_scope": calc["scope"],
            "core_scope": core["scope"],
            "device_name": calc["device_name"],
            "timing_method": calc["timing_method"],
            "warmup_steps": calc["warmup_steps"],
            "iterations_per_sample": calc["iterations_per_sample"],
            "num_samples": calc["num_samples"],
            "calculator_median_ms_per_step": calc["timings"]["median_ms_per_step"],
            "core_median_ms_per_step": core["timings"]["median_ms_per_step"],
        })


def main() -> None:
    args = build_parser().parse_args()
    raw = load_raw_structure(args)
    artifact = generate_reference_artifact(
        raw, args, args.output, benchmark=args.benchmark
    )
    print_summary(artifact)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(artifact, indent=2, default=str), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
