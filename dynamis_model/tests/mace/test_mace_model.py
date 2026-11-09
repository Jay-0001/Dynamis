"""Distributed torchrun driver and pytest entrypoints for MACEModel.

This file owns only the distributed MACEModel path. Standalone monolithic MACE
reference generation lives in ``mace_standalone_reference.py`` so that the
standalone baseline can be exercised without Dynamis graph construction in the
same driver.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import re
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from dynamis_model.comm.feature_push import FeaturePushToExchange
from dynamis_model.nnp.mace import (
    COMPUTE_DTYPE,
    MACEModel,
    MACEStableEnergyRuntime,
    load_raw_mace_model,
    resolve_mace_model_name,
    resolve_model_head_index,
    resolve_model_head_name,
)

# MACEModel returns MD units; the references in this file are raw MACE eV.
# Import the shared constant rather than re-spelling it so the two can't drift.
from dynamis_model.units import EV_PER_A_TO_AMU_A_PER_FS2 as EV_TO_MD
from dynamis_state import (
    DynamisState,
    MolecularInfo,
    SFCCurveType,
    SFCPartitioner,
    SimulationConfig,
    TorchDistEnv,
    radius_graph_pbc_cuda,
)
from dynamis_state.masses import TORCH_ATOMIC_MASSES

try:
    from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS

    HAS_ASE = True
except Exception:  # pragma: no cover - optional dependency
    ASE_ATOMIC_NUMBERS = {}
    HAS_ASE = False

ARTIFACT_VERSION = 4
TEST_FILE_DIR = Path(__file__).resolve().parent
ELEMENT_TO_Z = dict(ASE_ATOMIC_NUMBERS)


@dataclass
class PreparedExperiment:
    raw: Dict[str, Any]
    sim: SimulationConfig
    state: DynamisState
    adapter: MACEModel
    device: torch.device
    model_id: Dict[str, Any]


def natural_chain12() -> Dict[str, Any]:
    """A contained nonperiodic C4H8 fragment with covalent-scale spacing."""
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
    subparsers = parser.add_subparsers(dest="command", required=True)
    distributed_parser = subparsers.add_parser(
        "run-distributed",
        help="Run distributed MACEModel and compare against a detached standalone reference artifact.",
    )
    distributed_parser.add_argument(
        "--builtin-system", choices=BUILTIN_SYSTEMS, default="natural_chain12"
    )
    distributed_parser.add_argument(
        "--xyz-path", type=Path, help="XYZ/extxyz input; overrides --builtin-system."
    )
    distributed_parser.add_argument(
        "--model-family", choices=["mp", "off", "omol", "polar"], default="mp"
    )
    distributed_parser.add_argument(
        "--model", default=None, help="Checkpoint name (defaults by family)."
    )
    distributed_parser.add_argument(
        "--head-name", default="", help="Optional explicit MACE head name."
    )
    distributed_parser.add_argument(
        "--energy-mode", choices=["legacy", "stable"], default="legacy",
        help="Use the original float32 node sum or separate float64 baseline/interaction sums.",
    )
    distributed_parser.add_argument(
        "--enable-cueq",
        action="store_true",
        help="Request cuEquivariance conversion when loading the distributed raw MACE model.",
    )
    distributed_parser.add_argument("--ghost-cutoff-buffer", type=float, default=0.5)
    distributed_parser.add_argument(
        "--reference-artifact",
        type=Path,
        help="Detached standalone reference artifact.",
    )
    distributed_parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Time only the distributed adapter loop.",
    )
    distributed_parser.add_argument("--benchmark-warmup", type=int, default=3)
    distributed_parser.add_argument("--benchmark-iterations", type=int, default=10)
    distributed_parser.add_argument("--benchmark-samples", type=int, default=5)
    distributed_parser.add_argument(
        "--time-initialize-buffers",
        action="store_true",
        help="Also report init-only timing for adapter wrapper construction + initialize_buffers.",
    )
    distributed_parser.add_argument("--atol", type=float, default=1e-5)
    distributed_parser.add_argument("--rtol", type=float, default=1e-5)
    distributed_parser.add_argument(
        "--stable-energy-atol", type=float, default=1e-3,
        help="Absolute MD-energy tolerance for the controlled float64 total.",
    )
    distributed_parser.add_argument("--strict-parity", action="store_true")
    distributed_parser.add_argument(
        "--graph-edge-test", action="store_true",
        help="Audit assembled owned directed edges and owned per-atom energies against standalone graphs.",
    )
    distributed_parser.add_argument(
        "--strict-graph-interior", action="store_true",
        help="Enable the graph audit and fail if assembled owned edges differ from native MACE more than 1e-4 Angstrom inside the cutoff.",
    )
    distributed_parser.add_argument("--output-json", type=Path)
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
        f"torchrun working directory: {Path.cwd()}\n"
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


def _assign_if_present(obj: object, name: str, value: torch.Tensor) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def build_state(
    raw: Dict[str, Any],
    cutoff: float,
    ghost_cutoff_buffer: float,
    torch_env: TorchDistEnv,
) -> Tuple[DynamisState, SimulationConfig]:
    device = torch_env.device
    numbers = torch.tensor(raw["atomic_numbers"], dtype=torch.long, device=device)
    positions = torch.tensor(raw["positions"], dtype=COMPUTE_DTYPE, device=device)
    global_mol = MolecularInfo()
    global_mol.num_atoms = positions.shape[0]
    global_mol.atom_positions = positions
    global_mol.atom_types = numbers
    global_mol.atom_velocities = torch.zeros_like(positions)
    _assign_if_present(global_mol, "atom_forces", torch.zeros_like(positions))
    _assign_if_present(
        global_mol,
        "atom_vertex_numbers",
        torch.arange(positions.shape[0], device=device),
    )
    _assign_if_present(
        global_mol, "atom_masses", TORCH_ATOMIC_MASSES.to(device)[numbers].float()
    )

    sim = SimulationConfig()
    sim.cutoff = cutoff
    sim.ghost_cutoff_buffer = ghost_cutoff_buffer
    sim.cell = torch.tensor(raw["cell"], dtype=COMPUTE_DTYPE, device=device)
    sim.pbc = torch.tensor(raw["pbc"], dtype=torch.bool, device=device)

    partitioner = SFCPartitioner(SFCCurveType.Morton)
    info = partitioner.partition(global_mol, torch_env.dist_env)
    partitioner.build_halo_maps(info, global_mol, torch_env.dist_env, sim)
    local_indices = info.partition_indices.long()
    local_mol = MolecularInfo()
    local_mol.num_atoms = local_indices.numel()
    local_mol.atom_positions = positions.index_select(0, local_indices)
    local_mol.atom_types = numbers.index_select(0, local_indices)
    local_mol.atom_velocities = torch.zeros_like(local_mol.atom_positions)
    _assign_if_present(
        local_mol, "atom_forces", torch.zeros_like(local_mol.atom_positions)
    )
    _assign_if_present(local_mol, "atom_vertex_numbers", local_indices)
    _assign_if_present(
        local_mol,
        "atom_masses",
        TORCH_ATOMIC_MASSES.to(device)[local_mol.atom_types].float(),
    )

    state = DynamisState()
    state.dist_env = torch_env.dist_env
    state.sim_config = sim
    state.global_molecular_info = global_mol
    state.local_partition_info = info
    state.local_molecular_info = local_mol
    return state, sim


def radius_graph(
    positions: torch.Tensor,
    sim: SimulationConfig,
    cutoff: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data = SimpleNamespace(
        pos=positions,
        cell=sim.cell.reshape(1, 3, 3),
        pbc=sim.pbc.reshape(1, 3),
        natoms=torch.tensor([positions.shape[0]], device=positions.device),
    )
    with torch.no_grad():
        edge_index, offsets, _ = radius_graph_pbc_cuda(
            data,
            radius=cutoff,
            max_num_neighbors_threshold=0,
            enforce_max_neighbors_strictly=False,
        )
    return edge_index.long(), -offsets


def edge_records(
    edge_index: torch.Tensor,
    unit_shifts: torch.Tensor,
) -> set[Tuple[int, int, int, int, int]]:
    offsets = unit_shifts.round().long().cpu().tolist()
    pairs = edge_index.T.long().cpu().tolist()
    return {
        (int(pair[0]), int(pair[1]), int(offset[0]), int(offset[1]), int(offset[2]))
        for pair, offset in zip(pairs, offsets)
    }


def edge_record_rows(
    edge_index: torch.Tensor, unit_shifts: torch.Tensor
) -> torch.Tensor:
    """Directed sender/receiver and integer cell shift, preserving duplicates."""
    return torch.cat(
        (edge_index.T.long(), unit_shifts.round().long()), dim=1
    ).to(dtype=torch.int32, device="cpu")


def compare_graph_edges(
    explicit_rows: torch.Tensor,
    core_rows: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
) -> Dict[str, Any]:
    """Compare edge multisets and locate unmatched edges relative to cutoff."""
    explicit = Counter(map(tuple, explicit_rows.long().tolist()))
    core = Counter(map(tuple, core_rows.long().tolist()))
    positions = positions.detach().double().cpu()
    cell = cell.detach().reshape(3, 3).double().cpu()

    def describe_only(rows: Counter) -> Dict[str, Any]:
        count = sum(rows.values())
        if not count:
            return {"count": 0, "distance_min": None, "distance_max": None,
                    "within_1e_4_angstrom_of_cutoff": 0,
                    "inside_cutoff_by_more_than_1e_4_angstrom": 0,
                    "outside_cutoff_by_more_than_1e_4_angstrom": 0,
                    "examples": []}
        values = torch.tensor(list(rows.elements()), dtype=torch.long)
        displacement = (
            positions.index_select(0, values[:, 1])
            - positions.index_select(0, values[:, 0])
            + values[:, 2:].double() @ cell
        )
        distances = torch.linalg.vector_norm(displacement, dim=1)
        delta = distances - cutoff
        return {
            "count": count,
            "distance_min": float(distances.min()),
            "distance_max": float(distances.max()),
            "within_1e_4_angstrom_of_cutoff": int((delta.abs() <= 1e-4).sum()),
            "inside_cutoff_by_more_than_1e_4_angstrom": int((delta < -1e-4).sum()),
            "outside_cutoff_by_more_than_1e_4_angstrom": int((delta > 1e-4).sum()),
            "examples": [
                {"edge": values[i].tolist(), "distance": float(distances[i])}
                for i in range(min(5, count))
            ],
        }

    return {
        "cutoff": cutoff,
        "explicit_raw_count": int(explicit_rows.shape[0]),
        "core_raw_count": int(core_rows.shape[0]),
        "explicit_unique_count": len(explicit),
        "core_unique_count": len(core),
        "explicit_duplicate_count": sum(explicit.values()) - len(explicit),
        "core_duplicate_count": sum(core.values()) - len(core),
        "common_count_with_multiplicity": sum((explicit & core).values()),
        "explicit_only": describe_only(explicit - core),
        "core_only": describe_only(core - explicit),
    }


def explicit_mace_batch_from_tensors(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    model: torch.nn.Module,
    sim: SimulationConfig,
    edge_index: torch.Tensor,
    unit_shifts: torch.Tensor,
    head_name: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    device = positions.device
    model_numbers = model.atomic_numbers.to(device=device, dtype=torch.long)
    node_attrs = numbers[:, None].eq(model_numbers[None, :]).to(COMPUTE_DTYPE)
    cell = sim.cell.reshape(3, 3).to(dtype=COMPUTE_DTYPE)
    unit_shifts = unit_shifts.to(device=device, dtype=COMPUTE_DTYPE)
    head_index = resolve_model_head_index(model, head_name)
    return {
        "positions": positions,
        "edge_index": edge_index.to(device),
        "unit_shifts": unit_shifts,
        "shifts": unit_shifts @ cell,
        "cell": cell,
        "node_attrs": node_attrs,
        "batch": torch.zeros(numbers.numel(), dtype=torch.long, device=device),
        "ptr": torch.tensor([0, numbers.numel()], dtype=torch.long, device=device),
        "head": torch.tensor([head_index], dtype=torch.long, device=device),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _critical_path_seconds(local_seconds: float, device: torch.device) -> float:
    reduce_device = (
        device
        if dist.is_initialized() and dist.get_backend() == "nccl"
        else torch.device("cpu")
    )
    elapsed = torch.tensor(local_seconds, dtype=torch.float64, device=reduce_device)
    if dist.is_initialized():
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.cpu())


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


def load_and_validate_reference_artifact(
    artifact_path: Path,
    raw: Dict[str, Any],
    args: argparse.Namespace,
    require_standalone_benchmark: bool = False,
    expected_head_name: Optional[str] = None,
) -> Dict[str, Any]:
    args.model = resolve_mace_model_name(args.model_family, args.model)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    metadata = artifact.get("metadata", {})
    if expected_head_name is None:
        expected_head_name = resolve_model_head_name(
            load_raw_mace_model(
                args.model_family,
                args.model,
                "cpu",
                enable_cueq=args.enable_cueq,
            ).eval(),
            args.head_name or None,
        )
    expected_signature = _reference_signature(
        raw,
        args.model_family,
        args.model,
        expected_head_name,
        args.enable_cueq,
    )
    if metadata.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"Reference artifact version mismatch: expected {ARTIFACT_VERSION}, "
            f"got {metadata.get('artifact_version')!r}."
        )
    if metadata.get("signature") != expected_signature:
        expected_identity = {
            "system_name": raw["name"],
            "source_path": raw.get("source_path", f"builtin://{raw['name']}"),
            "num_atoms": len(raw["atomic_numbers"]),
            "model_family": args.model_family,
            "model_name": args.model,
            "head_name": expected_head_name,
            "enable_cueq": args.enable_cueq,
        }
        differences = [
            f"{key}: artifact={metadata.get(key)!r}, requested={value!r}"
            for key, value in expected_identity.items()
            if metadata.get(key) != value
        ]
        if not differences:
            differences = [
                "recorded identity fields match; check XYZ atomic numbers, "
                "positions, cell, and pbc for changes since reference generation"
            ]
        raise ValueError(
            "Reference artifact is stale or incompatible with the requested "
            "system/model. " + "; ".join(differences)
        )
    reference = artifact.get("reference", {})
    required_reference_keys = {
        "calculator_energy",
        "calculator_forces",
        "core_model_energy",
        "core_model_forces",
    }
    missing_reference_keys = sorted(
        required_reference_keys.difference(reference.keys())
    )
    if missing_reference_keys:
        raise ValueError(
            "Reference artifact is missing required monolithic reference keys: "
            + ", ".join(missing_reference_keys)
        )
    if require_standalone_benchmark and (
        artifact.get("calculator_benchmark") is None
        or artifact.get("core_model_benchmark") is None
    ):
        raise ValueError(
            "Benchmark mode requires a reference artifact generated with both calculator and core-model timing stats."
        )
    return artifact


def resolve_reference_artifact_path(args: argparse.Namespace) -> Path:
    if args.reference_artifact is not None:
        return args.reference_artifact
    env_candidates = [
        os.environ.get("DISTMACE_REFERENCE_ARTIFACT"),
        os.environ.get("DISTMACE_REFERENCE_DEFAULT_ARTIFACT"),
    ]
    for candidate in env_candidates:
        if candidate:
            return Path(candidate)
    raise ValueError(
        "run-distributed requires --reference-artifact. "
        "For pytest parity runs, set DISTMACE_REFERENCE_ARTIFACT or a test-specific artifact env var."
    )


def prepare_distributed(args: argparse.Namespace) -> PreparedExperiment:
    args.model = resolve_mace_model_name(args.model_family, args.model)
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        raise RuntimeError("Launch this test with torchrun and at least two ranks.")
    if not HAS_ASE and args.xyz_path is not None:
        raise ImportError("ASE is required to decode element symbols from XYZ inputs.")
    torch_env = TorchDistEnv(is_distributed=True)
    if dist.get_world_size() < 2:
        raise RuntimeError(
            "The distributed MACEModel test requires at least two ranks."
        )
    device = torch_env.device
    raw = load_raw_structure(args)
    model = (
        load_raw_mace_model(
            args.model_family,
            args.model,
            str(device),
            enable_cueq=args.enable_cueq,
        )
        .to(device=device)
        .eval()
    )
    cutoff = float(model.r_max.detach().cpu())
    state, sim = build_state(raw, cutoff, args.ghost_cutoff_buffer, torch_env)
    adapter = MACEModel(
        model, head_name=args.head_name or None, energy_mode=args.energy_mode
    )
    if not torch.isclose(
        torch.as_tensor(state.sim_config.cutoff), torch.as_tensor(adapter.cutoff)
    ):
        raise ValueError("Simulation halo cutoff must match the MACE checkpoint r_max.")
    adapter.initialize_buffers(
        state.local_partition_info, device, dist.get_world_size()
    )
    return PreparedExperiment(
        raw=raw,
        sim=sim,
        state=state,
        adapter=adapter,
        device=device,
        model_id={
            "model_family": args.model_family,
            "model_name": args.model,
            "head_name": adapter.head_name,
            "enable_cueq": args.enable_cueq,
            "energy_mode": args.energy_mode,
        },
    )


def gather_records(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Optional[Dict[str, Any]]] = [None] * dist.get_world_size()
    dist.all_gather_object(records, record)
    return [item for item in records if item is not None]


def assemble_forces(records: List[Dict[str, Any]], num_atoms: int) -> torch.Tensor:
    forces = torch.zeros(num_atoms, 3, dtype=COMPUTE_DTYPE)
    for record in records:
        indices = torch.tensor(record["local_global_indices"], dtype=torch.long)
        local_forces = torch.tensor(record["owned_local_forces"], dtype=COMPUTE_DTYPE)
        forces.index_copy_(0, indices, local_forces)
    return forces


def assemble_owned_node_energies(records: List[Dict[str, Any]], num_atoms: int) -> torch.Tensor:
    """Place each rank's owned atom energies in global atom order (MD units)."""
    values = torch.empty(num_atoms, dtype=torch.float64)
    visited = torch.zeros(num_atoms, dtype=torch.bool)
    for record in records:
        indices = torch.tensor(record["local_global_indices"], dtype=torch.long)
        energies = torch.tensor(record["owned_node_energy"], dtype=torch.float64)
        if (energies.numel() != indices.numel()
                or indices.unique().numel() != indices.numel()
                or visited[indices].any()):
            raise ValueError("Owned atom energy records are incomplete or overlapping.")
        values[indices] = energies
        visited[indices] = True
    if not visited.all():
        raise ValueError("Owned atom energy records omit global atoms.")
    return values


def owned_edge_audit_rows(prepared: PreparedExperiment) -> torch.Tensor:
    """Map the rank's owned-destination graph into global directed edge IDs."""
    state = prepared.state
    adapter = prepared.adapter
    local_ids = state.local_partition_info.partition_indices.long().to(prepared.device)
    if len(prepared.raw["atomic_numbers"]) > 2**24:
        raise ValueError("Float32 halo ID exchange cannot represent this many atoms exactly.")
    view = adapter._build_structural_view(state)
    graph = adapter._build_owned_graph(view, state)
    with torch.no_grad():
        if adapter.exchange_enabled:
            global_ids_float = FeaturePushToExchange.apply(
                local_ids.to(dtype=COMPUTE_DTYPE).unsqueeze(1),
                state, adapter.atom_type_exchanger,
            ).squeeze(1)
        else:
            global_ids_float = local_ids.to(dtype=COMPUTE_DTYPE)
    global_ids = global_ids_float.long()
    if (global_ids_float != global_ids).any() or global_ids.numel() != view.num_total:
        raise ValueError("Halo ID exchange produced invalid graph row IDs.")
    if not torch.equal(global_ids[:view.num_local], local_ids):
        raise ValueError("Owned graph rows do not match local partition indices.")
    if ((global_ids < 0) | (global_ids >= len(prepared.raw["atomic_numbers"]))).any():
        raise ValueError("Halo ID exchange produced an out-of-range atom ID.")
    global_edges = global_ids[graph.edge_index]
    return edge_record_rows(global_edges, graph.unit_shifts)


def edge_distance_bands(
    rows: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor,
    cutoff: float, margin: float = 1e-4,
) -> Dict[str, torch.Tensor]:
    """Split directed edges into safely interior, cutoff-boundary, and exterior."""
    rows = rows.long().cpu()
    if rows.numel() == 0:
        empty = torch.empty((0, 5), dtype=torch.int32)
        return {"interior": empty, "boundary": empty, "exterior": empty}
    positions = positions.double().cpu()
    cell = cell.reshape(3, 3).double().cpu()
    offsets = rows[:, 2:].double() @ cell
    distances = torch.linalg.vector_norm(
        positions[rows[:, 1]] - positions[rows[:, 0]] + offsets, dim=1
    )
    return {
        "interior": rows[distances < cutoff - margin].int(),
        "boundary": rows[(distances >= cutoff - margin) & (distances <= cutoff + margin)].int(),
        "exterior": rows[distances > cutoff + margin].int(),
    }


def nonperiodic_outside_cell_diagnostics(
    positions: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor,
) -> Dict[str, Any]:
    """Locate atoms an unconditional cell-wrap would move on open axes."""
    fractional = positions.detach().double().cpu() @ torch.linalg.inv(
        cell.detach().reshape(3, 3).double().cpu()
    )
    open_axes = ~pbc.detach().reshape(3).bool().cpu()
    outside = ((fractional < 0.0) | (fractional >= 1.0)) & open_axes
    return {
        "atom_count": int(outside.any(dim=1).sum()),
        "atom_count_by_axis": outside.sum(dim=0).tolist(),
        "fractional_min": fractional.min(dim=0).values.tolist(),
        "fractional_max": fractional.max(dim=0).values.tolist(),
    }


def build_owned_graph_audit(
    prepared: PreparedExperiment, records: List[Dict[str, Any]],
    explicit_reference: Dict[str, Any], artifact: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare assembled distributed graph and node outputs with native references."""
    rows = torch.tensor(
        [row for record in records for row in record["owned_edge_rows"]],
        dtype=torch.int32,
    ).reshape(-1, 5)
    native_rows = artifact["reference"].get("core_model_edge_rows")
    calculator_rows = artifact["reference"].get("calculator_graph_edge_rows")
    if native_rows is None or calculator_rows is None:
        raise ValueError(
            "Graph edge audit requires a freshly generated reference with core "
            "and calculator edge rows."
        )
    positions = prepared.state.global_molecular_info.atom_positions
    cell = prepared.sim.cell
    cutoff = float(prepared.adapter.cutoff)
    comparison: Dict[str, Any] = {
        "boundary_band_angstrom": 1e-4,
        "assembled_owned_edge_count": int(rows.shape[0]),
        "nonperiodic_outside_cell": nonperiodic_outside_cell_diagnostics(
            positions, cell, prepared.sim.pbc
        ),
    }
    assembled_bands = edge_distance_bands(rows, positions, cell, cutoff)
    comparison["assembled_owned_edge_counts_by_band"] = {
        key: int(value.shape[0]) for key, value in assembled_bands.items()
    }
    for name, other in (
        ("native_core", native_rows),
        ("public_calculator", calculator_rows),
        ("explicit_dynamis", explicit_reference["edge_rows"]),
    ):
        if other is None:
            continue
        comparison[f"assembled_vs_{name}"] = compare_graph_edges(
            rows, other, positions, cell, cutoff
        )
        other_bands = edge_distance_bands(other, positions, cell, cutoff)
        comparison[f"assembled_vs_{name}_by_band"] = {
            band: compare_graph_edges(rows_in_band, other_bands[band], positions, cell, cutoff)
            for band, rows_in_band in assembled_bands.items()
        }
    owned_values = assemble_owned_node_energies(records, len(prepared.raw["atomic_numbers"]))
    node_parity: Dict[str, Any] = {
        "assembled_owned_node_energy_sum_md": float(owned_values.sum()),
    }
    for name, reference in (
        ("native_core", artifact["reference"].get("core_model_node_energy_values")),
        ("public_calculator", artifact["reference"].get("calculator_node_energy_values")),
        ("explicit_dynamis", explicit_reference["node_energy_values"]),
    ):
        if reference is None:
            continue
        other = torch.as_tensor(reference).double().reshape(-1) * EV_TO_MD
        if other.shape != owned_values.shape:
            raise ValueError(f"{name} node energies do not match global atom count.")
        delta = owned_values - other
        node_parity[f"assembled_minus_{name}_node_sum_md"] = float(delta.sum())
        node_parity[f"assembled_vs_{name}_node_mae_md"] = float(delta.abs().mean())
        node_parity[f"assembled_vs_{name}_node_max_abs_md"] = float(delta.abs().max())
    comparison["node_parity"] = node_parity
    return comparison


def run_distributed_adapter_step(
    prepared: PreparedExperiment,
    graph_edge_test: bool = False,
) -> Tuple[float, torch.Tensor, List[Dict[str, Any]]]:
    try:
        result = prepared.adapter(
            prepared.state, return_energy_components=True,
            return_owned_node_energy=graph_edge_test,
        )
        if graph_edge_test:
            energy, forces, components, node_energies = result
        else:
            energy, forces, components = result
        record = {
            "rank": dist.get_rank(),
            "local_global_indices": prepared.state.local_partition_info.partition_indices.long()
            .cpu()
            .tolist(),
            "owned_local_energy": float(energy.cpu()),
            "owned_local_energy_dtype": str(energy.dtype),
            "model_parameter_dtype": str(
                next(prepared.adapter.model.parameters()).dtype
            ),
            "owned_local_forces": forces.detach().cpu().tolist(),
        }
        for key, value in components.items():
            record[key] = float(value.cpu())
        if graph_edge_test:
            record["owned_node_energy"] = node_energies.double().cpu().tolist()
            record["owned_edge_rows"] = owned_edge_audit_rows(prepared).tolist()
        records = gather_records(record)
        dist.barrier()
        return float(energy.cpu()), forces, records
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        if dist.is_initialized() and hasattr(dist, "abort"):
            dist.abort()
        raise


def build_explicit_global_reference(
    prepared: PreparedExperiment,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    from mace.tools.scatter import scatter_sum

    model = prepared.adapter.model
    cutoff = float(model.r_max.detach().cpu())
    positions = prepared.state.global_molecular_info.atom_positions.detach().clone()
    numbers = prepared.state.global_molecular_info.atom_types.detach().clone()
    explicit_edges, explicit_shifts = radius_graph(positions, prepared.sim, cutoff)
    explicit_batch = explicit_mace_batch_from_tensors(
        positions,
        numbers,
        model,
        prepared.sim,
        explicit_edges,
        explicit_shifts,
        head_name=args.head_name or prepared.adapter.head_name,
    )
    explicit_out = model(
        explicit_batch,
        training=False,
        compute_force=True,
        compute_stress=False,
        compute_hessian=False,
    )
    head_index = int(explicit_batch["head"].reshape(()).item())
    node_e0_raw = model.atomic_energies_fn(explicit_batch["node_attrs"])[
        :, head_index
    ].detach()
    node_e0 = node_e0_raw.double()
    # Reproduce ScaleShiftMACE.forward's E0 reduction. Its scatter_add_ can
    # accumulate differently from torch.sum even with the same input dtype.
    model_style_e0 = scatter_sum(
        node_e0_raw, explicit_batch["batch"], dim=0, dim_size=1
    ).to(explicit_batch["positions"].dtype)
    node_energy = explicit_out["node_energy"].detach().double().reshape(-1)
    node_interaction = node_energy - node_e0
    stable_energy = node_e0.sum() + node_interaction.sum()
    explicit_rows = edge_record_rows(explicit_edges, explicit_shifts)
    return {
        "energy": float(explicit_out["energy"].detach().cpu().reshape(())),
        "energy_tensor_dtype": str(explicit_out["energy"].dtype),
        "node_energy_tensor_dtype": str(explicit_out["node_energy"].dtype),
        "node_energy_sum": float(node_energy.sum().cpu()),
        "node_energy_sum_float32": float(node_energy.float().sum().cpu()),
        "stable_energy": float(stable_energy.cpu()),
        "node_energy_values": node_energy.cpu(),
        "node_e0_sum": float(node_e0.sum().cpu()),
        "node_e0_sum_float32": float(node_e0.float().sum().cpu()),
        "node_e0_tensor_dtype": str(node_e0_raw.dtype),
        "node_e0_model_style_scatter_sum": float(model_style_e0.cpu().reshape(())),
        "node_interaction_sum": float(node_interaction.sum().cpu()),
        "node_interaction_sum_float32": float(
            node_interaction.float().sum().cpu()
        ),
        "interaction_energy": float(
            explicit_out["interaction_energy"].detach().cpu().reshape(())
        ),
        "forces": explicit_out["forces"].detach().to(COMPUTE_DTYPE).cpu(),
        "edge_count": int(explicit_edges.shape[1]),
        "edge_unique_count": len(edge_records(explicit_edges, explicit_shifts)),
        "edge_rows": explicit_rows,
    }


def benchmark_distributed(
    prepared: PreparedExperiment,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    if (
        args.benchmark_warmup < 0
        or args.benchmark_iterations <= 0
        or args.benchmark_samples <= 0
    ):
        raise ValueError(
            "Benchmark warmup must be nonnegative; iterations and samples must be positive."
        )

    device = prepared.device
    adapter = prepared.adapter
    state = prepared.state
    raw = prepared.raw
    base_model = prepared.adapter.model

    for _ in range(args.benchmark_warmup):
        adapter(state)
    _synchronize(device)
    dist.barrier()

    distributed_samples: List[float] = []
    for _ in range(args.benchmark_samples):
        dist.barrier()
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(args.benchmark_iterations):
            adapter(state)
        _synchronize(device)
        local_seconds = (time.perf_counter() - start) / args.benchmark_iterations
        distributed_samples.append(_critical_path_seconds(local_seconds, device))

    stage_samples: Dict[str, List[float]] = {
        "structural_view_with_position_exchange": [],
        "owned_graph_build": [],
        "mace_forward_with_feature_exchange": [],
        "force_recovery_with_backward_exchange": [],
        "total_profiled_step": [],
    }
    last_view = None
    last_graph = None
    for _ in range(args.benchmark_samples):
        dist.barrier()
        _synchronize(device)
        total_start = time.perf_counter()

        stage_start = time.perf_counter()
        view = adapter._build_structural_view(state)
        _synchronize(device)
        stage_samples["structural_view_with_position_exchange"].append(
            time.perf_counter() - stage_start
        )

        stage_start = time.perf_counter()
        graph = adapter._build_owned_graph(view, state)
        _synchronize(device)
        stage_samples["owned_graph_build"].append(time.perf_counter() - stage_start)

        adapter.runtime.begin_forward(state, adapter.exchange_enabled)
        try:
            stage_start = time.perf_counter()
            outputs = adapter.runtime(graph.batch_dict)
            owned_energy = outputs["owned_local_energy"] * EV_TO_MD
            _synchronize(device)
            stage_samples["mace_forward_with_feature_exchange"].append(
                time.perf_counter() - stage_start
            )

            stage_start = time.perf_counter()
            adapter._recover_forces(owned_energy, view)
            _synchronize(device)
            stage_samples["force_recovery_with_backward_exchange"].append(
                time.perf_counter() - stage_start
            )
        finally:
            adapter.runtime.end_forward()

        stage_samples["total_profiled_step"].append(time.perf_counter() - total_start)
        last_view = view
        last_graph = graph

    if last_view is None or last_graph is None:
        raise RuntimeError("Stage profiling did not execute.")

    feature_width = int(adapter.runtime._feature_width or 0)
    feature_exchange_count = max(0, len(adapter.model.interactions) - 1)
    element_size = torch.empty((), dtype=COMPUTE_DTYPE).element_size()
    total_send = int(adapter.partition_validation["total_send"])
    total_recv = int(adapter.partition_validation["total_recv"])
    forward_send_bytes = (
        total_send * feature_width * element_size * feature_exchange_count
    )
    forward_recv_bytes = (
        total_recv * feature_width * element_size * feature_exchange_count
    )
    backward_send_bytes = forward_recv_bytes
    backward_recv_bytes = forward_send_bytes
    position_width = 3
    position_forward_send_bytes = total_send * position_width * element_size
    position_forward_recv_bytes = total_recv * position_width * element_size
    position_backward_send_bytes = position_forward_recv_bytes
    position_backward_recv_bytes = position_forward_send_bytes
    owned_atoms = int(last_view.num_local)
    total_atoms = int(last_view.num_total)
    halo_atoms = total_atoms - owned_atoms
    pre_filter_edges = int(last_graph.pre_filter_edge_count)
    owned_edges = int(last_graph.edge_index.shape[1])
    stage_timings = {
        name: _timing_statistics(values, owned_atoms)
        for name, values in stage_samples.items()
    }
    profiled_total_ms = stage_timings["total_profiled_step"]["median_ms_per_step"]
    for name, timing in stage_timings.items():
        timing["fraction_of_profiled_total"] = (
            1.0
            if name == "total_profiled_step"
            else timing["median_ms_per_step"] / profiled_total_ms
        )
    local_stage_profile = {
        "rank": dist.get_rank(),
        "timing_note": (
            "Diagnostic decomposition with a device synchronization after every stage; stage values "
            "include synchronization overhead and should not replace the uninterrupted distributed timing."
        ),
        "timings": stage_timings,
        "workload": {
            "owned_atoms": owned_atoms,
            "received_halo_atoms": halo_atoms,
            "total_local_plus_halo_atoms": total_atoms,
            "halo_to_owned_ratio": (
                0.0 if owned_atoms == 0 else halo_atoms / owned_atoms
            ),
            "pre_filter_graph_edges": pre_filter_edges,
            "owned_destination_edges": owned_edges,
            "owned_to_pre_filter_edge_ratio": (
                0.0 if pre_filter_edges == 0 else owned_edges / pre_filter_edges
            ),
            "send_rows_per_peer": adapter.partition_validation["send_counts_by_peer"],
            "receive_rows_per_peer": adapter.partition_validation[
                "recv_counts_by_peer"
            ],
            "feature_width": feature_width,
            "feature_exchanges_per_forward": feature_exchange_count,
            "feature_element_size_bytes": element_size,
            "forward_feature_send_bytes": forward_send_bytes,
            "forward_feature_receive_bytes": forward_recv_bytes,
            "backward_feature_send_bytes": backward_send_bytes,
            "backward_feature_receive_bytes": backward_recv_bytes,
            "total_feature_traffic_bytes": (
                forward_send_bytes
                + forward_recv_bytes
                + backward_send_bytes
                + backward_recv_bytes
            ),
            "position_width": position_width,
            "forward_position_send_bytes": position_forward_send_bytes,
            "forward_position_receive_bytes": position_forward_recv_bytes,
            "backward_position_send_bytes": position_backward_send_bytes,
            "backward_position_receive_bytes": position_backward_recv_bytes,
        },
    }
    stage_profiles: List[Optional[Dict[str, Any]]] = [None] * dist.get_world_size()
    dist.all_gather_object(stage_profiles, local_stage_profile)

    init_stats = None
    if args.time_initialize_buffers:
        init_samples: List[float] = []
        for _ in range(args.benchmark_samples):
            dist.barrier()
            _synchronize(device)
            start = time.perf_counter()
            init_adapter = MACEModel(
                base_model,
                head_name=args.head_name or None,
                energy_mode=args.energy_mode,
            )
            init_adapter.initialize_buffers(
                state.local_partition_info, device, dist.get_world_size()
            )
            _synchronize(device)
            local_seconds = time.perf_counter() - start
            init_samples.append(_critical_path_seconds(local_seconds, device))
        if dist.get_rank() == 0:
            init_stats = _timing_statistics(init_samples, len(raw["atomic_numbers"]))

    if dist.get_rank() != 0:
        return None
    distributed_stats = _timing_statistics(
        distributed_samples, len(raw["atomic_numbers"])
    )
    benchmark = {
        "scope": (
            "steady-state distributed MACEModel energy+forces; includes position exchange, "
            "owned-graph build, runtime feature exchange, and force recovery; excludes standalone "
            "reference generation, explicit parity reconstruction, and parity comparison"
        ),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        ),
        "timing_method": "perf_counter with device synchronization; distributed samples use slowest rank",
        "world_size": dist.get_world_size(),
        "warmup_steps": args.benchmark_warmup,
        "iterations_per_sample": args.benchmark_iterations,
        "num_samples": args.benchmark_samples,
        "distributed": distributed_stats,
        "per_rank_stage_profiles": [
            profile for profile in stage_profiles if profile is not None
        ],
    }
    if init_stats is not None:
        benchmark["distributed_initialize_buffers"] = init_stats
    return benchmark


def build_summary(
    prepared: PreparedExperiment,
    records: List[Dict[str, Any]],
    explicit_reference: Dict[str, Any],
    artifact: Dict[str, Any],
    args: argparse.Namespace,
    benchmark: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    num_atoms = len(prepared.raw["atomic_numbers"])
    records = sorted(records, key=lambda item: item["rank"])
    distributed_energy = sum(item["owned_local_energy"] for item in records)
    distributed_forces = assemble_forces(records, num_atoms)
    # MACEModel reports MD units; every reference here comes from a raw MACE
    # forward or a public calculator and is therefore eV / (eV per Angstrom).
    # Scale the references up front so nothing below mixes the two systems --
    # comparing MD-unit output against an unscaled eV reference is a ~99%
    # relative error that no tolerance would forgive.
    explicit_energy = float(explicit_reference["energy"]) * EV_TO_MD
    explicit_stable_energy = float(explicit_reference["stable_energy"]) * EV_TO_MD
    explicit_node_energy_sum = (
        float(explicit_reference["node_energy_sum"]) * EV_TO_MD
    )
    explicit_node_energy_sum_float32 = (
        float(explicit_reference["node_energy_sum_float32"]) * EV_TO_MD
    )
    explicit_interaction_energy = (
        float(explicit_reference["interaction_energy"]) * EV_TO_MD
    )
    explicit_node_e0_sum = float(explicit_reference["node_e0_sum"]) * EV_TO_MD
    explicit_node_e0_sum_float32 = (
        float(explicit_reference["node_e0_sum_float32"]) * EV_TO_MD
    )
    explicit_node_e0_model_style_scatter_sum = (
        float(explicit_reference["node_e0_model_style_scatter_sum"]) * EV_TO_MD
    )
    explicit_node_interaction_sum = (
        float(explicit_reference["node_interaction_sum"]) * EV_TO_MD
    )
    explicit_node_interaction_sum_float32 = (
        float(explicit_reference["node_interaction_sum_float32"]) * EV_TO_MD
    )
    explicit_forces = explicit_reference["forces"].to(dtype=COMPUTE_DTYPE) * EV_TO_MD
    calculator_energy = float(artifact["reference"]["calculator_energy"]) * EV_TO_MD
    calculator_controlled_ev = artifact["reference"].get(
        "calculator_node_energy_sum_float64"
    )
    calculator_controlled_energy = (
        float(calculator_controlled_ev) * EV_TO_MD
        if calculator_controlled_ev is not None else None
    )
    calculator_forces = (
        artifact["reference"]["calculator_forces"].to(dtype=COMPUTE_DTYPE) * EV_TO_MD
    )
    core_model_energy = float(artifact["reference"]["core_model_energy"]) * EV_TO_MD
    core_stable_ev = artifact["reference"].get(
        "core_model_stable_energy",
        artifact["reference"].get("core_model_node_energy_sum_float64"),
    )
    core_model_stable_energy = (
        float(core_stable_ev) * EV_TO_MD if core_stable_ev is not None else None
    )
    core_model_forces = (
        artifact["reference"]["core_model_forces"].to(dtype=COMPUTE_DTYPE) * EV_TO_MD
    )
    legacy_energy_close = torch.isclose(
        torch.tensor(distributed_energy, dtype=COMPUTE_DTYPE),
        torch.tensor(explicit_energy, dtype=COMPUTE_DTYPE),
        atol=args.atol,
        rtol=args.rtol,
    ).item()
    stable_energy_close = torch.isclose(
        torch.tensor(distributed_energy, dtype=torch.float64),
        torch.tensor(explicit_stable_energy, dtype=torch.float64),
        atol=args.stable_energy_atol,
        rtol=0.0,
    ).item()
    energy_close = (
        stable_energy_close if args.energy_mode == "stable" else legacy_energy_close
    )
    force_close = torch.allclose(
        distributed_forces,
        explicit_forces,
        atol=args.atol,
        rtol=args.rtol,
    )
    calculator_energy_close = bool(
        torch.isclose(
            torch.tensor(distributed_energy, dtype=COMPUTE_DTYPE),
            torch.tensor(calculator_energy, dtype=COMPUTE_DTYPE),
            atol=args.atol,
            rtol=args.rtol,
        ).item()
    )
    calculator_forces_close = bool(
        torch.allclose(
            distributed_forces, calculator_forces, atol=args.atol, rtol=args.rtol
        )
    )
    if not (energy_close and force_close):
        classification = (
            "stable_energy_mismatch"
            if args.energy_mode == "stable"
            else "post_graph_model_mismatch"
        )
    elif args.energy_mode == "legacy" and args.model_family == "off" and not (
        calculator_energy_close and calculator_forces_close
    ):
        classification = "public_calculator_mismatch"
    else:
        classification = "parity_ok"
    explicit_energy_abs = abs(distributed_energy - explicit_energy)
    calculator_energy_abs = abs(distributed_energy - calculator_energy)
    core_model_energy_abs = abs(distributed_energy - core_model_energy)
    explicit_force_delta = distributed_forces - explicit_forces
    calculator_force_delta = distributed_forces - calculator_forces
    core_model_force_delta = distributed_forces - core_model_forces
    core_edge_rows = artifact["reference"].get("core_model_edge_rows")
    edge_comparison = (
        compare_graph_edges(
            explicit_reference["edge_rows"],
            core_edge_rows,
            prepared.state.global_molecular_info.atom_positions,
            prepared.sim.cell,
            float(prepared.adapter.cutoff),
        )
        if core_edge_rows is not None
        else None
    )
    calculator_edge_rows = artifact["reference"].get("calculator_graph_edge_rows")
    calculator_edge_comparison = (
        compare_graph_edges(
            explicit_reference["edge_rows"], calculator_edge_rows,
            prepared.state.global_molecular_info.atom_positions,
            prepared.sim.cell, float(prepared.adapter.cutoff),
        ) if calculator_edge_rows is not None else None
    )
    if edge_comparison is not None and any(
        edge_comparison[side]["inside_cutoff_by_more_than_1e_4_angstrom"]
        for side in ("explicit_only", "core_only")
    ):
        classification = "native_graph_interior_mismatch"
    summary = {
        "system": prepared.raw["name"],
        "source_path": prepared.raw.get(
            "source_path", f"builtin://{prepared.raw['name']}"
        ),
        "num_atoms": num_atoms,
        "model": prepared.model_id,
        "reference_artifact": artifact["metadata"],
        "mismatch_classification": classification,
        "parity": {
            "distributed_energy": distributed_energy,
            "summed_owned_local_energy": distributed_energy,
            "explicit_dynamis_graph_energy": explicit_energy,
            "explicit_stable_energy": explicit_stable_energy,
            "explicit_stable_minus_node_sum": (
                explicit_stable_energy - explicit_node_energy_sum
            ),
            "explicit_legacy_minus_stable_energy": (
                explicit_energy - explicit_stable_energy
            ),
            "distributed_minus_explicit_stable_energy": (
                distributed_energy - explicit_stable_energy
            ),
            "explicit_stable_energy_abs_error": abs(
                distributed_energy - explicit_stable_energy
            ),
            "stable_reference_gap_reduction": (
                abs(distributed_energy - explicit_energy)
                - abs(distributed_energy - explicit_stable_energy)
            ),
            "core_model_stable_energy": core_model_stable_energy,
            "distributed_minus_core_model_stable_energy": (
                distributed_energy - core_model_stable_energy
                if core_model_stable_energy is not None else None
            ),
            "explicit_energy_tensor_dtype": explicit_reference["energy_tensor_dtype"],
            "explicit_node_energy_tensor_dtype": explicit_reference[
                "node_energy_tensor_dtype"
            ],
            "explicit_node_energy_sum": explicit_node_energy_sum,
            "explicit_node_energy_sum_float32": explicit_node_energy_sum_float32,
            "explicit_node_sum_float64_minus_float32": (
                explicit_node_energy_sum - explicit_node_energy_sum_float32
            ),
            "explicit_node_sum_minus_model_energy": (
                explicit_node_energy_sum - explicit_energy
            ),
            "distributed_minus_explicit_node_sum": (
                distributed_energy - explicit_node_energy_sum
            ),
            "distributed_minus_explicit_node_sum_float32": (
                distributed_energy - explicit_node_energy_sum_float32
            ),
            "explicit_interaction_energy": explicit_interaction_energy,
            "explicit_node_e0_sum": explicit_node_e0_sum,
            "explicit_node_e0_sum_float32": explicit_node_e0_sum_float32,
            "explicit_node_e0_tensor_dtype": explicit_reference[
                "node_e0_tensor_dtype"
            ],
            "explicit_node_e0_model_style_scatter_sum": (
                explicit_node_e0_model_style_scatter_sum
            ),
            "explicit_node_e0_scatter_minus_sum_float64": (
                explicit_node_e0_model_style_scatter_sum - explicit_node_e0_sum
            ),
            "explicit_node_e0_scatter_minus_model_e0": (
                explicit_node_e0_model_style_scatter_sum
                - (explicit_energy - explicit_interaction_energy)
            ),
            "explicit_node_interaction_sum": explicit_node_interaction_sum,
            "explicit_node_interaction_sum_float32": (
                explicit_node_interaction_sum_float32
            ),
            "explicit_node_interaction_minus_model_interaction": (
                explicit_node_interaction_sum - explicit_interaction_energy
            ),
            "explicit_node_e0_minus_model_e0": (
                explicit_node_e0_sum
                - (explicit_energy - explicit_interaction_energy)
            ),
            "calculator_mace_energy": calculator_energy,
            "calculator_controlled_per_atom_energy": calculator_controlled_energy,
            "calculator_legacy_minus_controlled_energy": (
                calculator_energy - calculator_controlled_energy
                if calculator_controlled_energy is not None else None
            ),
            "distributed_minus_calculator_controlled_energy": (
                distributed_energy - calculator_controlled_energy
                if calculator_controlled_energy is not None else None
            ),
            "core_model_mace_energy": core_model_energy,
            "distributed_minus_explicit_energy": distributed_energy - explicit_energy,
            "distributed_minus_calculator_energy": distributed_energy
            - calculator_energy,
            "distributed_minus_core_model_energy": distributed_energy
            - core_model_energy,
            "explicit_energy_abs_error": explicit_energy_abs,
            "calculator_energy_abs_error": calculator_energy_abs,
            "core_model_energy_abs_error": core_model_energy_abs,
            "explicit_energy_abs_error_per_atom": explicit_energy_abs / num_atoms,
            "calculator_energy_abs_error_per_atom": calculator_energy_abs / num_atoms,
            "core_model_energy_abs_error_per_atom": core_model_energy_abs / num_atoms,
            "explicit_force_mae": float(explicit_force_delta.abs().mean()),
            "explicit_force_max_abs": float(explicit_force_delta.abs().max()),
            "calculator_force_mae": float(calculator_force_delta.abs().mean()),
            "calculator_force_max_abs": float(calculator_force_delta.abs().max()),
            "core_model_force_mae": float(core_model_force_delta.abs().mean()),
            "core_model_force_max_abs": float(core_model_force_delta.abs().max()),
            "explicit_energy_close": bool(legacy_energy_close),
            "explicit_stable_energy_close": bool(stable_energy_close),
            "selected_reference_energy_close": bool(energy_close),
            "explicit_forces_close": bool(force_close),
            "calculator_energy_close": calculator_energy_close,
            "calculator_forces_close": calculator_forces_close,
            "explicit_edge_count": int(explicit_reference["edge_count"]),
            "core_model_edge_count": int(
                artifact["reference"]["core_model_edge_count"]
            ),
            # Reference-vs-reference diagnostics, scaled so the whole parity
            # block is reported in one unit system.
            "calculator_minus_core_model_energy": float(
                artifact["reference"]["calculator_minus_core_model_energy"]
            )
            * EV_TO_MD,
            "calculator_minus_core_model_force_mae": float(
                artifact["reference"]["calculator_minus_core_model_force_mae"]
            )
            * EV_TO_MD,
            "calculator_minus_core_model_force_max_abs": float(
                artifact["reference"]["calculator_minus_core_model_force_max_abs"]
            )
            * EV_TO_MD,
        },
        "ranks": [
            {key: value for key, value in record.items()
             if key not in {"owned_edge_rows", "owned_node_energy"}}
            for record in records
        ],
    }
    if args.energy_mode == "stable":
        distributed_baseline = sum(
            item["owned_baseline_energy"] for item in records
        )
        distributed_interaction = sum(
            item["owned_interaction_energy"] for item in records
        )
        summary["parity"].update(
            distributed_baseline_energy=distributed_baseline,
            distributed_interaction_energy=distributed_interaction,
            distributed_minus_explicit_baseline=(
                distributed_baseline - explicit_node_e0_sum
            ),
            distributed_minus_explicit_interaction=(
                distributed_interaction - explicit_node_interaction_sum
            ),
            distributed_total_minus_components=(
                distributed_energy - distributed_baseline - distributed_interaction
            ),
        )
    if edge_comparison is not None:
        summary["edge_comparison"] = edge_comparison
    if calculator_edge_comparison is not None:
        summary["calculator_edge_comparison"] = calculator_edge_comparison
    summary["standalone_diagnostics"] = artifact.get("diagnostics", {})
    core_node_values = artifact["reference"].get("core_model_node_energy_values")
    if core_node_values is not None:
        explicit_node_values = explicit_reference["node_energy_values"]
        if core_node_values.shape != explicit_node_values.shape:
            raise ValueError("Core and explicit node-energy arrays have different shapes.")
        node_delta = (explicit_node_values - core_node_values).double() * EV_TO_MD
        summary["parity"]["explicit_node_minus_core_node_sum"] = float(
            node_delta.sum()
        )
        summary["parity"]["explicit_vs_core_node_energy_mae"] = float(
            node_delta.abs().mean()
        )
        summary["parity"]["explicit_vs_core_node_energy_max_abs"] = float(
            node_delta.abs().max()
        )
    if benchmark is not None:
        calculator_benchmark = artifact["calculator_benchmark"]
        core_model_benchmark = artifact["core_model_benchmark"]
        calculator_stats = calculator_benchmark["timings"]
        core_model_stats = core_model_benchmark["timings"]
        calculator_speedup = (
            calculator_stats["median_ms_per_step"]
            / benchmark["distributed"]["median_ms_per_step"]
        )
        core_model_speedup = (
            core_model_stats["median_ms_per_step"]
            / benchmark["distributed"]["median_ms_per_step"]
        )
        benchmark["calculator_baseline"] = {
            "scope": calculator_benchmark["scope"],
            "timings": calculator_stats,
            "speedup_over_distributed": calculator_speedup,
            "parallel_efficiency": calculator_speedup / dist.get_world_size(),
        }
        benchmark["core_model_baseline"] = {
            "scope": core_model_benchmark["scope"],
            "timings": core_model_stats,
            "speedup_over_distributed": core_model_speedup,
            "parallel_efficiency": core_model_speedup / dist.get_world_size(),
        }
        summary["benchmark"] = benchmark
    parity = summary["parity"]
    definitions = (
        "calculator_mace_energy", "calculator_controlled_per_atom_energy",
        "core_model_mace_energy", "core_model_stable_energy",
        "explicit_dynamis_graph_energy", "explicit_stable_energy",
        "distributed_energy",
    )
    precision = (
        "explicit_energy_tensor_dtype", "explicit_node_energy_tensor_dtype",
        "explicit_node_e0_tensor_dtype",
    )
    groups: Dict[str, Dict[str, Any]] = {
        "configuration_and_precision": {
            "model": summary["model"], "num_atoms": num_atoms,
            "energy_unit": "amu * Angstrom^2 / fs^2",
            "reference_precision": artifact.get("precision"),
            **{key: parity[key] for key in precision if key in parity},
        },
        "energy_definitions": {key: parity[key] for key in definitions if key in parity},
        "matched_parity_comparisons": {},
        "reduction_decomposition": {},
        "per_atom_and_force_parity": {},
        "graph_diagnostics": {},
        "repeatability": artifact.get("diagnostics", {}).get("calculator_repeatability"),
    }
    for key, value in parity.items():
        if key in definitions or key in precision:
            continue
        if "force" in key or "node_energy_mae" in key or "node_energy_max_abs" in key:
            target = "per_atom_and_force_parity"
        elif "edge_count" in key:
            target = "graph_diagnostics"
        elif any(word in key for word in (
            "baseline", "interaction", "node_sum", "node_e0", "node_energy_sum",
            "scatter", "components", "legacy_minus_controlled", "legacy_minus_stable",
            "float32", "float64",
        )):
            target = "reduction_decomposition"
        else:
            target = "matched_parity_comparisons"
        groups[target][key] = value
    calculator_core = artifact.get("diagnostics", {}).get("calculator_core") or {}
    for key, value in calculator_core.items():
        if "edge_count" in key:
            target = "graph_diagnostics"
        elif "force" in key or "node_energy_mae" in key or "node_energy_max_abs" in key:
            target = "per_atom_and_force_parity"
        else:
            target = "matched_parity_comparisons"
        groups[target][f"reference.{key}"] = value
    groups["graph_diagnostics"]["explicit_vs_core"] = edge_comparison
    groups["graph_diagnostics"]["explicit_vs_calculator"] = calculator_edge_comparison
    if args.graph_edge_test:
        graph_audit = build_owned_graph_audit(prepared, records, explicit_reference, artifact)
        summary["owned_graph_audit"] = graph_audit
        interior = graph_audit["assembled_vs_native_core_by_band"]["interior"]
        if interior["explicit_only"]["count"] or interior["core_only"]["count"]:
            summary["mismatch_classification"] = "native_graph_interior_mismatch"
        groups["graph_diagnostics"]["assembled_owned_edges"] = {
            key: value for key, value in graph_audit.items() if key != "node_parity"
        }
        groups["per_atom_and_force_parity"].update(graph_audit["node_parity"])
    groups["reduction_decomposition"]["float64_diagnostic_ev"] = (
        artifact.get("diagnostics", {}).get("float64")
    )
    summary["report_sections"] = groups
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("=" * 60)
    print("Dynamis MACE Explicit-Graph Experiment")
    print("=" * 60)
    print(f"system: {summary['system']}")
    print(f"source_path: {summary['source_path']}")
    print(f"mismatch_classification: {summary['mismatch_classification']}")
    for title, values in summary["report_sections"].items():
        if values is None:
            continue
        print("\n" + "-" * 60 + f"\n{title.replace('_', ' ').title()}\n" + "-" * 60)
        for key, value in values.items():
            if key == "assembled_owned_edges":
                print(f"assembled_owned_edge_count: {value['assembled_owned_edge_count']}")
                print(f"nonperiodic_outside_cell: {value['nonperiodic_outside_cell']}")
                for source in ("native_core", "public_calculator", "explicit_dynamis"):
                    bands = value.get(f"assembled_vs_{source}_by_band")
                    if bands is None:
                        continue
                    for band, comparison in bands.items():
                        print(
                            f"assembled_vs_{source}.{band}: "
                            f"owned={comparison['explicit_raw_count']}, "
                            f"reference={comparison['core_raw_count']}, "
                            f"owned_only={comparison['explicit_only']['count']}, "
                            f"reference_only={comparison['core_only']['count']}"
                        )
                continue
            print(f"{key}: {value}")
    print("\n" + "-" * 60 + "\nPer-rank contributions\n" + "-" * 60)
    for record in summary["ranks"]:
        print(
            f"rank {record['rank']}: local_atoms={len(record['local_global_indices'])}, "
            f"owned_local_energy={record['owned_local_energy']}, "
            f"energy_dtype={record['owned_local_energy_dtype']}, "
            f"model_dtype={record['model_parameter_dtype']}, "
            f"baseline={record.get('owned_baseline_energy')}, "
            f"interaction={record.get('owned_interaction_energy')}"
        )
    if "benchmark" in summary:
        benchmark = summary["benchmark"]
        print("\n" + "-" * 60 + "\nPerformance Benchmark\n" + "-" * 60)
        print(f"scope: {benchmark['scope']}")
        print(f"device_name: {benchmark['device_name']}")
        print(f"timing_method: {benchmark['timing_method']}")
        print(f"world_size: {benchmark['world_size']}")
        print(f"warmup_steps: {benchmark['warmup_steps']}")
        print(f"iterations_per_sample: {benchmark['iterations_per_sample']}")
        print(f"num_samples: {benchmark['num_samples']}")
        print(
            f"distributed_median_ms_per_step: {benchmark['distributed']['median_ms_per_step']}"
        )
        print("\nPer-rank stage diagnostics:")
        for profile in benchmark["per_rank_stage_profiles"]:
            workload = profile["workload"]
            print(
                f"rank {profile['rank']}: owned={workload['owned_atoms']}, "
                f"halos={workload['received_halo_atoms']}, total={workload['total_local_plus_halo_atoms']}, "
                f"pre_filter_edges={workload['pre_filter_graph_edges']}, "
                f"owned_destination_edges={workload['owned_destination_edges']}"
            )
            print(
                f"  send_rows_per_peer={workload['send_rows_per_peer']}, "
                f"receive_rows_per_peer={workload['receive_rows_per_peer']}, "
                f"feature_width={workload['feature_width']}"
            )
            print(
                "  feature_bytes: "
                f"forward_send={workload['forward_feature_send_bytes']}, "
                f"forward_receive={workload['forward_feature_receive_bytes']}, "
                f"backward_send={workload['backward_feature_send_bytes']}, "
                f"backward_receive={workload['backward_feature_receive_bytes']}"
            )
            for stage_name, timing in profile["timings"].items():
                print(
                    f"  {stage_name}_median_ms={timing['median_ms_per_step']}, "
                    f"fraction={timing['fraction_of_profiled_total']}"
                )
        print(f"calculator_scope: {benchmark['calculator_baseline']['scope']}")
        print(
            f"calculator_median_ms_per_step: {benchmark['calculator_baseline']['timings']['median_ms_per_step']}"
        )
        print(
            f"speedup_calculator_over_distributed: {benchmark['calculator_baseline']['speedup_over_distributed']}"
        )
        print(
            f"calculator_parallel_efficiency: {benchmark['calculator_baseline']['parallel_efficiency']}"
        )
        print(f"core_model_scope: {benchmark['core_model_baseline']['scope']}")
        print(
            f"core_model_median_ms_per_step: {benchmark['core_model_baseline']['timings']['median_ms_per_step']}"
        )
        print(
            f"speedup_core_model_over_distributed: {benchmark['core_model_baseline']['speedup_over_distributed']}"
        )
        print(
            f"core_model_parallel_efficiency: {benchmark['core_model_baseline']['parallel_efficiency']}"
        )
        if "distributed_initialize_buffers" in benchmark:
            print(
                "distributed_initialize_buffers_median_ms: "
                f"{benchmark['distributed_initialize_buffers']['median_ms_per_step']}"
            )


def graph_edge_test_requested(args: argparse.Namespace) -> bool:
    return bool(args.graph_edge_test or args.strict_graph_interior)


def run_distributed(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    # A strict edge check necessarily performs the graph audit. Accept the
    # strict flag on its own so diagnostic shell scripts cannot fail before
    # model execution merely because the companion flag was omitted.
    args.graph_edge_test = graph_edge_test_requested(args)
    prepared = prepare_distributed(args)
    check_off_medium_fixture = (
        args.model_family == "off"
        and args.model == "medium"
        and args.xyz_path is None
        and args.builtin_system == "natural_chain12"
        and dist.get_world_size() == 2
    )
    if check_off_medium_fixture:
        assert len(prepared.adapter.model.interactions) == 2
        assert sum(
            prepared.adapter.runtime._should_exchange_after_product(index)
            for index in range(len(prepared.adapter.model.interactions))
        ) == 1
        assert prepared.adapter.exchange_enabled
        assert prepared.adapter.partition_validation["total_recv"] > 0
    artifact_path = resolve_reference_artifact_path(args)
    artifact = load_and_validate_reference_artifact(
        artifact_path,
        prepared.raw,
        args,
        require_standalone_benchmark=args.benchmark,
        expected_head_name=prepared.adapter.head_name,
    )
    _, _, records = run_distributed_adapter_step(prepared, graph_edge_test=args.graph_edge_test)
    if check_off_medium_fixture:
        num_atoms = len(prepared.raw["atomic_numbers"])
        assert all(
            0 < len(record["local_global_indices"]) < num_atoms
            for record in records
        )
        assert prepared.adapter.runtime._feature_width is not None
    benchmark = benchmark_distributed(prepared, args) if args.benchmark else None
    if dist.get_rank() != 0:
        return None
    explicit_reference = build_explicit_global_reference(prepared, args)
    summary = build_summary(
        prepared, records, explicit_reference, artifact, args, benchmark=benchmark
    )
    print_summary(summary)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.strict_parity:
        assert summary["mismatch_classification"] == "parity_ok", summary[
            "mismatch_classification"
        ]
    if args.strict_graph_interior:
        bands = summary["owned_graph_audit"]["assembled_vs_native_core_by_band"]
        interior = bands["interior"]
        extra = interior["explicit_only"]["count"]
        missing = interior["core_only"]["count"]
        if extra or missing:
            raise AssertionError(
                f"Assembled owned graph has {missing} missing and {extra} extra "
                "strictly interior directed edges versus native MACE; "
                "inspect owned_graph_audit in the output JSON."
            )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "run-distributed":
            run_distributed(args)
        else:  # pragma: no cover - argparse enforces the command
            raise ValueError(f"Unsupported command: {args.command!r}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _require_reference_env(*names: str) -> str:
    import pytest

    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    pytest.skip(
        "Set a detached standalone reference artifact env var before running parity tests. "
        f"Tried: {', '.join(names)}"
    )


def test_mace_model_parity_builtin() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        import pytest

        pytest.skip("Run with torchrun --nproc-per-node=2")
    artifact = _require_reference_env(
        "DISTMACE_REFERENCE_BUILTIN_ARTIFACT",
        "DISTMACE_REFERENCE_ARTIFACT",
        "DISTMACE_REFERENCE_DEFAULT_ARTIFACT",
    )
    args = build_parser().parse_args(
        ["run-distributed", "--reference-artifact", artifact, "--strict-parity"]
    )
    run_distributed(args)


def test_mace_model_parity_halo_fixture() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        import pytest

        pytest.skip("Run with torchrun --nproc-per-node=2")
    artifact = _require_reference_env(
        "DISTMACE_REFERENCE_HALO_ARTIFACT",
        "DISTMACE_REFERENCE_ARTIFACT",
        "DISTMACE_REFERENCE_DEFAULT_ARTIFACT",
    )
    args = build_parser().parse_args(
        [
            "run-distributed",
            "--xyz-path",
            "nonperiodic_compact_chain12.xyz",
            "--reference-artifact",
            artifact,
            "--strict-parity",
        ]
    )
    run_distributed(args)


def test_off23_medium_parity_builtin() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        import pytest

        pytest.skip("Run with torchrun --nproc-per-node=2")
    artifact = _require_reference_env("DISTMACE_OFF_REFERENCE_ARTIFACT")
    args = build_parser().parse_args(
        [
            "run-distributed",
            "--model-family", "off",
            "--reference-artifact", artifact,
            "--strict-parity",
        ]
    )
    run_distributed(args)


def test_off23_medium_stable_energy_builtin() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        import pytest

        pytest.skip("Run with torchrun --nproc-per-node=2")
    artifact = _require_reference_env("DISTMACE_OFF_REFERENCE_ARTIFACT")
    args = build_parser().parse_args(
        [
            "run-distributed",
            "--model-family", "off",
            "--energy-mode", "stable",
            "--reference-artifact", artifact,
            "--strict-parity",
        ]
    )
    summary = run_distributed(args)
    if summary is not None:
        assert abs(summary["parity"]["explicit_stable_minus_node_sum"]) < 1e-9
        assert summary["parity"]["explicit_stable_energy_close"]


def test_mace_model_reinitialize_buffers() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        import pytest

        pytest.skip("Run with torchrun --nproc-per-node=2")
    args = build_parser().parse_args(["run-distributed"])
    prepared = prepare_distributed(args)
    energy0, forces0 = prepared.adapter(prepared.state)
    prepared.adapter.initialize_buffers(
        prepared.state.local_partition_info,
        prepared.device,
        dist.get_world_size(),
    )
    energy1, forces1 = prepared.adapter(prepared.state)
    assert torch.isclose(energy0.cpu(), energy1.cpu(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(forces0.cpu(), forces1.cpu(), atol=1e-6, rtol=1e-6)


def test_mace_model_rejects_specialized_scaleshift_subclass(monkeypatch) -> None:
    import importlib
    import pytest

    mace_module = importlib.import_module("dynamis_model.nnp.mace")

    class OrdinaryScaleShiftMACE:
        r_max = 5.0
        atomic_numbers = (1, 6)
        node_embedding = spherical_harmonics = radial_embedding = object()
        interactions = products = readouts = (object(), object())
        atomic_energies_fn = scale_shift = object()

    class SpecializedScaleShiftMACE(OrdinaryScaleShiftMACE):
        pass

    monkeypatch.setattr(mace_module, "ScaleShiftMACE", OrdinaryScaleShiftMACE)
    MACEModel._validate_model_contract(OrdinaryScaleShiftMACE())
    with pytest.raises(TypeError, match="specialized model class SpecializedScaleShiftMACE"):
        MACEModel._validate_model_contract(SpecializedScaleShiftMACE())


def test_compare_graph_edges_preserves_direction_multiplicity_and_cutoff() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cell = torch.eye(3) * 20.0
    explicit = torch.tensor(
        [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 2, 0, 0, 0]],
        dtype=torch.int32,
    )
    core = torch.tensor(
        [[0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [2, 0, 0, 0, 0]],
        dtype=torch.int32,
    )
    result = compare_graph_edges(explicit, core, positions, cell, cutoff=5.0)
    assert result["explicit_duplicate_count"] == 1
    assert result["common_count_with_multiplicity"] == 2
    assert result["explicit_only"]["count"] == 1
    assert result["explicit_only"]["within_1e_4_angstrom_of_cutoff"] == 1
    assert result["core_only"]["count"] == 1
    assert result["core_only"]["inside_cutoff_by_more_than_1e_4_angstrom"] == 1


def test_stable_runtime_reduces_owned_components_in_float64() -> None:
    runtime = MACEStableEnergyRuntime(torch.nn.Identity())
    baseline = torch.tensor([-1.0e8, -1.0, 1.0e8, 99.0], dtype=torch.float32)
    interaction = torch.tensor(
        [0.25, 0.5, 0.75, 100.0], dtype=torch.float32, requires_grad=True
    )
    outputs = runtime._reduce_owned_energy(baseline, interaction, num_local=3)
    expected = baseline[:3].double().sum() + interaction[:3].double().sum()
    legacy_energy = (baseline[:3] + interaction[:3]).sum()
    assert outputs["owned_local_energy"].dtype == torch.float64
    assert torch.equal(outputs["owned_local_energy"], expected)
    assert abs(float(outputs["owned_local_energy"] - expected)) < abs(
        float(legacy_energy.double() - expected)
    )
    assert torch.equal(outputs["owned_baseline_energy"], torch.tensor(-1.0))
    outputs["owned_local_energy"].backward()
    assert torch.equal(interaction.grad, torch.tensor([1.0, 1.0, 1.0, 0.0]))


def test_stable_energy_mode_is_opt_in() -> None:
    parser = build_parser()
    assert parser.parse_args(["run-distributed"]).energy_mode == "legacy"
    assert (
        parser.parse_args(["run-distributed", "--energy-mode", "stable"]).energy_mode
        == "stable"
    )


def test_graph_edge_audit_flags_are_opt_in() -> None:
    args = build_parser().parse_args(["run-distributed"])
    assert not args.graph_edge_test
    assert not args.strict_graph_interior
    args = build_parser().parse_args(
        ["run-distributed", "--graph-edge-test", "--strict-graph-interior"]
    )
    assert args.graph_edge_test and args.strict_graph_interior


def test_strict_graph_interior_implies_graph_audit() -> None:
    args = build_parser().parse_args(["run-distributed", "--strict-graph-interior"])
    assert args.strict_graph_interior and not args.graph_edge_test
    assert graph_edge_test_requested(args)


def test_edge_distance_bands_include_exact_cutoff() -> None:
    positions = torch.tensor(
        [[0, 0, 0], [4.999, 0, 0], [5.0, 0, 0], [5.001, 0, 0]],
        dtype=torch.float64,
    )
    rows = torch.tensor([[0, i, 0, 0, 0] for i in (1, 2, 3)], dtype=torch.int32)
    bands = edge_distance_bands(rows, positions, torch.eye(3), 5.0)
    assert bands["interior"].shape[0] == 1
    assert bands["boundary"].shape[0] == 1
    assert bands["exterior"].shape[0] == 1


def test_nonperiodic_outside_cell_diagnostic_detects_wrap_risk() -> None:
    positions = torch.tensor([[-1.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    result = nonperiodic_outside_cell_diagnostics(
        positions, torch.eye(3) * 10.0, torch.tensor([False, False, False])
    )
    assert result["atom_count"] == 1
    assert result["atom_count_by_axis"] == [1, 0, 0]
    periodic_result = nonperiodic_outside_cell_diagnostics(
        positions, torch.eye(3) * 10.0, torch.tensor([True, False, False])
    )
    assert periodic_result["atom_count"] == 0


def test_nonperiodic_graph_preserves_edges_across_coordinate_origin() -> None:
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA radius-graph regression requires a GPU.")
    device = torch.device("cuda")
    sim = SimulationConfig()
    sim.cell = torch.eye(3, device=device) * 30.0
    positions = torch.tensor(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device
    )
    expected = {(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)}
    for pbc in ((False, False, False), (False, True, False)):
        sim.pbc = torch.tensor(pbc, dtype=torch.bool, device=device)
        for shifted in (positions, positions + 5.0):
            edges, shifts = radius_graph(shifted, sim, cutoff=3.0)
            assert edge_records(edges, shifts) == expected


def test_mace_loader_uses_family_default_and_preserves_explicit_name(monkeypatch) -> None:
    import importlib

    mace_module = importlib.import_module("dynamis_model.nnp.mace")
    calls = []

    def loader(family):
        def load(**kwargs):
            calls.append((family, kwargs))
            return torch.nn.Linear(1, 1).double()

        return load

    monkeypatch.setattr(mace_module, "_require_mace", lambda: None)
    monkeypatch.setattr(mace_module, "mace_mp", loader("mp"))
    monkeypatch.setattr(mace_module, "mace_off", loader("off"))
    mace_module.load_raw_mace_model("mp", device="cpu")
    mace_module.load_raw_mace_model("off", device="cpu")
    model = mace_module.load_raw_mace_model("off", "small", device="cpu")
    assert [kwargs["model"] for _, kwargs in calls] == [
        "medium-mpa-0", "medium", "small"
    ]
    assert all(kwargs["return_raw_model"] for _, kwargs in calls)
    assert all(kwargs["default_dtype"] == "float32" for _, kwargs in calls)
    assert next(model.parameters()).dtype == torch.float32


def test_reference_artifact_rejects_missing_benchmark_stats(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.pt"
    raw = natural_chain12()
    torch.save(
        {
            "metadata": {
                "artifact_version": ARTIFACT_VERSION,
                "signature": _reference_signature(
                    raw, "mp", "medium-mpa-0", "Default", False
                ),
            },
            "reference": {
                "calculator_energy": 0.0,
                "calculator_forces": torch.zeros(1, 3, dtype=COMPUTE_DTYPE),
                "core_model_energy": 0.0,
                "core_model_forces": torch.zeros(1, 3, dtype=COMPUTE_DTYPE),
            },
            "calculator_benchmark": None,
            "core_model_benchmark": None,
        },
        artifact_path,
    )
    args = build_parser().parse_args(
        [
            "run-distributed",
            "--reference-artifact",
            str(artifact_path),
            "--benchmark",
        ]
    )
    import pytest

    with pytest.raises(ValueError, match="timing stats"):
        load_and_validate_reference_artifact(
            artifact_path,
            raw,
            args,
            require_standalone_benchmark=True,
            expected_head_name="Default",
        )


def test_reference_artifact_reports_cueq_identity_mismatch(tmp_path: Path) -> None:
    import pytest

    raw = natural_chain12()
    artifact_path = tmp_path / "artifact.pt"
    torch.save(
        {
            "metadata": {
                "artifact_version": ARTIFACT_VERSION,
                "system_name": raw["name"],
                "source_path": f"builtin://{raw['name']}",
                "num_atoms": len(raw["atomic_numbers"]),
                "model_family": "off",
                "model_name": "medium",
                "head_name": "default",
                "enable_cueq": True,
                "signature": _reference_signature(
                    raw, "off", "medium", "default", True
                ),
            }
        },
        artifact_path,
    )
    args = build_parser().parse_args(["run-distributed", "--model-family", "off"])
    with pytest.raises(ValueError, match="enable_cueq: artifact=True, requested=False"):
        load_and_validate_reference_artifact(
            artifact_path, raw, args, expected_head_name="default"
        )


def test_reference_artifact_uses_distributed_model_head(
    tmp_path: Path, monkeypatch
) -> None:
    raw = natural_chain12()
    artifact_path = tmp_path / "artifact.pt"
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "signature": _reference_signature(raw, "off", "medium", "default", True),
    }
    torch.save(
        {
            "metadata": metadata,
            "reference": {
                "calculator_energy": 0.0,
                "calculator_forces": torch.zeros(12, 3, dtype=COMPUTE_DTYPE),
                "core_model_energy": 0.0,
                "core_model_forces": torch.zeros(12, 3, dtype=COMPUTE_DTYPE),
            },
        },
        artifact_path,
    )
    args = build_parser().parse_args(
        ["run-distributed", "--model-family", "off", "--enable-cueq"]
    )

    def fail_if_reloaded(*args, **kwargs):
        raise AssertionError("Validation must reuse the distributed model head.")

    monkeypatch.setitem(globals(), "load_raw_mace_model", fail_if_reloaded)
    loaded = load_and_validate_reference_artifact(
        artifact_path, raw, args, expected_head_name="default"
    )
    assert loaded["metadata"]["signature"] == metadata["signature"]


if __name__ == "__main__":
    main()
