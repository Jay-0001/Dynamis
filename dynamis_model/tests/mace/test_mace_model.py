"""Two-rank MACE parity against an independently generated raw-core artifact."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from dynamis_model.comm.feature_push import FeaturePushToExchange
from dynamis_model.nnp.mace import (
    COMPUTE_DTYPE, MACEModel, MACEScaleShiftRuntime,
    load_raw_mace_model, resolve_mace_model_name,
    resolve_model_head_index, resolve_model_head_name,
)
from dynamis_model.units import EV_PER_A_TO_AMU_A_PER_FS2 as EV_TO_MD
from dynamis_state import (
    DynamisState, MolecularInfo, SFCCurveType, SFCPartitioner,
    SimulationConfig, TorchDistEnv,
)
from dynamis_state.masses import TORCH_ATOMIC_MASSES

if __package__:
    from .mace_standalone_reference import (
        ARTIFACT_VERSION, load_raw_structure, natural_chain12, reference_signature,
    )
else:
    from mace_standalone_reference import (
        ARTIFACT_VERSION, load_raw_structure, natural_chain12, reference_signature,
    )

ENERGY_ATOL_MD = 1e-3
FORCE_ATOL_MD = 1e-5
FORCE_RTOL = 1e-5
GRAPH_BOUNDARY_BAND_ANGSTROM = 1e-4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-distributed")
    run.add_argument("--builtin-system", choices=["natural_chain12"], default="natural_chain12")
    run.add_argument("--xyz-path", type=Path)
    run.add_argument("--model-family", choices=["mp", "off"], default="mp")
    run.add_argument("--model", default=None)
    run.add_argument("--head-name", default="")
    run.add_argument("--enable-cueq", action="store_true")
    run.add_argument("--ghost-cutoff-buffer", type=float, default=0.5)
    run.add_argument("--reference-artifact", type=Path, required=True)
    run.add_argument("--strict-parity", action="store_true")
    run.add_argument("--graph-edge-test", action="store_true")
    run.add_argument("--strict-graph-interior", action="store_true")
    run.add_argument("--energy-atol", type=float, default=ENERGY_ATOL_MD)
    run.add_argument("--force-atol", type=float, default=FORCE_ATOL_MD)
    run.add_argument("--force-rtol", type=float, default=FORCE_RTOL)
    run.add_argument("--benchmark", action="store_true")
    run.add_argument("--benchmark-warmup", type=int, default=3)
    run.add_argument("--benchmark-iterations", type=int, default=10)
    run.add_argument("--benchmark-samples", type=int, default=5)
    run.add_argument("--output-json", type=Path)
    return parser


def build_state(raw, cutoff: float, ghost_buffer: float, torch_env: TorchDistEnv):
    device = torch_env.device
    numbers = torch.tensor(raw["atomic_numbers"], dtype=torch.long, device=device)
    positions = torch.tensor(raw["positions"], dtype=COMPUTE_DTYPE, device=device)
    global_mol = MolecularInfo()
    global_mol.num_atoms = len(numbers)
    global_mol.atom_positions = positions
    global_mol.atom_types = numbers
    global_mol.atom_velocities = torch.zeros_like(positions)
    if hasattr(global_mol, "atom_forces"):
        global_mol.atom_forces = torch.zeros_like(positions)
    if hasattr(global_mol, "atom_vertex_numbers"):
        global_mol.atom_vertex_numbers = torch.arange(len(numbers), device=device)
    if hasattr(global_mol, "atom_masses"):
        global_mol.atom_masses = TORCH_ATOMIC_MASSES.to(device)[numbers].float()

    sim = SimulationConfig()
    sim.cutoff = cutoff
    sim.ghost_cutoff_buffer = ghost_buffer
    sim.cell = torch.tensor(raw["cell"], dtype=COMPUTE_DTYPE, device=device)
    sim.pbc = torch.tensor(raw["pbc"], dtype=torch.bool, device=device)
    partitioner = SFCPartitioner(SFCCurveType.Morton)
    info = partitioner.partition(global_mol, torch_env.dist_env)
    partitioner.build_halo_maps(info, global_mol, torch_env.dist_env, sim)
    owned = info.partition_indices.long()
    local_mol = MolecularInfo()
    local_mol.num_atoms = owned.numel()
    local_mol.atom_positions = positions.index_select(0, owned)
    local_mol.atom_types = numbers.index_select(0, owned)
    local_mol.atom_velocities = torch.zeros_like(local_mol.atom_positions)
    if hasattr(local_mol, "atom_forces"):
        local_mol.atom_forces = torch.zeros_like(local_mol.atom_positions)
    if hasattr(local_mol, "atom_vertex_numbers"):
        local_mol.atom_vertex_numbers = owned
    if hasattr(local_mol, "atom_masses"):
        local_mol.atom_masses = TORCH_ATOMIC_MASSES.to(device)[local_mol.atom_types].float()

    state = DynamisState()
    state.dist_env = torch_env.dist_env
    state.sim_config = sim
    state.global_molecular_info = global_mol
    state.local_partition_info = info
    state.local_molecular_info = local_mol
    return state


def load_reference(path: Path, raw, args, head_name: str, cutoff: float):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    metadata = artifact.get("metadata", {})
    expected = {
        "artifact_version": ARTIFACT_VERSION,
        "model_family": args.model_family,
        "model_name": args.model,
        "head_name": head_name,
        "enable_cueq": args.enable_cueq,
        "num_atoms": len(raw["atomic_numbers"]),
        "signature": reference_signature(raw, args.model_family, args.model, head_name, args.enable_cueq),
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches or abs(float(metadata.get("cutoff_angstrom", -1)) - cutoff) > 1e-6:
        raise ValueError(f"Raw-core reference artifact is stale or incompatible: {mismatches or ['cutoff_angstrom']}")
    reference = artifact.get("reference", {})
    required = ("energy_ev", "forces_ev_per_angstrom", "edge_rows")
    if any(key not in reference for key in required):
        raise ValueError("Raw-core reference artifact is incomplete; regenerate it with the production writer.")
    if args.benchmark and artifact.get("benchmark") is None:
        raise ValueError("--benchmark requires a reference artifact generated with --benchmark.")
    return artifact


def owned_edge_rows(state: DynamisState, adapter: MACEModel, device: torch.device) -> torch.Tensor:
    owned = state.local_partition_info.partition_indices.long().to(device)
    if state.global_molecular_info.num_atoms > 2**24:
        raise ValueError("Float32 halo ID exchange cannot exactly represent the global atom IDs.")
    view = adapter._build_structural_view(state)
    graph = adapter._build_owned_graph(view, state)
    with torch.no_grad():
        ids = (
            FeaturePushToExchange.apply(
                owned.float().unsqueeze(1), state, adapter.atom_type_exchanger
            ).squeeze(1)
            if adapter.exchange_enabled else owned.float()
        )
    global_ids = ids.long()
    if ids.numel() != view.num_total or not torch.equal(ids, global_ids.float()):
        raise ValueError("Halo atom IDs are inconsistent with the graph view.")
    edges = global_ids[graph.edge_index]
    return torch.cat(
        (edges.T.long(), graph.unit_shifts.round().long()), dim=1
    ).int().cpu()


def _edge_bands(rows: torch.Tensor, raw, cutoff: float) -> dict[str, Counter]:
    rows = rows.long().reshape(-1, 5)
    if rows.numel() == 0:
        return {"interior": Counter(), "boundary": Counter(), "exterior": Counter()}
    positions = torch.tensor(raw["positions"], dtype=torch.float64)
    cell = torch.tensor(raw["cell"], dtype=torch.float64)
    distances = torch.linalg.vector_norm(
        positions[rows[:, 1]] - positions[rows[:, 0]] + rows[:, 2:].double() @ cell,
        dim=1,
    )
    interior = distances < cutoff - GRAPH_BOUNDARY_BAND_ANGSTROM
    exterior = distances > cutoff + GRAPH_BOUNDARY_BAND_ANGSTROM
    return {
        "interior": Counter(map(tuple, rows[interior].tolist())),
        "boundary": Counter(map(tuple, rows[~(interior | exterior)].tolist())),
        "exterior": Counter(map(tuple, rows[exterior].tolist())),
    }


def compare_owned_edges(records, native_rows: torch.Tensor, raw, cutoff: float):
    assembled = torch.tensor(
        [row for record in records for row in record["owned_edge_rows"]],
        dtype=torch.int32,
    ).reshape(-1, 5)
    assembled_bands = _edge_bands(assembled, raw, cutoff)
    native_bands = _edge_bands(native_rows, raw, cutoff)
    assembled_interior = assembled_bands["interior"]
    native_interior = native_bands["interior"]
    return {
        "assembled_directed_edges": int(assembled.shape[0]),
        "native_directed_edges": int(native_rows.shape[0]),
        "interior_extra": sum((assembled_interior - native_interior).values()),
        "interior_missing": sum((native_interior - assembled_interior).values()),
        "boundary_extra": sum((assembled_bands["boundary"] - native_bands["boundary"]).values()),
        "boundary_missing": sum((native_bands["boundary"] - assembled_bands["boundary"]).values()),
        "exterior_extra": sum(assembled_bands["exterior"].values()),
        "boundary_band_angstrom": GRAPH_BOUNDARY_BAND_ANGSTROM,
    }


def benchmark_adapter(adapter, state, device, args):
    if args.benchmark_warmup < 0 or args.benchmark_iterations <= 0 or args.benchmark_samples <= 0:
        raise ValueError("Benchmark counts require nonnegative warmup and positive iterations/samples.")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(args.benchmark_warmup):
        adapter(state)
    samples = []
    for _ in range(args.benchmark_samples):
        synchronize()
        start = time.perf_counter()
        for _ in range(args.benchmark_iterations):
            adapter(state)
        synchronize()
        elapsed = torch.tensor(
            (time.perf_counter() - start) * 1000.0 / args.benchmark_iterations,
            dtype=torch.float64, device=device,
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        samples.append(float(elapsed.cpu()))
    return {"median_ms_per_step": statistics.median(samples), "samples_ms": samples}


def run_distributed(args: argparse.Namespace):
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        raise RuntimeError("Launch the distributed MACE test with at least two torchrun ranks.")
    args.model = resolve_mace_model_name(args.model_family, args.model)
    torch_env = TorchDistEnv(is_distributed=True)
    device = torch_env.device
    raw = load_raw_structure(args)
    model = load_raw_mace_model(
        args.model_family, args.model, str(device), enable_cueq=args.enable_cueq
    ).to(device=device).eval()
    adapter = MACEModel(model, head_name=args.head_name or None)
    state = build_state(raw, adapter.cutoff, args.ghost_cutoff_buffer, torch_env)
    if not torch.isclose(torch.as_tensor(state.sim_config.cutoff), torch.as_tensor(adapter.cutoff)):
        raise ValueError("Simulation halo cutoff differs from checkpoint r_max.")
    adapter.initialize_buffers(state.local_partition_info, device, dist.get_world_size())
    artifact = load_reference(args.reference_artifact, raw, args, adapter.head_name, adapter.cutoff)

    energy, forces, components = adapter(state, return_energy_components=True)
    owned = state.local_partition_info.partition_indices.long().cpu()
    record = {
        "rank": dist.get_rank(),
        "atom_ids": owned.tolist(),
        "forces": forces.cpu().tolist(),
        "baseline_md": float(components["owned_baseline_energy"].cpu()),
        "interaction_md": float(components["owned_interaction_energy"].cpu()),
        "local_energy_md": float(energy.cpu()),
        "halo_atoms": adapter.partition_validation["total_recv"],
    }
    if args.graph_edge_test or args.strict_graph_interior:
        record["owned_edge_rows"] = owned_edge_rows(state, adapter, device).tolist()
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, record)
    timing = benchmark_adapter(adapter, state, device, args) if args.benchmark else None
    if dist.get_rank() != 0:
        return None

    n_atoms = len(raw["atomic_numbers"])
    assembled = torch.empty((n_atoms, 3), dtype=COMPUTE_DTYPE)
    visited = torch.zeros(n_atoms, dtype=torch.bool)
    for item in records:
        indices = torch.tensor(item["atom_ids"], dtype=torch.long)
        if visited[indices].any():
            raise ValueError("Distributed force records contain duplicate owned atoms.")
        assembled[indices] = torch.tensor(item["forces"], dtype=COMPUTE_DTYPE).reshape(-1, 3)
        visited[indices] = True
    if not visited.all():
        raise ValueError("Distributed force records omit owned atoms.")
    distributed_energy = sum(item["baseline_md"] for item in records) + sum(item["interaction_md"] for item in records)
    reference = artifact["reference"]
    reference_energy = float(reference["energy_ev"]) * EV_TO_MD
    reference_forces = reference["forces_ev_per_angstrom"].float() * EV_TO_MD
    energy_error = distributed_energy - reference_energy
    force_error = assembled - reference_forces
    force_ok = torch.allclose(assembled, reference_forces, atol=args.force_atol, rtol=args.force_rtol)
    energy_ok = abs(energy_error) <= args.energy_atol
    graph = (
        compare_owned_edges(records, reference["edge_rows"], raw, adapter.cutoff)
        if args.graph_edge_test or args.strict_graph_interior else None
    )
    summary = {
        "model": {"family": args.model_family, "checkpoint": args.model,
                  "head": adapter.head_name, "cueq": args.enable_cueq},
        "system": {"name": raw["name"], "num_atoms": n_atoms,
                   "cutoff_angstrom": adapter.cutoff},
        "energy": {"distributed_md": distributed_energy,
                   "raw_core_controlled_md": reference_energy,
                   "error_md": energy_error, "close": energy_ok},
        "forces": {"mae_md": float(force_error.abs().mean()),
                   "max_abs_md": float(force_error.abs().max()), "close": bool(force_ok)},
        "ranks": [{"rank": item["rank"], "owned_atoms": len(item["atom_ids"]),
                   "halo_atoms": item["halo_atoms"]} for item in records],
        "graph": graph,
        "benchmark": ({"distributed": timing, "raw_core": artifact["benchmark"]}
                      if timing else None),
    }
    if args.model_family == "off" and args.model == "medium" and args.xyz_path is None:
        if len(model.interactions) != 2 or sum(
            adapter.runtime._should_exchange_after_product(i)
            for i in range(len(model.interactions))
        ) != 1:
            raise AssertionError("OFF23 medium must have two interactions and one feature exchange.")
        if any(item["owned_atoms"] == 0 or item["halo_atoms"] == 0 for item in summary["ranks"]):
            raise AssertionError("The OFF23 halo fixture must place owned and halo atoms on both ranks.")
    print(json.dumps(summary, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.strict_parity and not (energy_ok and force_ok):
        raise AssertionError("MACE energy or force parity failed against the raw-core reference.")
    if args.strict_graph_interior and graph and (
        graph["interior_extra"] or graph["interior_missing"] or graph["exterior_extra"]
    ):
        raise AssertionError("Distributed owned graph differs from native MACE away from the cutoff boundary.")
    return summary


def test_stable_reduction_keeps_interaction_gradient():
    runtime = MACEScaleShiftRuntime(torch.nn.Identity())
    baseline = torch.tensor([-1.0e8, -1.0, 1.0e8, 99.0], dtype=torch.float32)
    interaction = torch.tensor([0.25, 0.5, 0.75, 100.0], requires_grad=True)
    result = runtime._reduce_owned_energy(baseline, interaction, num_local=3)
    assert result["owned_local_energy"].dtype == torch.float64
    assert result["owned_local_energy"] == baseline[:3].double().sum() + interaction[:3].double().sum()
    result["owned_local_energy"].backward()
    assert torch.equal(interaction.grad, torch.tensor([1.0, 1.0, 1.0, 0.0]))


def test_default_model_and_family_resolution():
    args = build_parser().parse_args(["run-distributed", "--reference-artifact", "ref.pt"])
    assert args.model_family == "mp"
    assert resolve_mace_model_name(args.model_family, args.model) == "medium-mpa-0"
    assert resolve_mace_model_name("off", None) == "medium"


def test_off_checkpoint_name_selection(monkeypatch):
    import dynamis_model.nnp.mace as mace_module

    requested = []

    class RawModel:
        def to(self, **kwargs):
            return self

    def load_off(**kwargs):
        requested.append(kwargs["model"])
        assert kwargs["return_raw_model"] is True
        return RawModel()

    monkeypatch.setattr(mace_module, "_MACE_IMPORT_ERROR", None)
    monkeypatch.setattr(mace_module, "mace_off", load_off)
    for name in (None, "small", "medium", "large"):
        load_raw_mace_model("off", name, device="cpu")
    assert requested == ["medium", "small", "medium", "large"]


def test_selected_head_is_explicit_and_validated():
    class Heads:
        heads = ["default", "special"]

    assert resolve_model_head_name(Heads()) == "default"
    assert resolve_model_head_index(Heads(), "special") == 1
    import pytest
    with pytest.raises(ValueError, match="not available"):
        resolve_model_head_name(Heads(), "missing")


def test_reference_artifact_rejects_stale_version_and_head(tmp_path):
    import pytest

    raw = natural_chain12()
    args = build_parser().parse_args(
        ["run-distributed", "--reference-artifact", str(tmp_path / "reference.pt")]
    )
    args.model = resolve_mace_model_name(args.model_family, args.model)
    metadata = {
        "artifact_version": ARTIFACT_VERSION - 1,
        "model_family": args.model_family,
        "model_name": args.model,
        "head_name": "default",
        "enable_cueq": False,
        "num_atoms": len(raw["atomic_numbers"]),
        "cutoff_angstrom": 6.0,
        "signature": reference_signature(raw, args.model_family, args.model, "default", False),
    }
    reference = {"energy_ev": 0.0, "forces_ev_per_angstrom": torch.zeros(12, 3),
                 "edge_rows": torch.empty((0, 5), dtype=torch.int32)}
    torch.save({"metadata": metadata, "reference": reference}, args.reference_artifact)
    with pytest.raises(ValueError, match="stale or incompatible"):
        load_reference(args.reference_artifact, raw, args, "default", 6.0)
    metadata["artifact_version"] = ARTIFACT_VERSION
    torch.save({"metadata": metadata, "reference": reference}, args.reference_artifact)
    with pytest.raises(ValueError, match="stale or incompatible"):
        load_reference(args.reference_artifact, raw, args, "special", 6.0)


def test_specialized_scaleshift_subclass_is_rejected(monkeypatch):
    import pytest
    import dynamis_model.nnp.mace as mace_module

    class Ordinary:
        r_max = atomic_numbers = node_embedding = spherical_harmonics = radial_embedding = object()
        interactions = products = readouts = (object(), object())
        atomic_energies_fn = scale_shift = object()

    class Specialized(Ordinary):
        pass

    monkeypatch.setattr(mace_module, "ScaleShiftMACE", Ordinary)
    MACEModel._validate_model_contract(Ordinary())
    with pytest.raises(TypeError, match="specialized model class Specialized"):
        MACEModel._validate_model_contract(Specialized())


def test_interior_graph_excludes_cutoff_boundary():
    raw = {"positions": [[0, 0, 0], [4, 0, 0], [5, 0, 0]],
           "cell": [[20, 0, 0], [0, 20, 0], [0, 0, 20]]}
    distributed = [{"owned_edge_rows": [[0, 1, 0, 0, 0], [0, 2, 0, 0, 0]]}]
    native = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.int32)
    result = compare_owned_edges(distributed, native, raw, 5.0)
    assert result["interior_extra"] == result["interior_missing"] == 0
    assert result["boundary_extra"] == 1
    reversed_native = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.int32)
    reversed_result = compare_owned_edges(distributed, reversed_native, raw, 5.0)
    assert reversed_result["interior_extra"] == 1
    assert reversed_result["interior_missing"] == 1


def test_reinitialize_buffers_preserves_energy_and_forces():
    import pytest

    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("Launch with torchrun and two CUDA ranks.")
    torch_env = TorchDistEnv(is_distributed=True)
    raw = natural_chain12()
    model = load_raw_mace_model(device=str(torch_env.device)).to(torch_env.device).eval()
    adapter = MACEModel(model)
    state = build_state(raw, adapter.cutoff, 0.5, torch_env)
    adapter.initialize_buffers(state.local_partition_info, torch_env.device, dist.get_world_size())
    before_energy, before_forces = adapter(state)
    adapter.initialize_buffers(state.local_partition_info, torch_env.device, dist.get_world_size())
    after_energy, after_forces = adapter(state)
    torch.testing.assert_close(after_energy, before_energy, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(after_forces, before_forces, atol=1e-6, rtol=1e-6)


def main():
    args = build_parser().parse_args()
    try:
        run_distributed(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
