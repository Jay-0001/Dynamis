"""Trace native MACE and the Dynamis adapter on one GPU with no halo atoms."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist

TEST_DIR = Path(__file__).resolve().parents[2] / "dynamis_model" / "tests" / "mace"
sys.path.insert(0, str(TEST_DIR))
from mace_standalone_reference import build_core_batch, load_raw_structure  # noqa: E402
from precision_settings import configure_precision, precision_snapshot  # noqa: E402
from test_mace_model import build_state  # noqa: E402
from dynamis_model.nnp.mace import (  # noqa: E402
    COMPUTE_DTYPE, MACEModel, load_raw_mace_model, prepare_graph,
    resolve_mace_model_name, resolve_model_head_name,
)
from dynamis_model.units import EV_PER_A_TO_AMU_A_PER_FS2 as EV_TO_MD  # noqa: E402
from dynamis_state import TorchDistEnv  # noqa: E402


def rows(batch: dict, global_ids: torch.Tensor) -> torch.Tensor:
    edge_ids = global_ids[batch["edge_index"]]
    return torch.cat((edge_ids.T.long(), batch["unit_shifts"].round().long()), dim=1).cpu()


def edge_vectors(batch: dict) -> torch.Tensor:
    context = prepare_graph(
        batch, compute_virials=False, compute_stress=False,
        compute_displacement=False, lammps_mliap=False,
    )
    return context.vectors.detach().float().cpu()


def canonical_order(edge_rows: torch.Tensor) -> list[int]:
    return sorted(range(len(edge_rows)), key=lambda index: (tuple(edge_rows[index].tolist()), index))


def compare(name: str, native: torch.Tensor, adapter: torch.Tensor) -> dict:
    if native.shape != adapter.shape:
        return {"name": name, "native_shape": list(native.shape),
                "adapter_shape": list(adapter.shape), "shape_match": False}
    difference = (native.float() - adapter.float()).abs()
    return {"name": name, "shape": list(native.shape), "shape_match": True,
            "max_abs": float(difference.max()) if difference.numel() else 0.0,
            "mean_abs": float(difference.mean()) if difference.numel() else 0.0}


def tensor_output(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item.detach().cpu().clone()
    return None


def capture_modules(model, target: dict) -> list:
    modules = [("node_embedding", model.node_embedding),
               ("spherical_harmonics", model.spherical_harmonics),
               ("radial_embedding", model.radial_embedding)]
    for index, (interaction, product) in enumerate(zip(model.interactions, model.products)):
        modules += [(f"interaction_{index}", interaction), (f"product_{index}", product)]
    modules += [(f"readout_{index}", module) for index, module in enumerate(model.readouts)]
    modules += [("scale_shift", model.scale_shift)]
    handles = []
    for name, module in modules:
        def hook(_module, _inputs, output, stage=name):
            captured = tensor_output(output)
            if captured is not None:
                target[stage] = captured
        handles.append(module.register_forward_hook(hook))
    return handles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-seed", type=int, default=1234)
    parser.add_argument("--molecule-count", type=int, default=64)
    parser.add_argument("--xyz-path", type=Path)
    parser.add_argument("--model-family", choices=("mp", "off"), default="mp")
    parser.add_argument("--model", default=None)
    parser.add_argument("--head-name", default="")
    parser.add_argument("--enable-cueq", action="store_true")
    parser.add_argument("--precision-mode", choices=("default", "no_tf32"), default="no_tf32")
    parser.add_argument("--stage-atol", type=float, default=1e-7)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_precision(args.precision_mode)
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The zero-halo trace requires exactly one torchrun rank.")
    environment = TorchDistEnv(is_distributed=True)
    try:
        device = environment.device
        raw = load_raw_structure(args)
        model_name = resolve_mace_model_name(args.model_family, args.model)
        model = load_raw_mace_model(
            args.model_family, model_name, str(device), enable_cueq=args.enable_cueq,
        ).to(device=device, dtype=COMPUTE_DTYPE).eval()
        head_name = resolve_model_head_name(model, args.head_name or None)
        native_batch = build_core_batch(raw, model, device, head_name)
        native_rows = rows(native_batch, torch.arange(len(raw["atomic_numbers"]), device=device))
        native_vectors = edge_vectors(native_batch)
        native_stages = {}
        handles = capture_modules(model, native_stages)
        native_output = model(
            native_batch, training=False, compute_force=True,
            compute_stress=False, compute_hessian=False,
        )
        for handle in handles:
            handle.remove()
        native_forces = native_output["forces"].detach().float().cpu()

        adapter = MACEModel(model, head_name=head_name)
        state = build_state(raw, adapter.cutoff, 0.5, environment)
        adapter.initialize_buffers(state.local_partition_info, device, 1)
        if adapter.exchange_enabled or adapter.partition_validation["total_recv"]:
            raise AssertionError("Expected zero halo atoms in the one-GPU trace")
        owned = state.local_partition_info.partition_indices.long()
        if owned.numel() != len(raw["atomic_numbers"]) or not torch.equal(
            owned.sort().values, torch.arange(owned.numel(), device=device)
        ):
            raise AssertionError("One-GPU partition did not own every atom exactly once")
        view = adapter._build_structural_view(state)
        graph = adapter._build_owned_graph(view, state)
        adapter_rows = rows(graph.batch_dict, owned)
        adapter_vectors = edge_vectors(graph.batch_dict)
        adapter_stages = {}
        handles = capture_modules(model, adapter_stages)
        adapter_energy_md, adapter_force_md = adapter(state)
        for handle in handles:
            handle.remove()
        aligned_adapter_forces = torch.empty_like(native_forces)
        aligned_adapter_forces[owned.cpu()] = adapter_force_md.detach().float().cpu() / EV_TO_MD

        native_counter = Counter(map(tuple, native_rows.tolist()))
        adapter_counter = Counter(map(tuple, adapter_rows.tolist()))
        same_edge_multiset = native_counter == adapter_counter
        same_order = torch.equal(native_rows, adapter_rows)
        comparisons = []
        if same_edge_multiset:
            native_order = canonical_order(native_rows)
            adapter_order = canonical_order(adapter_rows)
            comparisons.append(compare("edge_displacements", native_vectors[native_order],
                                       adapter_vectors[adapter_order]))
        else:
            comparisons.append({"name": "edge_displacements", "comparable": False})
        node_order = owned.cpu().argsort()
        for name in native_stages:
            if name not in adapter_stages:
                comparisons.append({"name": name, "present_in_adapter": False})
                continue
            lhs, rhs = native_stages[name], adapter_stages[name]
            if name in ("spherical_harmonics", "radial_embedding"):
                if not same_edge_multiset:
                    comparisons.append({"name": name, "comparable": False})
                    continue
                lhs = lhs[canonical_order(native_rows)]
                rhs = rhs[canonical_order(adapter_rows)]
            elif rhs.ndim and rhs.shape[0] == owned.numel():
                rhs = rhs[node_order]
            comparisons.append(compare(name, lhs, rhs))
        comparisons.append(compare("position_gradients_ev_per_angstrom", native_forces,
                                   aligned_adapter_forces))
        first_value_divergence = next((item["name"] for item in comparisons
                                       if item.get("shape_match") is False or
                                       item.get("max_abs", 0.0) > args.stage_atol), None)
        first_divergence = (
            "edge_membership" if not same_edge_multiset else
            "ordered_edges" if not same_order else first_value_divergence
        )
        result = {
            "model_family": args.model_family, "model": model_name,
            "backend": "cueq" if args.enable_cueq else "e3nn",
            "num_atoms": int(owned.numel()), "halo_atoms": adapter.partition_validation["total_recv"],
            "precision": precision_snapshot(args.precision_mode),
            "native_edges": len(native_rows), "adapter_edges": len(adapter_rows),
            "edge_multiset_equal": same_edge_multiset,
            "ordered_edges_equal": same_order,
            "first_order_difference": next((index for index, (a, b) in enumerate(zip(native_rows.tolist(), adapter_rows.tolist())) if a != b), None),
            "missing_edges": sum((native_counter - adapter_counter).values()),
            "extra_edges": sum((adapter_counter - native_counter).values()),
            "native_energy_ev": float(native_output["energy"].detach().reshape(-1).sum().cpu()),
            "adapter_energy_ev": float(adapter_energy_md.cpu()) / EV_TO_MD,
            "stage_atol": args.stage_atol,
            "first_divergence": first_divergence,
            "first_value_divergence": first_value_divergence,
            "comparisons": comparisons,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "trace.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        torch.save({"native_edge_rows": native_rows, "adapter_edge_rows": adapter_rows,
                    "native_displacements": native_vectors, "adapter_displacements": adapter_vectors,
                    "native_stages": native_stages, "adapter_stages": adapter_stages,
                    "owned_global_ids": owned.cpu(), "native_position_gradients": -native_forces,
                    "adapter_position_gradients": -aligned_adapter_forces},
                   args.output_dir / "trace.pt")
        print(json.dumps(result, indent=2))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
