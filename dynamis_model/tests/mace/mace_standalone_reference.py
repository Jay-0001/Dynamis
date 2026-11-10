"""Write an independent raw-MACE reference for distributed parity tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dynamis_model.nnp.mace import (
    COMPUTE_DTYPE,
    load_raw_mace_model,
    resolve_mace_model_name,
    resolve_model_head_name,
)

ARTIFACT_VERSION = 5
TEST_FILE_DIR = Path(__file__).resolve().parent


def natural_chain12() -> dict[str, Any]:
    return {
        "name": "natural_chain12",
        "atomic_numbers": [6] * 4 + [1] * 8,
        "positions": [
            [2.00, 2.00, 2.00], [3.52, 2.12, 2.00],
            [5.01, 1.90, 2.18], [6.50, 2.08, 1.95],
            [1.45, 2.92, 2.20], [1.52, 1.08, 1.80],
            [3.52, 3.08, 1.55], [3.55, 1.22, 2.62],
            [5.00, 2.82, 2.72], [5.02, 0.95, 1.58],
            [7.03, 2.96, 2.20], [7.02, 1.18, 1.72],
        ],
        "cell": [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]],
        "pbc": [False, False, False],
    }


def resolve_xyz_path(path: Path) -> Path:
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, TEST_FILE_DIR / path, TEST_FILE_DIR / "data" / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"XYZ structure does not exist: {path}")


def load_xyz(path: Path) -> dict[str, Any]:
    from ase.io import read

    path = resolve_xyz_path(path)
    atoms = read(path)
    positions = np.asarray(atoms.positions, dtype=np.float32)
    if not atoms.cell.any():
        lengths = np.maximum(positions.max(axis=0) + 10.0, 20.0)
        atoms.set_cell(np.diag(lengths))
    if not np.any(atoms.pbc):
        cell = np.asarray(atoms.cell, dtype=np.float64)
        fractional = positions.astype(np.float64) @ np.linalg.inv(cell)
        if np.any(positions < 0) or np.any(fractional < 0) or np.any(fractional >= 1):
            raise ValueError(
                "Nonperiodic MACE test atoms must lie within the positive simulation cell."
            )
    return {
        "name": path.stem,
        "source_path": str(path),
        "atomic_numbers": atoms.numbers.astype(int).tolist(),
        "positions": positions.tolist(),
        "cell": np.asarray(atoms.cell, dtype=np.float32).tolist(),
        "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
    }


def load_raw_structure(args: argparse.Namespace) -> dict[str, Any]:
    return load_xyz(args.xyz_path) if args.xyz_path else natural_chain12()


def reference_signature(raw: dict[str, Any], family: str, model: str, head: str, cueq: bool) -> str:
    payload = {key: raw[key] for key in ("atomic_numbers", "positions", "cell", "pbc")}
    payload.update(model_family=family, model_name=model, head_name=head, enable_cueq=cueq)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_core_batch(
    raw: dict[str, Any], model: torch.nn.Module, device: torch.device, head_name: str
) -> dict[str, torch.Tensor]:
    from mace.data import AtomicData, Configuration
    from mace.tools import AtomicNumberTable, torch_tools
    from mace.tools.torch_geometric.batch import Batch

    configuration = Configuration(
        atomic_numbers=np.asarray(raw["atomic_numbers"], dtype=np.int64),
        positions=np.asarray(raw["positions"], dtype=np.float32),
        properties={},
        property_weights={},
        cell=np.asarray(raw["cell"], dtype=np.float32),
        pbc=tuple(raw["pbc"]),
        weight=1.0,
        config_type="dynamis_raw_mace_reference",
        head=head_name,
    )
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers.cpu().tolist()])
    with torch_tools.default_dtype("float32"):
        atomic_data = AtomicData.from_config(
            config=configuration,
            z_table=z_table,
            cutoff=float(model.r_max.detach().cpu()),
            heads=list(getattr(model, "heads", [])) or [head_name],
        )
    batch = Batch.from_data_list([atomic_data]).to(device).to_dict()
    return {
        key: value.to(dtype=COMPUTE_DTYPE) if torch.is_floating_point(value) else value
        for key, value in batch.items()
    }


def evaluate_raw_core(
    raw: dict[str, Any], model: torch.nn.Module, device: torch.device, head_name: str
) -> dict[str, Any]:
    batch = build_core_batch(raw, model, device, head_name)
    output = model(
        batch, training=False, compute_force=True,
        compute_stress=False, compute_hessian=False,
    )
    node_energy = output["node_energy"].detach().reshape(-1).double()
    head_index = int(batch["head"].reshape(()).item())
    baseline = model.atomic_energies_fn(batch["node_attrs"])[
        :, head_index
    ].detach().reshape(-1).double()
    baseline_energy = baseline.sum()
    interaction_energy = (node_energy - baseline).sum()
    edge_rows = torch.cat(
        (batch["edge_index"].T.long(), batch["unit_shifts"].round().long()), dim=1
    ).int().cpu()
    return {
        "energy_ev": float((baseline_energy + interaction_energy).cpu()),
        "forces_ev_per_angstrom": output["forces"].detach().float().cpu(),
        "edge_rows": edge_rows,
        "model_parameter_dtype": str(next(model.parameters()).dtype),
        "position_dtype": str(batch["positions"].dtype),
        "energy_accumulator_dtype": str(node_energy.dtype),
    }


def benchmark_raw_core(raw, model, device, head_name, warmup, iterations, samples):
    if warmup < 0 or iterations <= 0 or samples <= 0:
        raise ValueError("Benchmark counts require nonnegative warmup and positive iterations/samples.")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        evaluate_raw_core(raw, model, device, head_name)
    timings = []
    for _ in range(samples):
        synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            evaluate_raw_core(raw, model, device, head_name)
        synchronize()
        timings.append(1000.0 * (time.perf_counter() - start) / iterations)
    return {"median_ms_per_step": statistics.median(timings), "samples_ms": timings}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builtin-system", choices=["natural_chain12"], default="natural_chain12")
    parser.add_argument("--xyz-path", type=Path)
    parser.add_argument("--model-family", choices=["mp", "off"], default="mp")
    parser.add_argument("--model", default=None)
    parser.add_argument("--head-name", default="")
    parser.add_argument("--enable-cueq", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    parser.add_argument("--benchmark-iterations", type=int, default=10)
    parser.add_argument("--benchmark-samples", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw = load_raw_structure(args)
    model_name = resolve_mace_model_name(args.model_family, args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_raw_mace_model(
        args.model_family, model_name, str(device), enable_cueq=args.enable_cueq
    ).to(device=device, dtype=COMPUTE_DTYPE).eval()
    head_name = resolve_model_head_name(model, args.head_name or None)
    reference = evaluate_raw_core(raw, model, device, head_name)
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "system_name": raw["name"],
        "num_atoms": len(raw["atomic_numbers"]),
        "model_family": args.model_family,
        "model_name": model_name,
        "head_name": head_name,
        "enable_cueq": args.enable_cueq,
        "cutoff_angstrom": float(model.r_max.detach().cpu()),
        "signature": reference_signature(raw, args.model_family, model_name, head_name, args.enable_cueq),
    }
    benchmark = (
        benchmark_raw_core(raw, model, device, head_name, args.benchmark_warmup,
                           args.benchmark_iterations, args.benchmark_samples)
        if args.benchmark else None
    )
    artifact = {"metadata": metadata, "reference": reference, "benchmark": benchmark}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    summary = {
        "metadata": metadata,
        "reference": {"controlled_energy_ev": reference["energy_ev"],
                      "directed_edges": int(reference["edge_rows"].shape[0]),
                      "model_parameter_dtype": reference["model_parameter_dtype"],
                      "position_dtype": reference["position_dtype"],
                      "energy_accumulator_dtype": reference["energy_accumulator_dtype"]},
        "benchmark": benchmark,
    }
    print(json.dumps(summary, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
