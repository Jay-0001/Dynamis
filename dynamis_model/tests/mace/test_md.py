"""
Run a world-size=1 MACE reference simulation and compare it against a MACEModel
trajectory written by run_distributed_trajectory.py. The reference runs a raw
MACE checkpoint on its native graph at each step.

Usage:
    python dynamis_model/tests/mace/test_md.py --path <temp dir path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from dynamis_integrator import Verlet
from dynamis_model.nnp.mace import resolve_mace_model_name

if __package__:
    from .md_test_common import (
        SEED,
        REBALANCE_INTERVAL,
        GeneratedMaceSimulator,
        build_native_mace_reference,
        compare_trajectories,
        generate_test_structure,
        get_test_system_spec,
        install_trajectory_simulator_import_compatibility,
        make_config,
        molecular_info_from_structure,
        real_singleton_torch_dist_env,
        run_md_loop,
        teardown_singleton_dist_env,
    )
    from .mace_test_systems import structure_signature
else:
    from md_test_common import (
        SEED,
        REBALANCE_INTERVAL,
        GeneratedMaceSimulator,
        build_native_mace_reference,
        compare_trajectories,
        generate_test_structure,
        get_test_system_spec,
        install_trajectory_simulator_import_compatibility,
        make_config,
        molecular_info_from_structure,
        real_singleton_torch_dist_env,
        run_md_loop,
        teardown_singleton_dist_env,
    )
    from mace_test_systems import structure_signature


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--system-seed", type=int, default=SEED)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--system", default="silicon", choices=["silicon", "helium", "organic"]
    )
    parser.add_argument("--head-name", default="")
    parser.add_argument(
        "--model-family", default="mp", choices=["mp", "off"]
    )
    parser.add_argument(
        "--enable-cueq",
        action="store_true",
        help="Use cuEquivariance in both the singleton reference and distributed artifact.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.model = resolve_mace_model_name(args.model_family, args.model)
    spec = get_test_system_spec(args.system)
    raw = generate_test_structure(args.system, args.system_seed)
    install_trajectory_simulator_import_compatibility()
    output_dir = Path(args.path) / "ref_out"
    dist_dump = Path(args.path) / "dist_traj.pt"

    try:
        torch_dist_env = real_singleton_torch_dist_env()
        dist_env = torch_dist_env.dist_env
        assert dist_env.world_size == 1

        torch.manual_seed(SEED)
        model = build_native_mace_reference(
            device=torch_dist_env.device,
            model=args.model,
            head_name=args.head_name or None,
            model_family=args.model_family,
            system=args.system,
            enable_cueq=args.enable_cueq,
        )
        print(
            f"Running raw-core singleton MACE reference trajectory for {args.system}."
        )
        integrator = Verlet(dt=spec["timestep_fs"])
        config = make_config(
            str(output_dir),
            model.cutoff,
            rebalance_interval=REBALANCE_INTERVAL,
            timestep_fs=spec["timestep_fs"],
            side_length_angstrom=spec["side_length_angstrom"],
            total_steps=spec["step_count"],
        )

        sim = GeneratedMaceSimulator(
            model, integrator, dist_env, config, torch_dist_env.device,
            periodic=spec["pbc"],
        )
        sim.setup(molecular_info_from_structure(raw))

        ref_traj = run_md_loop(sim, config, steps=spec["step_count"])
        assert (
            dist_dump.exists()
        ), f"Distributed trajectory artifact not found: {dist_dump}"
        artifact = torch.load(dist_dump, weights_only=False)
        if not isinstance(artifact, dict) or set(artifact) != {"metadata", "frames"}:
            raise ValueError(
                "Distributed trajectory artifact must contain exactly 'metadata' and 'frames'."
            )
        expected_metadata = {
            "artifact_version": 5,
            "model": args.model,
            "model_family": args.model_family,
            "system": args.system,
            "system_seed": args.system_seed,
            "head_name": args.head_name,
            "enable_cueq": args.enable_cueq,
            "structure_signature": structure_signature(raw),
            "timestep_fs": config.timestep_fs,
            "step_count": spec["step_count"],
            "rebalance_interval": REBALANCE_INTERVAL,
            "world_size": 2,
            "adapter_identity": "MACEModel",
            "force_unit": "amu*angstrom/fs^2",
        }
        recorded_metadata = dict(artifact["metadata"])
        owned_atoms = recorded_metadata.pop("owned_atoms_by_rank", None)
        if (
            not isinstance(owned_atoms, list)
            or len(owned_atoms) != 2
            or any(not isinstance(count, int) or count < 0 for count in owned_atoms)
            or sum(owned_atoms) != spec["n_atoms"]
        ):
            raise ValueError(f"Invalid distributed ownership counts: {owned_atoms}")
        if recorded_metadata != expected_metadata:
            raise ValueError(
                "Distributed trajectory metadata does not match the singleton reference: "
                f"expected {expected_metadata}, got {recorded_metadata}"
            )
        compare_trajectories(
            ref_traj,
            artifact["frames"],
            report_path=str(output_dir / "comparison_report.csv"),
            side_length_angstrom=spec["side_length_angstrom"],
            expected_steps=spec["step_count"],
            periodic=spec["pbc"],
        )
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
    finally:
        teardown_singleton_dist_env()

    print(f"PASSED: {len(ref_traj)}-step trajectory matches within tolerance schedule")
    sys.exit(0)


if __name__ == "__main__":
    main()
