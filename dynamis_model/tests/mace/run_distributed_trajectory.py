"""
Run the distributed MACEModel simulation and write a merged trajectory.

Usage:
    torchrun --nproc_per_node=2 dynamis_model/tests/mace/run_distributed_trajectory.py --path <temp dir path>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from dynamis_integrator import Verlet
from dynamis_state import TorchDistEnv
from dynamis_model.nnp.mace import resolve_mace_model_name

if __package__:
    from .md_test_common import (
        SEED,
        REBALANCE_INTERVAL,
        GeneratedMaceSimulator,
        build_mace_model,
        generate_test_structure,
        get_test_system_spec,
        install_trajectory_simulator_import_compatibility,
        make_config,
        molecular_info_from_structure,
        run_md_loop,
    )
    from .mace_test_systems import structure_signature
else:
    from md_test_common import (
        SEED,
        REBALANCE_INTERVAL,
        GeneratedMaceSimulator,
        build_mace_model,
        generate_test_structure,
        get_test_system_spec,
        install_trajectory_simulator_import_compatibility,
        make_config,
        molecular_info_from_structure,
        run_md_loop,
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
        help="Convert the distributed raw MACE model to cuEquivariance.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.model = resolve_mace_model_name(args.model_family, args.model)
    spec = get_test_system_spec(args.system)
    raw = generate_test_structure(args.system, args.system_seed)
    install_trajectory_simulator_import_compatibility()
    output_dir = Path(args.path) / "dist_out"
    dist_dump = Path(args.path) / "dist_traj.pt"

    try:
        torch_dist_env = TorchDistEnv(is_distributed=True)
        dist_env = torch_dist_env.dist_env

        torch.manual_seed(SEED)
        model = build_mace_model(
            device=torch_dist_env.device,
            model=args.model,
            head_name=args.head_name or None,
            model_family=args.model_family,
            system=args.system,
            enable_cueq=args.enable_cueq,
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

        local_frames = run_md_loop(sim, config, steps=spec["step_count"])
        local_payload = [
            (
                frame["energy"],
                frame["vid"],
                frame["pos"],
                frame["vel"],
            )
            for frame in local_frames
        ]

        dist.barrier()
        gathered = [None] * dist_env.world_size
        dist.all_gather_object(gathered, local_payload)

        if dist_env.global_rank == 0:
            merged = []
            for step in range(len(local_payload)):
                rank_local_energies = [
                    float(gathered[rank][step][0])
                    for rank in range(dist_env.world_size)
                ]
                vids = np.concatenate(
                    [gathered[rank][step][1] for rank in range(dist_env.world_size)]
                )
                pos = np.concatenate(
                    [gathered[rank][step][2] for rank in range(dist_env.world_size)]
                )
                vel = np.concatenate(
                    [gathered[rank][step][3] for rank in range(dist_env.world_size)]
                )
                order = np.argsort(vids)
                frame = {
                    "energy": float(sum(rank_local_energies)),
                    "pos": pos[order],
                    "vel": vel[order],
                    "vid": vids[order],
                }
                merged.append(frame)
            artifact = {
                "metadata": {
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
                    "world_size": dist_env.world_size,
                    "owned_atoms_by_rank": [len(gathered[rank][0][1]) for rank in range(dist_env.world_size)],
                    "adapter_identity": "MACEModel",
                    "force_unit": "amu*angstrom/fs^2",
                },
                "frames": merged,
            }
            torch.save(artifact, dist_dump)
            print(f"Owned atoms by rank: {artifact['metadata']['owned_atoms_by_rank']}")
            print(
                f"Wrote merged {len(merged)}-step distributed trajectory to {dist_dump}"
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
