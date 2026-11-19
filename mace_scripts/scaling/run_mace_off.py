#!/usr/bin/env python3
"""Run one synchronized MACE OFF strong-scaling measurement under torchrun."""

import argparse
import hashlib
import json
import os
import re
import socket
from pathlib import Path

import torch
import torch.distributed as dist

from dynamis import Simulator, SimulatorConfig
from dynamis_integrator import Verlet
from dynamis_model.nnp.mace import MACEModel, load_raw_mace_model
from dynamis_state import TIMING_ENABLED, TimingRegistry, TorchDistEnv


class SynchronizedSimulator(Simulator):
    def step(self, step_size=None):
        result = super().step(step_size)
        # Simulator.run() stops its host iteration timer after this returns.
        torch.cuda.synchronize(self.device)
        return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_xyz(path: Path) -> tuple[int, float]:
    with path.open(encoding="utf-8") as handle:
        atoms = int(handle.readline().strip())
        header = handle.readline()
    match = re.search(r'Lattice="([^"]+)"', header)
    if not match or 'pbc="T T T"' not in header:
        raise ValueError("XYZ needs a cubic Lattice header and pbc=\"T T T\"")
    cell = [float(value) for value in match.group(1).split()]
    if len(cell) != 9 or any(abs(cell[i]) > 1e-6 for i in (1, 2, 3, 5, 6, 7)):
        raise ValueError("Only an axis-aligned cubic XYZ cell is supported")
    side = cell[0]
    if side <= 0 or abs(cell[4] - side) > 1e-6 or abs(cell[8] - side) > 1e-6:
        raise ValueError("Only a positive cubic XYZ cell is supported")
    return atoms, side


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--timestep-fs", type=float, default=0.1)
    parser.add_argument("--rebalance-interval", type=int, default=0)
    parser.add_argument("--model", default="medium", help="MACE OFF checkpoint name")
    parser.add_argument("--enable-cueq", action="store_true", help="Convert the MACE OFF checkpoint to CuEq")
    args = parser.parse_args()
    if args.steps <= args.warmup_steps or args.warmup_steps < 0:
        parser.error("require 0 <= warmup-steps < steps")
    if args.timestep_fs <= 0 or args.rebalance_interval < 0:
        parser.error("timestep must be positive and rebalance interval nonnegative")
    if args.expected_gpus < 1:
        parser.error("expected GPU count must be positive")
    xyz = args.xyz.resolve(strict=True)
    output = args.output_dir.resolve()
    if not TIMING_ENABLED:
        raise RuntimeError("dynamis_state was built without DYNAMIS_TIMING; rebuild with timing enabled")
    if torch.cuda.device_count() != args.expected_gpus:
        raise RuntimeError(f"Expected {args.expected_gpus} visible GPUs; got {torch.cuda.device_count()}")
    if int(os.environ.get("WORLD_SIZE", "0")) != args.expected_gpus:
        raise RuntimeError("torchrun WORLD_SIZE must match --expected-gpus")
    atoms, side = read_xyz(xyz)
    xyz_hash = sha256(xyz)
    manifest = xyz.with_suffix(".json")
    if not manifest.exists():
        raise FileNotFoundError(f"Missing preparation manifest: {manifest}")
    expected = json.loads(manifest.read_text(encoding="utf-8"))
    if expected.get("sha256") != xyz_hash or expected.get("atoms") != atoms:
        raise ValueError("XYZ does not match its preparation manifest")
    env = TorchDistEnv(is_distributed=True)
    try:
        if env.global_rank == 0:
            output.mkdir(parents=True, exist_ok=False)
            (output / "run_metadata.json").write_text(json.dumps({
                "model_family": "off", "model": args.model, "xyz": str(xyz),
                "backend": "cueq" if args.enable_cueq else "e3nn",
                "xyz_sha256": xyz_hash, "atoms": atoms, "box_angstrom": side,
                "world_size": env.world_size, "steps": args.steps,
                "allocated_gpus": int(os.environ.get("SCALING_ALLOCATED_GPUS", env.world_size)),
                "warmup_steps": args.warmup_steps, "timestep_fs": args.timestep_fs,
                "rebalance_interval": args.rebalance_interval,
                "gpu_name": torch.cuda.get_device_name(env.device),
                "host": socket.gethostname(), "torch_version": torch.__version__,
                "cuda_runtime_version": torch.version.cuda,
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }, indent=2) + "\n", encoding="utf-8")
        dist.barrier()
        raw = load_raw_mace_model(
            "off", args.model, device=str(env.device), enable_cueq=args.enable_cueq
        )
        model = MACEModel(raw)
        config = SimulatorConfig(
            timestep_fs=args.timestep_fs, total_steps=args.steps,
            side_length_angstrom=side, cutoff_angstrom=model.cutoff,
            rebalance_interval=args.rebalance_interval,
            output_dir=str(output), traj_interval=0, log_interval=0,
        )
        sim = SynchronizedSimulator(model, Verlet(dt=args.timestep_fs), env.dist_env, config, env.device)
        try:
            sim.setup(str(xyz))
            dist.barrier()
            sim.run()
            dist.barrier()
        finally:
            TimingRegistry.instance().close()
        if env.global_rank == 0:
            print(f"Completed {args.steps} steps with {args.expected_gpus} GPU(s): {output}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
