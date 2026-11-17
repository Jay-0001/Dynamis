#!/usr/bin/env python3
"""Summarize synchronized host iteration timers for one MACE scaling job."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(directory: Path) -> dict:
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    world_size = metadata["world_size"]
    steps = metadata["steps"]
    warmup = metadata["warmup_steps"]
    if not (0 <= warmup < steps):
        raise ValueError("Invalid warmup or step count in run metadata")
    observed = {}
    for rank in range(world_size):
        path = directory / f"timing_rank{rank}.jsonl"
        records = {}
        header = None
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("type") == "meta":
                    header = row
                elif row.get("type") == "rec" and row.get("key") == "iteration" and row.get("where") == "host":
                    step = row["step"]
                    duration = row.get("time_ns")
                    if step in records or not isinstance(duration, int) or duration <= 0 or row.get("error"):
                        raise ValueError(f"Invalid or duplicate host iteration in {path}:{line_number}")
                    records[step] = duration
        if header is None or header.get("rank") != rank or header.get("world_size") != world_size:
            raise ValueError(f"Rank/world-size metadata mismatch in {path}")
        if set(records) != set(range(steps)):
            raise ValueError(f"Missing or extra iteration steps in {path}: found {len(records)} of {steps}")
        observed[rank] = records
    critical_ms = [max(observed[rank][step] for rank in observed) / 1e6 for step in range(warmup, steps)]
    median_ms = statistics.median(critical_ms)
    return {
        **metadata,
        "metric": "median over measured steps of max per-rank synchronized host iteration duration",
        "measured_steps": len(critical_ms),
        "median_critical_path_ms_per_step": median_ms,
        "mean_critical_path_ms_per_step": statistics.mean(critical_ms),
        "min_critical_path_ms_per_step": min(critical_ms),
        "max_critical_path_ms_per_step": max(critical_ms),
        "median_steps_per_second": 1000 / median_ms,
        "critical_path_ms_by_step": critical_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="one job output directory")
    args = parser.parse_args()
    result = summarize(args.directory)
    path = args.directory / "strong_scaling_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{result['world_size']} GPU(s), {result['atoms']} atoms, "
          f"{result['measured_steps']} measured steps: "
          f"{result['median_critical_path_ms_per_step']:.3f} ms/step "
          f"({result['median_steps_per_second']:.3f} steps/s)")
    print(f"Summary: {path}")


if __name__ == "__main__":
    main()
