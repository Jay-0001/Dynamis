"""Create a reproducible nonperiodic XYZ with no pair near the OFF cutoff."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def jitter_xyz(source: Path, output: Path, seed: int = 2026,
               amplitude: float = 0.01, cutoff: float = 5.0,
               boundary_band: float = 1e-4) -> dict:
    lines = source.read_text(encoding="utf-8").splitlines()
    count = int(lines[0].strip())
    if len(lines) != count + 2 or amplitude <= 0:
        raise ValueError("Expected one XYZ frame and positive jitter amplitude.")
    if 'pbc="T' in lines[1] or 'pbc="t' in lines[1]:
        raise ValueError("This jitter fixture generator expects a nonperiodic XYZ.")
    tokens = [line.split() for line in lines[2:]]
    if any(len(row) < 4 for row in tokens):
        raise ValueError("Every atom row needs a symbol and three coordinates.")
    positions = np.asarray([[float(value) for value in row[1:4]] for row in tokens])
    rng = np.random.default_rng(seed)
    for attempt in range(1, 21):
        candidate = (positions + rng.uniform(-amplitude, amplitude, positions.shape)).astype(np.float32)
        pairs = cKDTree(candidate).query_pairs(cutoff + boundary_band, output_type="ndarray")
        if len(pairs):
            distances = np.linalg.norm(
                candidate[pairs[:, 0]].astype(np.float64)
                - candidate[pairs[:, 1]].astype(np.float64), axis=1,
            )
            minimum_gap = float(np.min(np.abs(distances - cutoff)))
        else:
            minimum_gap = float("inf")
        if minimum_gap > boundary_band:
            break
    else:
        raise ValueError("Could not clear the cutoff band in 20 deterministic attempts.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{count}\n{lines[1]} jitter_seed={seed} jitter_amplitude={amplitude}\n")
        for row, xyz in zip(tokens, candidate):
            handle.write(f"{row[0]} {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}\n")
    return {"atoms": count, "seed": seed, "attempt": attempt,
            "minimum_gap_from_cutoff_angstrom": minimum_gap, "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--amplitude", type=float, default=0.01)
    args = parser.parse_args()
    print(jitter_xyz(args.input, args.output, args.seed, args.amplitude))
