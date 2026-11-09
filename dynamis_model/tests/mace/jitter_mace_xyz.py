"""Create a reproducible nonperiodic XYZ with no pair near the OFF cutoff."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def jitter_xyz(source: Path, output: Path, seed: int = 2026,
               amplitude: float = 0.01, cutoff: float = 5.0,
               boundary_band: float = 1e-3) -> dict:
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
    candidate = (positions + rng.uniform(-amplitude, amplitude, positions.shape)).astype(np.float32)
    initial_boundary_pairs = None
    resampled_atoms = 0
    for attempt in range(101):
        pairs = cKDTree(candidate).query_pairs(cutoff + boundary_band, output_type="ndarray")
        if len(pairs):
            distances = np.linalg.norm(
                candidate[pairs[:, 0]].astype(np.float64)
                - candidate[pairs[:, 1]].astype(np.float64), axis=1,
            )
            minimum_gap = float(np.min(np.abs(distances - cutoff)))
            boundary_pairs = pairs[np.abs(distances - cutoff) <= boundary_band]
        else:
            minimum_gap = float("inf")
            boundary_pairs = np.empty((0, 2), dtype=np.int64)
        if initial_boundary_pairs is None:
            initial_boundary_pairs = len(boundary_pairs)
        if len(boundary_pairs) == 0:
            break
        if attempt == 100:
            raise ValueError(
                f"Could not clear {len(boundary_pairs)} cutoff-boundary pairs "
                f"after {attempt} targeted repair rounds (minimum gap {minimum_gap:.3g} A)."
            )
        # Resampling one endpoint per offending pair preserves the jitter of
        # the rest of the lattice. Redrawing all atoms almost surely creates
        # another boundary pair when the structure contains thousands of atoms.
        affected = np.unique(boundary_pairs[:, 1])
        candidate[affected] = (
            positions[affected]
            + rng.uniform(-amplitude, amplitude, (len(affected), 3))
        ).astype(np.float32)
        resampled_atoms += len(affected)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{count}\n{lines[1]} jitter_seed={seed} jitter_amplitude={amplitude}\n")
        for row, xyz in zip(tokens, candidate):
            handle.write(f"{row[0]} {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}\n")
    return {"atoms": count, "seed": seed, "repair_rounds": attempt,
            "initial_boundary_pairs": initial_boundary_pairs,
            "resampled_atom_events": resampled_atoms,
            "minimum_gap_from_cutoff_angstrom": minimum_gap, "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--amplitude", type=float, default=0.01)
    args = parser.parse_args()
    print(jitter_xyz(args.input, args.output, args.seed, args.amplitude))
