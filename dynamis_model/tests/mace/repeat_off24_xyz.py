"""Tile the OFF24 C/H cutoff probe into a deterministic, sparse XYZ system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .mace_test_systems import OFF24_BRIDGE_PATH, load_xyz_structure
else:
    from mace_test_systems import OFF24_BRIDGE_PATH, load_xyz_structure


def repeat_off24_xyz(output: Path, copies: int) -> dict:
    if copies < 1:
        raise ValueError("--copies must be positive.")
    source = load_xyz_structure(OFF24_BRIDGE_PATH)
    base = np.asarray(source["positions"], dtype=np.float64)
    numbers = source["atomic_numbers"]
    if len(base) != 24 or any(source["pbc"]):
        raise ValueError("OFF24 source fixture must be the 24-atom nonperiodic probe.")

    distances = np.linalg.norm(base[:, None] - base[None, :], axis=-1)
    pairs = distances[np.triu_indices(len(base), k=1)]
    if not ((5.0 < pairs) & (pairs < 6.0)).any() or not (pairs > 6.0).any():
        raise ValueError("OFF24 probe must contain both selected and excluded pairs.")
    if np.min(np.abs(pairs - 6.0)) <= 1e-3:
        raise ValueError("OFF24 probe lies too close to its 6 Å cutoff.")

    width = int(np.ceil(copies ** (1.0 / 3.0)))
    while width**3 < copies:
        width += 1
    spacing = 26.0
    side = 36.0 + spacing * (width - 1)
    positions = []
    for index in range(copies):
        grid = np.array((index % width, (index // width) % width, index // (width * width)))
        positions.append(base + spacing * grid)
    positions = np.concatenate(positions)
    if np.min(positions) <= 6.5 or np.max(positions) >= side - 6.5:
        raise ValueError("Repeated atoms must remain clear of the cell boundary.")

    from ase.data import chemical_symbols

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{len(positions)}\n")
        handle.write(
            f'Lattice="{side:g} 0 0 0 {side:g} 0 0 0 {side:g}" '
            'Properties=species:S:1:pos:R:3 pbc="F F F"\n'
        )
        for index, position in enumerate(positions):
            handle.write(
                f"{chemical_symbols[numbers[index % len(base)]]} "
                + " ".join(f"{value:.8f}" for value in position)
                + "\n"
            )
    return {
        "output": str(output), "copies": copies, "atoms": len(positions),
        "pairs_between_5_and_6_angstrom_per_copy": int(((5.0 < pairs) & (pairs < 6.0)).sum()),
        "nearest_6_angstrom_gap": float(np.min(np.abs(pairs - 6.0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--copies", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(repeat_off24_xyz(args.output, args.copies), indent=2))


if __name__ == "__main__":
    main()
