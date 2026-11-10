"""Deterministic, in-memory structures shared by MACE parity and MD tests."""

from __future__ import annotations

import hashlib
import json

import numpy as np

DEFAULT_SEED = 1234
ORGANIC_ATOMIC_NUMBERS = [6] * 4 + [1] * 8
_CHAIN_TEMPLATE = np.array(
    [
        [2.00, 2.00, 2.00], [3.52, 2.12, 2.00],
        [5.01, 1.90, 2.18], [6.50, 2.08, 1.95],
        [1.45, 2.92, 2.20], [1.52, 1.08, 1.80],
        [3.52, 3.08, 1.55], [3.55, 1.22, 2.62],
        [5.00, 2.82, 2.72], [5.02, 0.95, 1.58],
        [7.03, 2.96, 2.20], [7.02, 1.18, 1.72],
    ],
    dtype=np.float64,
)


def organic_chain12(seed: int = DEFAULT_SEED) -> dict:
    """Perturb a C4H8 geometry while keeping it clear of cell and cutoff bands."""
    rng = np.random.default_rng(seed)
    for _ in range(100):
        positions = _CHAIN_TEMPLATE + rng.normal(0.0, 0.005, _CHAIN_TEMPLATE.shape)
        positions += 10.0 - (positions.min(axis=0) + positions.max(axis=0)) / 2.0
        positions = positions.astype(np.float32)
        differences = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(differences, axis=-1)
        pairs = distances[np.triu_indices(len(positions), k=1)]
        if (
            np.all(positions > 6.5)
            and np.all(positions < 13.5)
            and np.all(pairs > 0.75)
            and all(np.all(np.abs(pairs - cutoff) > 1e-3) for cutoff in (4.5, 5.0, 6.0))
        ):
            return {
                "name": "organic_chain12",
                "atomic_numbers": ORGANIC_ATOMIC_NUMBERS.copy(),
                "positions": positions.tolist(),
                "cell": np.diag([20.0, 20.0, 20.0]).tolist(),
                "pbc": [False, False, False],
            }
    raise RuntimeError("Could not generate a C/H chain clear of the cutoff bands.")


def organic_molecules(seed: int = DEFAULT_SEED, count: int = 1) -> dict:
    """Place deterministic C/H fragments inside a nonperiodic positive cell.

    Thirteen-Angstrom center spacing keeps distinct fragments beyond the 6 A
    MPA cutoff. The cell also leaves more than 6 A around its outermost atoms.
    """
    if count < 1:
        raise ValueError("Molecule count must be positive.")
    if count == 1:
        return organic_chain12(seed)
    width = int(np.ceil(count ** (1.0 / 3.0)))
    while width**3 < count:
        width += 1
    positions = []
    for index in range(count):
        molecule = np.asarray(organic_chain12(seed + index)["positions"], dtype=np.float32)
        grid = np.array((index % width, (index // width) % width, index // (width * width)))
        center = 10.0 + 13.0 * grid
        positions.append(molecule - 10.0 + center)
    side = 13.0 * (width - 1) + 20.0
    all_positions = np.concatenate(positions).astype(np.float32)
    if np.any(all_positions <= 6.5) or np.any(all_positions >= side - 6.5):
        raise AssertionError("Generated molecules do not fit within the safe cell interior.")
    return {
        "name": f"organic_chain12_x{count}",
        "atomic_numbers": ORGANIC_ATOMIC_NUMBERS * count,
        "positions": all_positions.tolist(),
        "cell": np.diag([side] * 3).tolist(),
        "pbc": [False, False, False],
    }


def structure_signature(raw: dict) -> str:
    payload = {key: raw[key] for key in ("atomic_numbers", "positions", "cell", "pbc")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
