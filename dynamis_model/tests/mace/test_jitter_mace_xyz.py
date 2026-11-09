"""Regression tests for the OFF cutoff-boundary fixture generator."""

from __future__ import annotations

import importlib.util
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _load_generator():
    path = Path(__file__).with_name("jitter_mace_xyz.py")
    spec = importlib.util.spec_from_file_location("jitter_mace_xyz", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.jitter_xyz


def test_large_lattice_clears_boundary_deterministically(tmp_path: Path) -> None:
    # Many exact 5 A pairs make whole-lattice redraws unlikely to ever succeed.
    positions = np.array(list(product(range(10), repeat=3)), dtype=float) * 2.5
    source = tmp_path / "lattice.xyz"
    with source.open("w", encoding="utf-8") as handle:
        handle.write(f'{len(positions)}\npbc="F F F"\n')
        for xyz in positions:
            handle.write(f"C {xyz[0]} {xyz[1]} {xyz[2]}\n")
    jitter_xyz = _load_generator()
    first = tmp_path / "first.xyz"
    second = tmp_path / "second.xyz"
    result = jitter_xyz(source, first, seed=2026)
    jitter_xyz(source, second, seed=2026)
    assert first.read_bytes() == second.read_bytes()
    assert result["initial_boundary_pairs"] > 0
    assert result["repair_rounds"] > 0
    written = np.array(
        [[float(value) for value in line.split()[1:4]]
         for line in first.read_text().splitlines()[2:]], dtype=np.float32,
    )
    pairs = cKDTree(written).query_pairs(5.001, output_type="ndarray")
    distances = np.linalg.norm(
        written[pairs[:, 0]].astype(float) - written[pairs[:, 1]].astype(float), axis=1
    )
    assert np.min(np.abs(distances - 5.0)) > 1e-3
    assert np.max(np.abs(written.astype(float) - positions)) < 0.01001
