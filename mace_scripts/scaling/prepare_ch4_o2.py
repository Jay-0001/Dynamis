#!/usr/bin/env python3
"""Create one reproducible CH4/O2 XYZ for all MACE OFF strong-scaling jobs."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "experiments/exp_datasets/scripts/03_generate_ch4.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-k", type=int, default=45, help="Target atom count in thousands (default: 45k for 8 GPUs)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "mace_data/strong_scaling_off")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.size_k < 1:
        parser.error("--size-k must be positive")
    num_ch4 = round(args.size_k * 1000 / 9)
    atoms = num_ch4 * 9
    box = round(60 * (args.size_k / 8) ** (1 / 3))
    xyz = args.out_dir.resolve() / f"ch4_{args.size_k}k_splited.xyz"
    manifest = xyz.with_suffix(".json")
    if (xyz.exists() or manifest.exists()) and not args.overwrite:
        parser.error(f"{xyz} or its manifest already exists; pass --overwrite to regenerate")
    subprocess.run(
        [sys.executable, str(GENERATOR), "--single", "--num-c", str(num_ch4),
         "--box", str(box), "--variant", "splited", "--seed", "42",
         "--momenta-std", "0", "--out", str(xyz)],
        check=True,
    )
    with xyz.open(encoding="utf-8") as handle:
        actual_atoms = int(handle.readline())
        header = handle.readline()
    if actual_atoms != atoms or f'Lattice="{box:.1f} ' not in header:
        raise RuntimeError("Generated XYZ atom count or cubic cell differs from the requested size")
    details = {
        "xyz": str(xyz), "sha256": sha256(xyz), "atoms": atoms,
        "num_ch4": num_ch4, "num_o2": 2 * num_ch4, "box_angstrom": box,
        "variant": "splited", "seed": 42, "momenta_std": 0,
        "generator": str(GENERATOR.relative_to(ROOT)),
    }
    manifest.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(f"XYZ: {xyz}\nSHA256: {details['sha256']}\nAtoms: {atoms}\nManifest: {manifest}")


if __name__ == "__main__":
    main()
