"""Capture exact software, device, source, and cached checkpoint provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


def command(*args: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    destination = Path(sys.argv[1])
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "mace"
    cache_files = {
        str(path.relative_to(cache_root)): sha256(path)
        for path in sorted(cache_root.rglob("*")) if path.is_file()
    } if cache_root.is_dir() else {}
    xyz_name = os.environ.get("FORCE_XYZ_PATH", "")
    xyz_path = Path(xyz_name) if xyz_name else None
    result = {
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--short"),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "packages": {name: version(name) for name in (
            "mace-torch", "e3nn", "cuequivariance", "cuequivariance-torch",
            "cuequivariance-ops-cu12", "cuequivariance-ops-torch-cu12",
            "cuequivariance-ops-cu13", "cuequivariance-ops-torch-cu13",
            "numpy", "ase", "dynamis-model", "dynamis-state",
        )},
        "model_family": os.environ["FORCE_MODEL_FAMILY"],
        "model_name": os.environ["FORCE_MODEL"],
        "seed": os.environ["FORCE_SEED"],
        "xyz_path": xyz_name or None,
        "xyz_sha256": sha256(xyz_path) if xyz_path and xyz_path.is_file() else None,
        "mace_cache_root": str(cache_root),
        "mace_cache_sha256": cache_files,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
