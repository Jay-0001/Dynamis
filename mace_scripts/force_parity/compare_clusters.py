"""Compare matching Delta and Beta native references and adapter results."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

EV_TO_MD = 0.0096485332


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    delta, beta = map(Path, sys.argv[1:3])
    delta_summary, beta_summary = read_json(delta / "summary.json"), read_json(beta / "summary.json")
    delta_prov, beta_prov = read_json(delta / "provenance.json"), read_json(beta / "provenance.json")
    delta_cases = {(row["case"], row["backend"], row["ranks"], row["reduction"]): row
                   for row in delta_summary["cases"]}
    beta_cases = {(row["case"], row["backend"], row["ranks"], row["reduction"]): row
                  for row in beta_summary["cases"]}
    rows = []
    for key in sorted(delta_cases.keys() & beta_cases.keys()):
        d, b = delta_cases[key], beta_cases[key]
        reference_dir = Path(key[0]) / key[1]
        d_ref = torch.load(delta / reference_dir / "reference.pt", map_location="cpu", weights_only=False)
        b_ref = torch.load(beta / reference_dir / "reference.pt", map_location="cpu", weights_only=False)
        d_force = d_ref["reference"]["forces_ev_per_angstrom"].float()
        b_force = b_ref["reference"]["forces_ev_per_angstrom"].float()
        same_signature = d["reference_signature"] == b["reference_signature"]
        native_force_difference = (
            float(((d_force - b_force).abs() * EV_TO_MD).max())
            if same_signature and d_force.shape == b_force.shape else None
        )
        rows.append({
            "case": key[0], "backend": key[1], "ranks": key[2], "reduction": key[3],
            "same_reference_signature": same_signature,
            "native_energy_beta_minus_delta_ev": b["reference_energy_ev"] - d["reference_energy_ev"],
            "native_force_beta_delta_max_abs_md": native_force_difference,
            "adapter_force_max_abs_delta_md": d["force_max_abs_md"],
            "adapter_force_max_abs_beta_md": b["force_max_abs_md"],
            "adapter_force_violations_delta": d["force_violations"],
            "adapter_force_violations_beta": b["force_violations"],
        })
    if not rows:
        raise SystemExit("No matching case/backend/rank/reduction results between Delta and Beta")
    delta_hashes = delta_prov["mace_cache_sha256"]
    beta_hashes = beta_prov["mace_cache_sha256"]
    shared_cache_names = sorted(delta_hashes.keys() & beta_hashes.keys())
    result = {
        "delta": str(delta), "beta": str(beta),
        "same_model_name": delta_prov["model_name"] == beta_prov["model_name"],
        "shared_cache_names": shared_cache_names,
        "matching_shared_cache_hashes": bool(shared_cache_names) and all(
            delta_hashes[name] == beta_hashes[name] for name in shared_cache_names
        ),
        "delta_checkpoint_hashes": delta_hashes,
        "beta_checkpoint_hashes": beta_hashes,
        "cases": rows,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
