"""Summarize independent native repeats or matched adapter runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def main() -> None:
    root = Path(sys.argv[1])
    mode = sys.argv[2]
    summary = {"mode": mode, "cases": {}}
    for precision_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        precision = precision_dir.name
        summary["cases"][precision] = {}
        native = {}
        for backend_dir in sorted(path for path in precision_dir.iterdir() if path.is_dir()):
            backend = backend_dir.name
            if mode == "native":
                trials = sorted(backend_dir.glob("trial*.pt"))
                if not trials:
                    continue
                artifacts = [torch.load(path, map_location="cpu", weights_only=False) for path in trials]
                signature = artifacts[0]["metadata"]["signature"]
                if any(item["metadata"]["signature"] != signature for item in artifacts):
                    raise ValueError(f"Fixture changed across {precision}/{backend} trials")
                base = artifacts[0]["reference"]
                base_forces = base["forces_ev_per_angstrom"]
                native[backend] = base
                summary["cases"][precision][backend] = {
                    "precision_by_trial": [item["metadata"]["precision"] for item in artifacts],
                    "precision_settings_consistent": all(
                        item["metadata"]["precision"] == artifacts[0]["metadata"]["precision"]
                        for item in artifacts
                    ),
                    "trials": len(artifacts),
                    "energies_ev": [item["reference"]["energy_ev"] for item in artifacts],
                    "max_energy_drift_ev": max(abs(item["reference"]["energy_ev"] - base["energy_ev"]) for item in artifacts),
                    "max_force_drift_ev_per_angstrom": max(float((item["reference"]["forces_ev_per_angstrom"] - base_forces).abs().max()) for item in artifacts),
                }
            else:
                native_json = json.loads((backend_dir / "reference.json").read_text())
                adapter = json.loads((backend_dir / "adapter.json").read_text())
                summary["cases"][precision][backend] = {
                    "native_precision": native_json["metadata"]["precision"],
                    "adapter_precision": adapter["precision"],
                    "energy_error_md": adapter["energy"]["error_md"],
                    "energy_close": adapter["energy"]["close"],
                    "force_mae_md": adapter["forces"]["mae_md"],
                    "force_max_abs_md": adapter["forces"]["max_abs_md"],
                    "force_violating_components": adapter["forces"]["violating_components"],
                    "force_close": adapter["forces"]["close"],
                    "graph": adapter["graph"],
                }
        if mode == "native" and "e3nn" in native and "cueq" in native:
            a, b = native["e3nn"], native["cueq"]
            summary["cases"][precision]["backend_gap"] = {
                "energy_abs_ev": abs(a["energy_ev"] - b["energy_ev"]),
                "force_mae_ev_per_angstrom": float((a["forces_ev_per_angstrom"] - b["forces_ev_per_angstrom"]).abs().mean()),
                "force_max_abs_ev_per_angstrom": float((a["forces_ev_per_angstrom"] - b["forces_ev_per_angstrom"]).abs().max()),
                "ordered_edges_equal": bool(torch.equal(a["edge_rows"], b["edge_rows"])),
            }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
