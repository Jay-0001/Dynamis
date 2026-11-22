"""Summarize a force-parity matrix and compare old/current force tensors."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


def main() -> None:
    root = Path(sys.argv[1])
    cases = []
    comparisons = []
    for reference_path in sorted(root.glob("*/**/reference.json")):
        backend_dir = reference_path.parent
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        case_name = backend_dir.parent.name
        backend = backend_dir.name
        for run_path in sorted(backend_dir.glob("*rank_*/distributed.json")):
            run = json.loads(run_path.read_text(encoding="utf-8"))
            cases.append({
                "case": case_name,
                "backend": backend,
                "ranks": len(run["ranks"]),
                "reduction": run["model"]["adapter_reduction"],
                "reference_energy_ev": reference["reference"]["controlled_energy_ev"],
                "reference_signature": reference["metadata"]["signature"],
                "energy_error_md": run["energy"]["error_md"],
                "reported_local_sum_md": run["energy"]["reported_local_sum_md"],
                "force_mae_md": run["forces"]["mae_md"],
                "force_max_abs_md": run["forces"]["max_abs_md"],
                "force_violations": run["forces"]["violating_components"],
                "force_close": run["forces"]["close"],
                "graph": run["graph"],
                "result": str(run_path.relative_to(root)),
            })
        for legacy_path in sorted(backend_dir.glob("*rank_legacy/forces.pt")):
            current_path = Path(str(legacy_path).replace("rank_legacy", "rank_current"))
            if not current_path.is_file():
                continue
            legacy = torch.load(legacy_path, map_location="cpu", weights_only=True)
            current = torch.load(current_path, map_location="cpu", weights_only=True)
            difference = (legacy["distributed_md"] - current["distributed_md"]).abs()
            comparisons.append({
                "case": case_name,
                "backend": backend,
                "ranks": int(legacy_path.parent.name.split("rank_")[0]),
                "legacy_current_force_mae_md": float(difference.mean()),
                "legacy_current_force_max_abs_md": float(difference.max()),
            })
    if not cases:
        raise SystemExit(f"No distributed parity results found under {root}")
    result = {"cases": cases, "legacy_current_comparisons": comparisons}
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
