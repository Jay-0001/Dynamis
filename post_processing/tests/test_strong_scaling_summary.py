"""Check that scaling results use the slowest rank after warmup."""

import json
import tempfile
import unittest
from pathlib import Path

from post_processing.strong_scaling_summary import summarize


class StrongScalingSummaryTest(unittest.TestCase):
    def test_critical_path_and_warmup(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "run_metadata.json").write_text(json.dumps({
                "world_size": 2, "steps": 3, "warmup_steps": 1, "atoms": 9,
            }), encoding="utf-8")
            for rank, times in enumerate(([100, 500, 100], [200, 100, 300])):
                rows = [{"type": "meta", "rank": rank, "world_size": 2}]
                rows.extend({
                    "type": "rec", "key": "iteration", "where": "host",
                    "step": step, "time_ns": ms * 1_000_000,
                } for step, ms in enumerate(times))
                (directory / f"timing_rank{rank}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
                )
            result = summarize(directory)
            self.assertEqual(result["critical_path_ms_by_step"], [500, 300])
            self.assertEqual(result["median_critical_path_ms_per_step"], 400)
            self.assertEqual(result["median_steps_per_second"], 2.5)
            (directory / "timing_rank1.jsonl").unlink()
            with self.assertRaises(FileNotFoundError):
                summarize(directory)


if __name__ == "__main__":
    unittest.main()
