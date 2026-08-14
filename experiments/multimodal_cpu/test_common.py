"""Unit tests for dependency-free measurement helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from experiments.multimodal_cpu.common import (
    flatten_record,
    grid_metrics,
    request_output_metrics,
)


class CommonHelpersTest(unittest.TestCase):
    def test_grid_metrics_uses_spatial_merge(self) -> None:
        self.assertEqual(
            grid_metrics([[1, 8, 12]], merge_size=2),
            {
                "grid_thw": [[1, 8, 12]],
                "patch_count": 96,
                "spatial_merge_size": 2,
                "visual_token_count": 24,
            },
        )

    def test_request_output_metrics_calculates_tpot(self) -> None:
        output = SimpleNamespace(
            request_id="request-1",
            prompt_token_ids=[1, 2],
            outputs=[SimpleNamespace(token_ids=[3, 4, 5])],
            finished=True,
            metrics=SimpleNamespace(
                first_token_latency=0.1,
                queued_ts=1.0,
                scheduled_ts=1.02,
                first_token_ts=1.1,
                last_token_ts=1.14,
            ),
        )
        metrics = request_output_metrics(output)
        self.assertAlmostEqual(metrics["ttft_ms"], 100.0)
        self.assertAlmostEqual(metrics["queue_ms"], 20.0)
        self.assertAlmostEqual(metrics["tpot_ms"], 20.0)

    def test_flatten_record_preserves_lists_as_json(self) -> None:
        self.assertEqual(
            flatten_record({"shape": {"grid": [[1, 2, 3]]}}),
            {"shape.grid": "[[1, 2, 3]]"},
        )


if __name__ == "__main__":
    unittest.main()
