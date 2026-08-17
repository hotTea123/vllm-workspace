"""Unit tests for NPU baseline record association."""

from __future__ import annotations

import unittest

from experiments.multimodal_cpu.run_npu_baseline import (
    _runtime_batch_for_output,
)


class NpuBaselineTest(unittest.TestCase):
    def test_selects_first_worker_for_internal_request_id(self) -> None:
        batches = [
            {
                "worker_rank": 1,
                "request_ids": ["1-acde1234"],
                "batch_id": 0,
            },
            {
                "worker_rank": 0,
                "request_ids": ["1-acde1234"],
                "batch_id": 0,
            },
        ]

        selected = _runtime_batch_for_output(batches, "1")

        self.assertEqual(selected["worker_rank"], 0)


if __name__ == "__main__":
    unittest.main()
