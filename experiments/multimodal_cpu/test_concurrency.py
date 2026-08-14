"""Unit tests for concurrency result aggregation."""

from __future__ import annotations

import unittest

from experiments.multimodal_cpu.run_concurrency import (
    aggregate_worker_batch_stats,
)


class ConcurrencyTest(unittest.TestCase):
    def test_tp_workers_use_slowest_batch_time(self) -> None:
        workers = [
            [
                {
                    "batch_id": 1,
                    "encoder_forward_secs": 0.1,
                    "num_items": 2,
                    "num_requests": 2,
                    "num_encoder_tokens": 48,
                    "request_ids": ["a", "b"],
                }
            ],
            [
                {
                    "batch_id": 1,
                    "encoder_forward_secs": 0.12,
                    "num_items": 2,
                    "num_requests": 2,
                    "num_encoder_tokens": 48,
                    "request_ids": ["a", "b"],
                }
            ],
        ]
        result = aggregate_worker_batch_stats(workers)
        self.assertEqual(result[0]["encoder_forward_ms"], 120.0)
        self.assertEqual(result[0]["worker_ranks"], [0, 1])


if __name__ == "__main__":
    unittest.main()
