"""Unit tests for concurrency result aggregation."""

from __future__ import annotations

import unittest

from experiments.multimodal_cpu.run_concurrency import (
    aggregate_worker_batch_stats,
    batch_scale_estimate,
    request_runtime_scales,
    runtime_batch_input_scale,
)


class ConcurrencyTest(unittest.TestCase):
    def test_tp_workers_use_slowest_batch_time(self) -> None:
        workers = [
            {
                **self.batch_record(),
                "worker_rank": 0,
                "encoder_forward_secs": 0.1,
            },
            {
                **self.batch_record(),
                "worker_rank": 1,
                "encoder_forward_secs": 0.12,
            },
        ]
        result = aggregate_worker_batch_stats(workers)
        self.assertEqual(result[0]["encoder_forward_ms"], 120.0)
        self.assertEqual(result[0]["worker_ranks"], [0, 1])
        self.assertEqual(len(result[0]["rank_records"]), 2)

    def test_request_scale_is_sliced_from_runtime_batch(self) -> None:
        batch = aggregate_worker_batch_stats(
            [
                {
                    **self.batch_record(),
                    "worker_rank": 0,
                    "encoder_forward_secs": 0.1,
                }
            ]
        )[0]
        scales = request_runtime_scales([batch], ["a", "b"], 14, 2)
        self.assertEqual(scales["a"]["grid_thw"], [[1, 2, 3]])
        self.assertEqual(scales["a"]["patch_count"], 6)
        self.assertEqual(scales["a"]["pixel_values"]["tensor_bytes"], 96)
        self.assertEqual(scales["b"]["encoder_output"]["shape"], [3, 8])

        batch_scale = runtime_batch_input_scale(batch, 14, 2)
        self.assertEqual(batch_scale["patch_count"], 14)
        self.assertEqual(batch_scale["encoder_output"]["shape"], [5, 8])

    def test_batch_estimate_scales_auxiliary_values(self) -> None:
        estimate = {
            "source": "collect_scale_estimate",
            "grid_thw": [[1, 2, 3]],
            "patch_count": 6,
            "visual_token_count": 2,
            "processed_width": 42,
            "processed_height": 28,
            "pixel_values": {
                "shape": [6, 4],
                "numel": 24,
                "tensor_bytes": 96,
            },
            "encoder_output_estimate": {
                "shape": [2, 8],
                "tensor_bytes": 32,
            },
        }
        scaled = batch_scale_estimate(estimate, 2)
        self.assertEqual(scaled["patch_count"], 12)
        self.assertEqual(scaled["pixel_values"]["shape"], [12, 4])
        self.assertEqual(scaled["encoder_output_estimate"]["tensor_bytes"], 64)
        self.assertNotIn("processed_width", scaled)
        self.assertEqual(estimate["patch_count"], 6)

    @staticmethod
    def batch_record() -> dict:
        return {
            "batch_id": 1,
            "modality": "image",
            "num_items": 2,
            "num_requests": 2,
            "num_encoder_tokens": 5,
            "request_ids": ["a-12345678", "b-12345678"],
            "item_request_ids": ["a-12345678", "b-12345678"],
            "grid_thw": [[1, 2, 3], [1, 2, 4]],
            "input_tensors": {
                "pixel_values": {
                    "shape": [14, 4],
                    "dtype": "torch.float32",
                    "device": "npu:0",
                    "numel": 56,
                    "element_size_bytes": 4,
                    "tensor_bytes": 224,
                }
            },
            "input_tensor_bytes": 224,
            "output_tensors": [
                {
                    "shape": [2, 8],
                    "dtype": "torch.bfloat16",
                    "device": "npu:0",
                    "numel": 16,
                    "element_size_bytes": 2,
                    "tensor_bytes": 32,
                },
                {
                    "shape": [3, 8],
                    "dtype": "torch.bfloat16",
                    "device": "npu:0",
                    "numel": 24,
                    "element_size_bytes": 2,
                    "tensor_bytes": 48,
                },
            ],
            "output_tensor_bytes": 80,
        }


if __name__ == "__main__":
    unittest.main()
