"""Unit tests for dependency-free measurement helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from experiments.multimodal_cpu.common import (
    compare_input_scales,
    flatten_record,
    grid_metrics,
    merge_request_stage_stats,
    normalize_internal_request_id,
    request_output_metrics,
    runtime_input_scale_from_batch,
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

    def test_merge_request_stage_stats_across_id_domains(self) -> None:
        stats = {
            "renderer0-mm-7": {"preprocessor_total_secs": 0.2},
            "1-acde1234": {
                "encoder_forward_secs": 0.1,
                "num_encoder_calls": 1,
            },
        }

        merged = merge_request_stage_stats(stats, ["1"])

        self.assertEqual(normalize_internal_request_id("1-acde1234"), "1")
        self.assertEqual(
            merged,
            {
                "1": {
                    "preprocessor_total_secs": 0.2,
                    "encoder_forward_secs": 0.1,
                    "num_encoder_calls": 1,
                }
            },
        )

    def test_runtime_input_scale_and_estimate_difference(self) -> None:
        batch = {
            "batch_id": 3,
            "worker_rank": 0,
            "modality": "image",
            "num_items": 1,
            "num_encoder_tokens": 4,
            "grid_thw": [[1, 4, 4]],
            "input_tensors": {
                "pixel_values": {
                    "shape": [16, 1176],
                    "dtype": "torch.float32",
                    "tensor_bytes": 75264,
                }
            },
            "input_tensor_bytes": 75388,
            "output_tensors": [
                {
                    "shape": [4, 3584],
                    "dtype": "torch.bfloat16",
                    "tensor_bytes": 28672,
                }
            ],
            "output_tensor_bytes": 28672,
        }
        estimate = {
            "processed_width": 56,
            "processed_height": 56,
            "grid_thw": [[1, 4, 4]],
            "patch_count": 16,
            "visual_token_count": 4,
            "pixel_values": {
                "shape": [16, 1176],
                "dtype": "torch.float32",
                "tensor_bytes": 75264,
            },
            "encoder_output_estimate": {
                "shape": [4, 3584],
                "tensor_bytes": 28000,
            },
        }

        actual = runtime_input_scale_from_batch(batch, 14, 2)
        comparison = compare_input_scales(actual, estimate)

        self.assertEqual(actual["source"], "vllm_runtime")
        self.assertEqual(actual["patch_count"], 16)
        self.assertEqual(actual["processed_width"], 56)
        self.assertEqual(actual["encoder_output"]["shape"], [4, 3584])
        self.assertTrue(comparison["grid_thw"]["matches"])
        self.assertEqual(
            comparison["encoder_output.tensor_bytes"],
            {
                "actual": 28672,
                "estimate": 28000,
                "matches": False,
                "difference": 672,
                "relative_difference": 0.024,
            },
        )


if __name__ == "__main__":
    unittest.main()
