"""Unit tests for the NPU baseline report."""

from __future__ import annotations

import statistics
import unittest

from experiments.multimodal_cpu.analyze_npu_baseline import (
    metric_stats,
    percentile,
    render_report,
)


def _record(repeat: int, latency_offset: float = 0.0) -> dict:
    return {
        "experiment": "npu_baseline",
        "model": "Qwen2.5-VL-7B-Instruct",
        "repeat": repeat,
        "tensor_parallel_size": 1,
        "expected_output_tokens": 100,
        "wall_e2e_ms": 2500.0 + latency_offset,
        "media": {
            "image_path": "/data/image.jpg",
            "original_file_bytes": 54497,
            "original_format": "JPEG",
            "original_mode": "RGB",
            "original_width": 1024,
            "original_height": 576,
        },
        "input_scale": {
            "source": "vllm_runtime",
            "grid_thw": [[1, 42, 74]],
            "patch_size": 14,
            "patch_count": 3108,
            "spatial_merge_size": 2,
            "visual_token_count": 777,
            "processed_width": 1036,
            "processed_height": 588,
            "pixel_values": {
                "shape": [3108, 1176],
                "dtype": "torch.bfloat16",
                "tensor_bytes": 7310016,
            },
            "encoder_input_tensors": {
                "image_grid_thw": {
                    "shape": [1, 3],
                    "dtype": "torch.int64",
                    "tensor_bytes": 24,
                }
            },
            "encoder_input_tensor_bytes": 7310040,
            "encoder_output": {
                "shape": [777, 3584],
                "dtype": "torch.bfloat16",
                "tensor_bytes": 5569536,
            },
        },
        "request": {
            "finished": True,
            "output_token_count": 100,
            "prompt_token_count": 802,
            "ttft_ms": 290.0 + latency_offset,
            "scheduled_to_first_token_ms": 180.0 + latency_offset,
            "decode_ms": 2210.0 + latency_offset,
            "tpot_ms": 22.0 + latency_offset,
            "engine_e2e_ms": 2500.0 + latency_offset,
        },
        "stages": {
            "num_encoder_calls": 1,
            "preprocessor_total_ms": 105.0 + latency_offset,
            "encoder_forward_ms": 70.0 + latency_offset,
        },
    }


class AnalyzeNpuBaselineTest(unittest.TestCase):
    def test_percentiles_use_linear_interpolation(self) -> None:
        values = list(range(1, 11))

        self.assertEqual(percentile(values, 0.5), 5.5)
        self.assertAlmostEqual(percentile(values, 0.9), 9.1)

    def test_metric_stats_use_sample_standard_deviation(self) -> None:
        values = [1.0, 2.0, 3.0]
        mean, _, _, cv_percent = metric_stats(values)

        self.assertEqual(mean, 2.0)
        self.assertAlmostEqual(
            cv_percent, statistics.stdev(values) / mean * 100
        )

    def test_report_contains_requested_tables(self) -> None:
        report = render_report([_record(0), _record(1, 1.0)])

        self.assertIn("| 原始图片 | 1024×576，JPEG RGB |", report)
        self.assertIn(
            "| Spatial Merge | 2（2×2，共4个Patch合并） |", report
        )
        self.assertIn(
            "| Encoder总输入 | — | — | 7,310,040 Bytes，6.97 MiB |",
            report,
        )
        self.assertIn("| Preprocessor | 105.50 ms | 105.50 | 105.90 |", report)


if __name__ == "__main__":
    unittest.main()
