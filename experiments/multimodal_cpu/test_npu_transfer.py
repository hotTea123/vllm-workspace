"""Unit tests for NPU transfer case construction."""

from __future__ import annotations

import unittest

from experiments.multimodal_cpu.bench_npu_transfer import (
    authoritative_input_scale,
    build_cases,
)


class NpuTransferTest(unittest.TestCase):
    def test_real_cases_use_patch_and_encoder_shapes(self) -> None:
        scale = {
            "patch_count": 96,
            "pixel_values": {"shape": [96, 1176], "dtype": "torch.float32"},
            "encoder_output": {
                "shape": [24, 3584],
                "dtype": "torch.bfloat16",
            },
        }
        cases = build_cases(scale, 1280, "bfloat16", "1")
        self.assertEqual(cases[1]["shape"], [24, 3584])
        self.assertEqual(cases[1]["dtype"], "bfloat16")
        self.assertEqual(cases[1]["shape_source"], "vllm_runtime")
        self.assertEqual(cases[2]["shape"], [96, 1280])
        self.assertIn("derived_from_runtime", cases[2]["shape_source"])

    def test_requires_runtime_input_scale(self) -> None:
        runtime = {"source": "vllm_runtime", "patch_count": 96}
        self.assertIs(
            authoritative_input_scale({"input_scale": runtime}), runtime
        )
        with self.assertRaisesRegex(ValueError, "only an estimate"):
            authoritative_input_scale(
                {"source": "collect_scale_estimate", "patch_count": 96}
            )


if __name__ == "__main__":
    unittest.main()
