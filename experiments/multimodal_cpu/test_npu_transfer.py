"""Unit tests for NPU transfer case construction."""

from __future__ import annotations

import unittest

from experiments.multimodal_cpu.bench_npu_transfer import build_cases


class NpuTransferTest(unittest.TestCase):
    def test_real_cases_use_patch_and_encoder_shapes(self) -> None:
        scale = {
            "patch_count": 96,
            "pixel_values": {"shape": [96, 1176], "dtype": "torch.float32"},
            "encoder_output_estimate": {"shape": [24, 3584]},
        }
        cases = build_cases(scale, 1280, "bfloat16", "1")
        self.assertEqual(cases[1]["shape"], [24, 3584])
        self.assertEqual(cases[2]["shape"], [96, 1280])


if __name__ == "__main__":
    unittest.main()
