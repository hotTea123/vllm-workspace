"""Unit tests for CPU roofline arithmetic."""

from __future__ import annotations

import unittest

from experiments.multimodal_cpu.bench_cpu_roofline import (
    authoritative_input_scale,
    case_storage_bytes,
)


class CpuRooflineTest(unittest.TestCase):
    def test_case_storage_includes_inputs_weight_and_output(self) -> None:
        self.assertEqual(case_storage_bytes(2, 3, 4, 2), 52)

    def test_requires_runtime_input_scale(self) -> None:
        runtime = {"source": "vllm_runtime", "patch_count": 128}
        self.assertIs(
            authoritative_input_scale({"input_scale": runtime}), runtime
        )
        with self.assertRaisesRegex(ValueError, "only an estimate"):
            authoritative_input_scale(
                {"source": "collect_scale_estimate", "patch_count": 128}
            )


if __name__ == "__main__":
    unittest.main()
