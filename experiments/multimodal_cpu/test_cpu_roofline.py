"""Unit tests for CPU roofline arithmetic."""

from __future__ import annotations

import unittest

from experiments.multimodal_cpu.bench_cpu_roofline import case_storage_bytes


class CpuRooflineTest(unittest.TestCase):
    def test_case_storage_includes_inputs_weight_and_output(self) -> None:
        self.assertEqual(case_storage_bytes(2, 3, 4, 2), 52)


if __name__ == "__main__":
    unittest.main()
