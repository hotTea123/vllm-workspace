"""Unit tests for vision-memory and KV-cache calculations."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from experiments.multimodal_cpu.analyze_memory_kv import (
    calculate_kv_bytes_per_token,
    kv_heads_per_rank,
    tensor_storage_bytes,
)


class MemoryKvTest(unittest.TestCase):
    def test_tensor_storage_bytes(self) -> None:
        self.assertEqual(tensor_storage_bytes([2, 3], "BF16"), 12)

    def test_kv_heads_can_be_replicated(self) -> None:
        self.assertEqual(kv_heads_per_rank(2, 4), 1)

    def test_kv_bytes_per_token(self) -> None:
        config = SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            hidden_size=1024,
        )
        result = calculate_kv_bytes_per_token(config, 2, 2)
        self.assertEqual(result["kv_bytes_per_token_per_npu"], 2048)


if __name__ == "__main__":
    unittest.main()
