"""Benchmark CPU ceilings with the real Qwen2.5-VL ViT matrix shapes."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from experiments.multimodal_cpu.common import write_csv, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure CPU bandwidth and ViT-shaped GEMM throughput."
    )
    parser.add_argument(
        "--input-scale-jsonl",
        required=True,
        help="JSONL produced by run_npu_baseline.",
    )
    parser.add_argument(
        "--model",
        help="Model path. Defaults to the model field in the input-scale row.",
    )
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--threads", default="1,8,16,32,64", help="Comma-separated thread counts."
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--memory-size-mb",
        type=int,
        default=256,
        help="Payload size for the sustained memory-copy case.",
    )
    parser.add_argument(
        "--max-case-gb",
        type=float,
        default=8.0,
        help="Skip a case when its input, weight, and output exceed this size.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-prefix", default="results/cpu_roofline")
    return parser.parse_args()


def load_first_jsonl(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"No rows found in {path}.")


def authoritative_input_scale(record: dict[str, Any]) -> dict[str, Any]:
    """Extract the real vLLM runtime scale from a baseline row."""
    scale = record.get("input_scale")
    if not isinstance(scale, dict) or scale.get("source") != "vllm_runtime":
        raise ValueError(
            "--input-scale-jsonl must contain run_npu_baseline rows with "
            "input_scale.source='vllm_runtime'; collect_input_scale output is "
            "only an estimate."
        )
    return scale


def measure(
    operation: Callable[[], None], warmups: int, iterations: int
) -> tuple[float, float]:
    for _ in range(warmups):
        operation()
    samples_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        samples_ms.append((time.perf_counter() - start) * 1000)
    p90_index = max(0, math.ceil(0.9 * len(samples_ms)) - 1)
    return statistics.median(samples_ms), sorted(samples_ms)[p90_index]


def case_storage_bytes(
    m: int, k: int, n: int, element_size: int
) -> int:
    return (m * k + k * n + m * n) * element_size


def benchmark_gemm(
    torch: Any,
    name: str,
    m: int,
    k: int,
    n: int,
    dtype: Any,
    warmups: int,
    iterations: int,
    max_case_bytes: int,
) -> dict[str, Any]:
    element_size = torch.empty((), dtype=dtype).element_size()
    storage_bytes = case_storage_bytes(m, k, n, element_size)
    base = {
        "operator": name,
        "shape": [m, k, n],
        "case_storage_bytes": storage_bytes,
        "flops": 2 * m * k * n,
    }
    if storage_bytes > max_case_bytes:
        return {**base, "status": "skipped_max_case_bytes"}

    lhs = torch.zeros((m, k), dtype=dtype)
    rhs = torch.zeros((k, n), dtype=dtype)
    output = torch.empty((m, n), dtype=dtype)

    def operation() -> None:
        torch.mm(lhs, rhs, out=output)

    try:
        median_ms, p90_ms = measure(operation, warmups, iterations)
    except RuntimeError as error:
        return {**base, "status": "unsupported", "error": str(error)}

    return {
        **base,
        "status": "ok",
        "median_ms": median_ms,
        "p90_ms": p90_ms,
        "gflops": base["flops"] / (median_ms / 1000) / 1e9,
    }


def benchmark_memory_copy(
    torch: Any,
    name: str,
    numel: int,
    dtype: Any,
    warmups: int,
    iterations: int,
) -> dict[str, Any]:
    source = torch.zeros(numel, dtype=dtype)
    destination = torch.empty_like(source)
    element_size = source.element_size()

    def operation() -> None:
        destination.copy_(source)

    median_ms, p90_ms = measure(operation, warmups, iterations)
    payload_bytes = numel * element_size
    traffic_bytes = payload_bytes * 2
    return {
        "operator": name,
        "shape": [numel],
        "status": "ok",
        "payload_bytes": payload_bytes,
        "estimated_read_write_bytes": traffic_bytes,
        "median_ms": median_ms,
        "p90_ms": p90_ms,
        "payload_gbps": payload_bytes / (median_ms / 1000) / 1e9,
        "estimated_traffic_gbps": traffic_bytes / (median_ms / 1000) / 1e9,
    }


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoConfig

    baseline_record = load_first_jsonl(args.input_scale_jsonl)
    scale = authoritative_input_scale(baseline_record)
    model = args.model or baseline_record["model"]
    config = AutoConfig.from_pretrained(
        model, trust_remote_code=args.trust_remote_code
    )
    vision = config.vision_config
    patch_tokens = int(scale["patch_count"])
    hidden_size = int(vision.hidden_size)
    intermediate_size = int(vision.intermediate_size)
    dtype = getattr(torch, args.dtype)
    thread_counts = [int(item) for item in args.threads.split(",")]
    max_case_bytes = int(args.max_case_gb * 1024**3)
    element_size = torch.empty((), dtype=dtype).element_size()
    stream_copy_numel = args.memory_size_mb * 1024**2 // element_size

    torch.set_num_interop_threads(1)
    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None
    )
    environment = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "torch_version": torch.__version__,
        "mkldnn_available": torch.backends.mkldnn.is_available(),
    }
    shapes = (
        ("qkv_projection", patch_tokens, hidden_size, 3 * hidden_size),
        ("attention_output", patch_tokens, hidden_size, hidden_size),
        ("mlp_gate_up", patch_tokens, hidden_size, 2 * intermediate_size),
        ("mlp_down", patch_tokens, intermediate_size, hidden_size),
    )

    records: list[dict[str, Any]] = []
    for threads in thread_counts:
        torch.set_num_threads(threads)
        common = {
            "experiment": "cpu_roofline",
            "model": model,
            "dtype": args.dtype,
            "threads": threads,
            "patch_tokens": patch_tokens,
            "visual_tokens": int(scale["visual_token_count"]),
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "environment": environment,
            "input_scale": scale,
            "input_scale_estimate": baseline_record.get(
                "input_scale_estimate"
            ),
            "input_scale_comparison": baseline_record.get(
                "input_scale_comparison"
            ),
        }
        records.append(
            {
                **common,
                **benchmark_memory_copy(
                    torch,
                    "stream_copy",
                    stream_copy_numel,
                    dtype,
                    args.warmups,
                    args.iterations,
                ),
            }
        )
        records.append(
            {
                **common,
                **benchmark_memory_copy(
                    torch,
                    "activation_copy",
                    patch_tokens * hidden_size,
                    dtype,
                    args.warmups,
                    args.iterations,
                ),
            }
        )
        for name, m, k, n in shapes:
            records.append(
                {
                    **common,
                    **benchmark_gemm(
                        torch,
                        name,
                        m,
                        k,
                        n,
                        dtype,
                        args.warmups,
                        args.iterations,
                        max_case_bytes,
                    ),
                }
            )

    prefix = Path(args.output_prefix)
    write_jsonl(prefix.with_suffix(".jsonl"), records)
    write_csv(prefix.with_suffix(".csv"), records)
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
