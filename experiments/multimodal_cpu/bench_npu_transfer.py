"""Measure synchronized CPU/NPU transfers for real multimodal tensors."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from experiments.multimodal_cpu.common import write_csv, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CPU/NPU transfer latency and effective bandwidth."
    )
    parser.add_argument("--input-scale-jsonl", required=True)
    parser.add_argument(
        "--model",
        help="Model path. Defaults to the model field in the input-scale row.",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--sweep-mb",
        default="1,4,16,64,256",
        help="Additional payload sizes used to fit fixed transfer overhead.",
    )
    parser.add_argument("--pageable", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-prefix", default="results/npu_transfer")
    return parser.parse_args()


def load_first_jsonl(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"No rows found in {path}.")


def authoritative_input_scale(record: dict[str, Any]) -> dict[str, Any]:
    """Extract tensor metadata captured by the real vLLM runtime."""
    scale = record.get("input_scale")
    if not isinstance(scale, dict) or scale.get("source") != "vllm_runtime":
        raise ValueError(
            "--input-scale-jsonl must contain run_npu_baseline rows with "
            "input_scale.source='vllm_runtime'; collect_input_scale output is "
            "only an estimate."
        )
    return scale


def percentile_90(samples: list[float]) -> float:
    index = max(0, math.ceil(0.9 * len(samples)) - 1)
    return sorted(samples)[index]


def measure(
    operation: Callable[[], None],
    synchronize: Callable[[], None],
    warmups: int,
    iterations: int,
) -> tuple[float, float]:
    for _ in range(warmups):
        operation()
        synchronize()

    samples_ms = []
    for _ in range(iterations):
        synchronize()
        start = time.perf_counter()
        operation()
        synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples_ms), percentile_90(samples_ms)


def normalize_dtype_name(dtype: str) -> str:
    return dtype.removeprefix("torch.")


def build_cases(
    scale: dict[str, Any], hidden_size: int, model_dtype: str, sweep_mb: str
) -> list[dict[str, Any]]:
    pixel = scale["pixel_values"]
    encoder_output = scale["encoder_output"]
    if "shape" not in encoder_output:
        raise ValueError(
            "The baseline row must describe exactly one runtime encoder output "
            "tensor. Run the single-image NPU baseline first."
        )
    cases = [
        {
            "tensor_role": "pixel_input",
            "shape": pixel["shape"],
            "dtype": normalize_dtype_name(pixel["dtype"]),
            "shape_source": "vllm_runtime",
        },
        {
            "tensor_role": "encoder_output",
            "shape": encoder_output["shape"],
            "dtype": normalize_dtype_name(encoder_output["dtype"]),
            "shape_source": "vllm_runtime",
        },
        {
            "tensor_role": "vit_cut_activation",
            "shape": [int(scale["patch_count"]), hidden_size],
            "dtype": model_dtype,
            "shape_source": (
                "derived_from_runtime_patch_count_and_model_config"
            ),
        },
    ]
    cases.extend(
        {
            "tensor_role": f"sweep_{size_mb}mb",
            "payload_bytes": size_mb * 1024**2,
            "dtype": model_dtype,
        }
        for size_mb in (int(item) for item in sweep_mb.split(","))
    )
    return cases


def benchmark_case(
    torch: Any,
    case: dict[str, Any],
    device: str,
    pinned: bool,
    warmups: int,
    iterations: int,
) -> list[dict[str, Any]]:
    dtype = getattr(torch, case["dtype"])
    element_size = torch.empty((), dtype=dtype).element_size()
    if "shape" in case:
        numel = math.prod(case["shape"])
    else:
        numel = math.ceil(case["payload_bytes"] / element_size)
    payload_bytes = numel * element_size

    cpu_source = torch.empty(numel, dtype=dtype, pin_memory=pinned)
    cpu_destination = torch.empty(numel, dtype=dtype, pin_memory=pinned)
    npu_tensor = torch.empty(numel, dtype=dtype, device=device)
    synchronize = torch.npu.synchronize

    operations = {
        "h2d": lambda: npu_tensor.copy_(
            cpu_source, non_blocking=pinned
        ),
        "d2h": lambda: cpu_destination.copy_(
            npu_tensor, non_blocking=pinned
        ),
        "roundtrip": lambda: (
            cpu_destination.copy_(npu_tensor, non_blocking=pinned),
            npu_tensor.copy_(cpu_source, non_blocking=pinned),
        ),
    }
    records = []
    for direction, operation in operations.items():
        median_ms, p90_ms = measure(
            operation, synchronize, warmups, iterations
        )
        direction_bytes = payload_bytes * (2 if direction == "roundtrip" else 1)
        records.append(
            {
                **case,
                "direction": direction,
                "pinned_memory": pinned,
                "payload_bytes": payload_bytes,
                "transferred_bytes": direction_bytes,
                "median_ms": median_ms,
                "p90_ms": p90_ms,
                "effective_gbps": direction_bytes / (median_ms / 1000) / 1e9,
            }
        )
    return records


def main() -> None:
    args = parse_args()
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoConfig

    baseline_record = load_first_jsonl(args.input_scale_jsonl)
    scale = authoritative_input_scale(baseline_record)
    model = args.model or baseline_record["model"]
    config = AutoConfig.from_pretrained(
        model, trust_remote_code=args.trust_remote_code
    )
    cases = build_cases(
        scale,
        int(config.vision_config.hidden_size),
        args.model_dtype,
        args.sweep_mb,
    )
    device_name = torch.npu.get_device_name(int(args.device.rsplit(":", 1)[-1]))

    records: list[dict[str, Any]] = []
    for case in cases:
        for result in benchmark_case(
            torch,
            case,
            args.device,
            not args.pageable,
            args.warmups,
            args.iterations,
        ):
            records.append(
                {
                    "experiment": "npu_transfer",
                    "model": model,
                    "device": args.device,
                    "device_name": device_name,
                    "input_scale": scale,
                    "input_scale_estimate": baseline_record.get(
                        "input_scale_estimate"
                    ),
                    "input_scale_comparison": baseline_record.get(
                        "input_scale_comparison"
                    ),
                    **result,
                }
            )

    prefix = Path(args.output_prefix)
    write_jsonl(prefix.with_suffix(".jsonl"), records)
    write_csv(prefix.with_suffix(".csv"), records)
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
