"""Render three summary tables from an NPU baseline JSONL file."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


LATENCY_METRICS = (
    ("Preprocessor", "stages.preprocessor_total_ms", 2),
    ("Encoder Forward", "stages.encoder_forward_ms", 2),
    ("TTFT", "request.ttft_ms", 2),
    ("Scheduled→First", "request.scheduled_to_first_token_ms", 2),
    ("Decode", "request.decode_ms", 2),
    ("TPOT", "request.tpot_ms", 3),
    ("Engine E2E", "request.engine_e2e_ms", 2),
    ("Wall E2E", "wall_e2e_ms", 2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze one cache-miss NPU baseline JSONL file."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}."
                ) from error
    if not records:
        raise ValueError(f"No records found in {path}.")
    return records


def _value(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        value = value[part]
    return value


def _validate(records: Sequence[Mapping[str, Any]]) -> None:
    first = records[0]
    if first["experiment"] != "npu_baseline":
        raise ValueError("The input is not an npu_baseline result.")

    invariant_paths = (
        "experiment",
        "model",
        "expected_output_tokens",
        "tensor_parallel_size",
        "media.image_path",
        "media.original_file_bytes",
        "media.original_format",
        "media.original_mode",
        "media.original_width",
        "media.original_height",
        "input_scale.source",
        "input_scale.grid_thw",
        "input_scale.patch_size",
        "input_scale.patch_count",
        "input_scale.spatial_merge_size",
        "input_scale.visual_token_count",
        "input_scale.processed_width",
        "input_scale.processed_height",
        "input_scale.pixel_values.shape",
        "input_scale.pixel_values.dtype",
        "input_scale.pixel_values.tensor_bytes",
        "input_scale.encoder_input_tensors.image_grid_thw.shape",
        "input_scale.encoder_input_tensors.image_grid_thw.dtype",
        "input_scale.encoder_input_tensors.image_grid_thw.tensor_bytes",
        "input_scale.encoder_input_tensor_bytes",
        "input_scale.encoder_output.shape",
        "input_scale.encoder_output.dtype",
        "input_scale.encoder_output.tensor_bytes",
        "request.prompt_token_count",
    )
    for path in invariant_paths:
        expected = _value(first, path)
        if any(_value(record, path) != expected for record in records[1:]):
            raise ValueError(f"Records do not share one value for {path}.")

    repeats = [int(record["repeat"]) for record in records]
    if len(set(repeats)) != len(repeats):
        raise ValueError("Duplicate repeat IDs found.")
    for record in records:
        if not record["request"]["finished"]:
            raise ValueError(f"Request in repeat {record['repeat']} did not finish.")
        if (
            record["request"]["output_token_count"]
            != record["expected_output_tokens"]
        ):
            raise ValueError(
                f"Output length mismatch in repeat {record['repeat']}."
            )
        if record["stages"]["num_encoder_calls"] != 1:
            raise ValueError(
                f"Expected one encoder call in repeat {record['repeat']}."
            )
        if record["input_scale"]["source"] != "vllm_runtime":
            raise ValueError("input_scale must come from vllm_runtime.")


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (
        ordered[upper] - ordered[lower]
    )


def metric_stats(values: Sequence[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    cv_percent = standard_deviation / mean * 100 if mean else 0.0
    return mean, percentile(values, 0.5), percentile(values, 0.9), cv_percent


def _shape(value: Sequence[int]) -> str:
    return "[" + ",".join(str(int(item)) for item in value) + "]"


def _dtype(value: str) -> str:
    labels = {
        "torch.bfloat16": "BF16",
        "torch.float16": "FP16",
        "torch.float32": "FP32",
        "torch.int64": "INT64",
        "torch.int32": "INT32",
    }
    return labels.get(value, value.removeprefix("torch.").upper())


def _bytes(value: int, include_mib: bool = False) -> str:
    rendered = f"{int(value):,} Bytes"
    if include_mib:
        rendered += f"，{value / (1024**2):.2f} MiB"
    return rendered


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(records: Sequence[Mapping[str, Any]]) -> str:
    _validate(records)
    first = records[0]
    media = first["media"]
    scale = first["input_scale"]
    pixel_values = scale["pixel_values"]
    grid_tensor = scale["encoder_input_tensors"]["image_grid_thw"]
    encoder_output = scale["encoder_output"]
    grid = scale["grid_thw"][0]
    merge_size = int(scale["spatial_merge_size"])
    prompt_tokens = int(first["request"]["prompt_token_count"])
    visual_tokens = int(scale["visual_token_count"])

    input_scale_table = _table(
        ("指标", "数值"),
        (
            (
                "原始图片",
                f"{media['original_width']}×{media['original_height']}"
                f"，{media['original_format']} {media['original_mode']}",
            ),
            ("文件大小", _bytes(media["original_file_bytes"])),
            (
                "处理后尺寸",
                f"{scale['processed_width']}×{scale['processed_height']}",
            ),
            ("grid_thw", _shape(grid)),
            ("Patch Size", f"{scale['patch_size']}×{scale['patch_size']}"),
            ("Patch数量", f"{int(scale['patch_count']):,}"),
            (
                "Spatial Merge",
                f"{merge_size}（{merge_size}×{merge_size}"
                f"，共{merge_size**2}个Patch合并）",
            ),
            ("Visual Token", f"{visual_tokens:,}"),
            (
                "非视觉Token",
                f"{prompt_tokens - visual_tokens:,}（含文本/控制Token）",
            ),
            ("Prompt Token", f"{prompt_tokens:,}"),
        ),
    )

    tensor_table = _table(
        ("Tensor", "Shape", "类型", "大小"),
        (
            (
                "Pixel Values",
                _shape(pixel_values["shape"]),
                _dtype(pixel_values["dtype"]),
                _bytes(pixel_values["tensor_bytes"]),
            ),
            (
                "Grid Tensor",
                _shape(grid_tensor["shape"]),
                _dtype(grid_tensor["dtype"]),
                _bytes(grid_tensor["tensor_bytes"]),
            ),
            (
                "Encoder总输入",
                "—",
                "—",
                _bytes(scale["encoder_input_tensor_bytes"], include_mib=True),
            ),
            (
                "Encoder输出",
                _shape(encoder_output["shape"]),
                _dtype(encoder_output["dtype"]),
                _bytes(encoder_output["tensor_bytes"], include_mib=True),
            ),
        ),
    )

    latency_rows: list[tuple[str, str, str, str, str]] = []
    for label, path, digits in LATENCY_METRICS:
        values = [float(_value(record, path)) for record in records]
        mean, p50, p90, cv_percent = metric_stats(values)
        cv_digits = 3 if cv_percent < 0.1 else 2
        latency_rows.append(
            (
                label,
                f"{mean:.{digits}f} ms",
                f"{p50:.{digits}f}",
                f"{p90:.{digits}f}",
                f"{cv_percent:.{cv_digits}f}%",
            )
        )
    latency_table = _table(
        ("指标", "平均", "P50", "P90", "波动率"), latency_rows
    )

    return "\n\n".join(
        (
            "## 输入规模",
            input_scale_table,
            "## Encoder Tensor规模",
            tensor_table,
            "## 时延统计",
            latency_table,
        )
    )


def main() -> None:
    args = parse_args()
    report = render_report(load_records(args.input))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
