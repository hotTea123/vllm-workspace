"""Shared measurement helpers for multimodal CPU-offload experiments."""

from __future__ import annotations

import csv
import io
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def load_image_with_metrics(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load one image and separate file-read, decode, and RGB-convert time."""
    from PIL import Image

    image_path = Path(path)
    read_start = time.perf_counter()
    payload = image_path.read_bytes()
    read_ms = (time.perf_counter() - read_start) * 1000

    decode_start = time.perf_counter()
    image = Image.open(io.BytesIO(payload))
    image.load()
    decode_ms = (time.perf_counter() - decode_start) * 1000

    original_mode = image.mode
    convert_start = time.perf_counter()
    rgb_image = image.convert("RGB")
    convert_ms = (time.perf_counter() - convert_start) * 1000

    metadata = {
        "image_path": str(image_path.resolve()),
        "original_file_bytes": len(payload),
        "original_format": image.format,
        "original_mode": original_mode,
        "original_width": image.width,
        "original_height": image.height,
        "media_read_ms": read_ms,
        "media_decode_ms": decode_ms,
        "media_convert_rgb_ms": convert_ms,
    }
    return rgb_image, metadata


def tensor_metadata(tensor: Any) -> dict[str, Any]:
    """Return shape, dtype, element count, and storage bytes for a tensor."""
    shape = [int(dim) for dim in tensor.shape]
    numel = int(tensor.numel())
    element_size = int(tensor.element_size())
    return {
        "shape": shape,
        "dtype": str(tensor.dtype),
        "numel": numel,
        "element_size_bytes": element_size,
        "tensor_bytes": numel * element_size,
    }


def grid_metrics(grid_thw: Any, merge_size: int) -> dict[str, Any]:
    """Calculate patch and post-merge visual-token counts from image_grid_thw."""
    if hasattr(grid_thw, "tolist"):
        grid = grid_thw.tolist()
    else:
        grid = grid_thw

    if grid and isinstance(grid[0], int):
        grid = [grid]
    normalized = [[int(value) for value in item] for item in grid]
    patch_count = sum(t * h * w for t, h, w in normalized)
    merge_unit = merge_size**2
    if patch_count % merge_unit:
        raise ValueError(
            f"Patch count {patch_count} is not divisible by merge unit "
            f"{merge_unit}."
        )
    return {
        "grid_thw": normalized,
        "patch_count": patch_count,
        "spatial_merge_size": merge_size,
        "visual_token_count": patch_count // merge_unit,
    }


def request_output_metrics(output: Any) -> dict[str, Any]:
    """Extract engine-side latency and token counts from a RequestOutput."""
    generated_tokens = sum(len(item.token_ids) for item in output.outputs)
    result: dict[str, Any] = {
        "request_id": output.request_id,
        "prompt_token_count": len(output.prompt_token_ids),
        "output_token_count": generated_tokens,
        "finished": bool(output.finished),
    }

    metrics = output.metrics
    if metrics is None:
        return result

    first_token_ts = getattr(metrics, "first_token_ts", 0.0)
    last_token_ts = getattr(metrics, "last_token_ts", 0.0)
    queued_ts = getattr(metrics, "queued_ts", 0.0)
    scheduled_ts = getattr(metrics, "scheduled_ts", 0.0)
    ttft_s = getattr(metrics, "first_token_latency", 0.0)
    decode_s = max(0.0, last_token_ts - first_token_ts)

    result.update(
        {
            "ttft_ms": ttft_s * 1000,
            "queue_ms": max(0.0, scheduled_ts - queued_ts) * 1000,
            "scheduled_to_first_token_ms": max(
                0.0, first_token_ts - scheduled_ts
            )
            * 1000,
            "decode_ms": decode_s * 1000,
            "tpot_ms": (
                decode_s * 1000 / (generated_tokens - 1)
                if generated_tokens > 1
                else 0.0
            ),
            "engine_e2e_ms": (ttft_s + decode_s) * 1000,
        }
    )
    return result


def flatten_record(
    value: Mapping[str, Any], prefix: str = ""
) -> dict[str, Any]:
    """Flatten nested mappings for CSV output while preserving lists as JSON."""
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            flattened.update(flatten_record(item, full_key))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            flattened[full_key] = json.dumps(item, ensure_ascii=False)
        else:
            flattened[full_key] = item
    return flattened


def write_jsonl(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write measurement records as UTF-8 JSON Lines."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write measurement records as a flattened UTF-8 CSV file."""
    flattened = [flatten_record(record) for record in records]
    if not flattened:
        raise ValueError("At least one record is required for CSV output.")

    fieldnames = sorted({key for record in flattened for key in record})
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)
