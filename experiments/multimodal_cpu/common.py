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


def normalize_internal_request_id(request_id: str) -> str:
    """Strip the V1 engine's eight-character request-id suffix."""
    external_id, separator, suffix = request_id.rpartition("-")
    if separator and external_id and len(suffix) == 8:
        return external_id
    return request_id


def merge_request_stage_stats(
    all_stats: Mapping[str, Mapping[str, Any]], request_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Associate offline renderer and encoder stats with output request IDs.

    Offline ``LLM.generate`` renders prompts in input order. Preprocessor stats
    use renderer-local IDs, while encoder stats use randomized internal request
    IDs. This preserves that ordering invariant without requiring those
    unrelated ID domains to match.
    """
    merged = {request_id: {} for request_id in request_ids}
    unassigned_preprocessor: list[Mapping[str, Any]] = []

    for stats_id, stats in all_stats.items():
        normalized_id = normalize_internal_request_id(stats_id)
        target_id = normalized_id if normalized_id in merged else None
        if target_id is not None:
            merged[target_id].update(stats)
        elif "preprocessor_total_secs" in stats:
            unassigned_preprocessor.append(stats)

    missing_preprocessor_ids = [
        request_id
        for request_id in request_ids
        if "preprocessor_total_secs" not in merged[request_id]
    ]
    for request_id, stats in zip(
        missing_preprocessor_ids, unassigned_preprocessor, strict=False
    ):
        merged[request_id].update(stats)

    return merged


def collect_encoder_batch_stats(llm: Any) -> list[dict[str, Any]]:
    """Drain actual encoder batch metadata from every model worker."""
    worker_stats = llm.llm_engine.collective_rpc(
        "get_encoder_batch_timing_stats"
    )
    return [
        {**item, "worker_rank": worker_rank}
        for worker_rank, rank_stats in enumerate(worker_stats)
        for item in rank_stats
    ]


def runtime_input_scale_from_batch(
    batch: Mapping[str, Any], patch_size: int, spatial_merge_size: int
) -> dict[str, Any]:
    """Build authoritative Qwen-VL input scale from one runtime batch."""
    grid = [[int(dim) for dim in item] for item in batch.get("grid_thw", [])]
    patch_count = sum(t * h * w for t, h, w in grid)
    processed_items = [
        {
            "processed_width": w * patch_size,
            "processed_height": h * patch_size,
        }
        for _, h, w in grid
    ]
    input_tensors = dict(batch.get("input_tensors", {}))
    pixel_values = input_tensors.get("pixel_values")
    if pixel_values is None:
        pixel_values = input_tensors.get("pixel_values_videos")

    output_tensors = list(batch.get("output_tensors", []))
    encoder_output: dict[str, Any] = {
        "items": output_tensors,
        "tensor_bytes": int(batch.get("output_tensor_bytes", 0)),
    }
    if len(output_tensors) == 1:
        encoder_output.update(output_tensors[0])

    result = {
        "source": "vllm_runtime",
        "worker_rank": int(batch.get("worker_rank", 0)),
        "encoder_batch_id": int(batch["batch_id"]),
        "modality": batch.get("modality"),
        "num_items": int(batch.get("num_items", len(grid))),
        "grid_thw": grid,
        "patch_size": patch_size,
        "patch_count": patch_count,
        "spatial_merge_size": spatial_merge_size,
        "visual_token_count": int(batch.get("num_encoder_tokens", 0)),
        "processed_items": processed_items,
        "pixel_values": pixel_values,
        "encoder_input_tensors": input_tensors,
        "encoder_input_tensor_bytes": int(batch.get("input_tensor_bytes", 0)),
        "encoder_output": encoder_output,
    }
    if len(processed_items) == 1:
        result.update(processed_items[0])
    return result


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
