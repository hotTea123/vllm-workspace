"""Run closed-batch multimodal concurrency sweeps with encoder batch stats."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiments.multimodal_cpu.collect_input_scale import collect as collect_scale
from experiments.multimodal_cpu.common import (
    collect_encoder_batch_stats,
    compare_input_scales,
    load_image_with_metrics,
    merge_request_stage_stats,
    normalize_internal_request_id,
    request_output_metrics,
    runtime_input_scale_from_batch,
    write_csv,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure multimodal batching and latency under concurrency."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--concurrency", default="1,4,8,16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-prefix", default="results/concurrency")
    return parser.parse_args()


def qwen_prompt(question: str) -> str:
    placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{placeholder}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def unique_image(image: Any, sample_index: int) -> Any:
    result = image.copy()
    red, green, blue = result.getpixel((0, 0))
    result.putpixel(
        (0, 0),
        (
            red ^ (sample_index & 0xFF),
            green ^ ((sample_index >> 8) & 0xFF),
            blue ^ ((sample_index >> 16) & 0xFF),
        ),
    )
    return result


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = max(0, math.ceil(fraction * len(values)) - 1)
    return sorted(values)[index]


def aggregate_worker_batch_stats(
    worker_stats: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge TP-rank batches while retaining every rank's runtime metadata."""
    batches: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}
    for item in worker_stats:
        rank = int(item["worker_rank"])
        request_ids = tuple(item["request_ids"])
        key = (int(item["batch_id"]), request_ids)
        if key not in batches:
            batches[key] = {
                **item,
                "representative_worker_rank": rank,
                "encoder_forward_ms": item["encoder_forward_secs"] * 1000,
                "worker_ranks": [rank],
                "rank_records": [dict(item)],
            }
            continue
        batch = batches[key]
        batch["encoder_forward_ms"] = max(
            batch["encoder_forward_ms"],
            item["encoder_forward_secs"] * 1000,
        )
        batch["worker_ranks"].append(rank)
        batch["rank_records"].append(dict(item))

    for batch in batches.values():
        batch.pop("encoder_forward_secs", None)
    return sorted(batches.values(), key=lambda item: item["batch_id"])


def collect_batch_stats(llm: Any) -> list[dict[str, Any]]:
    return aggregate_worker_batch_stats(collect_encoder_batch_stats(llm))


def _metadata_for_shape(
    metadata: dict[str, Any], shape: list[int]
) -> dict[str, Any]:
    result = dict(metadata)
    numel = math.prod(shape)
    result.update(
        {
            "shape": shape,
            "numel": numel,
            "tensor_bytes": numel * int(metadata["element_size_bytes"]),
        }
    )
    return result


def runtime_item_input_scale(
    batch: dict[str, Any],
    item_index: int,
    patch_size: int,
    spatial_merge_size: int,
) -> dict[str, Any]:
    """Slice one request item's scale from an actual runtime encoder batch."""
    grid = batch["grid_thw"]
    if item_index >= len(grid) or item_index >= len(batch["output_tensors"]):
        raise RuntimeError("Runtime encoder item metadata is incomplete.")

    item_grid = [int(dim) for dim in grid[item_index]]
    temporal, height, width = item_grid
    patch_count = temporal * height * width
    input_tensors = batch["input_tensors"]
    pixel_key = (
        "pixel_values"
        if "pixel_values" in input_tensors
        else "pixel_values_videos"
    )
    pixel_batch = input_tensors.get(pixel_key)
    if not isinstance(pixel_batch, dict) or not pixel_batch.get("shape"):
        raise RuntimeError("No runtime Pixel Tensor metadata found for the item.")
    total_patch_count = sum(math.prod(item) for item in grid)
    if int(pixel_batch["shape"][0]) != total_patch_count:
        raise RuntimeError(
            "Qwen-VL Pixel Tensor leading dimension does not match grid_thw."
        )
    pixel_item = _metadata_for_shape(
        pixel_batch, [patch_count, *pixel_batch["shape"][1:]]
    )
    output_item = dict(batch["output_tensors"][item_index])
    visual_token_count = int(output_item["shape"][0])
    processed_width = width * patch_size
    processed_height = height * patch_size
    return {
        "source": "vllm_runtime",
        "scope": "encoder_batch_item",
        "runtime_derivation": "sliced_from_encoder_batch_metadata",
        "worker_rank": int(batch["representative_worker_rank"]),
        "encoder_batch_id": int(batch["batch_id"]),
        "encoder_batch_item_index": item_index,
        "modality": batch.get("modality"),
        "num_items": 1,
        "grid_thw": [item_grid],
        "patch_size": patch_size,
        "patch_count": patch_count,
        "spatial_merge_size": spatial_merge_size,
        "visual_token_count": visual_token_count,
        "processed_width": processed_width,
        "processed_height": processed_height,
        "processed_items": [
            {
                "processed_width": processed_width,
                "processed_height": processed_height,
            }
        ],
        "pixel_values": pixel_item,
        "encoder_input_tensors": {pixel_key: pixel_item},
        "encoder_input_tensor_bytes": pixel_item["tensor_bytes"],
        "encoder_input_tensor_scope": "pixel_tensor_only",
        "encoder_output": {
            "items": [output_item],
            **output_item,
        },
    }


def request_runtime_scales(
    batches: list[dict[str, Any]],
    request_ids: list[str],
    patch_size: int,
    spatial_merge_size: int,
) -> dict[str, dict[str, Any]]:
    matches: dict[str, list[dict[str, Any]]] = {
        request_id: [] for request_id in request_ids
    }
    for batch in batches:
        for item_index, internal_id in enumerate(batch["item_request_ids"]):
            request_id = normalize_internal_request_id(internal_id)
            if request_id in matches:
                matches[request_id].append(
                    runtime_item_input_scale(
                        batch,
                        item_index,
                        patch_size,
                        spatial_merge_size,
                    )
                )

    result: dict[str, dict[str, Any]] = {}
    for request_id, scales in matches.items():
        if len(scales) != 1:
            raise RuntimeError(
                "Expected one runtime encoder item for request "
                f"{request_id}, got {len(scales)}."
            )
        result[request_id] = scales[0]
    return result


def batch_scale_estimate(
    item_estimate: dict[str, Any], num_items: int
) -> dict[str, Any]:
    """Scale the auxiliary one-image estimate to a closed encoder batch."""
    estimate = copy.deepcopy(item_estimate)
    estimate["source"] = "collect_scale_estimate_batch"
    estimate["scope"] = "encoder_batch"
    estimate["num_items"] = num_items
    estimate["grid_thw"] = estimate["grid_thw"] * num_items
    estimate["patch_count"] *= num_items
    estimate["visual_token_count"] *= num_items
    for key in ("pixel_values", "encoder_output_estimate"):
        metadata = estimate[key]
        metadata["shape"][0] *= num_items
        if "numel" in metadata:
            metadata["numel"] *= num_items
        metadata["tensor_bytes"] *= num_items
    if num_items > 1:
        estimate.pop("processed_width", None)
        estimate.pop("processed_height", None)
    return estimate


def runtime_batch_input_scale(
    batch: dict[str, Any], patch_size: int, spatial_merge_size: int
) -> dict[str, Any]:
    """Build actual batch scale and a logical combined output shape."""
    scale = runtime_input_scale_from_batch(
        batch, patch_size, spatial_merge_size
    )
    scale["scope"] = "encoder_batch"
    outputs = scale["encoder_output"]["items"]
    if outputs and all(
        item["shape"][1:] == outputs[0]["shape"][1:] for item in outputs
    ):
        scale["encoder_output"].update(
            {
                "shape": [
                    sum(int(item["shape"][0]) for item in outputs),
                    *outputs[0]["shape"][1:],
                ],
                "dtype": outputs[0]["dtype"],
                "shape_source": "combined_from_runtime_items",
            }
        )
    return scale


def summarize_requests(requests: list[dict[str, Any]]) -> dict[str, float]:
    ttft = [float(item["ttft_ms"]) for item in requests]
    tpot = [float(item["tpot_ms"]) for item in requests]
    return {
        "ttft_p50_ms": statistics.median(ttft),
        "ttft_p90_ms": percentile(ttft, 0.9),
        "ttft_p99_ms": percentile(ttft, 0.99),
        "tpot_p50_ms": statistics.median(tpot),
        "tpot_p90_ms": percentile(tpot, 0.9),
        "tpot_p99_ms": percentile(tpot, 0.99),
    }


def timing_stats_ms(stats: dict[str, Any]) -> dict[str, Any]:
    converted = {}
    for key, value in stats.items():
        if key.endswith("_secs"):
            converted[key.removesuffix("_secs") + "_ms"] = value * 1000
        else:
            converted[key] = value
    return converted


def main() -> None:
    args = parse_args()
    from vllm import LLM, SamplingParams
    from vllm.benchmarks.mm_processor import get_timing_stats_from_engine

    concurrency_levels = [int(item) for item in args.concurrency.split(",")]
    base_image, media_metrics = load_image_with_metrics(args.image)
    mm_processor_kwargs = {
        key: value
        for key, value in {
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        }.items()
        if value is not None
    }
    scale_estimate = collect_scale(
        SimpleNamespace(
            model=args.model,
            image=args.image,
            prompt=args.prompt,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            trust_remote_code=args.trust_remote_code,
            model_dtype_bytes=2,
        )
    )
    engine_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": max(concurrency_levels),
        "trust_remote_code": args.trust_remote_code,
        "mm_processor_kwargs": mm_processor_kwargs,
        "mm_processor_cache_gb": 0,
        "enable_prefix_caching": False,
        "enable_mm_processor_stats": True,
        "limit_mm_per_prompt": {"image": 1},
    }
    if args.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(**engine_kwargs)
    sampling_params = SamplingParams(
        temperature=0.0,
        min_tokens=args.output_tokens,
        max_tokens=args.output_tokens,
        ignore_eos=True,
    )
    prompt = qwen_prompt(args.prompt)
    vision_config = llm.llm_engine.vllm_config.model_config.hf_config.vision_config
    patch_size = int(vision_config.patch_size)
    spatial_merge_size = int(vision_config.spatial_merge_size)

    for warmup in range(args.warmups):
        llm.reset_mm_cache()
        llm.llm_engine.reset_encoder_cache()
        llm.generate(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": unique_image(base_image, warmup + 1)
                    },
                }
            ],
            sampling_params,
            use_tqdm=False,
        )
    get_timing_stats_from_engine(llm.llm_engine)
    collect_batch_stats(llm)

    records: list[dict[str, Any]] = []
    image_index = args.warmups + 1
    for concurrency in concurrency_levels:
        for repeat in range(args.repeats):
            llm.reset_mm_cache()
            llm.llm_engine.reset_encoder_cache()
            requests = []
            for _ in range(concurrency):
                requests.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {
                            "image": unique_image(base_image, image_index)
                        },
                    }
                )
                image_index += 1

            wall_start = time.perf_counter()
            outputs = llm.generate(
                requests, sampling_params, use_tqdm=False
            )
            wall_s = time.perf_counter() - wall_start
            all_stage_stats = get_timing_stats_from_engine(llm.llm_engine)
            batch_stats = collect_batch_stats(llm)
            request_metrics = [request_output_metrics(output) for output in outputs]
            request_ids = [item["request_id"] for item in request_metrics]
            stage_stats = merge_request_stage_stats(
                all_stage_stats, request_ids
            )
            runtime_scales = request_runtime_scales(
                batch_stats,
                request_ids,
                patch_size,
                spatial_merge_size,
            )

            for metrics in request_metrics:
                if metrics["output_token_count"] != args.output_tokens:
                    raise RuntimeError("Output length isolation failed.")
                stats = stage_stats.get(metrics["request_id"], {})
                if stats.get("num_encoder_calls") != 1:
                    raise RuntimeError(
                        "Encoder cache isolation failed for request "
                        f"{metrics['request_id']}."
                    )
                input_scale = runtime_scales[metrics["request_id"]]
                records.append(
                    {
                        "record_type": "request",
                        "experiment": "concurrency",
                        "model": args.model,
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "media": media_metrics,
                        "input_scale": input_scale,
                        "input_scale_estimate": scale_estimate,
                        "input_scale_comparison": compare_input_scales(
                            input_scale, scale_estimate
                        ),
                        "request": metrics,
                        "stages": timing_stats_ms(stats),
                    }
                )

            records.append(
                {
                    "record_type": "summary",
                    "experiment": "concurrency",
                    "model": args.model,
                    "concurrency": concurrency,
                    "repeat": repeat,
                    "wall_e2e_ms": wall_s * 1000,
                    "request_throughput_per_s": concurrency / wall_s,
                    "output_token_throughput_per_s": (
                        concurrency * args.output_tokens / wall_s
                    ),
                    **summarize_requests(request_metrics),
                }
            )
            for batch in batch_stats:
                input_scale = runtime_batch_input_scale(
                    batch, patch_size, spatial_merge_size
                )
                input_scale_estimate = batch_scale_estimate(
                    scale_estimate, int(batch["num_items"])
                )
                records.append(
                    {
                        "record_type": "encoder_batch",
                        "experiment": "concurrency",
                        "model": args.model,
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "input_scale": input_scale,
                        "input_scale_estimate": input_scale_estimate,
                        "input_scale_comparison": compare_input_scales(
                            input_scale, input_scale_estimate
                        ),
                        **batch,
                    }
                )

    prefix = Path(args.output_prefix)
    write_jsonl(prefix.with_suffix(".jsonl"), records)
    write_csv(prefix.with_suffix(".csv"), records)
    print(json.dumps(records[-1], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
