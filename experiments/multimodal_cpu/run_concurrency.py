"""Run closed-batch multimodal concurrency sweeps with encoder batch stats."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiments.multimodal_cpu.collect_input_scale import collect as collect_scale
from experiments.multimodal_cpu.common import (
    load_image_with_metrics,
    request_output_metrics,
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
    worker_stats: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge identical TP-rank batches using the slowest synchronized rank."""
    batches: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}
    for rank, rank_stats in enumerate(worker_stats):
        for item in rank_stats:
            request_ids = tuple(item["request_ids"])
            key = (int(item["batch_id"]), request_ids)
            if key not in batches:
                batches[key] = {
                    **item,
                    "encoder_forward_ms": item["encoder_forward_secs"] * 1000,
                    "worker_ranks": [rank],
                }
                continue
            batch = batches[key]
            batch["encoder_forward_ms"] = max(
                batch["encoder_forward_ms"],
                item["encoder_forward_secs"] * 1000,
            )
            batch["worker_ranks"].append(rank)

    for batch in batches.values():
        batch.pop("encoder_forward_secs", None)
    return sorted(batches.values(), key=lambda item: item["batch_id"])


def collect_batch_stats(llm: Any) -> list[dict[str, Any]]:
    worker_stats = llm.llm_engine.collective_rpc(
        "get_encoder_batch_timing_stats"
    )
    return aggregate_worker_batch_stats(worker_stats)


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
    scale = collect_scale(
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
            stage_stats = get_timing_stats_from_engine(llm.llm_engine)
            batch_stats = collect_batch_stats(llm)
            request_metrics = [request_output_metrics(output) for output in outputs]

            for metrics in request_metrics:
                if metrics["output_token_count"] != args.output_tokens:
                    raise RuntimeError("Output length isolation failed.")
                stats = stage_stats.get(metrics["request_id"], {})
                if stats.get("num_encoder_calls") != 1:
                    raise RuntimeError(
                        "Encoder cache isolation failed for request "
                        f"{metrics['request_id']}."
                    )
                records.append(
                    {
                        "record_type": "request",
                        "experiment": "concurrency",
                        "model": args.model,
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "media": media_metrics,
                        "input_scale": scale,
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
            records.extend(
                {
                    "record_type": "encoder_batch",
                    "experiment": "concurrency",
                    "model": args.model,
                    "concurrency": concurrency,
                    "repeat": repeat,
                    "input_patch_count_per_request": scale["patch_count"],
                    "batch_patch_count_estimate": (
                        scale["patch_count"] * batch["num_items"]
                    ),
                    **batch,
                }
                for batch in batch_stats
            )

    prefix = Path(args.output_prefix)
    write_jsonl(prefix.with_suffix(".jsonl"), records)
    write_csv(prefix.with_suffix(".csv"), records)
    print(json.dumps(records[-1], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
