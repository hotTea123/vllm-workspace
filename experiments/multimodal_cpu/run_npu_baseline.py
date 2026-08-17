"""Run a cache-miss, single-request multimodal NPU baseline."""

from __future__ import annotations

import argparse
import json
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
        description="Measure single-request Qwen2.5-VL NPU latency."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-prefix", default="results/npu_baseline")
    return parser.parse_args()


def qwen_prompt(question: str) -> str:
    """Build the Qwen2.5-VL chat prompt used by the vLLM example."""
    placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{placeholder}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def unique_image(image: Any, sample_index: int) -> Any:
    """Change one pixel so every measured request has a distinct media hash."""
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


def _stage_stats_for_output(llm: Any, request_id: str) -> dict[str, Any]:
    from vllm.benchmarks.mm_processor import get_timing_stats_from_engine

    all_stats = get_timing_stats_from_engine(llm.llm_engine)
    stats = merge_request_stage_stats(all_stats, [request_id])[request_id]
    if not stats:
        raise RuntimeError(
            "No multimodal timing stats found for request "
            f"{request_id}; available IDs: {list(all_stats)}."
        )

    converted: dict[str, Any] = {}
    for key, value in stats.items():
        if key.endswith("_secs"):
            converted[key.removesuffix("_secs") + "_ms"] = value * 1000
        else:
            converted[key] = value
    return converted


def _runtime_batch_for_output(
    batch_stats: list[dict[str, Any]], request_id: str
) -> dict[str, Any]:
    matches = [
        batch
        for batch in batch_stats
        if request_id
        in {
            normalize_internal_request_id(internal_id)
            for internal_id in batch["request_ids"]
        }
    ]
    if not matches:
        raise RuntimeError(
            f"No runtime encoder batch found for request {request_id}."
        )
    first_rank = min(int(batch["worker_rank"]) for batch in matches)
    first_rank_matches = [
        batch for batch in matches if int(batch["worker_rank"]) == first_rank
    ]
    if len(first_rank_matches) != 1:
        raise RuntimeError(
            "Expected one encoder batch for the single-image request, got "
            f"{len(first_rank_matches)} on worker rank {first_rank}."
        )
    return first_rank_matches[0]


def _reset_request_caches(llm: Any) -> None:
    llm.reset_mm_cache()
    llm.llm_engine.reset_encoder_cache()


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

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

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        mm_processor_kwargs=mm_processor_kwargs,
        mm_processor_cache_gb=0,
        enable_prefix_caching=False,
        enable_mm_processor_stats=True,
        limit_mm_per_prompt={"image": 1},
    )
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

    for warmup_index in range(args.warmups):
        _reset_request_caches(llm)
        llm.generate(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": unique_image(base_image, warmup_index + 1)
                    },
                }
            ],
            sampling_params,
            use_tqdm=False,
        )

    # Clear warmup timing records without depending on warmup request IDs.
    from vllm.benchmarks.mm_processor import get_timing_stats_from_engine

    get_timing_stats_from_engine(llm.llm_engine)
    collect_encoder_batch_stats(llm)

    records: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        _reset_request_caches(llm)
        request_image = unique_image(base_image, args.warmups + repeat + 1)
        wall_start = time.perf_counter()
        outputs = llm.generate(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": request_image},
                }
            ],
            sampling_params,
            use_tqdm=False,
        )
        wall_ms = (time.perf_counter() - wall_start) * 1000
        output = outputs[0]
        output_metrics = request_output_metrics(output)
        if output_metrics["output_token_count"] != args.output_tokens:
            raise RuntimeError(
                "Output length isolation failed: expected "
                f"{args.output_tokens}, got {output_metrics['output_token_count']}."
            )

        stage_metrics = _stage_stats_for_output(llm, output.request_id)
        encoder_batches = collect_encoder_batch_stats(llm)
        runtime_batch = _runtime_batch_for_output(
            encoder_batches, output.request_id
        )
        input_scale = runtime_input_scale_from_batch(
            runtime_batch, patch_size, spatial_merge_size
        )
        input_scale_comparison = compare_input_scales(
            input_scale, scale_estimate
        )
        if stage_metrics.get("num_encoder_calls") != 1:
            raise RuntimeError(
                "Encoder cache isolation failed: expected one encoder call, got "
                f"{stage_metrics.get('num_encoder_calls')}."
            )

        records.append(
            {
                "experiment": "npu_baseline",
                "model": args.model,
                "repeat": repeat,
                "tensor_parallel_size": args.tensor_parallel_size,
                "expected_output_tokens": args.output_tokens,
                "wall_e2e_ms": wall_ms,
                "media": media_metrics,
                "input_scale": input_scale,
                "input_scale_estimate": scale_estimate,
                "input_scale_comparison": input_scale_comparison,
                "encoder_batches": encoder_batches,
                "request": output_metrics,
                "stages": stage_metrics,
            }
        )

    return records


def main() -> None:
    args = parse_args()
    records = run(args)
    prefix = Path(args.output_prefix)
    write_jsonl(prefix.with_suffix(".jsonl"), records)
    write_csv(prefix.with_suffix(".csv"), records)
    print(json.dumps(records[-1], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
