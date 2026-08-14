"""Capture one Qwen2.5-VL vision-forward trace on the target NPU."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from experiments.multimodal_cpu.collect_input_scale import collect as collect_scale
from experiments.multimodal_cpu.common import (
    load_image_with_metrics,
    write_csv,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Qwen2.5-VL ViT stages.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--profile-dir", required=True)
    return parser.parse_args()


def qwen_prompt(question: str) -> str:
    placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{placeholder}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    args = parse_args()
    profile_dir = Path(args.profile_dir).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)

    # This existing vLLM switch makes record_function_or_nullcontext emit scopes.
    os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"

    from vllm import LLM, SamplingParams

    image, media_metrics = load_image_with_metrics(args.image)
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
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        mm_processor_kwargs=mm_processor_kwargs,
        mm_processor_cache_gb=0,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": str(profile_dir),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_memory": True,
            "torch_profiler_use_gzip": False,
        },
    )
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=1, ignore_eos=True
    )
    request = {
        "prompt": qwen_prompt(args.prompt),
        "multi_modal_data": {"image": image},
    }

    for _ in range(args.warmups):
        llm.reset_mm_cache()
        llm.llm_engine.reset_encoder_cache()
        llm.generate([request], sampling_params, use_tqdm=False)

    llm.reset_mm_cache()
    llm.llm_engine.reset_encoder_cache()
    llm.start_profile()
    llm.generate([request], sampling_params, use_tqdm=False)
    llm.stop_profile()
    time.sleep(2)

    manifest = {
        "experiment": "vit_layer_profile",
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "profile_dir": str(profile_dir),
        "media": media_metrics,
        "input_scale": scale,
        "scope_prefix": "mm.vit.",
    }
    write_jsonl(profile_dir / "manifest.jsonl", [manifest])
    write_csv(profile_dir / "manifest.csv", [manifest])
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
