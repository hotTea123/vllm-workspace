"""Collect actual Qwen2.5-VL processor shapes for one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.multimodal_cpu.common import (
    grid_metrics,
    load_image_with_metrics,
    tensor_metadata,
    write_csv,
    write_jsonl,
)


def _get_attr(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect real processor dimensions and visual-token counts."
    )
    parser.add_argument("--model", required=True, help="Model name or local path.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--prompt", default="Describe this image.", help="Text paired with the image."
    )
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--model-dtype-bytes",
        type=int,
        default=2,
        help="Bytes per element used to estimate device-side encoder output.",
    )
    parser.add_argument("--output-prefix", default="results/input_scale")
    return parser.parse_args()


def collect(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoConfig, AutoProcessor

    image, image_metrics = load_image_with_metrics(args.image)
    processor_kwargs = {"trust_remote_code": args.trust_remote_code}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels

    processor = AutoProcessor.from_pretrained(args.model, **processor_kwargs)
    config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    )

    image_processor = processor.image_processor
    vision_config = _get_attr(config, "vision_config", {})
    merge_size = int(
        _get_attr(
            image_processor,
            "merge_size",
            _get_attr(vision_config, "spatial_merge_size", 1),
        )
    )
    patch_size = int(
        _get_attr(
            image_processor,
            "patch_size",
            _get_attr(vision_config, "patch_size", 1),
        )
    )

    grid = grid_metrics(inputs["image_grid_thw"], merge_size)
    first_grid = grid["grid_thw"][0]
    out_hidden_size = int(
        _get_attr(
            vision_config,
            "out_hidden_size",
            _get_attr(config, "hidden_size", 0),
        )
    )

    record = {
        "source": "collect_scale_estimate",
        "model": args.model,
        "prompt": args.prompt,
        **image_metrics,
        "processed_width": first_grid[2] * patch_size,
        "processed_height": first_grid[1] * patch_size,
        "patch_size": patch_size,
        **grid,
        "pixel_values": tensor_metadata(inputs["pixel_values"]),
        "encoder_output_estimate": {
            "shape": [grid["visual_token_count"], out_hidden_size],
            "element_size_bytes": args.model_dtype_bytes,
            "tensor_bytes": (
                grid["visual_token_count"]
                * out_hidden_size
                * args.model_dtype_bytes
            ),
        },
    }
    return record


def main() -> None:
    args = parse_args()
    record = collect(args)
    prefix = Path(args.output_prefix)
    write_jsonl(prefix.with_suffix(".jsonl"), [record])
    write_csv(prefix.with_suffix(".csv"), [record])
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
