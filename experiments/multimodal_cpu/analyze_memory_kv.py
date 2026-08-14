"""Calculate Qwen2.5-VL vision weight bytes and equivalent KV capacity."""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.multimodal_cpu.common import write_csv, write_jsonl


SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

KV_DTYPE_BYTES = {"fp8": 1, "float16": 2, "bfloat16": 2, "float32": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure vision checkpoint bytes and calculate KV capacity."
    )
    parser.add_argument("--model", required=True, help="Local model directory.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--kv-dtype", choices=tuple(KV_DTYPE_BYTES), default="bfloat16"
    )
    parser.add_argument(
        "--vision-prefixes",
        default="model.visual.,visual.",
        help="Comma-separated checkpoint prefixes counted as vision weights.",
    )
    parser.add_argument(
        "--tokens-per-request",
        type=int,
        default=8192,
        help="Used only to convert released bytes to equivalent request slots.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-prefix", default="results/memory_kv")
    return parser.parse_args()


def tensor_storage_bytes(shape: list[int], dtype: str) -> int:
    try:
        element_size = SAFETENSORS_DTYPE_BYTES[dtype]
    except KeyError as error:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}") from error
    return math.prod(shape) * element_size


def vision_component(key: str) -> str:
    tail = key.split("visual.", 1)[-1]
    return tail.split(".", 1)[0]


def inspect_vision_checkpoint(
    model_dir: str | Path, prefixes: tuple[str, ...]
) -> dict[str, Any]:
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise ValueError(
            "No .safetensors files found. Pass the local model directory; "
            "PyTorch .bin checkpoints are not read by this probe."
        )

    total_bytes = 0
    total_parameters = 0
    tensor_count = 0
    components: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensor_count": 0, "parameter_count": 0, "storage_bytes": 0}
    )
    for checkpoint in files:
        with checkpoint.open("rb") as source:
            header_size = struct.unpack("<Q", source.read(8))[0]
            header = json.loads(source.read(header_size))
        for key, tensor_info in header.items():
            if key == "__metadata__" or not key.startswith(prefixes):
                continue
            shape = [int(dim) for dim in tensor_info["shape"]]
            dtype = tensor_info["dtype"]
            parameters = math.prod(shape)
            storage_bytes = tensor_storage_bytes(shape, dtype)
            component = vision_component(key)

            total_bytes += storage_bytes
            total_parameters += parameters
            tensor_count += 1
            components[component]["tensor_count"] += 1
            components[component]["parameter_count"] += parameters
            components[component]["storage_bytes"] += storage_bytes

    if not tensor_count:
        raise ValueError(
            f"No vision tensors matched prefixes {prefixes}. Inspect the "
            "checkpoint names and pass --vision-prefixes."
        )
    return {
        "tensor_count": tensor_count,
        "parameter_count": total_parameters,
        "checkpoint_storage_bytes": total_bytes,
        "components": dict(components),
    }


def kv_heads_per_rank(num_kv_heads: int, tensor_parallel_size: int) -> int:
    if num_kv_heads >= tensor_parallel_size:
        if num_kv_heads % tensor_parallel_size:
            raise ValueError("KV heads must be divisible by tensor parallel size.")
        return num_kv_heads // tensor_parallel_size
    if tensor_parallel_size % num_kv_heads:
        raise ValueError(
            "Tensor parallel size must be divisible by KV heads when KV heads "
            "are replicated."
        )
    return 1


def calculate_kv_bytes_per_token(
    text_config: Any, tensor_parallel_size: int, element_size: int
) -> dict[str, int]:
    num_layers = int(text_config.num_hidden_layers)
    num_attention_heads = int(text_config.num_attention_heads)
    num_kv_heads = int(
        getattr(text_config, "num_key_value_heads", num_attention_heads)
    )
    head_dim = int(
        getattr(
            text_config,
            "head_dim",
            int(text_config.hidden_size) // num_attention_heads,
        )
    )
    per_rank_heads = kv_heads_per_rank(num_kv_heads, tensor_parallel_size)
    per_npu = 2 * num_layers * per_rank_heads * head_dim * element_size
    return {
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_kv_heads,
        "kv_heads_per_npu": per_rank_heads,
        "head_dim": head_dim,
        "kv_bytes_per_token_per_npu": per_npu,
        "kv_bytes_per_token_all_npus": per_npu * tensor_parallel_size,
    }


def main() -> None:
    args = parse_args()
    from transformers import AutoConfig

    prefixes = tuple(
        item.strip() for item in args.vision_prefixes.split(",") if item.strip()
    )
    checkpoint = inspect_vision_checkpoint(args.model, prefixes)
    config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    text_config = getattr(config, "text_config", config)
    kv = calculate_kv_bytes_per_token(
        text_config,
        args.tensor_parallel_size,
        KV_DTYPE_BYTES[args.kv_dtype],
    )

    vision_bytes = checkpoint["checkpoint_storage_bytes"]
    lower_released_per_npu = math.ceil(
        vision_bytes / args.tensor_parallel_size
    )
    upper_released_per_npu = vision_bytes
    kv_bytes_per_token = kv["kv_bytes_per_token_per_npu"]
    lower_tokens = lower_released_per_npu // kv_bytes_per_token
    upper_tokens = upper_released_per_npu // kv_bytes_per_token
    record = {
        "experiment": "memory_kv",
        "model": str(Path(args.model).resolve()),
        "tensor_parallel_size": args.tensor_parallel_size,
        "kv_dtype": args.kv_dtype,
        "vision_checkpoint": checkpoint,
        "kv": kv,
        "released_weight_bytes_per_npu_range": [
            lower_released_per_npu,
            upper_released_per_npu,
        ],
        "equivalent_kv_tokens_per_npu_range": [lower_tokens, upper_tokens],
        "equivalent_request_slots_range": [
            lower_tokens / args.tokens_per_request,
            upper_tokens / args.tokens_per_request,
        ],
        "assumption": (
            "The lower bound assumes all vision weights are evenly TP-sharded; "
            "the upper bound assumes they are replicated. Runtime allocation, "
            "alignment, workspaces, and activation peaks are not included."
        ),
    }

    prefix = Path(args.output_prefix)
    write_jsonl(prefix.with_suffix(".jsonl"), [record])
    write_csv(prefix.with_suffix(".csv"), [record])
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
