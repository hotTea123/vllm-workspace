"""Send cache-isolated streaming requests to an online vLLM server."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure streaming TTFT and TPOT against vLLM's Chat API."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--phase", choices=("warmup", "measure"), required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY"))
    return parser.parse_args()


def _image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def _delta_has_token(delta: dict[str, Any]) -> bool:
    return any(
        value not in (None, "", [], {})
        for value in (
            delta.get("content"),
            delta.get("reasoning_content"),
            delta.get("tool_calls"),
        )
    )


def consume_sse(
    lines: Iterable[bytes],
    now_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Consume one OpenAI-compatible SSE response and timestamp token chunks."""
    first_token_ns: int | None = None
    last_token_ns: int | None = None
    response_id: str | None = None
    output_token_count: int | None = None

    for raw_line in lines:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        event_ns = now_ns()
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break

        event = json.loads(data)
        if error := event.get("error"):
            raise RuntimeError(f"Server returned an error: {error}")
        if event_id := event.get("id"):
            if response_id is not None and response_id != event_id:
                raise RuntimeError("Response ID changed within one SSE stream.")
            response_id = str(event_id)

        usage = event.get("usage")
        if usage and usage.get("completion_tokens") is not None:
            output_token_count = int(usage["completion_tokens"])

        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if _delta_has_token(delta):
                if first_token_ns is None:
                    first_token_ns = event_ns
                last_token_ns = event_ns

    return {
        "response_id": response_id,
        "output_token_count": output_token_count,
        "first_token_ns": first_token_ns,
        "last_token_ns": last_token_ns,
        "stream_end_ns": now_ns(),
    }


def _request_payload(
    model: str,
    prompt: str,
    image_data_url: str,
    media_uuid: str,
    output_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                        "uuid": media_uuid,
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": 0.0,
        "min_tokens": output_tokens,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def send_request(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    image_data_url: str,
    phase: str,
    run_id: str,
    index: int,
    output_tokens: int,
    timeout: float,
    api_key: str | None,
) -> dict[str, Any]:
    request_name = f"mmcpu-{phase}-{run_id}-{index:04d}"
    payload = _request_payload(
        model, prompt, image_data_url, request_name, output_tokens
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Request-Id": request_name,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start_epoch_ms = time.time_ns() // 1_000_000
    start_ns = time.perf_counter_ns()
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            stream = consume_sse(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error

    first_token_ns = stream["first_token_ns"]
    last_token_ns = stream["last_token_ns"]
    end_ns = stream["stream_end_ns"]
    actual_tokens = stream["output_token_count"]
    if first_token_ns is None or last_token_ns is None:
        raise RuntimeError(f"No generated token found for {request_name}.")
    if stream["response_id"] is None:
        raise RuntimeError(f"No response ID found for {request_name}.")
    if actual_tokens is None:
        raise RuntimeError(
            "No completion token count found; the server must support "
            "stream_options.include_usage."
        )
    if actual_tokens != output_tokens:
        raise RuntimeError(
            f"Output length isolation failed for {request_name}: "
            f"expected {output_tokens}, got {actual_tokens}."
        )

    elapsed_before_first_ns = first_token_ns - start_ns
    elapsed_before_last_ns = last_token_ns - start_ns
    return {
        "experiment": "online_cpu",
        "phase": phase,
        "repeat": index,
        "request_id_header": request_name,
        "response_id": stream["response_id"],
        "media_uuid": request_name,
        "expected_output_tokens": output_tokens,
        "output_token_count": actual_tokens,
        "request_start_epoch_ms": start_epoch_ms,
        "first_token_epoch_ms": start_epoch_ms + elapsed_before_first_ns // 1_000_000,
        "last_token_epoch_ms": start_epoch_ms + elapsed_before_last_ns // 1_000_000,
        "request_end_epoch_ms": start_epoch_ms + (end_ns - start_ns) // 1_000_000,
        "ttft_ms": elapsed_before_first_ns / 1_000_000,
        "tpot_ms": (last_token_ns - first_token_ns)
        / max(actual_tokens - 1, 1)
        / 1_000_000,
    }


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.requests <= 0:
        raise ValueError("--requests must be greater than zero.")
    if args.output_tokens <= 1:
        raise ValueError("--output-tokens must be greater than one for TPOT.")
    if not args.image.is_file():
        raise ValueError(f"Image does not exist: {args.image}")
    if args.phase == "measure" and args.output is None:
        raise ValueError("--output is required for the measure phase.")
    if args.output is not None and args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")

    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    image_data_url = _image_data_url(args.image)
    run_id = args.run_id or uuid.uuid4().hex
    records: list[dict[str, Any]] = []

    output_file = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_file = args.output.open("x", encoding="utf-8")
    try:
        for index in range(args.requests):
            record = send_request(
                endpoint=endpoint,
                model=args.model,
                prompt=args.prompt,
                image_data_url=image_data_url,
                phase=args.phase,
                run_id=run_id,
                index=index,
                output_tokens=args.output_tokens,
                timeout=args.timeout,
                api_key=args.api_key,
            )
            records.append(record)
            if output_file is not None:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
            print(
                f"[{args.phase} {index + 1}/{args.requests}] "
                f"TTFT={record['ttft_ms']:.2f} ms "
                f"TPOT={record['tpot_ms']:.3f} ms"
            )
    finally:
        if output_file is not None:
            output_file.close()

    return records


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
