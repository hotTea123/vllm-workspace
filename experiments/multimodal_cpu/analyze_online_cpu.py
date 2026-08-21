"""Summarize online request, host CPU, and encoder timing records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.multimodal_cpu.analyze_npu_baseline import metric_stats
from experiments.multimodal_cpu.common import normalize_internal_request_id


ENCODER_LOG_MARKER = "MM_ENCODER_TIMING "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze one online multimodal CPU measurement."
    )
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--host-cpu", required=True, type=Path)
    parser.add_argument("--server-log", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _load_encoder_events(path: Path) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            _, marker, payload = line.partition(ENCODER_LOG_MARKER)
            if not marker:
                continue
            event, _ = decoder.raw_decode(payload.lstrip())
            if event.get("event") == "mm_encoder_timing":
                events.append(event)
    return events


def _load_cpu_samples(
    path: Path, start_epoch_ms: int, end_epoch_ms: int
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        samples = [
            row
            for row in csv.DictReader(source)
            if start_epoch_ms
            <= int(row["timestamp_epoch_ms"])
            <= end_epoch_ms
        ]
    if not samples:
        raise ValueError("No host CPU samples fall inside the measured request window.")
    return samples


def _encoder_ms_by_request(
    requests: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    measured_ids = {str(record["response_id"]) for record in requests}
    by_request: defaultdict[str, list[float]] = defaultdict(list)
    for event in events:
        if int(event["num_requests"]) != 1:
            continue
        for request_id in event["request_ids"]:
            normalized = normalize_internal_request_id(str(request_id))
            if normalized in measured_ids:
                by_request[normalized].append(float(event["encoder_forward_ms"]))

    missing = measured_ids - by_request.keys()
    if missing:
        raise ValueError(
            "Encoder cache isolation failed; no encoder timing event for: "
            + ", ".join(sorted(missing))
        )
    return {
        request_id: max(rank_values)
        for request_id, rank_values in by_request.items()
    }


def _validate_requests(records: Sequence[Mapping[str, Any]]) -> None:
    if any(record["experiment"] != "online_cpu" for record in records):
        raise ValueError("The request file is not an online_cpu result.")
    if any(record["phase"] != "measure" for record in records):
        raise ValueError("The request file contains non-measure records.")
    if any(not record.get("response_id") for record in records):
        raise ValueError("A request is missing its server response ID.")
    media_uuids = [str(record["media_uuid"]) for record in records]
    if len(set(media_uuids)) != len(media_uuids):
        raise ValueError("Media UUIDs are not unique across measured requests.")
    for record in records:
        if record["output_token_count"] != record["expected_output_tokens"]:
            raise ValueError(
                f"Output length mismatch in repeat {record['repeat']}."
            )


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _metric_row(label: str, values: Sequence[float], digits: int) -> tuple[str, ...]:
    mean, p50, p90, cv_percent = metric_stats(values)
    return (
        label,
        f"{mean:.{digits}f}",
        f"{p50:.{digits}f}",
        f"{p90:.{digits}f}",
        f"{max(values):.{digits}f}",
        f"{cv_percent:.2f}%",
    )


def render_report(
    requests: Sequence[Mapping[str, Any]],
    cpu_samples: Sequence[Mapping[str, str]],
    encoder_ms: Mapping[str, float],
) -> str:
    start_epoch_ms = min(int(row["request_start_epoch_ms"]) for row in requests)
    end_epoch_ms = max(int(row["request_end_epoch_ms"]) for row in requests)
    metrics = (
        _metric_row("TTFT (ms)", [float(row["ttft_ms"]) for row in requests], 2),
        _metric_row("TPOT (ms)", [float(row["tpot_ms"]) for row in requests], 3),
        _metric_row(
            "Encoder Forward (ms)",
            [encoder_ms[str(row["response_id"])] for row in requests],
            2,
        ),
        _metric_row(
            "Container CPU (%)",
            [float(row["cpu_percent"]) for row in cpu_samples],
            2,
        ),
        _metric_row(
            "Effective CPU cores",
            [float(row["effective_cores"]) for row in cpu_samples],
            2,
        ),
    )
    overview = _table(
        ("项目", "数值"),
        (
            ("正式请求数", str(len(requests))),
            ("唯一Media UUID数", str(len({row["media_uuid"] for row in requests}))),
            ("匹配Encoder记录的请求数", str(len(encoder_ms))),
            ("CPU窗口样本数", str(len(cpu_samples))),
            ("正式窗口时长", f"{(end_epoch_ms - start_epoch_ms) / 1000:.3f} s"),
        ),
    )
    metric_table = _table(
        ("指标", "平均", "P50", "P90", "最大", "波动率"), metrics
    )
    return "\n\n".join(("## 在线CPU实验概览", overview, "## 指标统计", metric_table))


def analyze(
    request_path: Path, cpu_path: Path, server_log_path: Path
) -> str:
    requests = _load_jsonl(request_path)
    _validate_requests(requests)
    start_epoch_ms = min(int(row["request_start_epoch_ms"]) for row in requests)
    end_epoch_ms = max(int(row["request_end_epoch_ms"]) for row in requests)
    cpu_samples = _load_cpu_samples(cpu_path, start_epoch_ms, end_epoch_ms)
    encoder_ms = _encoder_ms_by_request(
        requests, _load_encoder_events(server_log_path)
    )
    return render_report(requests, cpu_samples, encoder_ms)


def main() -> None:
    args = parse_args()
    report = analyze(args.requests, args.host_cpu, args.server_log)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
