"""Unit tests for the online CPU measurement path."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.multimodal_cpu.analyze_online_cpu import analyze
from experiments.multimodal_cpu.run_online_cpu import (
    _request_payload,
    consume_sse,
)


class OnlineCpuClientTest(unittest.TestCase):
    def test_consume_sse_uses_usage_and_content_timestamps(self) -> None:
        response_id = "chatcmpl-request-1"
        events = (
            {"id": response_id, "choices": [{"delta": {"role": "assistant"}}]},
            {"id": response_id, "choices": [{"delta": {"content": "A"}}]},
            {"id": response_id, "choices": [{"delta": {"content": "B"}}]},
            {"id": response_id, "choices": [], "usage": {"completion_tokens": 2}},
        )
        lines = [f"data: {json.dumps(event)}\n".encode() for event in events]
        lines.append(b"data: [DONE]\n")
        timestamps = iter((100, 200, 300, 400, 500, 600))

        actual = consume_sse(lines, lambda: next(timestamps))

        self.assertEqual(actual["response_id"], response_id)
        self.assertEqual(actual["output_token_count"], 2)
        self.assertEqual(actual["first_token_ns"], 200)
        self.assertEqual(actual["last_token_ns"], 300)
        self.assertEqual(actual["stream_end_ns"], 600)

    def test_request_payload_uses_media_uuid_without_changing_image(self) -> None:
        payload = _request_payload(
            "model", "prompt", "data:image/jpeg;base64,abc", "unique-media", 100
        )
        image_part = payload["messages"][1]["content"][0]

        self.assertEqual(image_part["uuid"], "unique-media")
        self.assertEqual(
            image_part["image_url"]["url"], "data:image/jpeg;base64,abc"
        )
        self.assertTrue(payload["ignore_eos"])
        self.assertEqual(payload["min_tokens"], 100)
        self.assertEqual(payload["max_tokens"], 100)


class OnlineCpuAnalysisTest(unittest.TestCase):
    def test_analysis_filters_cpu_window_and_takes_slowest_rank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            cpu_path = root / "cpu.csv"
            server_log_path = root / "server.log"
            response_ids = (
                "chatcmpl-mmcpu-measure-run-0000",
                "chatcmpl-mmcpu-measure-run-0001",
            )
            request_records = [
                {
                    "experiment": "online_cpu",
                    "phase": "measure",
                    "repeat": index,
                    "response_id": response_id,
                    "media_uuid": f"media-{index}",
                    "expected_output_tokens": 100,
                    "output_token_count": 100,
                    "request_start_epoch_ms": 1000 + index * 1000,
                    "request_end_epoch_ms": 2000 + index * 1000,
                    "ttft_ms": 100.0 + index,
                    "tpot_ms": 20.0 + index,
                }
                for index, response_id in enumerate(response_ids)
            ]
            requests_path.write_text(
                "".join(json.dumps(row) + "\n" for row in request_records),
                encoding="utf-8",
            )

            with cpu_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=(
                        "timestamp_epoch_ms",
                        "timestamp_iso",
                        "container",
                        "cpu_percent",
                        "effective_cores",
                        "memory_usage",
                        "pids",
                    ),
                )
                writer.writeheader()
                for timestamp in (500, 1500, 2500, 3500):
                    writer.writerow(
                        {
                            "timestamp_epoch_ms": timestamp,
                            "timestamp_iso": "time",
                            "container": "container",
                            "cpu_percent": "200",
                            "effective_cores": "2",
                            "memory_usage": "1GiB / 2GiB",
                            "pids": "10",
                        }
                    )

            encoder_events = (
                {
                    "event": "mm_encoder_timing",
                    "worker_rank": 0,
                    "request_ids": [response_ids[0] + "-deadbeef"],
                    "num_requests": 1,
                    "encoder_forward_ms": 10.0,
                },
                {
                    "event": "mm_encoder_timing",
                    "worker_rank": 1,
                    "request_ids": [response_ids[0] + "-deadbeef"],
                    "num_requests": 1,
                    "encoder_forward_ms": 12.0,
                },
                {
                    "event": "mm_encoder_timing",
                    "worker_rank": 0,
                    "request_ids": [response_ids[1] + "-cafebabe"],
                    "num_requests": 1,
                    "encoder_forward_ms": 20.0,
                },
            )
            server_log_path.write_text(
                "".join(
                    "INFO MM_ENCODER_TIMING " + json.dumps(event) + "\n"
                    for event in encoder_events
                ),
                encoding="utf-8",
            )

            report = analyze(requests_path, cpu_path, server_log_path)

            self.assertIn("| 正式请求数 | 2 |", report)
            self.assertIn("| CPU窗口样本数 | 2 |", report)
            self.assertIn("Encoder Forward (ms)", report)
            self.assertIn("16.00", report)


if __name__ == "__main__":
    unittest.main()
