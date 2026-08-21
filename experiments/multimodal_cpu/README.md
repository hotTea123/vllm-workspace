# Multimodal CPU-offload measurement branches

These experiments collect the minimum data required to estimate whether all or
part of a Qwen2.5-VL vision encoder should move from NPU to CPU. They do not
implement CPU offload.

## Branches

| Branch | Question answered |
| --- | --- |
| `exp/mm-common-base` | Shared collection code and tests; not a runnable experiment. |
| `exp/mm-npu-baseline` | Where does single-request NPU latency go? |
| `exp/mm-vit-layer-profile` | Which ViT stages and operators dominate? |
| `exp/mm-cpu-roofline` | What is the CPU execution lower bound? |
| `exp/mm-npu-transfer` | What does CPU/NPU transfer cost? |
| `exp/mm-memory-kv` | How much NPU memory and KV capacity can offload recover? |
| `exp/mm-concurrency` | How do batching, queueing, and tail latency change? |
| `zkx/mm-online-cpu` | What CPU resources does the warmed online service consume? |

The runnable experiment branches start from
`exp/mm-common-base` and preserve these common fields:

- source image dimensions, format, and bytes;
- file read, media decode, and RGB conversion time;
- processed dimensions and `image_grid_thw`;
- patch count and post-merge visual-token count;
- pixel tensor shape, dtype, and bytes;
- actual ViT input tensor shape, dtype, device, and bytes;
- actual encoder-output tensor shape, dtype, device, and bytes.

Runtime `input_scale`, captured at the real vLLM encoder boundary, is the source
of truth. The experiment does not calculate or record a separate processor
estimate.

## Measurement rules

1. Warm up model loading, compilation, and kernels before recording steady-state
   samples.
2. Disable application caches for cache-miss baselines, but do not clear compiled
   kernel or weight caches between samples.
3. Preserve raw per-request rows. Summaries must be derived from raw data rather
   than replacing it.
4. Treat `scheduled_to_first_token` as a mixed interval containing encoder,
   embedding integration, language-model prefill, and first-token work.
5. Run hardware measurements on the target Kunpeng/Ascend host. Source-only
   checks on Windows do not constitute performance results.
6. Use only values captured from real vLLM runtime calls in subsequent modeling.

## NPU single-request baseline

This branch fixes output length at 256 tokens, disables multimodal processor and
prefix caches, resets the encoder cache before every sample, and verifies one
encoder invocation per request.

```bash
python -m experiments.multimodal_cpu.run_npu_baseline \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --image /data/image.jpg \
  --tensor-parallel-size 1 \
  --warmups 2 \
  --repeats 10 \
  --output-prefix /results/qwen25vl_7b_2k
```

The raw rows contain `preprocessor_total_ms`, individual processor stages,
`encoder_forward_ms`, TTFT, TPOT, queue time, engine E2E time, and caller wall
time. `encoder_forward_ms` is the synchronized whole encoder path; use the ViT
layer-profile branch for a pure stage/operator breakdown.

Each measured request records `input_scale` from the real vLLM encoder call. It
contains the actual grid, device input tensors, and encoder output tensors, and
is the only input-scale source for Roofline and transfer experiments.

Request metrics are enabled explicitly. The experiment fails instead of writing
an incomplete row if TTFT or TPOT is unavailable.

The branch delegates encoder timing and batch-metadata RPCs through Ascend's
`NPUWorker`; it does not add a second device synchronization to the hot path.

## Result analysis and tests

Generate the input-scale, encoder-tensor, and latency summary tables from an
NPU baseline JSONL file:

```bash
python -m experiments.multimodal_cpu.analyze_npu_baseline \
  --input /results/qwen25vl_72b_baseline.jsonl \
  --output /results/qwen25vl_72b_baseline_report.md
```

Run the analyzer unit tests:

```bash
python -m unittest \
  experiments.multimodal_cpu.test_analyze_npu_baseline
```

For full process-tree CPU and NPU utilization, run the experiment under the
host's `pidstat`/`perf` and `npu-smi` collectors. Parent-process CPU time alone is
not representative because vLLM workers may be separate processes.

## Warmed online CPU measurement

The `zkx/mm-online-cpu` branch measures only host CPU resources, streaming TTFT,
TPOT, and synchronized whole-encoder time. It does not repeat input-scale
or tensor-size collection from the offline baseline.

Start the server inside the inference container and save its log:

```bash
bash experiments/multimodal_cpu/start_online_cpu_server.sh \
  --model /models/Qwen2.5-VL-72B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 16635 \
  --port 12346 \
  -- --max-num-batched-tokens 16635 \
  2>&1 | tee /results/online_cpu_server.log
```

The launcher always uses the following isolation settings:

- `--mm-processor-cache-gb 0`;
- `--no-enable-prefix-caching`;
- `compile_mm_encoder=false`;
- `cudagraph_mm_encoder=false`;
- `--enable-mm-processor-stats` for synchronized encoder timing.

It does not disable the request-local encoder cache because vLLM needs that
storage to splice ViT outputs into language-model prefill. Instead, the client
sends the exact same image with a unique media `uuid` in every request. vLLM
uses that UUID as the encoder-cache key, so different requests cannot share the
cached encoder output.

After the server is ready, run the measurement on the Docker host:

```bash
bash experiments/multimodal_cpu/run_online_cpu_measurement.sh \
  --container vllm-container \
  --base-url http://127.0.0.1:12346 \
  --model /models/Qwen2.5-VL-72B-Instruct \
  --image /data/1-1024x576.jpg \
  --warmups 4 \
  --repeats 10 \
  --output-tokens 100 \
  --output-prefix /results/qwen72b_online_cpu
```

The orchestration script finishes all warmups before starting
`collect_host_cpu.sh`, then stops CPU sampling immediately after the last
measured response. It writes:

- `/results/qwen72b_online_cpu.requests.jsonl` for TTFT, TPOT, request IDs,
  media UUIDs, and millisecond window timestamps;
- `/results/qwen72b_online_cpu.host_cpu.csv` for container CPU percentage,
  effective CPU cores, memory usage, and PIDs.

Generate the final cache-isolation and performance summary after making the
server log available on the host:

```bash
python -m experiments.multimodal_cpu.analyze_online_cpu \
  --requests /results/qwen72b_online_cpu.requests.jsonl \
  --host-cpu /results/qwen72b_online_cpu.host_cpu.csv \
  --server-log /results/online_cpu_server.log \
  --output /results/qwen72b_online_cpu_report.md
```

The analyzer restricts CPU samples to the measured request timestamps, requires
one encoder timing match for every measured response, and takes the maximum
encoder time across tensor-parallel ranks for each request.
