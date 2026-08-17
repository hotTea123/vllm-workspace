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

The other six branches are runnable experiments. They start from
`exp/mm-common-base` and preserve these common fields:

- source image dimensions, format, and bytes;
- file read, media decode, and RGB conversion time;
- processed dimensions and `image_grid_thw`;
- patch count and post-merge visual-token count;
- pixel tensor shape, dtype, and bytes;
- actual ViT input tensor shape, dtype, device, and bytes;
- actual encoder-output tensor shape, dtype, device, and bytes.

Runtime `input_scale`, captured at the real vLLM encoder boundary, is the source
of truth. `input_scale_estimate` is an independent processor estimate used only
for comparison. `input_scale_comparison` records differences and never fails an
experiment merely because the estimate differs.

## Auxiliary input-scale estimate

Run from the workspace root inside the target container, with the local `vllm`
source installed in that container:

```bash
python -m experiments.multimodal_cpu.collect_input_scale \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --image /data/image.jpg \
  --output-prefix /results/qwen25vl_7b_2k
```

The command writes one JSONL file for programmatic analysis and one UTF-8 CSV
file for spreadsheet inspection. It does not execute a real vLLM inference and
must not be used as the authoritative Roofline or transfer input. The NPU
baseline records runtime values, this estimate, and their differences together.
Use the same `min_pixels` and `max_pixels` as the vLLM run.

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
6. Preserve estimate/runtime differences and continue the experiment; use only
   runtime values in subsequent modeling.

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

Each measured request also records `input_scale`, `input_scale_estimate`, and
`input_scale_comparison`. `input_scale` comes from the real vLLM encoder call
and contains the actual grid, device input tensors, and encoder output tensors.
Roofline and transfer experiments must consume this field. Estimate/runtime
differences are recorded without failing the experiment.

The branch delegates encoder timing and batch-metadata RPCs through Ascend's
`NPUWorker`; it does not add a second device synchronization to the hot path.

For full process-tree CPU and NPU utilization, run the experiment under the
host's `pidstat`/`perf` and `npu-smi` collectors. Parent-process CPU time alone is
not representative because vLLM workers may be separate processes.
