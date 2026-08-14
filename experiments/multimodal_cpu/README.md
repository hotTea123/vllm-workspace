# Multimodal CPU-offload measurement branches

These experiments collect the minimum data required to estimate whether all or
part of a Qwen2.5-VL vision encoder should move from NPU to CPU. They do not
implement CPU offload.

## Branches

| Branch | Question answered |
| --- | --- |
| `exp/mm-common-base` | What shape and byte volume does each request create? |
| `exp/mm-npu-baseline` | Where does single-request NPU latency go? |
| `exp/mm-vit-layer-profile` | Which ViT stages and operators dominate? |
| `exp/mm-cpu-roofline` | What is the CPU execution lower bound? |
| `exp/mm-npu-transfer` | What does CPU/NPU transfer cost? |
| `exp/mm-memory-kv` | How much NPU memory and KV capacity can offload recover? |
| `exp/mm-concurrency` | How do batching, queueing, and tail latency change? |

All experiment branches start from `exp/mm-common-base`. The unmodified fields
below are therefore present in every branch:

- source image dimensions, format, and bytes;
- file read, media decode, and RGB conversion time;
- processed dimensions and `image_grid_thw`;
- patch count and post-merge visual-token count;
- pixel tensor shape, dtype, and bytes;
- estimated encoder-output tensor shape and bytes.

## Common input-scale probe

Run from the workspace root inside the target container, with the local `vllm`
source installed in that container:

```bash
python -m experiments.multimodal_cpu.collect_input_scale \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --image /data/image.jpg \
  --output-prefix /results/qwen25vl_7b_2k
```

The command writes one JSONL file for programmatic analysis and one UTF-8 CSV
file for spreadsheet inspection. Use the same `min_pixels` and `max_pixels` as
the vLLM run when those processor overrides are configured.

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

## CPU roofline probe

This branch does not move ViT to CPU. It measures sustained CPU memory-copy and
GEMM throughput with the real pre-merge patch count and Qwen vision dimensions.
Use the target runtime's PyTorch CPU backend so the result includes the kernels
and SVE support that a future offload implementation would actually use.

```bash
numactl --cpunodebind=0 --membind=0 \
  python -m experiments.multimodal_cpu.bench_cpu_roofline \
  --input-scale-jsonl /results/qwen25vl_2k_input_scale.jsonl \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --dtype bfloat16 \
  --threads 1,8,16,32,64 \
  --output-prefix /results/qwen25vl_2k_cpu_roofline
```

The four GEMMs represent QKV projection, attention output projection, MLP
gate/up projection, and MLP down projection. The benchmark intentionally uses
`patch_count`, not post-merge visual tokens, because Qwen2.5-VL's transformer
blocks run before the patch merger. Run `perf stat` around this command to verify
cycles, instructions, cache misses, and available Arm vector events. A fast
roofline result is only a lower bound; it is not evidence that a complete CPU
ViT will reach the same latency.
