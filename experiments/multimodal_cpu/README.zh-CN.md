# 多模态 CPU 卸载测量分支

这些实验用于采集判断 Qwen2.5-VL 视觉编码器是否适合全部或部分从 NPU
卸载到 CPU 所需的最小数据集。实验本身不会实现 CPU 卸载。

## 分支说明

| 分支 | 需要回答的问题 |
| --- | --- |
| `exp/mm-common-base` | 共享采集代码和测试；不是需要单独运行的实验。 |
| `exp/mm-npu-baseline` | 单请求在 NPU 上的时延花在哪里？ |
| `exp/mm-vit-layer-profile` | 哪些 ViT 阶段和算子占用主要时间？ |
| `exp/mm-cpu-roofline` | CPU 执行 ViT 的理论时间下界是多少？ |
| `exp/mm-npu-transfer` | CPU 与 NPU 之间的数据传输代价是多少？ |
| `exp/mm-memory-kv` | 卸载可以释放多少 NPU 显存和 KV Cache 容量？ |
| `exp/mm-concurrency` | 批处理、排队和尾时延如何随并发变化？ |

其余六个分支是需要实际运行的实验。它们都基于 `exp/mm-common-base`，并保留
以下公共字段：

- 原始图片尺寸、格式和文件字节数；
- 文件读取、媒体解码和 RGB 转换时间；
- 处理后的图片尺寸和 `image_grid_thw`；
- Patch 数和合并后的 Visual Token 数；
- Pixel Tensor 的形状、数据类型和字节数；
- ViT 实际输入 Tensor 的形状、数据类型、设备和字节数；
- Encoder 实际输出 Tensor 的形状、数据类型、设备和字节数。

运行时 `input_scale` 是权威数据源。它由真实 vLLM Encoder 调用采集；
`input_scale_estimate` 来自独立 Processor，仅用于辅助校验；
`input_scale_comparison` 记录两者差异，不会因不一致终止实验。

## 辅助输入规模估算

在目标容器内从工作区根目录运行，并确保容器安装或加载的是当前本地
`vllm` 源码：

```bash
python -m experiments.multimodal_cpu.collect_input_scale \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --image /data/image.jpg \
  --output-prefix /results/qwen25vl_7b_2k
```

命令会生成一个供程序分析的 JSONL 文件，以及一个便于在表格软件中查看的
UTF-8 CSV 文件。它不会执行真实 vLLM 推理，因此不能作为 Roofline 或传输
实验的权威输入。NPU 基线会在真实推理过程中同时生成实际值、估算值和差异。
如果 vLLM 实验配置了 `min_pixels` 和 `max_pixels`，估算必须使用相同参数。

## 测量规则

1. 记录稳态数据前，先完成模型加载、编译和算子预热。
2. 缓存未命中基线应关闭业务缓存，但不要在每个样本之间清除已编译算子或
   权重缓存。
3. 保留逐请求原始记录；汇总结果必须从原始数据计算，不能替代原始数据。
4. `scheduled_to_first_token` 是混合区间，可能同时包含 Encoder、Embedding
   融合、语言模型 Prefill 和首 Token 生成，不能把它直接标为纯 Prefill。
5. 硬件性能必须在目标鲲鹏/昇腾主机上测量；Windows 上的源码检查不能作为
   性能结果。
6. 输入规模估算与运行时实际值不一致时保留差异并继续实验，后续建模只使用
   运行时实际值。

## CPU Roofline 测试

该分支不会把 ViT 实际卸载到 CPU，而是使用真实的合并前 Patch 数量和 Qwen
视觉模型维度，测量 CPU 的持续内存拷贝带宽和 GEMM 吞吐。测试必须使用目标
运行环境的 PyTorch CPU 后端，这样结果才能反映未来卸载实现实际使用的算子
库和 SVE 支持情况。

```bash
numactl --cpunodebind=0 --membind=0 \
  python -m experiments.multimodal_cpu.bench_cpu_roofline \
  --input-scale-jsonl /results/qwen25vl_2k_input_scale.jsonl \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --dtype bfloat16 \
  --threads 1,8,16,32,64 \
  --output-prefix /results/qwen25vl_2k_cpu_roofline
```

四个 GEMM 分别代表 QKV 投影、Attention 输出投影、MLP Gate/Up 投影和 MLP
Down 投影。测试使用 `patch_count`，而不是合并后的 Visual Token 数，因为
Qwen2.5-VL 的 Transformer Block 位于 Patch Merger 之前。可在命令外层运行
`perf stat`，检查周期数、指令数、Cache Miss 和可用的 Arm 向量事件。
Roofline 结果即使很快，也只代表理论下界，不能证明完整 CPU ViT 能达到相同
时延。
