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

## ViT 层级与算子分析

该分支为 Qwen2.5-VL 的输入类型转换、PatchEmbed、元数据准备、Token 重排、
每个 ViT Block、Merger 和输出重排增加 Profiler 范围。只有启用已有的
`VLLM_CUSTOM_SCOPES_FOR_PROFILING` 开关时这些范围才会生效；运行脚本会在
导入 vLLM 前自动启用该开关。

```bash
python -m experiments.multimodal_cpu.run_vit_profile \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --image /data/image.jpg \
  --profile-dir /results/vit_profile_2k
```

使用受支持的 Profiler 界面打开生成的 torch-npu Trace，并按 `mm.vit.`
过滤。应比较每个范围下实际嵌套的 NPU Kernel；由于 NPU 算子是异步下发的，
不能把某个范围的主机侧持续时间直接报告为 NPU 执行时间。

Manifest 中的 `input_scale` 来自本次 Profile 的真实 Encoder 调用。
`input_scale_estimate` 和 `input_scale_comparison` 仅用于展示估算差异，不能
替代运行时实际值。
