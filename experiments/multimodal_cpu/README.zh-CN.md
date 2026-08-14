# 多模态 CPU 卸载测量分支

这些实验用于采集判断 Qwen2.5-VL 视觉编码器是否适合全部或部分从 NPU
卸载到 CPU 所需的最小数据集。实验本身不会实现 CPU 卸载。

## 分支说明

| 分支 | 需要回答的问题 |
| --- | --- |
| `exp/mm-common-base` | 每个请求实际产生什么形状和多少字节的数据？ |
| `exp/mm-npu-baseline` | 单请求在 NPU 上的时延花在哪里？ |
| `exp/mm-vit-layer-profile` | 哪些 ViT 阶段和算子占用主要时间？ |
| `exp/mm-cpu-roofline` | CPU 执行 ViT 的理论时间下界是多少？ |
| `exp/mm-npu-transfer` | CPU 与 NPU 之间的数据传输代价是多少？ |
| `exp/mm-memory-kv` | 卸载可以释放多少 NPU 显存和 KV Cache 容量？ |
| `exp/mm-concurrency` | 批处理、排队和尾时延如何随并发变化？ |

所有实验分支都基于 `exp/mm-common-base`，因此每个分支都会保留以下公共字段：

- 原始图片尺寸、格式和文件字节数；
- 文件读取、媒体解码和 RGB 转换时间；
- 处理后的图片尺寸和 `image_grid_thw`；
- Patch 数和合并后的 Visual Token 数；
- Pixel Tensor 的形状、数据类型和字节数；
- Encoder 输出 Tensor 的估算形状和字节数。

## 公共输入规模采集

在目标容器内从工作区根目录运行，并确保容器安装或加载的是当前本地
`vllm` 源码：

```bash
python -m experiments.multimodal_cpu.collect_input_scale \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --image /data/image.jpg \
  --output-prefix /results/qwen25vl_7b_2k
```

命令会生成一个供程序分析的 JSONL 文件，以及一个便于在表格软件中查看的
UTF-8 CSV 文件。如果 vLLM 实验配置了 `min_pixels` 和 `max_pixels`，这里必须
使用相同的处理器参数。

## 测量规则

1. 记录稳态数据前，先完成模型加载、编译和算子预热。
2. 缓存未命中基线应关闭业务缓存，但不要在每个样本之间清除已编译算子或
   权重缓存。
3. 保留逐请求原始记录；汇总结果必须从原始数据计算，不能替代原始数据。
4. `scheduled_to_first_token` 是混合区间，可能同时包含 Encoder、Embedding
   融合、语言模型 Prefill 和首 Token 生成，不能把它直接标为纯 Prefill。
5. 硬件性能必须在目标鲲鹏/昇腾主机上测量；Windows 上的源码检查不能作为
   性能结果。

## 闭合批次并发扫描

该分支为每次分组后的 Encoder 调用增加一条结构化记录，包括批次 ID、请求
ID、媒体项数量、合并后的 Encoder Token 数和同步批次时间。同时为昇腾
Worker 补充对上游已有逐请求 Encoder 计时注册表的委托接口。

```bash
python -m experiments.multimodal_cpu.run_concurrency \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --image /data/image.jpg \
  --concurrency 1,4,8,16 \
  --repeats 3 \
  --output-prefix /results/qwen25vl_7b_concurrency
```

同一个并发级别下的所有请求会在离线引擎循环开始前提交，因此该实验测量的
是闭合批次和引擎真实的 Encoder 分组行为，不包含网络请求到达模式。开放式
在线负载应另行使用 `vllm bench serve` 测试；本分支得到的批处理服务曲线
用于卸载理论模型。实验期间应同时运行 `pidstat`/`perf` 和 `npu-smi`，采集
完整进程树的 CPU 与 NPU 利用率。
