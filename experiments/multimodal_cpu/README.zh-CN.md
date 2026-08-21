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
| `zkx/mm-online-cpu` | 预热后的在线服务消耗多少 CPU 资源？ |

可运行的实验分支基于 `exp/mm-common-base`，并保留
以下公共字段：

- 原始图片尺寸、格式和文件字节数；
- 文件读取、媒体解码和 RGB 转换时间；
- 处理后的图片尺寸和 `image_grid_thw`；
- Patch 数和合并后的 Visual Token 数；
- Pixel Tensor 的形状、数据类型和字节数；
- ViT 实际输入 Tensor 的形状、数据类型、设备和字节数；
- Encoder 实际输出 Tensor 的形状、数据类型、设备和字节数。

运行时 `input_scale` 是权威数据源。它由真实 vLLM Encoder 调用采集；实验不再
计算或记录独立 Processor 估算值。

## 测量规则

1. 记录稳态数据前，先完成模型加载、编译和算子预热。
2. 缓存未命中基线应关闭业务缓存，但不要在每个样本之间清除已编译算子或
   权重缓存。
3. 保留逐请求原始记录；汇总结果必须从原始数据计算，不能替代原始数据。
4. `scheduled_to_first_token` 是混合区间，可能同时包含 Encoder、Embedding
   融合、语言模型 Prefill 和首 Token 生成，不能把它直接标为纯 Prefill。
5. 硬件性能必须在目标鲲鹏/昇腾主机上测量；Windows 上的源码检查不能作为
   性能结果。
6. 后续建模只使用真实 vLLM 运行时调用采集的数据。

## NPU 单请求基线

该分支将输出长度固定为 256 Token，关闭多模态处理器缓存和 Prefix Cache，
在每个样本前重置 Encoder Cache，并验证每个请求只执行一次 Encoder。

```bash
python -m experiments.multimodal_cpu.run_npu_baseline \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --image /data/image.jpg \
  --tensor-parallel-size 1 \
  --warmups 2 \
  --repeats 10 \
  --output-prefix /results/qwen25vl_7b_2k
```

原始记录包含 `preprocessor_total_ms`、各预处理阶段时间、
`encoder_forward_ms`、TTFT、TPOT、排队时间、引擎端到端时间和调用方观测的
Wall Time。`encoder_forward_ms` 是同步后的完整 Encoder 路径时间；纯 ViT
阶段和算子拆解应使用 ViT 层级分析分支。

每条正式请求都会记录来自真实 vLLM Encoder 调用的 `input_scale`，其中包含
实际 Grid、设备输入 Tensor 和 Encoder 输出 Tensor。后续 Roofline 和传输
实验只使用该字段作为输入规模数据源。

请求时延统计会被显式启用；如果 TTFT 或 TPOT 不可用，实验会直接失败，不会
写出指标不完整的记录。

该分支还通过昇腾 `NPUWorker` 委托调用 Encoder 计时和 Batch 元数据 RPC，
未在推理热路径中增加第二次设备同步。

## 结果分析与测试

使用 NPU 基线生成的 JSONL 输出输入规模、Encoder Tensor 规模和
时延统计表：

```bash
python -m experiments.multimodal_cpu.analyze_npu_baseline \
  --input /results/qwen25vl_72b_baseline.jsonl \
  --output /results/qwen25vl_72b_baseline_report.md
```

运行分析脚本的单元测试：

```bash
python -m unittest \
  experiments.multimodal_cpu.test_analyze_npu_baseline
```

如需采集完整进程树的 CPU 和 NPU 利用率，应在主机侧同时运行 `pidstat`/
`perf` 和 `npu-smi`。vLLM Worker 可能运行在独立进程中，因此仅统计父进程
CPU 时间不具有代表性。

## 预热后的在线 CPU 测量

`zkx/mm-online-cpu` 分支只采集宿主机 CPU 资源、流式 TTFT、TPOT
和同步后的 Encoder 整体时间，不重复采集离线基线已经获得的输入规模和
Tensor 大小。

在推理容器内启动服务并保存服务日志：

```bash
bash experiments/multimodal_cpu/start_online_cpu_server.sh \
  --model /models/Qwen2.5-VL-72B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 16635 \
  --port 12346 \
  -- --max-num-batched-tokens 16635 \
  2>&1 | tee /results/online_cpu_server.log
```

启动脚本始终使用以下隔离配置：

- `--mm-processor-cache-gb 0`；
- `--no-enable-prefix-caching`；
- `compile_mm_encoder=false`；
- `cudagraph_mm_encoder=false`；
- 通过 `--enable-mm-processor-stats` 启用同步 Encoder 计时。

不能关闭请求内部的 Encoder Cache，因为 vLLM 需要它把 ViT 输出插入语言模型
Prefill。客户端改为在每次请求中发送完全相同的图片，但设置唯一的媒体
`uuid`。vLLM 使用该 UUID 作为 Encoder Cache 键，因此不同请求无法复用
Encoder 结果。

服务就绪后，在 Docker 宿主机执行：

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

编排脚本会先完成全部 Warmup，再启动 `collect_host_cpu.sh`；第10个正式
请求完成后立即停止 CPU 采集。输出：

- `/results/qwen72b_online_cpu.requests.jsonl`：TTFT、TPOT、请求ID、
  Media UUID 和毫秒级测量窗口；
- `/results/qwen72b_online_cpu.host_cpu.csv`：容器 CPU 百分比、有效CPU核数、
  内存用量和PID数。

把服务日志放到宿主机可访问的路径后，生成缓存隔离和性能汇总：

```bash
python -m experiments.multimodal_cpu.analyze_online_cpu \
  --requests /results/qwen72b_online_cpu.requests.jsonl \
  --host-cpu /results/qwen72b_online_cpu.host_cpu.csv \
  --server-log /results/online_cpu_server.log \
  --output /results/qwen72b_online_cpu_report.md
```

分析脚本仅保留正式请求时间窗口内的 CPU 样本；每个正式请求都必须匹配
一条 Encoder 计时记录，同一请求在多个 TP rank 上的 Encoder 时间取最大值。
