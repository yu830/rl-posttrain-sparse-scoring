# 显存预算约束的 Teacher Scoring

> 10 月 30 日新增内容为 AI 辅助草稿，等待 Human 审核；此前已审核报告仍保留为独立快照。

这条实验路径把行分块前移到 LM Head 之前，同时保留学生候选 ID 上的全词表归一化教师 logprob 和教师自己的 Top-K ID。目标是开放的 [verl #6920](https://github.com/verl-project/verl/pull/6920) 固定提交 `87cd055e872f9fdd7b976fcd797a0934b6dba265` 中的冻结 Teacher。补丁需要显式启用，尚未合入上游；行分块本身已有先例。

适配器在 FSDP 包装前安装。普通模型入口执行 backbone 与最终 norm，然后用原 LM Head 逐块投影；块小于总行数时不拼接完整 logits。Worker 维持原 nested offsets 和两个字段的 CPU 输出。无预算且一个块覆盖所有行时，集成路径改用原生 dense forward，并在 FSDP 返回后归约。

目前仅支持共享输入/输出 embedding 的 Qwen2、普通稠密头、BF16/FP32、单 GPU / world size 1、无缓存及无教师梯度。多 rank、未共享权重、LoRA、量化/QAT、Liger、融合模型 forward 和序列并行会被明确拒绝。多序列使用 padded 输入和真实 attention mask；多序列 packed/varlen 注意力尚未验证，当前拒绝。预算只估算读取头分配，不能保证完整模型、其他进程、碎片或 CUDA workspace 的总显存上界。

## 10 月 30 日补充：稳定性、容量与消费链

首次真实 Worker 的 CUDA `unknown error` 仍未解释。受控 2×2 实验对 dense/chunked 与 FSDP2 CPU offload on/off 的每种组合各使用 3 个新进程、每进程 2 次预热和 30 次校验调用。384 次调用均匹配同一紧凑输出参考，实际 FSDP 策略与请求一致。另有两个真实 Worker 进程分别完成 32 次逐次哈希和 offsets 检查。这些测试未重现异常，不能证明根因或长期稳定性。

初版直接模型与真实 Worker 的 BF16 输出出现差异：固定 Worker 会显式执行 `model.to(dtype)`，同时将 Qwen2 旋转频率缓冲区从 FP32 转为 BF16；原直接路径没有这一步。所测 fixture 中，对齐该单一构建步骤后，31/32 个频率值发生变化，全部 808 个候选分数和 101 行教师 Top-K 与 Worker 完全一致。这解释了该 fixture 的路径差异，不能解释 CUDA 异常。下方较早的直接模型测量仅适用于其记录的缓冲区配置。

容量实验限制的是 **PyTorch 缓存 allocator fraction**，不是整卡内存硬上限。每个负载都在新进程运行，固定单条合成序列、BF16、K=32、B=256 和模型 32768 context 上限；分块正确性另与较大参考预算下的完整 dense 输出比较。

| allocator fraction | CPU offload | dense 最大已测成功 / 下一格点 OOM | 分块最大可运行 | 分块完成 dense 参考对照至 |
| --- | --- | ---: | ---: | ---: |
| 0.25 | on | 1536 / 2048 | 32768 | 8192 |
| 0.25 | off | 512 / 1024 | 16384 / 16896 OOM | 4096 |
| 0.50 | on | 8704 / 9216 | 32768 | 9216 |

以上是 512-token 边界分辨率下的已测格点，不证明未测形状的单调性，也不是整卡自然最大长度。fraction 0.25 下，已完成数值对照的 8192 与 dense 已测成功的 1536 对应约 5.33 倍的**观察长度比**；32768 行尚无同长度完整 dense 正确性参考。fraction 0.50 下，9216 行分块与 fraction 0.55 的完整 dense 参考相比，294912 个候选值及 9216 行教师 Top-K 均一致；0.55 只作正确性参考，不参与容量比较。

fraction 0.50、offload on、N=8192 的独立进程中，dense/chunked 的单次前向 PyTorch allocated 峰值为 3596.83/681.66 MiB，单次前向 reserved 峰值为 3846/1572 MiB；包含模型加载和包装的进程观察 allocated 峰值为 3596.83/958.48 MiB。`nvidia-smi` 100 ms 全设备最大采样值为 5504/3229 MiB。采样包含其他进程且可能漏峰，不能将差值写成精确进程 VRAM 节省。dense 在 fraction 0.50 下 OOM 时 CUDA 仍报告约 3401 MiB 空闲，该 OOM 是 allocator 预算边界，不是整卡耗尽。

固定 verl 的真实 `TeacherFSDPWorker` dense 与 chunked 输出随后直接在内存中传给原 `compute_reverse_kl_topk` 和真实 Qwen Student。两条 padded 序列 offsets 为 [0,23,45]，含 25 个 response 预测行；360 个教师候选值、教师 Top-K、loss 以及所比较的 6989056 个 logits 梯度和 896 个 RMSNorm 梯度在两条路径一致。AdamW 仅更新最终 RMSNorm，两边梯度非零、各有 1 个 BF16 参数值改变。这是本地直接 Worker→Loss→受限更新，不是 Ray 调度或完整 RL 训练。

在 N=1024/K=32/B=256 的配对实验中，已显式对齐 Worker 缓冲区转换。offload on 的 A0/A1 `perf_counter` p50 为 192.580/192.978 ms、allocated 峰值为 1517.09/664.97 MiB；offload off 为 87.241/78.709 ms、2467.26/1615.48 MiB。两种 offload 设置在不同进程运行；offload off 的 A/B 中 30/32 个 CUDA Event 大于包围它的 perf-counter 区间，时延只作描述，不宣称加速。

## 10 月 29 日直接模型测量（较早的缓冲区配置）

下表是审查后适配器的复测，不是服务或 RL 训练基准。条件：固定 Qwen2.5-0.5B-Instruct、BF16、合成文本分词和确定性候选 ID、RTX 5070 Laptop 8 GiB、PyTorch 2.14.1+cu130、Transformers 5.19.0、单卡 FSDP2 与 CPU 参数 offload、SDPA、关闭 KV cache。A0 执行完整模型并归约完整 logits；A1 对同一完整模型前向使用行分块。两者均包含紧凑结果复制到 CPU。每格包含四组 A/A 和四组交错 ABBA/BAAB，每个 A/B 标签 8 个样本。时延为同步主机 p50；峰值是 PyTorch allocated 绝对值，不是整卡占用。

| 行数 | K | 块大小 | A0 峰值 MiB | A1 峰值 MiB | A0 p50 ms | A1 p50 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 32 | 256 | 552.04 | 552.04 | 196.421 | 196.893 |
| 512 | 32 | 256 | 774.05 | 664.14 | 174.285 | 171.881 |
| 1024 | 32 | 256 | 1517.09 | 665.31 | 148.338 | 149.474 |
| 2048 | 32 | 256 | 1814.31 | 667.65 | 202.120 | 202.617 |

1024 行时，本范围内的 allocated 绝对峰值下降 56.15%；这不是此前仅输出头回放的 75% 增量下降。512 行时若块大小为 512，绝对峰值反而增加约 33.7%，因此无预算时启用短负载 dense 回退。128 行的 A/A 主机 p50 为 118.614 和 115.205 ms，但随后 A/B 主机 p50 约为 196 ms；阶段间漂移使这些数据不能证明普遍加速。Reserved 会受到 allocator 历史影响。早期矩阵覆盖行数 128/512/1024/2048、K=8/32 和块大小 64/128/256/512，32 个组合在记录的契约下通过候选 logprob、教师 Top-K 和头部形状检查。审查后适配器复测相同行数、K=32 与块大小 64/256，8 个组合均通过；表格采用后一次结果。早期临时探针的原始样本因 WSL 临时目录被清空而丢失；表格来自后续保留下来的重跑。更早的一次计时探针出现 CUDA Event 大于同步主机区间的异常，原因未知。

归档矩阵脚本在块覆盖全部行时，错误地把该头部调用标为未物化完整 logits；实际记录的形状显示它执行了完整投影。当前脚本已修正该标记，并在后续检查中验证了两种情况。

使用当前脚本的后续 1024 行确认中，perf-counter 主机 p50 为 163.110/168.292 ms（A0/A1），CLOCK_MONOTONIC_RAW 主机 p50 为 150.562/155.345 ms。A/A 的 8 个样本中，CUDA Event 有 7 个大于 perf-counter 区间，0 个大于 raw 时钟区间。另一个 10 样本 CUDA sleep 对照中，raw 时间有 10 个大于 perf-counter，Event 有 9 个大于 perf-counter、0 个大于 raw。差异根因尚未确认；时延仅作描述，不能证明加速。

另一项 1024 行、K=32 的 FSDP2 读取头探针，对 64/128/256 MiB 估算预算分别选择 35/70/141 行；测得读取头 allocated 增量约为 51.03/101.75/204.64 MiB，完整前向峰值约为 552.28 MiB。该结果不能证明所有负载或后端的硬预算。

已直接实例化打补丁的真实 `TeacherFSDPWorker`，在单进程 FSDP2 CPU offload 和两条不同长度的 padded 输入上运行；所测 dense 回退与分块路径的候选 logprob、教师 Top-K 哈希和 offsets 一致。独立的真实模型消费检查使用学生实际 Top-K、mask 和一次 AdamW 更新，但仅训练学生的最终 RMSNorm 权重：loss 和所比较梯度一致，梯度非零，BF16 参数中有 1 个值发生变化。这不是完整模型训练。首次重复 Worker 运行发生 CUDA `unknown error` 并终止，后续直接运行完成，但错误原因尚未确认，少量成功不能证明长期稳定性。

另将记录的真实模型教师输出送入固定版本的 verl `compute_reverse_kl_topk` 原函数，并使用真实 Qwen Student 进行对照。项目实现与上游函数、以及上游函数消费 A0 与 A1 的 loss、mass、overlap、logits 梯度和最终 RMSNorm 梯度均通过比较；该直接函数测试没有运行完整 Worker-to-Trainer 链路。

最初的项目消费实现对 BF16 学生 logprob 的两条梯度分支分别转换 FP32：标量 loss 与上游相同，但 RMSNorm 和 logits 梯度的相对 L2 分别约差 1.07% 和 1.80%。改为像上游一样复用同一次 FP32 转换后，这组输入上的差异消失。原差异保留为负结果；逐值一致仅适用于所测输入和环境。

## 复现

使用已有锁定的 G2 环境和本地模型快照：

```bash
python scripts/full_teacher_benchmark.py --model-dir "$MODEL_DIR" --out "$OUT_DIR/full-forward.json" \
  --rows 128 512 1024 2048 --ks 8 32 --row-chunks 64 128 256 512 \
  --warmup 2 --aa-blocks 4 --ab-blocks 4 --fsdp2 --cpu-offload
python scripts/teacher_budget_probe.py --model-dir "$MODEL_DIR" --out "$OUT_DIR/budget.json" \
  --rows 1024 --k 32 --budgets-mib 64 128 256 --max-chunk-rows 512 --fsdp2 --cpu-offload
python scripts/teacher_consumer_step.py --model-dir "$MODEL_DIR" --out "$OUT_DIR/consumer.json" \
  --device cuda --dtype bfloat16 --optimizer adamw --learning-rate 0.01
```

以下命令复现对齐 Worker 缓冲区的设置及直接 Worker-to-Loss 检查。每种因子条件使用独立进程；`nvidia-smi` 采样是全设备观察。

```bash
python scripts/factorial_teacher_probe.py --model-dir "$MODEL_DIR" --out "$OUT_DIR/dense-on.json" \
  --mode dense --cpu-offload on --buffer-cast worker --fixture fixtures/s1-worker-padded-64-37.json \
  --lengths 64 37 --k 8 --chunk-rows 16 --attention eager --warmup 2 --repeats 30 --sample-device-ms 100
python scripts/factorial_teacher_probe.py --model-dir "$MODEL_DIR" --out "$OUT_DIR/chunk-on.json" \
  --mode chunked --cpu-offload on --buffer-cast worker --fixture fixtures/s1-worker-padded-64-37.json \
  --expected "$OUT_DIR/dense-on.json" --lengths 64 37 --k 8 --chunk-rows 16 --attention eager --warmup 2 --repeats 30
python scripts/worker_to_student_step.py --verl-dir "$VERL_DIR" --model-dir "$MODEL_DIR" \
  --out "$OUT_DIR/worker-to-student.json" --dtype bfloat16 --topk 8 --chunk-rows 8
python scripts/full_teacher_benchmark.py --model-dir "$MODEL_DIR" --out "$OUT_DIR/timing-on.json" \
  --rows 1024 --ks 32 --row-chunks 256 --warmup 2 --aa-blocks 6 --ab-blocks 8 \
  --fsdp2 --cpu-offload --worker-buffer-cast
```

其他因子条件使用 `--cpu-offload off`。容量运行使用 `--purpose capacity --rows N --k 32 --chunk-rows 256 --attention sdpa --allocator-fraction 0.25 --save-output "$OUT_DIR/output.pt"`；使用 `scripts/compare_capacity_outputs.py` 对照相同输入的输出。allocator fraction 应明确设定并记录其他 GPU 活动；allocator OOM 不自动等于整卡 OOM。

[中文阶段增补](reports/memory-bounded-stage-20261030.zh-CN.pdf)与[可编辑 Markdown](reports/memory-bounded-stage-20261030.zh-CN.md)等待 Human 审核。运行 `python scripts/build_stage_report_20261030.py --output-dir "$OUT_DIR/reports"` 可重新生成 PDF；本次构建使用 ReportLab 4.4.9 和 Windows 宋体/黑体。可用 `--cjk-font` 指定其他完整中文字体，替换字体可能改变分页。

[固定 verl 补丁与 Worker 设置](../integrations/verl_6920/README.zh-CN.md)记录源码版本、配置和离线校验。真实 Worker 运行是直接进程冒烟，不代表 Ray 远程调度或多卡运行。此前已审核的技术报告和架构图仍描述原快照，尚未修订以声明这些新测量。
