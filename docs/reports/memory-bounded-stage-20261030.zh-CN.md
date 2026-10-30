---
title: 显存预算约束的 Teacher Scoring
subtitle: 受控稳定性、分配器预算下的容量与真实 Worker-to-Student 验证
author: Jiacheng Yu
edition: 2026 年 10 月 30 日 | 简体中文
status: AI 辅助工程草稿；新增结果待 Human 审核
footer: 草稿 - 待 Human 审核
scope: 在 RTX 5070 Laptop 单 GPU 上，固定 Qwen2.5-0.5B 与 verl 源码，分别检验稳定性、容量和受限 Student 更新。
history: 此前已审核的 c1ab1f4 中英文报告保持原样。本增补是新稿，不继承旧稿的人工审核状态。
---
## 1. 问题与执行契约

Teacher 即使只返回紧凑分数，也可能先分配完整的 token×vocabulary logits。本项目把行分块前移到稠密 LM Head 之前，同时保留两项输出：Student 候选 ID 上的全词表归一化 logprob，以及 Teacher 自身的 Top-K ID。此前仅凭融合 selected logits 替换完整契约的方案仍为 NO-GO。分块本身已有先例；本项目的工程价值在于固定 Teacher 执行链与消费接口的接入。

本阶段分别检验首次 CUDA 异常能否在受控条件下复现、明确 allocator 预算时可执行的 Token 长度，以及真实 Worker 输出能否保留 Student 更新。直接模型/FSDP 实验与实际 verl Worker 执行在报告中分别标明。

表 1. 固定执行条件。

| 项目 | 固定条件 |
| --- | --- |
| 硬件 | NVIDIA RTX 5070 Laptop GPU，8151 MiB |
| 模型 | Qwen2.5-0.5B-Instruct，BF16，共享 embedding |
| 模型版本 | `7ae557604adf67be50417f59c2c2f167def9a775` |
| 框架 | PyTorch 2.14.1+cu130；Transformers 5.19.0 |
| verl 源码 | PR #6920，`87cd055e872f9fdd7b976fcd797a0934b6dba265` |
| 支持范围 | World size 1、无 KV cache、padded 文本输入 |

### 2. 受控稳定性（S1）

直接模型实验比较 Dense/Chunked 与 FSDP2 CPU offload 开/关四种条件，每种条件使用三个新进程。各进程在同一 padded 64/37-token 输入、K=8、B=16 下执行两次预热与 30 次计数调用。每次完成后记录输出哈希和 offsets，并检查 root、decoder 的实际 offload policy。

表 2. 按入口与条件记录的稳定性完成调用。

| 执行入口 | 条件 | 新进程数 | 完成调用数 |
| --- | --- | ---: | ---: |
| 直接 Qwen + FSDP2 | Dense，offload 开 | 3 | 96 |
| 直接 Qwen + FSDP2 | Chunked，offload 开 | 3 | 96 |
| 直接 Qwen + FSDP2 | Dense，offload 关 | 3 | 96 |
| 直接 Qwen + FSDP2 | Chunked，offload 关 | 3 | 96 |
| 实际 verl Worker | Dense，offload 开 | 1 | 32 |
| 实际 verl Worker | Chunked，offload 开 | 1 | 32 |

12 个直接模型进程完成 384 次调用并通过所检紧凑输出契约；另两个实际 Worker 进程共完成 64 次调用。原 CUDA unknown error 在这一有界集合中未复现。其根因仍未知：结果既未确认 offload/分块是原因，也未证明长期生产稳定性。

<!-- pagebreak -->

## 3. 明确 allocator 预算下的容量

容量扫描的每个点均使用新进程、BF16、K=32、B=256、SDPA 与一条因果序列。通过 `torch.cuda.set_per_process_memory_fraction(0.25)` 设置本进程 PyTorch 缓存 allocator 限额，对该设备相当于名义 2,129,510,400 字节。它约束 allocator，不是整卡 GPU 显存硬隔离。

扫描逐步增加 N，并以 512-token 粒度收窄失败区间。成功表示所要求的紧凑输出路径完成且保存了有限值；没有同 N dense 真值的长度仍不能据此判定完整数值正确。

表 3. Allocator fraction 0.25 下的容量扫描。

| CPU offload | 路径 | 最大已测成功 N | 最小已测 OOM N | 同 N dense 校验到达 |
| --- | --- | ---: | ---: | ---: |
| 开 | Dense | 1536 | 2048 | 参考路径 |
| 开 | Chunked | 32768 | 未观察到 | 8192 |
| 关 | Dense | 512 | 1024 | 参考路径 |
| 关 | Chunked | 16384 | 16896 | 4096 |

Offload 开启的 Chunked 路径到达本次测试的模型 context 上限 32768，尚无已测失败上界。关闭 offload 时，16384 成功、16896 OOM 构成已测区间；不能据此保证任意中间形状或未测输入。

### 3.1 容量结果的正确性边界

N=1024 的 offload 开启比较来自同为 0.25 的两条路径。其余更长 dense 参考使用独立的 0.55 allocator 比例，在相同 N、模型和执行策略下，与 0.25 Chunked 运行保存的全部紧凑行、候选及 Teacher Top-K 集合比较。

表 4. 修正版同 N 紧凑输出对照。

| CPU offload | 修正版比较器通过的同 N 对照 | 失败的 dense 参考 |
| --- | --- | --- |
| 开 | 1024、2048、4096、8192；logprob 最大差 0 | 12288：OOM |
| 关 | 2048、4096；logprob 最大差 0 | 8192：OOM |

因此，有数值正确性支撑的已测长度，从 0.25 Dense 的 1536 扩展到 offload 开启时的 8192，从 512 扩展到关闭时的 4096。更长 Chunked 成功仍只作为可运行容量证据。高预算参考不能变成同预算的内存性能比较。比较器补齐 shape/签名校验后，六组结果重新运行并通过，旧比较记录保留。

<!-- pagebreak -->

## 3.2 第二 allocator 预算：0.50

预登记的后续检查在 CPU offload 开启时使用 0.50 PyTorch allocator 比例。Dense 在 N=8704 成功，在下一 512-token 格点 N=9216 OOM。Chunked 在已测 8192/16384/24576/32768 点均完成；32768 的成功仍只作为可运行证据，不代表该长度已经数值等价验证。

表 5. Allocator fraction 0.50 下的容量观察。

| 条件 | 最大已测成功 N | 下一已测 OOM N | Chunked 的 dense 校验 N |
| --- | ---: | ---: | ---: |
| Dense，fraction 0.50 | 8704 | 9216 | 参考路径 |
| Chunked，fraction 0.50 | 32768 | 未观察到 | 9216 |

N=8192 时两条路径同用 0.50，全部保存的候选 logprob 与 Teacher Top-K 行一致。为检验超过 Dense 0.50 边界的点，另用 fraction 0.55 的 Dense 在 N=9216 作参考，与 fraction 0.50 的 Chunked 输出比较：294,912 个候选数值和全部 9216 行 Teacher Top-K 集合逐值一致。因此在本测试格点上，有正确性支撑的长度比最大已测 Dense 成功 8704 多 512 tokens，约 5.88%；它不是对全部中间形状或精确最大值的保证。

### N=8192 的同长度显存观察

表 6. N=8192 的独立进程显存观察。

| 测量 | Dense，fraction 0.50 | Chunked，fraction 0.50 |
| --- | ---: | ---: |
| Forward-call allocated 峰值，MiB | 3596.83 | 681.66 |
| 含加载/包装的进程 allocator 观察峰值，MiB | 3596.83 | 958.48 |
| Forward-call reserved 峰值，MiB | 3846 | 1572 |
| 整设备 used 采样最大值，MiB | 5504 | 3229 |
| 设备采样数 | 36 | 33 |

两组使用相同 N 的独立新进程，紧凑输出已经比较。进程 allocator 观察峰值包含加载、包装与调用；较低的调用峰值并不意味着初始化分配消失。整设备最大采样值仍受 100 ms 采样范围限制，可能包含无关占用并漏峰，不是精确进程峰值，也不是净 VRAM 收益归因。

两个 allocator 比例给出了不同的已测容量区间。0.50 对照加强了“结果依赖所声明内存限额”的观察，并没有验证最长 Chunked 成功点；两组预算下的 dense 参考边界均继续明确保留。


<!-- pagebreak -->

## 4. 显存与时延对应不同测量口径

独立的配对 T1 实验采用 Worker 对齐的 buffer 构建、N=1024、K=32、B=256，测量完整直接 Qwen/FSDP2 前向至紧凑 CPU 输出；加载和校验在计时区间之外。各 offload 设置包含六组 A/A 与八组交错 ABBA/BAAB，每个 A/B 标签 16 个样本。

表 7. N=1024 配对完整前向的 allocated 峰值。

| CPU offload | A0 allocated 峰值 MiB | A1 allocated 峰值 MiB | 观察到的下降 |
| --- | ---: | ---: | ---: |
| 开 | 1517.09 | 664.97 | 56.17% |
| 关 | 2467.26 | 1615.48 | 34.52% |

表 8. T1 配对主机时钟样本的描述性统计。

| CPU offload / 时钟 | A0 p50 / p95 ms | A1 p50 / p95 ms |
| --- | ---: | ---: |
| 开 / perf_counter | 192.580 / 193.953 | 192.978 / 198.150 |
| 开 / MONOTONIC_RAW | 190.491 / 191.858 | 190.926 / 196.081 |
| 关 / perf_counter | 87.241 / 114.847 | 78.709 / 116.953 |
| 关 / MONOTONIC_RAW | 87.567 / 115.198 | 78.986 / 117.407 |

两组 T1 的候选 logprob 最大差均为 0，Teacher Top-K 在全部 1024 行一致。关闭 offload 的组中，30/32 个 A/B 样本和 12/12 个 A/A 样本的 CUDA Event 却长于包围它的 perf-counter 区间。主机时钟分歧尚未解决。因此这些是描述性时延，不支持通用加速，也不宜把不同 offload 进程直接解释为生产性能优劣。

### N=1024 的独立进程观察

M1 的 offload 开启 Dense/Chunked 对照说明 allocator 与设备采样必须分开报告。

表 9. N=1024 的独立进程显存观察。

| 测量 | Dense | Chunked |
| --- | ---: | ---: |
| Forward-call allocated 峰值，MiB | 1517.09 | 665.31 |
| 含加载/包装的进程 allocator 观察峰值，MiB | 1517.09 | 958.48 |
| Forward-call reserved 峰值，MiB | 1748 | 1572 |
| 整设备 used 采样最大值，MiB | 3429 | 3258 |
| 设备采样数 | 40 | 30 |

设备采样每 100 ms 查询一次 nvidia-smi，包含桌面及其他进程，且可能漏掉瞬时峰值。两个采样最大值相差 171 MiB，不能据此宣称净进程 VRAM 收益；同样不能把 allocated 下降扩大为整卡显存或 RL 训练收益。

<!-- pagebreak -->

## 5. 真实 Worker 输出进入受限 Student 更新

I1 补齐本地内存内的消费链。真实 Student 先生成一次候选，然后两个实际固定 verl TeacherFSDPWorker 分别执行 Dense 回退和 Chunked 评分；它们当前返回的 CPU nested TensorDict 直接映射回原 Student padded 行，并送入固定 `compute_reverse_kl_topk` 原函数。此路径没有用记录的 JSON 教师分数代替 Worker 输出。

输入为两条 prompt/response 序列，offsets 为 [0,23,45]，padded 模型行数 46，K=8，激活的响应预测行 25。Student 先按温度 0.8 缩放，再 log-softmax、Top-K；Teacher 温度为 1。候选不再额外位移，只有 response mask 按 next-token 预测行对齐。

两条 Student 路径均恢复相同最终 RMSNorm 初值和新的 AdamW 状态，学习率 0.01，weight decay 0。损失保留候选支持集归一化、logprob 下限 -10、token loss 非负截断及最大损失截断 10；Teacher 张量保持 detach。

表 10. 真实 Worker-to-Student 受限更新。

| 观察量 | Dense Worker 输出 | Chunked Worker 输出 |
| --- | ---: | ---: |
| Mask 后标量 loss | 0.0216006283 | 0.0216006283 |
| 最终 RMSNorm 梯度 L2 | 0.0124774044 | 0.0124774044 |
| 非零 RMSNorm 梯度元素 | 896 | 896 |
| 一步后改变的 BF16 参数值 | 1 | 1 |
| 参数更新 L2 | 0.0099792480 | 0.0099792480 |

本 fixture 中，候选 logprob、Teacher Top-K、mass、overlap、token loss、全部 6,989,056 个 logits 梯度元素、896 个 RMSNorm 梯度元素和更新结果逐值一致。两个 Worker Python 对象均在顺序生命周期边界被回收；实际 outer/head 前后 hook 正常完成，Teacher 未产生梯度。

该证据覆盖本地直接 Worker-to-Loss-to-受限更新。它没有运行 Ray 调度、完整 Trainer、全参数优化或 RL step；训练证据明确限定为一个参数组的一步更新。

<!-- pagebreak -->

## 6. 解释、负结果与后续工作

本阶段形成三项具体结果：受控 2×2 集合没有复现原生 CUDA abort；head 前行分块在明确 allocator 限额下允许更长的已测输入，但 fraction 0.25 的 dense 参考正确性只证实到 offload 开启 N=8192、关闭 N=4096，fraction 0.50 的开启组证实到 N=9216；真实 Worker 当前输出保留了固定上游损失与受限 Student 更新。

在比较直接模型与 Worker 之前发现了构建差异：固定 Worker 在加载后显式执行 `model.to(bfloat16)`，使非持久 rotary-frequency buffers 舍入为 BF16，原直接路径则保留 FP32。应用 Worker 的构建策略后，所测 fixture 的输出差异消失。这一数值解释与 CUDA unknown error 的归因是两个问题。

首次 CUDA unknown error、同期 WSL dxg 观察和双时钟分歧继续保持开放。超过 dense 参考上限的更长 Chunked 输入仍为数值等价 UNKNOWN。FP32 GPU、FSDP1、多 GPU/TP、packed varlen 注意力、LoRA/量化/QAT、Ray 传输、代表性 RL 轨迹和完整模型训练，均不在本阶段已验证支持范围内。

此前 56.17% 的 allocated 测量和当前容量区间均有明确条件；它们没有证实整卡 VRAM 下降、新融合 Kernel、普遍时延改善或生产可用。原 fused selected-logit 路线对完整 Teacher 契约仍为 NO-GO。

### 复现与证据标识

代码基线为现有[项目仓库](https://github.com/yu830/rl-posttrain-sparse-scoring)的 `06f0abc0945a98fe94d732f4ee0ab875948cdb43`。下列新增脚本位于包含本增补的后续版本，父基线本身不包含它们；复现时须使用相应后续提交或源码包，源码哈希随原始测量记录。主要入口为：

- `scripts/factorial_teacher_probe.py`：每个新进程只执行一个 Dense/Chunked 与 offload 条件。
- `scripts/compare_capacity_outputs.py`：与同 N dense 输出比较全部紧凑候选行和 Teacher Top-K 集合。
- `scripts/worker_to_student_step.py`：真实本地 Worker 输出进入固定上游 loss 和一次受限更新。
- `scripts/build_stage_report_20261030.py`：从两份 Markdown 正文重新生成本报告 PDF。

证据标识包括 `S1_SUMMARY.json`、`capacity_runner.jsonl`、`compare_validated_summary.json`、`compare_on_n9216.json`、`T1_worker_aligned_on/off.json` 与 `worker_to_student_01.json`，分别保留成功、OOM、正确性边界及未运行项目。

### 参考来源

- [verl PR #6920 固定源码](https://github.com/verl-project/verl/tree/87cd055e872f9fdd7b976fcd797a0934b6dba265)。
- [固定 FSDP reverse-KL loss](https://github.com/verl-project/verl/blob/87cd055e872f9fdd7b976fcd797a0934b6dba265/verl/trainer/distillation/fsdp/losses.py)，Git blob `97d3ebf485a5ce9e1b1a435cce427d6d89284dc6`。
- [Qwen2.5-0.5B-Instruct 模型快照](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)。

本增补的新增实验正文等待 Human 审核。旧 c1ab1f4 报告及其审核状态单独保留。
