# G2 技术准入观察

以下新增观察尚未并入作者审核过的报告快照。G3 **尚未准入**；现有 G2 证据对提出的新融合教师评分方案给出 **NO-GO**。对于下述 verl 消费者的完整契约，单纯返回候选 logits **不能作为直接替换方案（NO-GO）**。

## 后续：完整教师契约的输出头回放

本次有界回放比较固定版本 #6920 的教师输出头（A0）与在同一稠密输出头之前按 token 行分块（A1）。输入是真实固定 Qwen 模型的最终归一化 hidden states 和权重，但文本是合成测试输入，不是实际 RL 轨迹。计时不含 Transformer、FSDP、传输或训练。两条路径均返回全词表归一化的候选 logprob 和教师自身 Top-K；训练器随后把教师 logprob 转为 FP32，并在下游计算 teacher mass、overlap 和带截断的损失。固定源码中学生温度 tensor 会转成 logits dtype 后再除，教师 worker 实际使用原始 logits（温度为 1）。

在 RTX 5070 Laptop 8 GiB 上，BF16 的 N=32/128/512/1024、K=8/32、行块大小 64/256 均通过记录的 A1 输出、质量、重叠和损失检查。完整学生输出头/logits 反向检查每个单元只覆盖前 32 行，并非完整训练步骤。N=1024、K=32、行块 256 的同组 A0/A1 主机同步计时中位数为 26.869/27.045 ms；PyTorch allocated 绝对峰值为 2501.4/1387.8 MiB（包含常驻 tensor），增量峰值为 1484.82/371.25 MiB。这些是本测试输入的教师输出头结果，不是端到端吞吐。

隔离的 A2 使用 #7809 未修改的 Triton 候选 logprob 算子，再增加一次按行分块的稠密投影以取得教师 Top-K。在相同 BF16 单元中，虽然候选 logprob 达到逐元素预算，学生 logits 梯度相对 L2 误差在全部 8 个单元超过冻结的 0.05 门限，因此不是正确的直接替换。FP32 的 A2 检查均通过；N=1024、K=32 直接配对 A1/64 与 A2 的主机时间中位数为 58.654/161.552 ms。这些结果尚未确立新的优化实现。

最终每个单元均包含 A/A 和交错 ABBA；主机时间在不同 block 间有变化，不能把微小差异当作普遍加速。较早的 BF16 运行曾出现 CUDA Event 时间超过包围它的同步主机时间；修正输入后的运行未复现，原因仍未知。Reserved 显存受前序样本的 allocator 缓存影响；WSL 未提供进程级显存占用。现阶段对进入 G3 的结论为 **NO-GO**；没有声称完整 FSDP 教师或 RL 训练收益。

复现时使用下方环境与固定的 MODEL_DIR，将 `WORK_DIR` 和 `A2_DIR` 指向可写临时目录。加载器会检查两个上游 Git blob，拒绝不匹配源码，也不把稠密回退冒充 Triton：

```bash
mkdir -p "$WORK_DIR" "$A2_DIR"
curl -fsSL https://raw.githubusercontent.com/verl-project/verl/522514fe21300c4cf7b3800fd6cd827eb1caa2c9/verl/utils/kernel/linear_topk_log_probs.py -o "$A2_DIR/linear_topk_log_probs.py"
curl -fsSL https://raw.githubusercontent.com/verl-project/verl/522514fe21300c4cf7b3800fd6cd827eb1caa2c9/verl/utils/kernel/topk_log_probs_kernels.py -o "$A2_DIR/topk_log_probs_kernels.py"
python -m pytest -q tests/test_scoring_reference.py tests/test_teacher_stage_benchmark.py tests/test_teacher_stage_attribution.py
python scripts/teacher_stage_benchmark.py --model-dir "$MODEL_DIR" --dtype bfloat16 --a2-upstream-dir "$A2_DIR" --out "$WORK_DIR/teacher-bf16.json" --profile
python scripts/teacher_stage_benchmark.py --model-dir "$MODEL_DIR" --dtype float32 --a2-upstream-dir "$A2_DIR" --out "$WORK_DIR/teacher-fp32.json" --profile
python scripts/diagnose_teacher_stage_attribution.py --model-dir "$MODEL_DIR" --out "$WORK_DIR/hf-stage-attribution.json"
```

BF16 对照因 A2 超出冻结梯度预算而以非零状态退出，这是需要保留的负结果，不应放宽门限。阶段诊断脚本另行重放此前的固定输入：HF 单条与批处理的差异在第一 decoder 层后已出现；原 SGLang score 请求的 hidden 未直接捕获。[基线脚本](../scripts/teacher_stage_benchmark.py)和[阶段诊断脚本](../scripts/diagnose_teacher_stage_attribution.py)记录了完整范围与控制条件。

## 消费者和语义

在 [verl main 2ab3529](https://github.com/verl-project/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/trainer/distillation/losses.py) 中，所核查 Top-K 路径使用教师选择的支持集及前向 KL 或尾部感知前向 KL；主线教师管理器接收 token 序列，不接收学生候选 ID。

开放的 [verl PR #6920](https://github.com/verl-project/verl/pull/6920)，head 为 [87cd055e](https://github.com/verl-project/verl/tree/87cd055e872f9fdd7b976fcd797a0934b6dba265)，实际实现了学生 Top-K 条件反向 KL。提交者报告执行过端到端训练，本项目**尚未独立复现**。[训练器](https://github.com/verl-project/verl/blob/87cd055e872f9fdd7b976fcd797a0934b6dba265/verl/trainer/ppo/ray_trainer.py)在 rollout 后通过无梯度学生预计算取得候选 ID，向冻结的 FSDP 教师发送对齐 ID，并把返回值写入更新批次。[教师 worker](https://github.com/verl-project/verl/blob/87cd055e872f9fdd7b976fcd797a0934b6dba265/verl/workers/teacher/fsdp_teacher_worker.py)要求 ID 对齐到因果 logits 行，输出全词表归一化的候选 logprob 和教师 Top-K ID。未来服务如何绑定策略版本以及处理异步更新仍需验证。

该 [loss](https://github.com/verl-project/verl/blob/87cd055e872f9fdd7b976fcd797a0934b6dba265/verl/trainer/distillation/fsdp/losses.py)先将双方概率重归一化到学生支持集。因此，支持集、顺序和温度相同时，候选集条件教师概率足以计算这个**损失项**；但仅凭选中 logits 无法还原原有教师全词表概率质量或教师自身 Top-K 诊断，它们依赖未选 logits。该 PR 还可在归一化后截断 logprob；原条件评分参考不包含截断损失；上述教师输出头回放也检查了截断，但不宣称完整 FSDP 集成。本项目尚未证明与该 PR 的完整契约等价。

另一个开放的 [verl PR #7806](https://github.com/verl-project/verl/pull/7806) 使用 rollout Top-K 和不同的截断／全归一化目标，并报告过多 GPU 实验；这些均是第三方观察。[Rethinking OPD 作者代码](https://github.com/Thinking-Space/Rethinking-OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/verl/verl/workers/actor/dp_actor.py)使用全归一化 log-ratio 奖励和 PPO surrogate，不能悄悄用条件教师分数替换该目标。

## 现有评分和实现

SGLang [46caa545](https://github.com/sgl-project/sglang/tree/46caa54562a97e71d9b937c62875c90b39274b0d) 的 /v1/score 已支持每个 item 的候选列表和可选归一化。[PR #38119](https://github.com/sgl-project/sglang/pull/38119) 与 [vLLM PR #56984](https://github.com/vllm-project/vllm/pull/56984) 已提供逐位置接口先例。[vLLM 固定版本文档](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/docs/training/prompt_token_id_logprobs.md)说明 next-token 位移和逐行全归一化 logprob。[IndexedLogits](https://github.com/pettyjohnjn/indexed_logits/tree/6020cb446cde8cb2b74a8fdf90db905a320936c3) 已有候选 logit CUDA 原语，其核查文件未声明许可证；本项目没有复制第三方内核。

开放的 [verl PR #7809](https://github.com/verl-project/verl/pull/7809)，head 为 [522514fe](https://github.com/verl-project/verl/blob/522514fe21300c4cf7b3800fd6cd827eb1caa2c9/verl/utils/kernel/linear_topk_log_probs.py)，已在 Megatron 学生侧实现词表分块、全词表归一化的候选 logprob，无需物化完整学生 logits，并包含反向路径。其报告的 GPU 验证属于第三方证据，不是本项目结果。任何词表流式输出头方案都必须对照这个先例；学生侧契约并不等于教师概率质量和教师自身 Top-K 诊断。

## 实际执行的观察

| 观察 | 结果及范围 |
| --- | --- |
| CPU/GPU 合成参考 | 覆盖固定支持集、FP64/FP32/FP16/BF16、损失／梯度及失败输入；见下方命令和结果记录。这不是优化内核。 |
| 本地真实服务 | RTX 5070 Laptop 上以 BF16 运行 Qwen2.5-0.5B-Instruct；一次 /v1/score 请求包含 8 个合成因果前缀，每行 4 个候选 ID，返回 HTTP 200。候选条件分数与返回的全词表 logprob 再归一化结果一致。 |
| 独立真实模型对照 | **未通过**预先冻结的 BF16 逐元素容差 atol=0.1、rtol=0.01：32 个 SGLang 全 logprob 中有 2 个超出 Transformers BF16 批处理 eager 前向的阈值，最大绝对差为 0.37914085。没有放宽容差。 |
| 精度诊断 | 同一 Transformers BF16 模型的单条和批处理前向最大差为 0.24671745；SGLang BF16 与 Transformers FP32 单条前向最大差为 0.15135193。这些观察提示执行形状及精度值得排查，但不能证明根因或构成正确性通过。 |
| 剖析 | 单独执行一次合成回放并清空 radix cache，观察到两次 [4,896] × [896,151936] BF16 稠密输出头矩阵乘，对应 GPU 内核约 993.408 和 994.713 微秒。它只是轨迹观察，不是 A/A、A/B、关键路径归因或预期加速。 |
| 训练与性能 | 在这一早期阶段，尚未建立本地真实训练集成、A/A、A/B、代表性负载覆盖、RL 单步计时或项目加速结论。 |

[评分记录](../results/sglang-teacher-smoke-20261029.json)包含实际输入、响应和时间；[未通过的对照](../results/teacher-hf-comparison-20261029.json)、[诊断](../results/teacher-precision-diagnostic-20261029.json)、[合成数值记录](../results/g2-reference-20261029.json)和[轨迹摘要](../results/sglang-profile-summary-20261029.json)分别保留各自范围。原始轨迹和未脱敏失败日志保存在私有项目证据区。单次受冷启动和预热影响的请求耗时不是性能基准。

## FP32 精度对照

下列实验重放了相同的 8 个前缀和候选 ID，使用同一固定模型。除第一项启动失败外，FP32 行均将 SGLang 融合算子强制为 PyTorch 回退。TF32 和 IEEE 分别使用全新的独立 Triton 编译缓存；对照采用原先冻结的 FP32 容差 atol=5e-5、rtol=2e-5，未修改阈值。

| 配置 | 结果 | 最大绝对 logprob 误差 | 超出容差 |
| --- | --- | ---: | ---: |
| FP32，默认融合算子 | 服务启动失败：已安装的 RMSNorm 内核不支持 Float | — | — |
| FP32，Triton 注意力，默认精度 | 未通过 | 0.00966835 | 30 / 32 |
| FP32，Triton 注意力，显式 TF32 | 未通过 | 0.00966835 | 30 / 32 |
| FP32，Triton 注意力，显式 IEEE | 通过 | 0.00001717 | 0 / 32 |
| FP32，PyTorch 原生注意力 | 通过 | 0.00002670 | 0 / 32 |
| BF16，原生注意力、默认融合算子 | 未通过 | 0.30104256 | 3 / 32 |
| BF16，原生注意力及融合算子 | 未通过 | 0.45039749 | 5 / 32 |

在这 32 个数值上，默认与显式 TF32 的 SGLang 输出相同；显式 TF32 与 IEEE 最大差为 0.00967407。这些对照支持本次 FP32 差异与 Triton FP32 dot 精度模式有关，但没有定位到唯一内核，也没有解释 BF16 的失败。TRITON_F32_DEFAULT 是服务进程内所有符合条件的 Triton FP32 dot 的全局设置。固定版本的 [SGLang extend-attention 源码](https://github.com/sgl-project/sglang/blob/46caa54562a97e71d9b937c62875c90b39274b0d/python/sglang/kernels/ops/attention/extend_attention.py#L795)中有未显式指定输入精度的 dot；[Triton 文档](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)说明了 NVIDIA 上的 FP32 默认值。这是有范围的数值诊断，不是生产补丁或性能结果。较早对照记录中名为 tf32 的字段仅指独立 PyTorch 参考，不代表 SGLang Triton 设置。

[精度对照记录](../results/precision-controls-20261029.json)由[汇总脚本](../scripts/summarize_precision_controls.py)从原始 JSON 生成；其中省略含本机路径的 traceback，源文件哈希对应私有原件。完整输入、输出、配置和逐元素结果仍可查看。这些较早的精度对照本身没有使 G3 准入；后续教师输出头回放见上文。

## 复现

使用 Python 3.12 和记录的[依赖版本](../requirements-g2.lock)。该列表是平台相关的 uv pip freeze 快照，不含 wheel 哈希，并非跨平台锁文件。SGLang 单独使用固定提交 46caa54562a97e71d9b937c62875c90b39274b0d 安装，设置 SGLANG_BUILD_RUST_EXTS=none；本次构建未编译可选 Rust 扩展。从 Qwen/Qwen2.5-0.5B-Instruct（Apache-2.0）同一修订 7ae557604adf67be50417f59c2c2f167def9a775 下载模型和分词器。实测 model.safetensors SHA-256 为 fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe；tokenizer.json SHA-256 为 c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539。

在新的 Linux 环境中，将 SGLANG_SRC 指向新的空检出目录、MODEL_DIR 指向新的模型目录，然后执行：

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install uv==0.12.3
    uv pip install -r requirements-g2.lock
    git init "$SGLANG_SRC"
    git -C "$SGLANG_SRC" remote add origin https://github.com/sgl-project/sglang.git
    git -C "$SGLANG_SRC" fetch --depth=1 origin 46caa54562a97e71d9b937c62875c90b39274b0d
    git -C "$SGLANG_SRC" checkout --detach FETCH_HEAD
    SGLANG_BUILD_RUST_EXTS=none uv pip install --no-deps -e "$SGLANG_SRC/python"
    python -c 'import os; from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775", local_dir=os.environ["MODEL_DIR"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])'

这些安装命令描述记录的环境；尚未独立重测从快照开始的全新安装。

在仓库根目录激活虚拟环境，并让 MODEL_DIR 指向固定快照：

    python -m pytest -q tests/test_scoring_reference.py
    python scripts/measure_reference.py --out results/g2-reference-20261029.json
    python scripts/sglang_teacher_smoke.py --model-dir "$MODEL_DIR" --out results/sglang-teacher-smoke-20261029.json --server-log /tmp/g2-sglang.log
    python scripts/verify_teacher_hf.py --model-dir "$MODEL_DIR" --smoke results/sglang-teacher-smoke-20261029.json --out results/teacher-hf-comparison-20261029.json

在本次环境下，最后一条命令预期复现已记录的失败，不能解释为测试通过。精度诊断脚本记录单条和 FP32 补充对照；评分脚本的可选 profile 目录可执行单独回放，轨迹汇总脚本接收轨迹和词表大小。请使用独立输出目录保留历史记录。


复现成对的 FP32 精度对照时，先将 IEEE_CACHE 与 TF32_CACHE 指向两个新的、不同的目录：

    python scripts/sglang_teacher_smoke.py --dtype float32 --force-native-fused-ops --attention-backend triton --triton-f32-default ieee --triton-cache-dir "$IEEE_CACHE" --model-dir "$MODEL_DIR" --out results/sglang-teacher-fp32-triton-ieee-20261029.json --server-log /tmp/g2-ieee.log
    python scripts/verify_teacher_hf.py --dtype float32 --model-dir "$MODEL_DIR" --smoke results/sglang-teacher-fp32-triton-ieee-20261029.json --out results/teacher-hf-fp32-triton-ieee-20261029.json
    python scripts/sglang_teacher_smoke.py --dtype float32 --force-native-fused-ops --attention-backend triton --triton-f32-default tf32 --triton-cache-dir "$TF32_CACHE" --model-dir "$MODEL_DIR" --out results/sglang-teacher-fp32-triton-tf32-20261029.json --server-log /tmp/g2-tf32.log
    python scripts/verify_teacher_hf.py --dtype float32 --model-dir "$MODEL_DIR" --smoke results/sglang-teacher-fp32-triton-tf32-20261029.json --out results/teacher-hf-fp32-triton-tf32-20261029.json

本次 TF32 对照未通过记录的容差；这两项都不作为性能计时。
