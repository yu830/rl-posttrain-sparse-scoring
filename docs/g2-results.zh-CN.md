# G2 技术准入观察

以下新增观察尚未并入作者审核过的报告快照。G3 的结论为 **HOLD / NEEDS VERIFICATION**。对于下述 verl 消费者的完整契约，单纯返回候选 logits **不能作为直接替换方案（NO-GO）**。

## 消费者和语义

在 [verl main 2ab3529](https://github.com/verl-project/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/trainer/distillation/losses.py) 中，所核查 Top-K 路径使用教师选择的支持集及前向 KL 或尾部感知前向 KL；主线教师管理器接收 token 序列，不接收学生候选 ID。

开放的 [verl PR #6920](https://github.com/verl-project/verl/pull/6920)，head 为 [87cd055e](https://github.com/verl-project/verl/tree/87cd055e872f9fdd7b976fcd797a0934b6dba265)，实际实现了学生 Top-K 条件反向 KL。提交者报告执行过端到端训练，本项目**尚未独立复现**。[训练器](https://github.com/verl-project/verl/blob/87cd055e872f9fdd7b976fcd797a0934b6dba265/verl/trainer/ppo/ray_trainer.py)在 rollout 后通过无梯度学生预计算取得候选 ID，向冻结的 FSDP 教师发送对齐 ID，并把返回值写入更新批次。[教师 worker](https://github.com/verl-project/verl/blob/87cd055e872f9fdd7b976fcd797a0934b6dba265/verl/workers/teacher/fsdp_teacher_worker.py)要求 ID 对齐到因果 logits 行，输出全词表归一化的候选 logprob 和教师 Top-K ID。未来服务如何绑定策略版本以及处理异步更新仍需验证。

该 [loss](https://github.com/verl-project/verl/blob/87cd055e872f9fdd7b976fcd797a0934b6dba265/verl/trainer/distillation/fsdp/losses.py)先将双方概率重归一化到学生支持集。因此，支持集、顺序和温度相同时，候选集条件教师概率足以计算这个**损失项**；但仅凭选中 logits 无法还原原有教师全词表概率质量或教师自身 Top-K 诊断，它们依赖未选 logits。该 PR 还可在归一化后截断 logprob；本参考实现不包含截断损失，测试展示截断可以改变损失值。本项目尚未证明与该 PR 的完整契约等价。

另一个开放的 [verl PR #7806](https://github.com/verl-project/verl/pull/7806) 使用 rollout Top-K 和不同的截断／全归一化目标，并报告过多 GPU 实验；这些均是第三方观察。[Rethinking OPD 作者代码](https://github.com/Thinking-Space/Rethinking-OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/verl/verl/workers/actor/dp_actor.py)使用全归一化 log-ratio 奖励和 PPO surrogate，不能悄悄用条件教师分数替换该目标。

## 现有评分和实现

SGLang [46caa545](https://github.com/sgl-project/sglang/tree/46caa54562a97e71d9b937c62875c90b39274b0d) 的 /v1/score 已支持每个 item 的候选列表和可选归一化。[PR #38119](https://github.com/sgl-project/sglang/pull/38119) 与 [vLLM PR #56984](https://github.com/vllm-project/vllm/pull/56984) 已提供逐位置接口先例。[vLLM 固定版本文档](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/docs/training/prompt_token_id_logprobs.md)说明 next-token 位移和逐行全归一化 logprob。[IndexedLogits](https://github.com/pettyjohnjn/indexed_logits/tree/6020cb446cde8cb2b74a8fdf90db905a320936c3) 已有候选 logit CUDA 原语，其核查文件未声明许可证；本项目没有复制第三方内核。

## 实际执行的观察

| 观察 | 结果及范围 |
| --- | --- |
| CPU/GPU 合成参考 | 覆盖固定支持集、FP64/FP32/FP16/BF16、损失／梯度及失败输入；见下方命令和结果记录。这不是优化内核。 |
| 本地真实服务 | RTX 5070 Laptop 上以 BF16 运行 Qwen2.5-0.5B-Instruct；一次 /v1/score 请求包含 8 个合成因果前缀，每行 4 个候选 ID，返回 HTTP 200。候选条件分数与返回的全词表 logprob 再归一化结果一致。 |
| 独立真实模型对照 | **未通过**预先冻结的 BF16 逐元素容差 atol=0.1、rtol=0.01：32 个 SGLang 全 logprob 中有 2 个超出 Transformers BF16 批处理 eager 前向的阈值，最大绝对差为 0.37914085。没有放宽容差。 |
| 精度诊断 | 同一 Transformers BF16 模型的单条和批处理前向最大差为 0.24671745；SGLang BF16 与 Transformers FP32 单条前向最大差为 0.15135193。这些观察提示执行形状及精度值得排查，但不能证明根因或构成正确性通过。 |
| 剖析 | 单独执行一次合成回放并清空 radix cache，观察到两次 [4,896] × [896,151936] BF16 稠密输出头矩阵乘，对应 GPU 内核约 993.408 和 994.713 微秒。它只是轨迹观察，不是 A/A、A/B、关键路径归因或预期加速。 |
| 训练与性能 | 尚未建立本地真实训练集成、A/A 噪声门槛、A/B 对比、代表性负载覆盖、RL 单步计时或项目加速结论。 |

[评分记录](../results/sglang-teacher-smoke-20261029.json)包含实际输入、响应和时间；[未通过的对照](../results/teacher-hf-comparison-20261029.json)、[诊断](../results/teacher-precision-diagnostic-20261029.json)、[合成数值记录](../results/g2-reference-20261029.json)和[轨迹摘要](../results/sglang-profile-summary-20261029.json)分别保留各自范围。原始轨迹和未脱敏失败日志保存在私有项目证据区。单次受冷启动和预热影响的请求耗时不是性能基准。

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
