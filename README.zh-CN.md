[English](./README.md) | [简体中文](./README.zh-CN.md)

# 候选集条件教师评分

本项目研究语言模型后训练中的教师评分系统。当前方向是**显存预算约束的 Teacher Scoring**：在单 GPU Qwen2/FSDP 路径中将行分块前移到 LM Head 之前，同时保留全词表归一化候选 logprob 和教师 Top-K。该路径仍处于实验阶段；多卡行为、生产稳定性和完整 RL 训练尚未确立。

## 当前实现与测量

[显存预算约束集成及完整前向结果](./docs/memory-bounded-teacher.zh-CN.md)说明固定 verl 补丁、已测 FSDP2/CPU offload 边界、预算局限、负结果与复现命令。此前仅用融合候选 logits 替换完整教师契约的方案未通过，新路径不改变这一结论。

## 行业问题

在线策略蒸馏（On-Policy Distillation，OPD）的消费者可能需要在每个因果位置请求不同 token ID 的教师分数。位置选择、候选选择和概率归一化是三个不同问题。紧凑响应本身不能避免稠密 LM Head 计算；候选集条件概率也不等于全词表概率。一个开放的 verl 训练实现已提供具体的条件损失消费者，但其完整诊断仍需全词表信息。

## 已有工作

- 在[所核查的 SGLang 版本](https://github.com/sgl-project/sglang/blob/4517a5ff1bbd1f944d6bde1819300ed14af7b698/python/sglang/srt/managers/tokenizer_manager_score_mixin.py)中，`/score` 已支持逐项标签和可选的候选集归一化；[位置选择](https://github.com/sgl-project/sglang/blob/4517a5ff1bbd1f944d6bde1819300ed14af7b698/python/sglang/srt/layers/logits_processor.py)已存在，但使用稠密输出头。
- [SGLang PR #38119](https://github.com/sgl-project/sglang/pull/38119) 在 `sglang-miles` 上实现了全词表归一化的逐位置 ID；2026 年 10 月 29 日核查时仍开放。[vLLM PR #56984](https://github.com/vllm-project/vllm/pull/56984) 于 2026 年 10 月 3 日合并逐行接口。
- [OmniLens](https://arxiv.org/abs/2608.10260v1) 与 [IndexedLogits](https://github.com/pettyjohnjn/indexed_logits/tree/6020cb446cde8cb2b74a8fdf90db905a320936c3) 已有候选 logits 工作；本项目不将该原语称为原创。
- 开放的 [verl PR #7809](https://github.com/verl-project/verl/pull/7809) 已为 Megatron 学生蒸馏实现词表分块、全归一化的候选 logprob 路径；其 GPU 结果属于第三方声明，不是本项目结果。

## 早期研究思路

对于隐藏状态行 `h_t` 和候选集 `C_t`，拟议研究契约获取有序 ID 对应的教师 logits，并只在候选集内归一化。语义一致的稠密参考必须保留因果位移、掩码、模型输出头变换与候选顺序。潜在候选投影需与稠密、分块稠密、eager 索引以及已有融合实现比较。

条件反向 KL 忽略支持集总概率和尾部项，因而不同于全词表 KL。候选数为一时，其损失与局部梯度为零。所核查的 [Rethinking OPD 作者代码](https://github.com/Thinking-Space/Rethinking-OPD/tree/ac26e38d6f1572eb027597b48a9f4e01f6915ef8)使用全归一化 log-ratio 奖励和 PPO surrogate；仅条件化的教师分数不能等价替换。开放的 [verl PR #6920](https://github.com/verl-project/verl/pull/6920)实现了学生 Top-K 条件反向 KL，但也计算仅凭候选 logits 无法恢复的教师概率质量及教师 Top-K 诊断。该 PR 报告的训练运行尚未在本项目独立复现。

## 架构

![当前显存受控 Teacher Scoring 架构](./docs/architecture/memory-bounded.zh-CN.png)

当前架构图展示单 GPU FSDP2 执行边界、输出头前的行分块与稠密回退，以及 response mask 的消费边界。[可编辑的双语拓扑](./docs/architecture/memory-bounded-topology.json)与[中文 SVG](./docs/architecture/memory-bounded.zh-CN.svg)对应同一张图；不支持或未验证的路径均明确标注。

### 早期研究快照

![中文架构图：上游已有路径、拟议组件及已有先例](./docs/architecture/architecture.zh-CN.png)

这张历史图区分早期研究快照中的**上游已有**、**拟议且尚未实现**和**已有先例**。[可编辑拓扑](./docs/architecture/topology.json)生成中英文两个版本；也可查看[中文 SVG](./docs/architecture/architecture.zh-CN.svg)。

## 复现

[实验版 Worker 复现说明](./docs/experimental-reproduction.md)列出固定环境、长序列 Dense/Chunked 对照、无人工 fraction 的 ABBA 对照和 response mask 测试。

以下命令重建文档，不会运行评分服务或基准测试。依赖：XeLaTeX、BibTeX、Poppler（`pdftoppm`、`pdfinfo`、`pdftotext`）、带 Pillow 的 Python 3，以及随仓库提供的有限字符集字体。例如：

```bash
bash scripts/build_assets.sh
python3 scripts/verify_assets.py
```

[构建环境锁定记录](./docs/build-environment.md)列出文档构建所用的版本与哈希。[随附字体子集](./assets/fonts/README.md)已固定版本，采用 OFL 1.1 许可。添加新文字需要额外字形时，可从 [Noto CJK 项目](https://github.com/notofonts/noto-cjk)安装完整字体，再将 `CJK_FONT` 指向其绝对路径。公开的[基准测试方法](./docs/benchmark-methodology.zh-CN.md)说明拟议实验、基线与报告边界。

独立的 [G2 参考、真实教师评分命令与观察](./docs/g2-results.zh-CN.md)记录了单独的 Python 环境和固定模型。这些检查不是优化后的服务实现。

## 历史 G2 结果与局限

| 项目 | 状态 |
| --- | --- |
| 合成数值与梯度参考 | 记录的测试输入在 CPU 和 GPU 上通过 |
| SGLang 教师评分 | 使用合成候选 ID 的一次真实 BF16 HTTP 冒烟请求通过 |
| 真实模型数值对照 | **未通过：**32 个数值中有 2 个超过冻结的 BF16 容差 |
| FP32 精度诊断 | IEEE Triton 与原生注意力通过；TF32 Triton 超出固定容差 |
| 代表性性能与真实训练 | **尚未证实**；一次单独轨迹仅用于探索 |
| 本项目加速、显存或质量收益 | **尚未证实** |

此前 G3 融合方案对完整教师契约仍为 **NO-GO**；当前显存受控路径单独评估。报告中的运算量和张量大小示例仅为解析值；第三方测量不属于本项目结果。作者已审核提交 `c1ab1f4308c2ac1aecf47e222f210a16d0a5a898` 的两份 AI 辅助报告；后续实质性技术修订须重新审核。报告描述已审核快照，新增 G2 观察另行记录，尚未经过作者审核。

## 报告

[显存受控实验技术报告（PDF）](./docs/reports/memory-bounded-experimental.zh-CN.pdf) · [可编辑 LaTeX](./docs/reports/memory-bounded-experimental.zh-CN.tex)。本版由 AI 辅助整理，新增结果等待 Human 审核。

下方旧报告保留原已审核研究快照：

[![中文技术报告封面](./assets/report-cover.zh-CN.png)](./docs/reports/technical-report.zh-CN.pdf)

[阅读中文 PDF](./docs/reports/technical-report.zh-CN.pdf) · [中文 TeX](./docs/reports/technical-report.zh-CN.tex) · [共享公式](./docs/reports/common-equations.tex) · [参考文献](./docs/reports/references.bib)
