[English](./README.md) | [简体中文](./README.zh-CN.md)

# 候选集条件教师评分

本项目研究语言模型后训练中的教师评分系统：逐位置候选契约与候选词表投影是否能在保持目标语义的同时改善真实消费者的评分路径。项目目前处于**研究阶段**：尚无本项目优化实现，也没有实测加速结果。

## 行业问题

在线策略蒸馏（On-Policy Distillation，OPD）的消费者可能需要在每个因果位置请求不同 token ID 的教师分数。位置选择、候选选择和概率归一化是三个不同问题。紧凑响应本身不能避免稠密 LM Head 计算；候选集条件概率也不等于全词表概率。真实训练消费者是否存在仍待证明。

## 已有工作

- 在[所核查的 SGLang 版本](https://github.com/sgl-project/sglang/blob/4517a5ff1bbd1f944d6bde1819300ed14af7b698/python/sglang/srt/managers/tokenizer_manager_score_mixin.py)中，`/score` 已支持逐项标签和可选的候选集归一化；[位置选择](https://github.com/sgl-project/sglang/blob/4517a5ff1bbd1f944d6bde1819300ed14af7b698/python/sglang/srt/layers/logits_processor.py)已存在，但使用稠密输出头。
- [SGLang PR #38119](https://github.com/sgl-project/sglang/pull/38119) 在 `sglang-miles` 上实现了全词表归一化的逐位置 ID；2026 年 10 月 29 日核查时仍开放。[vLLM PR #56984](https://github.com/vllm-project/vllm/pull/56984) 于 2026 年 10 月 3 日合并逐行接口。
- [OmniLens](https://arxiv.org/abs/2608.10260v1) 与 [IndexedLogits](https://github.com/pettyjohnjn/indexed_logits/tree/6020cb446cde8cb2b74a8fdf90db905a320936c3) 已有候选 logits 工作；本项目不将该原语称为原创。

## 技术思路

对于隐藏状态行 `h_t` 和候选集 `C_t`，拟议研究契约获取有序 ID 对应的教师 logits，并只在候选集内归一化。语义一致的稠密参考必须保留因果位移、掩码、模型输出头变换与候选顺序。潜在候选投影需与稠密、分块稠密、eager 索引以及已有融合实现比较。

条件反向 KL 忽略支持集总概率和尾部项，因而不同于全词表 KL。候选数为一时，其损失与局部梯度为零。所核查的 [Rethinking OPD 作者代码](https://github.com/Thinking-Space/Rethinking-OPD/tree/ac26e38d6f1572eb027597b48a9f4e01f6915ef8)使用全归一化 log-ratio 奖励和 PPO surrogate；仅条件化的教师分数不能等价替换。真正消费该条件目标的训练端仍未核实。

## 架构

![中文架构图：上游已有路径、拟议组件及已有先例](./docs/architecture/architecture.zh-CN.png)

图中区分**上游已有**、**拟议且尚未实现**和**已有先例**。[可编辑拓扑](./docs/architecture/topology.json)生成中英文两个版本；也可查看[中文 SVG](./docs/architecture/architecture.zh-CN.svg)。

## 复现

以下命令重建文档，不会运行评分服务或基准测试。依赖：XeLaTeX、BibTeX、Poppler（`pdftoppm`、`pdfinfo`、`pdftotext`）、带 Pillow 的 Python 3，以及随仓库提供的有限字符集字体。例如：

```bash
bash scripts/build_assets.sh
python3 scripts/verify_assets.py
```

[构建环境锁定记录](./docs/build-environment.md)列出本次交付所用的版本与哈希。[随附字体子集](./assets/fonts/README.md)已固定版本，采用 OFL 1.1 许可。添加新文字需要额外字形时，可从 [Noto CJK 项目](https://github.com/notofonts/noto-cjk)安装完整字体，再将 `CJK_FONT` 指向其绝对路径。公开的[基准测试方法](./docs/benchmark-methodology.zh-CN.md)说明拟议实验、基线与报告边界。

## 已核实结果与局限

| 项目 | 状态 |
| --- | --- |
| 源码与文献研究 | 报告引用固定快照和分析；未运行复现 |
| CPU 数值或梯度测试 | **尚未执行** |
| GPU 内核或服务基准测试 | **尚未实测** |
| HTTP 评分或真实训练 | **尚未执行** |
| 本项目加速、显存或质量收益 | **尚未证实** |

拟议优化与真实消费者均待核实。报告中的运算量和张量大小示例仅为解析值；第三方测量不属于本项目结果。报告为 AI 辅助初稿，作者审核待完成。

## 报告

[![中文技术报告封面](./assets/report-cover.zh-CN.png)](./docs/reports/technical-report.zh-CN.pdf)

[阅读中文 PDF](./docs/reports/technical-report.zh-CN.pdf) · [中文 TeX](./docs/reports/technical-report.zh-CN.tex) · [共享公式](./docs/reports/common-equations.tex) · [参考文献](./docs/reports/references.bib)
