[English](./README.md) | [简体中文](./README.zh-CN.md)

# Candidate-Conditioned Teacher Scoring

A systems research project on teacher scoring for language-model post-training. It asks whether an explicit per-position candidate contract and a selected vocabulary projection can improve a real consumer's scoring path while preserving the intended objective. The project is in the **research stage**: there is no project optimization implementation or measured speedup.

## Industrial problem

An on-policy distillation (OPD) consumer may need teacher scores for a different set of token IDs at each causal position. Position selection, candidate selection, and probability normalization are separate problems. A compact response does not by itself avoid dense LM-head computation, and candidate-conditioned probabilities are not full-vocabulary probabilities. An open verl training implementation now provides a concrete conditional-loss consumer; its complete diagnostics still require full-vocabulary information.

## Prior art

- At [the audited SGLang revision](https://github.com/sgl-project/sglang/blob/4517a5ff1bbd1f944d6bde1819300ed14af7b698/python/sglang/srt/managers/tokenizer_manager_score_mixin.py), `/score` already supports per-item labels and optional candidate normalization; [position selection](https://github.com/sgl-project/sglang/blob/4517a5ff1bbd1f944d6bde1819300ed14af7b698/python/sglang/srt/layers/logits_processor.py) exists with dense readout.
- [SGLang PR #38119](https://github.com/sgl-project/sglang/pull/38119) implements per-position IDs on `sglang-miles` with full normalization; it was open when checked on 29 October 2026. [vLLM PR #56984](https://github.com/vllm-project/vllm/pull/56984) merged a per-row interface on 3 October 2026.
- [OmniLens](https://arxiv.org/abs/2608.10260v1) and [IndexedLogits](https://github.com/pettyjohnjn/indexed_logits/tree/6020cb446cde8cb2b74a8fdf90db905a320936c3) provide prior selected-logit work. This project does not claim that primitive as novel.
- Open [verl PR #7809](https://github.com/verl-project/verl/pull/7809) already implements a vocabulary-tiled, full-normalized selected-logprob path for Megatron student distillation. Its GPU results are third-party claims, not project results.

## Technical approach

For hidden row `h_t` and candidate set `C_t`, the proposed research contract obtains teacher logits for the ordered IDs and normalizes only over that set. A same-semantics dense reference must preserve causal shifts, masks, model-specific head transforms and candidate order. A potential selected projection must be compared with dense, chunked-dense, eager indexed and existing fused baselines.

Conditional reverse KL differs from full-vocabulary KL because it omits support mass and tail contributions. With one candidate, its loss and local gradient are zero. The inspected [Rethinking OPD author code](https://github.com/Thinking-Space/Rethinking-OPD/tree/ac26e38d6f1572eb027597b48a9f4e01f6915ef8) uses full-normalized log-ratio rewards and a PPO surrogate; conditional-only teacher scores are not a drop-in equivalent. Open [verl PR #6920](https://github.com/verl-project/verl/pull/6920) implements student-top-K conditional reverse KL, while also calculating teacher mass and teacher top-K diagnostics that selected logits alone cannot reproduce. Its reported training run has not been independently reproduced here.

## Architecture

![English architecture: existing upstream paths, proposed components and prior art](./docs/architecture/architecture.en.png)

The diagram distinguishes **existing upstream**, **proposed and unimplemented**, and **prior art**. [Editable topology](./docs/architecture/topology.json) generates both language editions; [English SVG](./docs/architecture/architecture.en.svg) is also available.

## Reproduction

These commands rebuild the documentation, not a scoring service or benchmark. Requirements: XeLaTeX, BibTeX, Poppler (`pdftoppm`, `pdfinfo`, `pdftotext`), Python 3 with Pillow, and the bundled limited-glyph font. For example:

```bash
bash scripts/build_assets.sh
python3 scripts/verify_assets.py
```

The [build environment lock](./docs/build-environment.md) records the documentation build versions and hashes. The [bundled font subset](./assets/fonts/README.md) is pinned and licensed under OFL 1.1. For new text requiring additional glyphs, install a full font from the [Noto CJK project](https://github.com/notofonts/noto-cjk) and set `CJK_FONT` to its absolute path. The public [benchmark methodology](./docs/benchmark-methodology.en.md) defines proposed experiments, baselines and reporting boundaries.

The independent [G2 reference, real teacher-scoring commands and observations](./docs/g2-results.en.md) document the separate Python environment and fixed model. These checks are not an optimized serving implementation.

## Verified results and limitations

| Item | Status |
| --- | --- |
| Synthetic numerical and gradient reference | CPU and GPU checks passed on the recorded fixtures |
| SGLang teacher scoring | One real BF16 HTTP smoke request passed using synthetic candidate IDs |
| Real-model numerical comparison | **FAILED:** 2 of 32 values exceeded the frozen BF16 budget |
| Representative performance and real training | **NOT ESTABLISHED**; one separate trace is exploratory |
| Project speedup, memory or quality gain | **NOT ESTABLISHED** |

G3 remains **HOLD / NEEDS VERIFICATION**. Arithmetic or tensor-size examples in the report are analytical only; third-party measurements are not project results. The author reviewed both AI-assisted report editions at commit `c1ab1f4308c2ac1aecf47e222f210a16d0a5a898`; later substantive technical revisions require a new review. The reports describe that reviewed snapshot, while the new G2 observations are recorded separately and have not been author-reviewed.

## Report

[![English technical report cover](./assets/report-cover.en.png)](./docs/reports/technical-report.en.pdf)

[Read the English PDF](./docs/reports/technical-report.en.pdf) · [English TeX](./docs/reports/technical-report.en.tex) · [Shared equations](./docs/reports/common-equations.tex) · [Bibliography](./docs/reports/references.bib)
