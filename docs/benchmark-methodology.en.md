[English](./benchmark-methodology.en.md) | [简体中文](./benchmark-methodology.zh-CN.md)

# Benchmark methodology

No experiments below have been run. This document describes a reproducible future comparison; it is not evidence of performance or training benefit.

1. Define one real consumer's loss, candidate-generation timing, policy/model/tokenizer revisions, positions, masks and output normalization. Preserve the next-token shift. Conditional and full-vocabulary log-probabilities are different contracts.
2. Compare the same conditional target using dense projection plus gather, chunked dense, eager indexed gather and the existing IndexedLogits baseline when compatible. Native full-normalized scoring is context, not a same-target speed ratio. Include candidate movement, validation and fallback work at the corresponding timing boundary.
3. Establish CPU FP64 reference values and student gradients. Check singleton and full-vocabulary supports, permutations, duplicate/invalid IDs, ragged masks, omitted logits and causal alignment. Set precision-specific tolerances before comparison, and record unsupported model heads.
4. Freeze code, environment, model, tokenizer, hardware, traces, concurrency and workload dimensions. Separate tuning from evaluation. Report excluded or OOM cells and fallback counts. Use real candidate traces for service claims and distinguish teacher replay from real training.
5. Measure an A/A control before randomized paired A/B blocks. Report noise, p50/p95 request latency, kernel latency, throughput, allocated/reserved memory, payload bytes, numerical error, failures and negative results. Profile separately from headline measurements. If controls are unstable, the comparison is inconclusive.

See the [English report](./reports/technical-report.en.pdf) for definitions, analytical examples and limitations.
