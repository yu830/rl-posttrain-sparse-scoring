---
title: Memory-Bounded Teacher Scoring
subtitle: Controlled stability, allocator-bounded capacity and direct Worker-to-Student validation
author: Jiacheng Yu
edition: 30 October 2026 | English
footer: Engineering supplement
scope: Single-GPU Qwen2.5-0.5B experiments on an RTX 5070 Laptop, with fixed verl source and explicitly bounded correctness claims.
---
## 1. Question and execution contract

A Teacher may return only compact scores while still allocating a full token-by-vocabulary logits tensor. This project moves row chunking before the dense LM Head and retains two required outputs: full-vocabulary-normalized log probabilities at Student candidate IDs, and the Teacher's own Top-K IDs. The earlier selected-logit fusion proposal remains NO-GO for this complete contract. Row chunking itself is established prior art; the engineering contribution is its integration with the fixed Teacher execution and consumer interfaces.

The present stage asks whether the initial CUDA failure recurs under controlled conditions, whether chunking permits longer inputs under a stated allocator limit, and whether actual Worker outputs preserve a real Student update. Direct model/FSDP experiments and actual verl Worker execution are identified separately throughout.

Table 1. Fixed execution conditions.

| Item | Fixed condition |
| --- | --- |
| Hardware | NVIDIA RTX 5070 Laptop GPU, 8151 MiB |
| Model | Qwen2.5-0.5B-Instruct, BF16, tied embeddings |
| Model revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Frameworks | PyTorch 2.14.1+cu130; Transformers 5.19.0 |
| verl source | PR #6920 at `87cd055e872f9fdd7b976fcd797a0934b6dba265` |
| Support | World size one, no KV cache, padded text inputs |

### 2. Controlled stability (S1)

The direct-model factorial test used Dense/Chunked and FSDP2 CPU offload on/off, with three new processes per condition. Each process made two warmup calls and 30 counted calls on the same padded 64/37-token fixture, K=8 and B=16. Every completed call retained output hashes and offsets. The root and decoder offload policies were inspected rather than inferred from a flag.

Table 2. Completed stability calls by entry and condition.

| Execution entry | Conditions | Fresh processes | Completed calls |
| --- | --- | ---: | ---: |
| Direct Qwen + FSDP2 | Dense, offload on | 3 | 96 |
| Direct Qwen + FSDP2 | Chunked, offload on | 3 | 96 |
| Direct Qwen + FSDP2 | Dense, offload off | 3 | 96 |
| Direct Qwen + FSDP2 | Chunked, offload off | 3 | 96 |
| Actual verl Worker | Dense, offload on | 1 | 32 |
| Actual verl Worker | Chunked, offload on | 1 | 32 |

The 12 direct processes completed 384 calls with the checked compact output contract; the two actual Worker processes completed another 64 calls. The original CUDA unknown error did not recur in this bounded set. Its cause remains unknown: these observations neither identify offload/chunking as the cause nor prove long-term production stability.

<!-- pagebreak -->

## 3. Capacity under an explicit allocator limit

Each capacity point ran in a new process with BF16, K=32, B=256, SDPA and a single causal sequence. The process used `torch.cuda.set_per_process_memory_fraction(0.25)`, corresponding to a nominal 2,129,510,400-byte PyTorch caching-allocator allowance on this device. This is an allocator constraint, not a hard partition of total GPU memory.

The scan increased N and narrowed failing brackets in 512-token steps. A success means the requested compact contract executed and finite outputs were saved. It does not establish correctness at lengths without a same-N dense reference.

Table 3. Capacity scan at allocator fraction 0.25.

| CPU offload | Path | Largest tested success N | Smallest tested OOM N | Same-N dense validation reached |
| --- | --- | ---: | ---: | ---: |
| On | Dense | 1536 | 2048 | Reference path |
| On | Chunked | 32768 | Not observed | 8192 |
| Off | Dense | 512 | 1024 | Reference path |
| Off | Chunked | 16384 | 16896 | 4096 |

The offload-on Chunked path reached the tested model-context ceiling of 32768 without a measured failing upper bound. The offload-off bracket is a tested success at 16384 and OOM at 16896; intermediate or untested shapes are not guaranteed by that bracket.

### 3.1 Correctness bounds on the capacity result

The N=1024 offload-on comparison used two 0.25 runs. Additional longer Dense references used a separate 0.55 allocator fraction at the same N, model and execution policy. They were compared against saved Chunked outputs from the 0.25 runs across every compact row and candidate, including exact Teacher Top-K sets.

Table 4. Revised same-N compact-output comparisons.

| CPU offload | Same-N comparisons that passed | Dense reference that failed |
| --- | --- | --- |
| On | 1024, 2048, 4096, 8192; max logprob difference 0 | 12288: OOM |
| Off | 2048, 4096; max logprob difference 0 | 8192: OOM |

Thus the correctness-supported tested lengths extend from the 0.25 Dense result of 1536 to 8192 with offload on, and from 512 to 4096 with offload off. Larger Chunked successes remain execution-capacity evidence with an open dense-correctness boundary. The higher-budget reference does not turn those comparisons into a same-budget memory benchmark. After correcting a shape/signature-validation gap in the comparator, all six cases were rerun and passed; earlier comparison records were retained.

<!-- pagebreak -->

## 3.2 Second allocator budget: 0.50

A preregistered follow-up used a 0.50 PyTorch allocator fraction with CPU offload on. Dense completed N=8704 and failed at the next 512-token grid point, N=9216. Chunked completed the tested 8192/16384/24576/32768 points. The 32768 success remains runability evidence; it is not a numerical-equivalence claim at that length.

Table 5. Capacity observations at allocator fraction 0.50.

| Condition | Largest tested success N | Next tested OOM N | Dense-checked Chunked N |
| --- | ---: | ---: | ---: |
| Dense, fraction 0.50 | 8704 | 9216 | Reference path |
| Chunked, fraction 0.50 | 32768 | Not observed | 9216 |

At N=8192 both paths used fraction 0.50, and every saved candidate logprob and Teacher Top-K row matched. To check a point beyond Dense's 0.50 boundary, a fraction-0.55 Dense reference at N=9216 was compared with the fraction-0.50 Chunked output. All 294,912 candidate values and all 9216 Teacher Top-K sets matched exactly. The correctness-supported tested length is therefore 512 tokens, or 5.88%, beyond the largest tested Dense success at 8704 for this grid; it is not a claim about every intervening shape or the exact maximum.

### Same-N memory observations at N=8192

Table 6. Independent-process memory observations at N=8192.

| Measurement | Dense, fraction 0.50 | Chunked, fraction 0.50 |
| --- | ---: | ---: |
| Forward-call allocated peak, MiB | 3596.83 | 681.66 |
| Recorded process allocator peak incl. load/wrap, MiB | 3596.83 | 958.48 |
| Forward-call reserved peak, MiB | 3846 | 1572 |
| Sampled whole-device used maximum, MiB | 5504 | 3229 |
| Device samples | 36 | 33 |

These were separate new processes with the same N and checked compact outputs. The recorded process allocator peak includes loading/wrapping as well as calls; the lower call-only peak does not remove initialization allocations. The sampled whole-device maxima retain the 100 ms sampler's scope and may include unrelated use or miss peaks. They are not exact process-memory peaks or a net-VRAM attribution.

The two allocator fractions expose different tested capacity brackets. The 0.50 check strengthens the observation that the result depends on the declared memory allowance; it does not validate the longest successful Chunked length. The dense reference limits remain explicit in both budget studies.


<!-- pagebreak -->

## 4. Memory and timing describe different quantities

A separate paired T1 experiment used Worker-aligned buffer construction, N=1024, K=32 and B=256. It measured a complete direct Qwen/FSDP2 forward through compact CPU outputs, excluding loading and validation. Each offload setting had six A/A blocks and eight interleaved ABBA/BAAB blocks, giving 16 samples per A/B label.

Table 7. Paired complete-forward allocated peaks at N=1024.

| CPU offload | A0 allocated peak MiB | A1 allocated peak MiB | Observed decrease |
| --- | ---: | ---: | ---: |
| On | 1517.09 | 664.97 | 56.17% |
| Off | 2467.26 | 1615.48 | 34.52% |

Table 8. Descriptive paired T1 host-clock samples.

| CPU offload / clock | A0 p50 / p95 ms | A1 p50 / p95 ms |
| --- | ---: | ---: |
| On / perf_counter | 192.580 / 193.953 | 192.978 / 198.150 |
| On / MONOTONIC_RAW | 190.491 / 191.858 | 190.926 / 196.081 |
| Off / perf_counter | 87.241 / 114.847 | 78.709 / 116.953 |
| Off / MONOTONIC_RAW | 87.567 / 115.198 | 78.986 / 117.407 |

Both T1 settings had zero candidate-logprob difference and matching Teacher Top-K at all 1024 rows. The offload-off group nevertheless had CUDA Event durations exceeding the enclosing perf-counter span in 30/32 A/B samples and 12/12 A/A samples. The host-clock disagreement remains unresolved. These are descriptive timings, not a supported general speedup or a comparison of separate offload runs as production performance.

### Independent-process observations at N=1024

The M1 offload-on Dense/Chunked runs provide an example of why allocator and device observations must remain distinct.

Table 9. Independent-process memory observations at N=1024.

| Measurement | Dense | Chunked |
| --- | ---: | ---: |
| Forward-call allocated peak, MiB | 1517.09 | 665.31 |
| Recorded process allocator peak incl. load/wrap, MiB | 1517.09 | 958.48 |
| Forward-call reserved peak, MiB | 1748 | 1572 |
| Sampled whole-device used maximum, MiB | 3429 | 3258 |
| Device samples | 40 | 30 |

The device sampler queried nvidia-smi every 100 ms. It includes the desktop and other processes and can miss transient peaks. The 171 MiB difference between sampled maxima is not a controlled net process-VRAM saving. Likewise, the allocated reductions above must not be called total-card VRAM or RL training gains.

<!-- pagebreak -->

## 5. Actual Worker output to a restricted Student update

I1 closes a local in-memory consumption path. A real Student generated the candidates once, then two actual fixed verl TeacherFSDPWorker instances executed Dense fallback and Chunked scoring. Their current CPU nested TensorDict outputs were scattered back to the original padded Student rows and passed directly to the pinned `compute_reverse_kl_topk` function. No recorded JSON teacher scores were substituted into this path.

The fixture had two prompt/response sequences, offsets [0,23,45], 46 padded model rows, K=8 and 25 active response-prediction rows. Student candidate selection used temperature 0.8 followed by log-softmax and Top-K; the Teacher used temperature 1. The candidate rows were not shifted again. Only the response mask received next-token alignment.

Each Student path restored the same final RMSNorm weight and fresh AdamW state, with learning rate 0.01 and weight decay 0. The loss retained support normalization, log-probability clamp -10, nonnegative token-loss clamp and maximum loss clamp 10. The Teacher tensors were detached.

Table 10. Actual Worker-to-Student restricted update.

| Observed quantity | Dense Worker output | Chunked Worker output |
| --- | ---: | ---: |
| Masked scalar loss | 0.0216006283 | 0.0216006283 |
| Final RMSNorm gradient L2 | 0.0124774044 | 0.0124774044 |
| Nonzero RMSNorm gradient elements | 896 | 896 |
| Changed BF16 parameter values after one step | 1 | 1 |
| Parameter-update L2 | 0.0099792480 | 0.0099792480 |

Candidate logprob, Teacher Top-K, mass, overlap, token losses, all 6,989,056 Student logits-gradient elements, 896 RMSNorm-gradient elements and the resulting updates matched exactly in this fixture. Both Worker Python objects were collected at the sequential lifecycle boundaries; normal outer/head hook completion and absence of Teacher gradients were checked.

This establishes a direct local Worker-to-Loss-to-restricted-update check. It does not execute Ray dispatch, a full Trainer, all-parameter optimization or an RL step. A single-step update of one parameter group is the explicit scope of the training evidence.

<!-- pagebreak -->

## 6. Interpretation, negative results and remaining work

The stage provides three concrete results. First, the controlled 2x2 set did not reproduce the initial native CUDA abort. Second, pre-head row chunking permitted longer tested inputs under the stated allocator allowance, with dense-reference correctness established through N=8192 on / N=4096 off at fraction 0.25, and through N=9216 on at fraction 0.50. Third, current real Worker outputs preserved the fixed upstream loss and a restricted real Student update.

A construction mismatch was found before comparing the direct model with the Worker: the fixed Worker explicitly applies `model.to(bfloat16)` after loading, which rounds its non-persistent rotary-frequency buffers. The original direct path kept those buffers in FP32. Applying the Worker construction policy removed the observed output difference in the tested fixture. This numerical explanation is distinct from the CUDA unknown-error investigation.

The initial CUDA unknown error, WSL dxg observations and dual-clock disagreement remain open. Longer Chunked inputs beyond the dense-reference ceiling retain UNKNOWN numerical equivalence. FP32 GPU behavior, FSDP1, multi-GPU/TP, packed varlen attention, LoRA/quantization/QAT, Ray transport, representative RL trajectories and full-model training remain outside this stage's verified support.

The preceding 56.17% allocated result and the present capacity brackets are conditional measurements. They do not establish total VRAM savings, a new fused kernel, universal latency improvement or production readiness. The original fused selected-logit proposal remains NO-GO for the full Teacher contract.

### Reproduction and evidence identifiers

The code baseline is `06f0abc0945a98fe94d732f4ee0ab875948cdb43` in the existing [project repository](https://github.com/yu830/rl-posttrain-sparse-scoring). The new scripts below belong to the follow-up revision containing this supplement; they are not present in the baseline alone. Follow-up source hashes are recorded with the raw measurements. Use that corresponding revision or source archive for reproduction. The principal entries are:

- `scripts/factorial_teacher_probe.py`: one new Dense/Chunked and offload condition per process.
- `scripts/compare_capacity_outputs.py`: all saved candidate rows and Teacher Top-K sets against same-N dense output.
- `scripts/worker_to_student_step.py`: actual local Worker outputs into the fixed upstream loss and one restricted update.
- `scripts/build_stage_report_20261030.py`: regenerate these PDFs from the two Markdown sources.

Evidence identifiers are `S1_SUMMARY.json`, `capacity_runner.jsonl`, `compare_validated_summary.json`, `compare_on_n9216.json`, `T1_worker_aligned_on/off.json` and `worker_to_student_01.json`. They distinguish observed success, OOM, correctness limits and unrun conditions.

### References

- [verl PR #6920, fixed source](https://github.com/verl-project/verl/tree/87cd055e872f9fdd7b976fcd797a0934b6dba265).
- [Pinned FSDP reverse-KL loss](https://github.com/verl-project/verl/blob/87cd055e872f9fdd7b976fcd797a0934b6dba265/verl/trainer/distillation/fsdp/losses.py), Git blob `97d3ebf485a5ce9e1b1a435cce427d6d89284dc6`.
- [Qwen2.5-0.5B-Instruct model snapshot](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775).
