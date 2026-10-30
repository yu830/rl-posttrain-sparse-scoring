# Memory-Bounded Teacher Scoring

> The 30 October follow-up is AI-assisted and awaits Human review. The earlier reviewed report remains a separate snapshot.

This experimental path moves row chunking in front of the LM head while retaining the teacher's full-vocabulary-normalized log probabilities at student candidate IDs and the teacher's own Top-K IDs. It targets the frozen teacher in the open [verl #6920](https://github.com/verl-project/verl/pull/6920) at `87cd055e872f9fdd7b976fcd797a0934b6dba265`. The patch is opt-in and is not an upstream merge. Row chunking itself is established prior art.

The adapter is installed before FSDP wrapping. Its ordinary model call executes the backbone and final norm, then invokes the existing LM head on a bounded number of rows. It returns compact scores without concatenating a full logits tensor when the chunk is smaller than the input. The worker preserves its nested offsets and two-field CPU output. If an unbudgeted tile covers all rows, the integration uses native dense forward and reduces logits after FSDP returns.

Current support is limited to tied-embedding Qwen2, ordinary dense head, BF16/FP32, one GPU / world size one, no cache and no teacher gradients. Unsupported multi-rank, untied head, LoRA, quantized/QAT, Liger, fused model forward and sequence-parallel configurations are rejected. Multiple sequences are supported through padded input and the real attention mask; packed multi-sequence varlen attention is not yet verified and is rejected. The budget estimates readout allocations; it is not a hard bound on full-model memory, other processes, fragmentation or CUDA workspace.

## 30 October follow-up: stability, capacity and consumer

The first Worker CUDA `unknown error` remains unexplained. A controlled 2×2 experiment used three fresh FSDP2 processes for each dense/chunked and CPU-offload on/off condition, with two warmups and 30 checked calls per process. All 384 calls matched the same compact reference and the actual FSDP policies matched the requested settings. Two additional real Worker processes completed 32 calls each with per-call output hashes and offsets checked. These tests did not reproduce the failure and do not prove its cause or long-term stability.

The initial direct-model path and the actual Worker produced different BF16 scores. The pinned Worker explicitly calls `model.to(dtype)`, which also casts Qwen2 rotary frequency buffers from FP32 to BF16; the original direct path did not. In the tested fixture, matching that single build step changed 31/32 frequency values and restored exact Worker/direct equality for all 808 candidate scores and 101 teacher Top-K rows. This explains that fixture's path difference, not the CUDA exception. The earlier direct-model measurements below remain valid only for their recorded buffer configuration.

The capacity experiment used a **PyTorch caching allocator fraction**, not an all-device memory limit. Each workload ran in a fresh process with the same single-sequence synthetic input, BF16, K=32, B=256 and model context limit 32768. Dense and chunked were also compared against complete dense outputs obtained under a separately larger reference budget.

| Allocator fraction | CPU offload | Dense largest tested pass / next tested OOM | Chunked largest runnable | Chunked dense-reference correctness checked through |
| --- | --- | ---: | ---: | ---: |
| 0.25 | on | 1536 / 2048 | 32768 | 8192 |
| 0.25 | off | 512 / 1024 | 16384 / OOM at 16896 | 4096 |
| 0.50 | on | 8704 / 9216 | 32768 | 9216 |

The successful lengths are observed grid points at 512-token boundary resolution, not a proof of a monotonic or device-wide maximum. Under fraction 0.25, the verified 8192 versus dense's tested 1536 points are a 5.33× observed length ratio; the 32768 run was not checked against full dense logits at that length. At fraction 0.50 the 9216-row chunked result matched a dense reference run at fraction 0.55 for all 294912 candidate values and all 9216 teacher Top-K sets; fraction 0.55 was used only for correctness, not the capacity comparison.

At N=8192 under fraction 0.50, CPU offload on, fresh-process dense/chunked forward-call PyTorch peak allocated was 3596.83/681.66 MiB and forward-call peak reserved was 3846/1572 MiB. Including model load and wrapping, the observed process allocator peak was 3596.83/958.48 MiB. The corresponding 100 ms `nvidia-smi` all-device maximum observations were 5504/3229 MiB. These are sampled total-device values including other processes and may miss transient peaks; their difference is not an exact process VRAM saving. At the dense fraction-0.50 OOM, CUDA still reported approximately 3401 MiB free, confirming that this observation is an allocator-budget boundary rather than total-device exhaustion.

The real pinned `TeacherFSDPWorker` dense and chunked outputs were then passed directly in memory to the fixed verl `compute_reverse_kl_topk` function and a real Qwen Student update. On two padded sequences with offsets [0,23,45] and 25 response prediction rows, 360 teacher candidate scores, teacher Top-K, loss and the checked 6989056 logits / 896 RMSNorm gradients agreed between paths. The AdamW step trained only the final RMSNorm; both variants had nonzero gradients and changed one BF16 parameter value. This is a direct local Worker-to-loss-to-limited-update integration, not Ray dispatch or a full RL training step.

For N=1024/K=32/B=256, a later paired test explicitly matched the Worker buffer cast. With offload on, A0/A1 `perf_counter` p50 was 192.580/192.978 ms and allocated peak 1517.09/664.97 MiB; with offload off it was 87.241/78.709 ms and 2467.26/1615.48 MiB. The two offload settings were separate processes. In the offload-off run CUDA Event exceeded the surrounding perf-counter span in 30/32 A/B samples; timings remain descriptive and are not a speedup claim.

## 29 October direct-model measurement (earlier buffer configuration)

The table below is a follow-up experiment on the reviewed adapter, not a service or RL training benchmark. It used the fixed Qwen2.5-0.5B-Instruct snapshot, BF16, synthetic tokenized text and deterministic candidate IDs, RTX 5070 Laptop 8 GiB, PyTorch 2.14.1+cu130, Transformers 5.19.0, FSDP2 world size one with CPU parameter offload, SDPA and no KV cache. A0 executes a complete model forward and reduces the full logits; A1 executes the complete same-model forward with row chunking. Both include compact CPU copies. Each cell has four A/A blocks and four alternating ABBA/BAAB blocks, eight samples per A/B label. Values are synchronized host p50; peak is absolute PyTorch allocated memory, not total device memory.

| Rows | K | Chunk | A0 peak MiB | A1 peak MiB | A0 p50 ms | A1 p50 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 32 | 256 | 552.04 | 552.04 | 196.421 | 196.893 |
| 512 | 32 | 256 | 774.05 | 664.14 | 174.285 | 171.881 |
| 1024 | 32 | 256 | 1517.09 | 665.31 | 148.338 | 149.474 |
| 2048 | 32 | 256 | 1814.31 | 667.65 | 202.120 | 202.617 |

For 1024 rows the observed absolute allocated-peak reduction was 56.15% for this scope; it is not the earlier head-only 75% incremental figure. At 512 rows, a full-row chunk of 512 instead increased the absolute peak by about 33.7%, motivating the unbudgeted dense fallback. The 128-row A/A host p50 values were 118.614 and 115.205 ms, while the later A/B host p50 was about 196 ms; this phase-to-phase drift prevents a general speedup conclusion. Reserved memory includes allocator history. The earlier matrix covered 128/512/1024/2048 rows, K=8/32 and chunk rows 64/128/256/512; all 32 tested candidate-logprob, teacher Top-K and head-shape checks passed. A follow-up on the reviewed adapter covered the same row counts at K=32 and chunks 64/256; all eight cases passed and the table reports that follow-up. Earlier transient probe samples were lost after WSL temporary storage was cleared; the recorded matrix is a subsequent run. An earlier timer probe also observed CUDA Event values larger than synchronized host spans; its cause remains unknown.

The archived matrix script incorrectly labeled full-row chunks as avoiding a full logits head call; the recorded call shapes show the full projection when chunk size covers all rows. The current script corrects this label, and a follow-up check verified both cases.

A later 1024-row confirmation using the current script measured perf-counter host p50 163.110/168.292 ms (A0/A1) and CLOCK_MONOTONIC_RAW host p50 150.562/155.345 ms. During A/A, CUDA Event exceeded perf-counter host time in 7/8 samples and exceeded the raw clock in 0/8. In a separate ten-sample CUDA sleep control, raw time exceeded perf-counter time in 10/10 samples, while Event exceeded perf-counter time in 9/10 and raw time in 0/10. The source of the disagreement is unresolved; latency is descriptive and does not establish a speedup.

A separate FSDP2 readout probe at 1024 rows and K=32 selected 35/70/141 rows for estimated budgets of 64/128/256 MiB. Measured readout allocated increments were about 51.03/101.75/204.64 MiB respectively; complete-forward peaks were about 552.28 MiB. This does not establish a hard budget across workloads or backends.

The patched real `TeacherFSDPWorker` was invoked directly in one process using FSDP2 CPU offload and a padded two-sequence input. Dense fallback and chunked outputs had identical candidate-logprob and teacher-Top-K hashes and offsets in the tested fixture. A separate real-model consumer check used actual student Top-K selection, masking and one AdamW update of only the student's final RMSNorm weight: loss and compared gradients agreed, with a nonzero gradient and one BF16 parameter value changed. This is not a complete training step for the model. One initial repeated worker run terminated with CUDA `unknown error`; later direct runs completed, but the failure is not explained and stability is not established by those few runs.

The recorded real-model teacher outputs were also replayed into the actual pinned verl `compute_reverse_kl_topk` function with a real Qwen student. Project-versus-upstream and upstream A0-versus-A1 comparisons passed for loss, mass, overlap, logits gradients and final RMSNorm gradients. This direct function check does not execute the complete worker-to-trainer pipeline.

An initial project-consumer implementation cast the BF16 student log probabilities to FP32 separately on two gradient branches: its scalar loss matched upstream, but the RMSNorm and logits gradients differed by about 1.07% and 1.80% in relative L2. Reusing the FP32 conversion as upstream does removed those differences in this fixture. The earlier difference is retained as a negative result; exact agreement is limited to the tested input and environment.

## Reproduction

Use the existing locked G2 environment and local model snapshot. The full-model measurement is:

```bash
python scripts/full_teacher_benchmark.py --model-dir "$MODEL_DIR" --out "$OUT_DIR/full-forward.json" \
  --rows 128 512 1024 2048 --ks 8 32 --row-chunks 64 128 256 512 \
  --warmup 2 --aa-blocks 4 --ab-blocks 4 --fsdp2 --cpu-offload
python scripts/teacher_budget_probe.py --model-dir "$MODEL_DIR" --out "$OUT_DIR/budget.json" \
  --rows 1024 --k 32 --budgets-mib 64 128 256 --max-chunk-rows 512 --fsdp2 --cpu-offload
python scripts/teacher_consumer_step.py --model-dir "$MODEL_DIR" --out "$OUT_DIR/consumer.json" \
  --device cuda --dtype bfloat16 --optimizer adamw --learning-rate 0.01
```

The following commands reproduce the Worker-aligned buffer setting and the direct Worker-to-loss check. Repeat each factorial condition in a separate process; `nvidia-smi` sampling is an all-device observation.

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

Use `--cpu-offload off` for the other factorial conditions. Capacity runs use `--purpose capacity --rows N --k 32 --chunk-rows 256 --attention sdpa --allocator-fraction 0.25 --save-output "$OUT_DIR/output.pt"`; compare same-input outputs with `scripts/compare_capacity_outputs.py`. Select an allocator fraction deliberately and record other GPU activity; an allocator OOM is not automatically an all-device OOM.

The [English stage supplement](reports/memory-bounded-stage-20261030.en.pdf) and [editable Markdown](reports/memory-bounded-stage-20261030.en.md) are pending Human review. To regenerate the PDF, run `python scripts/build_stage_report_20261030.py --output-dir "$OUT_DIR/reports"`; the measured build used ReportLab 4.4.9 and Windows SimSun/SimHei fonts. A different full Chinese font can be supplied with `--cjk-font`; font substitution may change pagination.

The [pinned verl patch and worker setup](../integrations/verl_6920/README.md) contain the required source revision, configuration and offline verification. Actual worker execution is a direct-process smoke, not a Ray dispatch or multi-GPU run. The earlier reviewed technical reports and diagrams describe their original snapshot and have not been revised to assert these newer measurements.
