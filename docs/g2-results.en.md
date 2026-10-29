# G2 admission observations

These new observations have not been incorporated into the author-reviewed report snapshot. The evidence supports **HOLD / NEEDS VERIFICATION** for G3. A selected-logits-only response is **NO-GO as a drop-in replacement for the complete contract** of the verl consumer described below.

## Consumer and semantics

At [verl main 2ab3529](https://github.com/verl-project/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/trainer/distillation/losses.py), the inspected top-K path uses teacher-selected support and forward-KL or tail-aware forward-KL. The main teacher manager requests scores for a token sequence without receiving student candidate IDs.

The open [verl PR #6920](https://github.com/verl-project/verl/pull/6920), head [87cd055e](https://github.com/verl-project/verl/tree/87cd055e872f9fdd7b976fcd797a0934b6dba265), actually implements student-top-K conditional reverse-KL. Its submitter reports an end-to-end training run; this project has **not independently reproduced** it. The [trainer](https://github.com/verl-project/verl/blob/87cd055e872f9fdd7b976fcd797a0934b6dba265/verl/trainer/ppo/ray_trainer.py) obtains candidate IDs in a no-gradient student prepass after rollout, sends aligned IDs to a frozen FSDP teacher, and carries the values into the update batch. The [teacher worker](https://github.com/verl-project/verl/blob/87cd055e872f9fdd7b976fcd797a0934b6dba265/verl/workers/teacher/fsdp_teacher_worker.py) expects IDs aligned to causal-logit rows and produces full-normalized selected logprobs plus teacher top-K IDs. A future service must still establish policy-version binding and behavior under asynchronous updates.

The [loss](https://github.com/verl-project/verl/blob/87cd055e872f9fdd7b976fcd797a0934b6dba265/verl/trainer/distillation/fsdp/losses.py) renormalizes both sides over the student support. With identical ordered support and temperature, conditional teacher probabilities suffice for that **loss term**. They cannot recover the original full-vocabulary teacher mass or teacher top-K diagnostics: those depend on omitted logits. The PR can clamp log-probabilities after normalization. This project's reference does not implement that clamped loss; a test demonstrates clamping can change its value. Full PR-contract equivalence has not been established.

The separate open [verl PR #7806](https://github.com/verl-project/verl/pull/7806) uses rollout top-K with a different truncated/full-normalized objective and reports multi-GPU experiments; those claims are third-party observations. The [Rethinking OPD author code](https://github.com/Thinking-Space/Rethinking-OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/verl/verl/workers/actor/dp_actor.py) uses full-normalized log-ratio rewards and a PPO surrogate. Conditional teacher scores must not silently replace that objective.

## Existing serving and implementation work

SGLang [46caa545](https://github.com/sgl-project/sglang/tree/46caa54562a97e71d9b937c62875c90b39274b0d) already supports /v1/score with candidate lists per item and optional normalization. [PR #38119](https://github.com/sgl-project/sglang/pull/38119) and [vLLM PR #56984](https://github.com/vllm-project/vllm/pull/56984) are prior per-position interfaces. [vLLM's fixed-revision documentation](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/docs/training/prompt_token_id_logprobs.md) specifies next-token alignment and per-row full logprobs. [IndexedLogits](https://github.com/pettyjohnjn/indexed_logits/tree/6020cb446cde8cb2b74a8fdf90db905a320936c3) already provides a selected-logit CUDA primitive; its inspected files did not declare a license. No third-party kernel has been copied.

Open [verl PR #7809](https://github.com/verl-project/verl/pull/7809), head [522514fe](https://github.com/verl-project/verl/blob/522514fe21300c4cf7b3800fd6cd827eb1caa2c9/verl/utils/kernel/linear_topk_log_probs.py), already implements vocabulary-tiled, globally normalized selected logprobs without materializing full student logits for Megatron, including a backward path. Its reported GPU validation is third-party evidence, not a project result. It is relevant prior art for any proposed streaming-head alternative; its student-side contract is not identical to the teacher's full-mass and teacher-top-K diagnostics.

## Executed observations

| Observation | Result and scope |
| --- | --- |
| CPU/GPU synthetic reference | Fixed supports, FP64/FP32/FP16/BF16, loss/gradient and failure cases; see the commands and record below. This is not an optimized kernel. |
| Real local service | Qwen2.5-0.5B-Instruct in BF16 on an RTX 5070 Laptop; one /v1/score request for eight synthetic causal prefixes and four candidate IDs per row returned HTTP 200. Candidate-conditioned scores matched a renormalization of its returned full logprobs. |
| Independent real-model comparison | **FAILED** the frozen elementwise BF16 budget atol=0.1, rtol=0.01: 2 of 32 SGLang full logprobs differed from a Transformers BF16 batched eager forward beyond the budget; maximum absolute difference was 0.37914085. No tolerance was widened. |
| Precision diagnosis | The same Transformers BF16 model's singleton and batched forwards differed by up to 0.24671745; SGLang BF16 versus Transformers FP32-single differed by up to 0.15135193. These observations suggest execution shape and precision matter, but do not establish a root cause or correctness pass. |
| Profiling | One separate synthetic replay after flushing radix cache recorded two BF16 dense head matmuls of [4,896] × [896,151936]; their correlated GPU kernels lasted approximately 993.408 and 994.713 microseconds. This is a trace observation, not A/A, A/B, critical-path attribution, or an expected speedup. |
| Training and performance | No local real training integration, A/A noise gate, A/B comparison, representative sweep, RL step timing or project speedup has been established. |

The actual inputs, response and timings are in [the smoke record](../results/sglang-teacher-smoke-20261029.json). The [failed comparison](../results/teacher-hf-comparison-20261029.json), [diagnostic](../results/teacher-precision-diagnostic-20261029.json), [synthetic numerical record](../results/g2-reference-20261029.json), and [trace summary](../results/sglang-profile-summary-20261029.json) preserve their respective scope. The raw trace and unredacted failure log are in the private project evidence workspace. One cold/warmup-affected request is not a performance benchmark.

## FP32 precision controls

The same eight prefixes and candidate IDs were replayed with the original fixed model. In the FP32 rows below, SGLang's fused operations were forced to their PyTorch fallback except for the first startup failure. The TF32 and IEEE runs used separate fresh Triton compilation caches. The comparison used the frozen FP32 budget atol=5e-5, rtol=2e-5; no threshold was changed.

| Configuration | Result | Max absolute logprob error | Out of budget |
| --- | --- | ---: | ---: |
| FP32, default fused operations | Server startup failed: the installed RMSNorm kernel did not dispatch Float | — | — |
| FP32, Triton attention, default precision | FAILED | 0.00966835 | 30 / 32 |
| FP32, Triton attention, explicit TF32 | FAILED | 0.00966835 | 30 / 32 |
| FP32, Triton attention, explicit IEEE | PASS | 0.00001717 | 0 / 32 |
| FP32, torch-native attention | PASS | 0.00002670 | 0 / 32 |
| BF16, torch-native attention, default fused operations | FAILED | 0.30104256 | 3 / 32 |
| BF16, torch-native attention and fused operations | FAILED | 0.45039749 | 5 / 32 |

On these 32 values the default and explicit-TF32 SGLang outputs were identical; explicit TF32 versus IEEE differed by at most 0.00967407. These controls support a Triton FP32 dot-precision explanation for this FP32 discrepancy; they do not identify a single kernel or explain the BF16 failures. The global TRITON_F32_DEFAULT setting affects all eligible Triton FP32 dot operations in the server. The pinned [SGLang extend-attention source](https://github.com/sgl-project/sglang/blob/46caa54562a97e71d9b937c62875c90b39274b0d/python/sglang/kernels/ops/attention/extend_attention.py#L795) contains a dot without an explicit input precision; [Triton documentation](https://triton-lang.org/main/python-api/generated/triton.language.dot.html) describes its NVIDIA FP32 default. This is a scoped numerical diagnosis, not a proposed production patch or performance result. In earlier comparison records the field named tf32 refers to the independent PyTorch reference only, not the SGLang Triton setting.

[Precision control records](../results/precision-controls-20261029.json) are generated from the raw JSON by [the summarizer](../scripts/summarize_precision_controls.py); local-path tracebacks are omitted, and source hashes identify the private originals. The full input, output, configuration and elementwise results remain available. The diagnosis does not change the G3 hold.

## Reproduce

Use Python 3.12 and the observed [dependency versions](../requirements-g2.lock). This is a platform-specific uv pip freeze snapshot without wheel hashes, not a cross-platform lock. Install SGLang separately from commit 46caa54562a97e71d9b937c62875c90b39274b0d with SGLANG_BUILD_RUST_EXTS=none; the observed build did not compile optional Rust extensions. Download Qwen/Qwen2.5-0.5B-Instruct (Apache-2.0) and its tokenizer from revision 7ae557604adf67be50417f59c2c2f167def9a775. The observed model.safetensors SHA-256 was fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe; tokenizer.json SHA-256 was c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539.

For a new Linux environment, set SGLANG_SRC to a new empty checkout path and MODEL_DIR to a new model directory, then:

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

These installation commands describe the recorded setup; a fresh installation from the snapshot has not been independently retested.

From the repository root, with the virtual environment active and MODEL_DIR pointing to the fixed snapshot:

    python -m pytest -q tests/test_scoring_reference.py
    python scripts/measure_reference.py --out results/g2-reference-20261029.json
    python scripts/sglang_teacher_smoke.py --model-dir "$MODEL_DIR" --out results/sglang-teacher-smoke-20261029.json --server-log /tmp/g2-sglang.log
    python scripts/verify_teacher_hf.py --model-dir "$MODEL_DIR" --smoke results/sglang-teacher-smoke-20261029.json --out results/teacher-hf-comparison-20261029.json

The last command is expected to report the recorded failure on the observed setup; do not reinterpret it as passing. The precision diagnostic records singleton and FP32 comparisons. The smoke script's optional profile directory runs a separate replay, and the trace-summary script accepts the trace and vocabulary size. Use a separate output directory to preserve historical records.


For the paired FP32 precision controls, set IEEE_CACHE and TF32_CACHE to two new, distinct directories before running:

    python scripts/sglang_teacher_smoke.py --dtype float32 --force-native-fused-ops --attention-backend triton --triton-f32-default ieee --triton-cache-dir "$IEEE_CACHE" --model-dir "$MODEL_DIR" --out results/sglang-teacher-fp32-triton-ieee-20261029.json --server-log /tmp/g2-ieee.log
    python scripts/verify_teacher_hf.py --dtype float32 --model-dir "$MODEL_DIR" --smoke results/sglang-teacher-fp32-triton-ieee-20261029.json --out results/teacher-hf-fp32-triton-ieee-20261029.json
    python scripts/sglang_teacher_smoke.py --dtype float32 --force-native-fused-ops --attention-backend triton --triton-f32-default tf32 --triton-cache-dir "$TF32_CACHE" --model-dir "$MODEL_DIR" --out results/sglang-teacher-fp32-triton-tf32-20261029.json --server-log /tmp/g2-tf32.log
    python scripts/verify_teacher_hf.py --dtype float32 --model-dir "$MODEL_DIR" --smoke results/sglang-teacher-fp32-triton-tf32-20261029.json --out results/teacher-hf-fp32-triton-tf32-20261029.json

The TF32 comparison failed the recorded budget in this observation. Neither control is timed as a benchmark.
