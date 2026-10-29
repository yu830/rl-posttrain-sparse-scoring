# Memory-bounded teacher integration for verl #6920

This patch targets **only** `verl-project/verl` commit
`87cd055e872f9fdd7b976fcd797a0934b6dba265`. It connects the project's Qwen2
scorer to the actual teacher engine builder and `TeacherFSDPWorker`.
The adapter is installed before FSDP wrapping, and each request enters
`self.engine.module(...)` through the ordinary outer FSDP forward.
The original parameter tree, tied embeddings, no-grad/eval context, nested
output offsets, collecting-rank check and CPU return are retained.

The returned TensorDict still contains exactly `teacher_on_student_logp` and
`teacher_topk_ids`. IDs address existing causal-logit rows; this integration
adds no shift or temperature transformation.

## Apply to the pinned source

Set `SCORING_DIR` and `VERL_DIR` to the existing scoring repository and a
separate verl checkout. The scoring repository must remain on `PYTHONPATH`;
the patch imports `reference` and `integrations.verl_6920` from it.

```bash
git -C "$VERL_DIR" fetch origin 87cd055e872f9fdd7b976fcd797a0934b6dba265
git -C "$VERL_DIR" checkout --detach 87cd055e872f9fdd7b976fcd797a0934b6dba265
git -C "$VERL_DIR" apply --check "$SCORING_DIR/integrations/verl_6920/memory-bounded-teacher.patch"
git -C "$VERL_DIR" apply "$SCORING_DIR/integrations/verl_6920/memory-bounded-teacher.patch"
export PYTHONPATH="$SCORING_DIR:$VERL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export VERL_LOGGING_LEVEL=INFO
```

Use a clean checkout for application. The patch preserves the upstream
CRLF files; a whitespace check for those files is
`git -c core.whitespace=cr-at-eol diff --check`.
`source-manifest.json` records the required Git blobs.

## Explicit configuration

The default `teacher_memory_bounded: false` keeps the original worker branch.
The following settings merge into the existing distillation config; retain
the existing model path, routing key, resource and loss settings:

```yaml
distillation:
  teacher_backend: fsdp
  teacher_models:
    teacher_model:
      use_remove_padding: false
      override_config:
        attn_implementation: eager
  teacher_fsdp_config:
    teacher_memory_bounded: true
    teacher_lm_head_chunk_rows: 64
    teacher_head_memory_budget_mib: 128
```

The teacher model options use the same `use_remove_padding` and
`override_config` names/defaults as upstream `HFModelConfig`. The existing
trainer passes them to that constructor and passes the three scoring options
through its existing `FSDPEngineConfig` conversion. Omitting them retains
`use_remove_padding=True`, the empty override map and the upstream attention
default. Padded inputs may contain multiple sequences: candidates are
scattered using the engine's actual attention mask and restored in nested
row order. A single sequence also works with remove-padding enabled.
Multi-sequence packed/rmpad execution is rejected until actual varlen
attention behavior is verified.

`teacher_lm_head_chunk_rows` caps the rows projected in one head call.
`teacher_head_memory_budget_mib` can reduce that count before the forward;
it estimates head allocations, including padded model rows. It is **not a
hard bound on total GPU memory**, FSDP parameter residency, backbone
activations, CUDA workspace, fragmentation or other processes.
An impossible budget raises before the model forward. The existing
`distillation.teacher_chunk_size` retains its original dense gather meaning.

With no explicit budget, a selected tile covering all projected model rows
uses `dense_full_row_fallback`: native dense forward followed by the existing
gather outside FSDP. This avoids retaining a complete logits tile plus
reduction buffers while FSDP parameters are still resident. With any explicit
budget, this automatic dense fallback is disabled. INFO logs report the
execution path, requested/selected rows, estimated bytes, budget and sizing
reason. Errors and OOMs propagate; the outer FSDP forward is never retried.

Only the adapter's ordinary Qwen2 linear-head contract is accepted. LoRA,
quantization/QAT, alternative/fused model forwards and sequence parallelism
are rejected. This is an integration for the pinned consumer, not a general
model wrapper.

## Offline verification and actual runtime scope

```bash
python -m integrations.verl_6920.verify_patch --verl-dir "$VERL_DIR"
VERL_6920_SOURCE="$VERL_DIR" python -m pytest -q tests/test_verl_patch.py
```

The verifier reads pinned Git blobs, applies the patch to temporary copies,
and parses/compiles the Python sources. It does not alter the given checkout.
CPU tests exercise real nested-tensor layout conversion, fallback/budget
policy and the actual trainer constructor expression. They also compare the
disabled scoring branch, output reconstruction and CPU-return code with the
original AST. These checks **do not execute a verl worker or establish GPU
performance**; runtime evidence must be reported separately.

An actual worker run needs verl's pinned-source runtime dependencies, including
`tensordict`, `peft`, `codetiming`, `hydra-core`, `torchdata`, `ray`,
`accelerate` and the other imports required by its `requirements.txt`.
Importing a patched source file with missing dependencies is not a worker run.
A local direct invocation must construct the real `TeacherFSDPWorker`, call
`reset()` to build its real engine, and call `compute_logprobs_at_ids()` with
a real nested TensorDict. Such a run covers the actual worker/engine path;
it does not establish Ray dispatch, multi-GPU sharding or a complete RL step.

The direct worker and actual upstream-loss checks can be reproduced as follows
after applying the patch and setting the environment above. Use fresh output
filenames; these commands do not exercise Ray remote dispatch.

```bash
python scripts/verl_teacher_worker_smoke.py --verl-dir "$VERL_DIR" --model-dir "$MODEL_DIR" \
  --mode dense-fallback --warmup 1 --repeats 2 --out "$OUT_DIR/worker-dense.json"
python scripts/verl_teacher_worker_smoke.py --verl-dir "$VERL_DIR" --model-dir "$MODEL_DIR" \
  --mode chunked --chunk-rows 16 --head-memory-budget-mib 64 --warmup 1 --repeats 2 \
  --out "$OUT_DIR/worker-chunked.json"
python scripts/verify_verl_consumer.py --verl-dir "$VERL_DIR" --input "$OUT_DIR/consumer.json" \
  --device cuda --out "$OUT_DIR/verl-consumer.json"
```