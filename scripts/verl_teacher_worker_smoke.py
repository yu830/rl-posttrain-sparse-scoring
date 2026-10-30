#!/usr/bin/env python3
"""Run the real patched verl #6920 teacher worker in one direct process.

Each invocation builds the actual FSDP2/offload engine through reset() and
scores a padded batch of two unequal sequence lengths. Run chunked and
dense-fallback modes in separate processes and retain both JSON files.
This is not Ray remote dispatch, multi-GPU validation or a complete RL step.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time
import traceback

PINNED_VERL = "87cd055e872f9fdd7b976fcd797a0934b6dba265"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
PROJECT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.STDOUT).strip()


def json_safe(value):
    # Retain nonfinite failure evidence while writing standards-compliant JSON.
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def tensor_sha256(tensor) -> str:
    raw = tensor.detach().cpu().contiguous()
    header = json.dumps({"shape": list(raw.shape), "dtype": str(raw.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(header + raw.view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def checkpoint(result, out):
    """Persist completed stages even if a native CUDA failure aborts Python."""
    path = out.with_name(out.name + ".partial.json")
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(json_safe(result), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def snapshot_output(output):
    if output is None:
        return None
    snapshot = {}
    for key in sorted(output.keys()):
        nested = output[key]
        values = nested.values()
        snapshot[key] = {
            "dtype": str(values.dtype), "device": str(values.device),
            "offsets": nested.offsets().cpu().tolist(), "shape": list(values.shape),
            "values_sha256": tensor_sha256(values),
            "values": values.float().tolist() if values.is_floating_point() else values.tolist(),
        }
    return snapshot


def compact_output_signature(snapshot):
    """Keep the exact CPU output contract without duplicating tensor values."""
    if snapshot is None:
        return None
    required = {"teacher_on_student_logp", "teacher_topk_ids"}
    if set(snapshot) != required:
        raise ValueError("unexpected Worker output signature fields")
    result = {}
    for name, value in snapshot.items():
        shape = value.get("shape", value.get("values_shape"))
        if shape is None:
            raise ValueError("Worker output signature is missing the shape")
        result[name] = {key: value[key] for key in ("dtype", "device", "offsets", "values_sha256")}
        result[name]["shape"] = shape
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-dir", type=Path, required=True, help="patched checkout at the fixed verl commit")
    parser.add_argument("--model-dir", type=Path, required=True, help="existing pinned Qwen2.5-0.5B snapshot")
    parser.add_argument("--out", type=Path, required=True, help="new JSON path; existing evidence is never overwritten")
    parser.add_argument("--mode", choices=("chunked", "dense-fallback"), required=True)
    parser.add_argument("--chunk-rows", type=int, default=8, help="row cap for chunked mode")
    parser.add_argument("--head-memory-budget-mib", type=float, default=None)
    parser.add_argument("--gather-chunk-rows", type=int, default=32)
    parser.add_argument("--lengths", type=int, nargs=2, default=(64, 37), metavar=("FIRST", "SECOND"))
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--invalid-candidate", choices=("negative", "outside"), help="verify pre-FSDP rejection")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=0, help="0 chooses a free localhost port")
    parser.add_argument("--no-hooks", action="store_true", help="omit optional call-shape instrumentation")
    parser.add_argument("--expected", type=Path, help="successful Worker JSON with identical fixture")
    return parser.parse_args()


def run(args, result):
    result["stage"] = "source_and_model_validation"
    if git_head(args.verl_dir) != PINNED_VERL:
        raise ValueError("--verl-dir is not at the pinned verl commit")
    if args.lengths[0] == args.lengths[1] or any(not 0 < n <= 512 for n in args.lengths):
        raise ValueError("provide two different sequence lengths between 1 and 512")
    if args.topk <= 0 or args.chunk_rows <= 0 or args.gather_chunk_rows <= 0:
        raise ValueError("topk and row counts must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats positive")
    if args.invalid_candidate and args.no_hooks:
        raise ValueError("--invalid-candidate requires hooks to verify pre-FSDP rejection")
    if args.mode == "dense-fallback" and args.head_memory_budget_mib is not None:
        raise ValueError("dense-fallback requires no explicit head budget")
    if not 0 <= args.master_port <= 65535:
        raise ValueError("master-port must be between 0 and 65535")
    result["model"] = {
        "id": MODEL_ID, "revision": MODEL_REVISION,
        "artifacts_sha256": {name: sha256(args.model_dir / name)
                             for name in ("model.safetensors", "tokenizer.json", "config.json")},
    }
    if result["model"]["artifacts_sha256"]["model.safetensors"] != MODEL_SHA256:
        raise ValueError("model.safetensors differs from the pinned model")
    if result["model"]["artifacts_sha256"]["tokenizer.json"] != TOKENIZER_SHA256:
        raise ValueError("tokenizer.json differs from the pinned tokenizer")
    upstream_paths = (
        "verl/workers/teacher/fsdp_teacher_worker.py", "verl/workers/engine/fsdp/transformer_impl.py",
        "verl/workers/config/engine.py", "verl/workers/config/distillation.py", "verl/trainer/ppo/ray_trainer.py",
    )
    result["sources_sha256"] = {name: sha256(args.verl_dir / name) for name in upstream_paths}
    result["project_sources_sha256"] = {
        name: sha256(PROJECT / name) for name in (
            "reference/memory_bounded_teacher.py", "reference/memory_budget.py",
            "integrations/verl_6920/bridge.py", "scripts/verl_teacher_worker_smoke.py",
        )
    }
    result["project_git_head"] = git_head(PROJECT)
    result["verl_git_head"] = PINNED_VERL
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, str(args.verl_dir))
    os.environ.setdefault("VERL_LOGGING_LEVEL", "INFO")
    result["stage"] = "real_worker_import"
    import torch
    import ray
    from tensordict import TensorDict
    from verl.trainer.config import CheckpointConfig
    from verl.utils import tensordict_utils as tu
    from verl.utils.profiler import ProfilerConfig
    from verl.workers.config import FSDPEngineConfig, HFModelConfig, TrainingWorkerConfig
    worker_module = importlib.import_module("verl.workers.teacher.fsdp_teacher_worker")
    TeacherFSDPWorker = worker_module.TeacherFSDPWorker
    loaded_worker_path = Path(inspect.getfile(TeacherFSDPWorker)).resolve()
    if not loaded_worker_path.is_relative_to(args.verl_dir):
        raise RuntimeError(f"worker was imported from a different source: {loaded_worker_path}")
    result["loaded_worker_source"] = str(loaded_worker_path)
    result["versions"] = {"python": sys.version, "torch": torch.__version__, "cuda_build": torch.version.cuda,
                          "transformers": importlib.import_module("transformers").__version__}
    if torch.distributed.is_initialized():
        raise RuntimeError("run the smoke entry in a fresh process without an existing process group")
    if ray.is_initialized():
        raise RuntimeError("this entry exercises direct-process invocation; run it without an initialized Ray runtime")
    if any(key.startswith("RAY_EXPERIMENTAL_NOSET_") and value for key, value in os.environ.items()):
        raise RuntimeError("direct invocation cannot use Ray actor-only NOSET_VISIBLE_DEVICES flags")
    if not torch.cuda.is_available():
        raise RuntimeError("this smoke entry requires a CUDA GPU")
    torch.cuda.set_device(0)
    torch.manual_seed(731)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result["gpu"] = {"name": torch.cuda.get_device_name(0), "device_index": 0,
                     "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}

    result["stage"] = "fixture_and_real_configs"
    model_config = HFModelConfig(
        path=str(args.model_dir), use_remove_padding=False, enable_gradient_checkpointing=False,
        use_fused_kernels=False, override_config={"attn_implementation": "eager", "use_cache": False},
    )
    tokenizer = model_config.tokenizer
    vocabulary = model_config.hf_config.vocab_size
    if not 0 < args.topk <= vocabulary:
        raise ValueError("topk must not exceed the actual model vocabulary")
    texts = (
        "A frozen teacher scores every candidate at the original causal position. Memory measurements include the full model forward. ",
        "Different sequence lengths require the real attention mask. Candidate order and output offsets must remain aligned. ",
    )
    tokens = []
    candidates = []
    for sequence, (text, length) in enumerate(zip(texts, args.lengths)):
        encoded = tokenizer.encode(text * (length + 1), add_special_tokens=False)[:length]
        if len(encoded) != length:
            raise RuntimeError("the tokenizer did not produce the requested fixture length")
        tokens.append(encoded)
        rows = []
        for position, token in enumerate(encoded):
            ids = [(token + 131 * position + 17 * sequence + 9973 * column) % vocabulary
                   for column in range(args.topk)]
            if args.topk > 1:
                ids[-1] = ids[0]  # Preserve ordered duplicate candidate semantics.
            rows.append(ids)
        candidates.append(rows)
    if args.invalid_candidate:
        candidates[0][0][0] = -1 if args.invalid_candidate == "negative" else vocabulary
    fixture = {"input_ids": tokens, "position_ids": [list(range(n)) for n in args.lengths],
               "topk_ids": candidates, "causal_alignment": "row i addresses logits at input position i; no shift"}
    fixture_bytes = json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode()
    result["fixture"] = fixture
    result["fixture_sha256"] = hashlib.sha256(fixture_bytes).hexdigest()
    expected_signature = None
    if args.expected is not None:
        reference_path = args.expected.resolve()
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if reference.get("status") != "PASS_SMOKE" or reference.get("fixture_sha256") != result["fixture_sha256"]:
            raise ValueError("expected Worker result is not a successful identical fixture")
        expected_signature = compact_output_signature(reference["outputs"])
        result["expected_evidence"] = {"path": str(reference_path), "sha256": sha256(reference_path),
                                       "mode": reference.get("mode")}
    result["correctness_scope"] = ("STRICT_EXTERNAL_WORKER_HASH" if expected_signature is not None else
                                   "REPEAT_CONSISTENCY_ONLY_NO_EXTERNAL_REFERENCE")
    result["per_call_signatures"] = []
    result["input_offsets"] = [0, args.lengths[0], sum(args.lengths)]
    model_rows = len(args.lengths) * max(args.lengths)
    requested_rows = model_rows if args.mode == "dense-fallback" else args.chunk_rows
    from reference.memory_budget import select_chunk_rows
    from integrations.verl_6920.bridge import select_teacher_execution_path
    decision = select_chunk_rows(
        model_rows, vocabulary, args.topk, getattr(torch, args.dtype),
        requested_rows=requested_rows, head_memory_budget_mib=args.head_memory_budget_mib,
    )
    expected_path = select_teacher_execution_path(num_model_rows=model_rows, decision=decision)
    if args.mode == "chunked" and expected_path != "chunked_lm_head":
        raise ValueError("chunked mode needs a row cap below all projected rows, or an explicit budget")
    result["expected_path"] = expected_path
    result["chunk_decision"] = dict(decision.__dict__)
    engine_config = FSDPEngineConfig(
        strategy="fsdp2", forward_only=True, model_dtype=args.dtype, dtype=args.dtype,
        mixed_precision={"param_dtype": args.dtype, "reduce_dtype": "float32", "buffer_dtype": "float32"},
        fsdp_size=1, ulysses_sequence_parallel_size=1, use_torch_compile=False,
        use_dynamic_bsz=False, infer_micro_batch_size_per_gpu=2,
        teacher_memory_bounded=True, teacher_lm_head_chunk_rows=requested_rows,
        teacher_head_memory_budget_mib=args.head_memory_budget_mib,
    )
    config = TrainingWorkerConfig(
        model_type="language_model", model_config=model_config, engine_config=engine_config,
        optimizer_config=None, checkpoint_config=CheckpointConfig(),
        profiler_config=ProfilerConfig(tool=None, enable=False, save_path=None, tool_config={}),
    )
    result["configuration"] = {
        "strategy": "fsdp2", "forward_only": True, "world_size": 1, "use_remove_padding": False,
        "attention": "eager", "use_cache": False, "model_dtype": args.dtype,
        "teacher_lm_head_chunk_rows": requested_rows,
        "teacher_head_memory_budget_mib": args.head_memory_budget_mib,
        "budget_scope": "estimated head allocations; not a total-device guarantee",
        "gather_chunk_rows": args.gather_chunk_rows, "hooks_enabled": not args.no_hooks,
    }
    if args.master_port:
        port = args.master_port
    else:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
    os.environ.update(WORLD_SIZE="1", RANK="0", LOCAL_WORLD_SIZE="1", LOCAL_RANK="0",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), DIST_INIT_METHOD="env://")
    result["owns_process_group"] = True
    result["stage"] = "real_worker_constructor"
    worker = TeacherFSDPWorker(config)
    result["execution"]["real_worker_constructed"] = True
    result["stage"] = "real_reset_builds_fsdp2"
    worker.reset()
    result["execution"]["reset_completed"] = True
    result["actual_engine_class"] = f"{type(worker.engine).__module__}.{type(worker.engine).__name__}"
    result["actual_module_class"] = f"{type(worker.engine.module).__module__}.{type(worker.engine.module).__name__}"
    result["actual_fsdp2_cpu_offload_policy"] = worker.engine._uses_fsdp2_cpu_offload_policy
    if not worker.engine._uses_fsdp2_cpu_offload_policy:
        raise RuntimeError("the real engine did not select its forward-only FSDP2 CPU offload policy")
    checkpoint(result, args.out)

    hook_calls = []
    handles = []
    if not args.no_hooks:
        for name, module in (("outer", worker.engine.module), ("backbone", worker.engine.module.model),
                             ("head", worker.engine.module.get_output_embeddings())):
            def pre_hook(module, inputs, label=name):
                shape = list(inputs[0].shape) if inputs and isinstance(inputs[0], torch.Tensor) else None
                hook_calls.append({"module": label, "event": "pre", "input_shape": shape})
            def post_hook(module, inputs, output, label=name):
                hook_calls.append({"module": label, "event": "post"})
            handles.extend((module.register_forward_pre_hook(pre_hook), module.register_forward_hook(post_hook)))
    result["hook_calls"] = hook_calls
    path_messages = []
    class DecisionHandler(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if message.startswith("memory_bounded_teacher "):
                path_messages.append(message)
    handler = DecisionHandler()
    worker_module.logger.setLevel(logging.INFO)
    worker_module.logger.addHandler(handler)
    result["worker_decision_logs"] = path_messages
    try:
        def make_input():
            nested = lambda rows: torch.nested.nested_tensor(
                [torch.tensor(row, dtype=torch.int64) for row in rows], layout=torch.jagged
            )
            data = TensorDict({"input_ids": nested(tokens), "position_ids": nested(fixture["position_ids"]),
                               "topk_ids": nested(candidates)}, batch_size=[2])
            tu.assign_non_tensor(data, temperature=1.0, use_remove_padding=False, use_fused_kernels=False,
                                 teacher_chunk_size=args.gather_chunk_rows)
            return data
        result["stage"] = "real_compute_logprobs_at_ids"
        samples = []
        output = None
        for index in range(args.warmup + args.repeats):
            data = make_input()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved()}
            result["stage"] = f"real_compute_logprobs_at_ids_call_{index}"
            checkpoint(result, args.out)
            started = time.perf_counter()
            try:
                output = worker.compute_logprobs_at_ids(data)
            except ValueError as error:
                if not args.invalid_candidate or "outside the vocabulary" not in str(error):
                    raise
                outer_pre_calls = sum(event["module"] == "outer" and event["event"] == "pre"
                                      for event in hook_calls)
                if outer_pre_calls:
                    raise RuntimeError("invalid candidates reached the outer FSDP forward") from error
                result["checks"] = {"invalid_candidate_rejected": True, "outer_fsdp_pre_calls": outer_pre_calls}
                result["observed_error"] = str(error)
                result["status"] = "PASS_REJECTION"
                checkpoint(result, args.out)
                return
            if args.invalid_candidate:
                raise AssertionError("invalid candidates were unexpectedly accepted")
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            result["execution"]["compute_calls_completed"] += 1
            result["last_completed_output"] = snapshot_output(output)
            signature = compact_output_signature(result["last_completed_output"])
            if expected_signature is None:
                expected_signature = signature
            match = signature == expected_signature
            result["per_call_signatures"].append({"index": index, "phase": "warmup" if index < args.warmup else "measured",
                                                  "signature": signature, "matches_reference": match})
            checkpoint(result, args.out)
            if not match:
                raise RuntimeError("Worker output hash, dtype, shape or offsets differ from the reference")
            if index >= args.warmup:
                samples.append({"host_wall_ms": elapsed, "before": before,
                                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        result["samples"] = samples
        result["latency_median_ms"] = statistics.median(sample["host_wall_ms"] for sample in samples)
        result["measurement_scope"] = (
            "complete direct worker compute_logprobs_at_ids call, including input preparation, real backbone/head, "
            "FSDP2 offload and nested CPU return; optional hooks included; first call may include lazy setup"
        )
        result["stage"] = "validate_real_cpu_outputs"
        if output is None:
            raise RuntimeError("the single collecting rank returned no output")
        keys = set(output.keys())
        if keys != {"teacher_on_student_logp", "teacher_topk_ids"}:
            raise RuntimeError(f"unexpected worker output schema: {sorted(keys)}")
        result["outputs"] = {}
        checks = {"exact_output_keys": True, "finite_logprobs": True, "cpu_values": True,
                  "matching_offsets": True, "compact_value_shapes": True, "no_grad_outputs": True}
        for key in sorted(keys):
            nested = output[key]
            values = nested.values()
            offsets = nested.offsets().cpu().tolist()
            checks["cpu_values"] &= values.device.type == "cpu"
            checks["matching_offsets"] &= offsets == result["input_offsets"]
            checks["compact_value_shapes"] &= list(values.shape) == [sum(args.lengths), args.topk]
            checks["no_grad_outputs"] &= not values.requires_grad
            if key == "teacher_on_student_logp":
                checks["finite_logprobs"] &= bool(torch.isfinite(values).all())
            result["outputs"][key] = {"dtype": str(values.dtype), "device": str(values.device),
                                      "offsets": offsets, "values_shape": list(values.shape),
                                      "values_sha256": tensor_sha256(values),
                                      "values": values.float().tolist() if values.is_floating_point() else values.tolist()}
        observed = re.findall(r"(?:^| )path=([^ ]+)", path_messages[-1]) if path_messages else []
        result["observed_path"] = observed[-1] if observed else "NOT_OBSERVED"
        checks["observed_expected_path"] = result["observed_path"] == expected_path
        if not args.no_hooks:
            calls = args.warmup + args.repeats
            checks["one_outer_and_backbone_per_call"] = all(
                sum(event["module"] == module and event["event"] == event_type for event in hook_calls) == calls
                for module in ("outer", "backbone") for event_type in ("pre", "post")
            )
            head_shapes = [event["input_shape"] for event in hook_calls
                           if event["module"] == "head" and event["event"] == "pre"]
            checks["head_path_shape"] = (
                len(head_shapes) == calls * math.ceil(model_rows / decision.rows)
                and sum(shape[0] for shape in head_shapes) == calls * model_rows
                and all(len(shape) == 2 and shape[0] <= decision.rows for shape in head_shapes)
                if expected_path == "chunked_lm_head" else
                len(head_shapes) == calls and all(len(shape) == 3 and shape[:2] == [2, max(args.lengths)]
                                                  for shape in head_shapes)
            )
            checks["head_post_hooks_complete"] = sum(
                event["module"] == "head" and event["event"] == "post" for event in hook_calls
            ) == len(head_shapes)
        result["checks"] = checks
        if not all(checks.values()):
            raise AssertionError(f"worker smoke checks failed: {checks}")
        result["status"] = "PASS_SMOKE"
    finally:
        worker_module.logger.removeHandler(handler)
        for handle in handles:
            handle.remove()


def main():
    args = parse_args()
    args.verl_dir = args.verl_dir.resolve()
    args.model_dir = args.model_dir.resolve()
    args.out = args.out.resolve()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing evidence: {args.out}")
    result = {
        "status": "RUNNING", "stage": "start", "started_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv, "pid": os.getpid(), "mode": args.mode,
        "execution": {"real_worker_constructed": False, "reset_completed": False, "compute_calls_completed": 0},
        "scope": {"entry": "real_worker_direct_process", "ray_remote_dispatch": "NOT_RUN",
                  "multi_gpu_sharding": "NOT_RUN", "complete_rl_training": "NOT_RUN",
                  "numerical_reference_comparison": "NOT_RUN_by_this_script"},
    }
    try:
        run(args, result)
    except Exception as error:
        result["status"] = "FAIL"
        result["exception"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if isinstance(error, subprocess.CalledProcessError):
            result["exception"]["subprocess_output"] = error.output
            result["exception"]["subprocess_stderr"] = error.stderr
    finally:
        torch = sys.modules.get("torch")
        if result.get("owns_process_group") and torch is not None and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception as error:
                result["cleanup_exception"] = {"type": type(error).__name__, "message": str(error),
                                               "traceback": traceback.format_exc()}
                result["status"] = "FAIL"
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(json_safe(result), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": result["status"], "stage": result["stage"], "out": str(args.out)}))
    return 0 if result["status"] in {"PASS_SMOKE", "PASS_REJECTION"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
