#!/usr/bin/env python3
"""Run actual patched verl Teacher workers directly into a real Student update.

The same in-memory student candidates enter dense-fallback and row-chunked
TeacherFSDPWorker calls. Their returned CPU nested tensors are mapped into the
Student batch and consumed by the pinned compute_reverse_kl_topk function.
Only final Student RMSNorm is trained. This is a local world-size-one check,
not Ray dispatch, full-parameter training, or a complete RL trainer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import weakref

import torch

ROOT = Path(__file__).resolve().parents[1]
VERL_SHA = "87cd055e872f9fdd7b976fcd797a0934b6dba265"
LOSS_PATH = "verl/trainer/distillation/fsdp/losses.py"
LOSS_BLOB = "97d3ebf485a5ce9e1b1a435cce427d6d89284dc6"
MODEL_SHA = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
TOKENIZER_SHA = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
FIXTURES = (
    ("Explain how a GPU computes matrix multiplication. ",
     "It moves matrix tiles into fast memory and accumulates their products."),
    ("Why does causal language modeling use shifted labels? ",
     "Each hidden state predicts the next token without seeing that token."),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value):
    raw = value.detach().cpu().contiguous()
    header = json.dumps({"dtype": str(raw.dtype), "shape": list(raw.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + raw.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def checkpoint(report, output):
    partial = output.with_name(output.name + ".partial.json")
    temporary = partial.with_name(partial.name + ".tmp")
    partial.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(json_safe(report), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, partial)


def local_module(name, relative_path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def padded_fixture(sequences, prompt_lengths, pad_token_id):
    """Use right padding, matching the actual engine's nested-to-padded input."""
    if len(sequences) != len(prompt_lengths) or not sequences:
        raise ValueError("sequence and prompt counts must match and be nonempty")
    lengths = [len(row) for row in sequences]
    if any(not 0 < prompt < length for prompt, length in zip(prompt_lengths, lengths)):
        raise ValueError("each fixture needs a nonempty prompt and response")
    width = max(lengths)
    tokens = torch.full((len(sequences), width), pad_token_id, dtype=torch.int64)
    attention = torch.zeros_like(tokens)
    positions = torch.zeros_like(tokens)
    response_mask = torch.zeros_like(tokens, dtype=torch.bool)
    for row, (sequence, prompt) in enumerate(zip(sequences, prompt_lengths)):
        length = len(sequence)
        tokens[row, :length] = torch.tensor(sequence, dtype=torch.int64)
        attention[row, :length] = 1
        positions[row, :length] = torch.arange(length)
        # Causal-logit row i predicts token i+1; do not shift candidate IDs.
        response_mask[row, prompt - 1:length - 1] = True
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)
    return {"input_ids": tokens, "attention_mask": attention, "position_ids": positions}, response_mask, offsets


def compact_candidates(candidate_ids, attention_mask):
    if candidate_ids.dtype != torch.int64 or candidate_ids.ndim != 3:
        raise ValueError("Student candidate IDs must be int64 [B,T,K]")
    if tuple(candidate_ids.shape[:-1]) != tuple(attention_mask.shape):
        raise ValueError("Student candidate rows must match attention mask")
    return candidate_ids[attention_mask.bool()].detach().cpu()


def restore_worker_values(nested, offsets, attention_mask, candidate_width):
    """Scatter the real returned nested values into the same padded Student rows."""
    if not nested.is_nested:
        raise ValueError("the actual Worker must return nested tensors")
    observed_offsets = nested.offsets().detach().cpu()
    if not torch.equal(observed_offsets, offsets.cpu()):
        raise ValueError("Worker output offsets changed the original sequence layout")
    values = nested.values()
    expected_rows = int(offsets[-1])
    if tuple(values.shape) != (expected_rows, candidate_width):
        raise ValueError("Worker compact shape does not match Student candidate rows")
    if values.device.type != "cpu" or values.requires_grad:
        raise ValueError("Worker output must be detached CPU values")
    mask = attention_mask.bool().cpu()
    if not torch.equal(mask.sum(-1), offsets.diff().cpu()):
        raise ValueError("Student valid-row lengths do not match Worker offsets")
    padded = torch.zeros((*mask.shape, candidate_width), dtype=values.dtype)
    padded[mask] = values
    return padded


def worker_output_snapshot(output):
    return {
        key: {
            "dtype": str(value.values().dtype), "offsets": value.offsets().cpu().tolist(),
            "values_shape": list(value.values().shape), "values_sha256": tensor_hash(value.values()),
            "values": value.values().float().tolist() if value.values().is_floating_point()
            else value.values().tolist(),
        }
        for key, value in output.items()
    }


def response_diagnostics(outputs, response_mask, topk):
    """Aggregate padded outputs with the pinned trainer's response-only semantics.

    Raw outputs remain untouched. The loss range is measured after the reverse-KL
    nonnegative clamp but before loss_max_clamp, as in trainer/distillation/losses.py
    at verl 87cd055 (lines 300-307, 479-516). This is a local-batch diagnostic,
    not distributed Metric aggregation or the optimization loss itself.
    """
    mask = response_mask.bool()
    if not mask.any():
        raise ValueError("response diagnostics require at least one response token")
    if topk <= 0:
        raise ValueError("topk must be positive")
    metrics = {}
    with torch.no_grad():
        for name in ("student_mass", "teacher_mass"):
            valid = outputs[name][mask]
            metrics[f"distillation/{name}"] = valid.mean().item()
            metrics[f"distillation/{name}_min"] = valid.min().item()
            metrics[f"distillation/{name}_max"] = valid.max().item()
        valid_loss = outputs["distillation_losses"][mask].clamp_min(0)
        metrics["distillation/loss_min"] = valid_loss.min().item()
        metrics["distillation/loss_max"] = valid_loss.max().item()
        overlap_count = outputs.get("overlap_count")
        overlap_advantage = outputs.get("overlap_token_advantage")
        if overlap_count is not None and overlap_advantage is not None:
            metrics["distillation/overlap_ratio"] = (overlap_count[mask].float().mean() / topk).item()
            overlap_mask = mask & (overlap_count > 0)
            metrics["distillation/overlap_token_advantage"] = (
                overlap_advantage[overlap_mask].mean().item() if overlap_mask.any() else 0.0
            )
    return metrics


def load_upstream(directory, report):
    head = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if head != VERL_SHA:
        raise ValueError("the actual verl checkout is not at the pinned commit")
    source = directory / LOSS_PATH
    raw = source.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != LOSS_BLOB:
        raise ValueError("the actual upstream loss file differs from its pinned blob")
    sys.path.insert(0, str(directory))
    sys.path.insert(0, str(ROOT))
    loss_module = importlib.import_module("verl.trainer.distillation.fsdp.losses")
    worker_module = importlib.import_module("verl.workers.teacher.fsdp_teacher_worker")
    if Path(inspect.getfile(loss_module.compute_reverse_kl_topk)).resolve() != source.resolve():
        raise RuntimeError("the loss function was imported from a different checkout")
    expected_worker = directory / "verl/workers/teacher/fsdp_teacher_worker.py"
    if Path(inspect.getfile(worker_module.TeacherFSDPWorker)).resolve() != expected_worker.resolve():
        raise RuntimeError("the Worker was imported from a different checkout")
    report["upstream"] = {
        "commit": head, "loss_git_blob": blob, "loss_sha256": sha256(source),
        "worker_sha256": sha256(expected_worker),
        "engine_sha256": sha256(directory / "verl/workers/engine/fsdp/transformer_impl.py"),
        "loss_function": f"{loss_module.__name__}.compute_reverse_kl_topk",
        "worker_class": f"{worker_module.__name__}.TeacherFSDPWorker",
    }
    return loss_module.compute_reverse_kl_topk, worker_module

def run_teacher(mode, args, model_config, tokens, positions, candidate_rows, offsets, report):
    from tensordict import TensorDict
    from verl.trainer.config import CheckpointConfig
    from verl.utils import tensordict_utils as tu
    from verl.utils.profiler import ProfilerConfig
    from verl.workers.config import FSDPEngineConfig, TrainingWorkerConfig
    worker_module = importlib.import_module("verl.workers.teacher.fsdp_teacher_worker")
    total_model_rows = len(tokens) * max(map(len, tokens))
    rows = total_model_rows if mode == "dense" else args.chunk_rows
    engine = FSDPEngineConfig(
        strategy="fsdp2", forward_only=True, model_dtype=args.dtype, dtype=args.dtype,
        mixed_precision={"param_dtype": args.dtype, "reduce_dtype": "float32", "buffer_dtype": "float32"},
        fsdp_size=1, ulysses_sequence_parallel_size=1, use_torch_compile=False,
        use_dynamic_bsz=False, infer_micro_batch_size_per_gpu=len(tokens),
        teacher_memory_bounded=True, teacher_lm_head_chunk_rows=rows,
        teacher_head_memory_budget_mib=None,
    )
    config = TrainingWorkerConfig(
        model_type="language_model", model_config=model_config, engine_config=engine,
        optimizer_config=None, checkpoint_config=CheckpointConfig(),
        profiler_config=ProfilerConfig(tool=None, enable=False, save_path=None, tool_config={}),
    )
    # Each worker owns a separate immutable config. Both are constructed in the
    # same process group; the caller destroys that group once at final cleanup.
    worker = None
    handler = None
    handles = []
    observed = {"path_logs": [], "head_shapes": [], "head_post": 0, "outer_pre": 0, "outer_post": 0}
    result = {"status": "RUNNING", "requested_chunk_rows": rows, "observed": observed}
    report.setdefault("teacher_runs", {})[mode] = result
    expected_path = "dense_full_row_fallback" if mode == "dense" else "chunked_lm_head"

    class PathHandler(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if message.startswith("memory_bounded_teacher "):
                observed["path_logs"].append(message)

    try:
        report["stage"] = f"{mode}_worker_constructor"
        checkpoint(report, args.out)
        worker = worker_module.TeacherFSDPWorker(config)
        worker.reset()
        result["actual_engine"] = f"{type(worker.engine).__module__}.{type(worker.engine).__name__}"
        result["actual_module"] = f"{type(worker.engine.module).__module__}.{type(worker.engine.module).__name__}"
        result["cpu_offload_policy"] = bool(worker.engine._uses_fsdp2_cpu_offload_policy)
        if not result["cpu_offload_policy"]:
            raise RuntimeError("the expected real forward-only FSDP2 CPU-offload policy was not selected")
        handler = PathHandler()
        worker_module.logger.setLevel(logging.INFO)
        worker_module.logger.addHandler(handler)
        if not args.no_hooks:
            def outer_pre(module, positional):
                observed["outer_pre"] += 1

            def outer_post(module, positional, output):
                observed["outer_post"] += 1

            def head_pre(module, positional):
                observed["head_shapes"].append(list(positional[0].shape))

            def head_post(module, positional, output):
                observed["head_post"] += 1

            handles = [
                worker.engine.module.register_forward_pre_hook(outer_pre),
                worker.engine.module.register_forward_hook(outer_post),
                worker.engine.module.get_output_embeddings().register_forward_pre_hook(head_pre),
                worker.engine.module.get_output_embeddings().register_forward_hook(head_post),
            ]
        nested = lambda rows: torch.nested.nested_tensor(
            [torch.tensor(row, dtype=torch.int64) for row in rows], layout=torch.jagged)
        data = TensorDict({
            "input_ids": nested(tokens), "position_ids": nested(positions), "topk_ids": nested(candidate_rows),
        }, batch_size=[len(tokens)])
        tu.assign_non_tensor(data, temperature=1.0, use_remove_padding=False, use_fused_kernels=False,
                             teacher_chunk_size=args.gather_chunk_rows)
        if not torch.equal(data["input_ids"].offsets(), offsets):
            raise ValueError("the actual nested Worker input changed offsets")
        report["stage"] = f"{mode}_real_worker_compute"
        checkpoint(report, args.out)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved()}
        start = time.perf_counter()
        output = worker.compute_logprobs_at_ids(data)
        torch.cuda.synchronize()
        result["diagnostic_measurement"] = {
            "host_ms": (time.perf_counter() - start) * 1000, "before": before,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "scope": "one actual Worker call with a resident Student; not an isolated performance comparison",
        }
        if output is None or set(output.keys()) != {"teacher_on_student_logp", "teacher_topk_ids"}:
            raise RuntimeError("the actual collecting Worker returned an unexpected schema")
        if (output["teacher_on_student_logp"].values().dtype != getattr(torch, args.dtype)
                or output["teacher_topk_ids"].values().dtype != torch.int64):
            raise RuntimeError("the actual Worker output dtypes changed the scoring contract")
        if not torch.isfinite(output["teacher_on_student_logp"].values()).all():
            raise ValueError("the actual Worker returned nonfinite teacher scores")
        result["output"] = worker_output_snapshot(output)
        if not any(f"path={expected_path} " in message for message in observed["path_logs"]):
            raise RuntimeError("the expected actual Worker execution path was not observed")
        if not args.no_hooks:
            if (observed["outer_pre"], observed["outer_post"]) != (1, 1):
                raise RuntimeError("the actual outer FSDP call did not complete normally")
            shapes = observed["head_shapes"]
            if mode == "chunked":
                head_ok = (len(shapes) == math.ceil(total_model_rows / rows)
                           and all(len(shape) == 2 and shape[0] <= rows for shape in shapes)
                           and sum(shape[0] for shape in shapes) == total_model_rows)
            else:
                head_ok = len(shapes) == 1 and shapes[0][:2] == [len(tokens), max(map(len, tokens))]
            if not head_ok or observed["head_post"] != len(shapes):
                raise RuntimeError("the actual head calls did not match the selected execution path")
        result["teacher_gradients_absent"] = all(parameter.grad is None
                                                for parameter in worker.engine.module.parameters())
        if not result["teacher_gradients_absent"]:
            raise RuntimeError("the actual frozen Teacher unexpectedly accumulated gradients")
        result["status"] = "PASS_CALL"
        checkpoint(report, args.out)
        # Return the actual in-memory TensorDict. No JSON serialization/reload is
        # involved in the data passed to the loss.
        return output
    finally:
        for handle in handles:
            handle.remove()
        if handler is not None:
            worker_module.logger.removeHandler(handler)
        if worker is not None:
            had_error = sys.exc_info()[0] is not None
            reference = weakref.ref(worker)
            worker = None
            try:
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                result["worker_object_collected"] = reference() is None
                result["after_release_allocated_bytes"] = torch.cuda.memory_allocated()
                if not result["worker_object_collected"] and not had_error:
                    raise RuntimeError("the previous Teacher Worker was not released")
            except Exception as cleanup_error:
                result["cleanup_error"] = {"type": type(cleanup_error).__name__, "message": str(cleanup_error)}
                if not had_error:
                    raise


def update_student(student, initial_norm, teacher_output, inputs, ids, response_mask, offsets, upstream, args):
    norm = student.model.norm.weight
    with torch.no_grad():
        norm.copy_(initial_norm)
    student.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW([norm], lr=args.learning_rate, weight_decay=0)
    teacher_logp = restore_worker_values(
        teacher_output["teacher_on_student_logp"], offsets, inputs["attention_mask"].cpu(), args.topk)
    teacher_ids = restore_worker_values(
        teacher_output["teacher_topk_ids"], offsets, inputs["attention_mask"].cpu(), args.topk)
    if not torch.isfinite(teacher_logp).all():
        raise ValueError("the actual Worker returned nonfinite teacher scores")
    logits = student(**inputs, use_cache=False).logits
    temperature = torch.full(logits.shape[:-1], 0.8, dtype=torch.float32, device=logits.device)
    logits.div_(temperature.clamp_min(1e-8).unsqueeze(-1).to(logits.dtype))
    logits.retain_grad()
    config = SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=-10))
    # These scores are the current real Worker's CPU tensors, scattered only to
    # the original padded rows; the FP32 cast follows the fixed trainer wire.
    outputs = upstream(
        student_logits=logits, teacher_on_student_logp=teacher_logp.to(logits.device).detach().float(),
        student_topk_ids=ids, config=config, data_format="bshd",
        teacher_topk_ids=teacher_ids.to(logits.device),
    )
    per_token = outputs["distillation_losses"].clamp_min(0).clamp(-10, 10)
    loss = per_token[response_mask].mean()
    loss.backward()
    if norm.grad is None or logits.grad is None:
        raise RuntimeError("the actual Student path did not produce required gradients")
    norm_gradient = norm.grad.detach().cpu().float().clone()
    logits_gradient = logits.grad.detach().cpu().float().clone()
    optimizer.step()
    updated = norm.detach().cpu().float().clone()
    initial = initial_norm.detach().cpu().float()
    values = {
        "loss": loss.detach().cpu().float(), "norm_gradient": norm_gradient,
        "logits_gradient": logits_gradient, "updated_norm": updated, "norm_update": updated - initial,
        **{name: value.detach().cpu() for name, value in outputs.items()},
    }
    raw = {
        "loss": loss.item(), "norm_gradient": norm_gradient.tolist(), "updated_norm": updated.tolist(),
        "gradient_l2": norm_gradient.norm().item(),
        "gradient_nonzero_elements": norm_gradient.count_nonzero().item(),
        "changed_parameter_elements": (updated != initial).sum().item(),
        "parameter_delta_l2": (updated - initial).norm().item(),
        "hashes": {name: tensor_hash(value) for name, value in values.items()},
        # Keep full tensors as evidence; only response_metrics are aggregate diagnostics.
        "diagnostics": {name: value.detach().cpu().tolist() for name, value in outputs.items()},
        "diagnostics_scope": "raw unmasked per-token evidence; prompt/padding values are not metrics",
        "response_metrics": response_diagnostics(outputs, response_mask, args.topk),
        "response_metric_scope": "local batch; response mask, plus overlap_count > 0 for advantage",
    }
    del logits, optimizer, teacher_logp, teacher_ids
    return values, raw


def compare_updates(first, second, dtype):
    helper = local_module("_i1_student_consumer", "scripts/teacher_consumer_step.py")
    atol, rtol = helper.TOLERANCES[dtype]
    checks = {}
    for name in first:
        if name == "overlap_count":
            checks[name] = {"pass": torch.equal(first[name], second[name]), "kind": "exact_integer"}
        elif "gradient" in name or name == "norm_update":
            checks[name] = helper.compare_gradient(first[name], second[name], dtype)
        else:
            checks[name] = helper.compare(first[name], second[name], atol, rtol)
    return {"pass": all(value["pass"] for value in checks.values()), "checks": checks}

def run(args, report):
    import ray
    import transformers
    from transformers import AutoModelForCausalLM
    from verl.workers.config import HFModelConfig

    upstream, _ = load_upstream(args.verl_dir, report)
    if torch.distributed.is_initialized() or ray.is_initialized():
        raise RuntimeError("run this direct-process check in a fresh process")
    if not torch.cuda.is_available():
        raise RuntimeError("the real Worker path requires CUDA; CPU helpers are tested separately")
    if any(name.startswith("RAY_EXPERIMENTAL_NOSET_") and value for name, value in os.environ.items()):
        raise RuntimeError("Ray actor-only visible-device settings cannot be used for this direct invocation")
    torch.cuda.set_device(0)
    torch.manual_seed(731)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report["environment"] = {
        "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "total_gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    model_hashes = {name: sha256(args.model_dir / name)
                    for name in ("model.safetensors", "tokenizer.json", "config.json")}
    if model_hashes["model.safetensors"] != MODEL_SHA or model_hashes["tokenizer.json"] != TOKENIZER_SHA:
        raise ValueError("the model/tokenizer files do not match the fixed Qwen snapshot")
    report["model"] = {"directory": str(args.model_dir), "sha256": model_hashes}
    report["stage"] = "real_student_and_shared_fixture"
    checkpoint(report, args.out)
    dtype = getattr(torch, args.dtype)
    teacher_model_config = HFModelConfig(
        path=str(args.model_dir), use_remove_padding=False, enable_gradient_checkpointing=False,
        use_fused_kernels=False, override_config={"attn_implementation": "eager", "use_cache": False},
    )
    tokenizer = teacher_model_config.tokenizer
    if args.topk > teacher_model_config.hf_config.vocab_size:
        raise ValueError("topk exceeds the fixed model vocabulary")
    sequences, prompt_lengths = [], []
    for prompt, response in FIXTURES:
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        suffix = tokenizer.encode(response, add_special_tokens=False)
        sequence = (prefix + suffix)[:args.max_tokens]
        sequences.append(sequence)
        prompt_lengths.append(len(prefix))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("the tokenizer has no padding or EOS ID")
    cpu_inputs, cpu_response_mask, offsets = padded_fixture(sequences, prompt_lengths, pad)
    student = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=dtype, attn_implementation="eager"
    ).to("cuda").eval()
    student.requires_grad_(False)
    student.model.norm.weight.requires_grad_(True)
    initial_norm = student.model.norm.weight.detach().clone()
    inputs = {name: value.to("cuda") for name, value in cpu_inputs.items()}
    response_mask = cpu_response_mask.to("cuda")
    with torch.no_grad():
        logits = student(**inputs, use_cache=False).logits
        temperatures = torch.full(logits.shape[:-1], 0.8, device="cuda", dtype=torch.float32)
        logits.div_(temperatures.clamp_min(1e-8).unsqueeze(-1).to(logits.dtype))
        ids = logits.log_softmax(-1).topk(args.topk, dim=-1).indices
        del logits, temperatures
    valid_candidates = compact_candidates(ids.cpu(), cpu_inputs["attention_mask"])
    candidate_rows = [valid_candidates[int(offsets[index]):int(offsets[index + 1])].tolist()
                      for index in range(len(sequences))]
    model_rows = cpu_inputs["input_ids"].numel()
    if not args.chunk_rows < model_rows:
        raise ValueError("chunk rows must be below the actual padded model-row count")
    report["fixture"] = {
        "kind": "tokenized fixed prompt/response text; synthetic, not sampled RL trajectories",
        "texts": FIXTURES, "tokens": sequences, "prompt_lengths": prompt_lengths,
        "input_offsets": offsets.tolist(), "model_rows_including_padding": model_rows,
        "active_response_rows": cpu_response_mask.sum().item(),
        "inputs": {name: value.tolist() for name, value in cpu_inputs.items()},
        "response_logit_row_mask": cpu_response_mask.tolist(),
        "student_candidate_ids": ids.cpu().tolist(),
        "hashes": {**{name: tensor_hash(value) for name, value in cpu_inputs.items()},
                   "response_mask": tensor_hash(cpu_response_mask), "candidate_ids": tensor_hash(ids)},
        "causal_alignment": "same logit rows; only response mask is next-token aligned",
    }
    report["student"] = {
        "temperature": 0.8, "teacher_temperature": 1.0,
        "trainable_parameters": [name for name, value in student.named_parameters() if value.requires_grad],
        "optimizer": {"type": "AdamW", "learning_rate": args.learning_rate, "weight_decay": 0},
        "initial_norm_sha256": tensor_hash(initial_norm),
    }
    if args.master_port:
        port = args.master_port
    else:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
    os.environ.update(WORLD_SIZE="1", RANK="0", LOCAL_WORLD_SIZE="1", LOCAL_RANK="0",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), DIST_INIT_METHOD="env://")
    report["owns_process_group"] = True
    checkpoint(report, args.out)
    positions = [list(range(len(row))) for row in sequences]
    worker_outputs = {}
    for mode in ("dense", "chunked"):
        worker_outputs[mode] = run_teacher(
            mode, args, teacher_model_config, sequences, positions, candidate_rows, offsets, report)
    helper = local_module("_i1_student_consumer", "scripts/teacher_consumer_step.py")
    atol, rtol = helper.TOLERANCES[args.dtype]
    dense_values = worker_outputs["dense"]["teacher_on_student_logp"].values()
    chunked_values = worker_outputs["chunked"]["teacher_on_student_logp"].values()
    teacher_checks = {
        "candidate_logprobs": helper.compare(dense_values, chunked_values, atol, rtol),
        "topk_set": {"pass": torch.equal(
            worker_outputs["dense"]["teacher_topk_ids"].values().sort(-1).values,
            worker_outputs["chunked"]["teacher_topk_ids"].values().sort(-1).values),
            "scope": "strict set equality; exact boundary ties may conservatively fail"},
    }
    report["teacher_comparison"] = teacher_checks
    updates = {}
    report["student_updates"] = {}
    for mode in ("dense", "chunked"):
        report["stage"] = f"{mode}_actual_upstream_loss_and_student_update"
        checkpoint(report, args.out)
        updates[mode], raw = update_student(
            student, initial_norm, worker_outputs[mode], inputs, ids, response_mask, offsets, upstream, args)
        report["student_updates"][mode] = raw
        checkpoint(report, args.out)
    report["student_comparison"] = compare_updates(updates["dense"], updates["chunked"], args.dtype)
    report["checks"] = {
        "teacher_scores_within_frozen_budget": teacher_checks["candidate_logprobs"]["pass"],
        "teacher_topk_sets_equal": teacher_checks["topk_set"]["pass"],
        "student_outputs_gradients_and_update_within_budget": report["student_comparison"]["pass"],
        "both_steps_have_nonzero_gradient": all(
            value["gradient_l2"] > 0 and math.isfinite(value["gradient_l2"])
            for value in report["student_updates"].values()),
        "both_steps_change_a_real_parameter": all(
            value["changed_parameter_elements"] > 0 for value in report["student_updates"].values()),
    }
    report["status"] = "PASS" if all(report["checks"].values()) else "FAIL"
    report["stage"] = "completed"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--chunk-rows", type=int, default=8)
    parser.add_argument("--gather-chunk-rows", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--no-hooks", action="store_true")
    args = parser.parse_args()
    for name in ("verl_dir", "model_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    partial = args.out.with_name(args.out.name + ".partial.json")
    if args.out.is_relative_to(ROOT.resolve()) or args.out.exists() or partial.exists():
        parser.error("--out must be a new file outside the repository, without a previous partial result")
    if args.topk <= 0 or args.chunk_rows <= 0 or args.gather_chunk_rows <= 0 or args.max_tokens <= 1:
        parser.error("token and chunk counts must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if not 0 <= args.master_port <= 65535:
        parser.error("master-port must be between 0 and 65535")
    return args


def main():
    args = parse_args()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(args.verl_dir))
    os.environ.setdefault("VERL_LOGGING_LEVEL", "INFO")
    report = {
        "status": "RUNNING", "stage": "start", "argv": sys.argv,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": __doc__,
        "sources": {name: sha256(ROOT / name) for name in (
            "scripts/worker_to_student_step.py", "scripts/teacher_consumer_step.py",
            "reference/memory_bounded_teacher.py", "reference/memory_budget.py",
            "integrations/verl_6920/bridge.py")},
        "limitations": [
            "direct local Worker invocation; Ray dispatch and full trainer remain NOT_RUN",
            "one GPU, FSDP2 CPU offload, tied Qwen2, padded attention",
            "only Student final RMSNorm is trained; no full-parameter or RL update claim",
            "single diagnostic timings include a resident Student; no performance comparison claim",
            "the initial CUDA unknown error is not explained by a successful run here",
        ],
    }
    try:
        run(args, report)
    except Exception as error:
        report["status"] = "ERROR"
        report["exception"] = {
            "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        if report.get("owns_process_group") and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception as error:
                report["cleanup_exception"] = {
                    "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
                report["status"] = "ERROR"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(json_safe(report), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": report["status"], "out": str(args.out), "stage": report["stage"]}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())