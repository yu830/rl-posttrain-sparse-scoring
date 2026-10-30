#!/usr/bin/env python3
"""One fresh-process Dense/Chunked x FSDP2 CPU-offload condition.

This is a real complete Qwen model/FSDP forward, not a verl Worker. Run one
condition per process and retain its exit code/stderr externally: a native
CUDA abort can bypass Python cleanup, but the fsynced JSONL journal survives.
All-device nvidia-smi observations include other processes and are sampled,
not an exact process VRAM peak. The optional allocator fraction limits only
PyTorch's caching allocator, not all allocations or the entire GPU.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference.memory_bounded_teacher import install_memory_bounded_teacher
from scripts.full_teacher_benchmark import (
    MODEL_REVISION, MODEL_SHA256, TOKENIZER_SHA256, PROMPTS,
    candidate_scores, dense_scores, is_out_of_memory, sha256, wrap_fsdp2,
)


class OutputMismatch(RuntimeError):
    pass


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    header = json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(header + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


class JsonlJournal:
    """One task-owned stream; every complete event is flushed and fsynced."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open("x", encoding="utf-8")
        self.lock = threading.Lock()
        self.sequence = 0

    def record(self, event, **fields):
        with self.lock:
            item = {"sequence": self.sequence, "event": event,
                    "unix_time_ns": time.time_ns(), "perf_counter_ns": time.perf_counter_ns(), **fields}
            self.stream.write(json.dumps(json_safe(item), ensure_ascii=False, allow_nan=False) + "\n")
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.sequence += 1

    def close(self):
        with self.lock:
            self.stream.close()


def policy_description(policy):
    from torch.distributed.fsdp import CPUOffloadPolicy
    return {"class": f"{type(policy).__module__}.{type(policy).__name__}",
            "cpu_offload": isinstance(policy, CPUOffloadPolicy),
            "pin_memory": getattr(policy, "pin_memory", None)}


def describe_actual_fsdp_policies(model, requested):
    groups = []
    for name, module in [("root", model), *[(f"decoder_{i}", x) for i, x in enumerate(model.model.layers)]]:
        for group in module._get_fsdp_state()._fsdp_param_groups:
            groups.append({"module": name, **policy_description(group.offload_policy)})
    if not groups or any(group["cpu_offload"] != requested for group in groups):
        raise RuntimeError("the actual FSDP2 offload policies do not match the requested condition")
    return groups


def cuda_memory():
    free, total = torch.cuda.mem_get_info()
    return {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cuda_free_bytes": free, "cuda_total_bytes": total}


def safe_cuda_memory():
    try:
        return cuda_memory()
    except Exception as error:
        return {"acquisition": "UNAVAILABLE", "error": repr(error)}


def read_device_observation(device_index):
    query = ["nvidia-smi", f"--id={device_index}",
             "--query-gpu=timestamp,index,name,memory.used,memory.free,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"]
    completed = subprocess.run(query, text=True, capture_output=True, timeout=5, check=False)
    result = {"command": query, "exit_code": completed.returncode,
              "stdout": completed.stdout, "stderr": completed.stderr}
    if completed.returncode:
        result["status"] = "UNAVAILABLE"
        return result
    rows = list(csv.reader(completed.stdout.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != 7:
        result.update(status="UNAVAILABLE", error="unexpected nvidia-smi CSV shape")
        return result
    fields = [value.strip() for value in rows[0]]
    try:
        result.update(status="OBSERVED", timestamp=fields[0], index=int(fields[1]), name=fields[2],
                      memory_used_mib=float(fields[3]), memory_free_mib=float(fields[4]),
                      memory_total_mib=float(fields[5]), utilization_percent=float(fields[6]))
    except ValueError:
        result.update(status="UNAVAILABLE", error="nvidia-smi returned nonnumeric fields")
    return result


class DeviceSampler:
    def __init__(self, interval_ms, device_index, journal):
        self.interval = interval_ms / 1000
        self.device_index, self.journal = device_index, journal
        self.stop_event = threading.Event()
        self.thread = None
        self.observations = []
        self.failure = None

    def sample(self):
        try:
            observation = read_device_observation(self.device_index)
            self.observations.append(observation)
            self.journal.record("all_device_memory_observation", observation=observation)
            if observation["status"] != "OBSERVED":
                self.failure = observation
                return False
            return True
        except Exception as error:
            self.failure = {"status": "UNAVAILABLE", "error": repr(error), "traceback": traceback.format_exc()}
            self.journal.record("all_device_memory_unavailable", observation=self.failure)
            return False

    def start(self):
        if self.interval <= 0:
            self.sample()
            return
        def loop():
            while not self.stop_event.is_set():
                if not self.sample():
                    break
                self.stop_event.wait(self.interval)
        self.thread = threading.Thread(target=loop, name="task-nvidia-smi-sampler", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=6)
            if self.thread.is_alive():
                self.failure = {"status": "UNAVAILABLE", "error": "sampler did not stop within its subprocess timeout"}
        if self.thread is None and self.failure is None:
            self.sample()
        valid = [item for item in self.observations if item.get("status") == "OBSERVED"]
        return {"interval_ms": self.interval * 1000, "sample_count": len(self.observations),
                "scope": "sampled all-device use including other processes; not an exact process peak",
                "max_observed_memory_used_mib": max((x["memory_used_mib"] for x in valid), default=None),
                "min_observed_memory_free_mib": min((x["memory_free_mib"] for x in valid), default=None),
                "failure": self.failure,
                "instrumentation_overhead": "host sampling and journal I/O overlap model calls"}


def compact_signature(output, valid_mask, lengths, vocab_size, k, logits_dtype):
    if valid_mask.dtype != torch.bool or valid_mask.sum(-1).tolist() != list(lengths):
        raise OutputMismatch("valid-mask lengths do not match offsets")
    if output.candidate_logprobs.dtype != logits_dtype:
        raise OutputMismatch("candidate logprob dtype differs from the head contract")
    expected_shape = (*valid_mask.shape, k)
    if tuple(output.candidate_logprobs.shape) != expected_shape or tuple(output.teacher_topk_ids.shape) != expected_shape:
        raise OutputMismatch("compact output shape differs from padded input rows")
    logp = output.candidate_logprobs[valid_mask]
    topk = output.teacher_topk_ids[valid_mask]
    if logp.device.type != "cpu" or topk.device.type != "cpu":
        raise OutputMismatch("the measured path did not return compact CPU outputs")
    if not torch.isfinite(logp).all() or topk.dtype != torch.int64:
        raise OutputMismatch("nonfinite candidate output or incorrect Top-K dtype")
    if not 0 <= topk.min().item() <= topk.max().item() < vocab_size:
        raise OutputMismatch("teacher Top-K IDs are outside the vocabulary")
    if topk.shape[-1] > 1 and not topk.sort(-1).values.diff(dim=-1).ne(0).all():
        raise OutputMismatch("teacher Top-K contains duplicate IDs")
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return {"padded_shape": list(expected_shape), "values_shape": list(logp.shape), "offsets": offsets,
            "candidate_logprobs": {"dtype": str(logp.dtype), "sha256": tensor_hash(logp)},
            "teacher_topk_ids": {"dtype": str(topk.dtype), "sha256": tensor_hash(topk)},
            "finite_logprobs": True, "teacher_topk_unique_and_in_range": True}


def compare_signatures(actual, expected):
    return {"match": actual == expected,
            "candidate_logprobs_hash_equal": actual["candidate_logprobs"] == expected["candidate_logprobs"],
            "teacher_topk_hash_equal": actual["teacher_topk_ids"] == expected["teacher_topk_ids"],
            "shape_offsets_equal": all(actual[key] == expected[key] for key in ("padded_shape", "values_shape", "offsets"))}


def validate_head_calls(calls, mode, batch, width, chunk_rows):
    rows = batch * width
    if not calls:
        return False
    if mode == "dense":
        return len(calls) == 1 and calls[0]["input_shape"][:2] == [batch, width] and len(calls[0]["input_shape"]) == 3
    shapes = [call["input_shape"] for call in calls]
    return (len(shapes) == math.ceil(rows / chunk_rows)
            and all(len(shape) == 2 and 0 < shape[0] <= chunk_rows for shape in shapes)
            and sum(shape[0] for shape in shapes) == rows)


@contextmanager
def observe_calls(model, journal, trace_hooks):
    events, handles = [], []
    for label, module in (("outer", model), ("backbone", model.model), ("head", model.lm_head)):
        def pre(module, args, kwargs, name=label):
            value = args[0] if args else kwargs.get("input_ids", kwargs.get("inputs_embeds"))
            event = {"module": name, "event": "pre", "input_shape": list(value.shape) if value is not None else None}
            events.append(event)
            if trace_hooks:
                journal.record("forward_hook", hook=event)
        def post(module, args, kwargs, output, name=label):
            event = {"module": name, "event": "post"}
            if name == "head":
                event["output_shape"] = list(output.shape)
            events.append(event)
            if trace_hooks:
                journal.record("forward_hook", hook=event)
        handles.extend((module.register_forward_pre_hook(pre, with_kwargs=True),
                        module.register_forward_hook(post, with_kwargs=True)))
    try:
        yield events
    finally:
        for handle in handles:
            handle.remove()


def build_fixture(tokenizer, lengths, k, vocab_size, device, recorded=None):
    width = max(lengths)
    # The pinned Worker fixture used prepare_model_inputs' default pad_token_id=0.
    # Reproduce padded input bytes, not just the unpadded tokens and mask.
    pad_id = 0 if recorded is not None else (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("the tokenizer has no padding/EOS token")
    tokens = torch.full((len(lengths), width), pad_id, dtype=torch.int64)
    mask = torch.zeros_like(tokens)
    positions = torch.zeros_like(tokens)
    ids = torch.zeros((*tokens.shape, k), dtype=torch.int64)
    if recorded is not None:
        source = recorded.get("fixture", recorded)
        for field in ("input_ids", "position_ids", "topk_ids"):
            if not isinstance(source.get(field), list) or len(source[field]) != len(lengths):
                raise ValueError(f"recorded fixture {field} batch differs from requested lengths")
        for row, length in enumerate(lengths):
            raw_tokens, raw_positions, raw_ids = (source[field][row] for field in
                                                  ("input_ids", "position_ids", "topk_ids"))
            if len(raw_tokens) != length or len(raw_positions) != length or len(raw_ids) != length:
                raise ValueError("recorded fixture sequence length differs from requested lengths")
            if any(len(candidates) != k for candidates in raw_ids):
                raise ValueError("recorded fixture candidate width differs from K")
            all_ids = [*raw_tokens, *raw_positions, *[token for candidates in raw_ids for token in candidates]]
            if any(type(value) is not int for value in all_ids):
                raise ValueError("recorded fixture must contain integer token/position/candidate IDs")
            if any(not 0 <= value < vocab_size for value in [*raw_tokens, *[x for candidates in raw_ids for x in candidates]]):
                raise ValueError("recorded fixture has token/candidate IDs outside the vocabulary")
            if any(value < 0 for value in raw_positions):
                raise ValueError("recorded fixture has negative position IDs")
            tokens[row, :length] = torch.tensor(raw_tokens)
            positions[row, :length] = torch.tensor(raw_positions)
            ids[row, :length] = torch.tensor(raw_ids)
            mask[row, :length] = 1
    else:
        for row, length in enumerate(lengths):
            text_tokens = tokenizer.encode(PROMPTS[row % len(PROMPTS)], add_special_tokens=False)
            values = (text_tokens * math.ceil(length / len(text_tokens)))[:length]
            tokens[row, :length] = torch.tensor(values)
            mask[row, :length] = 1
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        stride = 7919 if math.gcd(7919, vocab_size) == 1 else 1
        ids = (tokens.unsqueeze(-1) + torch.arange(k) * stride
               + torch.arange(width).reshape(1, width, 1) * 17) % vocab_size
    identity = {"lengths": list(lengths), "k": k, "padded_rows": tokens.numel(),
                "pad_token_id": pad_id, "valid_rows": int(mask.sum()), "input_ids_sha256": tensor_hash(tokens),
                "attention_mask_sha256": tensor_hash(mask), "position_ids_sha256": tensor_hash(positions),
                "candidate_ids_sha256": tensor_hash(ids),
                "kind": ("recorded worker fixture, replayed through a direct model/FSDP entry" if recorded is not None else
                         "deterministic synthetic tokenized text and vocabulary candidates; not RL trajectories")}
    return ({"input_ids": tokens.to(device), "attention_mask": mask.to(device),
             "position_ids": positions.to(device), "use_cache": False},
            ids.to(device), mask.bool(), identity)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new final JSON path outside the repository")
    parser.add_argument("--events", type=Path, help="new JSONL path, default OUT.events.jsonl")
    parser.add_argument("--mode", choices=("dense", "chunked"), required=True)
    parser.add_argument("--cpu-offload", choices=("on", "off"), required=True)
    parser.add_argument("--purpose", choices=("stability", "capacity"), default="stability")
    parser.add_argument("--rows", type=int, default=1024, help="single-sequence length unless --lengths is supplied")
    parser.add_argument("--lengths", type=int, nargs="+", help="padded independent sequence lengths; no packed attention")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--lse-chunk", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--buffer-cast", choices=("worker", "preserve"), default="worker",
                        help="worker reproduces the pinned engine model.to(dtype) before CUDA")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--allocator-fraction", type=float,
                        help="PyTorch caching allocator fraction only; not total VRAM or a device hard cap")
    parser.add_argument("--fixture", type=Path, help="recorded Worker JSON fixture; pass matching --lengths and --k")
    parser.add_argument("--expected", type=Path, help="prior result JSON for the identical fixture/output contract")
    parser.add_argument("--save-output", type=Path, help="save first compact CPU output to this new torch file")
    parser.add_argument("--sample-device-ms", type=float, default=0,
                        help="optional nvidia-smi all-device sampling; overlaps host timing")
    parser.add_argument("--no-hooks", action="store_true")
    parser.add_argument("--trace-hooks", action="store_true", help="fsync every hook for native-abort diagnosis")
    args = parser.parse_args(argv)
    args.lengths = args.lengths or [args.rows]
    args.warmup = (2 if args.purpose == "stability" else 0) if args.warmup is None else args.warmup
    args.repeats = (30 if args.purpose == "stability" else 1) if args.repeats is None else args.repeats
    if min(args.lengths + [args.k, args.chunk_rows, args.lse_chunk, args.repeats]) <= 0 or args.warmup < 0:
        parser.error("lengths/K/chunks/repeats must be positive; warmup must be nonnegative")
    if args.allocator_fraction is not None and (not math.isfinite(args.allocator_fraction) or not 0 < args.allocator_fraction <= 1):
        parser.error("allocator-fraction must be finite in (0,1]")
    if not math.isfinite(args.sample_device_ms) or args.sample_device_ms < 0:
        parser.error("sample-device-ms must be finite and nonnegative")
    if args.trace_hooks and args.no_hooks:
        parser.error("--trace-hooks requires hooks")
    args.model_dir = args.model_dir.resolve()
    if args.fixture is not None:
        args.fixture = args.fixture.resolve()
    args.out = args.out.resolve()
    args.events = (args.events or args.out.with_name(args.out.name + ".events.jsonl")).resolve()
    if args.out == args.events or any(path.is_relative_to(ROOT) for path in (args.out, args.events)):
        parser.error("out and events must be distinct paths outside the source repository")
    paths = [args.out, args.events]
    if args.save_output is not None:
        args.save_output = args.save_output.resolve()
        paths.append(args.save_output)
        if args.save_output.is_relative_to(ROOT) or len(set(paths)) != len(paths):
            parser.error("save-output must be a distinct path outside the repository")
    if any(path.exists() for path in paths):
        parser.error("output evidence already exists; use new paths")
    return args


def run(args, result, journal):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU performance substitution")
    if torch.distributed.is_initialized():
        raise RuntimeError("run this probe in a fresh process without an existing process group")
    torch.cuda.set_device(0)
    if args.allocator_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.allocator_fraction, 0)
    torch.manual_seed(20261030)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result.update(torch=torch.__version__, transformers=transformers.__version__, cuda_runtime=torch.version.cuda,
                  device_name=torch.cuda.get_device_name(), device_capability=list(torch.cuda.get_device_capability()),
                  selected_environment={key: os.environ.get(key) for key in
                                        ("CUDA_LAUNCH_BLOCKING", "TORCH_SHOW_CPP_STACKTRACES",
                                         "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")})
    result["memory_before_load"] = cuda_memory()
    result["allocator_limit"] = {
        "fraction": args.allocator_fraction,
        "nominal_bytes": (args.allocator_fraction * result["memory_before_load"]["cuda_total_bytes"]
                          if args.allocator_fraction is not None else None),
        "scope": "PyTorch caching allocator only; not all CUDA allocations or a total-device hard guarantee",
    }
    journal.record("cuda_initialized", memory=result["memory_before_load"], environment=result["selected_environment"])
    artifact_hashes = {name: sha256(args.model_dir / name) for name in ("model.safetensors", "tokenizer.json")}
    if artifact_hashes != {"model.safetensors": MODEL_SHA256, "tokenizer.json": TOKENIZER_SHA256}:
        raise ValueError("model/tokenizer artifacts differ from the fixed snapshot")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=getattr(torch, args.dtype),
        attn_implementation=args.attention).eval()
    result["floating_buffers_after_load"] = {
        name: {"dtype": str(value.dtype), "sha256": tensor_hash(value)}
        for name, value in model.named_buffers() if value.is_floating_point()
    }
    if args.buffer_cast == "worker":
        # Match the pinned verl FSDP engine's explicit module.to(torch_dtype).
        model.to(getattr(torch, args.dtype))
    result["floating_buffers_after_cast"] = {
        name: {"dtype": str(value.dtype), "sha256": tensor_hash(value)}
        for name, value in model.named_buffers() if value.is_floating_point()
    }
    model = model.to("cuda")
    model.requires_grad_(False)
    install_memory_bounded_teacher(model)
    with tempfile.TemporaryDirectory(prefix="factorial-fsdp2-", dir=args.out.parent) as rendezvous:
        torch.distributed.init_process_group("nccl", rank=0, world_size=1,
                                            init_method="file://" + str(Path(rendezvous) / "init"))
        try:
            model = wrap_fsdp2(model, getattr(torch, args.dtype), args.cpu_offload == "on")
            result["actual_fsdp_policies"] = describe_actual_fsdp_policies(model, args.cpu_offload == "on")
            result["memory_after_model_wrap"] = cuda_memory()
            journal.record("model_wrapped", policies=result["actual_fsdp_policies"],
                           memory=result["memory_after_model_wrap"])
            if args.k > model.config.vocab_size:
                raise ValueError("K exceeds the model vocabulary")
            recorded = json.loads(args.fixture.read_text(encoding="utf-8")) if args.fixture else None
            if args.fixture:
                result["recorded_fixture_evidence"] = {"path": str(args.fixture), "sha256": sha256(args.fixture)}
            inputs, ids, valid_mask, fixture = build_fixture(tokenizer, args.lengths, args.k,
                                                           model.config.vocab_size, "cuda", recorded=recorded)
            result["identity"] = {"model_revision": MODEL_REVISION, "artifact_hashes": artifact_hashes,
                                  "dtype": args.dtype, "attention": args.attention,
                                  "buffer_cast": args.buffer_cast, "fixture": fixture}
            expected = None
            if args.expected:
                reference = json.loads(args.expected.read_text(encoding="utf-8"))
                if reference.get("identity") != result["identity"]:
                    raise ValueError("expected result describes different model/input/dtype/attention")
                if not reference.get("status", "").startswith("PASS") or not reference.get("reference_output"):
                    raise ValueError("expected result has no successful output contract")
                expected = reference["reference_output"]
                result["expected_evidence"] = {"path": str(args.expected), "sha256": sha256(args.expected),
                                               "mode": reference["configuration"]["mode"],
                                               "cpu_offload": reference["configuration"]["cpu_offload"]}
            result["correctness_scope"] = ("STRICT_EXTERNAL_OUTPUT_HASH" if expected else
                                           "REPEAT_CONSISTENCY_ONLY_NO_DENSE_REFERENCE")
            result["samples"] = []
            result["completed_calls"] = 0
            fn = (lambda: dense_scores(model, inputs, ids, args.lse_chunk, "cpu")) if args.mode == "dense" else (
                lambda: candidate_scores(model, inputs, ids, args.chunk_rows, "cpu"))
            for index in range(args.warmup + args.repeats):
                hooks = []
                journal.record("call_begin", index=index, phase="warmup" if index < args.warmup else "measured")
                with (observe_calls(model, journal, args.trace_hooks) if not args.no_hooks else empty_hooks()) as hooks:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    before = cuda_memory()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    perf_start = time.perf_counter_ns()
                    raw_clock = getattr(time, "CLOCK_MONOTONIC_RAW", None)
                    raw_start = time.clock_gettime_ns(raw_clock) if raw_clock is not None else None
                    start.record()
                    with torch.no_grad():
                        output = fn()
                    end.record()
                    torch.cuda.synchronize()
                    raw_end = time.clock_gettime_ns(raw_clock) if raw_clock is not None else None
                    perf_end = time.perf_counter_ns()
                    after = cuda_memory()
                signature = compact_signature(output, valid_mask, args.lengths, model.config.vocab_size,
                                              args.k, getattr(torch, args.dtype))
                if expected is None:
                    expected = signature
                    result["reference_output"] = signature
                comparison = compare_signatures(signature, expected)
                head_pre = [event for event in hooks if event["module"] == "head" and event["event"] == "pre"]
                hook_checks = {
                    "normal_outer_and_backbone_hooks": all(
                        sum(event["module"] == module and event["event"] == phase for event in hooks) == 1
                        for module in ("outer", "backbone") for phase in ("pre", "post")),
                    "head_path_shape": validate_head_calls(head_pre, args.mode, len(args.lengths),
                                                          max(args.lengths), args.chunk_rows),
                    "head_post_count": sum(event["module"] == "head" and event["event"] == "post"
                                           for event in hooks) == len(head_pre),
                } if not args.no_hooks else {"status": "NOT_RUN"}
                sample = {
                    "index": index, "phase": "warmup" if index < args.warmup else "measured",
                    "status": "OK", "perf_counter_ms": (perf_end - perf_start) / 1e6,
                    "raw_host_ms": (raw_end - raw_start) / 1e6 if raw_end is not None else None,
                    "cuda_event_ms": start.elapsed_time(end),
                    "before": before, "after": after,
                    "incremental_peak_allocated_bytes": after["peak_allocated_bytes"] - before["allocated_bytes"],
                    "output": signature, "comparison": comparison,
                    "hook_checks": hook_checks, "head_calls": head_pre,
                }
                result["completed_calls"] += 1
                result["last_output"] = signature
                result.setdefault("reference_output", signature)
                if args.save_output is not None and index == 0:
                    artifact = {"identity": result["identity"], "output": signature,
                                "candidate_logprobs": output.candidate_logprobs[valid_mask].contiguous(),
                                "teacher_topk_ids": output.teacher_topk_ids[valid_mask].contiguous()}
                    with args.save_output.open("xb") as stream:
                        torch.save(artifact, stream)
                    result["output_artifact"] = {"path": str(args.save_output), "sha256": sha256(args.save_output)}
                if not comparison["match"] or (not args.no_hooks and not all(hook_checks.values())):
                    sample["status"] = "MISMATCH"
                    journal.record("call_mismatch", sample=sample)
                    raise OutputMismatch("output hash/shape/offset or actual hook boundary differed")
                journal.record("call_complete", sample=sample)
                if index >= args.warmup:
                    result["samples"].append(sample)
                del output
            result["memory_after_calls"] = cuda_memory()
            result["summary"] = summarize_samples(result["samples"])
            result["observed_process_allocator_peak_bytes"] = max(
                result["memory_after_model_wrap"]["peak_allocated_bytes"],
                *[sample["after"]["peak_allocated_bytes"] for sample in result["samples"]])
            result["status"] = "PASS_REFERENCE_EQUIVALENT" if args.expected else "PASS_REPEAT_ONLY"
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


def summarize_samples(samples):
    def quantile(values, q):
        ordered = sorted(values)
        position = (len(ordered) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    metrics = ("perf_counter_ms", "cuda_event_ms", "incremental_peak_allocated_bytes")
    if all(sample["raw_host_ms"] is not None for sample in samples):
        metrics += ("raw_host_ms",)
    return {"count": len(samples),
            "metrics": {name: {"p50": quantile([x[name] for x in samples], 0.5),
                               "p95": quantile([x[name] for x in samples], 0.95)} for name in metrics},
            "cuda_event_exceeds_perf_counter_samples": sum(x["cuda_event_ms"] > x["perf_counter_ms"] for x in samples),
            "cuda_event_exceeds_raw_host_samples": (sum(x["cuda_event_ms"] > x["raw_host_ms"] for x in samples)
                                                     if "raw_host_ms" in metrics else None)}


@contextmanager
def empty_hooks():
    yield []


def main(argv=None):
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.events.parent.mkdir(parents=True, exist_ok=True)
    if args.save_output is not None:
        args.save_output.parent.mkdir(parents=True, exist_ok=True)
    faulthandler.enable()
    journal = JsonlJournal(args.events)
    sampler = DeviceSampler(args.sample_device_ms, 0, journal)
    result = {"status": "RUNNING", "scope": __doc__, "pid": os.getpid(), "platform": platform.platform(),
              "configuration": {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()},
              "source_git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "source_git_status": subprocess.check_output(["git", "-C", str(ROOT), "status", "--short"], text=True),
              "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in
                                (Path(__file__), ROOT / "reference/memory_bounded_teacher.py",
                                 ROOT / "scripts/full_teacher_benchmark.py",
                                 ROOT / "scripts/teacher_stage_benchmark.py")},
              "measurement_scope": "complete outer FSDP/model forward through compact CPU outputs; validation/hashing outside timing",
              "limitations": ["single-GPU FSDP2, no verl Worker/Ray or training claim",
                              "sampled device memory includes all processes and may miss transient peaks",
                              "hooks/sampler/JSONL fsync are diagnostic instrumentation; not a production latency benchmark"]}
    try:
        journal.record("process_started", configuration=result["configuration"],
                       source_git_head=result["source_git_head"], source_hashes=result["source_hashes"])
        sampler.start()
        run(args, result, journal)
    except Exception as error:
        result["status"] = "OOM" if is_out_of_memory(error) else ("MISMATCH" if isinstance(error, OutputMismatch) else "ERROR")
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        result["memory_at_failure"] = safe_cuda_memory() if torch.cuda.is_available() else {"acquisition": "UNAVAILABLE"}
        journal.record("process_error", status=result["status"], error=result["error"],
                       completed_calls=result.get("completed_calls", 0), memory=result["memory_at_failure"])
    finally:
        result["all_device_sampler"] = sampler.stop()
        result["journal"] = {"path": str(args.events), "sha256_before_final_event": sha256(args.events)}
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(json_safe(result), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        journal.record("process_finished", status=result["status"], final_json=str(args.out), sha256=sha256(args.out))
        journal.close()
    print(json.dumps({"status": result["status"], "completed_calls": result.get("completed_calls", 0),
                      "out": str(args.out), "events": str(args.events)}), flush=True)
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())