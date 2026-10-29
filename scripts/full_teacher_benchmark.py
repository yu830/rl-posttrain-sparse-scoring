#!/usr/bin/env python3
"""Benchmark a real complete Qwen Teacher forward and compact scoring output.

A0 calls the ordinary CausalLM forward, then reduces full logits. A1 calls the
same outer model/FSDP forward with the installed opt-in readout adapter. The
boundary includes the Transformer, final norm, LM head, FP32 LSE/gather, teacher
Top-K, and optional compact-output CPU copies. It excludes tokenization, model
loading, candidate construction, correctness checks, Ray dispatch and training.
This is not an execution of the verl TeacherFSDPWorker.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import traceback

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reference.memory_bounded_teacher import install_memory_bounded_teacher
from scripts.teacher_stage_benchmark import (
    MODEL_REVISION, MODEL_SHA256, TOKENIZER_SHA256, PROMPTS, TOLERANCES,
    selected_full_logprobs, topk_tie_valid,
)


@dataclass
class Scores:
    candidate_logprobs: torch.Tensor
    teacher_topk_ids: torch.Tensor


class ForwardFailure(RuntimeError):
    def __init__(self, sample):
        super().__init__(sample["error"])
        self.sample = sample


def is_out_of_memory(error):
    """Preserve an OOM classification across the FSDP stop-and-propagate boundary."""
    current = error
    while current is not None:
        if isinstance(current, torch.cuda.OutOfMemoryError):
            return True
        if isinstance(current, ForwardFailure) and current.sample.get("status") == "OOM":
            return True
        current = current.__cause__
    return False


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def gpu_snapshot():
    result = {"unix_time": time.time(), "cuda_free_total_bytes": list(torch.cuda.mem_get_info())}
    for label, command in (
        ("device", ["nvidia-smi", "--query-gpu=timestamp,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw", "--format=csv"]),
        ("processes", ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"]),
    ):
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        result[label] = {"command": command, "exit_code": completed.returncode,
                         "stdout": completed.stdout, "stderr": completed.stderr}
    return result


def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize(samples):
    successful = [sample for sample in samples if sample["status"] == "OK"]
    metrics = ("wall_ms", "cuda_event_ms", "peak_allocated_bytes",
               "incremental_peak_allocated_bytes", "peak_reserved_bytes")
    if successful and all(sample.get("raw_host_ms") is not None for sample in successful):
        metrics += ("raw_host_ms",)
    return {
        label: {
            "count": len(group),
            "metrics": {metric: {"p50": percentile([x[metric] for x in group], 0.5),
                                 "p95": percentile([x[metric] for x in group], 0.95)}
                        for metric in metrics},
            "cuda_event_exceeds_host_samples": sum(x["cuda_event_ms"] > x["wall_ms"] for x in group),
        }
        for label in dict.fromkeys(x["label"] for x in successful)
        for group in [[x for x in successful if x["label"] == label]]
    }


def comparison_summary(summary):
    if not all(label in summary for label in ("A0", "A1")):
        return {"status": "UNAVAILABLE"}
    base, candidate = [summary[label]["metrics"] for label in ("A0", "A1")]
    base_peak = base["peak_allocated_bytes"]["p50"]
    return {
        "status": "DESCRIPTIVE_ONLY",
        "host_p50_change_fraction": candidate["wall_ms"]["p50"] / base["wall_ms"]["p50"] - 1,
        "absolute_peak_allocated_reduction_fraction": 1 - candidate["peak_allocated_bytes"]["p50"] / base_peak,
        "absolute_peak_allocated_saved_bytes": base_peak - candidate["peak_allocated_bytes"]["p50"],
        "same_process_reserved_memory_is_not_isolated": True,
    }


def timed(fn, label, block, phase):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before_allocated = torch.cuda.memory_allocated()
    sample = {"label": label, "block": block, "phase": phase,
              "before_allocated_bytes": before_allocated,
              "before_reserved_bytes": torch.cuda.memory_reserved()}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin = time.perf_counter()
    raw_clock = getattr(time, "CLOCK_MONOTONIC_RAW", None)
    raw_begin = time.clock_gettime_ns(raw_clock) if raw_clock is not None else None
    start.record()
    try:
        output = fn()
        end.record()
        torch.cuda.synchronize()
        raw_end = time.clock_gettime_ns(raw_clock) if raw_clock is not None else None
        host_elapsed_ms = (time.perf_counter() - begin) * 1000
        sample.update(status="OK", wall_ms=host_elapsed_ms,
                      raw_host_ms=(raw_end - raw_begin) / 1e6 if raw_end is not None else None,
                      cuda_event_ms=start.elapsed_time(end),
                      after_allocated_bytes=torch.cuda.memory_allocated(),
                      after_reserved_bytes=torch.cuda.memory_reserved(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        sample["incremental_peak_allocated_bytes"] = sample["peak_allocated_bytes"] - before_allocated
        del output
        return sample
    except Exception as exc:
        sample.update(status="OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "ERROR",
                      elapsed_host_ms_until_error=(time.perf_counter() - begin) * 1000,
                      error=repr(exc), traceback=traceback.format_exc(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        raise ForwardFailure(sample) from exc


def move_scores(output, device):
    if device == "cuda":
        return Scores(output.candidate_logprobs, output.teacher_topk_ids)
    return Scores(output.candidate_logprobs.cpu(), output.teacher_topk_ids.cpu())


def dense_scores(module, inputs, ids, lse_chunk, return_device):
    output = module(**inputs)
    logits = output.logits
    flat = logits.reshape(-1, logits.shape[-1])
    flat_ids = ids.reshape(-1, ids.shape[-1])
    logp = selected_full_logprobs(flat, flat_ids, lse_chunk).reshape(ids.shape)
    topk = flat.topk(ids.shape[-1], dim=-1).indices.reshape(ids.shape)
    return move_scores(Scores(logp, topk), return_device)


def candidate_scores(module, inputs, ids, chunk, return_device):
    output = module(**inputs, teacher_topk_ids=ids, teacher_chunk_rows=chunk)
    return move_scores(output, return_device)


@contextmanager
def observe_head(head):
    """Record actual untimed head calls; hooks remain active around __call__."""
    calls = []

    def record(module, inputs, output):
        calls.append({"input_shape": list(inputs[0].shape), "output_shape": list(output.shape),
                      "output_dtype": str(output.dtype), "output_numel": output.numel()})

    handle = head.register_forward_hook(record)
    try:
        yield calls
    finally:
        handle.remove()


def correctness(module, head, inputs, ids, chunk, lse_chunk, dtype):
    atol, rtol = TOLERANCES[dtype]
    with torch.no_grad(), observe_head(head) as calls:
        candidate = candidate_scores(module, inputs, ids, chunk, "cpu")
    # The compact candidate is on CPU, so validation does not retain a second
    # full logits tensor while running the dense reference.
    with torch.no_grad():
        raw = module(**inputs)
        logits = raw.logits
        flat = logits.reshape(-1, logits.shape[-1])
        flat_ids = ids.reshape(-1, ids.shape[-1])
        expected = selected_full_logprobs(flat, flat_ids, lse_chunk).reshape(ids.shape).cpu()
        expected_topk = flat.topk(ids.shape[-1], -1).indices.reshape(ids.shape).cpu()
        valid = topk_tie_valid(flat, candidate.teacher_topk_ids.to(flat.device).reshape(flat_ids.shape),
                              ids.shape[-1]).cpu()
        del raw, logits, flat
    actual = candidate.candidate_logprobs
    logp_close = torch.isclose(actual, expected, atol=atol, rtol=rtol)
    expected_mass = expected.float().exp().sum(-1)
    actual_mass = actual.float().exp().sum(-1)
    mass_close = torch.isclose(actual_mass, expected_mass, atol=atol, rtol=rtol)
    n = ids.numel() // ids.shape[-1]
    observed_rows = [math.prod(call["input_shape"][:-1]) for call in calls]
    boundary_ok = bool(calls) and sum(observed_rows) == n and max(observed_rows) <= min(n, chunk)
    exact_rows = expected_topk.sort(-1).values.eq(candidate.teacher_topk_ids.sort(-1).values).all(-1)
    checks = {
        "candidate_logprobs": {"within_budget": bool(logp_close.all()),
                               "max_abs": (actual.float() - expected.float()).abs().max().item(),
                               "failed_values": int((~logp_close).sum()),
                               "dtype_equal": actual.dtype == expected.dtype,
                               "baseline_sha256": tensor_sha256(expected),
                               "candidate_sha256": tensor_sha256(actual)},
        "teacher_topk_ids": {"within_budget": bool(valid.all()), "rows": n,
                             "exact_set_equal_rows": int(exact_rows.sum()),
                             "baseline_tie_valid_rows": int(valid.sum()),
                             "candidate_sha256": tensor_sha256(candidate.teacher_topk_ids)},
        "teacher_mass": {"within_budget": bool(mass_close.all()),
                         "max_abs": (actual_mass - expected_mass).abs().max().item()},
        "head_boundary": {"within_budget": boundary_ok, "calls": calls,
                           "requested_chunk_rows": chunk, "total_rows": n,
                           "largest_observed_head_rows": max(observed_rows) if observed_rows else None,
                           "no_full_N_by_V_head_call": bool(observed_rows) and all(rows < n for rows in observed_rows),
                           "full_logits_equivalent_due_to_chunk_size": n <= chunk},
        "loss_gradient_and_training_update": {"status": "NOT_RUN",
                                               "reason": "separate training-consumer acceptance is required"},
    }
    checks["status"] = "PASS" if all(
        check["within_budget"] for check in checks.values()
        if isinstance(check, dict) and "within_budget" in check
    ) and checks["candidate_logprobs"]["dtype_equal"] else "FAILED"
    return checks


def wrap_fsdp2(model, dtype, cpu_offload):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy, fully_shard

    mesh = init_device_mesh("cuda", (1,))
    options = {"mesh": mesh, "mp_policy": MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype),
               "offload_policy": CPUOffloadPolicy() if cpu_offload else OffloadPolicy(),
               "reshard_after_forward": True}
    # This accesses the module tree only to install wrappers. All measured
    # execution below enters the outer module; embedding/head remain together
    # in the root group, preserving the pinned Qwen's tied weights.
    for layer in model.model.layers:
        fully_shard(layer, **options)
    fully_shard(model, **options)
    return model


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True, help="raw JSON path, normally outside the source repository")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--rows", nargs="+", type=int, default=[128, 512, 1024, 2048])
    parser.add_argument("--ks", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--row-chunks", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--lse-chunk", type=int, default=1024)
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--aa-blocks", type=int, default=6, help="one sample per A0 label per block")
    parser.add_argument("--ab-blocks", type=int, default=10, help="ABBA/BAAB blocks; two samples per label per block")
    parser.add_argument("--single-path", choices=("A0", "A1"), help="capacity measurement in a fresh process; no A/B correctness claim")
    parser.add_argument("--fsdp2", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true", help="requires --fsdp2")
    parser.add_argument("--return-device", choices=("cuda", "cpu"), default="cpu")
    args = parser.parse_args()
    if min(args.rows + args.ks + args.row_chunks + [args.lse_chunk, args.warmup,
                                                   args.aa_blocks, args.ab_blocks]) <= 0:
        parser.error("dimensions and sample counts must be positive")
    if args.cpu_offload and not args.fsdp2:
        parser.error("--cpu-offload requires --fsdp2")
    return args


def main():
    args = get_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    result = {"status": "RUNNING", "argv": sys.argv, "configuration": vars(args),
              "scope": __doc__, "model_revision": MODEL_REVISION, "seed": 20261029,
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "python": platform.python_version(), "platform": platform.platform(),
              "pid": os.getpid(), "script_sha256": sha256(__file__),
              "source_git_head": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "source_git_status": subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--short"], text=True),
              "tolerance": dict(zip(("atol", "rtol"), TOLERANCES[dtype])),
              "measurement_boundary": {
                  "included": ["outer model/FSDP call", "transformer and final norm", "full teacher readout contract",
                               "compact CPU copies" if args.return_device == "cpu" else "compact GPU output"],
                  "excluded": ["model loading", "tokenization/input preparation", "candidate construction",
                               "correctness probes", "Ray/verl worker dispatch", "loss/backward/optimizer"],
                  "memory": "absolute and incremental PyTorch allocator peaks; reserved inherits this process's history; device snapshots are not peaks",
                  "inputs": "already-resident CUDA input IDs, attention mask, positions and candidate IDs",
              },
              "fsdp": {"enabled": args.fsdp2, "world_size": 1, "cpu_offload": args.cpu_offload,
                       "root_and_decoder_reshard_after_forward": True if args.fsdp2 else None,
                       "groups": "decoder layers then root (tied embedding/head together)" if args.fsdp2 else None,
                       "multi_gpu_sharding_and_communication": "NOT_RUN"},
              "engineering_targets": {"absolute_peak_reduction_fraction": 0.20,
                                      "maximum_length_increase_ratio": 1.5,
                                      "main_workload_host_overhead_fraction": 0.10,
                                      "kind": "predeclared targets, not achieved results"},
              "source_hashes": {str(path.relative_to(REPO_ROOT)): sha256(path)
                                for path in (REPO_ROOT / "reference/memory_bounded_teacher.py",
                                             REPO_ROOT / "scripts/teacher_stage_benchmark.py")},
              "cells": []}

    def save():
        temporary = out.with_suffix(out.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + chr(10), encoding="utf-8")
        temporary.replace(out)

    distributed = False
    rendezvous_dir = None
    active_stage = "initialization"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA device is required; no CPU performance substitution")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers
        result["transformers"] = transformers.__version__
        torch.manual_seed(20261029)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        result["tf32_matmul"] = False
        result["autocast"] = False
        result["gpu_before_load"] = gpu_snapshot()
        model_dir = Path(args.model_dir)
        result["model_hashes"] = {name: sha256(model_dir / name)
                                  for name in ("model.safetensors", "tokenizer.json")}
        if result["model_hashes"] != {"model.safetensors": MODEL_SHA256, "tokenizer.json": TOKENIZER_SHA256}:
            raise ValueError("model/tokenizer hashes do not match the fixed snapshot")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=dtype, attn_implementation=args.attention, local_files_only=True,
        ).to("cuda").eval()
        install_memory_bounded_teacher(model)
        head = model.lm_head
        if args.fsdp2:
            rendezvous_dir = tempfile.TemporaryDirectory(prefix="teacher-fsdp2-")
            torch.distributed.init_process_group(
                "nccl", init_method="file://" + str(Path(rendezvous_dir.name) / "rendezvous"),
                rank=0, world_size=1,
            )
            distributed = True
            model = wrap_fsdp2(model, dtype, args.cpu_offload)
        result["model"] = {"class": type(model).__name__, "vocab_size": model.config.vocab_size,
                           "tie_word_embeddings": model.config.tie_word_embeddings,
                           "use_cache": False, "attention": args.attention,
                           "layers": len(model.model.layers)}
        # One ordinary causal sequence avoids a fictitious packed-SDPA path.
        # Different N use exact prefixes of this deterministic token fixture.
        prompt = " ".join(PROMPTS)
        source_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        repeats = math.ceil(max(args.rows) / len(source_tokens))
        token_fixture = (source_tokens * repeats)[:max(args.rows)]
        result["fixture"] = {
            "kind": "synthetic repeated natural-language prompt; not an RL trajectory",
            "batch": 1, "tokens": token_fixture, "packed_multi_sequence_attention": "NOT_RUN",
            "candidate_source": "deterministic distinct vocabulary IDs derived from input token and row; not observed student top-K",
        }
        save()
        for n in args.rows:
            tokens = torch.tensor([token_fixture[:n]], device="cuda", dtype=torch.long)
            inputs = {"input_ids": tokens, "attention_mask": torch.ones_like(tokens),
                      "position_ids": torch.arange(n, device="cuda").unsqueeze(0), "use_cache": False}
            for k in args.ks:
                vocab = model.config.vocab_size
                if k > vocab:
                    raise ValueError("K exceeds vocabulary")
                # Coprime stride ensures no within-row duplicate candidates.
                stride = 7919 if math.gcd(7919, vocab) == 1 else 1
                ids = (tokens.unsqueeze(-1) + torch.arange(k, device="cuda") * stride
                       + torch.arange(n, device="cuda").reshape(1, n, 1) * 17) % vocab
                cell = {"N": n, "K": k, "status": "RUNNING", "variants": [],
                        "input_ids_sha256": tensor_sha256(tokens),
                        "candidate_ids_sha256": tensor_sha256(ids),
                        "candidate_ids": ids.cpu().tolist(),
                        "gpu_before": gpu_snapshot()}
                result["cells"].append(cell)
                save()
                dense_fn = lambda: dense_scores(model, inputs, ids, args.lse_chunk, args.return_device)
                try:
                    if args.single_path is not None:
                        # Capacity probes use a new process after a failed FSDP
                        # run. They do not establish A/B equivalence by themselves.
                        for chunk in args.row_chunks if args.single_path == "A1" else [None]:
                            label = args.single_path
                            variant = {"name": label, "chunk_rows": chunk, "status": "RUNNING",
                                       "samples": [], "correctness": {"status": "NOT_RUN",
                                       "reason": "single-path capacity measurement; use a completed A/B check for equivalence"}}
                            cell["variants"].append(variant)
                            fn = dense_fn if label == "A0" else (
                                lambda: candidate_scores(model, inputs, ids, chunk, args.return_device))
                            with torch.no_grad():
                                active_stage = f"N={n},K={k},B={chunk},single-path warmup"
                                for _ in range(args.warmup):
                                    fn()
                                for block in range(args.ab_blocks * 2):
                                    active_stage = f"N={n},K={k},B={chunk},single-path sample={block}"
                                    variant["samples"].append(timed(fn, label, block, "capacity"))
                                    save()
                            variant["summary"] = summarize(variant["samples"])
                            variant["status"] = "MEASURED_ONLY"
                        cell["status"] = "MEASURED_ONLY"
                        cell["gpu_after"] = gpu_snapshot()
                        save()
                        print(json.dumps({"N": n, "K": k, "status": cell["status"], "out": str(out)}), flush=True)
                        del ids
                        continue
                    active_stage = f"N={n},K={k},A/A warmup"
                    with torch.no_grad():
                        for _ in range(args.warmup):
                            dense_fn()
                        aa = []
                        cell["AA"] = {"samples": aa}
                        for block in range(args.aa_blocks):
                            order = ("A0a", "A0b") if block % 2 == 0 else ("A0b", "A0a")
                            for label in order:
                                active_stage = f"N={n},K={k},A/A block={block},label={label}"
                                aa.append(timed(dense_fn, label, block, "AA"))
                                save()
                    cell["AA"]["summary"] = summarize(aa)
                    for chunk in args.row_chunks:
                        variant = {"name": "A1", "chunk_rows": chunk, "status": "RUNNING",
                                   "samples": []}
                        cell["variants"].append(variant)
                        active_stage = f"N={n},K={k},B={chunk},correctness"
                        variant["correctness"] = correctness(model, head, inputs, ids, chunk,
                                                             args.lse_chunk, dtype)
                        save()
                        candidate_fn = lambda: candidate_scores(model, inputs, ids, chunk, args.return_device)
                        with torch.no_grad():
                            active_stage = f"N={n},K={k},B={chunk},AB warmup"
                            for _ in range(args.warmup):
                                dense_fn()
                                candidate_fn()
                            for block in range(args.ab_blocks):
                                order = ("A0", "A1", "A1", "A0") if block % 2 == 0 else ("A1", "A0", "A0", "A1")
                                for label in order:
                                    active_stage = f"N={n},K={k},B={chunk},ABBA block={block},label={label}"
                                    fn = dense_fn if label == "A0" else candidate_fn
                                    variant["samples"].append(timed(fn, label, block, "ABBA"))
                                    save()
                        variant["summary"] = summarize(variant["samples"])
                        variant["comparison"] = comparison_summary(variant["summary"])
                        variant["status"] = variant["correctness"]["status"]
                        variant["gpu_after"] = gpu_snapshot()
                        save()
                    cell["status"] = "PASS" if all(v["status"] == "PASS" for v in cell["variants"]) else "FAILED_CORRECTNESS"
                except Exception as exc:
                    cell.update(status="OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "ERROR",
                                failed_stage=active_stage, error=repr(exc), traceback=traceback.format_exc())
                    if isinstance(exc, ForwardFailure):
                        cell["failed_sample"] = exc.sample
                        cell["status"] = exc.sample["status"]
                    save()
                    if args.fsdp2:
                        raise RuntimeError("FSDP forward failed; stop rather than reuse uncertain hook/offload state") from exc
                    gc.collect()
                    torch.cuda.empty_cache()
                cell["gpu_after"] = gpu_snapshot()
                save()
                print(json.dumps({"N": n, "K": k, "status": cell["status"], "out": str(out)}), flush=True)
                del ids
            del inputs, tokens
        success_status = "MEASURED_ONLY" if args.single_path else "PASS"
        result["status"] = success_status if result["cells"] and all(
            c["status"] == success_status for c in result["cells"]
        ) else "COMPLETED_WITH_FAILURES"
        result["gpu_final"] = gpu_snapshot()
    except Exception as exc:
        result.update(status="OOM" if is_out_of_memory(exc) else "ERROR",
                      failed_stage=active_stage, error=repr(exc), traceback=traceback.format_exc())
    finally:
        save()
        if distributed:
            torch.distributed.destroy_process_group()
        if rendezvous_dir is not None:
            rendezvous_dir.cleanup()
    print(json.dumps({"status": result["status"], "out": str(out)}), flush=True)
    return 0 if result["status"] in {"PASS", "MEASURED_ONLY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())