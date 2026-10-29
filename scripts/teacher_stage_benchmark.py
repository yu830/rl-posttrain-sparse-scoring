#!/usr/bin/env python3
"""Real-transformer teacher-head replay; no serving or training speedup claim.

A0 materializes [N,V] before token-chunked FP32 LSE/gather. A1 chunks
hidden rows *before* the full-vocabulary head. Both retain teacher top-K and
cast selected full logprobs back to the head dtype, as verl #6920 does.
Model loading, transformer forward, candidate selection, correctness checks,
and loss/backward probes are outside the timed teacher-head boundary.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
import types

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.scoring import causal_positions

MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
A2_BLOBS = {"linear_topk_log_probs.py": "8463ac83431e22cf63fa0349d73726b8d9e0564b",
            "topk_log_probs_kernels.py": "2fa63d2bd2bae31416d4645f1d3a071d9d77aa16"}
TOLERANCES = {torch.bfloat16: (0.1, 0.01), torch.float32: (5e-5, 2e-5)}
PROMPTS = (
    "Explain how a GPU computes matrix multiplication, including memory movement and synchronization. ",
    "Describe the causal language modeling objective and why the hidden state predicts the next token. ",
    "Compare a full vocabulary probability distribution with one conditioned on a set of candidate tokens. ",
    "Give a careful account of numerical precision, rounding, and stable logsumexp in machine learning. ",
    "Explain reinforcement learning with a frozen teacher model and a student policy. ",
    "Describe the difference between allocated memory, reserved memory, and total device memory. ",
    "Explain how a reproducible benchmark controls warmup, execution order, and background activity. ",
    "Discuss sparse readout interfaces, causal alignment, token masks, and output diagnostics. ",
)


@dataclass
class TeacherOutput:
    candidate_logprobs: torch.Tensor
    teacher_topk_ids: torch.Tensor


def selected_full_logprobs(logits, ids, lse_chunk, out=None):
    """Faithful #6920 utility: FP32 LSE/gather, then output-dtype cast."""
    if lse_chunk <= 0:
        raise ValueError("lse_chunk must be positive")
    if out is None:
        out = torch.empty(ids.shape, device=logits.device, dtype=logits.dtype)
    for start in range(0, logits.shape[0], lse_chunk):
        z = logits[start:start + lse_chunk].float()
        log_z = z.logsumexp(-1, keepdim=True)
        selected = z.gather(-1, ids[start:start + lse_chunk])
        out[start:start + lse_chunk] = (selected - log_z).to(logits.dtype)
    return out


def teacher_stage(hidden, weight, ids, *, topk, row_chunk=None, lse_chunk=1024):
    """Inputs are validated supports on an ordinary unquantized linear head."""
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden and weight must be [N,D] and [V,D]")
    if ids.shape[0] != hidden.shape[0] or ids.ndim != 2 or ids.dtype != torch.int64:
        raise ValueError("ids must be int64 [N,K]")
    if not 0 < topk <= weight.shape[0] or row_chunk is not None and row_chunk <= 0:
        raise ValueError("invalid topk or row_chunk")
    if row_chunk is None:
        logits = torch.nn.functional.linear(hidden, weight)
        logps = selected_full_logprobs(logits, ids, lse_chunk)
        topids = torch.topk(logits, topk, dim=-1).indices
        return TeacherOutput(logps, topids)
    step = row_chunk
    logps = torch.empty(ids.shape, device=hidden.device, dtype=hidden.dtype)
    topids = torch.empty((hidden.shape[0], topk), device=hidden.device, dtype=torch.int64)
    for start in range(0, hidden.shape[0], step):
        logits = torch.nn.functional.linear(hidden[start:start + step], weight)
        selected_full_logprobs(logits, ids[start:start + step], lse_chunk, out=logps[start:start + step])
        topids[start:start + step] = torch.topk(logits, topk, dim=-1).indices
        del logits
    return TeacherOutput(logps, topids)


def topk_tie_valid(reference_logits, actual_ids, k):
    """Set contract: all scores above kth threshold, any exact boundary ties.

    Numeric tolerances do not turn merely nearby scores into ties. Ordering
    inside equal-score groups is deliberately unconstrained by torch.topk.
    """
    values = reference_logits.gather(-1, actual_ids)
    threshold = torch.topk(reference_logits, k, dim=-1).values[:, -1:]
    unique = actual_ids.sort(-1).values.diff(dim=-1).ne(0).all(-1)
    above_count = reference_logits.gt(threshold).sum(-1)
    actual_above = values.gt(threshold).sum(-1)
    descending = values.diff(dim=-1).le(0).all(-1)
    return unique & descending & values.ge(threshold).all(-1) & actual_above.eq(above_count)


def scale_student_logits_(logits, temperature):
    """Match #6920's non-fused engine temperature dtype and in-place divide.

    The engine clamps the per-token temperature tensor in its source dtype,
    then casts it to the logits dtype before division. A Python scalar divide
    is observably different for BF16 (including the representation of 0.8).
    """
    return logits.div_(temperature.clamp(min=1e-8).unsqueeze(-1).to(logits.dtype))


def consumer_loss(student_full_logprobs, teacher_full_logprobs, candidate_mask,
                  token_mask, clamp_min=None, loss_max_clamp=None):
    """Unclamped oracle, or #6920 post-normalization clamp diagnostic.

    Teacher FP32 matches ray_trainer.py's .float() before the loss; student
    normalization retains its logit dtype until reverse KL casts to FP32.
    Candidate padding is a reference stress
    probe; #6920's actual top-K support is rectangular. token_mask excludes
    padding/prompt positions from the mean; it does not change causal rows.
    """
    student = student_full_logprobs
    teacher = teacher_full_logprobs.detach().float()
    teacher_mass = teacher.exp().masked_fill(~candidate_mask, 0).sum(-1)
    logp = student.masked_fill(~candidate_mask, -torch.inf)
    logq = teacher.masked_fill(~candidate_mask, -torch.inf)
    logp = logp - logp.logsumexp(-1, keepdim=True)
    logq = logq - logq.logsumexp(-1, keepdim=True)
    if clamp_min is not None:
        logp = logp.clamp_min(clamp_min).masked_fill(~candidate_mask, 0)
        logq = logq.clamp_min(clamp_min).masked_fill(~candidate_mask, 0)
    logp = logp.float().masked_fill(~candidate_mask, 0)
    logq = logq.float().masked_fill(~candidate_mask, 0)
    per_token = (logp.exp() * (logp - logq)).masked_fill(~candidate_mask, 0).sum(-1)
    if loss_max_clamp is not None:
        per_token = per_token.clamp_min(0).clamp(-loss_max_clamp, loss_max_clamp)
    loss = per_token.masked_fill(~token_mask, 0).sum() / token_mask.sum()
    return loss, teacher_mass


def overlap_diagnostics(student_lp, teacher_lp, student_ids, teacher_ids, clamp_min=-10):
    """#6920 diagnostics prior to the wrapper's nonnegative loss clamp."""
    teacher_lp = teacher_lp.float()
    student_lp = student_lp - student_lp.logsumexp(-1, keepdim=True)
    teacher_lp = teacher_lp - teacher_lp.logsumexp(-1, keepdim=True)
    if clamp_min is not None:
        student_lp = student_lp.clamp_min(clamp_min)
        teacher_lp = teacher_lp.clamp_min(clamp_min)
    overlap = (student_ids[:, :, None] == teacher_ids[:, None, :]).any(-1)
    count = overlap.sum(-1)
    # This diagnostic precedes reverse_kl_divergence's explicit FP32 cast.
    token_rkl = student_lp.exp() * (student_lp - teacher_lp)
    advantage = (-token_rkl * overlap).sum(-1) / count.clamp_min(1)
    return count, torch.where(count > 0, advantage, torch.zeros_like(advantage))


def load_a2(directory):
    """Load untouched pinned Apache-2.0 upstream modules in isolation.

    Only CUDA Triton dispatch is accepted here; the portable dense fallback
    is not relabeled as a fused baseline.
    """
    for name, expected in A2_BLOBS.items():
        raw = (Path(directory) / name).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if actual != expected:
            raise ValueError(f"{name}: upstream Git blob mismatch: {actual} != {expected}")
    package = types.ModuleType("g2_upstream_a2")
    package.__path__ = [str(Path(directory).resolve())]
    sys.modules[package.__name__] = package
    kernels = importlib.import_module("g2_upstream_a2.topk_log_probs_kernels")
    linear = importlib.import_module("g2_upstream_a2.linear_topk_log_probs")
    def stage(hidden, weight, ids, *, topk, row_chunk):
        if not kernels.can_use_triton(hidden, ids.shape[-1]):
            raise RuntimeError("upstream cannot dispatch Triton for this input; dense fallback not measured as A2")
        logps = linear.linear_topk_log_probs(hidden, weight, ids, temperature=1.0)
        indices = torch.empty((hidden.shape[0], topk), device=hidden.device, dtype=torch.int64)
        for start in range(0, hidden.shape[0], row_chunk):
            logits = torch.nn.functional.linear(hidden[start:start + row_chunk], weight)
            indices[start:start + row_chunk] = logits.topk(topk, -1).indices
            del logits
        # Explicit #6920 wire-dtype adaptation. Native upstream is FP32 and
        # accumulates before BF16-logit rounding, so equivalence is tested.
        return TeacherOutput(logps.to(hidden.dtype), indices)
    stage.native_logprobs = lambda hidden, weight, ids: linear.linear_topk_log_probs(hidden, weight, ids, temperature=1.0)
    return stage


def gpu_snapshot():
    # WSL may not expose per-process accounting: preserve exact stdout/stderr.
    queries = {
        "device": ["nvidia-smi", "--query-gpu=timestamp,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit", "--format=csv"],
        "processes": ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"],
        "full": ["nvidia-smi"],
    }
    result = {"unix_time": time.time(), "cuda_free_total_bytes": list(torch.cuda.mem_get_info())}
    for name, command in queries.items():
        p = subprocess.run(command, text=True, capture_output=True, check=False)
        result[name] = {"command": command, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    return result


def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize(samples):
    return {name: {metric: {"p50": percentile([s[metric] for s in samples if s["label"] == name], .5),
                             "p95": percentile([s[metric] for s in samples if s["label"] == name], .95)}
                   for metric in ("wall_ms", "cuda_event_ms", "peak_allocated_bytes", "peak_reserved_bytes")}
            for name in dict.fromkeys(s["label"] for s in samples)}


def timed(fn, label, block):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved()}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin = time.perf_counter()
    start.record()
    output = fn()
    end.record()
    end.synchronize()
    wall = (time.perf_counter() - begin) * 1000
    sample = {"label": label, "block": block, "wall_ms": wall, "cuda_event_ms": start.elapsed_time(end),
              "before": before, "allocated_bytes": torch.cuda.memory_allocated(),
              "reserved_bytes": torch.cuda.memory_reserved(), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    del output
    return sample


def compare_outputs(a, b, hidden, weight, ids, student_lp, *, lse_chunk):
    atol, rtol = TOLERANCES[hidden.dtype]
    comparisons = {}
    for field in ("candidate_logprobs",):
        x, y = getattr(a, field), getattr(b, field)
        comparisons[field] = {"max_abs": (x.float() - y.float()).abs().max().item(),
                              "within_budget": bool(torch.isclose(x, y, atol=atol, rtol=rtol).all())}
        comparisons[field].update(worst_difference(x, y))
    valid_rows, strict_equal = [], []
    # Validation is untimed. Match A0's GEMM shape exactly: BF16 shape changes
    # may alter logits, which must not be misclassified as exact ties.
    with torch.no_grad():
        reference_logits = torch.nn.functional.linear(hidden, weight)
    for start in range(0, hidden.shape[0], 32):
        actual = b.teacher_topk_ids[start:start + 32]
        a_ids = a.teacher_topk_ids[start:start + 32]
        valid = topk_tie_valid(reference_logits[start:start + 32], actual, actual.shape[-1])
        valid_rows.extend(valid.cpu().tolist())
        strict_equal.extend(a_ids.sort(-1).values.eq(actual.sort(-1).values).all(-1).cpu().tolist())
    del reference_logits
    comparisons["teacher_topk_ids"] = {"exact_set_equal_rows": sum(strict_equal),
                                         "tie_valid_rows": sum(valid_rows), "rows": len(valid_rows),
                                         "within_budget": all(valid_rows)}
    for field, x, y in (("teacher_mass_full_support", a.candidate_logprobs.float().exp().sum(-1), b.candidate_logprobs.float().exp().sum(-1)),
                        *[(name, xa, xb) for name, xa, xb in zip(("overlap_count", "overlap_token_advantage"),
                          overlap_diagnostics(student_lp, a.candidate_logprobs, ids, a.teacher_topk_ids),
                          overlap_diagnostics(student_lp, b.candidate_logprobs, ids, b.teacher_topk_ids))]):
        comparisons[field] = {"max_abs": (x.float() - y.float()).abs().max().item(),
                              "within_budget": bool(torch.isclose(x.float(), y.float(), atol=atol, rtol=rtol).all())}
    cmask = torch.ones_like(ids, dtype=torch.bool)
    cmask[::5, -1] = False
    tmask = torch.arange(ids.shape[0], device=ids.device).remainder(7).ne(0)
    loss_results = []
    for output in (a, b):
        student = student_lp.detach().clone().requires_grad_(True)
        loss, mass = consumer_loss(student, output.candidate_logprobs, cmask, tmask, clamp_min=-10, loss_max_clamp=10)
        grad = torch.autograd.grad(loss, student)[0]
        loss_results.append((loss.detach(), mass, grad.detach()))
    for name, x, y in zip(("student_loss_ragged_probe", "teacher_mass_ragged_probe", "student_selected_logprob_gradient"), *loss_results):
        comparisons[name] = {"max_abs": (x - y).abs().max().item(),
                             "within_budget": bool(torch.isclose(x, y, atol=atol, rtol=rtol).all())}
    comparisons["student_probe"] = {"loss": loss_results[0][0].item(),
                                     "gradient_l2": loss_results[0][2].norm().item(),
                                     "active_token_rows": tmask.sum().item(),
                                     "candidate_mask": "last candidate excluded every fifth row"}
    # Untimed real-head backward probe on 32 aligned rows. Unlike the local
    # selected-logprob gradient above, this traverses the full student head,
    # full-vocabulary log_softmax, gather, K renormalization and clamp.
    model_grads = []
    probe_rows = min(32, hidden.shape[0])
    for output in (a, b):
        sh = hidden[:probe_rows].detach().clone().requires_grad_(True)
        student_logits = torch.nn.functional.linear(sh, weight.detach())
        temperature = torch.full((probe_rows,), .8, device=sh.device, dtype=torch.float32)
        scale_student_logits_(student_logits, temperature)
        selected = student_logits.log_softmax(-1).gather(-1, ids[:probe_rows].clone())
        loss, _ = consumer_loss(selected, output.candidate_logprobs[:probe_rows],
                                torch.ones_like(selected, dtype=torch.bool), tmask[:probe_rows],
                                clamp_min=-10, loss_max_clamp=10)
        gh, gz = torch.autograd.grad(loss, (sh, student_logits))
        model_grads.append((loss.detach(), gh.detach(), gz.detach()))
    for name, x, y in zip(("student_full_path_loss", "student_hidden_gradient", "student_full_logits_gradient"), *model_grads):
        comparisons[name] = {"rows": probe_rows, "max_abs": (x.float() - y.float()).abs().max().item(),
                             "within_budget": bool(torch.isclose(x, y, atol=atol, rtol=rtol).all())}
        comparisons[name].update(worst_difference(x, y))
        if "gradient" in name:
            norm = x.double().norm().item()
            error_norm = (x.double() - y.double()).norm().item()
            relative = error_norm / norm if norm else (0.0 if error_norm == 0 else None)
            budget = .05 if hidden.dtype == torch.bfloat16 else 1e-4
            comparisons[name].update({"baseline_l2": norm, "relative_l2": relative, "relative_l2_budget": budget})
            comparisons[name]["within_budget"] &= relative is not None and relative <= budget
    comparisons["status"] = "PASS" if all(v["within_budget"] for v in comparisons.values() if isinstance(v, dict) and "within_budget" in v) else "FAILED"
    return comparisons


def worst_difference(baseline, candidate):
    delta = (baseline.double() - candidate.double()).abs()
    index = int(delta.flatten().argmax().item())
    return {"shape": list(baseline.shape), "worst_flat_index": index,
            "baseline_value": baseline.flatten()[index].item(),
            "candidate_value": candidate.flatten()[index].item(),
            "baseline_dtype": str(baseline.dtype), "candidate_dtype": str(candidate.dtype)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--rows", type=int, nargs="+", default=[32, 128, 512, 1024])
    parser.add_argument("--ks", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--row-chunks", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--lse-chunk", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--aa-blocks", type=int, default=6)
    parser.add_argument("--ab-blocks", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--a2-upstream-dir", help="isolated directory with unmodified pinned #7809 modules")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if min(args.rows + args.ks + args.row_chunks + [args.lse_chunk, args.warmup, args.aa_blocks, args.ab_blocks]) <= 0:
        parser.error("all dimensions and counts must be positive")
    dtype = getattr(torch, args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261029)
    random.seed(20261029)
    result = {"status": "RUNNING", "argv": sys.argv, "model_revision": MODEL_REVISION,
              "seed": 20261029, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "python": platform.python_version(), "platform": platform.platform(), "pid": os.getpid(), "dtype": args.dtype,
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "tf32_matmul": False, "autocast": False, "configuration": vars(args),
              "tolerance": dict(zip(("atol", "rtol"), TOLERANCES[dtype])),
              "scope": "teacher LM-head + full-vocabulary FP32 chunked LSE/gather + raw teacher top-K; excludes transformer, candidate prepass, transfers, loss/backward, profiler",
              "memory_scope": "PyTorch allocator absolute peaks include resident real model and inputs; per-sample before values permit incremental peaks. Reserved peaks inherit prior allocations (including A0); not isolated process reservations. nvidia-smi totals are boundary snapshots; WSL per-process memory can be unavailable.",
              "A2": {"status": "NOT_RUN", "reason": "No installed verified fused implementation of full teacher contract (full normalized candidates plus independent teacher top-K); selected-only/student kernels do not establish compatibility."},
              "cells": [], "script_sha256": sha256(__file__)}
    def save():
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers
        result["transformers"] = transformers.__version__
        result["gpu"] = torch.cuda.get_device_name()
        result["gpu_before_load"] = gpu_snapshot()
        model_dir = Path(args.model_dir)
        result["hashes"] = {name: sha256(model_dir / name) for name in ("model.safetensors", "tokenizer.json")}
        if result["hashes"] != {"model.safetensors": MODEL_SHA256, "tokenizer.json": TOKENIZER_SHA256}:
            raise ValueError("model/tokenizer hashes do not match fixed snapshot")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, attn_implementation="eager", local_files_only=True).to("cuda").eval()
        weight = model.get_output_embeddings().weight
        if model.get_output_embeddings().bias is not None:
            raise ValueError("this benchmark supports the pinned bias-free Qwen linear head")
        per_sequence = math.ceil(max(args.rows) / len(PROMPTS))
        sequences = []
        for prompt in PROMPTS:
            row = tokenizer.encode(prompt * (per_sequence // 10 + 2), add_special_tokens=False)
            while len(row) < per_sequence + 1:
                row += row
            sequences.append(row[:per_sequence + 1])
        tokens = torch.tensor(sequences, dtype=torch.long, device="cuda")
        positions = torch.arange(1, per_sequence + 1, device="cuda")
        causal = causal_positions(positions, tokens.shape[1])
        with torch.inference_mode():
            hidden_states = model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).last_hidden_state
            hidden = hidden_states[:, causal].reshape(-1, hidden_states.shape[-1]).clone()
        del hidden_states
        result["inputs"] = {"source": "fixed natural-language token sequences; real pinned Transformers eager-attention hidden states after final norm",
                            "tokens": tokens.cpu().tolist(), "token_positions": positions.cpu().tolist(),
                            "causal_hidden_positions": causal.cpu().tolist(), "batch": len(PROMPTS),
                            "hidden_shape": list(hidden.shape), "head_shape": list(weight.shape),
                            "hidden_sha256": hashlib.sha256(hidden.cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),
                            "student": "same real hidden/head, per-token FP32 temperature 0.8 clamped then cast to logit dtype before in-place divide; real dense student top-K candidate prepass; nonzero loss probe, no training claim"}
        result["gpu_after_hidden"] = gpu_snapshot()
        a2 = None
        if args.a2_upstream_dir:
            try:
                a2 = load_a2(args.a2_upstream_dir)
                result["A2"] = {"status": "AVAILABLE_FOR_TEST", "revision": "522514fe21300c4cf7b3800fd6cd827eb1caa2c9",
                                "contract": "unmodified upstream Triton selected full-normalized FP32 logprobs, explicit output cast to model dtype, plus separate row-chunked dense teacher topK",
                                "source_sha256": {name: sha256(Path(args.a2_upstream_dir) / name) for name in ("linear_topk_log_probs.py", "topk_log_probs_kernels.py")}}
                result["A2"]["source_git_blobs_verified"] = A2_BLOBS
            except Exception as exc:
                result["A2"] = {"status": "UNAVAILABLE", "error": repr(exc), "traceback": traceback.format_exc()}
        save()
        for n in args.rows:
            for k in args.ks:
                h = hidden[:n]
                cell = {"N": n, "K": k, "lse_chunk": args.lse_chunk,
                        "theoretical_dense_logits_bytes": n * weight.shape[0] * weight.element_size(),
                        "gpu_before": gpu_snapshot(), "variants": []}
                result["cells"].append(cell)
                try:
                    with torch.inference_mode():
                        # Candidate selection is a real dense student prepass,
                        # deliberately outside the teacher-stage timing.
                        candidates, student_lps = [], []
                        temperature = torch.full((n,), .8, device=h.device, dtype=torch.float32)
                        for start in range(0, n, 64):
                            z = torch.nn.functional.linear(h[start:start + 64], weight)
                            scale_student_logits_(z, temperature[start:start + 64])
                            full_student_lp = z.log_softmax(-1)
                            ids = full_student_lp.topk(k, -1).indices
                            candidates.append(ids)
                            student_lps.append(full_student_lp.gather(-1, ids))
                        ids = torch.cat(candidates)
                        student_lp = torch.cat(student_lps).clone()
                        del z, full_student_lp, candidates, student_lps
                        baseline = teacher_stage(h, weight, ids, topk=k, lse_chunk=args.lse_chunk)
                        cell["candidate_ids_sha256"] = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
                        cell["candidate_ids_first_32"] = ids[:32].cpu().tolist()
                    aa = []
                    with torch.inference_mode():
                        fn = lambda: teacher_stage(h, weight, ids, topk=k, lse_chunk=args.lse_chunk)
                        for _ in range(args.warmup):
                            fn()
                        for block in range(args.aa_blocks):
                            for label in (("A0a", "A0b") if block % 2 == 0 else ("A0b", "A0a")):
                                aa.append(timed(fn, label, block))
                    cell["AA"] = {"samples": aa, "summary": summarize(aa)}
                    for chunk in args.row_chunks:
                        variant = {"name": "A1", "row_chunk": chunk,
                                   "theoretical_peak_logits_bytes": min(n, chunk) * weight.shape[0] * weight.element_size()}
                        cell["variants"].append(variant)
                        with torch.inference_mode():
                            candidate = teacher_stage(h, weight, ids, topk=k, row_chunk=chunk, lse_chunk=args.lse_chunk)
                        variant["correctness"] = compare_outputs(baseline, candidate, h, weight, ids, student_lp, lse_chunk=args.lse_chunk)
                        del candidate
                        functions = {"A0": lambda: teacher_stage(h, weight, ids, topk=k, lse_chunk=args.lse_chunk),
                                     "A1": lambda: teacher_stage(h, weight, ids, topk=k, row_chunk=chunk, lse_chunk=args.lse_chunk)}
                        samples = []
                        with torch.inference_mode():
                            for _ in range(args.warmup):
                                functions["A0"]()
                                functions["A1"]()
                            for block in range(args.ab_blocks):
                                order = ("A0", "A1", "A1", "A0") if block % 2 == 0 else ("A1", "A0", "A0", "A1")
                                for label in order:
                                    samples.append(timed(functions[label], label, block))
                        variant.update({"samples": samples, "summary": summarize(samples), "gpu_after": gpu_snapshot()})
                        save()
                    if a2 is not None:
                        variant = {"name": "A2_upstream_plus_dense_topk", "row_chunk": args.row_chunks[0], "samples": []}
                        cell["variants"].append(variant)
                        try:
                            with torch.inference_mode():
                                candidate = a2(h, weight, ids, topk=k, row_chunk=args.row_chunks[0])
                            variant["correctness"] = compare_outputs(baseline, candidate, h, weight, ids, student_lp, lse_chunk=args.lse_chunk)
                            del candidate
                            with torch.inference_mode():
                                native_logps = a2.native_logprobs(h, weight, ids)
                                variant["native_fp32_logprobs_vs_A0"] = worst_difference(baseline.candidate_logprobs.float(), native_logps)
                                variant["native_fp32_logprobs_vs_A0"]["max_abs"] = (baseline.candidate_logprobs.float() - native_logps).abs().max().item()
                                del native_logps
                            functions = {"A1": lambda: teacher_stage(h, weight, ids, topk=k, row_chunk=args.row_chunks[0], lse_chunk=args.lse_chunk),
                                         "A2": lambda: a2(h, weight, ids, topk=k, row_chunk=args.row_chunks[0])}
                            with torch.inference_mode():
                                for _ in range(args.warmup):
                                    functions["A1"]()
                                    functions["A2"]()
                                for block in range(args.ab_blocks):
                                    for label in (("A1", "A2", "A2", "A1") if block % 2 == 0 else ("A2", "A1", "A1", "A2")):
                                        variant["samples"].append(timed(functions[label], label, block))
                            variant["summary"] = summarize(variant["samples"])
                            variant["status"] = variant["correctness"]["status"]
                        except Exception as exc:
                            variant.update({"status": "FAILED", "correctness": {"status": "FAILED"}, "error": repr(exc), "traceback": traceback.format_exc()})
                        variant["gpu_after"] = gpu_snapshot()
                        save()
                    if args.profile and n == max(args.rows) and k == max(args.ks):
                        try:
                            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                            with torch.inference_mode(), torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
                                for label, chunk in (("A0", None), ("A1", args.row_chunks[0])):
                                    with torch.profiler.record_function(label):
                                        teacher_stage(h, weight, ids, topk=k, row_chunk=chunk, lse_chunk=args.lse_chunk)
                                        torch.cuda.synchronize()
                                if a2 is not None:
                                    with torch.profiler.record_function("A2_upstream_plus_dense_topk"):
                                        a2(h, weight, ids, topk=k, row_chunk=args.row_chunks[0])
                                        torch.cuda.synchronize()
                            trace = out.with_suffix(".trace.json")
                            prof.export_chrome_trace(str(trace))
                            cell["profile"] = {"status": "ACQUIRED", "path": str(trace), "cuda_events": sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)}
                        except Exception as exc:
                            cell["profile"] = {"status": "UNAVAILABLE", "error": repr(exc), "traceback": traceback.format_exc()}
                    cell["status"] = "PASS" if all(v["correctness"]["status"] == "PASS" for v in cell["variants"]) else "FAILED_CORRECTNESS"
                    del baseline, ids, student_lp
                except torch.cuda.OutOfMemoryError as exc:
                    cell.update({"status": "OOM", "error": repr(exc), "traceback": traceback.format_exc()})
                    torch.cuda.empty_cache()
                cell["gpu_after"] = gpu_snapshot()
                save()
        result["gpu_final"] = gpu_snapshot()
        result["status"] = "PASS" if all(c["status"] == "PASS" for c in result["cells"]) else "COMPLETED_WITH_FAILURES"
    except Exception as exc:
        result.update({"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()})
    save()
    print(json.dumps({"status": result["status"], "out": str(out), "cells": len(result["cells"])}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
