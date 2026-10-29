#!/usr/bin/env python3
"""One real student RMSNorm update with dense versus row-chunked teacher scores.

This is a fixed-input distillation consumer check, not RL training or a latency
benchmark. Only the student's final RMSNorm weight is trained to fit an 8 GiB
device. The teacher runs its complete transformer forward in both paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference.memory_bounded_teacher import TeacherScores, install_memory_bounded_teacher

UPSTREAM_SHA = "87cd055e872f9fdd7b976fcd797a0934b6dba265"
TOLERANCES = {"bfloat16": (0.1, 0.01), "float32": (5e-5, 2e-5)}
GRADIENT_RELATIVE_L2_MAX = {"bfloat16": 0.05, "float32": 1e-4}
FIXTURES = (
    ("Explain how a GPU computes matrix multiplication. ",
     "It moves matrix tiles into fast memory and accumulates their products."),
    ("Why does causal language modeling use shifted labels? ",
     "Each hidden state predicts the next token without seeing that token."),
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def command(args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    return {"command": args, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value


def compare(reference, candidate, atol, rtol):
    reference, candidate = reference.detach().cpu().double(), candidate.detach().cpu().double()
    if reference.shape != candidate.shape:
        return {"pass": False, "reason": "shape mismatch", "reference_shape": list(reference.shape),
                "candidate_shape": list(candidate.shape)}
    finite = torch.isfinite(reference) & torch.isfinite(candidate)
    difference = (candidate - reference).abs()
    denominator = reference.norm().item()
    delta_norm = (candidate - reference).norm().item()
    relative_l2 = delta_norm / denominator if denominator else (0.0 if delta_norm == 0 else math.inf)
    failed = ~finite | (difference > atol + rtol * reference.abs())
    return {"pass": not failed.any().item(), "elements": reference.numel(),
            "nonfinite_elements": (~finite).sum().item(), "failed_elements": failed.sum().item(),
            "max_abs": difference.max().item() if difference.numel() else 0.0,
            "relative_l2": relative_l2, "atol": atol, "rtol": rtol}


def compare_gradient(reference, candidate, dtype):
    """Apply the pre-existing dtype-specific element and relative-L2 gates."""
    atol, rtol = TOLERANCES[dtype]
    result = compare(reference, candidate, atol, rtol)
    result["relative_l2_max"] = GRADIENT_RELATIVE_L2_MAX[dtype]
    result["pass"] = result["pass"] and result["relative_l2"] <= result["relative_l2_max"]
    return result


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def observed_call(fn, device):
    synchronize(device)
    before = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        before = {"allocated_bytes": torch.cuda.memory_allocated(device),
                  "reserved_bytes": torch.cuda.memory_reserved(device)}
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    observation = {"wall_ms": (time.perf_counter() - start) * 1000,
                   "scope": "one diagnostic call; not a repeated performance benchmark", "before": before}
    if device.type == "cuda":
        observation.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result, observation


def build_inputs(tokenizer, *, tiny, max_tokens, device):
    if tiny:
        sequences = [[2, 5, 7, 11, 13, 17, 19, 23], [3, 9, 12, 14, 18, 22]]
        prompt_lengths = [4, 3]
    else:
        sequences, prompt_lengths = [], []
        for prompt, response in FIXTURES:
            prefix = tokenizer.encode(prompt, add_special_tokens=False)
            suffix = tokenizer.encode(response, add_special_tokens=False)
            sequence = (prefix + suffix)[:max_tokens]
            if len(sequence) <= len(prefix):
                raise ValueError("max_tokens must leave at least one response token")
            sequences.append(sequence)
            prompt_lengths.append(len(prefix))
    width = max(map(len, sequences))
    pad_token_id = 0 if tiny else tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("the tokenizer has no padding or EOS token")
    tokens = torch.full((len(sequences), width), pad_token_id, dtype=torch.int64)
    attention = torch.zeros_like(tokens)
    response_rows = torch.zeros_like(tokens, dtype=torch.bool)
    for row, (sequence, prompt_length) in enumerate(zip(sequences, prompt_lengths)):
        offset = width - len(sequence)
        tokens[row, offset:] = torch.tensor(sequence)
        attention[row, offset:] = 1
        # Logit row i predicts token i+1. IDs below already address logit rows;
        # only the response mask receives this next-token alignment.
        response_rows[row, offset + prompt_length - 1:offset + len(sequence) - 1] = True
    positions = (attention.cumsum(-1) - 1).clamp_min(0)
    if not response_rows.any():
        raise ValueError("the fixture has no active response prediction rows")
    inputs = {"input_ids": tokens.to(device), "attention_mask": attention.to(device),
              "position_ids": positions.to(device)}
    metadata = {"fixture_kind": "synthetic token IDs" if tiny else "tokenized fixed prompt/response text; not RL trajectories",
                "texts": None if tiny else FIXTURES, "input_ids": tokens.tolist(),
                "attention_mask": attention.tolist(), "position_ids": positions.tolist(),
                "response_logit_row_mask": response_rows.tolist(), "prompt_lengths": prompt_lengths,
                "active_response_rows": response_rows.sum().item(),
                "hashes": {key: tensor_hash(value) for key, value in inputs.items()}}
    metadata["hashes"]["response_logit_row_mask"] = tensor_hash(response_rows)
    return inputs, response_rows.to(device), metadata


def select_student_candidates(scaled_student_logits, topk):
    """Match the pinned #6920 full log-softmax then Top-K selection order."""
    return F.log_softmax(scaled_student_logits, dim=-1).topk(topk, dim=-1).indices


def dense_teacher(teacher, inputs, ids, chunk_rows):
    logits = teacher(**inputs, use_cache=False).logits
    flat = logits.reshape(-1, logits.shape[-1])
    flat_ids = ids.reshape(-1, ids.shape[-1])
    logps = torch.empty_like(ids, dtype=logits.dtype).reshape_as(flat_ids)
    for start in range(0, flat.shape[0], chunk_rows):
        fp32 = flat[start:start + chunk_rows].float()
        logps[start:start + chunk_rows] = (
            fp32.gather(-1, flat_ids[start:start + chunk_rows]) - fp32.logsumexp(-1, keepdim=True)
        ).to(logits.dtype)
    return TeacherScores(logps.reshape_as(ids).cpu(), logits.topk(ids.shape[-1], -1).indices.cpu(), chunk_rows)


def consumer(student_logits, teacher_scores, ids, response_rows):
    # #6920 engine temperature semantics: clamp FP32 tensor, cast to logits
    # dtype, then divide before the loss's dtype-preserving log_softmax.
    temperature = torch.full(student_logits.shape[:-1], 0.8, dtype=torch.float32, device=student_logits.device)
    student_logits.div_(temperature.clamp_min(1e-8).unsqueeze(-1).to(student_logits.dtype))
    student_logits.retain_grad()
    student_logp = F.log_softmax(student_logits, dim=-1).gather(-1, ids)
    teacher_logp = teacher_scores.candidate_logprobs.to(student_logits.device).detach().float()
    teacher_ids = teacher_scores.teacher_topk_ids.to(student_logits.device)
    student_mass = student_logp.exp().sum(-1)
    teacher_mass = teacher_logp.exp().sum(-1)
    logq = (student_logp - student_logp.logsumexp(-1, keepdim=True)).clamp_min(-10)
    logp = (teacher_logp - teacher_logp.logsumexp(-1, keepdim=True)).clamp_min(-10)
    # reverse_kl_divergence casts both sides to FP32 after normalization/clamp.
    # Match upstream: share each FP32 cast so BF16 backward accumulates before
    # converting the combined gradient back to the input dtype.
    logq_fp32 = logq.float()
    logp_fp32 = logp.float()
    token_loss = (logq_fp32.exp() * (logq_fp32 - logp_fp32)).sum(-1)
    # Keep the original mixed-dtype overlap diagnostic before loss-wrapper clamp.
    overlap = (ids.unsqueeze(-1) == teacher_ids.unsqueeze(-2)).any(-1)
    overlap_count = overlap.sum(-1)
    token_rkl = logq.exp() * (logq - logp)
    advantage = (-token_rkl * overlap).sum(-1) / overlap_count.clamp_min(1)
    advantage = torch.where(overlap_count > 0, advantage, torch.zeros_like(advantage))
    wrapped_loss = token_loss.clamp_min(0).clamp(-10, 10)
    loss = wrapped_loss[response_rows].mean()
    return loss, {"student_mass": student_mass.detach().cpu(), "teacher_mass": teacher_mass.detach().cpu(),
                  "overlap_count": overlap_count.detach().cpu(), "overlap_token_advantage": advantage.detach().cpu(),
                  "token_loss_before_wrapper": token_loss.detach().cpu(), "masked_loss": loss.detach().cpu()}


def update_student(student, initial_norm, teacher_scores, inputs, ids, response_rows, learning_rate, optimizer_name):
    norm = student.model.norm.weight
    with torch.no_grad():
        norm.copy_(initial_norm)
    student.zero_grad(set_to_none=True)
    optimizer = (torch.optim.AdamW([norm], lr=learning_rate, weight_decay=0)
                 if optimizer_name == "adamw" else torch.optim.SGD([norm], lr=learning_rate))
    logits = student(**inputs, use_cache=False).logits
    loss, diagnostics = consumer(logits, teacher_scores, ids, response_rows)
    loss.backward()
    if norm.grad is None or logits.grad is None:
        raise RuntimeError("student RMSNorm or logits gradient was not produced")
    gradient = norm.grad.detach().cpu().float().clone()
    logits_gradient = logits.grad.detach().cpu().float().clone()
    optimizer.step()
    updated = norm.detach().cpu().float().clone()
    initial = initial_norm.detach().cpu().float()
    result = {"loss": loss.detach().cpu().float(), "diagnostics": diagnostics,
              "norm_gradient": gradient, "logits_gradient": logits_gradient,
              "updated_norm": updated, "norm_update": updated - initial,
              "raw": {"loss": loss.item(), "norm_gradient": gradient.tolist(), "updated_norm": updated.tolist(),
                      "gradient_norm": gradient.norm().item(), "gradient_nonzero_elements": gradient.count_nonzero().item(),
                      "changed_parameter_elements": (updated != initial).sum().item(),
                      "parameter_delta_l2": (updated - initial).norm().item(),
                      "hashes": {"norm_gradient": tensor_hash(gradient), "logits_gradient": tensor_hash(logits_gradient),
                                 "updated_norm": tensor_hash(updated)}}}
    del logits, optimizer
    return result


def run(args, report):
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(731)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(731)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report["environment"].update(cuda=torch.version.cuda, device_name=torch.cuda.get_device_name(device),
                                      capability=list(torch.cuda.get_device_capability(device)))
    if args.tiny:
        config = Qwen2Config(vocab_size=97, hidden_size=24, intermediate_size=40, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
                             attention_dropout=0.0, max_position_embeddings=64)
        config._attn_implementation = "eager"
        teacher = Qwen2ForCausalLM(config).to(dtype=dtype, device=device)
        student = Qwen2ForCausalLM(config).to(dtype=dtype, device=device)
        student.load_state_dict(teacher.state_dict())
        tokenizer = None
        report["weight_source"] = "random tiny CPU contract fixture; no real-model claim"
    else:
        if args.model_dir is None:
            raise ValueError("--model-dir is required unless --tiny is selected")
        directory = Path(args.model_dir).resolve()
        files = sorted(set(directory.glob("*.safetensors")) | set(directory.glob("*.json")))
        report["weight_source"] = {"directory": str(directory), "files": [
            {"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)} for path in files]}
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            directory, local_files_only=True, dtype=dtype, attn_implementation="sdpa").to(device)
        student = AutoModelForCausalLM.from_pretrained(
            directory, local_files_only=True, dtype=dtype, attn_implementation="sdpa").to(device)
    teacher.eval().requires_grad_(False)
    student.eval().requires_grad_(False)
    student.model.norm.weight.requires_grad_(True)
    report["teacher_student_same_initial_weights"] = all(
        torch.equal(teacher.state_dict()[name], value) for name, value in student.state_dict().items())
    if not report["teacher_student_same_initial_weights"]:
        raise RuntimeError("teacher/student initial weights differ")
    install_memory_bounded_teacher(teacher)
    inputs, response_rows, report["inputs"] = build_inputs(
        tokenizer, tiny=args.tiny, max_tokens=args.max_tokens, device=device)
    with torch.no_grad():
        logits = student(**inputs, use_cache=False).logits
        temperature = torch.full(logits.shape[:-1], 0.8, device=device, dtype=torch.float32)
        logits.div_(temperature.clamp_min(1e-8).unsqueeze(-1).to(logits.dtype))
        ids = select_student_candidates(logits, args.topk)
        del logits
    report["candidate_ids"] = ids.cpu().tolist()
    report["candidate_ids_sha256"] = tensor_hash(ids)
    initial_norm = student.model.norm.weight.detach().clone()
    report["initial_norm"] = initial_norm.float().cpu().tolist()
    report["trainable_parameters"] = [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                                       for name, value in student.named_parameters() if value.requires_grad]
    with torch.no_grad():
        dense, dense_observation = observed_call(
            lambda: dense_teacher(teacher, inputs, ids, args.lse_chunk_rows), device)
        report["teacher"] = {"A0": {"observation": dense_observation,
                                       "candidate_logprobs": dense.candidate_logprobs.float().tolist(),
                                       "teacher_topk_ids": dense.teacher_topk_ids.tolist()}}

        def chunked_teacher():
            value = teacher(**inputs, teacher_topk_ids=ids, teacher_chunk_rows=args.chunk_rows)
            return TeacherScores(value.candidate_logprobs.cpu(), value.teacher_topk_ids.cpu(), value.chunk_rows)

        chunked, chunked_observation = observed_call(chunked_teacher, device)
    report["teacher"]["A1"] = {"observation": chunked_observation,
                                  "candidate_logprobs": chunked.candidate_logprobs.float().tolist(),
                                  "teacher_topk_ids": chunked.teacher_topk_ids.tolist()}
    atol, rtol = TOLERANCES[args.dtype]
    checks = {"teacher_candidate_logprobs": compare(dense.candidate_logprobs, chunked.candidate_logprobs, atol, rtol)}
    # Top-K ties may permute; compare set membership, and report exact IDs too.
    checks["teacher_topk_set"] = {"pass": torch.equal(dense.teacher_topk_ids.sort(-1).values,
                                                       chunked.teacher_topk_ids.sort(-1).values),
                                   "scope": "strict set equality; alternative exact-boundary ties can cause a conservative failure"}
    outputs = []
    for name, scores in (("A0", dense), ("A1", chunked)):
        result, observation = observed_call(lambda: update_student(
            student, initial_norm, scores, inputs, ids, response_rows, args.learning_rate, args.optimizer), device)
        report.setdefault("student_updates", {})[name] = dict(result["raw"], observation=observation,
            diagnostics={key: value.tolist() for key, value in result["diagnostics"].items()})
        checks[f"{name}_nonzero_gradient_and_update"] = {"pass": result["raw"]["gradient_norm"] > 0
            and math.isfinite(result["raw"]["gradient_norm"]) and result["raw"]["changed_parameter_elements"] > 0}
        outputs.append(result)
    for key in ("loss", "norm_gradient", "logits_gradient", "updated_norm", "norm_update"):
        if "gradient" in key or key == "norm_update":
            checks[key] = compare_gradient(outputs[0][key], outputs[1][key], args.dtype)
        else:
            checks[key] = compare(outputs[0][key], outputs[1][key], atol, rtol)
    for key in outputs[0]["diagnostics"]:
        checks[key] = compare(outputs[0]["diagnostics"][key], outputs[1]["diagnostics"][key], atol, rtol)
    report["checks"] = checks
    report["teacher_gradients_absent"] = all(value.grad is None for value in teacher.parameters())
    report["status"] = "PASS" if all(value["pass"] for value in checks.values()) and report["teacher_gradients_absent"] else "FAIL"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=tuple(TOLERANCES), default="bfloat16")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--chunk-rows", type=int, default=16)
    parser.add_argument("--lse-chunk-rows", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    output = args.out.resolve()
    if output.is_relative_to(ROOT):
        parser.error("--out must be outside the repository")
    if args.topk <= 0 or args.chunk_rows <= 0 or args.lse_chunk_rows <= 0 or args.max_tokens <= 1:
        parser.error("topk/chunk sizes must be positive and max-tokens must exceed one")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if args.tiny and args.device != "cpu":
        parser.error("--tiny is a CPU-only contract smoke mode")
    report = {"status": "STARTED", "scope": "complete HF Teacher forward plus one final-RMSNorm-only student update; no FSDP/Ray/RL training claim",
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "frozen_tolerances": {"atol": TOLERANCES[args.dtype][0], "rtol": TOLERANCES[args.dtype][1],
                                    "gradient_relative_l2_max": GRADIENT_RELATIVE_L2_MAX[args.dtype]},
              "consumer_semantics": {"student_temperature": 0.8, "teacher_temperature": 1.0,
                                     "log_prob_min_clamp_after_support_normalization": -10,
                                     "loss_wrapper_nonnegative_clamp": True, "loss_max_clamp": 10,
                                     "aggregation": "mean over next-token-aligned response rows",
                                     "optimizer_weight_decay": 0},
              "sources": {"verl_head": UPSTREAM_SHA,
                          "losses_blob": "97d3ebf485a5ce9e1b1a435cce427d6d89284dc6",
                          "teacher_worker_blob": "bd03d07fbedbc2be72f77f425a6f4aaa3c8f3f07",
                          "local_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in
                                          (Path(__file__).resolve(), ROOT / "reference/memory_bounded_teacher.py")}},
              "git": {"head": command(["git", "rev-parse", "HEAD"]), "status": command(["git", "status", "--short", "--branch"])},
              "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                              "transformers": transformers.__version__},
              "limitations": ["fixed synthetic text, not sampled RL trajectories", "only student final RMSNorm weight is updated",
                              "optimizer state only covers the final RMSNorm; no full-model training claim", "single diagnostic timings are not performance evidence",
                              "strict top-K set gate may conservatively fail alternative exact ties"]}
    exit_code = 0
    try:
        run(args, report)
        exit_code = 0 if report["status"] == "PASS" else 1
    except Exception as error:
        report["status"] = "ERROR"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        exit_code = 1
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(safe_json(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(output), "scope": report["scope"]}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
