#!/usr/bin/env python3
"""Probe readout-budget estimates during a real complete Qwen2 Teacher forward.

This diagnostic resets the CUDA allocator high-water mark at the first LM-head
pre-hook, then reads it after the outer model/FSDP post-forward hooks. It is not
a latency benchmark. The estimate excludes total model memory, other processes,
external CUDA allocations/workspaces and fragmentation; it is no OOM guarantee.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from reference.memory_budget import select_chunk_rows
from scripts.full_teacher_benchmark import (
    MODEL_REVISION, MODEL_SHA256, TOKENIZER_SHA256, PROMPTS,
    gpu_snapshot, sha256, tensor_sha256, wrap_fsdp2,
)


def memory_snapshot():
    return {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def probe_once(module, head, inputs, ids, decision):
    state = {"status": "RUNNING", "head_calls": [], "normal_outer_post_forward": False}
    handles = []

    def before_head(layer, positional):
        torch.cuda.synchronize()
        if not state["head_calls"]:
            state["first_head_pre_hook"] = memory_snapshot()
            torch.cuda.reset_peak_memory_stats()
            state["head_peak_reset_at_unix_time"] = time.time()
        state["head_calls"].append({"input_shape": list(positional[0].shape),
                                    "before_projection_allocated_bytes": torch.cuda.memory_allocated()})

    def after_head(layer, positional, output):
        state["head_calls"][-1].update(output_shape=list(output.shape),
                                       output_dtype=str(output.dtype),
                                       after_projection_allocated_bytes=torch.cuda.memory_allocated())

    def after_outer(outer, positional, output):
        # Registered after fully_shard installs its own root hooks, so this
        # reads after normal FSDP reshard/offload post-forward processing.
        torch.cuda.synchronize()
        state["outer_post_forward"] = memory_snapshot()
        state["normal_outer_post_forward"] = True

    try:
        handles = [head.register_forward_pre_hook(before_head),
                   head.register_forward_hook(after_head),
                   module.register_forward_hook(after_outer)]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        state["before_outer_forward"] = memory_snapshot()
        with torch.no_grad():
            output = module(**inputs, teacher_topk_ids=ids, teacher_chunk_rows=decision.rows)
        if not state["normal_outer_post_forward"] or not state["head_calls"]:
            raise RuntimeError("the real outer/head hook boundary was not observed")
        before_head_allocated = state["first_head_pre_hook"]["allocated_bytes"]
        actual_head_increment = state["outer_post_forward"]["peak_allocated_bytes"] - before_head_allocated
        # The supported adapter allocates its int64 Top-K output before its
        # first head call. It remains live through this interval, so add its
        # bytes back when comparing with an estimate of the whole readout.
        preallocated_topk_bytes = output.teacher_topk_ids.numel() * output.teacher_topk_ids.element_size()
        readout_increment = actual_head_increment + preallocated_topk_bytes
        observed_rows = [math.prod(call["input_shape"][:-1]) for call in state["head_calls"]]
        total_rows = ids.numel() // ids.shape[-1]
        shape_ok = (output.chunk_rows == decision.rows and sum(observed_rows) == total_rows and
                    max(observed_rows) <= min(total_rows, decision.rows) and
                    all(call.get("output_shape") == [rows, module.config.vocab_size]
                        for call, rows in zip(state["head_calls"], observed_rows)))
        logp = output.candidate_logprobs.detach().cpu()
        topk = output.teacher_topk_ids.detach().cpu()
        output_ok = (list(logp.shape) == list(ids.shape) and list(topk.shape) == list(ids.shape)
                     and logp.dtype == torch.bfloat16 and topk.dtype == torch.int64
                     and bool(torch.isfinite(logp).all()))
        estimate_ok = readout_increment <= decision.estimated_bytes
        budget_ok = decision.budget_bytes is None or readout_increment <= decision.budget_bytes
        pre_head_peak = state["first_head_pre_hook"]["peak_allocated_bytes"]
        head_peak = state["outer_post_forward"]["peak_allocated_bytes"]
        state.update(
            status="PASS" if shape_ok and output_ok and estimate_ok and budget_ok else "FAILED",
            head_increment_allocated_bytes=actual_head_increment,
            known_preallocated_topk_output_bytes=preallocated_topk_bytes,
            conservative_readout_increment_allocated_bytes=readout_increment,
            full_forward_peak_allocated_bytes=max(pre_head_peak, head_peak),
            full_forward_increment_allocated_bytes=max(pre_head_peak, head_peak)
                                                   - state["before_outer_forward"]["allocated_bytes"],
            estimated_minus_observed_bytes=decision.estimated_bytes - readout_increment,
            observed_over_estimate_ratio=readout_increment / decision.estimated_bytes,
            within_estimate=estimate_ok,
            within_budget=budget_ok,
            budget_minus_observed_bytes=None if decision.budget_bytes is None else decision.budget_bytes - readout_increment,
            selected_chunk_rows=decision.rows, returned_chunk_rows=output.chunk_rows,
            head_shape_bound_pass=shape_ok,
            output_structure_finite_pass=output_ok,
            output={"candidate_logprobs_sha256": tensor_sha256(logp),
                    "teacher_topk_ids_sha256": tensor_sha256(topk),
                    "shape": list(logp.shape), "logprob_dtype": str(logp.dtype),
                    "teacher_topk_dtype": str(topk.dtype)},
            numerical_equivalence={"status": "NOT_RUN",
                                   "reason": "this budget probe is separate from the dense-reference correctness gate"},
        )
    except Exception as exc:
        state.update(status="OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "ERROR",
                     error=repr(exc), traceback=traceback.format_exc())
        try:
            state["memory_at_error_without_synchronization"] = memory_snapshot()
        except Exception as memory_exc:
            state["failure_memory_query_error"] = repr(memory_exc)
    finally:
        for handle in handles:
            handle.remove()
    return state


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True, help="raw JSON path outside the repository")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--budgets-mib", type=float, nargs="+", required=True)
    parser.add_argument("--max-chunk-rows", type=int, default=512)
    parser.add_argument("--safety-factor", type=float, default=1.25)
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--fsdp2", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()
    if min(args.rows, args.k, args.max_chunk_rows, args.repeats) <= 0 or args.warmup < 0:
        parser.error("dimensions/repeats must be positive and warmup nonnegative")
    if any(not math.isfinite(value) for value in args.budgets_mib):
        parser.error("budgets must be finite")
    if not math.isfinite(args.safety_factor) or args.safety_factor < 1:
        parser.error("safety factor must be finite and at least 1")
    if args.cpu_offload and not args.fsdp2:
        parser.error("--cpu-offload requires --fsdp2")
    if Path(args.out).resolve().is_relative_to(REPO_ROOT.resolve()):
        parser.error("--out must be outside the source repository")
    return args


def main():
    args = get_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "RUNNING", "argv": sys.argv, "configuration": vars(args),
        "scope": __doc__, "pid": os.getpid(), "seed": 20261029,
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "model_revision": MODEL_REVISION, "dtype": "bfloat16", "use_cache": False,
        "source_git_head": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_git_status": subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--short"], text=True),
        "source_hashes": {str(path.relative_to(REPO_ROOT)): sha256(path) for path in (
            Path(__file__), REPO_ROOT / "reference/memory_budget.py",
            REPO_ROOT / "reference/memory_bounded_teacher.py",
            REPO_ROOT / "scripts/full_teacher_benchmark.py",
        )},
        "measurement": {
            "kind": "diagnostic allocation probe; hook synchronizations invalidate performance comparisons",
            "start": "first LM-head pre-hook after backbone/final norm; GPU work synchronized, then peak reset",
            "end": "outer module post-forward hook registered after FSDP hooks; GPU work synchronized",
            "head_increment": "post-forward head-interval peak allocated minus first-head allocated baseline",
            "budget_comparison": "head increment plus known int64 Top-K output allocated before the first head call",
            "estimate_scope": "compact outputs plus conservative row-tile PyTorch allocations",
            "accounting_assumption": "the source-hashed adapter allocates only its int64 Top-K output before the first head call",
            "excluded": ["total model/hidden/input residency", "other GPU processes",
                         "external CUDA allocator/workspace", "allocator fragmentation"],
            "oom_guarantee": False,
            "latency_and_training_benefit": "NOT_MEASURED",
        },
        "fsdp": {"enabled": args.fsdp2, "world_size": 1, "cpu_offload": args.cpu_offload,
                 "multi_gpu_sharding_and_communication": "NOT_RUN"},
        "cases": [],
    }

    def save():
        temporary = out.with_suffix(out.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + chr(10), encoding="utf-8")
        temporary.replace(out)

    distributed = False
    rendezvous = None
    active_case = None
    active_stage = "initialization"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA device is required; CPU allocation results cannot replace this probe")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers
        result["transformers"] = transformers.__version__
        torch.manual_seed(20261029)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        result["tf32_matmul"] = False
        result["autocast"] = False
        result["gpu_before_load"] = gpu_snapshot()
        result["device"] = {"name": torch.cuda.get_device_name(),
                            "capability": list(torch.cuda.get_device_capability())}
        model_dir = Path(args.model_dir)
        result["model_hashes"] = {name: sha256(model_dir / name)
                                  for name in ("model.safetensors", "tokenizer.json")}
        if result["model_hashes"] != {"model.safetensors": MODEL_SHA256, "tokenizer.json": TOKENIZER_SHA256}:
            raise ValueError("model/tokenizer hashes do not match the fixed snapshot")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, attn_implementation=args.attention, local_files_only=True,
        ).to("cuda").eval()
        install_memory_bounded_teacher(model)
        head = model.lm_head
        if args.fsdp2:
            rendezvous = tempfile.TemporaryDirectory(prefix="teacher-budget-fsdp2-")
            torch.distributed.init_process_group(
                "nccl", init_method="file://" + str(Path(rendezvous.name) / "rendezvous"),
                rank=0, world_size=1,
            )
            distributed = True
            model = wrap_fsdp2(model, torch.bfloat16, args.cpu_offload)
        prompt_tokens = tokenizer.encode(" ".join(PROMPTS), add_special_tokens=False)
        tokens_list = (prompt_tokens * math.ceil(args.rows / len(prompt_tokens)))[:args.rows]
        tokens = torch.tensor([tokens_list], dtype=torch.long, device="cuda")
        inputs = {"input_ids": tokens, "attention_mask": torch.ones_like(tokens),
                  "position_ids": torch.arange(args.rows, device="cuda").unsqueeze(0), "use_cache": False}
        vocab = model.config.vocab_size
        if args.k > vocab:
            raise ValueError("candidate width exceeds vocabulary")
        stride = 7919 if math.gcd(7919, vocab) == 1 else 1
        ids = (tokens.unsqueeze(-1) + torch.arange(args.k, device="cuda") * stride
               + torch.arange(args.rows, device="cuda").reshape(1, args.rows, 1) * 17) % vocab
        result["input"] = {
            "kind": "one synthetic natural-language sequence and deterministic distinct candidate IDs; not an RL trajectory",
            "input_ids": tokens_list, "candidate_ids": ids.cpu().tolist(),
            "input_ids_sha256": tensor_sha256(tokens), "candidate_ids_sha256": tensor_sha256(ids),
            "attention_mask_sha256": tensor_sha256(inputs["attention_mask"]),
            "position_ids_sha256": tensor_sha256(inputs["position_ids"]),
            "shape": list(tokens.shape), "candidate_shape": list(ids.shape),
        }
        save()
        for budget in args.budgets_mib:
            case = {"budget_mib": budget, "status": "RUNNING", "samples": []}
            result["cases"].append(case)
            active_case = case
            active_stage = "budget_selection"
            try:
                decision = select_chunk_rows(
                    args.rows, vocab, args.k, torch.bfloat16,
                    requested_rows=args.max_chunk_rows,
                    head_memory_budget_mib=budget, safety_factor=args.safety_factor,
                )
                case["decision"] = asdict(decision)
            except ValueError as exc:
                case.update(status="REJECTED_BUDGET", error=repr(exc))
                save()
                continue
            active_stage = "warmup"
            with torch.no_grad():
                for _ in range(args.warmup):
                    model(**inputs, teacher_topk_ids=ids, teacher_chunk_rows=decision.rows)
            active_stage = "head_allocation_probe"
            for repeat in range(args.repeats):
                sample = probe_once(model, head, inputs, ids, decision)
                sample["repeat"] = repeat
                case["samples"].append(sample)
                save()
                if sample["status"] in {"ERROR", "OOM"}:
                    # A failed outer FSDP forward may not have run cleanup hooks.
                    # Do not retry that process or turn a lower B into a pass.
                    case["status"] = sample["status"]
                    raise RuntimeError("probe forward failed; stop without retrying uncertain model/FSDP state")
            case["status"] = "PASS" if all(sample["status"] == "PASS" for sample in case["samples"]) else "FAILED_ESTIMATE_OR_BOUNDARY"
            case["gpu_after"] = gpu_snapshot()
            save()
            print(json.dumps({"budget_mib": budget, "chunk_rows": decision.rows,
                              "status": case["status"], "out": str(out)}), flush=True)
        result["status"] = "PASS" if result["cases"] and all(case["status"] == "PASS" for case in result["cases"]) else "COMPLETED_WITH_FAILURES"
        result["gpu_final"] = gpu_snapshot()
    except Exception as exc:
        if active_case is not None and active_case["status"] == "RUNNING":
            active_case.update(status="OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "ERROR",
                               failed_stage=active_stage, error=repr(exc))
        result["failed_stage"] = active_stage
        result.update(status="OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "ERROR",
                      error=repr(exc), traceback=traceback.format_exc())
    finally:
        save()
        if distributed:
            torch.distributed.destroy_process_group()
        if rendezvous is not None:
            rendezvous.cleanup()
    print(json.dumps({"status": result["status"], "out": str(out)}), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())