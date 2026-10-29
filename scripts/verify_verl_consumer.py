#!/usr/bin/env python3
"""Compare the real pinned verl reverse-KL consumer with the project consumer.

This entry replays recorded teacher scores with a real student model. It imports
and executes verl's compute_reverse_kl_topk; it does not execute a trainer, Ray
dispatch, or a FSDP worker-to-trainer pipeline. Only final RMSNorm is trainable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VERL_SHA = "87cd055e872f9fdd7b976fcd797a0934b6dba265"
LOSS_PATH = "verl/trainer/distillation/fsdp/losses.py"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value):
    import torch
    raw = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(raw.dtype), tuple(raw.shape))).encode())
    digest.update(raw.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_recorded_fixture(source):
    """Check the recorded input bytes before allocating a real model."""
    import torch

    recorded = source["inputs"]
    hashes = recorded["hashes"]
    for name in ("input_ids", "attention_mask", "position_ids"):
        value = torch.tensor(recorded[name], dtype=torch.int64)
        if tensor_hash(value) != hashes[name]:
            raise ValueError(f"recorded {name} tensor hash does not match fixture contents")
    response_mask = torch.tensor(recorded["response_logit_row_mask"], dtype=torch.bool)
    if tensor_hash(response_mask) != hashes["response_logit_row_mask"]:
        raise ValueError("recorded response mask hash does not match fixture contents")
    candidate_ids = torch.tensor(source["candidate_ids"], dtype=torch.int64)
    if tensor_hash(candidate_ids) != source["candidate_ids_sha256"]:
        raise ValueError("recorded candidate IDs hash does not match fixture contents")


def git_output(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def load_project_consumer():
    # Both repositories have a scripts directory; use an explicit file import
    # rather than allowing the upstream package to shadow this project's code.
    name = "_memory_bounded_project_consumer_check"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/teacher_consumer_step.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def load_upstream(directory, report):
    directory = directory.resolve()
    if git_output(directory, "rev-parse", "HEAD") != VERL_SHA:
        raise ValueError("verl checkout is not at the pinned commit")
    expected_blob = git_output(directory, "rev-parse", f"HEAD:{LOSS_PATH}")
    source_bytes = (directory / LOSS_PATH).read_bytes()
    observed_blob = hashlib.sha1(b"blob " + str(len(source_bytes)).encode() + b"\0" + source_bytes).hexdigest()
    if observed_blob != expected_blob:
        raise ValueError("the actual upstream loss source differs from its pinned Git blob")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(directory))
    module = importlib.import_module("verl.trainer.distillation.fsdp.losses")
    function = module.compute_reverse_kl_topk
    loaded_path = Path(inspect.getfile(function)).resolve()
    if loaded_path != (directory / LOSS_PATH).resolve():
        raise RuntimeError(f"upstream function loaded from a different source: {loaded_path}")
    report["upstream"] = {
        "commit": VERL_SHA, "path": LOSS_PATH, "git_blob": observed_blob,
        "source_sha256": file_hash(loaded_path),
        "function": f"{module.__name__}.compute_reverse_kl_topk",
        "loaded_path": str(loaded_path),
    }
    return function


def tiny_fixture(torch, model, dtype, device):
    tokens = torch.tensor([[2, 5, 7, 11], [0, 3, 9, 12]], device=device)
    attention = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]], device=device)
    positions = (attention.cumsum(-1) - 1).clamp_min(0)
    with torch.no_grad():
        logits = model(input_ids=tokens, attention_mask=attention,
                       position_ids=positions, use_cache=False).logits
        ids = logits.log_softmax(-1).topk(4, -1).indices
        teacher = (logits.float().gather(-1, ids)
                   - logits.float().logsumexp(-1, keepdim=True)).to(dtype)
        own_ids = logits.topk(4, -1).indices
    return {
        "arguments": {"dtype": str(dtype).split(".")[-1]},
        "inputs": {"input_ids": tokens.cpu().tolist(), "attention_mask": attention.cpu().tolist(),
                   "position_ids": positions.cpu().tolist(),
                   "response_logit_row_mask": [[False, True, True, False], [False, False, True, False]]},
        "candidate_ids": ids.cpu().tolist(),
        "consumer_semantics": {"student_temperature": 0.8,
                              "log_prob_min_clamp_after_support_normalization": -10,
                              "loss_wrapper_nonnegative_clamp": True, "loss_max_clamp": 10},
        "teacher": {name: {"candidate_logprobs": teacher.float().cpu().tolist(),
                           "teacher_topk_ids": own_ids.cpu().tolist()} for name in ("A0", "A1")},
    }


def evaluate(torch, model, inputs, ids, response_mask, scores, semantics, implementation, upstream, dtype):
    project = load_project_consumer()
    consumer = project.consumer

    model.zero_grad(set_to_none=True)
    logits = model(**inputs, use_cache=False).logits
    if implementation == "project":
        # The project function owns its own temperature scaling.
        if semantics["student_temperature"] != 0.8:
            raise ValueError("the project consumer currently has fixed student temperature 0.8")
        loss, diagnostics = consumer(logits, scores, ids, response_mask)
    else:
        temperature = torch.full(logits.shape[:-1], semantics["student_temperature"],
                                 dtype=torch.float32, device=logits.device)
        logits.div_(temperature.clamp_min(1e-8).unsqueeze(-1).to(logits.dtype))
        logits.retain_grad()
        config = SimpleNamespace(distillation_loss=SimpleNamespace(
            log_prob_min_clamp=semantics["log_prob_min_clamp_after_support_normalization"]))
        outputs = upstream(
            student_logits=logits,
            teacher_on_student_logp=scores.candidate_logprobs.to(logits.device).detach().float(),
            student_topk_ids=ids, config=config, data_format="bshd",
            teacher_topk_ids=scores.teacher_topk_ids.to(logits.device),
        )
        token_loss = outputs["distillation_losses"]
        # The real function returns pre-wrapper token losses. Apply the same
        # recorded wrapper and mask for the scalar/gradient comparison.
        wrapped = token_loss.clamp_min(0) if semantics["loss_wrapper_nonnegative_clamp"] else token_loss
        bound = semantics["loss_max_clamp"]
        if bound is not None:
            wrapped = wrapped.clamp(-bound, bound)
        loss = wrapped[response_mask].mean()
        diagnostics = {
            "student_mass": outputs["student_mass"], "teacher_mass": outputs["teacher_mass"],
            "overlap_count": outputs["overlap_count"],
            "overlap_token_advantage": outputs["overlap_token_advantage"],
            "token_loss_before_wrapper": token_loss, "masked_loss": loss,
        }
    norm_gradient, logits_gradient = torch.autograd.grad(loss, (model.model.norm.weight, logits))
    values = {
        "loss": loss.detach().cpu(), "norm_gradient": norm_gradient.detach().cpu().float(),
        "logits_gradient": logits_gradient.detach().cpu().float(),
        **{name: value.detach().cpu() for name, value in diagnostics.items()},
    }
    raw = {
        "implementation": implementation, "loss": loss.detach().item(),
        "norm_gradient": values["norm_gradient"].tolist(),
        "norm_gradient_l2": values["norm_gradient"].norm().item(),
        "norm_gradient_nonzero": values["norm_gradient"].count_nonzero().item(),
        "logits_gradient_elements": values["logits_gradient"].numel(),
        "hashes": {name: tensor_hash(value) for name, value in values.items()},
        "diagnostics": {name: value.tolist() for name, value in values.items()
                        if name not in {"norm_gradient", "logits_gradient"}},
    }
    del logits, norm_gradient, logits_gradient
    return values, raw


def compare_runs(torch, first, second, dtype):
    project = load_project_consumer()
    compare, compare_gradient, TOLERANCES = project.compare, project.compare_gradient, project.TOLERANCES
    atol, rtol = TOLERANCES[dtype]
    checks = {}
    for name in first:
        if name == "overlap_count":
            checks[name] = {"pass": torch.equal(first[name], second[name]), "kind": "exact_integer"}
        elif "gradient" in name:
            checks[name] = compare_gradient(first[name], second[name], dtype)
        else:
            checks[name] = compare(first[name], second[name], atol, rtol)
    return {"pass": all(check["pass"] for check in checks.values()), "checks": checks}


def run(args, report):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM
    from reference.memory_bounded_teacher import TeacherScores
    project = load_project_consumer()
    TOLERANCES, GRADIENT_RELATIVE_L2_MAX = project.TOLERANCES, project.GRADIENT_RELATIVE_L2_MAX

    upstream = load_upstream(args.verl_dir, report)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(731)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report["device"] = torch.cuda.get_device_name()
    source = None if args.tiny else json.loads(args.input.read_text(encoding="utf-8"))
    if source is not None:
        if source.get("status") != "PASS":
            raise ValueError("recorded teacher/student fixture was not a PASS result")
        validate_recorded_fixture(source)
    dtype_name = args.dtype or ("float32" if args.tiny else source["arguments"]["dtype"])
    dtype = getattr(torch, dtype_name)
    report["frozen_tolerances"] = {
        "atol": TOLERANCES[dtype_name][0], "rtol": TOLERANCES[dtype_name][1],
        "gradient_relative_l2_max": GRADIENT_RELATIVE_L2_MAX[dtype_name],
    }
    if args.tiny:
        config = Qwen2Config(vocab_size=97, hidden_size=24, intermediate_size=40,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             tie_word_embeddings=True, attention_dropout=0.0, max_position_embeddings=32)
        config._attn_implementation = "eager"
        model = Qwen2ForCausalLM(config).to(device=device, dtype=dtype).eval()
        source = tiny_fixture(torch, model, dtype, device)
        report["fixture_scope"] = "random tiny CPU helper; no real-model or teacher integration claim"
    else:
        directory = args.model_dir or Path(source["weight_source"]["directory"])
        directory = directory.resolve()
        for artifact in source["weight_source"]["files"]:
            if file_hash(directory / artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"model fixture hash mismatch: {artifact['path']}")
        report["model_files"] = source["weight_source"]["files"]
        report["model_directory"] = str(directory)
        report["input_evidence_sha256"] = file_hash(args.input)
        report["fixture_scope"] = "recorded real-model teacher fixture; direct upstream consumer function only"
        model = AutoModelForCausalLM.from_pretrained(
            directory, local_files_only=True, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    model.requires_grad_(False)
    model.model.norm.weight.requires_grad_(True)
    inputs = {name: torch.tensor(source["inputs"][name], dtype=torch.int64, device=device)
              for name in ("input_ids", "attention_mask", "position_ids")}
    response_mask = torch.tensor(source["inputs"]["response_logit_row_mask"], dtype=torch.bool, device=device)
    ids = torch.tensor(source["candidate_ids"], dtype=torch.int64, device=device)
    if tuple(ids.shape[:-1]) != tuple(inputs["input_ids"].shape) or response_mask.shape != inputs["input_ids"].shape:
        raise ValueError("recorded candidate/mask rows do not match model inputs")
    if not bool(response_mask.any()):
        raise ValueError("the recorded response mask contains no active rows")
    if bool((response_mask & ~inputs["attention_mask"].bool()).any()):
        raise ValueError("the recorded response mask includes padding rows")

    semantics = source["consumer_semantics"]
    required = {"student_temperature": 0.8, "log_prob_min_clamp_after_support_normalization": -10,
                "loss_wrapper_nonnegative_clamp": True, "loss_max_clamp": 10}
    if any(semantics.get(key) != value for key, value in required.items()):
        raise ValueError("recorded semantics differ from the project consumer's fixed contract")
    report["consumer_semantics"] = semantics
    report["inputs"] = {name: value.detach().cpu().tolist() for name, value in inputs.items()}
    report["inputs"]["response_logit_row_mask"] = response_mask.cpu().tolist()
    report["candidate_ids"] = ids.cpu().tolist()
    report["input_tensor_hashes"] = {name: tensor_hash(value) for name, value in inputs.items()}
    report["candidate_ids_sha256"] = tensor_hash(ids)
    report["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                          "python": sys.version, "cuda_build": torch.version.cuda}
    report["trainable_parameters"] = [name for name, value in model.named_parameters() if value.requires_grad]
    report["runs"] = {}
    tensors = {}
    for teacher_name in ("A0", "A1"):
        teacher = source["teacher"][teacher_name]
        scores = TeacherScores(
            torch.tensor(teacher["candidate_logprobs"], dtype=getattr(torch, source["arguments"]["dtype"])),
            torch.tensor(teacher["teacher_topk_ids"], dtype=torch.int64), 0)
        for implementation in ("project", "upstream"):
            key = f"{implementation}_{teacher_name}"
            tensors[key], report["runs"][key] = evaluate(
                torch, model, inputs, ids, response_mask, scores, semantics, implementation, upstream, dtype_name)
    report["comparisons"] = {
        "project_vs_upstream_A0": compare_runs(torch, tensors["project_A0"], tensors["upstream_A0"], dtype_name),
        "project_vs_upstream_A1": compare_runs(torch, tensors["project_A1"], tensors["upstream_A1"], dtype_name),
        "upstream_A0_vs_A1": compare_runs(torch, tensors["upstream_A0"], tensors["upstream_A1"], dtype_name),
    }
    report["upstream_nonzero_norm_gradient"] = all(
        report["runs"][f"upstream_{name}"]["norm_gradient_nonzero"] > 0 for name in ("A0", "A1"))
    report["status"] = "PASS" if all(value["pass"] for value in report["comparisons"].values()) and report["upstream_nonzero_norm_gradient"] else "FAIL"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="recorded consumer-real-*.json fixture")
    source.add_argument("--tiny", action="store_true", help="CPU-only helper check with a tiny random model")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"))
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.verl_dir = args.verl_dir.resolve()
    if args.out.is_relative_to(ROOT.resolve()):
        parser.error("--out must be outside the source repository")
    if args.out.exists():
        parser.error("--out already exists; preserve previous evidence")
    if args.tiny and args.device != "cpu":
        parser.error("--tiny is a CPU-only helper")
    sys.path.insert(0, str(ROOT))
    report = {
        "status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": __doc__, "argv": sys.argv,
        "sources": {name: file_hash(ROOT / name) for name in
                    ("scripts/verify_verl_consumer.py", "scripts/teacher_consumer_step.py",
                     "reference/memory_bounded_teacher.py")},
        "source_git_head": git_output(ROOT, "rev-parse", "HEAD"),
        "source_git_status": git_output(ROOT, "status", "--short"),
        "limitations": ["no FSDP worker-to-trainer execution", "no Ray dispatch or complete RL step",
                        "only student final RMSNorm is trainable; no optimizer update in this comparison"],
    }
    try:
        run(args, report)
    except Exception as error:
        report["status"] = "ERROR"
        report["exception"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                            encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(args.out)}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())