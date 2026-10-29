#!/usr/bin/env python3
"""Bounded HF stage diagnosis on the exact saved G2 fixture; no service launch.

This collects HF intermediates only. Saved SGLang responses contain no hidden
states/logits and cannot establish the first HF/SGLang divergent model stage.
Frozen full-logprob budgets are reported, never used to change acceptance.
"""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

import torch
import torch.nn.functional as F

BUDGETS = {"bfloat16": (0.1, 0.01), "float32": (5e-5, 2e-5)}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    """Encode nonfinite values explicitly; raw tensors remain lossless in .pt."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": "NaN" if math.isnan(value) else "+Inf" if value > 0 else "-Inf"}
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def validate_capture_manifest(manifest, tensor_path, smoke_path, model_dir):
    """Bind a raw HF tensor to the exact fixture and currently loaded weights."""
    smoke = json.loads(Path(smoke_path).read_text())
    if manifest["status"] != "MEASURED":
        raise ValueError("HF manifest was not measured")
    if manifest["tensor_sha256"] != digest(tensor_path):
        raise ValueError("HF tensor SHA differs from its manifest")
    for key in ("model", "model_revision", "tokenizer_revision", "request"):
        if manifest[key] != smoke[key]:
            raise ValueError(f"HF manifest fixture differs at {key}")
    if manifest["predictor_positions"] != smoke["predictor_last_prefix_indices"]:
        raise ValueError("HF manifest predictor positions differ")
    if digest(smoke_path) not in [item["sha256"] for item in manifest["inputs"]]:
        raise ValueError("saved smoke hash is not bound in HF manifest")
    model_dir = Path(model_dir)
    if manifest["config_sha256"] != digest(model_dir / "config.json"):
        raise ValueError("HF model config hash differs")
    weights = [{"file": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
               for path in sorted(model_dir.glob("*.safetensors"))]
    if not weights or manifest["weight_files"] != weights:
        raise ValueError("HF model weight hashes differ")


def difference(actual, reference):
    """Exact differences and nonfinite classifications, without a new budget."""
    a, b = actual.detach().cpu().double(), reference.detach().cpu().double()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")
    finite = torch.isfinite(a) & torch.isfinite(b)
    mismatch = ((torch.isnan(a) != torch.isnan(b)) |
                (torch.isposinf(a) != torch.isposinf(b)) |
                (torch.isneginf(a) != torch.isneginf(b)))
    delta = (a[finite] - b[finite]).abs()
    unequal = finite & (a != b)
    locations = (unequal | mismatch).nonzero()
    return {"shape": list(a.shape), "elements": a.numel(),
            "actual_nonfinite": {"nan": int(torch.isnan(a).sum()), "posinf": int(torch.isposinf(a).sum()),
                                 "neginf": int(torch.isneginf(a).sum())},
            "reference_nonfinite": {"nan": int(torch.isnan(b).sum()), "posinf": int(torch.isposinf(b).sum()),
                                    "neginf": int(torch.isneginf(b).sum())},
            "finite_pairs": int(finite.sum()), "nonfinite_class_mismatches": int(mismatch.sum()),
            "unequal_finite_elements": int(unequal.sum()),
            "first_difference_index": locations[0].tolist() if locations.numel() else None,
            "max_abs_error": float(delta.max()) if delta.numel() else None,
            "mean_abs_error": float(delta.mean()) if delta.numel() else None}


def frozen_comparison(actual, reference, dtype):
    atol, rtol = BUDGETS[dtype]
    a, b = torch.as_tensor(actual, dtype=torch.float32), torch.as_tensor(reference, dtype=torch.float32)
    result = difference(a, b)
    within = torch.isclose(a, b, atol=atol, rtol=rtol)
    result.update({"atol": atol, "rtol": rtol, "elementwise_within_budget": within.tolist(),
                   "failed_elements": int((~within).sum())})
    return result


def select_predictors(values, lengths):
    """Select each row's last real prefix token, including right-padded batches."""
    if values.ndim != 3 or values.shape[0] != len(lengths):
        raise ValueError("expected [batch, sequence, feature] with one length per row")
    if any(n < 1 or n > values.shape[1] for n in lengths):
        raise ValueError("prefix lengths must index real sequence positions")
    return torch.stack([values[r, n - 1] for r, n in enumerate(lengths)])


def readout(logits, candidates):
    """Freeze logits; contrast FP32 log_softmax with FP64 CPU normalization."""
    z = logits.float()
    ids = torch.as_tensor(candidates, dtype=torch.long, device=z.device)
    if ids.ndim != 2 or ids.shape[0] != z.shape[0] or ids.numel() == 0:
        raise ValueError("expected a nonempty rectangular candidate matrix")
    if bool(((ids < 0) | (ids >= z.shape[1])).any()):
        raise ValueError("candidate outside vocabulary")
    logp = torch.log_softmax(z, dim=-1)
    selected_logits = z.gather(1, ids)
    selected_logp = logp.gather(1, ids)
    lse = torch.logsumexp(z, dim=-1)
    explicit = selected_logits - lse[:, None]
    scalar = torch.stack([torch.stack([logp[r, int(j)] for j in row])
                          for r, row in enumerate(candidates)])
    double_z = z.detach().cpu().double()
    double_lse = torch.logsumexp(double_z, dim=-1)
    independent = double_z.gather(1, ids.cpu()) - double_lse[:, None]
    return {"selected_logits": selected_logits.cpu(), "logsumexp": lse.cpu(),
            "selected_logprobs": selected_logp.cpu(),
            "fp64_cpu_logsumexp": double_lse, "fp64_cpu_selected_logprobs": independent,
            "log_softmax_vs_explicit_lse": difference(selected_logp, explicit),
            "fp32_vs_fp64_cpu_normalization": difference(selected_logp, independent),
            "gather_vs_scalar_index": difference(selected_logp, scalar)}


def validate_fixture(smoke, other):
    for item in (smoke, other):
        if item["status"] != "PASS":
            raise ValueError("smoke fixture was not recorded PASS")
        rows, candidates = item["request"]["items"], item["request"]["label_token_ids"]
        if not rows or any(not row for row in rows) or len(rows) != len(candidates):
            raise ValueError("empty or misaligned fixture")
        if len({len(row) for row in candidates}) != 1 or not candidates[0]:
            raise ValueError("candidates must be nonempty rectangular rows")
        if item["predictor_last_prefix_indices"] != [len(row) - 1 for row in rows]:
            raise ValueError("saved predictor positions disagree with prefix lengths")
        if item["request"].get("temperature") != 1.0:
            raise ValueError("this diagnosis requires the saved temperature 1.0")
    for key in ("model", "model_revision", "tokenizer_revision", "request"):
        if smoke[key] != other[key]:
            raise ValueError(f"BF16/FP32 fixtures differ at {key}")


def capture(model, rows, candidates, pad_id, mode):
    groups = [rows] if mode == "batch" else [[row] for row in rows]
    layers, native_logits = [], []
    for group in groups:
        lengths = list(map(len, group))
        max_len = max(lengths)
        ids = torch.tensor([row + [pad_id] * (max_len - len(row)) for row in group],
                           dtype=torch.long, device="cuda")
        kwargs = {"input_ids": ids, "output_hidden_states": True}
        # Match the saved HF verifier: mask supplied only in the padded batch.
        if mode == "batch":
            kwargs["attention_mask"] = torch.tensor(
                [[1] * n + [0] * (max_len - n) for n in lengths], device="cuda")
        output = model(**kwargs)
        layers.append(torch.stack([select_predictors(h, lengths) for h in output.hidden_states]).cpu())
        native_logits.append(select_predictors(output.logits, lengths).cpu())
        del output
    hidden = torch.cat(layers, dim=1)
    logits = torch.cat(native_logits, dim=0)
    final = hidden[-1].to("cuda")
    weight = model.get_output_embeddings().weight
    bias = model.get_output_embeddings().bias
    # Fix the same final hidden/weights and alter only head arithmetic/shape.
    head_native = F.linear(final, weight, bias).cpu()
    head_single = torch.cat([F.linear(row[None], weight, bias) for row in final]).cpu()
    head_fp32 = F.linear(final.float(), weight.float(), None if bias is None else bias.float()).cpu()
    stages = {name: readout(z.to("cuda"), candidates) for name, z in
              (("forward", logits), ("selected_hidden_native_head", head_native),
               ("selected_hidden_fp32_head", head_fp32), ("selected_hidden_single_row_head", head_single))}
    return {"hidden_states": hidden, "forward_logits": logits,
            "selected_hidden_native_head_logits": head_native,
            "selected_hidden_fp32_head_logits": head_fp32, "readouts": stages,
            "selected_hidden_single_row_head_logits": head_single,
            "batch_vs_single_row_head_same_hidden": difference(head_native, head_single),
            "forward_vs_selected_native_head": difference(logits, head_native),
            "native_vs_fp32_head_same_hidden": difference(head_native, head_fp32)}


def summarize(capture_result):
    return {"hidden_shape": list(capture_result["hidden_states"].shape),
            "hidden_dtype": str(capture_result["hidden_states"].dtype),
            "forward_logits_dtype": str(capture_result["forward_logits"].dtype),
            "forward_vs_selected_native_head": capture_result["forward_vs_selected_native_head"],
            "native_vs_fp32_head_same_hidden": capture_result["native_vs_fp32_head_same_hidden"],
            "batch_vs_single_row_head_same_hidden": capture_result["batch_vs_single_row_head_same_hidden"],
            "readouts": {name: {k: v.tolist() if isinstance(v, torch.Tensor) else v
                                 for k, v in data.items()}
                         for name, data in capture_result["readouts"].items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--smoke", default="results/sglang-teacher-smoke-20261029.json")
    p.add_argument("--controls", default="results/precision-controls-20261029.json")
    p.add_argument("--fp32-smoke", help="optional raw override; default uses tracked controls fp32_triton_ieee.smoke.data")
    p.add_argument("--bf16-reference", default="results/teacher-hf-comparison-20261029.json")
    p.add_argument("--fp32-reference", help="optional raw override; default uses tracked controls fp32_triton_ieee.comparison.data")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "NOT_RUN", "decision": "INVESTIGATE",
              "purpose": "HF stage attribution; frozen acceptance is unchanged",
              "nonfinite_encoding": "JSON uses {nonfinite_float: NaN/+Inf/-Inf}; .pt retains raw tensors",
              "command": [sys.executable, *sys.argv], "frozen_budgets": BUDGETS,
              "limitations": ["No SGLang intermediate capture; cross-runtime first divergence UNKNOWN",
                              "Synthetic 8-prefix fixture, one GPU, eager HF attention only",
                              "HF batch-vs-single changes padding, explicit mask and shape together to match saved runs",
                              "FP32-head intervention promotes the same dtype-loaded weights; not a new reference",
                              "Hidden tuple final entry is post-final-norm; earlier entries follow Transformers semantics"]}
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers
        paths = [args.smoke, args.bf16_reference, args.controls]
        paths.extend(path for path in (args.fp32_smoke, args.fp32_reference) if path)
        smoke = json.loads(Path(args.smoke).read_text())
        bf16_reference = json.loads(Path(args.bf16_reference).read_text())
        controls = json.loads(Path(args.controls).read_text())
        ieee = next(case for case in controls["cases"] if case["name"] == "fp32_triton_ieee")
        fp32_smoke = (json.loads(Path(args.fp32_smoke).read_text()) if args.fp32_smoke else ieee["smoke"]["data"])
        fp32_reference = (json.loads(Path(args.fp32_reference).read_text()) if args.fp32_reference
                          else ieee["comparison"]["data"])
        report["fp32_evidence_source"] = {
            "smoke": args.fp32_smoke or f"{args.controls}#cases[name=fp32_triton_ieee].smoke.data",
            "reference": args.fp32_reference or f"{args.controls}#cases[name=fp32_triton_ieee].comparison.data"}
        validate_fixture(smoke, fp32_smoke)
        for fixture, archived, dtype in ((smoke, bf16_reference, "bfloat16"),
                                          (fp32_smoke, fp32_reference, "float32")):
            command = fixture["server_command"]
            if command[command.index("--dtype") + 1] != dtype:
                raise ValueError("saved server dtype does not match diagnostic phase")
            if archived["model_revision"] != fixture["model_revision"]:
                raise ValueError("archive model revision differs")
            if archived["tolerance"] != dict(zip(("atol", "rtol"), BUDGETS[dtype])):
                raise ValueError("archive tolerance differs from frozen budget")
        report.update({"model": smoke["model"], "model_revision": smoke["model_revision"],
                       "tokenizer_revision": smoke["tokenizer_revision"],
                       "sglang_revision": smoke["sglang_commit"], "request": smoke["request"],
                       "predictor_positions": smoke["predictor_last_prefix_indices"],
                       "inputs": [{"path": str(Path(x).resolve()), "sha256": digest(x)} for x in paths],
                       "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                       "script_sha256": digest(__file__), "torch": torch.__version__,
                       "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda})
        model_dir = Path(args.model_dir)
        report["config_sha256"] = digest(model_dir / "config.json")
        report["weight_files"] = [{"file": x.name, "bytes": x.stat().st_size, "sha256": digest(x)}
                                  for x in sorted(model_dir.glob("*.safetensors"))]
        if not report["weight_files"]:
            raise ValueError("no safetensors weight files found")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for full fixture diagnosis; CPU unit tests are separate")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report["precision_controls"] = {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                        "float32_matmul_precision": torch.get_float32_matmul_precision()}
        report["gpu"] = torch.cuda.get_device_name()
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        rows, candidates = smoke["request"]["items"], smoke["request"]["label_token_ids"]
        raw = {}
        report["cases"] = {}
        with torch.inference_mode():
            for dtype, archived, served in (("bfloat16", bf16_reference, smoke),
                                            ("float32", fp32_reference, fp32_smoke)):
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_dir, dtype=getattr(torch, dtype), attn_implementation="eager",
                    local_files_only=True).to("cuda").eval()
                report["model_source"] = inspect.getfile(type(model))
                cases = {mode: capture(model, rows, candidates, tokenizer.pad_token_id, mode)
                         for mode in ("batch", "single")}
                raw[dtype] = cases
                batch, single = cases["batch"], cases["single"]
                summary = {mode: summarize(data) for mode, data in cases.items()}
                summary["batch_vs_single_hidden_by_tuple_index"] = [
                    difference(a, b) for a, b in zip(batch["hidden_states"], single["hidden_states"], strict=True)]
                summary["batch_vs_single_forward_logits"] = difference(batch["forward_logits"], single["forward_logits"])
                summary["batch_vs_single_readout_stages"] = {
                    key: difference(batch["readouts"]["forward"][key], single["readouts"]["forward"][key])
                    for key in ("selected_logits", "logsumexp", "selected_logprobs")}
                for mode, data in cases.items():
                    values = data["readouts"]["forward"]["selected_logprobs"]
                    summary[mode]["served_vs_hf_frozen_budget"] = frozen_comparison(
                        served["response"]["token_logprobs"], values, dtype)
                    summary[mode]["archived_hf_batch_vs_current_frozen_budget"] = frozen_comparison(
                        values, archived["reference_full_logprobs"], dtype)
                report["cases"][dtype] = summary
                del model
                torch.cuda.empty_cache()
        report["bf16_vs_fp32_by_mode"] = {
            mode: {"hidden_by_tuple_index": [
                difference(a, b) for a, b in zip(raw["bfloat16"][mode]["hidden_states"],
                                                raw["float32"][mode]["hidden_states"], strict=True)],
                   "selected_hidden_fp32_head_logits": difference(
                       raw["bfloat16"][mode]["selected_hidden_fp32_head_logits"],
                       raw["float32"][mode]["selected_hidden_fp32_head_logits"])}
            for mode in ("batch", "single")}
        tensor_path = out.with_suffix(".pt")
        torch.save(raw, tensor_path)
        report.update({"status": "MEASURED", "tensor_file": str(tensor_path.resolve()),
                       "tensor_sha256": digest(tensor_path),
                       "max_allocated_bytes": torch.cuda.max_memory_allocated()})
    except Exception as exc:
        report.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()})
    out.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "out": str(out), "error": report.get("error")}))
    return 0 if report["status"] == "MEASURED" else 1


if __name__ == "__main__":
    sys.exit(main())
