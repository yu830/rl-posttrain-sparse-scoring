#!/usr/bin/env python3
"""Diagnose the BF16 full-model discrepancy without changing acceptance budgets."""
import argparse
import json
from pathlib import Path
import sys
import traceback

import torch
from transformers import AutoModelForCausalLM


def forward_rows(model_dir, rows, candidates, dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=dtype, attn_implementation="eager", local_files_only=True
    ).to("cuda").eval()
    values = []
    with torch.inference_mode():
        for row, cand in zip(rows, candidates, strict=True):
            tokens = torch.tensor([row], dtype=torch.long, device="cuda")
            logits = model(input_ids=tokens).logits[0, -1].float()
            logp = torch.log_softmax(logits, dim=-1)
            values.append(logp[torch.tensor(cand, device="cuda")].cpu().tolist())
    del model
    torch.cuda.empty_cache()
    return values


def compare(a, b):
    delta = [abs(x - y) for ra, rb in zip(a, b, strict=True) for x, y in zip(ra, rb, strict=True)]
    return {"max_abs_error": max(delta), "mean_abs_error": sum(delta) / len(delta)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--smoke", required=True)
    p.add_argument("--hf-batched", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    smoke = json.loads(Path(args.smoke).read_text())
    batch = json.loads(Path(args.hf_batched).read_text())
    result = {"status": "NOT_RUN", "purpose": "diagnostic only; no relaxed numerical acceptance",
              "model": smoke["model"], "model_revision": smoke["model_revision"],
              "model_storage": "same fixed weights loaded in BF16 or FP32",
              "attention": "Transformers eager", "tf32": False}
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        rows = smoke["request"]["items"]
        candidates = smoke["request"]["label_token_ids"]
        served = smoke["response"]["token_logprobs"]
        bf16 = forward_rows(args.model_dir, rows, candidates, torch.bfloat16)
        fp32 = forward_rows(args.model_dir, rows, candidates, torch.float32)
        result.update({
            "hf_bf16_single_vs_hf_bf16_batch": compare(bf16, batch["reference_full_logprobs"]),
            "sglang_bf16_vs_hf_bf16_single": compare(served, bf16),
            "sglang_bf16_vs_hf_fp32_single": compare(served, fp32),
            "hf_bf16_single_vs_hf_fp32_single": compare(bf16, fp32),
            "hf_bf16_single_logprobs": bf16,
            "hf_fp32_single_logprobs": fp32,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "status": "MEASURED",
        })
    except Exception as exc:
        result.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()})
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "MEASURED" else 1


if __name__ == "__main__":
    sys.exit(main())
