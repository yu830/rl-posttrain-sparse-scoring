#!/usr/bin/env python3
"""Compare the SGLang smoke response with an independent Hugging Face forward."""
import argparse
import json
from pathlib import Path
import sys
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--smoke", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    smoke = json.loads(Path(args.smoke).read_text())
    result = {"status": "NOT_RUN", "reference": "Transformers eager attention, BF16 model and FP32 log_softmax",
              "model": smoke["model"], "model_revision": smoke["model_revision"],
              "tokenizer_revision": smoke["tokenizer_revision"],
              "tolerance": {"atol": 0.1, "rtol": 0.01}, "tf32": False}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        assert smoke["status"] == "PASS"
        assert torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
        ).to("cuda").eval()
        rows = smoke["request"]["items"]
        candidates = smoke["request"]["label_token_ids"]
        max_len = max(map(len, rows))
        pad_id = tokenizer.pad_token_id
        ids = torch.tensor([row + [pad_id] * (max_len - len(row)) for row in rows], device="cuda")
        mask = torch.tensor([[1] * len(row) + [0] * (max_len - len(row)) for row in rows], device="cuda")
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask).logits
            errors = []
            reference_rows = []
            comparisons = []
            for r, (row, cand, served) in enumerate(zip(rows, candidates, smoke["response"]["token_logprobs"], strict=True)):
                # Last prefix position predicts the next response token.
                logp = torch.log_softmax(logits[r, len(row) - 1].float(), dim=-1)
                reference = logp[torch.tensor(cand, device="cuda")].cpu()
                actual = torch.tensor(served, dtype=torch.float32)
                errors.extend((actual - reference).abs().tolist())
                reference_rows.append(reference.tolist())
                comparisons.append(torch.isclose(actual, reference, atol=0.1, rtol=0.01).tolist())
            result.update({"rows": len(rows), "candidate_count": sum(map(len, candidates)),
                           "max_abs_error": max(errors), "mean_abs_error": sum(errors) / len(errors),
                           "reference_full_logprobs": reference_rows, "elementwise_within_budget": comparisons,
                           "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                           "max_allocated_bytes": torch.cuda.max_memory_allocated()})
            if not all(all(row) for row in comparisons):
                raise AssertionError("one or more full logprob values exceed the preregistered BF16 budget")
            result["status"] = "PASS"
    except Exception as exc:
        result.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()})
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
