#!/usr/bin/env python3
"""Save small deterministic CPU/GPU numerical observations; not a benchmark."""
import argparse
import json
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.scoring import conditional_reverse_kl, dense_selected, eager_selected

SEED = 20261029


def maxerror(a, b):
    return float((a.detach().double() - b.detach().double()).abs().max().cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    gen = torch.Generator().manual_seed(SEED)
    base_h = torch.randn(5, 23, generator=gen, dtype=torch.float64)
    base_w = torch.randn(31, 23, generator=gen, dtype=torch.float64)
    base_b = torch.randn(31, generator=gen, dtype=torch.float64)
    ids = torch.stack([torch.randperm(31, generator=gen)[:7] for _ in range(5)])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    teacher = torch.randn(5, 7, generator=gen, dtype=torch.float64)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    records = []
    for device in devices:
        for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
            h, w, b = (x.to(device=device, dtype=dtype) for x in (base_h, base_w, base_b))
            d_ids, d_mask = ids.to(device), mask.to(device)
            # Compare on the same quantized inputs; report quantization separately.
            oracle = dense_selected(h.double(), w.double(), d_ids, d_mask, b.double())
            unquantized = dense_selected(base_h.to(device), base_w.to(device), d_ids, d_mask, base_b.to(device))
            dense = dense_selected(h, w, d_ids, d_mask, b)
            eager = eager_selected(h, w, d_ids, d_mask, b)
            s0 = oracle.detach().clone().requires_grad_(True)
            t0 = teacher.to(device).requires_grad_(True)
            l0 = conditional_reverse_kl(s0, t0, d_mask)
            g0 = torch.autograd.grad(l0.sum(), s0)[0]
            s1 = eager.detach().clone().requires_grad_(True)
            t1 = teacher.to(device=device, dtype=dtype).requires_grad_(True)
            loss = conditional_reverse_kl(s1, t1, d_mask)
            grad, teacher_grad = torch.autograd.grad(loss.sum(), (s1, t1), allow_unused=True)
            records.append({
                "device": device, "dtype": str(dtype), "rows": 5, "hidden": 23, "vocab": 31,
                "k_max": 7, "active_counts": mask.sum(-1).tolist(),
                "dense_vs_quantized_input_fp64_max_abs": maxerror(dense, oracle),
                "eager_vs_quantized_input_fp64_max_abs": maxerror(eager, oracle),
                "input_quantization_selected_max_abs": maxerror(oracle, unquantized),
                "loss_max_abs_including_teacher_input_quantization": maxerror(loss, l0),
                "student_gradient_max_abs_including_teacher_input_quantization": maxerror(grad, g0),
                "teacher_gradient_none": teacher_grad is None,
            })
    result = {"seed": SEED, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "python": platform.python_version(), "tf32_matmul": False, "autocast": False,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "purpose": "numerical observation only, not a performance benchmark", "records": records}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
