#!/usr/bin/env python3
"""Compare saved CPU compact outputs from independent capacity probes.

This accepts only tensor files written by factorial_teacher_probe and uses
weights_only loading. A top-K set difference is reported as an unresolved tie
or mismatch because compact outputs alone do not contain the full reference
logits needed to prove an exact boundary tie.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
TOLERANCES = {"bfloat16": (0.1, 0.01), "float32": (5e-5, 2e-5)}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    header = json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(header + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def validate_artifact(artifact):
    """Bind compact tensors to the declared fixture and producer signatures."""
    identity, output = artifact["identity"], artifact["output"]
    fixture = identity["fixture"]
    lengths, width = fixture["lengths"], fixture["k"]
    if (not isinstance(lengths, list) or not lengths
            or any(type(n) is not int or n <= 0 for n in lengths)
            or type(width) is not int or width <= 0):
        raise ValueError("invalid fixture lengths or candidate width")
    rows = sum(lengths)
    if fixture["valid_rows"] != rows or fixture["padded_rows"] != len(lengths) * max(lengths):
        raise ValueError("fixture row counts are inconsistent with lengths")
    if output["values_shape"] != [rows, width] or output["padded_shape"] != [len(lengths), max(lengths), width]:
        raise ValueError("recorded output shapes do not match fixture")
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    if output["offsets"] != offsets:
        raise ValueError("recorded output offsets do not match fixture")
    for key in ("candidate_logprobs", "teacher_topk_ids"):
        value = artifact[key]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or list(value.shape) != [rows, width]:
            raise ValueError(f"{key} shape/device does not match fixture")
        if output[key]["dtype"] != str(value.dtype) or output[key]["sha256"] != tensor_hash(value):
            raise ValueError(f"{key} bytes do not match producer signature")


def compare_artifacts(dense, candidate):
    validate_artifact(dense)
    validate_artifact(candidate)
    if dense["identity"] != candidate["identity"]:
        raise ValueError("artifacts describe different model, input, dtype or attention")
    dtype = dense["identity"]["dtype"]
    if dtype not in TOLERANCES:
        raise ValueError("unsupported dtype")
    left, right = dense["candidate_logprobs"], candidate["candidate_logprobs"]
    left_ids, right_ids = dense["teacher_topk_ids"], candidate["teacher_topk_ids"]
    if (left.device.type != "cpu" or right.device.type != "cpu"
            or left.shape != right.shape or left_ids.shape != right_ids.shape
            or left.shape != left_ids.shape or left_ids.dtype != torch.int64
            or right_ids.dtype != torch.int64 or str(left.dtype) != "torch." + dtype
            or left.dtype != right.dtype):
        raise ValueError("compact tensors have incompatible CPU shapes or dtypes")
    atol, rtol = TOLERANCES[dtype]
    finite = torch.isfinite(left) & torch.isfinite(right)
    close = finite & torch.isclose(left.float(), right.float(), atol=atol, rtol=rtol)
    delta = (left.float() - right.float()).abs()
    equal_rows = (left_ids.sort(-1).values == right_ids.sort(-1).values).all(-1)
    logprob_ok = bool(close.all())
    topk_ok = bool(equal_rows.all())
    return {
        "status": "PASS" if logprob_ok and topk_ok else "FAIL_OR_TIE_UNKNOWN",
        "scope": "all saved compact rows and candidates; exact top-K sets; no full-logits tie oracle",
        "rows": left.shape[0], "candidate_width": left.shape[-1],
        "candidate_logprobs": {"pass": logprob_ok, "atol": atol, "rtol": rtol,
                               "failed_values": int((~close).sum()),
                               "nonfinite_values": int((~finite).sum()),
                               "max_abs": float(delta[finite].max()) if bool(finite.any()) else None,
                               "exact_values": int((left == right).sum()), "total_values": left.numel()},
        "teacher_topk": {"exact_set_equal": topk_ok, "equal_rows": int(equal_rows.sum()),
                         "different_rows": int((~equal_rows).sum()), "total_rows": equal_rows.numel(),
                         "first_different_rows": (~equal_rows).nonzero().flatten()[:64].tolist(),
                         "tie_status_if_different": "UNKNOWN_without_full_reference_logits"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.is_relative_to(ROOT.resolve()) or args.out.exists():
        parser.error("out must be a new path outside the source repository")
    result = {"status": "RUNNING", "argv": sys.argv,
              "dense_sha256": sha256(args.dense), "candidate_sha256": sha256(args.candidate),
              "source_sha256": sha256(__file__)}
    try:
        dense = torch.load(args.dense, map_location="cpu", weights_only=True)
        candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
        result.update(compare_artifacts(dense, candidate))
    except Exception as error:
        result.update(status="ERROR", error={"type": type(error).__name__, "message": str(error),
                                             "traceback": traceback.format_exc()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "out": str(args.out)}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())