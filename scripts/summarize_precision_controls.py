#!/usr/bin/env python3
"""Assemble scored precision controls from their raw JSON records."""
import argparse
import hashlib
import json
from pathlib import Path

CASES = (
    ("bf16_triton_auto", "sglang-teacher-smoke-20261029.json", "teacher-hf-comparison-20261029.json"),
    ("bf16_torch_attention", "sglang-teacher-bf16-torch-attn-20261029.json", "teacher-hf-bf16-torch-attn-20261029.json"),
    ("bf16_torch_attention_and_fused_ops", "sglang-teacher-bf16-native-20261029.json", "teacher-hf-bf16-native-20261029.json"),
    ("fp32_default_fused_ops", "sglang-teacher-fp32-20261029.json", None),
    ("fp32_triton_default_precision", "sglang-teacher-fp32-native-20261029.json", "teacher-hf-fp32-native-20261029.json"),
    ("fp32_torch_attention", "sglang-teacher-fp32-torch-attn-20261029.json", "teacher-hf-fp32-torch-attn-20261029.json"),
    ("fp32_triton_ieee", "sglang-teacher-fp32-triton-ieee-20261029.json", "teacher-hf-fp32-triton-ieee-20261029.json"),
    ("fp32_triton_tf32", "sglang-teacher-fp32-triton-tf32-20261029.json", "teacher-hf-fp32-triton-tf32-20261029.json"),
)


def read_record(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    # Tracebacks may contain local workstation paths; retain them in private evidence.
    data.pop("traceback", None)
    return {"source_filename": path.name, "source_sha256": hashlib.sha256(raw).hexdigest(), "data": data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.input_dir)
    cases = []
    for name, smoke, comparison in CASES:
        row = {"name": name, "smoke": read_record(root / smoke)}
        if comparison is not None:
            row["comparison"] = read_record(root / comparison)
        cases.append(row)
    payload = {
        "schema_version": 1,
        "description": "Small real-model scoring controls, not a performance benchmark. Tracebacks omitted.",
        "source_note": "Each SHA-256 identifies the corresponding unredacted raw JSON.",
        "cases": cases,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Wrote {len(cases)} cases to {output}")


if __name__ == "__main__":
    main()
