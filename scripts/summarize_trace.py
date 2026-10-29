#!/usr/bin/env python3
"""Summarize a SGLang torch-profiler trace without interpreting sums as latency."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace", help="SGLang *.trace.json.gz")
    p.add_argument("--out", required=True)
    p.add_argument("--vocab-size", type=int, required=True)
    args = p.parse_args()
    path = Path(args.trace)
    payload = path.read_bytes()
    trace = json.loads(gzip.decompress(payload))
    categories = Counter()
    durations = defaultdict(float)
    kernels = defaultdict(lambda: {"count": 0, "sum_duration_us": 0.0})
    for event in trace["traceEvents"]:
        cat = event.get("cat", "<none>")
        categories[cat] += 1
        if event.get("ph") != "X" or not isinstance(event.get("dur"), (int, float)):
            continue
        duration = event["dur"]
        durations[cat] += duration
        if cat == "kernel":
            kernel = kernels[event.get("name", "<unknown>")]
            kernel["count"] += 1
            kernel["sum_duration_us"] += duration
    top = sorted(({"name": name, **data} for name, data in kernels.items()),
                 key=lambda row: row["sum_duration_us"], reverse=True)[:20]
    events = trace["traceEvents"]
    # Match recorded aten::mm input shapes, then correlate by profiler External id.
    head_ops = []
    for event in events:
        args_dict = event.get("args", {})
        dims = args_dict.get("Input Dims", [])
        if (event.get("cat") != "cpu_op" or event.get("name") != "aten::mm" or
                len(dims) != 2 or len(dims[1]) != 2 or dims[1][-1] != args.vocab_size):
            continue
        external_id = args_dict.get("External id")
        matches = [x for x in events if x.get("cat") == "kernel" and
                   x.get("args", {}).get("External id") == external_id]
        head_ops.append({"input_dims": dims, "gpu_kernels": [
            {"name": x.get("name"), "duration_us": x.get("dur")} for x in matches
        ]})
    result = {
        "trace_sha256": hashlib.sha256(payload).hexdigest(),
        "trace_bytes": len(payload),
        "event_counts_by_category": dict(categories),
        "sum_event_duration_us_by_category": dict(durations),
        "top_gpu_kernel_names_by_summed_duration_us": top,
        "head_matmuls_by_vocab_shape": head_ops,
        "interpretation": "One synthetic profiling replay. Event durations may overlap; sums are not end-to-end latency or a benchmark.",
    }
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
