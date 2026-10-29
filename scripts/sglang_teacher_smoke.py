#!/usr/bin/env python3
"""Run one real local SGLang teacher-scoring smoke request, not a benchmark.

The response prefixes and candidate IDs are deterministic synthetic fixtures;
they are not a captured student-top-k trace or a verified training consumer.
The script launches and terminates only its own loopback server process group.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

import requests
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SGLANG_REVISION = "46caa54562a97e71d9b937c62875c90b39274b0d"
PROMPT = "Explain why reproducible experiments need fixed model versions."
RESPONSE = " Fixed versions make comparisons reliable and repeatable."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="local snapshot at the pinned model revision")
    p.add_argument("--out", required=True)
    p.add_argument("--server-log", required=True)
    p.add_argument("--startup-timeout", type=float, default=420)
    p.add_argument("--profile-dir", help="optional separate SGLang torch-profiler trace directory")
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    response_ids = tokenizer.encode(RESPONSE, add_special_tokens=False)[:8]
    assert prompt_ids and len(response_ids) >= 3
    vocab = tokenizer.vocab_size
    candidates = []
    for position, target in enumerate(response_ids):
        # Same-target deduplication is explicit. These IDs are synthetic.
        row = list(dict.fromkeys([target, (target + 7 + position) % vocab,
                                  (target + 101 + position) % vocab, (target + 1009 + position) % vocab]))
        assert len(row) == 4
        candidates.append(row)
    request = {
        "model": MODEL_ID,
        "query": [],
        "items": [prompt_ids + response_ids[:i] for i in range(len(response_ids))],
        "label_token_ids": candidates,
        "apply_softmax": True,
        "return_token_logprobs": True,
        "temperature": 1.0,
    }
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", args.model_dir,
           "--served-model-name", MODEL_ID, "--host", "127.0.0.1", "--port", str(port),
           "--dtype", "bfloat16", "--context-length", "2048", "--mem-fraction-static", "0.45",
           "--max-running-requests", "4", "--max-total-tokens", "2048",
           "--chunked-prefill-size", "512", "--disable-cuda-graph", "--attention-backend", "triton"]
    safe_cmd = ["<PYTHON>" if x == sys.executable else "<MODEL_DIR>" if x == args.model_dir else x for x in cmd]
    result = {"status": "NOT_RUN", "model": MODEL_ID, "model_revision": MODEL_REVISION,
              "tokenizer_revision": MODEL_REVISION, "sglang_commit": SGLANG_REVISION,
              "workload_kind": "synthetic_prefixes_and_candidates_not_student_topk",
              "server_command": safe_cmd, "request": request,
              "response_token_ids": response_ids, "predictor_last_prefix_indices": [len(x) - 1 for x in request["items"]],
              "timing_scope": "one cold/warmup-affected loopback HTTP request; not a performance benchmark"}
    process = None
    try:
        with open(args.server_log, "w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + args.startup_timeout
            url = f"http://127.0.0.1:{port}"
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"SGLang exited before readiness with code {process.returncode}")
                try:
                    if requests.get(url + "/health", timeout=2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("SGLang server did not become ready")
            t0 = time.perf_counter()
            response = requests.post(url + "/v1/score", json=request, timeout=180)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            result.update({"http_status": response.status_code, "response_bytes": len(response.content),
                           "request_bytes": len(response.request.body), "elapsed_ms": elapsed_ms})
            response.raise_for_status()
            payload = response.json()
            result["response"] = payload
            scores, logprobs = payload["scores"], payload["token_logprobs"]
            if len(scores) != len(response_ids) or len(logprobs) != len(response_ids):
                raise AssertionError("number of scored rows differs from number of response positions")
            errors = []
            for row, probs, full_logp in zip(candidates, scores, logprobs, strict=True):
                if len(row) != len(probs) or len(row) != len(full_logp):
                    raise AssertionError("candidate, score, and logprob lengths differ")
                m = max(full_logp)
                exps = [math.exp(x - m) for x in full_logp]
                expected = [x / sum(exps) for x in exps]
                errors.extend(abs(a - b) for a, b in zip(probs, expected, strict=True))
                if not all(math.isfinite(x) for x in full_logp + probs):
                    raise AssertionError("non-finite scoring result")
            result["max_abs_conditional_vs_full_logprob_renormalized"] = max(errors)
            if max(errors) > 1e-5:
                raise AssertionError("conditional scores differ from renormalized full logprobs")
            result["status"] = "PASS"
            if args.profile_dir:
                profile_dir = Path(args.profile_dir)
                profile_dir.mkdir(parents=True, exist_ok=True)
                requests.post(url + "/flush_cache", timeout=60).raise_for_status()
                config = {"output_dir": str(profile_dir), "activities": ["CPU", "GPU"],
                          "with_stack": False, "record_shapes": True, "profile_id": "g2-smoke"}
                requests.post(url + "/start_profile", json=config, timeout=60).raise_for_status()
                profiled_response = requests.post(url + "/v1/score", json=request, timeout=180)
                profiled_response.raise_for_status()
                requests.post(url + "/stop_profile", timeout=180).raise_for_status()
                traces = sorted(profile_dir.glob("*.trace.json.gz"))
                if not traces:
                    raise RuntimeError("profiler returned but did not produce a trace")
                result["profile"] = {"kind": "separate replay, torch profiler, CPU and CUDA activities",
                                     "cache": "radix cache flushed immediately before profiling",
                                     "trace_filenames": [x.name for x in traces],
                                     "trace_bytes": [x.stat().st_size for x in traces],
                                     "profiled_http_status": profiled_response.status_code}
    except Exception as exc:
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
