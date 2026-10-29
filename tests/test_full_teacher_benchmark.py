import torch

from scripts.full_teacher_benchmark import ForwardFailure, is_out_of_memory, summarize


def test_oom_classification_survives_fsdp_stop_wrapper():
    gpu_oom = torch.cuda.OutOfMemoryError("allocation failed")
    stopped = RuntimeError("FSDP forward failed")
    stopped.__cause__ = gpu_oom
    assert is_out_of_memory(stopped)
    assert is_out_of_memory(ForwardFailure({"status": "OOM", "error": "allocation failed"}))
    assert not is_out_of_memory(RuntimeError("different failure"))

def test_summary_keeps_optional_raw_host_clock_separate():
    sample = {
        "status": "OK", "label": "A0", "wall_ms": 10.0, "raw_host_ms": 10.2,
        "cuda_event_ms": 10.1, "peak_allocated_bytes": 40,
        "incremental_peak_allocated_bytes": 20, "peak_reserved_bytes": 80,
    }
    summary = summarize([sample])["A0"]
    assert summary["metrics"]["wall_ms"]["p50"] == 10.0
    assert summary["metrics"]["raw_host_ms"]["p50"] == 10.2
    assert summary["cuda_event_exceeds_host_samples"] == 1
    sample["raw_host_ms"] = None
    assert "raw_host_ms" not in summarize([sample])["A0"]["metrics"]