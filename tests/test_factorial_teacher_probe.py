"""CPU contract checks for the one-condition probe; no GPU work is run here."""
import json
import subprocess
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy

from scripts import factorial_teacher_probe as probe


def test_journal_is_readable_before_close_and_preserves_nonfinite_failure(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = probe.JsonlJournal(path)
    journal.record("call_complete", output_hash="abc")
    journal.record("error", diagnostic=float("nan"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["sequence"] for row in rows] == [0, 1]
    assert rows[0]["event"] == "call_complete"
    assert rows[1]["diagnostic"] == "nan"
    journal.close()
    with pytest.raises(FileExistsError):
        probe.JsonlJournal(path)


def fixture_output():
    mask = torch.tensor([[True, True, True], [True, True, False]])
    logp = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / -10
    topk = torch.tensor([[[1, 2], [2, 3], [3, 4]], [[4, 5], [5, 6], [0, 1]]])
    return SimpleNamespace(candidate_logprobs=logp, teacher_topk_ids=topk), mask


def test_complete_output_signature_covers_valid_rows_offsets_and_bytes():
    output, mask = fixture_output()
    first = probe.compact_signature(output, mask, [3, 2], 10, 2, torch.float32)
    assert first["values_shape"] == [5, 2]
    assert first["offsets"] == [0, 3, 5]
    output.candidate_logprobs[1, 2] = 999  # excluded padding does not change the compact contract
    padding_changed = probe.compact_signature(output, mask, [3, 2], 10, 2, torch.float32)
    assert probe.compare_signatures(first, padding_changed)["match"]
    output.candidate_logprobs[0, 0, 0] = -0.123
    changed = probe.compact_signature(output, mask, [3, 2], 10, 2, torch.float32)
    assert not probe.compare_signatures(first, changed)["candidate_logprobs_hash_equal"]


@pytest.mark.parametrize("bad", ["shape", "dtype", "offsets", "topk_duplicate", "topk_range", "nan"])
def test_output_signature_rejects_material_contract_failure(bad):
    output, mask = fixture_output()
    lengths = [3, 2]
    if bad == "shape":
        output.candidate_logprobs = output.candidate_logprobs[..., :1]
    elif bad == "dtype":
        output.candidate_logprobs = output.candidate_logprobs.bfloat16()
    elif bad == "offsets":
        lengths = [2, 3]
    elif bad == "topk_duplicate":
        output.teacher_topk_ids[0, 0, 1] = 1
    elif bad == "topk_range":
        output.teacher_topk_ids[0, 0, 1] = 10
    else:
        output.candidate_logprobs[0, 0, 0] = float("nan")
    with pytest.raises(probe.OutputMismatch):
        probe.compact_signature(output, mask, lengths, 10, 2, torch.float32)


def test_head_boundary_distinguishes_full_projection_and_prehead_chunks():
    assert probe.validate_head_calls([{"input_shape": [2, 3, 4]}], "dense", 2, 3, 2)
    chunks = [{"input_shape": [2, 4]} for _ in range(3)]
    assert probe.validate_head_calls(chunks, "chunked", 2, 3, 2)
    assert not probe.validate_head_calls([{"input_shape": [6, 4]}], "chunked", 2, 3, 2)
    assert not probe.validate_head_calls(chunks[:-1], "chunked", 2, 3, 2)


def test_real_policy_types_are_distinguished():
    off = probe.policy_description(OffloadPolicy())
    on = probe.policy_description(CPUOffloadPolicy(pin_memory=True))
    assert off["cpu_offload"] is False and off["pin_memory"] is None
    assert on["cpu_offload"] is True and on["pin_memory"] is True


def args_for(tmp_path):
    return ["--model-dir", str(tmp_path / "model"), "--out", str(tmp_path / "result.json"),
            "--mode", "dense", "--cpu-offload", "off"]


def test_cli_default_scopes_and_allocator_limit(tmp_path):
    capacity = probe.parse_args([*args_for(tmp_path), "--purpose", "capacity", "--allocator-fraction", "0.5"])
    assert (capacity.warmup, capacity.repeats) == (0, 1)
    assert capacity.allocator_fraction == 0.5
    stability = probe.parse_args(args_for(tmp_path))
    assert (stability.warmup, stability.repeats) == (2, 30)
    assert stability.buffer_cast == "worker"
    assert stability.events.name == "result.json.events.jsonl"


@pytest.mark.parametrize("fraction", ["0", "-1", "1.1", "nan", "inf"])
def test_cli_rejects_invalid_allocator_fraction(tmp_path, fraction):
    with pytest.raises(SystemExit):
        probe.parse_args([*args_for(tmp_path), "--allocator-fraction", fraction])


def test_cli_preserves_prior_evidence_and_rejects_conflicting_hook_options(tmp_path):
    output = tmp_path / "result.json"
    output.write_text("prior")
    with pytest.raises(SystemExit):
        probe.parse_args(args_for(tmp_path))
    assert output.read_text() == "prior"
    with pytest.raises(SystemExit):
        probe.parse_args([*args_for(tmp_path / "other"), "--no-hooks", "--trace-hooks"])


def test_nvidia_smi_observation_is_all_device_and_failure_is_visible(monkeypatch):
    def completed(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "2026/10/30 00:00:00, 0, GPU, 1000, 7000, 8000, 2\n", "")
    monkeypatch.setattr(probe.subprocess, "run", completed)
    value = probe.read_device_observation(0)
    assert value["memory_used_mib"] == 1000
    assert value["status"] == "OBSERVED"
    monkeypatch.setattr(probe.subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 1, "", "failure"))
    assert probe.read_device_observation(0)["status"] == "UNAVAILABLE"


def test_fixture_is_padded_and_identical_across_conditions():
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1, encode=lambda *a, **kw: [2, 4, 6])
    inputs, ids, mask, identity = probe.build_fixture(tokenizer, [4, 2], 3, 17, "cpu")
    assert inputs["input_ids"].tolist() == [[2, 4, 6, 2], [2, 4, 0, 0]]
    assert mask.tolist() == [[True] * 4, [True, True, False, False]]
    assert identity["valid_rows"] == 6 and identity["padded_rows"] == 8
    assert ids.shape == (2, 4, 3)
    other = probe.build_fixture(tokenizer, [4, 2], 3, 17, "cpu")[-1]
    assert other == identity

def test_trace_hooks_are_durable_without_colliding_with_journal_event_name(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(3, 3)
            self.lm_head = torch.nn.Linear(3, 7)
        def forward(self, inputs):
            return self.lm_head(self.model(inputs))
    model = Model()
    journal = probe.JsonlJournal(tmp_path / "trace.jsonl")
    with probe.observe_calls(model, journal, True) as events:
        model(torch.ones(2, 3))
    rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
    journal.close()
    assert len(events) == len(rows) == 6
    assert all(row["event"] == "forward_hook" for row in rows)
    assert events[0]["module"] == "outer" and events[-1]["module"] == "outer"


def test_recorded_worker_fixture_preserves_duplicate_candidates_and_offsets():
    # The real Worker defaulted to pad ID 0 even when the tokenizer differs.
    tokenizer = SimpleNamespace(pad_token_id=16, eos_token_id=1)
    recorded = {"fixture": {
        "input_ids": [[2, 4, 6], [3]],
        "position_ids": [[0, 1, 2], [0]],
        "topk_ids": [[[1, 1], [2, 2], [3, 3]], [[4, 4]]],
    }}
    inputs, ids, mask, metadata = probe.build_fixture(tokenizer, [3, 1], 2, 17, "cpu", recorded)
    assert inputs["input_ids"].tolist() == [[2, 4, 6], [3, 0, 0]]
    assert ids[mask].tolist() == [[1, 1], [2, 2], [3, 3], [4, 4]]
    assert inputs["position_ids"].tolist() == [[0, 1, 2], [0, 0, 0]]
    assert metadata["valid_rows"] == 4 and metadata["pad_token_id"] == 0
    recorded["fixture"]["topk_ids"][0][0][0] = -1
    with pytest.raises(ValueError, match="outside the vocabulary"):
        probe.build_fixture(tokenizer, [3, 1], 2, 17, "cpu", recorded)