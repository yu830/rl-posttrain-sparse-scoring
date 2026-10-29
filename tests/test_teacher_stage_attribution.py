"""CPU checks for stage-selection and normalization diagnostics; no model load."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "stage_diagnosis", Path(__file__).resolve().parents[1] / "scripts/diagnose_teacher_stage_attribution.py")
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def test_predictor_selection_uses_real_prefix_end():
    values = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    assert torch.equal(stage.select_predictors(values, [2, 4]), torch.stack([values[0, 1], values[1, 3]]))
    with pytest.raises(ValueError):
        stage.select_predictors(values, [0, 4])


def test_normalization_and_gather_cover_unsorted_duplicates():
    logits = torch.tensor([[100.0, 0.0, 99.0], [-20.0, 3.0, 3.0]])
    ids = [[2, 0, 2], [1, 0, 2]]
    result = stage.readout(logits, ids)
    expected = torch.log_softmax(logits.double(), dim=-1).gather(1, torch.tensor(ids))
    torch.testing.assert_close(result["fp64_cpu_selected_logprobs"], expected, atol=1e-14, rtol=0)
    assert result["gather_vs_scalar_index"]["unequal_finite_elements"] == 0
    assert result["fp32_vs_fp64_cpu_normalization"]["max_abs_error"] < 1e-5
    with pytest.raises(ValueError):
        stage.readout(logits, [[0, -1], [1, 2]])


def test_frozen_budget_rejects_wrong_candidate():
    result = stage.frozen_comparison([[-4.0, -20.0]], [[-4.0, -20.5]], "bfloat16")
    assert result["failed_elements"] == 1
    assert result["atol"] == 0.1 and result["rtol"] == 0.01
    assert stage.BUDGETS["float32"] == (5e-5, 2e-5)


def test_nonfinite_mismatch_is_not_silent_success():
    result = stage.difference(torch.tensor([float("nan"), float("inf"), 1.0]),
                              torch.tensor([0.0, -float("inf"), 1.0]))
    assert result["nonfinite_class_mismatches"] == 2
    assert result["finite_pairs"] == 1
    assert result["first_difference_index"] == [0]


def test_fixture_rejects_changed_inputs_or_positions():
    fixture = {"status": "PASS", "model": "tiny", "model_revision": "fixed",
               "tokenizer_revision": "fixed", "predictor_last_prefix_indices": [1],
               "request": {"items": [[3, 4]], "label_token_ids": [[5, 6]], "temperature": 1.0}}
    stage.validate_fixture(fixture, copy.deepcopy(fixture))
    changed = copy.deepcopy(fixture)
    changed["request"]["items"][0][0] = 9
    with pytest.raises(ValueError, match="request"):
        stage.validate_fixture(fixture, changed)
    changed = copy.deepcopy(fixture)
    changed["predictor_last_prefix_indices"] = [0]
    with pytest.raises(ValueError, match="predictor"):
        stage.validate_fixture(fixture, changed)


def test_nonfinite_json_is_valid_and_preserves_classes():
    import json
    values = {"nested": [float("nan"), float("inf"), -float("inf"), 1.5]}
    encoded = json.dumps(stage.json_safe(values), allow_nan=False)
    assert json.loads(encoded)["nested"] == [
        {"nonfinite_float": "NaN"}, {"nonfinite_float": "+Inf"},
        {"nonfinite_float": "-Inf"}, 1.5]
    result = stage.difference(torch.tensor([float("nan"), float("inf")]),
                              torch.tensor([float("nan"), float("inf")]))
    assert result["actual_nonfinite"] == {"nan": 1, "posinf": 1, "neginf": 0}
    assert result["finite_pairs"] == 0 and result["max_abs_error"] is None


def manifest_fixture(tmp_path):
    import json
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"hidden_size": 3}')
    (model / "model.safetensors").write_bytes(b"fixed-weights")
    tensors = tmp_path / "hidden.pt"
    tensors.write_bytes(b"fixed-tensor-bytes")
    fixture = {"model": "tiny", "model_revision": "fixed", "tokenizer_revision": "fixed",
               "request": {"items": [[3, 4]], "label_token_ids": [[5, 6]]},
               "predictor_last_prefix_indices": [1]}
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps(fixture))
    manifest = dict(fixture, status="MEASURED", tensor_sha256=stage.digest(tensors),
                    predictor_positions=[1], inputs=[{"sha256": stage.digest(smoke)}],
                    config_sha256=stage.digest(model / "config.json"),
                    weight_files=[{"file": "model.safetensors", "bytes": 13,
                                   "sha256": stage.digest(model / "model.safetensors")}])
    return manifest, tensors, smoke, model


def test_manifest_binds_tensor_fixture_positions_and_weights(tmp_path):
    manifest, tensors, smoke, model = manifest_fixture(tmp_path)
    stage.validate_capture_manifest(manifest, tensors, smoke, model)
    for key, value, message in (("request", {"items": [[9, 4]], "label_token_ids": [[5, 6]]}, "request"),
                                ("predictor_positions", [0], "positions"),
                                ("model_revision", "other", "model_revision"),
                                ("tensor_sha256", "wrong", "tensor SHA"),
                                ("config_sha256", "wrong", "config hash")):
        changed = copy.deepcopy(manifest)
        changed[key] = value
        with pytest.raises(ValueError, match=message):
            stage.validate_capture_manifest(changed, tensors, smoke, model)
    (model / "model.safetensors").write_bytes(b"other-weights")
    with pytest.raises(ValueError, match="weight hashes"):
        stage.validate_capture_manifest(manifest, tensors, smoke, model)


def test_default_inputs_work_without_ignored_raw_files(tmp_path, monkeypatch):
    import json
    import shutil
    import sys
    root = Path(__file__).resolve().parents[1]
    results = tmp_path / "results"
    results.mkdir()
    # Exactly the three tracked defaults; ignored per-configuration JSONs absent.
    for name in ("sglang-teacher-smoke-20261029.json", "teacher-hf-comparison-20261029.json",
                 "precision-controls-20261029.json"):
        shutil.copyfile(root / "results" / name, results / name)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"not-loaded")
    out = tmp_path / "cpu-guard.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(root / "scripts/diagnose_teacher_stage_attribution.py"),
                                     "--model-dir", str(model), "--out", str(out)])
    monkeypatch.setattr(stage.subprocess, "check_output", lambda *a, **k: "fixed-head\n")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert stage.main() == 1
    result = json.loads(out.read_text())
    assert result["status"] == "FAILED" and "CUDA required" in result["error"]
    assert "cases[name=fp32_triton_ieee]" in result["fp32_evidence_source"]["smoke"]
    assert len(result["inputs"]) == 3
