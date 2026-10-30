"""CPU layout checks; these do not substitute for real Worker/GPU execution."""
import hashlib
import importlib
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.worker_to_student_step import (
    compact_candidates,
    padded_fixture,
    restore_worker_values,
    response_diagnostics,
    update_student,
    tensor_hash,
    worker_output_snapshot,
)


def test_right_padded_fixture_preserves_causal_response_rows():
    inputs, mask, offsets = padded_fixture([[2, 3, 4, 5], [6, 7, 8]], [2, 1], 0)
    assert inputs["input_ids"].tolist() == [[2, 3, 4, 5], [6, 7, 8, 0]]
    assert inputs["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert inputs["position_ids"].tolist() == [[0, 1, 2, 3], [0, 1, 2, 0]]
    assert mask.tolist() == [[False, True, True, False], [True, True, False, False]]
    assert offsets.tolist() == [0, 4, 7]


@pytest.mark.parametrize("prompts", [[0, 1], [4, 1], [2]])
def test_fixture_rejects_missing_response_or_wrong_prompt_count(prompts):
    with pytest.raises(ValueError):
        padded_fixture([[2, 3, 4, 5], [6, 7, 8]], prompts, 0)


def test_actual_nested_values_scatter_directly_to_valid_student_rows():
    offsets = torch.tensor([0, 3, 5])
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
                           [7.0, 8.0], [9.0, 10.0]])
    nested = torch.nested.nested_tensor_from_jagged(values, offsets)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    actual = restore_worker_values(nested, offsets, mask, 2)
    torch.testing.assert_close(actual[mask.bool()], values, atol=0, rtol=0)
    assert actual[1, 2].tolist() == [0.0, 0.0]
    # Reversing sequence boundaries while retaining total rows must fail.
    with pytest.raises(ValueError, match="offsets"):
        restore_worker_values(nested, torch.tensor([0, 2, 5]), mask, 2)
    with pytest.raises(ValueError, match="lengths"):
        restore_worker_values(nested, offsets, torch.tensor([[1, 1, 0], [1, 1, 1]]), 2)


def test_compact_candidates_preserve_order_and_duplicate_ids():
    candidates = torch.tensor([[[3, 3], [1, 2], [5, 4]], [[6, 7], [8, 8], [9, 0]]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    compact = compact_candidates(candidates, mask)
    assert compact.tolist() == [[3, 3], [1, 2], [5, 4], [6, 7], [8, 8]]
    with pytest.raises(ValueError, match="int64"):
        compact_candidates(candidates.int(), mask)
    with pytest.raises(ValueError, match="rows"):
        compact_candidates(candidates[:, :-1], mask)


def test_worker_snapshot_keeps_offsets_dtype_and_exact_tensor_hash():
    offsets = torch.tensor([0, 2, 3])
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.bfloat16)
    nested = torch.nested.nested_tensor_from_jagged(values, offsets)
    snapshot = worker_output_snapshot({"teacher_on_student_logp": nested})
    raw = snapshot["teacher_on_student_logp"]
    assert raw["offsets"] == [0, 2, 3]
    assert raw["dtype"] == "torch.bfloat16"
    assert raw["values_sha256"] == tensor_hash(values)
    assert raw["values"] == values.float().tolist()

@pytest.fixture
def response_case():
    inputs, mask, offsets = padded_fixture(
        [[2, 3, 4, 5, 6, 7], [8, 9, 10, 11]], [3, 2], 0)
    # Includes attention-valid prompt rows, final non-predicting rows and padding.
    outputs = {
        "student_mass": torch.full(mask.shape, -16.0),
        "teacher_mass": torch.full(mask.shape, 8.0),
        "distillation_losses": torch.full(mask.shape, 1000.0),
        "overlap_count": torch.full(mask.shape, 2, dtype=torch.int64),
        "overlap_token_advantage": torch.full(mask.shape, -1000.0),
    }
    for name, valid in {
        "student_mass": [0.25, 0.5, 0.75, 0.125, 0.875],
        "teacher_mass": [0.125, 0.25, 0.5, 0.75, 0.875],
        "distillation_losses": [-0.5, 0.25, 20.0, 0.5, 1.0],
        "overlap_count": [0, 1, 2, 0, 2],
        "overlap_token_advantage": [999.0, -0.5, 0.25, 999.0, 1.0],
    }.items():
        outputs[name][mask] = torch.tensor(valid, dtype=outputs[name].dtype)
    return inputs, mask, offsets, outputs


def test_response_metrics_hand_calculated_and_raw_values_unchanged(response_case):
    inputs, mask, _, outputs = response_case
    original = {name: value.clone() for name, value in outputs.items()}
    assert (inputs["attention_mask"].bool() & ~mask).any()  # real prompt rows
    assert (~inputs["attention_mask"].bool()).any()  # actual padding rows
    assert response_diagnostics(outputs, mask, topk=2) == {
        "distillation/student_mass": 0.5,
        "distillation/student_mass_min": 0.125,
        "distillation/student_mass_max": 0.875,
        "distillation/teacher_mass": 0.5,
        "distillation/teacher_mass_min": 0.125,
        "distillation/teacher_mass_max": 0.875,
        # Range follows clamp_min(0), BEFORE the optimization loss_max_clamp=10.
        "distillation/loss_min": 0.0,
        "distillation/loss_max": 20.0,
        "distillation/overlap_ratio": 0.5,
        # Positive-overlap response positions only: (-0.5 + 0.25 + 1) / 3.
        "distillation/overlap_token_advantage": 0.25,
    }
    for name in outputs:
        torch.testing.assert_close(outputs[name], original[name], atol=0, rtol=0)


@pytest.mark.parametrize("region", ["prompt", "padding"])
def test_nonresponse_extremes_cannot_change_any_metric(response_case, region):
    inputs, mask, _, outputs = response_case
    before = response_diagnostics(outputs, mask, topk=2)
    excluded = ((inputs["attention_mask"].bool() & ~mask) if region == "prompt"
                else ~inputs["attention_mask"].bool())
    for name, value in outputs.items():
        value[excluded] = 0 if name == "overlap_count" else float("nan")
    assert response_diagnostics(outputs, mask, topk=2) == before


def test_zero_overlap_response_fallback_ignores_nonresponse_overlap(response_case):
    _, mask, _, outputs = response_case
    outputs["overlap_count"][mask] = 0
    outputs["overlap_token_advantage"][mask] = float("nan")
    metrics = response_diagnostics(outputs, mask, topk=2)
    assert metrics["distillation/overlap_ratio"] == 0.0
    assert metrics["distillation/overlap_token_advantage"] == 0.0
    # Non-response positions still have positive overlap and extreme advantage.
    assert (outputs["overlap_count"][~mask] > 0).all()


def test_optional_overlap_and_empty_response_contract(response_case):
    _, mask, _, outputs = response_case
    outputs.pop("overlap_count")
    outputs.pop("overlap_token_advantage")
    assert len(response_diagnostics(outputs, mask, topk=2)) == 8
    with pytest.raises(ValueError, match="at least one response"):
        response_diagnostics(outputs, torch.zeros_like(mask), topk=2)


@pytest.fixture(scope="module")
def pinned_trainer_metrics():
    """Opt-in full pinned imports; never substitute a fake upstream function."""
    directory = os.environ.get("VERL_MASK_REGRESSION_SOURCE")
    if not directory:
        pytest.skip("set VERL_MASK_REGRESSION_SOURCE (and VERL_MASK_REGRESSION_DEPS if needed) for upstream CPU parity")
    directory = Path(directory).resolve()
    hashes = {
        "verl/trainer/distillation/losses.py": "27d368a4756b7763e3ad9d484d7aa47e9e07efc9e3a51f106c0e883ef0d47aa7",
        "verl/workers/utils/padding.py": "7fde938841e6eae368a16df1aab271723ba21329bac48fa492fc6912ba1038e6",
        "verl/utils/metric/utils.py": "b17a4ce737d154a2b7ccbf2dc238be48b0d580203133082d3f98e72e70e0e3bd",
    }
    for relative, expected in hashes.items():
        assert hashlib.sha256((directory / relative).read_bytes()).hexdigest() == expected
    # Add runtime dependencies after importing the local scripts namespace;
    # the upstream checkout also has a scripts package that would shadow it.
    sys.path.insert(0, str(directory))
    dependencies = os.environ.get("VERL_MASK_REGRESSION_DEPS")
    if dependencies:
        sys.path.insert(0, dependencies)
    module = importlib.import_module("verl.trainer.distillation.losses")
    assert Path(inspect.getfile(module)).resolve() == directory / "verl/trainer/distillation/losses.py"
    assert Path(inspect.getfile(module.no_padding_2_padding)).resolve() == directory / "verl/workers/utils/padding.py"
    assert Path(inspect.getfile(module.Metric)).resolve() == directory / "verl/utils/metric/utils.py"
    return module


@pytest.mark.parametrize("zero_overlap", [False, True])
def test_all_metrics_equal_actual_pinned_trainer(response_case, pinned_trainer_metrics, zero_overlap):
    from tensordict import TensorDict

    inputs, mask, _, outputs = response_case
    if zero_overlap:
        outputs["overlap_count"][mask] = 0
        outputs["overlap_token_advantage"][mask] = 999.0
    # The actual trainer consumes compact full-sequence outputs, slices the
    # causal response rows, then right-pads responses to their maximum length.
    nested = lambda rows: torch.nested.nested_tensor(rows, layout=torch.jagged)
    data = TensorDict({
        "prompts": nested([[2, 3, 4], [8, 9]]),
        "responses": nested([[5, 6, 7], [10, 11]]),
        "response_mask": torch.tensor([[True, True, True], [True, True, False]]),
    }, batch_size=[2])
    compact = {name: value[inputs["attention_mask"].bool()] for name, value in outputs.items()}
    module = pinned_trainer_metrics
    losses, expected = module.compute_reverse_kl_topk(
        config=None, distillation_config=SimpleNamespace(distillation_loss=SimpleNamespace(topk=2)),
        model_output=compact, data=data)
    expected.update(module.compute_distillation_loss_range(losses, data["response_mask"]))
    expected = {name: value.aggregate() if isinstance(value, module.Metric) else value
                for name, value in expected.items()}
    assert response_diagnostics(outputs, mask, topk=2) == expected


def test_student_update_reports_masked_metrics_without_changing_loss_or_gradient(response_case):
    inputs, mask, offsets, fixture_outputs = response_case

    class TinyStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.norm = torch.nn.LayerNorm(2, elementwise_affine=True, bias=False)

        def forward(self, input_ids, attention_mask, position_ids, use_cache):
            return SimpleNamespace(logits=self.model.norm.weight *
                                   torch.ones((*input_ids.shape, 2)))

    student = TinyStudent()
    initial = student.model.norm.weight.detach().clone()
    teacher_values = torch.ones((int(offsets[-1]), 2))
    teacher_output = {
        "teacher_on_student_logp": torch.nested.nested_tensor_from_jagged(teacher_values, offsets),
        "teacher_topk_ids": torch.nested.nested_tensor_from_jagged(teacher_values.long(), offsets),
    }
    ids = torch.zeros((*mask.shape, 2), dtype=torch.int64)

    def upstream(**kwargs):
        # An analytic CPU test double exercises reporting integration, not verl.
        logits = kwargs["student_logits"]
        return {**fixture_outputs, "distillation_losses": logits[..., 0] - 0.5 * logits[..., 1]}

    values, report = update_student(
        student, initial, teacher_output, inputs, ids, mask, offsets, upstream,
        SimpleNamespace(topk=2, learning_rate=0.01))
    expected_initial = initial.clone().requires_grad_()
    temperature = torch.full(mask.shape, 0.8).unsqueeze(-1)
    expected_logits = expected_initial * torch.ones((*mask.shape, 2)) / temperature
    expected_per_token = expected_logits[..., 0] - 0.5 * expected_logits[..., 1]
    expected_loss = expected_per_token.clamp_min(0).clamp(-10, 10)[mask].mean()
    expected_loss.backward()
    torch.testing.assert_close(values["loss"], expected_loss.detach(), atol=0, rtol=0)
    torch.testing.assert_close(values["norm_gradient"], expected_initial.grad, atol=0, rtol=0)
    assert not values["logits_gradient"][~mask].count_nonzero()
    assert report["response_metrics"]["distillation/teacher_mass_max"] == 0.875
    assert torch.tensor(report["diagnostics"]["teacher_mass"])[~mask].eq(8).all()
    assert report["response_metrics"]["distillation/overlap_token_advantage"] == 0.25
