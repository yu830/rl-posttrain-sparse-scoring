"""CPU layout checks; these do not substitute for real Worker/GPU execution."""
import pytest
import torch

from scripts.worker_to_student_step import (
    compact_candidates,
    padded_fixture,
    restore_worker_values,
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