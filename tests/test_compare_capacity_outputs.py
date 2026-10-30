import copy

import pytest
import torch

from scripts.compare_capacity_outputs import compare_artifacts, tensor_hash


def artifact():
    logp = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], dtype=torch.bfloat16)
    ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    return {
        "identity": {"dtype": "bfloat16", "fixture": {"lengths": [2], "k": 2,
                     "valid_rows": 2, "padded_rows": 2}, "buffer_cast": "worker"},
        "output": {"values_shape": [2, 2], "padded_shape": [1, 2, 2], "offsets": [0, 2],
                   "candidate_logprobs": {"dtype": str(logp.dtype), "sha256": tensor_hash(logp)},
                   "teacher_topk_ids": {"dtype": str(ids.dtype), "sha256": tensor_hash(ids)}},
        "candidate_logprobs": logp, "teacher_topk_ids": ids,
    }


def refresh(item, key):
    value = item[key]
    item["output"][key]["sha256"] = tensor_hash(value)


def test_compact_outputs_pass_and_differences_are_reported():
    dense, candidate = artifact(), artifact()
    assert compare_artifacts(dense, candidate)["status"] == "PASS"
    candidate["candidate_logprobs"][0, 0] = -3.0
    refresh(candidate, "candidate_logprobs")
    result = compare_artifacts(dense, candidate)
    assert result["status"] == "FAIL_OR_TIE_UNKNOWN"
    assert result["candidate_logprobs"]["failed_values"] == 1
    candidate = artifact()
    candidate["teacher_topk_ids"][1, 0] = 7
    refresh(candidate, "teacher_topk_ids")
    result = compare_artifacts(dense, candidate)
    assert result["teacher_topk"]["different_rows"] == 1
    assert result["teacher_topk"]["tie_status_if_different"] == "UNKNOWN_without_full_reference_logits"


def test_identity_mismatch_is_rejected():
    dense, candidate = artifact(), artifact()
    candidate["identity"]["buffer_cast"] = "preserve"
    with pytest.raises(ValueError, match="different model"):
        compare_artifacts(dense, candidate)


def test_nonfinite_output_is_a_recordable_failure():
    dense, candidate = artifact(), artifact()
    candidate["candidate_logprobs"][0, 0] = float("nan")
    refresh(candidate, "candidate_logprobs")
    result = compare_artifacts(dense, candidate)
    assert result["status"] == "FAIL_OR_TIE_UNKNOWN"
    assert result["candidate_logprobs"]["nonfinite_values"] == 1


def test_matching_but_truncated_artifacts_are_rejected():
    dense = artifact()
    dense["candidate_logprobs"] = dense["candidate_logprobs"][:1]
    dense["teacher_topk_ids"] = dense["teacher_topk_ids"][:1]
    dense["output"]["values_shape"] = [1, 2]
    for key in ("candidate_logprobs", "teacher_topk_ids"):
        refresh(dense, key)
    with pytest.raises(ValueError, match="shapes do not match fixture"):
        compare_artifacts(dense, copy.deepcopy(dense))


def test_mutated_tensor_fails_producer_hash():
    dense, candidate = artifact(), artifact()
    candidate["candidate_logprobs"][0, 0] = -1.5
    with pytest.raises(ValueError, match="producer signature"):
        compare_artifacts(dense, candidate)


def test_ordered_topk_difference_is_reported_separately_from_set_equality():
    dense, candidate = artifact(), artifact()
    candidate["teacher_topk_ids"][0] = candidate["teacher_topk_ids"][0].flip(0)
    refresh(candidate, "teacher_topk_ids")
    result = compare_artifacts(dense, candidate)
    # Compact IDs alone cannot prove whether a changed order is a true tie.
    assert result["status"] == "FAIL_OR_TIE_UNKNOWN"
    assert result["teacher_topk"]["exact_set_equal"]
    assert not result["teacher_topk"]["ordered_exact_equal"]
    assert result["teacher_topk"]["ordered_equal_rows"] == 1
