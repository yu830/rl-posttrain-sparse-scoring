import pytest
import torch

from scripts.verify_verl_consumer import tensor_hash, validate_recorded_fixture


def test_fixture_hash_validation_rejects_changed_candidates():
    inputs = {
        "input_ids": [[1, 2]],
        "attention_mask": [[1, 1]],
        "position_ids": [[0, 1]],
        "response_logit_row_mask": [[True, False]],
    }
    hashes = {name: tensor_hash(torch.tensor(inputs[name], dtype=torch.int64))
              for name in ("input_ids", "attention_mask", "position_ids")}
    hashes["response_logit_row_mask"] = tensor_hash(torch.tensor(inputs["response_logit_row_mask"], dtype=torch.bool))
    candidates = [[[3, 4], [5, 6]]]
    source = {"inputs": {**inputs, "hashes": hashes}, "candidate_ids": candidates,
              "candidate_ids_sha256": tensor_hash(torch.tensor(candidates, dtype=torch.int64))}
    validate_recorded_fixture(source)
    source["candidate_ids"][0][0][0] = 2
    with pytest.raises(ValueError, match="candidate IDs hash"):
        validate_recorded_fixture(source)