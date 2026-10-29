import math

import pytest
import torch

from reference.memory_budget import select_chunk_rows


def test_budget_selects_bounded_rows_and_reports_estimate():
    decision = select_chunk_rows(1024, 151936, 32, torch.bfloat16,
                                 requested_rows=512, head_memory_budget_mib=256)
    assert 0 < decision.rows < 512
    assert decision.reason == "budget_limited"
    assert decision.estimated_bytes <= decision.budget_bytes
    assert decision.budget_bytes == 256 * 1024**2


def test_fixed_full_and_budget_within_limit():
    fixed = select_chunk_rows(8, 64, 4, torch.float32, requested_rows=100)
    assert fixed.rows == 8
    assert fixed.reason == "fixed_rows"
    within = select_chunk_rows(8, 64, 4, torch.float32, head_memory_budget_mib=1)
    assert within.rows == 8
    assert within.reason == "within_budget"


@pytest.mark.parametrize("bad", [0, -1, math.inf, math.nan])
def test_rejects_bad_budget(bad):
    with pytest.raises(ValueError):
        select_chunk_rows(4, 32, 4, torch.float32, head_memory_budget_mib=bad)


def test_rejects_budget_too_small_for_one_row():
    with pytest.raises(ValueError, match="one estimated row"):
        select_chunk_rows(1024, 151936, 32, torch.bfloat16, head_memory_budget_mib=0.1)


@pytest.mark.parametrize("kwargs", [
    {"num_rows": 0},
    {"vocab_size": 0},
    {"candidate_width": 0},
    {"candidate_width": 33},
    {"requested_rows": 0},
    {"requested_rows": True},
    {"safety_factor": 0.9},
])
def test_rejects_invalid_shape_or_sizing(kwargs):
    args = dict(num_rows=4, vocab_size=32, candidate_width=4, logits_dtype=torch.float32)
    args.update(kwargs)
    with pytest.raises(ValueError):
        select_chunk_rows(**args)