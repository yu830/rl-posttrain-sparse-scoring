"""Conservative sizing for a teacher LM-head readout.

The budget covers a model of PyTorch allocations attributable to readout, not
all CUDA allocations, kernel workspace, or memory owned by other processes.
Actual high-water marks must be recorded and compared with this estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch


@dataclass(frozen=True)
class ChunkDecision:
    rows: int
    estimated_bytes: int
    budget_bytes: int | None
    reason: str


def select_chunk_rows(
    num_rows: int,
    vocab_size: int,
    candidate_width: int,
    logits_dtype: torch.dtype,
    *,
    requested_rows: int | None = None,
    head_memory_budget_mib: float | None = None,
    safety_factor: float = 1.25,
) -> ChunkDecision:
    """Choose a row count before the FSDP forward, or fail on an impossible budget.

    The estimate includes compact [N,K] outputs, a logits tile, up to two
    FP32 vocabulary-sized work tensors, and conservative per-row top-K
    bookkeeping. Backend workspaces and fragmentation are not bounded here.
    It is an estimate, not an OOM guarantee; calibrate it on the target GPU.
    """
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in
           (num_rows, vocab_size, candidate_width)):
        raise ValueError("row, vocabulary and candidate counts must be integers")
    if num_rows <= 0 or vocab_size <= 0 or not 0 < candidate_width <= vocab_size:
        raise ValueError("num_rows, vocab_size and candidate_width must be positive")
    if logits_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("unsupported logits dtype")
    if requested_rows is not None and (
        isinstance(requested_rows, bool) or not isinstance(requested_rows, Integral) or requested_rows <= 0
    ):
        raise ValueError("requested_rows must be a positive integer")
    if isinstance(safety_factor, bool) or not isinstance(safety_factor, Real) or not math.isfinite(safety_factor) or safety_factor < 1:
        raise ValueError("safety_factor must be finite and at least 1")

    dtype_bytes = torch.empty((), dtype=logits_dtype).element_size()
    compact_bytes = num_rows * candidate_width * (dtype_bytes + 8)
    row_bytes = vocab_size * (dtype_bytes + 8) + candidate_width * 24
    per_row_with_margin = math.ceil(row_bytes * safety_factor)
    requested = min(num_rows, requested_rows or num_rows)
    if head_memory_budget_mib is None:
        return ChunkDecision(requested, compact_bytes + requested * per_row_with_margin, None,
                             "fixed_rows" if requested_rows is not None else "full_rows")
    if (isinstance(head_memory_budget_mib, bool) or not isinstance(head_memory_budget_mib, Real)
            or not math.isfinite(head_memory_budget_mib) or head_memory_budget_mib <= 0):
        raise ValueError("head_memory_budget_mib must be finite and positive")
    budget_bytes = int(head_memory_budget_mib * (1024**2))
    max_rows = (budget_bytes - compact_bytes) // per_row_with_margin
    if max_rows < 1:
        raise ValueError("head memory budget cannot fit compact outputs and one estimated row")
    chosen = min(requested, max_rows)
    return ChunkDecision(chosen, compact_bytes + chosen * per_row_with_margin,
                         budget_bytes, "budget_limited" if chosen < requested else "within_budget")