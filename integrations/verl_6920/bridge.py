"""Layout conversion for the real, patched verl teacher worker.

These helpers contain no worker substitute. They preserve the rmpad output
order while allowing the installed scorer to consume ordinary [B,T,K] IDs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from reference.memory_budget import ChunkDecision


@dataclass(frozen=True)
class ScoringLayout:
    candidate_ids: torch.Tensor
    valid_mask: torch.Tensor | None

    def restore_values(self, values: torch.Tensor) -> torch.Tensor:
        """Discard padded rows and restore the worker's [total_nnz,K] order."""
        if tuple(values.shape) != tuple(self.candidate_ids.shape):
            raise ValueError("teacher scores do not match the prepared candidate layout")
        if self.valid_mask is None:
            return values.squeeze(0)
        return values[self.valid_mask.to(values.device)]


def prepare_scoring_layout(
    input_ids: torch.Tensor,
    topk_ids: torch.Tensor,
    model_inputs: dict,
    *,
    use_remove_padding: bool,
) -> ScoringLayout:
    """Prepare aligned IDs without changing causal positions or attention.

    Multi-sequence packed/rmpad attention is deliberately unsupported until
    its actual varlen execution has been verified. Padded inputs may have
    arbitrary valid positions as long as their per-sequence lengths agree
    with the nested input and candidate offsets.
    """
    if not input_ids.is_nested or not topk_ids.is_nested:
        raise ValueError("input_ids and topk_ids must be nested tensors")
    if topk_ids.dtype != torch.int64 or topk_ids.dim() != 3:
        raise ValueError("topk_ids must be nested int64 [B,seq_len_i,K]")
    offsets = input_ids.offsets()
    if not torch.equal(offsets, topk_ids.offsets().to(offsets.device)):
        raise ValueError("candidate offsets must match input_ids offsets")
    lengths = offsets.diff()
    if lengths.numel() == 0 or bool((lengths <= 0).any()):
        raise ValueError("teacher scoring requires nonempty sequences")
    values = topk_ids.values()
    if values.ndim != 2 or values.shape[-1] <= 0:
        raise ValueError("teacher scoring requires a nonempty candidate width")
    dense_inputs = model_inputs.get("input_ids")
    if not isinstance(dense_inputs, torch.Tensor) or dense_inputs.is_nested or dense_inputs.ndim != 2:
        raise ValueError("the prepared model input must be dense [B,T]")
    if use_remove_padding:
        if lengths.numel() != 1:
            raise NotImplementedError(
                "multi-sequence rmpad requires verified varlen attention; use padded teacher inputs"
            )
        candidate_ids = values.unsqueeze(0).to(dense_inputs.device)
        if tuple(candidate_ids.shape[:-1]) != tuple(dense_inputs.shape):
            raise ValueError("single-sequence rmpad candidates do not match prepared inputs")
        return ScoringLayout(candidate_ids, None)

    attention_mask = model_inputs.get("attention_mask")
    if not isinstance(attention_mask, torch.Tensor) or tuple(attention_mask.shape) != tuple(dense_inputs.shape):
        raise ValueError("padded teacher scoring requires the prepared [B,T] attention mask")
    valid_mask = attention_mask.to(device=dense_inputs.device, dtype=torch.bool)
    if not torch.equal(valid_mask.sum(-1).to(lengths.device), lengths):
        raise ValueError("prepared attention-mask lengths do not match nested input offsets")
    candidate_ids = torch.zeros(
        (*dense_inputs.shape, values.shape[-1]), dtype=values.dtype, device=dense_inputs.device
    )
    candidate_ids[valid_mask] = values.to(dense_inputs.device)
    return ScoringLayout(candidate_ids, valid_mask)


def validate_teacher_candidates(candidate_ids: torch.Tensor, vocab_size: int) -> None:
    """Reject invalid gather indices before entering the outer FSDP forward."""
    if not 0 < candidate_ids.shape[-1] <= vocab_size:
        raise ValueError("candidate width is outside the vocabulary")
    if candidate_ids.min().item() < 0 or candidate_ids.max().item() >= vocab_size:
        raise ValueError("candidate token IDs are outside the vocabulary")


def select_teacher_execution_path(*, num_model_rows: int, decision: ChunkDecision) -> str:
    """Keep native readout for a full-row tile only when no budget was requested.

    A complete tile retains all logits and can increase the FSDP high-water
    mark by reducing them while parameters remain unsharded. An explicit
    budget must not be silently replaced by an unbounded dense execution.
    Model rows include padded positions actually projected by the adapter.
    """
    if decision.budget_bytes is None and decision.rows >= num_model_rows:
        return "dense_full_row_fallback"
    return "chunked_lm_head"
