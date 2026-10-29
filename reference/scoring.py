"""Explicit conditional-scoring reference, not an optimized serving implementation.

This module operates on fixed per-row support. It does not implement candidate
selection, a training loop, or SGLang's model-specific logits processing.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Readout:
    # Contract: ((h @ W.T) + bias) * scale, then optional softcap.
    scale: float = 1.0
    softcap: float | None = None
    adapter: bool = False
    quantized_weights: bool = False

    def validate(self) -> None:
        if not math.isfinite(self.scale) or not self.scale > 0:
            raise ValueError("scale must be finite and positive")
        if self.softcap is not None and (not math.isfinite(self.softcap) or not self.softcap > 0):
            raise ValueError("softcap must be finite and positive")
        if self.adapter or self.quantized_weights:
            raise NotImplementedError("LoRA and quantized readouts require a model-specific reference")


def validate_support(ids: torch.Tensor, mask: torch.Tensor, vocab: int) -> None:
    if ids.dtype != torch.int64 or mask.dtype != torch.bool:
        raise ValueError("ids must be int64 and mask bool")
    if ids.ndim != 2 or mask.shape != ids.shape or ids.shape[1] == 0:
        raise ValueError("ids and mask must have matching nonempty [rows, K] shape")
    # Deliberately slow validation: this is a correctness oracle, not a kernel.
    for row, valid in zip(ids.detach().cpu().tolist(), mask.detach().cpu().tolist()):
        selected = [i for i, keep in zip(row, valid) if keep]
        if not selected or any(i < 0 or i >= vocab for i in selected):
            raise ValueError("each row needs in-range active candidate IDs")
        if len(selected) != len(set(selected)):
            raise ValueError("active candidate IDs must be unique within each row")


def _transform(z: torch.Tensor, spec: Readout) -> torch.Tensor:
    z = z * spec.scale
    if spec.softcap is not None:
        z = spec.softcap * torch.tanh(z / spec.softcap)
    return z


def _inputs(h, weight, ids, mask, bias, spec):
    spec.validate()
    if h.ndim != 2 or weight.ndim != 2 or h.shape[1] != weight.shape[1]:
        raise ValueError("h and weight must be [rows, hidden] and [vocab, hidden]")
    if ids.shape[0] != h.shape[0] or h.device != weight.device or ids.device != h.device or mask.device != h.device:
        raise ValueError("row counts and devices must agree")
    if h.dtype != weight.dtype or (bias is not None and (bias.shape != (weight.shape[0],) or bias.dtype != h.dtype or bias.device != h.device)):
        raise ValueError("weight, bias and hidden dtype/device must agree")
    validate_support(ids, mask, weight.shape[0])
    # Inactive entries may contain any padding sentinel; never index with them.
    return ids.masked_fill(~mask, 0)


def dense_selected(h, weight, ids, mask, bias=None, spec=Readout()):
    safe = _inputs(h, weight, ids, mask, bias, spec)
    full = h @ weight.T
    if bias is not None:
        full = full + bias
    return _transform(full, spec).gather(-1, safe).masked_fill(~mask, 0)


def eager_selected(h, weight, ids, mask, bias=None, spec=Readout()):
    safe = _inputs(h, weight, ids, mask, bias, spec)
    # Intentionally materializes [rows, K, hidden]: a correctness baseline.
    selected_weight = weight[safe]
    logits = torch.einsum("rkd,rd->rk", selected_weight, h)
    if bias is not None:
        logits = logits + bias[safe]
    return _transform(logits, spec).masked_fill(~mask, 0)


def conditional_log_probs(logits, mask, temperature=1.0):
    if not math.isfinite(temperature) or not temperature > 0 or logits.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("finite positive temperature and matching boolean mask required")
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("each row needs at least one candidate")
    # Stable FP32 accumulation for FP16/BF16; FP64 stays FP64.
    dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    scaled = logits.to(dtype) / temperature
    return torch.log_softmax(scaled.masked_fill(~mask, -torch.inf), dim=-1)


def conditional_reverse_kl(student_logits, teacher_logits, mask, student_temperature=1.0, teacher_temperature=1.0):
    logp = conditional_log_probs(student_logits, mask, student_temperature)
    logq = conditional_log_probs(teacher_logits.detach(), mask, teacher_temperature)
    # Avoid inf-inf on padding before arithmetic and autograd.
    p = logp.masked_fill(~mask, 0)
    q = logq.masked_fill(~mask, 0)
    return (torch.exp(p) * (p - q)).masked_fill(~mask, 0).sum(-1)


def causal_positions(token_positions: torch.Tensor, sequence_length: int) -> torch.Tensor:
    if token_positions.dtype != torch.int64 or bool(((token_positions < 1) | (token_positions >= sequence_length)).any()):
        raise ValueError("token positions must satisfy 1 <= t < sequence_length")
    return token_positions - 1
