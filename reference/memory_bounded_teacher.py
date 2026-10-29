"""Opt-in Qwen2 teacher scoring inside the ordinary outer model forward.

Install before FSDP wrapping. The original parameters and module tree are kept;
each row chunk calls ``lm_head`` as a module so its own FSDP hooks still run.
This adapter does not install a verl worker or implement packed attention.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM


@dataclass(frozen=True)
class TeacherScores:
    candidate_logprobs: torch.Tensor
    teacher_topk_ids: torch.Tensor
    chunk_rows: int


def install_memory_bounded_teacher(model: Qwen2ForCausalLM) -> Qwen2ForCausalLM:
    """Add scoring kwargs to one unwrapped, ordinary Qwen2 model instance.

    Without ``teacher_topk_ids``, the original forward is used. Scoring accepts
    int64 IDs shaped ``[B,T,K]`` and positive ``teacher_chunk_rows``, and requires
    eval/no-grad execution. IDs already address causal-logit rows; no shift or
    temperature scaling is applied. Cache, labels, partial-logit selection,
    adapters, quantization, and alternative model forwards are unsupported.
    """
    if type(model) is not Qwen2ForCausalLM:
        raise TypeError("install on an unwrapped Qwen2ForCausalLM before FSDP")
    if "forward" in model.__dict__:
        raise ValueError("the model already has an instance forward override")
    if model.forward.__module__ != Qwen2ForCausalLM.__module__:
        raise ValueError("an alternative Qwen2 forward is unsupported")
    if type(model.lm_head) is not nn.Linear:
        raise TypeError("scoring requires an ordinary nn.Linear LM head")
    if not model.config.tie_word_embeddings or model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError("scoring currently requires tied input/output embeddings")
    if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("scoring currently supports distributed world_size=1 only")
    if getattr(model, "is_quantized", False) or getattr(model, "peft_config", None):
        raise ValueError("quantized and adapter models are unsupported")

    original_forward = model.forward

    def scoring_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        *,
        teacher_topk_ids=None,
        teacher_chunk_rows=None,
        **kwargs,
    ):
        if teacher_topk_ids is None:
            if teacher_chunk_rows is not None:
                raise ValueError("teacher_chunk_rows requires teacher_topk_ids")
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        if self.training or torch.is_grad_enabled():
            raise RuntimeError("teacher scoring requires eval mode and torch.no_grad()")
        if labels is not None or past_key_values is not None or use_cache:
            raise ValueError("teacher scoring does not support labels or a KV cache")
        if not isinstance(logits_to_keep, int) or logits_to_keep != 0:
            raise ValueError("teacher scoring requires every causal-logit row")
        if not isinstance(teacher_chunk_rows, int) or isinstance(teacher_chunk_rows, bool) or teacher_chunk_rows <= 0:
            raise ValueError("teacher_chunk_rows must be a positive integer")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids and inputs_embeds")
        source = input_ids if input_ids is not None else inputs_embeds
        if not isinstance(source, torch.Tensor) or source.is_nested:
            raise ValueError("teacher inputs must be dense tensors")
        shape = source.shape if input_ids is not None else source.shape[:-1]
        expected_ndim = 2 if input_ids is not None else 3
        if source.ndim != expected_ndim or any(n <= 0 for n in shape):
            raise ValueError("teacher inputs must have nonempty [B,T] leading dimensions")
        ids = teacher_topk_ids
        if not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or ids.is_nested:
            raise ValueError("teacher_topk_ids must be a dense int64 [B,T,K] tensor")
        if ids.ndim != 3 or tuple(ids.shape[:-1]) != tuple(shape):
            raise ValueError("teacher_topk_ids must align with every input row")
        vocab_size = self.config.vocab_size
        k = ids.shape[-1]
        if not 0 < k <= vocab_size or ids.min().item() < 0 or ids.max().item() >= vocab_size:
            raise ValueError("teacher candidate width or token IDs are outside the vocabulary")
        # Only options with a defined compact-output contract are accepted.
        for name, value in kwargs.items():
            if name not in {"return_dict", "output_attentions", "output_hidden_states"}:
                raise ValueError(f"unsupported teacher scoring option: {name}")
            if (name == "return_dict" and value is not True) or (name != "return_dict" and value not in (False, None)):
                raise ValueError(f"unsupported teacher scoring option: {name}={value}")

        hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        ).last_hidden_state
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_ids = ids.reshape(-1, k).to(hidden.device)
        logprobs = None
        topk = torch.empty(flat_ids.shape, dtype=torch.int64, device=hidden.device)
        for start in range(0, flat_hidden.shape[0], teacher_chunk_rows):
            end = min(start + teacher_chunk_rows, flat_hidden.shape[0])
            logits = self.lm_head(flat_hidden[start:end])
            if logprobs is None:
                logprobs = torch.empty(flat_ids.shape, dtype=logits.dtype, device=logits.device)
            logits_fp32 = logits.float()
            selected = logits_fp32.gather(-1, flat_ids[start:end])
            logprobs[start:end] = (selected - logits_fp32.logsumexp(-1, keepdim=True)).to(logits.dtype)
            topk[start:end] = logits.topk(k, dim=-1).indices
            del logits, logits_fp32, selected
        return TeacherScores(logprobs.reshape(ids.shape), topk.reshape(ids.shape), teacher_chunk_rows)

    model.forward = MethodType(scoring_forward, model)
    return model
