"""CPU real-Qwen2 forward checks; these do not validate distributed FSDP."""
import pytest
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from reference.memory_bounded_teacher import TeacherScores, install_memory_bounded_teacher


@pytest.fixture
def model():
    torch.manual_seed(731)
    config = Qwen2Config(
        vocab_size=37, hidden_size=24, intermediate_size=40,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, tie_word_embeddings=True,
        attention_dropout=0.0, use_cache=True,
    )
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).eval()


@pytest.fixture
def inputs():
    tokens = torch.tensor([[2, 5, 7, 3, 9], [0, 0, 11, 13, 17]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]])
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    ids = torch.tensor([[[0, 2, 0], [4, 9, 2], [36, 2, 5], [9, 8, 7], [1, 3, 6]],
                        [[5, 4, 3], [6, 7, 8], [0, 10, 36], [11, 13, 17], [9, 10, 11]]])
    return dict(input_ids=tokens, attention_mask=mask, position_ids=positions), ids


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_rows", [1, 3, 64])
def test_complete_qwen_forward_matches_dense_and_bounds_each_projection(model, inputs, dtype, chunk_rows):
    model.to(dtype)
    batch, ids = inputs
    with torch.no_grad():
        dense = model(**batch, use_cache=False).logits
    expected = (dense.float().gather(-1, ids) - dense.float().logsumexp(-1, keepdim=True)).to(dtype)
    expected_topk_values = dense.topk(ids.shape[-1], -1).values
    projections = []
    handle = model.lm_head.register_forward_pre_hook(lambda module, args: projections.append(tuple(args[0].shape)))
    install_memory_bounded_teacher(model)
    with torch.no_grad():
        actual = model(**batch, teacher_topk_ids=ids, teacher_chunk_rows=chunk_rows)
    handle.remove()
    assert isinstance(actual, TeacherScores)
    assert actual.candidate_logprobs.dtype == dtype
    tolerance = dict(atol=0.04, rtol=0.01) if dtype == torch.bfloat16 else dict(atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual.candidate_logprobs, expected, **tolerance)
    # Exact tied-score permutations are valid; compare scores rather than IDs.
    torch.testing.assert_close(dense.gather(-1, actual.teacher_topk_ids), expected_topk_values, **tolerance)
    assert all(shape[0] <= chunk_rows and len(shape) == 2 for shape in projections)
    assert sum(shape[0] for shape in projections) == ids.shape[0] * ids.shape[1]
    assert actual.chunk_rows == chunk_rows
    assert not actual.candidate_logprobs.requires_grad
    assert not hasattr(actual, "logits")


def test_install_preserves_parameter_identity_tying_state_dict_and_standard_forward(model, inputs):
    batch, _ = inputs
    before = {name: id(p) for name, p in model.named_parameters()}
    state = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        expected = model(**batch, use_cache=True, output_hidden_states=True)
    assert install_memory_bounded_teacher(model) is model
    assert {name: id(p) for name, p in model.named_parameters()} == before
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert set(model.state_dict()) == set(state)
    with torch.no_grad():
        actual = model(**batch, use_cache=True, output_hidden_states=True)
    torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
    assert actual.past_key_values.get_seq_length() == expected.past_key_values.get_seq_length()
    for a, b in zip(actual.hidden_states, expected.hidden_states):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], atol=0, rtol=0)
    # The standard, non-scoring path remains a trainable causal LM.
    model.train()
    loss = model(**batch, labels=batch["input_ids"], use_cache=False).loss
    loss.backward()
    assert model.lm_head.weight.grad is not None
    assert model.lm_head.weight.grad.abs().sum() > 0


def test_outer_backbone_and_head_call_hooks_complete_and_no_request_state_leaks(model, inputs):
    batch, ids = inputs
    install_memory_bounded_teacher(model)
    events = []
    handles = []
    for name, module in [("outer", model), ("backbone", model.model), ("head", model.lm_head)]:
        handles.append(module.register_forward_pre_hook(lambda module, args, label=name: events.append((label, "pre"))))
        handles.append(module.register_forward_hook(lambda module, args, out, label=name: events.append((label, "post"))))
    with torch.no_grad():
        expected = model(**batch, teacher_topk_ids=ids, teacher_chunk_rows=3)
    assert events[0] == ("outer", "pre") and events[-1] == ("outer", "post")
    assert events.count(("backbone", "pre")) == events.count(("backbone", "post")) == 1
    assert events.count(("head", "pre")) == events.count(("head", "post")) == 4
    for handle in handles:
        handle.remove()

    def fail_projection(module, args):
        raise RuntimeError("injected head failure")

    handle = model.lm_head.register_forward_pre_hook(fail_projection)
    with torch.no_grad(), pytest.raises(RuntimeError, match="injected head failure"):
        model(**batch, teacher_topk_ids=ids, teacher_chunk_rows=3)
    handle.remove()
    with torch.no_grad():
        recovered = model(**batch, teacher_topk_ids=ids, teacher_chunk_rows=3)
        standard = model(**batch, use_cache=False)
    torch.testing.assert_close(recovered.candidate_logprobs, expected.candidate_logprobs, atol=0, rtol=0)
    assert standard.logits.shape == (*batch["input_ids"].shape, model.config.vocab_size)
    assert not any(name.startswith("teacher_") for name in model.__dict__)


@pytest.mark.parametrize("option,match", [
    ({"teacher_chunk_rows": 0}, "positive integer"),
    ({"teacher_chunk_rows": True}, "positive integer"),
    ({"labels": torch.ones(2, 5, dtype=torch.int64)}, "labels or a KV cache"),
    ({"use_cache": True}, "labels or a KV cache"),
    ({"logits_to_keep": 1}, "every causal-logit row"),
    ({"output_hidden_states": True}, "unsupported teacher scoring option"),
    ({"temperature": 0.8}, "unsupported teacher scoring option"),
])
def test_unsupported_scoring_options_fail_and_standard_forward_recovers(model, inputs, option, match):
    batch, ids = inputs
    install_memory_bounded_teacher(model)
    options = dict(teacher_topk_ids=ids, teacher_chunk_rows=2)
    options.update(option)
    with torch.no_grad(), pytest.raises(ValueError, match=match):
        model(**batch, **options)
    with torch.no_grad():
        assert model(**batch, use_cache=False).logits.shape == (2, 5, 37)


@pytest.mark.parametrize("kind", ["dtype", "rows", "zero_k", "negative", "outside"])
def test_invalid_candidate_support_rejected_before_backbone(model, inputs, kind):
    batch, ids = inputs
    if kind == "dtype":
        ids = ids.int()
    elif kind == "rows":
        ids = ids[:, :-1]
    elif kind == "zero_k":
        ids = ids[..., :0]
    else:
        ids = ids.clone()
        ids[0, 0, 0] = -1 if kind == "negative" else model.config.vocab_size
    called = []
    handle = model.model.register_forward_pre_hook(lambda module, args: called.append(True))
    install_memory_bounded_teacher(model)
    with torch.no_grad(), pytest.raises(ValueError):
        model(**batch, teacher_topk_ids=ids, teacher_chunk_rows=2)
    handle.remove()
    assert not called


def test_install_and_scoring_mode_guards(model, inputs):
    batch, ids = inputs
    with pytest.raises(TypeError, match="unwrapped Qwen2"):
        install_memory_bounded_teacher(nn.Linear(2, 2))
    install_memory_bounded_teacher(model)
    with pytest.raises(ValueError, match="already has an instance"):
        install_memory_bounded_teacher(model)
    with pytest.raises(RuntimeError, match="eval mode and torch.no_grad"):
        model(**batch, teacher_topk_ids=ids, teacher_chunk_rows=2)
    model.train()
    with torch.no_grad(), pytest.raises(RuntimeError, match="eval mode and torch.no_grad"):
        model(**batch, teacher_topk_ids=ids, teacher_chunk_rows=2)


def test_dense_and_chunked_teacher_preserve_consumer_loss_and_student_gradient(model, inputs):
    batch, ids = inputs
    with torch.no_grad():
        dense = model(**batch, use_cache=False).logits
        teacher_dense = dense.float().log_softmax(-1).gather(-1, ids)
    install_memory_bounded_teacher(model)
    with torch.no_grad():
        teacher_chunked = model(**batch, teacher_topk_ids=ids, teacher_chunk_rows=3).candidate_logprobs
    torch.manual_seed(27)
    student_base = torch.randn_like(dense)
    token_mask = batch["attention_mask"].bool()
    results = []
    for teacher in (teacher_dense, teacher_chunked):
        student = student_base.clone().requires_grad_()
        selected = student.log_softmax(-1).gather(-1, ids)
        p = (selected - selected.logsumexp(-1, keepdim=True)).clamp_min(-10)
        q = (teacher - teacher.logsumexp(-1, keepdim=True)).clamp_min(-10)
        loss = (p.exp() * (p - q)).sum(-1)[token_mask].mean()
        gradient, = torch.autograd.grad(loss, student)
        results.append((loss.detach(), gradient, teacher.exp().sum(-1)))
    for dense_result, chunked_result in zip(results[0], results[1]):
        torch.testing.assert_close(dense_result, chunked_result, atol=2e-6, rtol=2e-6)


def test_untied_or_multi_rank_models_are_rejected_before_install(model, monkeypatch):
    model.config.tie_word_embeddings = False
    with pytest.raises(ValueError, match="tied input/output embeddings"):
        install_memory_bounded_teacher(model)
    model.config.tie_word_embeddings = True
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="world_size=1"):
        install_memory_bounded_teacher(model)
    assert "forward" not in model.__dict__