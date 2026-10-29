"""Discriminating contract checks, separate from real-model GPU replay."""
import pytest
import torch

from scripts.teacher_stage_benchmark import (
    TOLERANCES, causal_positions, consumer_loss, selected_full_logprobs,
    teacher_stage, topk_tie_valid, overlap_diagnostics, scale_student_logits_,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n,k,row_chunk,lse_chunk", [(7, 3, 2, 4), (1, 1, 8, 2), (9, 17, 4, 3)])
def test_stage_matches_independent_fp64_oracle(dtype, n, k, row_chunk, lse_chunk):
    generator = torch.Generator().manual_seed(611)
    # Same quantized inputs; independent FP64 scalar dot products and LSE.
    h = torch.randn(n, 11, generator=generator).to(dtype)
    w = torch.randn(17, 11, generator=generator).to(dtype)
    ids = torch.stack([torch.randperm(17, generator=generator)[:k] for _ in range(n)])
    logits64 = torch.stack([torch.stack([sum(h[r, d].double() * w[v, d].double() for d in range(11))
                                        for v in range(17)]) for r in range(n)])
    expected = logits64.log_softmax(-1).gather(-1, ids)
    atol, rtol = TOLERANCES[dtype]
    outputs = [teacher_stage(h, w, ids, topk=k, row_chunk=c, lse_chunk=lse_chunk)
               for c in (None, row_chunk)]
    for output in outputs:
        assert output.candidate_logprobs.dtype == dtype
        torch.testing.assert_close(output.candidate_logprobs.double(), expected, atol=atol, rtol=rtol)
        # Top-K correctness against the same rounded-logit head contract.
        assert topk_tie_valid(h @ w.T, output.teacher_topk_ids, k).all()
    torch.testing.assert_close(outputs[0].candidate_logprobs, outputs[1].candidate_logprobs, atol=atol, rtol=rtol)


def test_full_normalization_retains_omitted_tail_and_duplicate_gather():
    logits = torch.tensor([[0., 0., 3.]])
    ids = torch.tensor([[0, 1, 0]])  # utility preserves gather duplicates
    actual = selected_full_logprobs(logits, ids, 1)
    expected = logits.log_softmax(-1).gather(-1, ids)
    torch.testing.assert_close(actual, expected)
    assert actual.exp().sum() < .2
    assert actual[0, 0] == actual[0, 2]
    # Conditional-only normalization would incorrectly sum to one.
    assert not torch.allclose(actual, logits.gather(-1, ids).log_softmax(-1))


def test_lse_gather_cast_occurs_after_fp32_normalization():
    logits = torch.tensor([[4.03125, -3.71875, 0.125]], dtype=torch.bfloat16)
    ids = torch.tensor([[1, 0]])
    expected = (logits.float().gather(-1, ids) - logits.float().logsumexp(-1, keepdim=True)).bfloat16()
    actual = selected_full_logprobs(logits, ids, 1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.dtype == torch.bfloat16


def test_topk_ties_are_sets_and_nearby_values_are_not_ties():
    logits = torch.tensor([[4., 3., 3., 1.], [4., 3., 3.00001, 1.]])
    assert topk_tie_valid(logits[:1], torch.tensor([[0, 2]]), 2).item()
    assert not topk_tie_valid(logits[:1], torch.tensor([[2, 0]]), 2).item()  # wrong descending order
    assert not topk_tie_valid(logits[:1], torch.tensor([[1, 2]]), 2).item()  # missing strict maximum
    assert not topk_tie_valid(logits[:1], torch.tensor([[0, 0]]), 2).item()
    assert not topk_tie_valid(logits[1:], torch.tensor([[0, 1]]), 2).item()
    assert topk_tie_valid(logits[1:], torch.tensor([[0, 2]]), 2).item()


def test_loss_gradient_teacher_detach_masks_and_analytic_oracle():
    s = torch.tensor([[-2., -3., -8.], [-4., -1., -7.], [-9., -5., -6.]], requires_grad=True)
    t = torch.tensor([[-5., -2., -6.], [-3., -2., -7.], [-4., -2., -3.]], requires_grad=True)
    cmask = torch.tensor([[True, True, False], [True, True, True], [True, False, True]])
    tmask = torch.tensor([True, False, True])
    loss, mass = consumer_loss(s, t, cmask, tmask)
    grad, tg = torch.autograd.grad(loss, (s, t), allow_unused=True)
    # FP64 independent categorical calculation over exactly the active IDs.
    sr = s.detach().double().requires_grad_(True)
    tr = t.detach().double()
    rows = []
    for r in (0, 2):
        ps = sr[r, cmask[r]].softmax(-1)
        qt = tr[r, cmask[r]].softmax(-1)
        rows.append((ps * (ps.log() - qt.log())).sum())
    oracle = torch.stack(rows).mean()
    expected_grad = torch.autograd.grad(oracle, sr)[0]
    torch.testing.assert_close(loss.double(), oracle, atol=5e-5, rtol=2e-5)
    torch.testing.assert_close(grad.double(), expected_grad, atol=5e-5, rtol=2e-5)
    torch.testing.assert_close(mass, t.detach().exp().masked_fill(~cmask, 0).sum(-1))
    assert tg is None and grad[1].count_nonzero() == 0 and grad[~cmask].count_nonzero() == 0
    # Excluding a token must change both aggregation and gradient.
    all_loss, _ = consumer_loss(s, t, cmask, torch.ones_like(tmask))
    assert abs((all_loss - loss).item()) > .01


def test_post_normalization_clamp_changes_loss_and_is_exercised():
    s = torch.tensor([[0., -30.]], requires_grad=True)
    t = torch.tensor([[-30., 0.]])
    cmask = torch.ones_like(s, dtype=torch.bool)
    tmask = torch.tensor([True])
    plain, _ = consumer_loss(s, t, cmask, tmask)
    clamped, _ = consumer_loss(s, t, cmask, tmask, clamp_min=-10)
    assert abs((plain - clamped).item()) > 1
    lp, lq = s.log_softmax(-1).clamp_min(-10), t.log_softmax(-1).clamp_min(-10)
    torch.testing.assert_close(clamped, (lp.exp() * (lp - lq)).sum())


def test_causal_positions_excludes_current_token_and_masks_do_not_shift():
    token_positions = torch.tensor([1, 3, 5])
    positions = causal_positions(token_positions, 6)
    # Distinguishable hidden values make an off-by-one observable.
    hidden = torch.arange(6.).reshape(6, 1)
    torch.testing.assert_close(hidden[positions].flatten(), torch.tensor([0., 2., 4.]))
    assert not torch.equal(hidden[positions], hidden[token_positions])
    with pytest.raises(ValueError):
        causal_positions(torch.tensor([0]), 6)


def test_invalid_chunks_rejected_without_changing_support_contract():
    with pytest.raises(ValueError):
        selected_full_logprobs(torch.ones(2, 3), torch.tensor([[0], [1]]), 0)
    with pytest.raises(ValueError):
        teacher_stage(torch.ones(2, 3), torch.ones(4, 3), torch.tensor([[0], [1]]), topk=1, row_chunk=0)


def test_complete_student_logits_backward_and_teacher_stop_gradient():
    student = torch.tensor([[.3, -.9, 1.7, -3.], [2., -1., .5, .2]], requires_grad=True)
    teacher = torch.tensor([[-2., -5.], [-4., -1.]], requires_grad=True)
    ids = torch.tensor([[2, 0], [0, 3]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    token_mask = torch.tensor([True, True])
    selected = student.log_softmax(-1).gather(-1, ids)
    loss, _ = consumer_loss(selected, teacher, mask, token_mask, clamp_min=-10)
    grad, tg = torch.autograd.grad(loss, (student, teacher), allow_unused=True)
    sr = student.detach().double().requires_grad_(True)
    # Independent conditional-on-selected-logits oracle; the full-vocabulary
    # normalizer must cancel before the post-normalization clamp.
    lp = sr.gather(-1, ids).log_softmax(-1).clamp_min(-10)
    lq = teacher.detach().double().log_softmax(-1).clamp_min(-10)
    oracle = (lp.exp() * (lp - lq)).sum(-1).mean()
    expected = torch.autograd.grad(oracle, sr)[0]
    torch.testing.assert_close(loss.double(), oracle, atol=5e-5, rtol=2e-5)
    torch.testing.assert_close(grad.double(), expected, atol=5e-5, rtol=2e-5)
    assert tg is None
    assert grad.abs().sum() > .01
    omitted = torch.ones_like(student, dtype=torch.bool).scatter(-1, ids, False)
    assert grad[omitted].abs().max() < 1e-6


def test_overlap_diagnostics_and_wrapper_clamp_are_distinct():
    student = torch.tensor([[0., -30.], [-2., -1.]])
    teacher = torch.tensor([[-30., 0.], [-3., -2.]])
    ids = torch.tensor([[0, 2], [1, 3]])
    teacher_ids = torch.tensor([[0, 1], [0, 2]])
    count, advantage = overlap_diagnostics(student, teacher, ids, teacher_ids)
    torch.testing.assert_close(count, torch.tensor([1, 0]))
    assert advantage[0] < -9 and advantage[1] == 0
    cmask, tmask = torch.ones_like(student, dtype=torch.bool), torch.tensor([True, True])
    loss, _ = consumer_loss(student, teacher, cmask, tmask)
    bounded, _ = consumer_loss(student, teacher, cmask, tmask, loss_max_clamp=10)
    assert loss > 10 and bounded <= 10


def test_bf16_student_temperature_is_cast_before_division():
    logits = torch.tensor([[0.4296875, -0.30078125, 0.76953125]], dtype=torch.bfloat16)
    temperature = torch.tensor([.8], dtype=torch.float32)
    actual = scale_student_logits_(logits.clone(), temperature)
    expected = logits.clone().div_(temperature.clamp(min=1e-8).unsqueeze(-1).to(logits.dtype))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual[0, 0].item() == .53515625
    assert (logits / .8)[0, 0].item() == .5390625
    assert not torch.equal(actual, logits / .8)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_student_temperature_full_path_matches_pinned_engine_gradient(dtype):
    original = torch.tensor([[.4296875, -.30078125, .76953125, -1.203125],
                             [.625, -1.125, .30078125, -.4296875]], dtype=dtype)
    temperature = torch.tensor([.8, 1.3], dtype=torch.float32)
    ids = torch.tensor([[2, 0, 1], [0, 2, 3]])
    teacher = torch.tensor([[-2., -3., -1.], [-4., -1., -2.]])
    observed = []
    for use_helper in (False, True):
        leaf = original.detach().clone().requires_grad_(True)
        logits = leaf * 1  # non-leaf output, as from the real model head
        if use_helper:
            scale_student_logits_(logits, temperature)
        else:
            # Literal pinned engine expression is the independent reference.
            logits.div_(temperature.clamp(min=1e-8).unsqueeze(-1).to(logits.dtype))
        selected = logits.log_softmax(-1).gather(-1, ids)
        loss, _ = consumer_loss(selected, teacher, torch.ones_like(ids, dtype=torch.bool),
                                torch.tensor([True, True]), clamp_min=-10, loss_max_clamp=10)
        gradient = torch.autograd.grad(loss, leaf)[0]
        observed.append((logits.detach(), loss.detach(), gradient))
    for expected, actual in zip(*observed):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert observed[1][2].abs().sum() > .01


def test_student_temperature_clamps_before_low_precision_cast():
    logits = torch.tensor([[1., -1.], [1., -1.]])
    temperatures = torch.tensor([0., -1.])
    actual = scale_student_logits_(logits, temperatures)
    torch.testing.assert_close(actual, torch.tensor([[1e8, -1e8], [1e8, -1e8]]), atol=0, rtol=0)
