import pytest
import torch

from reference.scoring import (Readout, causal_positions, conditional_log_probs,
                               conditional_reverse_kl, dense_selected, eager_selected)

# Preregistered budgets from the project evaluation protocol, before running.
TOLERANCES = {
    torch.float64: (1e-10, 1e-9),
    torch.float32: (2e-5, 2e-5),
    torch.float16: (2e-2, 2e-3),
    torch.bfloat16: (1e-1, 1e-2),
}
LOSS_TOLERANCES = {
    torch.float64: (1e-10, 1e-9, 1e-8),
    torch.float32: (5e-5, 2e-5, 1e-4),
    torch.float16: (3e-2, 3e-3, 1e-2),
    torch.bfloat16: (1e-1, 1e-2, 5e-2),
}
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def disable_tf32():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", list(TOLERANCES))
@pytest.mark.parametrize("rows,k", [(1, 1), (4, 3), (7, 8)])
def test_projection_against_fp64_independent_dot(device, dtype, rows, k):
    generator = torch.Generator().manual_seed(121 + rows + k)
    h = torch.randn(rows, 11, generator=generator).to(device=device, dtype=dtype)
    w = torch.randn(17, 11, generator=generator).to(device=device, dtype=dtype)
    bias = torch.randn(17, generator=generator).to(device=device, dtype=dtype)
    ids = torch.stack([torch.randperm(17, generator=generator)[:k] for _ in range(rows)]).to(device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    if rows > 1 and k > 1:
        mask[-1, -1] = False
        ids[-1, -1] = -999
    spec = Readout(scale=0.7, softcap=4.0)
    # Scalar FP64 dot products on the same quantized inputs form the oracle.
    expected = torch.zeros(rows, k, dtype=torch.float64, device=device)
    for r in range(rows):
        for c in range(k):
            if mask[r, c]:
                j = ids[r, c].item()
                raw = sum(h[r, i].double() * w[j, i].double() for i in range(11)) + bias[j].double()
                expected[r, c] = 4.0 * torch.tanh(raw * 0.7 / 4.0)
    atol, rtol = TOLERANCES[dtype]
    for fn in (dense_selected, eager_selected):
        torch.testing.assert_close(fn(h, w, ids, mask, bias, spec).double(), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES)
def test_loss_analytic_gradient_teacher_detached_and_finite_difference(device):
    student = torch.tensor([[0.4, -0.3, 0.8, -2.0], [2.0, -1.0, 0.2, 0.9]], dtype=torch.float64, device=device, requires_grad=True)
    teacher = torch.tensor([[0.1, 0.3, -0.7, 3.0], [-0.1, 2.0, -0.4, 0.5]], dtype=torch.float64, device=device, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool, device=device)
    temp = 0.7
    kl = conditional_reverse_kl(student, teacher, mask, student_temperature=temp, teacher_temperature=1.3)
    grad, teacher_grad = torch.autograd.grad(kl.sum(), (student, teacher), allow_unused=True)
    assert teacher_grad is None
    logp = conditional_log_probs(student.detach(), mask, temp).masked_fill(~mask, 0)
    logq = conditional_log_probs(teacher.detach(), mask, 1.3).masked_fill(~mask, 0)
    analytic = (logp.exp() * (logp - logq - kl.detach()[:, None]) / temp).masked_fill(~mask, 0)
    torch.testing.assert_close(grad, analytic, atol=1e-10, rtol=1e-9)
    eps = 1e-6
    for r, j in [(0, 0), (0, 2), (1, 1)]:
        plus, minus = student.detach().clone(), student.detach().clone()
        plus[r, j] += eps
        minus[r, j] -= eps
        finite = (conditional_reverse_kl(plus, teacher, mask, temp, 1.3).sum() -
                  conditional_reverse_kl(minus, teacher, mask, temp, 1.3).sum()) / (2 * eps)
        torch.testing.assert_close(grad[r, j], finite, atol=1e-8, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_full_vs_conditional_semantic_counterexample(device):
    p = torch.tensor([[0.4, 0.4, 0.2]], dtype=torch.float64, device=device)
    q = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float64, device=device)
    full = (p * (p.log() - q.log())).sum()
    mask = torch.ones((1, 2), dtype=torch.bool, device=device)
    conditional = conditional_reverse_kl(p[:, :2].log(), q[:, :2].log(), mask)
    assert full.item() > 0.8
    torch.testing.assert_close(conditional, torch.zeros_like(conditional), atol=1e-12, rtol=0)
    full_candidate = p[:, :2].log()
    normalized_candidate = conditional_log_probs(full_candidate, mask)
    assert not torch.allclose(full_candidate, normalized_candidate)


@pytest.mark.parametrize("device", DEVICES)
def test_k_one_zero_and_k_vocab_matches_full(device):
    s = torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float64, device=device, requires_grad=True)
    t = torch.tensor([[0.2, 0.7, -0.1]], dtype=torch.float64, device=device)
    singleton = conditional_reverse_kl(s[:, :1], t[:, :1], torch.ones((1, 1), dtype=torch.bool, device=device))
    assert singleton.item() == 0.0
    assert torch.autograd.grad(singleton.sum(), s, retain_graph=True)[0].abs().max().item() == 0.0
    all_ids = conditional_reverse_kl(s, t, torch.ones_like(s, dtype=torch.bool))
    sp, tp = torch.log_softmax(s, -1), torch.log_softmax(t, -1)
    torch.testing.assert_close(all_ids, (sp.exp() * (sp - tp)).sum(-1), atol=1e-12, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_padding_and_permutations(device):
    h, w = torch.tensor([[1., 2.], [-1., 3.]], device=device), torch.arange(10., device=device).view(5, 2)
    ids = torch.tensor([[1, 4, -99], [3, 0, 2]], dtype=torch.long, device=device)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool, device=device)
    selected = eager_selected(h, w, ids, mask)
    permutation = torch.tensor([1, 0, 2], device=device)
    torch.testing.assert_close(eager_selected(h, w, ids[:, permutation], mask[:, permutation]), selected[:, permutation])
    probs = conditional_log_probs(selected, mask).exp().masked_fill(~mask, 0)
    torch.testing.assert_close(probs.sum(-1), torch.ones(2, device=device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ids,mask", [([[1, 1]], [[1, 1]]), ([[-1, 1]], [[1, 1]]), ([[1, 5]], [[1, 1]]), ([[1, 2]], [[0, 0]])])
def test_reject_invalid_support(device, ids, mask):
    with pytest.raises(ValueError):
        eager_selected(torch.ones(1, 2, device=device), torch.ones(5, 2, device=device),
                       torch.tensor(ids, dtype=torch.long, device=device), torch.tensor(mask, dtype=torch.bool, device=device))


def test_causal_alignment_and_unsupported_features():
    torch.testing.assert_close(causal_positions(torch.tensor([1, 4, 7]), 8), torch.tensor([0, 3, 6]))
    for t in (torch.tensor([0]), torch.tensor([8])):
        with pytest.raises(ValueError):
            causal_positions(t, 8)
    h, w, ids, mask = torch.ones(1, 2), torch.ones(4, 2), torch.tensor([[0]]), torch.tensor([[True]])
    for spec in (Readout(adapter=True), Readout(quantized_weights=True)):
        with pytest.raises(NotImplementedError):
            dense_selected(h, w, ids, mask, spec=spec)


@pytest.mark.parametrize("device", DEVICES)
def test_dense_and_selected_hidden_weight_bias_gradients(device):
    gen = torch.Generator().manual_seed(301)
    h0 = torch.randn(3, 5, generator=gen, dtype=torch.float64).to(device)
    w0 = torch.randn(9, 5, generator=gen, dtype=torch.float64).to(device)
    b0 = torch.randn(9, generator=gen, dtype=torch.float64).to(device)
    teacher = torch.randn(3, 3, generator=gen, dtype=torch.float64).to(device)
    ids = torch.tensor([[1, 4, 6], [2, 4, 8], [4, 3, -1]], dtype=torch.long, device=device)
    mask = ids >= 0  # ID 4 intentionally occurs in more than one row.
    observed = []
    for implementation in (dense_selected, eager_selected):
        h, w, b = (x.detach().clone().requires_grad_(True) for x in (h0, w0, b0))
        logits = implementation(h, w, ids, mask, b, Readout(scale=0.7, softcap=4.0))
        loss = conditional_reverse_kl(logits, teacher, mask).sum()
        grads = torch.autograd.grad(loss, (h, w, b))
        observed.append((logits.detach(), loss.detach(), *(g.detach() for g in grads)))
    for dense_value, selected_value in zip(*observed):
        torch.testing.assert_close(dense_value, selected_value, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", list(LOSS_TOLERANCES))
def test_low_precision_conditional_loss_and_student_gradient(device, dtype):
    generator = torch.Generator().manual_seed(302)
    s = torch.randn(4, 6, generator=generator).to(device=device, dtype=dtype).requires_grad_(True)
    t = torch.randn(4, 6, generator=generator).to(device=device, dtype=dtype).requires_grad_(True)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool, device=device)
    s_ref = s.detach().double().requires_grad_(True)
    t_ref = t.detach().double()
    # FP64 normalizer and loss operate on exactly the same quantized values.
    logp_ref = torch.log_softmax((s_ref / 0.8).masked_fill(~mask, -torch.inf), dim=-1)
    logq_ref = torch.log_softmax((t_ref / 1.2).masked_fill(~mask, -torch.inf), dim=-1)
    lp_safe = logp_ref.masked_fill(~mask, 0)
    lq_safe = logq_ref.masked_fill(~mask, 0)
    ref = (lp_safe.exp() * (lp_safe - lq_safe)).masked_fill(~mask, 0).sum(-1)
    ref_grad = torch.autograd.grad(ref.sum(), s_ref)[0]
    actual_lp = conditional_log_probs(s, mask, 0.8)
    actual = conditional_reverse_kl(s, t, mask, 0.8, 1.2)
    grad, teacher_grad = torch.autograd.grad(actual.sum(), (s, t), allow_unused=True)
    atol, rtol, grad_l2 = LOSS_TOLERANCES[dtype]
    torch.testing.assert_close(actual_lp[mask].double(), logp_ref.detach()[mask], atol=atol, rtol=rtol)
    torch.testing.assert_close(actual.double(), ref.detach(), atol=atol, rtol=rtol)
    relative_l2 = (grad.double() - ref_grad).norm() / ref_grad.norm()
    assert relative_l2.item() <= grad_l2
    assert teacher_grad is None


def test_reject_nonfinite_readout_and_temperature():
    h, w, ids, mask = torch.ones(1, 2), torch.ones(4, 2), torch.tensor([[0]]), torch.tensor([[True]])
    for spec in (Readout(scale=float("inf")), Readout(softcap=float("nan"))):
        with pytest.raises(ValueError):
            dense_selected(h, w, ids, mask, spec=spec)
    with pytest.raises(ValueError):
        conditional_log_probs(torch.ones(1, 1), mask, temperature=float("inf"))


@pytest.mark.parametrize("device", DEVICES)
def test_post_normalization_clamp_is_a_distinct_loss(device):
    # verl PR #6920 supports clamping after normalization; this reference does not.
    student = torch.tensor([[0.0, -30.0]], dtype=torch.float64, device=device)
    teacher = torch.tensor([[-30.0, 0.0]], dtype=torch.float64, device=device)
    mask = torch.ones_like(student, dtype=torch.bool)
    unclamped = conditional_reverse_kl(student, teacher, mask)
    logp = conditional_log_probs(student, mask).clamp_min(-10.0)
    logq = conditional_log_probs(teacher, mask).clamp_min(-10.0)
    clamped = (logp.exp() * (logp - logq)).sum(-1)
    assert abs((unclamped - clamped).item()) > 1.0
