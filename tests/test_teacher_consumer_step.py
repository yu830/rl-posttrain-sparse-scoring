import torch

from scripts.teacher_consumer_step import compare_gradient, select_student_candidates


def test_frozen_fp32_gradient_relative_l2_gate_rejects_small_relative_drift():
    reference = torch.tensor([0.001, 0.002], dtype=torch.float32)
    candidate = reference + torch.tensor([1e-6, 0.0], dtype=torch.float32)
    fp32 = compare_gradient(reference, candidate, "float32")
    bf16 = compare_gradient(reference, candidate, "bfloat16")
    assert fp32["max_abs"] < fp32["atol"]
    assert fp32["relative_l2"] > fp32["relative_l2_max"] == 1e-4
    assert not fp32["pass"]
    assert bf16["relative_l2"] < bf16["relative_l2_max"] == 0.05
    assert bf16["pass"]

def test_student_candidates_follow_rounded_bf16_full_log_softmax_topk():
    # BF16 normalization can create ties that change the selected support.
    logits = torch.linspace(0, -0.015, 64).to(torch.bfloat16)
    expected = logits.log_softmax(-1).topk(8, dim=-1).indices
    direct_logits = logits.topk(8, dim=-1).indices
    assert not torch.equal(expected.sort().values, direct_logits.sort().values)
    torch.testing.assert_close(select_student_candidates(logits, 8), expected, atol=0, rtol=0)