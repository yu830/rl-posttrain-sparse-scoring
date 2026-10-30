import pytest

from scripts.verl_teacher_worker_smoke import compact_output_signature


def test_worker_signature_handles_checkpoint_and_final_output_shapes():
    fields = ("teacher_on_student_logp", "teacher_topk_ids")
    snapshot = {name: {"dtype": "torch.int64", "device": "cpu", "offsets": [0, 2, 5],
                       "shape": [5, 8], "values_sha256": name, "values": [[1]]} for name in fields}
    final = {name: {"dtype": value["dtype"], "device": value["device"], "offsets": value["offsets"],
                    "values_shape": value["shape"], "values_sha256": value["values_sha256"]}
             for name, value in snapshot.items()}
    assert compact_output_signature(snapshot) == compact_output_signature(final)
    final["teacher_topk_ids"]["offsets"] = [0, 3, 5]
    assert compact_output_signature(snapshot) != compact_output_signature(final)


def test_worker_signature_rejects_missing_fields():
    with pytest.raises(ValueError, match="signature"):
        compact_output_signature({})
