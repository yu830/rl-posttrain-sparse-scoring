"""CPU layout and pinned-source patch checks; these never import or run a verl worker."""
import ast
import os
from pathlib import Path

import pytest
import torch

from integrations.verl_6920.bridge import prepare_scoring_layout, select_teacher_execution_path, validate_teacher_candidates
from reference.memory_budget import select_chunk_rows
from integrations.verl_6920.verify_patch import verify_patch


def _nested(rows, dtype=torch.int64):
    return torch.nested.nested_tensor([torch.tensor(row, dtype=dtype) for row in rows], layout=torch.jagged)


def _padded_fixture():
    input_ids = _nested([[10, 11, 12], [20, 21]])
    topk_ids = _nested([[[1, 9], [2, 8], [3, 7]], [[4, 6], [5, 0]]])
    model_inputs = {
        "input_ids": torch.tensor([[10, 11, 12, 0], [0, 20, 0, 21]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [0, 1, 0, 1]]),
    }
    return input_ids, topk_ids, model_inputs


def test_padded_batch_preserves_candidate_order_real_mask_and_return_order():
    input_ids, topk_ids, model_inputs = _padded_fixture()
    layout = prepare_scoring_layout(input_ids, topk_ids, model_inputs, use_remove_padding=False)
    mask = model_inputs["attention_mask"].bool()
    assert layout.candidate_ids.shape == (2, 4, 2)
    torch.testing.assert_close(layout.candidate_ids[mask], topk_ids.values(), atol=0, rtol=0)
    assert not layout.candidate_ids[~mask].any()
    # Simulate compact score values only, not a worker/model execution.
    values = torch.arange(16).reshape(2, 4, 2)
    torch.testing.assert_close(layout.restore_values(values), values[mask], atol=0, rtol=0)
    assert layout.restore_values(values).shape == (5, 2)


def test_single_sequence_rmpad_mapping_and_restore():
    input_ids = _nested([[10, 11, 12]])
    topk_ids = _nested([[[1, 9], [2, 8], [3, 7]]])
    layout = prepare_scoring_layout(
        input_ids, topk_ids, {"input_ids": input_ids.values().unsqueeze(0)}, use_remove_padding=True
    )
    assert layout.valid_mask is None
    torch.testing.assert_close(layout.candidate_ids.squeeze(0), topk_ids.values(), atol=0, rtol=0)
    torch.testing.assert_close(layout.restore_values(layout.candidate_ids), topk_ids.values(), atol=0, rtol=0)


def test_multi_sequence_rmpad_is_rejected_before_model_execution():
    input_ids, topk_ids, _ = _padded_fixture()
    with pytest.raises(NotImplementedError, match="verified varlen attention"):
        prepare_scoring_layout(
            input_ids, topk_ids, {"input_ids": input_ids.values().unsqueeze(0)}, use_remove_padding=True
        )


def test_offsets_must_match_even_when_total_rows_match():
    input_ids, _, model_inputs = _padded_fixture()
    wrong = _nested([[[1, 9], [2, 8]], [[3, 7], [4, 6], [5, 0]]])
    with pytest.raises(ValueError, match="offsets"):
        prepare_scoring_layout(input_ids, wrong, model_inputs, use_remove_padding=False)


@pytest.mark.parametrize("mask", [None, torch.ones(2, 3), torch.ones(2, 4)])
def test_missing_or_inconsistent_attention_mask_is_rejected(mask):
    input_ids, topk_ids, model_inputs = _padded_fixture()
    model_inputs["attention_mask"] = mask
    with pytest.raises(ValueError, match="attention.mask"):
        prepare_scoring_layout(input_ids, topk_ids, model_inputs, use_remove_padding=False)


def test_candidate_dtype_and_restored_shape_are_checked():
    input_ids, topk_ids, model_inputs = _padded_fixture()
    with pytest.raises(ValueError, match="int64"):
        prepare_scoring_layout(input_ids, topk_ids.to(torch.int32), model_inputs, use_remove_padding=False)
    layout = prepare_scoring_layout(input_ids, topk_ids, model_inputs, use_remove_padding=False)
    with pytest.raises(ValueError, match="layout"):
        layout.restore_values(torch.zeros(5, 2))


@pytest.fixture(scope="module")
def checked_patch():
    source = os.environ.get("VERL_6920_SOURCE")
    if not source:
        pytest.skip("set VERL_6920_SOURCE to the pinned checkout for offline patch checks")
    return verify_patch(Path(source))


def _method(source, class_name, method_name):
    module = ast.parse(source)
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def _eval_block(method):
    return next(
        node for node in ast.walk(method)
        if isinstance(node, ast.With) and any("self.engine.eval_mode" in ast.unparse(item.context_expr) for item in node.items)
    )


def test_patch_applies_to_pinned_blobs_and_python_sources_compile(checked_patch):
    assert checked_patch["source_commit"] == "87cd055e872f9fdd7b976fcd797a0934b6dba265"
    assert checked_patch["patch_application"] == "PASS_on_temporary_pinned_blob_copies"
    assert checked_patch["python_syntax"] == "PASS"
    assert checked_patch["worker_runtime"] == "NOT_RUN"


def test_disabled_worker_scoring_branch_is_original_ast_and_output_schema_is_unchanged(checked_patch):
    path = "verl/workers/teacher/fsdp_teacher_worker.py"
    original = _method(checked_patch["original_sources"][path], "TeacherFSDPWorker", "compute_logprobs_at_ids")
    patched = _method(checked_patch["patched_sources"][path], "TeacherFSDPWorker", "compute_logprobs_at_ids")
    old_with, new_with = _eval_block(original), _eval_block(patched)
    assert ast.dump(ast.Tuple(elts=[item.context_expr for item in old_with.items])) == ast.dump(
        ast.Tuple(elts=[item.context_expr for item in new_with.items])
    )
    old_end = next(i for i, node in enumerate(old_with.body) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "cu_seqlens" for t in node.targets))
    enabled = new_with.body[1]
    assert isinstance(enabled, ast.If)
    assert ast.unparse(enabled.test) == "self.engine_config.teacher_memory_bounded"
    assert [ast.dump(n) for n in enabled.orelse] == [ast.dump(n) for n in old_with.body[1:old_end]]
    # Original nested offsets construction and the post-with collection/CPU return remain byte-independent AST matches.
    assert [ast.dump(n) for n in new_with.body[2:]] == [ast.dump(n) for n in old_with.body[old_end:]]
    old_tail, new_tail = original.body.index(old_with), patched.body.index(new_with)
    assert [ast.dump(n) for n in patched.body[new_tail + 1:]] == [ast.dump(n) for n in original.body[old_tail + 1:]]


def test_installation_precedes_fsdp_wrap_and_enabled_worker_keeps_outer_forward(checked_patch):
    path = "verl/workers/engine/fsdp/transformer_impl.py"
    method = _method(checked_patch["patched_sources"][path], "FSDPEngine", "_build_model_optimizer")
    install = next(n for n in ast.walk(method) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == "install_memory_bounded_teacher")
    wrap = next(n for n in ast.walk(method) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "_build_fsdp_module")
    assert install.lineno < wrap.lineno
    worker = _method(checked_patch["patched_sources"]["verl/workers/teacher/fsdp_teacher_worker.py"],
                     "TeacherFSDPWorker", "compute_logprobs_at_ids")
    enabled = _eval_block(worker).body[1]
    forwards = [n for n in ast.walk(enabled) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == "self.engine.module"]
    assert len(forwards) == 3  # enabled compact, enabled dense fallback, and disabled native branch
    compact = next(n for n in forwards if any(k.arg == "teacher_topk_ids" for k in n.keywords))
    assert {k.arg for k in compact.keywords} >= {"teacher_topk_ids", "teacher_chunk_rows", "use_cache"}
    assert not any(isinstance(n, ast.Try) for n in ast.walk(enabled))


def test_config_is_opt_in_and_yaml_fields_reach_existing_trainer_conversion(checked_patch):
    source = checked_patch["patched_sources"]["verl/workers/config/engine.py"]
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "FSDPEngineConfig")
    fields = {n.target.id: ast.literal_eval(n.value) for n in cls.body
              if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
              and n.target.id.startswith("teacher_")}
    assert fields == {"teacher_memory_bounded": False, "teacher_lm_head_chunk_rows": 256,
                      "teacher_head_memory_budget_mib": None}
    yaml = checked_patch["patched_sources"]["verl/trainer/config/distillation/distillation.yaml"]
    for field in fields:
        assert "  " + field + ":" in yaml


@pytest.mark.parametrize("requested_rows", [8, 32])
def test_complete_row_tile_without_budget_uses_observable_dense_fallback(requested_rows):
    decision = select_chunk_rows(8, 64, 4, torch.bfloat16, requested_rows=requested_rows)
    assert select_teacher_execution_path(num_model_rows=8, decision=decision) == "dense_full_row_fallback"


def test_explicit_budget_never_silently_selects_dense_fallback():
    full_tile = select_chunk_rows(8, 64, 4, torch.bfloat16, requested_rows=8, head_memory_budget_mib=1)
    assert full_tile.rows == 8
    assert select_teacher_execution_path(num_model_rows=8, decision=full_tile) == "chunked_lm_head"
    limited = select_chunk_rows(8, 64, 4, torch.bfloat16, requested_rows=3)
    assert select_teacher_execution_path(num_model_rows=8, decision=limited) == "chunked_lm_head"
    with pytest.raises(ValueError, match="one estimated row"):
        select_chunk_rows(8, 64, 4, torch.bfloat16, head_memory_budget_mib=0.0001)


@pytest.mark.parametrize("settings,expected_padding,expected_override", [
    ({}, True, {}),
    ({"use_remove_padding": False, "override_config": {"attn_implementation": "eager"}},
     False, {"attn_implementation": "eager"}),
])
def test_real_trainer_constructor_expression_transports_hf_options_with_original_defaults(
    checked_patch, settings, expected_padding, expected_override
):
    path = "verl/trainer/ppo/ray_trainer.py"
    method = _method(checked_patch["patched_sources"][path], "RayPPOTrainer", "_build_fsdp_teacher_training_config")
    call = next(n for n in ast.walk(method) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "HFModelConfig")

    class Settings(dict):
        def __getattr__(self, key):
            return self[key]

    teacher_yaml = Settings(model_path="existing-local-model", **settings)
    # Evaluate only the actual constructor expression with a recording callable.
    # This is a source transport unit test, not an HF model/worker substitute run.
    result = eval(compile(ast.Expression(call), path, "eval"),
                  {"HFModelConfig": lambda **kwargs: kwargs, "teacher_yaml": teacher_yaml})
    assert result == {"path": "existing-local-model", "use_remove_padding": expected_padding,
                      "override_config": expected_override}
    assert "omega_conf_to_dataclass(fsdp_cfg, dataclass_type=FSDPEngineConfig)" in ast.unparse(method)
    assert "engine_config=engine_config" in ast.unparse(method)
    # The old no-overrides behavior follows the existing HFModelConfig defaults.
    hf_source = checked_patch["original_sources"]["verl/workers/config/model.py"]
    assert "use_remove_padding: bool = True" in hf_source
    assert "override_config: dict = field(default_factory=dict)" in hf_source


@pytest.mark.parametrize("bad_id", [-1, 64])
@pytest.mark.parametrize("budget", [None, 1.0])
def test_invalid_candidate_is_rejected_before_either_execution_path(bad_id, budget):
    candidate_ids = torch.tensor([[[0, bad_id]]])
    decision = select_chunk_rows(1, 64, 2, torch.bfloat16, requested_rows=1,
                                 head_memory_budget_mib=budget)
    path = select_teacher_execution_path(num_model_rows=1, decision=decision)
    assert path == ("dense_full_row_fallback" if budget is None else "chunked_lm_head")
    with pytest.raises(ValueError, match="outside the vocabulary"):
        validate_teacher_candidates(candidate_ids, 64)


def test_patched_worker_validates_candidates_before_fsdp_entry(checked_patch):
    path = "verl/workers/teacher/fsdp_teacher_worker.py"
    worker = _method(checked_patch["patched_sources"][path], "TeacherFSDPWorker", "compute_logprobs_at_ids")
    enabled = _eval_block(worker).body[1]
    checks = [n for n in ast.walk(enabled) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name) and n.func.id == "validate_teacher_candidates"]
    forwards = [n for node in enabled.body for n in ast.walk(node) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == "self.engine.module"]
    assert len(checks) == 1
    assert checks[0].lineno < min(call.lineno for call in forwards)