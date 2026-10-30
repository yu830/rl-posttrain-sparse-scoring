import pytest
import torch

from scripts.long_worker_probe import build_fixture, fixture_identity, output_artifact, parse_args


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [3, 4, 5]


def test_fixture_preserves_padding_positions_and_ordered_duplicates():
    fixture = build_fixture(Tokenizer(), [4, 2], 3, 19)
    assert fixture['attention_mask'].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert fixture['position_ids'].tolist() == [[0, 1, 2, 3], [0, 1, 0, 0]]
    assert fixture['topk_ids'][1, 2:].eq(0).all()
    assert torch.equal(fixture['topk_ids'][..., 0], fixture['topk_ids'][..., -1])
    identity = fixture_identity(fixture, [4, 2], 3)
    assert identity['valid_rows'] == 6 and identity['padded_rows'] == 8


def make_output(logp=None, offsets=None):
    offsets = torch.tensor([0, 2, 3] if offsets is None else offsets)
    logp = torch.tensor([[-1., -2.], [-3., -4.], [-5., -6.]], dtype=torch.bfloat16) if logp is None else logp
    return {'teacher_on_student_logp': torch.nested.nested_tensor_from_jagged(logp, offsets),
            'teacher_topk_ids': torch.nested.nested_tensor_from_jagged(torch.tensor([[1, 2], [2, 3], [4, 5]]), offsets)}


def identity():
    return {'fixture': {'lengths': [2, 1], 'k': 2}, 'vocab_size': 10}


def test_full_output_and_bad_offsets_rejection():
    artifact = output_artifact(make_output(), identity())
    assert artifact['output']['offsets'] == [0, 2, 3]
    assert artifact['candidate_logprobs'].numel() == 6
    with pytest.raises(ValueError, match='contract'):
        output_artifact(make_output(offsets=[0, 1, 3]), identity())


def test_nonfinite_and_duplicate_topk_rejected():
    output = make_output()
    output['teacher_on_student_logp'].values()[1, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        output_artifact(output, identity())
    output = make_output()
    output['teacher_topk_ids'].values()[1, 1] = 2
    with pytest.raises(ValueError, match='not unique'):
        output_artifact(output, identity())


def test_no_fraction_default_and_invalid_fraction_rejection(tmp_path):
    base = ['--verl-dir', str(tmp_path), '--model-dir', str(tmp_path), '--out', str(tmp_path/'result.json'), '--mode', 'dense']
    assert parse_args(base).allocator_fraction is None
    with pytest.raises(SystemExit):
        parse_args(base + ['--allocator-fraction', 'nan'])


def test_sampler_failure_preserves_worker_result_but_fails_experiment(tmp_path, monkeypatch):
    import json
    from scripts import long_worker_probe as probe

    class FailingSampler:
        def __init__(self, *args):
            pass

        def start(self):
            pass

        def stop(self):
            return {'failure': {'status': 'UNAVAILABLE'}, 'sample_count': 0}

    def successful_worker(args, result, journal):
        result['status'] = 'PASS_WORKER_CONTRACT_REPEAT'
        result['all_calls'] = [{'output': 'preserved'}]

    monkeypatch.setattr(probe, 'DeviceSampler', FailingSampler)
    monkeypatch.setattr(probe, 'run', successful_worker)
    output = tmp_path / 'result.json'
    status = probe.main(['--verl-dir', str(tmp_path), '--model-dir', str(tmp_path),
                         '--out', str(output), '--mode', 'dense'])
    result = json.loads(output.read_text())
    assert status == 1
    assert result['status'] == 'ERROR_DEVICE_SAMPLER'
    assert result['worker_status_before_sampler_check'] == 'PASS_WORKER_CONTRACT_REPEAT'
    assert result['all_calls'] == [{'output': 'preserved'}]
