#!/usr/bin/env python3
"""Real pinned verl TeacherFSDPWorker with long padded fixtures and durable evidence.

One fresh process per mode. No allocator fraction is installed unless explicitly
requested. Complete CPU outputs are saved per call, including warmups. This is
not Ray dispatch, multi-GPU sharding, RL training, or a production timing claim.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.verl_teacher_worker_smoke import (
    PINNED_VERL, MODEL_REVISION, MODEL_SHA256, TOKENIZER_SHA256, git_head, sha256,
)
from scripts.factorial_teacher_probe import (
    DeviceSampler, JsonlJournal, cuda_memory, safe_cuda_memory, describe_actual_fsdp_policies,
    json_safe, tensor_hash, validate_head_calls, summarize_samples,
)
from scripts.full_teacher_benchmark import is_out_of_memory


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verl-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=('dense', 'chunked'), required=True)
    p.add_argument('--lengths', type=int, nargs='+', default=[8192])
    p.add_argument('--k', type=int, default=32)
    p.add_argument('--chunk-rows', type=int, default=256)
    p.add_argument('--gather-chunk-rows', type=int, default=1024)
    p.add_argument('--allocator-fraction', type=float)
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--sample-device-ms', type=float, default=100)
    a = p.parse_args(argv)
    if min([*a.lengths, a.k, a.chunk_rows, a.gather_chunk_rows, a.repeats]) <= 0 or a.warmup < 0:
        p.error('lengths/K/chunks/repeats must be positive; warmup nonnegative')
    if a.allocator_fraction is not None and (not math.isfinite(a.allocator_fraction) or not 0 < a.allocator_fraction <= 1):
        p.error('allocator fraction must be finite in (0,1]')
    if not math.isfinite(a.sample_device_ms) or a.sample_device_ms < 0:
        p.error('sample interval must be finite and nonnegative')
    if a.mode == 'chunked' and a.chunk_rows >= len(a.lengths) * max(a.lengths):
        p.error('chunked mode must tile fewer than the padded model rows')
    for key in ('verl_dir', 'model_dir', 'out'):
        setattr(a, key, getattr(a, key).resolve())
    if a.out.is_relative_to(ROOT) or a.out.exists():
        p.error('out must be a new path outside the source checkout')
    return a


def build_fixture(tokenizer, lengths, k, vocabulary):
    width = max(lengths)
    tokens = torch.zeros((len(lengths), width), dtype=torch.int64)
    mask = torch.zeros_like(tokens)
    positions = torch.zeros_like(tokens)
    candidates = torch.zeros((*tokens.shape, k), dtype=torch.int64)
    texts = (
        'A frozen teacher scores every candidate at the original causal position. Memory measurements include the full model forward. ',
        'Different sequence lengths require the real attention mask. Candidate order and output offsets must remain aligned. ',
    )
    for row, length in enumerate(lengths):
        base = tokenizer.encode(texts[row % len(texts)], add_special_tokens=False)
        encoded = (base * math.ceil(length / len(base)))[:length]
        tokens[row, :length] = torch.tensor(encoded)
        mask[row, :length] = 1
        positions[row, :length] = torch.arange(length)
        ids = (tokens[row, :length, None] + 131 * torch.arange(length)[:, None]
               + 17 * row + 9973 * torch.arange(k)[None, :]) % vocabulary
        if k > 1:
            ids[:, -1] = ids[:, 0]
        candidates[row, :length] = ids
    return {'input_ids': tokens, 'attention_mask': mask, 'position_ids': positions, 'topk_ids': candidates}


def fixture_identity(fixture, lengths, k):
    return {'lengths': list(lengths), 'k': k, 'padded_rows': len(lengths) * max(lengths),
            'valid_rows': sum(lengths), 'pad_token_id': 0,
            'kind': 'deterministic synthetic tokenized text; duplicate ordered candidate; not RL trajectories',
            **{key + '_sha256': tensor_hash(value) for key, value in fixture.items()}}


def output_artifact(output, identity):
    if output is None or set(output.keys()) != {'teacher_on_student_logp', 'teacher_topk_ids'}:
        raise ValueError('unexpected Worker output keys or missing collector output')
    lengths, k = identity['fixture']['lengths'], identity['fixture']['k']
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    result = {'identity': identity}
    signature = {'values_shape': [sum(lengths), k], 'padded_shape': [len(lengths), max(lengths), k], 'offsets': offsets}
    for field, key, dtype in [('teacher_on_student_logp', 'candidate_logprobs', torch.bfloat16),
                              ('teacher_topk_ids', 'teacher_topk_ids', torch.int64)]:
        nested = output[field]
        value = nested.values().detach().contiguous()
        if (nested.values().requires_grad or value.device.type != 'cpu' or value.dtype != dtype
                or list(value.shape) != [sum(lengths), k] or nested.offsets().cpu().tolist() != offsets):
            raise ValueError('Worker values violate shape/dtype/CPU/no-grad/offset contract')
        if key == 'candidate_logprobs' and not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite candidate logprob')
        if key == 'teacher_topk_ids':
            if not bool(((value >= 0) & (value < identity['vocab_size'])).all()):
                raise ValueError('Top-K ID outside vocabulary')
            if k > 1 and not bool(value.sort(-1).values.diff(dim=-1).ne(0).all()):
                raise ValueError('Top-K IDs are not unique')
        result[key] = value
        signature[key] = {'dtype': str(value.dtype), 'sha256': tensor_hash(value)}
    result['output'] = signature
    return result


@contextmanager
def observe_worker(module, fixture, journal):
    events, handles = [], []
    for name, target in [('outer', module), ('backbone', module.model), ('head', module.get_output_embeddings())]:
        def pre(m, args, kwargs, label=name):
            value = args[0] if args else kwargs.get('input_ids', kwargs.get('inputs_embeds'))
            event = {'module': label, 'event': 'pre', 'input_shape': list(value.shape) if value is not None else None}
            if label == 'outer':
                event['actual_inputs'] = {}
                for field in ('input_ids', 'attention_mask', 'position_ids'):
                    actual = kwargs.get(field)
                    if actual is None:
                        raise ValueError(f'actual Worker forward omitted {field}')
                    actual = actual.detach().cpu()
                    equal = torch.equal(actual, fixture[field])
                    event['actual_inputs'][field] = {'sha256': tensor_hash(actual), 'shape': list(actual.shape),
                                                      'equals_fixture': equal}
                    if not equal:
                        raise ValueError(f'actual Worker forward {field} differs from fixture')
            events.append(event)
            journal.record('forward_hook', hook=event)
        def post(m, args, kwargs, output, label=name):
            event = {'module': label, 'event': 'post'}
            if label == 'head':
                event['output_shape'] = list(output.shape)
            events.append(event)
            journal.record('forward_hook', hook=event)
        handles.extend((target.register_forward_pre_hook(pre, with_kwargs=True),
                        target.register_forward_hook(post, with_kwargs=True)))
    try:
        yield events
    finally:
        for handle in handles:
            handle.remove()


def run(args, result, journal):
    if git_head(args.verl_dir) != PINNED_VERL:
        raise ValueError('verl checkout differs from fixed commit')
    hashes = {name: sha256(args.model_dir / name) for name in ('model.safetensors', 'tokenizer.json', 'config.json')}
    if hashes['model.safetensors'] != MODEL_SHA256 or hashes['tokenizer.json'] != TOKENIZER_SHA256:
        raise ValueError('model/tokenizer snapshot differs from fixed artifacts')
    sys.path.insert(0, str(args.verl_dir))
    import ray
    from tensordict import TensorDict
    from verl.trainer.config import CheckpointConfig
    from verl.utils import tensordict_utils as tu
    from verl.utils.profiler import ProfilerConfig
    from verl.workers.config import FSDPEngineConfig, HFModelConfig, TrainingWorkerConfig
    worker_module = importlib.import_module('verl.workers.teacher.fsdp_teacher_worker')
    Worker = worker_module.TeacherFSDPWorker
    source = Path(inspect.getfile(Worker)).resolve()
    if not source.is_relative_to(args.verl_dir):
        raise ValueError('Worker imported from unexpected checkout')
    if torch.distributed.is_initialized() or ray.is_initialized():
        raise RuntimeError('requires fresh direct process, no existing distributed/Ray runtime')
    if any(k.startswith('RAY_EXPERIMENTAL_NOSET_') and v for k, v in os.environ.items()):
        raise ValueError('direct Worker cannot use Ray actor-only NOSET flags')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required, no CPU substitution')
    torch.cuda.set_device(0)
    if args.allocator_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.allocator_fraction, 0)
    result['allocator_fraction_installed'] = args.allocator_fraction is not None
    result['actual_allocator_fraction'] = torch.cuda.get_per_process_memory_fraction(0)
    result['selected_environment'] = {key: os.environ.get(key) for key in (
        'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF', 'CUDA_LAUNCH_BLOCKING', 'CUDA_VISIBLE_DEVICES', 'TMPDIR')}
    torch.manual_seed(731)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result['versions'] = {'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                          'transformers': importlib.import_module('transformers').__version__, 'python': sys.version}
    result['device'] = {'name': torch.cuda.get_device_name(), 'memory_before_load': cuda_memory()}
    result['sources'] = {'loaded_worker': str(source), 'worker_sha256': sha256(source), 'verl_head': PINNED_VERL,
                         'engine_sha256': sha256(args.verl_dir / 'verl/workers/engine/fsdp/transformer_impl.py')}
    model_config = HFModelConfig(path=str(args.model_dir), use_remove_padding=False,
        enable_gradient_checkpointing=False, use_fused_kernels=False,
        override_config={'attn_implementation': 'sdpa', 'use_cache': False})
    vocabulary = model_config.hf_config.vocab_size
    if args.k > vocabulary:
        raise ValueError('K exceeds actual vocabulary')
    fixture = build_fixture(model_config.tokenizer, args.lengths, args.k, vocabulary)
    result['identity'] = {'model_revision': MODEL_REVISION, 'artifact_hashes': hashes, 'dtype': 'bfloat16',
        'attention': 'sdpa', 'buffer_cast': 'worker', 'vocab_size': vocabulary,
        'worker_sha256': sha256(source), 'fixture': fixture_identity(fixture, args.lengths, args.k)}
    fixture_path = args.out.with_suffix('.fixture.pt')
    with fixture_path.open('xb') as stream:
        torch.save({'identity': result['identity'], **fixture}, stream)
    result['fixture_artifact'] = {'path': str(fixture_path), 'sha256': sha256(fixture_path)}
    rows = len(args.lengths) * max(args.lengths)
    engine_config = FSDPEngineConfig(strategy='fsdp2', forward_only=True, model_dtype='bfloat16', dtype='bfloat16',
        mixed_precision={'param_dtype': 'bfloat16', 'reduce_dtype': 'float32', 'buffer_dtype': 'float32'},
        fsdp_size=1, ulysses_sequence_parallel_size=1, use_torch_compile=False, use_dynamic_bsz=False,
        infer_micro_batch_size_per_gpu=len(args.lengths), teacher_memory_bounded=True,
        teacher_lm_head_chunk_rows=rows if args.mode == 'dense' else args.chunk_rows,
        teacher_head_memory_budget_mib=None)
    config = TrainingWorkerConfig(model_type='language_model', model_config=model_config, engine_config=engine_config,
        optimizer_config=None, checkpoint_config=CheckpointConfig(),
        profiler_config=ProfilerConfig(tool=None, enable=False, save_path=None, tool_config={}))
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    os.environ.update(WORLD_SIZE='1', RANK='0', LOCAL_WORLD_SIZE='1', LOCAL_RANK='0',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), DIST_INIT_METHOD='env://')
    result['stage'] = 'constructor_and_reset'
    journal.record('worker_reset_begin')
    worker = Worker(config)
    worker.reset()
    result['actual_engine_class'] = f'{type(worker.engine).__module__}.{type(worker.engine).__name__}'
    result['actual_attention'] = worker.engine.module.config._attn_implementation
    if result['actual_attention'] != 'sdpa' or not worker.engine._uses_fsdp2_cpu_offload_policy:
        raise RuntimeError('actual attention/offload differs from requested path')
    result['actual_fsdp_policies'] = describe_actual_fsdp_policies(worker.engine.module, True)
    result['memory_after_reset'] = cuda_memory()
    journal.record('worker_reset_complete', memory=result['memory_after_reset'], policies=result['actual_fsdp_policies'])
    path_messages = []
    class Handler(logging.Handler):
        def emit(self, record):
            if record.getMessage().startswith('memory_bounded_teacher '):
                path_messages.append(record.getMessage())
    handler = Handler()
    worker_module.logger.setLevel(logging.INFO)
    worker_module.logger.addHandler(handler)
    result['samples'], result['all_calls'] = [], []
    try:
        for index in range(args.warmup + args.repeats):
            nested = lambda field: torch.nested.nested_tensor(
                [fixture[field][i, :length].clone() for i, length in enumerate(args.lengths)], layout=torch.jagged)
            data = TensorDict({field: nested(field) for field in ('input_ids', 'position_ids', 'topk_ids')},
                              batch_size=[len(args.lengths)])
            tu.assign_non_tensor(data, temperature=1.0, use_remove_padding=False, use_fused_kernels=False,
                                 teacher_chunk_size=args.gather_chunk_rows)
            phase = 'warmup' if index < args.warmup else 'measured'
            result['stage'] = f'compute_call_{index}'
            journal.record('call_begin', index=index, phase=phase)
            with observe_worker(worker.engine.module, fixture, journal) as events:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = cuda_memory()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                perf_start, raw_start = time.perf_counter_ns(), time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                start.record()
                output = worker.compute_logprobs_at_ids(data)
                end.record()
                torch.cuda.synchronize()
                raw_end, perf_end = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), time.perf_counter_ns()
                after = cuda_memory()
            journal.record('call_returned', index=index, before=before, after=after,
                           perf_counter_ms=(perf_end-perf_start)/1e6, raw_host_ms=(raw_end-raw_start)/1e6,
                           cuda_event_ms=start.elapsed_time(end), hooks=events)
            # Preserve returned bytes even if schema/finiteness validation fails.
            raw_path = args.out.with_suffix(f'.raw-call{index:02d}.pt')
            raw = None if output is None else {
                field: {'values': output[field].values().detach().cpu(),
                        'offsets': output[field].offsets().detach().cpu()}
                for field in output.keys()}
            with raw_path.open('xb') as stream:
                torch.save(raw, stream)
            journal.record('raw_output_saved', index=index, path=str(raw_path), sha256=sha256(raw_path))
            artifact = output_artifact(output, result['identity'])
            output_path = args.out.with_suffix(f'.call{index:02d}.pt')
            with output_path.open('xb') as stream:
                torch.save(artifact, stream)
            head_pre = [x for x in events if x['module'] == 'head' and x['event'] == 'pre']
            expected_path = 'dense_full_row_fallback' if args.mode == 'dense' else 'chunked_lm_head'
            checks = {'one_outer_backbone': all(sum(x['module'] == name and x['event'] == phase_ for x in events) == 1
                       for name in ('outer', 'backbone') for phase_ in ('pre', 'post')),
                      'head_path': validate_head_calls(head_pre, args.mode, len(args.lengths), max(args.lengths), args.chunk_rows),
                      'head_post_count': sum(x['module'] == 'head' and x['event'] == 'post' for x in events) == len(head_pre),
                      'observed_path': bool(path_messages) and f'path={expected_path} ' in path_messages[-1],
                      'repeat_exact': not result['all_calls'] or artifact['output'] == result['all_calls'][0]['output']}
            sample = {'index': index, 'phase': phase, 'perf_counter_ms': (perf_end-perf_start)/1e6,
                'raw_host_ms': (raw_end-raw_start)/1e6, 'cuda_event_ms': start.elapsed_time(end),
                'host_clock_boundaries': {'perf_start_ns': perf_start, 'perf_end_ns': perf_end,
                                         'raw_start_ns': raw_start, 'raw_end_ns': raw_end},
                'before': before, 'after': after, 'incremental_peak_allocated_bytes': after['peak_allocated_bytes']-before['allocated_bytes'],
                'checks': checks, 'output': artifact['output'], 'artifact': {'path': str(output_path), 'sha256': sha256(output_path)},
                'hooks': events, 'decision_log': path_messages[-1] if path_messages else None}
            result['all_calls'].append(sample)
            if phase == 'measured':
                result['samples'].append(sample)
            journal.record('call_complete', sample=sample)
            if not all(checks.values()):
                raise ValueError(f'Worker path/repeat checks failed: {checks}')
            del output, artifact
        result['summary'] = summarize_samples(result['samples'])
        result['status'] = 'PASS_WORKER_CONTRACT_REPEAT'
    finally:
        worker_module.logger.removeHandler(handler)


def main(argv=None):
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    journal = JsonlJournal(args.out.with_suffix('.events.jsonl'))
    sampler = DeviceSampler(args.sample_device_ms, 0, journal)
    result = {'status': 'RUNNING', 'stage': 'start', 'pid': os.getpid(), 'argv': sys.argv,
        'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'project_git_head': git_head(ROOT), 'started_unix_ns': time.time_ns(),
        'source_hashes': {name: sha256(ROOT / name) for name in (
            'scripts/long_worker_probe.py', 'scripts/verl_teacher_worker_smoke.py', 'scripts/factorial_teacher_probe.py',
            'reference/memory_bounded_teacher.py', 'integrations/verl_6920/bridge.py')},
        'scope': {'real_worker_direct_process': True, 'ray': 'NOT_RUN', 'multi_gpu': 'NOT_RUN', 'RL_training': 'NOT_RUN'},
        'measurement_scope': 'synchronized complete Worker compute through nested CPU return; input creation/output checks/hash/save outside; hooks + sampler diagnostics overlap timing',
        'allocator_scope': 'fraction only applies to PyTorch allocator, never an entire device hard cap'}
    try:
        journal.record('process_started', configuration=result['configuration'], sources=result['source_hashes'])
        sampler.start()
        run(args, result, journal)
    except Exception as error:
        result['status'] = 'OOM' if is_out_of_memory(error) else 'ERROR'
        result['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        result['memory_at_failure'] = safe_cuda_memory()
        journal.record('process_error', status=result['status'], error=result['error'], memory=result['memory_at_failure'])
    finally:
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception as error:
                result['cleanup_error'] = repr(error)
                result['status'] = 'ERROR'
        result['all_device_sampler'] = sampler.stop()
        # Preserve the completed Worker result, but do not report the measured
        # experiment as a success if the required device observation is absent.
        if result['all_device_sampler']['failure'] is not None or result['all_device_sampler']['sample_count'] == 0:
            result['worker_status_before_sampler_check'] = result['status']
            if result['status'].startswith('PASS'):
                result['status'] = 'ERROR_DEVICE_SAMPLER'
        result['finished_unix_ns'] = time.time_ns()
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(json_safe(result), stream, indent=2, allow_nan=False)
            stream.write('\n')
        journal.record('process_finished', status=result['status'], out=str(args.out), sha256=sha256(args.out))
        journal.close()
    print(json.dumps({'status': result['status'], 'out': str(args.out)}), flush=True)
    return 0 if result['status'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
