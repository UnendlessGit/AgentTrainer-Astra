#!/usr/bin/env python3
"""Matched production actor comparison for the zero-residual queue warm start.

Usage: .venv/bin/python scripts/benchmark_queued_control.py --report .local/queue-actor.json

Replays one generated, causally admitted practice trajectory in both arms.
Measures BGRA ingestion, real feedback validation/feature preparation, compiled
policy/full-budget decoder, GPU completion, command decoding and JSON output.
No SCK, frame-ring publication, process IPC or learned-behavior claim is made.
"""
from pathlib import Path
from dataclasses import replace
from copy import deepcopy
from importlib.metadata import version
import argparse
import gc
import hashlib
import json
import platform
import resource
import statistics
import time
import uuid

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from astra.control_feedback import validate_control_feedback, validate_feedback_continuation
from astra.data.actions import decode_commands
from astra.data.observations import make_observation
from astra.environments.practice import PracticeConfig, PracticeEnvironment
from astra.inference import _PolicyExecution, prepare_metal_surface
from astra.model.actions import ActionVocabulary, PacketBatch
from astra.model.config import ModelConfig
from astra.model.policy import AgentPolicy
from astra.model.vision import VisualFeatures


WARM_CALLS = 3
TIMED_CALLS = 12
POLICY_SEED, QUEUE_SEED, SAMPLER_SEED, PRACTICE_SEED = 171, 172, 817, 807
MEMORY_LIMIT, CACHE_LIMIT = 2500 * 1024**2, 64 * 1024**2


def statistics_ms(samples):
    return dict(medianMS=statistics.median(samples), p95MS=float(np.percentile(samples, 95)),
                maximumMS=max(samples), samplesMS=samples)


def without_random_ids(value):
    """Only for the reproducibility digest; original inputs remain unchanged."""
    if isinstance(value, dict):
        return {key: without_random_ids(child) for key, child in value.items()}
    if isinstance(value, list):
        return [without_random_ids(child) for child in value]
    if isinstance(value, str) and len(value) == 36:
        try:
            uuid.UUID(value)
            return '<identity>'
        except ValueError:
            pass
    return value


def fixture_trace(config):
    environment = PracticeEnvironment(PracticeConfig(pixel_width=1280, pixel_height=720,
        period_ms=config.period_ms, lead_ms=config.lead_ms, relative_pointer=True))
    environment.enable_control_feedback(str(uuid.uuid5(uuid.NAMESPACE_URL, 'astra:queued-control-benchmark-v1')))
    observation = environment.reset(seed=PRACTICE_SEED)
    events, last_input = [], None
    fixtures, details = [], []
    digest = hashlib.sha256()
    for step in range(WARM_CALLS + TIMED_CALLS):
        feedback = observation.control_feedback
        validate_control_feedback(feedback, cutoff_nanos=observation.metadata['observedNanos'],
                                  surfaces=[observation.metadata['surface']], require_available=True)
        fixtures.append((observation, events, last_input))
        details.append(dict(cutoffNanos=observation.metadata['observedNanos'],
            packets=len(feedback['packets']), pendingPackets=sum(row.get('terminal') is None for row in feedback['packets']),
            commands=sum(len(row['packet']['commands']) for row in feedback['packets']),
            partialCommands=sum(progress['status'] == 'partial' for row in feedback['packets'] for progress in row['progress']),
            executedEvents=len(events), feedbackBytes=len(json.dumps(feedback, separators=(',', ':'), allow_nan=False).encode())))
        digest.update(observation.pixels.tobytes())
        digest.update(json.dumps(without_random_ids(dict(metadata=observation.metadata, controls=observation.control_state,
            events=events, feedback=feedback)), sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        if step == WARM_CALLS + TIMED_CALLS - 1:
            break
        if events:
            last_input = max(last_input or 0, max(event['observedNanos'] for event in events))
        # Full original packets create genuine pending trajectories and no-op
        # releases. They never click/complete the pointing task. These scripted
        # environment actions are not current policy samples or teacher inputs.
        commands = []
        for index in range(config.packet_capacity):
            offset = index * config.period_ms // (config.packet_capacity - 1)
            if index % 4 == 1:
                commands.append(dict(offsetMs=offset, operation='buttonUp', button=0))
            else:
                commands.append(dict(offsetMs=offset, operation='pointerAbsolute', surfaceID='practice',
                    x=.15 + .65 * ((step * 17 + index * 5) % 97) / 96,
                    y=.20 + .55 * ((step * 11 + index * 7) % 89) / 88))
        transition = environment.step(commands)
        if transition.outcome != 'continuing':
            raise AssertionError('The fixed practice trace ended before the benchmark observations')
        observation, events = transition.observation, transition.raw_events
    assert all(item['pendingPackets'] > 0 for item in details[WARM_CALLS:])
    assert any(item['partialCommands'] > 0 for item in details[WARM_CALLS:])
    return fixtures, dict(seed=PRACTICE_SEED, semanticTraceSHA256=digest.hexdigest(), steps=details,
        identityDigestNote='Random audit UUIDs are replaced only in this digest, never in actor inputs.',
        behavior='One original scripted practice trajectory; sampled benchmark outputs are not executed.')


def matched_policies(config, vocabulary):
    mx.random.seed(POLICY_SEED)
    old = AgentPolicy(config, vocabulary)
    backbone = Path(__file__).resolve().parents[1] / 'vendor/weights/convnext_tiny.safetensors'
    old.vision.backbone.load_pretrained(backbone)
    mx.random.seed(QUEUE_SEED)
    new = AgentPolicy(replace(config, schema_version=3), vocabulary)
    original, expanded = dict(tree_flatten(old.parameters())), dict(tree_flatten(new.parameters()))
    added = expanded.keys() - original.keys()
    if (not added or not original.keys() <= expanded.keys()
            or any(not name.startswith('temporal.queued_control.') for name in added)
            or any(original[name].shape != expanded[name].shape for name in original)):
        raise AssertionError('The matched arm changed an existing parameter layout')
    new.load_weights(list(original.items()), strict=False)
    old.eval(); new.eval(); mx.eval(old.parameters(), new.parameters())
    copied = dict(tree_flatten(new.parameters()))
    weights = hashlib.sha256()
    for name in sorted(original):
        expected, actual = np.asarray(original[name]), np.asarray(copied[name])
        np.testing.assert_array_equal(expected, actual)
        weights.update(name.encode() + b'\0' + str(expected.dtype).encode() + b'\0')
        weights.update(json.dumps(expected.shape).encode() + b'\0' + expected.tobytes())
    assert not np.count_nonzero(np.asarray(new.temporal.queued_control.output.weight))
    return {'schema2': old, 'schema3': new}, dict(sharedWeightsSHA256=weights.hexdigest(),
        sharedParameterNames=len(original), addedParameterNames=len(added),
        sharedParametersBitwiseEqual=True, queuedOutputProjectionZero=True,
        initialization='Production architecture with vendored pretrained ConvNeXt and fixed random remaining weights; no learned-quality claim.')


def run_arm(name, policy, fixtures, *, greedy):
    gc.collect(); mx.clear_cache(); mx.reset_peak_memory()
    active_before = mx.get_active_memory()
    runner = _PolicyExecution(policy, greedy=greedy, compiled=True)
    state, key = policy.temporal.initial_state(1), mx.random.key(SAMPLER_SEED)
    mx.eval(state, key)
    samples, warm, phases, snapshots, command_counts = [], [], [], [], []
    previous_feedback, warm_peak = None, 0
    for index, (raw, events, last_input) in enumerate(fixtures):
        if index == WARM_CALLS:
            warm_peak = mx.get_peak_memory(); mx.reset_peak_memory()
        start = time.perf_counter()
        owned = mx.array(raw.pixels)
        extra = {}
        if policy.config.schema_version == 3:
            # Same checks performed by InferenceSession before the shared
            # observation builder; the builder validates its semantic features.
            validate_control_feedback(raw.control_feedback, cutoff_nanos=raw.metadata['observedNanos'],
                geometry_revision=0, run_id=raw.control_feedback['runID'], surfaces=[raw.metadata['surface']], require_available=True)
            validate_feedback_continuation(previous_feedback, raw.control_feedback)
            extra['control_feedback'] = raw.control_feedback
        observation = make_observation([(owned, raw.metadata)], raw.control_state,
            cutoff_nanos=raw.metadata['observedNanos'], elapsed_seconds=policy.config.period_ms / 1000,
            reset=index == 0, config=policy.config, executed_events=events, last_input_nanos=last_input,
            surface_preparer=prepare_metal_surface, maximum_timestamp=2**64 - 1, **extra)
        built = time.perf_counter()
        output = runner(observation, state, key)
        graphed = time.perf_counter()
        mx.eval(output)
        evaluated = time.perf_counter()
        if not bool(output['finite'].item()):
            raise AssertionError('Nonfinite policy output')
        commands = decode_commands(PacketBatch(**output['packet']), config=policy.config,
            vocabulary=policy.actions.vocabulary, visual=VisualFeatures(**output['visual']), surfaces=[raw.metadata['surface']])
        encoded = json.dumps(dict(commands=commands, value=float(output['value'].item()),
            logProbability=float(output['log_probability'].item()), conditionalEntropy=float(output['conditional_entropy'].item())), allow_nan=False)
        state, key = output['state'], output['key']
        if policy.config.schema_version == 3:
            previous_feedback = deepcopy(raw.control_feedback)
        finished = time.perf_counter()
        milliseconds = (finished - start) * 1000
        command_counts.append(len(commands))
        if index < WARM_CALLS:
            warm.append(milliseconds)
        else:
            samples.append(milliseconds)
            phases.append(dict(prepareGraphMS=(built-start)*1000, policyGraphMS=(graphed-built)*1000,
                evaluateMS=(evaluated-graphed)*1000, decodeJSONAndRetainMS=(finished-evaluated)*1000))
        # Numerical comparison copies and logging are outside the timed actor.
        snapshots.append(dict(commands=commands, tokens={field: np.asarray(value).copy() for field, value in output['packet'].items()},
            logProbability=np.asarray(output['log_probability']).copy(), value=np.asarray(output['value']).copy(),
            entropy=np.asarray(output['conditional_entropy']).copy(), state=[np.asarray(value).copy() for value in state],
            nextKey=np.asarray(key).copy()))
        print(f'{name} step {index}: {milliseconds:.3f} ms, commands {len(commands)}, JSON bytes {len(encoded)}', flush=True)
        del output, observation, owned
    return dict(**statistics_ms(samples), initialSamplesMS=warm,
        phaseMediansMS={field: statistics.median(row[field] for row in phases) for field in phases[0]},
        commandCounts=command_counts, activeMLXBytesBeforeArm=active_before,
        peakMLXBytesWarmCalls=warm_peak, peakMLXBytesTimedCalls=mx.get_peak_memory(),
        processPeakRSSBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)), snapshots


def parity(expected, actual):
    errors = dict(logProbability=0., value=0., entropy=0., state=0.)
    commands_equal = tokens_equal = keys_equal = numerical = True
    for left, right in zip(expected, actual, strict=True):
        commands_equal &= left['commands'] == right['commands']
        tokens_equal &= all(np.array_equal(left['tokens'][field], right['tokens'][field]) for field in left['tokens'])
        keys_equal &= np.array_equal(left['nextKey'], right['nextKey'])
        fields = [(field, left[field], right[field]) for field in ('logProbability', 'value', 'entropy')]
        fields += [('state', a, b) for a, b in zip(left['state'], right['state'], strict=True)]
        for field, a, b in fields:
            errors[field] = max(errors[field], float(np.max(np.abs(a - b))))
            numerical &= bool(np.allclose(a, b, rtol=1e-5, atol=1e-4 if field in ('logProbability', 'entropy') else 1e-5))
    return dict(passed=bool(commands_equal and tokens_equal and keys_equal and numerical),
        commandsIdentical=bool(commands_equal), allPacketTokensIdentical=bool(tokens_equal), nextRNGKeysIdentical=bool(keys_equal),
        numericalWithinTolerance=bool(numerical), maximumAbsoluteErrors=errors,
        tolerance='rtol1e-5; atol1e-4 for joint log probability/entropy,1e-5 for value/recurrent state')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--order', choices=('schema2-first', 'schema3-first'), default='schema2-first')
    parser.add_argument('--greedy', action='store_true')
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Choose a new report path; existing measurement evidence is not overwritten')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    mx.set_memory_limit(MEMORY_LIMIT); mx.set_cache_limit(CACHE_LIMIT)
    config = replace(ModelConfig(), lead_ms=200).validate()
    vocabulary = ActionVocabulary(key_codes=(0, 13), mouse_buttons=(0,), absolute_pointer=True, relative_pointer=True, scroll=True)
    fixtures, fixture_report = fixture_trace(config)
    policies, weight_report = matched_policies(config, vocabulary)
    order = ['schema2', 'schema3'] if args.order == 'schema2-first' else ['schema3', 'schema2']
    results, observations = {}, {}
    for name in order:
        results[name], observations[name] = run_arm(name, policies[name], fixtures, greedy=args.greedy)
    agreement = parity(observations['schema2'], observations['schema3'])
    before, after = results['schema2']['medianMS'], results['schema3']['medianMS']
    report = dict(schemaVersion=1, passed=agreement['passed'], platform=platform.platform(), machine=platform.machine(),
        mlxVersion=version('mlx'), device=mx.device_info(), pixelSize=[1280, 720], warmCallsPerArm=WARM_CALLS,
        timedCallsPerArm=TIMED_CALLS, compiled=True, greedy=args.greedy, order=order,
        seeds=dict(policy=POLICY_SEED, queuedEncoder=QUEUE_SEED, sampler=SAMPLER_SEED),
        models={name: policy.config.to_dict() for name, policy in policies.items()},
        parameterCounts={name: policy.config.parameter_count for name, policy in policies.items()},
        weights=weight_report, fixture=fixture_report, results=results, parity=agreement,
        overhead=dict(medianMS=after-before, medianPercent=(after/before-1)*100),
        memory=dict(limitBytes=MEMORY_LIMIT, cacheLimitBytes=CACHE_LIMIT,
            note='Both immutable policies share copied common parameter arrays. MLX peaks reset per arm and after warm calls; process RSS is cumulative. These are not isolated-process memory deltas.'),
        scope='Owned BGRA ingestion, feedback validation/continuation, shared Metal preparation, full-budget compiled policy/decoder, GPU eval, decoded JSON and retained feedback. No SCK, ring publication, frame-reference validation or IPC timing. Parity copies/logging excluded. No deadline/parameter retuning or learning-quality claim.')
    encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
    with args.report.open('x') as destination:
        destination.write(encoded)
    print(encoded, end='')
    if not agreement['passed']:
        raise SystemExit('Zero-residual parity failed; inspect the saved report before interpreting latency')


if __name__ == '__main__':
    main()
