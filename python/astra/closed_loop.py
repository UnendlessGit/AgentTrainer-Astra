"""Frozen local practice evaluation using the production compiled policy path."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from .environments.practice import PracticeConfig, PracticeEnvironment, PracticeError
from .model.contexts import vocabulary


def _int(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('Closed-loop protocol integer is outside its supported range')
    return value


def validate_protocol(value):
    required = {'schemaVersion', 'task', 'periodMS', 'leadMS', 'delayMS', 'cueMS', 'timeLimitMS',
                'deterministic', 'policySeed', 'trials', 'contextSizes', 'contextVocabulary', 'contextIDs'}
    if type(value) is not dict or set(value) != required or value['schemaVersion'] != 1 or type(value['schemaVersion']) is not int:
        raise ValueError('Unsupported closed-loop protocol')
    _int(value['periodMS'], 1, 1000); _int(value['leadMS'], 0, 2000)
    _int(value['delayMS'], 1, 120000); _int(value['cueMS'], 1, 120000); _int(value['timeLimitMS'], 1, 600000)
    _int(value['policySeed'], 0, 1_000_000_000)
    if value['task'] not in ('pointing', 'delayed_memory') or type(value['deterministic']) is not bool:
        raise ValueError('Choose a supported practice task and action-selection mode')
    if value['task'] == 'delayed_memory' and value['timeLimitMS'] <= value['delayMS'] + value['cueMS']:
        raise ValueError('The memory protocol needs time to answer after its cue and delay')
    trials = value['trials']
    if type(trials) is not list or not 1 <= len(trials) <= 256:
        raise ValueError('Closed-loop protocols require 1–256 fixed trials')
    seen = set()
    for trial in trials:
        if type(trial) is not dict or set(trial) != {'seed', 'pixelWidth', 'pixelHeight', 'logicalBounds'}:
            raise ValueError('Invalid closed-loop trial layout')
        _int(trial['seed'], 0, 1_000_000_000)
        _int(trial['pixelWidth'], 32, 8192); _int(trial['pixelHeight'], 32, 8192)
        if trial['pixelWidth'] * trial['pixelHeight'] * 4 > 64 * 1024**2:
            raise ValueError('A practice layout exceeds its pixel budget')
        bounds = trial['logicalBounds']
        if type(bounds) is not list or len(bounds) != 4:
            raise ValueError('Practice layout requires four logical bounds')
        for i, number in enumerate(bounds): _int(number, -1_000_000 if i < 2 else 1, 1_000_000)
        identity = json.dumps(trial, sort_keys=True)
        if identity in seen: raise ValueError('A protocol cannot repeat an identical seed and layout')
        seen.add(identity)
    if len(trials) * math.ceil(value['timeLimitMS'] / value['periodMS']) > 200_000:
        raise ValueError('The evaluation exceeds its decision budget')
    sizes, contexts = value['contextSizes'], value['contextIDs']
    if type(sizes) is not list or len(sizes) > 32 or type(contexts) is not list or len(contexts) != len(sizes):
        raise ValueError('Evaluation contexts must match their frozen vocabulary')
    for size, choice in zip(sizes, contexts): _int(size, 1, 65536); _int(choice, 0, size - 1)
    if value['contextVocabulary'] is not None: vocabulary(value['contextVocabulary'], sizes)
    return copy.deepcopy(value)


def fingerprint(protocol):
    value = copy.deepcopy(protocol)
    if value['deterministic']: value['policySeed'] = 0
    if value['contextVocabulary'] is not None:
        value['contextVocabulary'] = [field.semantic_dict() for field in vocabulary(value['contextVocabulary'], value['contextSizes'])]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def evaluate(checkpoint_path: Path, protocol, *, cancelled=lambda: False, progress=lambda _: None):
    import mlx.core as mx
    from .checkpoints import load_checkpoint
    from .data.actions import decode_commands, ActionEncodingError
    from .data.observations import make_observation
    from .inference import _PolicyExecution, prepare_metal_surface
    from .model.actions import PacketBatch
    from .model.vision import VisualFeatures

    protocol = validate_protocol(protocol)
    loaded = load_checkpoint(checkpoint_path); policy = loaded.policy; config = policy.config
    expected_context = () if protocol['contextVocabulary'] is None else vocabulary(protocol['contextVocabulary'], protocol['contextSizes'])
    if (config.period_ms != protocol['periodMS'] or config.lead_ms != protocol['leadMS'] or
            config.context_sizes != tuple(protocol['contextSizes']) or config.context_vocabulary != expected_context):
        raise ValueError('Checkpoint timing or categorical context meaning differs from the fixed evaluation protocol')
    policy.eval(); execution = _PolicyExecution(policy, greedy=protocol['deterministic'])
    results = []
    for index, trial in enumerate(protocol['trials']):
        if cancelled(): raise InterruptedError('Closed-loop evaluation stopped before the next trial')
        environment = PracticeConfig(task=protocol['task'], seed=trial['seed'], pixel_width=trial['pixelWidth'],
            pixel_height=trial['pixelHeight'], logical_bounds=tuple(trial['logicalBounds']), period_ms=protocol['periodMS'],
            lead_ms=protocol['leadMS'], delay_ms=protocol['delayMS'], cue_ms=protocol['cueMS'], time_limit_ms=protocol['timeLimitMS'], shaping_scale=0)
        env = PracticeEnvironment(environment)
        if policy.actions.vocabulary != env.action_vocabulary:
            raise ValueError('Checkpoint controls do not match the selected practice task')
        current = env.reset(seed=trial['seed'])
        state = None; key = mx.random.key(0 if protocol['deterministic'] else protocol['policySeed'] + index)
        events = []; last_input = None; decisions = 0; total_return = 0.0
        outcome = None; fault = None; success = False
        try:
            while outcome is None:
                if cancelled(): raise InterruptedError('Closed-loop evaluation stopped during a trial')
                observed = make_observation([(mx.array(current.pixels), current.metadata)], current.control_state,
                    cutoff_nanos=current.metadata['observedNanos'], elapsed_seconds=config.period_ms / 1000,
                    reset=decisions == 0, config=config, context_ids=tuple(protocol['contextIDs']), executed_events=events,
                    last_input_nanos=last_input, surface_preparer=prepare_metal_surface)
                output = execution(observed, state, key); mx.eval(output)
                if not bool(output['finite'].item()):
                    outcome = 'numerical_fault'; fault = 'Policy returned nonfinite output'; break
                commands = decode_commands(PacketBatch(**output['packet']), config=config, vocabulary=policy.actions.vocabulary,
                    visual=VisualFeatures(**output['visual']), surfaces=[current.metadata['surface']])
                state, key = output['state'], output['key']
                transition = env.step(commands, episode_id=current.episode_id, provenance='agent')
                current = transition.observation; events = transition.raw_events; decisions += 1; total_return += transition.reward
                times = [event['observedNanos'] for event in events if event['origin'] in ('physical', 'agent')]
                if times: last_input = max(last_input or 0, max(times))
                if transition.outcome != 'continuing':
                    outcome = transition.outcome
                    success = outcome == 'terminated' and transition.reward > 0
        except (PracticeError, ActionEncodingError) as error:
            outcome = 'action_fault'; fault = str(error)[:2048]
        finally:
            if env.outcome == 'continuing': env.abort('Evaluation trial ended')
        result = dict(index=index, seed=trial['seed'], outcome=outcome, success=success, returnValue=total_return,
                      decisions=decisions, virtualDurationMS=env.elapsed_ms)
        if fault is not None: result['fault'] = fault
        results.append(result)
        progress(dict(phase='evaluating', completedTrials=len(results), totalTrials=len(protocol['trials']),
                      successes=sum(row['success'] for row in results)))
    if cancelled(): raise InterruptedError('Closed-loop evaluation stopped before result publication')
    return dict(checkpointID=loaded.manifest['id'], policySignature=loaded.manifest['policySignature'],
                protocolFingerprint=fingerprint(protocol), provenance='practice_closed_loop', trials=results)
