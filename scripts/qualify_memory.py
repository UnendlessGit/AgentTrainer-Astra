#!/usr/bin/env python3
"""Bounded, causal BC/checkpoint/closed-loop delayed-memory qualification.

Oracle actions supply demonstration/evaluation labels only. The actor sees
pixels and causal controls through the shared observation and policy path.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import platform
import resource
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))


def initialize_experimental_retention(policy, variant, *, maximum_horizon=512):
    """Experiment-only persistent GRU gate initialization; no schema/default change.

    MLX 0.32.2 orders biases r,z,n and uses h'=(1-z)n+z*h. Its native
    biases are near-zero uniform draws. Geometric horizons are a deterministic
    multi-timescale variant of the chrono principle, not the paper's sampler.
    """
    import mlx.core as mx
    import numpy as np
    if variant not in ('native', 'zero-update', 'geometric'):
        raise ValueError('Unknown experimental GRU initialization')
    if type(maximum_horizon) is not int or not 2 <= maximum_horizon <= 100000:
        raise ValueError('Experimental horizon must be a bounded decision count')
    if variant == 'native': return dict(variant=variant, changedParameters=[])
    changed = []
    for index, layer in enumerate(policy.temporal.layers):
        width = layer.hidden_size
        if layer.b is None or layer.b.shape != (3 * width,): raise ValueError('Unexpected GRU bias layout')
        original = layer.b
        if variant == 'zero-update':
            update_bias = mx.zeros((width,), dtype=mx.float32)
        else:
            # H=1 implies -infinity. A finite 1.01 lower bound keeps fast units
            # while every checkpoint parameter remains finite FP32.
            horizon = np.geomspace(1.01, maximum_horizon, width)
            update_bias = mx.array(np.log(horizon - 1).astype(np.float32))
        layer.b = mx.concatenate((original[:width], update_bias, original[2 * width:]))
        mx.eval(layer.b)
        np.testing.assert_array_equal(layer.b[:width], original[:width])
        np.testing.assert_array_equal(layer.b[2 * width:], original[2 * width:])
        assert layer.b.dtype == mx.float32 and bool(mx.all(mx.isfinite(layer.b)).item())
        changed.append(f'temporal.layers.{index}.b[{width}:{2 * width}]')
    return dict(variant=variant, minimumHorizon=1.01 if variant=='geometric' else 2,
                maximumHorizon=maximum_horizon if variant=='geometric' else 2, changedParameters=changed)


def memory_phase_fixture(environment, seed, cache_directory):
    """Cache exact passive cue/wait/choice pixels for paired diagnostics only."""
    from copy import deepcopy
    from types import SimpleNamespace
    import hashlib
    import numpy as np
    import astra.environments.practice as implementation
    from astra.environments.practice import PracticeEnvironment
    fingerprint = hashlib.sha256((environment.signature + hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest()).encode()).hexdigest()
    destination = cache_directory / fingerprint / f'{seed}.npz'
    if destination.exists():
        with np.load(destination, allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata'].item()))
            if metadata['fingerprint'] != fingerprint or metadata['seed'] != seed: raise ValueError('Mismatched diagnostic fixture cache')
            phases = [SimpleNamespace(pixels=archive[name].copy(), metadata=entry['metadata'], control_state=entry['controlState'])
                      for name, entry in zip(('cueA','cueB','wait','choice'), metadata['phases'])]
            return phases, metadata['labels']
    env = PracticeEnvironment(environment); current = env.reset(seed=seed)
    flipped = deepcopy(env); flipped._answer = 1 - env._answer; other = flipped._observe()
    assert current.control_state == other.control_state and not np.array_equal(current.pixels, other.pixels)
    phases = [current, other]
    for _ in range(environment.cue_ms // environment.period_ms):
        current = env.step([], episode_id=current.episode_id).observation
        other = flipped.step([], episode_id=other.episode_id).observation
    assert np.array_equal(current.pixels, other.pixels) and current.control_state == other.control_state
    phases.append(current)
    while env.elapsed_ms < environment.cue_ms + environment.delay_ms:
        before = current.pixels
        current = env.step([], episode_id=current.episode_id).observation
        other = flipped.step([], episode_id=other.episode_id).observation
        assert np.array_equal(current.pixels, other.pixels)
        if env.elapsed_ms < environment.cue_ms + environment.delay_ms: assert np.array_equal(before, current.pixels)
    phases.append(current)
    labels = [world.oracle_commands() for world in (env, flipped)]
    metadata = dict(fingerprint=fingerprint, seed=seed, labels=labels, provenance='counterfactual_oracle_evaluation_only',
                    phases=[dict(metadata=phase.metadata, controlState=phase.control_state) for phase in phases])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.'+str(uuid.uuid4())+'.tmp')
    with temporary.open('wb') as file:
        np.savez_compressed(file, **{name: phase.pixels for name,phase in zip(('cueA','cueB','wait','choice'),phases)},
                            metadata=np.array(json.dumps(metadata)))
    temporary.replace(destination)
    return phases, labels


def counterfactual_probe(checkpoint, *, seeds, output, budget_seconds=120):
    """Balanced cue-retention diagnostic; this is not closed-loop success.

    Both members of a pair share layout/control state and differ only in cue
    pixels. Oracle calls create evaluation labels, never model observations.
    Repeated identical waiting images reuse exact visual features while every
    recurrent decision still executes. No model or trainer implementation changes.
    """
    import mlx.core as mx
    import numpy as np
    from astra.checkpoints import load_checkpoint
    from astra.data.actions import encode_commands
    from astra.data.observations import make_observation
    from astra.environments.practice import PracticeConfig, PracticeEnvironment
    from astra.inference import prepare_metal_surface
    from astra.model.actions import flatten_visual
    from astra.model.observation import ObservationBatch

    began = time.perf_counter(); loaded = load_checkpoint(checkpoint); policy = loaded.policy; policy.eval()
    config = policy.config; delays = (2000, 8000, 30000)
    environment = PracticeConfig(task='delayed_memory', delay_ms=max(delays), time_limit_ms=32500)
    assert config.period_ms == environment.period_ms and config.lead_ms == environment.lead_ms
    results = []; checks = []
    def temporal(summary, controls, reset, state):
        # TemporalCore reads these causal fields, never surface images.
        observation = ObservationBatch((), controls, mx.full(controls.shape[:2], .1),
            mx.zeros((*controls.shape[:2], 0), dtype=mx.int32), reset, mx.ones(controls.shape[:2], dtype=mx.bool_))
        result = policy.temporal(summary, observation, state)
        return result.context, result.state
    temporal_step = mx.compile(temporal, inputs=policy.temporal.state)
    for seed in seeds:
        if time.perf_counter() - began > budget_seconds: break
        phases, commands = memory_phase_fixture(environment, seed, output.parent / 'memory-phase-cache')
        def encode(raw):
            observation = make_observation([(mx.array(raw.pixels), raw.metadata)], raw.control_state,
                cutoff_nanos=raw.metadata['observedNanos'], elapsed_seconds=.1, reset=False, config=config,
                surface_preparer=prepare_metal_surface)
            visual = policy.encode_visual(observation); mx.eval(visual.summary, visual.cells, observation.controls)
            return observation, visual
        cue_observations = [encode(phases[0]), encode(phases[1])]
        states = [policy.temporal.initial_state(1), policy.temporal.initial_state(1)]
        for step in range(environment.cue_ms // config.period_ms):
            for index in (0, 1):
                _, states[index] = temporal_step(cue_observations[index][1].summary, cue_observations[index][0].controls,
                                                mx.array([[step == 0]]), states[index])
            mx.eval(states)
        waiting, visual_wait = encode(phases[2])
        checks.append(dict(seed=seed, waitingPixelsEqual=True))
        choice_observation, visual_choice = encode(phases[3])
        assert np.array_equal(choice_observation.controls, waiting.controls)
        assert all(np.array_equal(value[0].controls, waiting.controls) for value in cue_observations)
        labels = [encode_commands(packet, config=config, vocabulary=policy.actions.vocabulary,
                   visual=flatten_visual(visual_choice), surfaces=[phases[3].metadata['surface']]) for packet in commands]
        active_steps = environment.cue_ms // config.period_ms
        for delay in delays:
            ready_step = (environment.cue_ms + delay) // config.period_ms
            while active_steps < ready_step:
                for index in (0, 1):
                    _, states[index] = temporal_step(visual_wait.summary, waiting.controls, mx.array([[False]]), states[index])
                mx.eval(states); active_steps += 1
            distances = [float(mx.sqrt(mx.mean((left-right)**2)).item()) for left, right in zip(*states)]
            pair_scores = []
            for index in (0, 1):
                context, _ = temporal_step(visual_choice.summary, choice_observation.controls, mx.array([[False]]), states[index])
                scores = [policy.actions.log_prob(context[:, 0], flatten_visual(visual_choice), label).log_probability for label in labels]
                mx.eval(scores)
                values = [float(score.item()) for score in scores]
                assert all(np.isfinite(values))
                margin = values[index] - values[1-index]
                pair_scores.append(dict(cueMember=index, correctLogProbability=values[index], wrongLogProbability=values[1-index],
                                        margin=margin, correct=margin > 0, tie=margin == 0))
            results.append(dict(seed=seed, delayMS=delay, recurrentRMSDifference=distances, choices=pair_scores))
        print(json.dumps(dict(phase='memoryProbe', layouts=len(checks), elapsedSeconds=time.perf_counter()-began)), flush=True)
    summary = []
    for delay in delays:
        choices = [choice for row in results if row['delayMS']==delay for choice in row['choices']]
        if choices: summary.append(dict(delayMS=delay, choices=len(choices), correct=sum(choice['correct'] for choice in choices),
            ties=sum(choice['tie'] for choice in choices), balancedAccuracy=sum(choice['correct'] + .5*choice['tie'] for choice in choices)/len(choices),
            meanMargin=float(np.mean([choice['margin'] for choice in choices])),
            meanStateRMSDifference=float(np.mean([value for row in results if row['delayMS']==delay for value in row['recurrentRMSDifference']]))))
    report = dict(schemaVersion=1, checkpoint=str(checkpoint), policySignature=loaded.manifest['policySignature'],
        model=config.to_dict(), environment=environment.to_dict(), seeds=list(seeds), completedLayouts=len(checks), summary=summary, trials=results,
        provenance='counterfactual_oracle_evaluation_only', waitingPixelsEqual=True,
        scope='Forced correct/wrong packet likelihood after passive cue-free delay, not freely acting success.',
        wallSeconds=time.perf_counter()-began, peakMLXBytes=mx.get_peak_memory())
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(phase='memoryProbeComplete', summary=summary, report=str(output))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-probe-cache', action='store_true', help='Prepare only exact CPU phase fixtures, without loading a policy')
    parser.add_argument('--probe-checkpoint', type=Path, help='Run only balanced paired-cue likelihood diagnostics at 2/8/30 seconds')
    parser.add_argument('--delay-ms', type=int, default=2000, choices=(2000, 8000, 30000))
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max-updates', type=int, help='Stop at an exact completed optimizer update for matched comparisons')
    parser.add_argument('--model-seed', type=int, default=834)
    parser.add_argument('--reference-initial', type=Path, help='Verify all non-update-bias tensors match this baseline initial checkpoint')
    parser.add_argument('--probe-before-training', action='store_true')
    parser.add_argument('--retention-init', choices=('native', 'zero-update', 'geometric'), default='native')
    parser.add_argument('--train-seeds', type=int, default=32)
    parser.add_argument('--heldout-seeds', type=int, default=32)
    parser.add_argument('--sequence-length', type=int, default=64)
    parser.add_argument('--training-budget-seconds', type=float, default=360)
    parser.add_argument('--budget-seconds', type=float, default=600)
    parser.add_argument('--small', action='store_true', help='Numerical exploration only; not a default-model learning result')
    parser.add_argument('--skip-baseline-policy', action='store_true', help='Still evaluate an answer-independent always-left chance baseline')
    args = parser.parse_args()
    if args.max_updates is not None and args.max_updates < 1: parser.error('Update limit must be positive')
    if not 0 <= args.model_seed < 2**63: parser.error('Model seed must be a bounded nonnegative integer')
    if not 1 <= args.train_seeds <= 1000 or not 1 <= args.heldout_seeds <= 1000:
        parser.error('Use one to 1000 independent episode seeds per split')
    if not 1 <= args.budget_seconds <= 3600:
        parser.error('Set an overall budget of at most one hour')
    if not args.probe_checkpoint and not args.prepare_probe_cache and not 1 <= args.training_budget_seconds < args.budget_seconds:
        parser.error('Set a positive training deadline within the overall budget')
    import mlx.core as mx
    from astra.checkpoints import save_checkpoint, load_checkpoint
    from astra.data.actions import decode_commands
    from astra.data.observations import make_observation
    from astra.environments.practice import PracticeConfig, PracticeEnvironment
    from astra.environments.demonstrations import PracticeDemonstrations
    from astra.inference import _PolicyExecution, prepare_metal_surface
    from astra.learning.behavioral import BehaviorConfig, BehaviorTrainer
    from astra.model.actions import PacketBatch
    from astra.model.config import ModelConfig
    from astra.model.policy import AgentPolicy
    from astra.model.vision import VisualFeatures

    mx.set_memory_limit(10 * 1024**3); mx.set_cache_limit(64 * 1024**2)
    if args.prepare_probe_cache:
        environment = PracticeConfig(task='delayed_memory', delay_ms=30000, time_limit_ms=32500)
        for seed in range(1000, 1000 + args.heldout_seeds):
            memory_phase_fixture(environment, seed, args.output.parent / 'memory-phase-cache')
            print(json.dumps(dict(phase='fixtureCache', seed=seed)), flush=True)
        return
    if args.probe_checkpoint:
        counterfactual_probe(args.probe_checkpoint, seeds=range(1000, 1000 + args.heldout_seeds), output=args.output, budget_seconds=args.budget_seconds)
        return
    began = time.perf_counter(); deadline = began + args.budget_seconds
    train_seeds = list(range(args.train_seeds)); heldout = list(range(1000, 1000 + args.heldout_seeds))
    config = ModelConfig.test_small() if args.small else ModelConfig()
    environment = PracticeConfig(task='delayed_memory', delay_ms=args.delay_ms, time_limit_ms=500 + args.delay_ms + 2000,
                                 pixel_width=64 if args.small else 1280, pixel_height=64 if args.small else 720)
    training = BehaviorConfig(epochs=args.epochs, lanes=2, sequence_length=args.sequence_length, seed=834).validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schemaVersion=1, status='preparing', configuration='numerical-test-small' if args.small else 'production-default',
                  model=config.to_dict(), environment=environment.to_dict(), training=asdict(training),
                  provenance='practice_oracle', trainSeeds=train_seeds, heldoutSeeds=heldout,
                  platform=platform.platform(), device=mx.device_info(), epochs=[], closedLoop={})
    def publish(phase, **values):
        report['status'] = phase; report['wallSeconds'] = time.perf_counter() - began
        report['peakMLXBytes'] = mx.get_peak_memory()
        report['processPeakRSSBytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)
        report.update(values)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'); temporary.replace(args.output)
        print(json.dumps(dict(phase=phase, elapsedSeconds=report['wallSeconds'], **values), allow_nan=False), flush=True)
    def within_budget():
        if time.perf_counter() >= deadline: raise InterruptedError('Overall qualification deadline reached')
    publish('preparing')
    source = PracticeDemonstrations(environment=environment, model=config,
                                   seeds_by_split={'train': train_seeds, 'validation': heldout},
                                   cancelled=lambda: time.perf_counter() >= deadline)
    episodes = source.episodes('train')
    lengths = [episode['steps'] for episode in episodes]
    answer_step = (environment.cue_ms + environment.delay_ms + environment.period_ms - 1) // environment.period_ms
    report['supervision'] = dict(trainingDecisions=sum(lengths), episodeLengths=lengths,
        answerDecisionsPerEpisode=1, answerDecisionFraction=1 / lengths[0], cueDecisions=environment.cue_ms // config.period_ms,
        answerStep=answer_step, answerChunkStart=answer_step // training.sequence_length * training.sequence_length,
        cueAndAnswerInSameTBPTTChunk=answer_step < training.sequence_length)
    report['tbpttCoverageAtDefaultCadence'] = [{
        'delayMS': delay, 'answerStep': (environment.cue_ms + delay) // config.period_ms,
        'answerChunkStart': ((environment.cue_ms + delay) // config.period_ms) // training.sequence_length * training.sequence_length,
        'cueAndAnswerInSameChunk': (environment.cue_ms + delay) // config.period_ms < training.sequence_length,
    } for delay in (2000, 8000, 30000)]
    mx.random.seed(args.model_seed)
    policy = AgentPolicy(config, source.vocabulary)
    report['initialization'] = initialize_experimental_retention(policy, args.retention_init)
    report['modelSeed'] = args.model_seed
    report['maximumUpdates'] = args.max_updates
    if not args.small: policy.vision.backbone.load_pretrained(ROOT / 'vendor/weights/convnext_tiny.safetensors')
    if args.reference_initial:
        from mlx.utils import tree_flatten
        reference = load_checkpoint(args.reference_initial).policy
        current_parameters, reference_parameters = dict(tree_flatten(policy.parameters())), dict(tree_flatten(reference.parameters()))
        assert current_parameters.keys() == reference_parameters.keys()
        checked = 0
        for name, value in current_parameters.items():
            previous = reference_parameters[name]
            assert value.shape == previous.shape and value.dtype == previous.dtype
            if name.startswith('temporal.layers.') and name.endswith('.b'):
                width = value.shape[0] // 3
                assert bool(mx.all(value[:width] == previous[:width]).item())
                assert bool(mx.all(value[2*width:] == previous[2*width:]).item())
            else:
                assert bool(mx.all(value == previous).item()), name
            checked += 1
        report['initialParity'] = dict(reference=str(args.reference_initial), checkedParameterLeaves=checked,
                                      allExceptPersistentUpdateBiasBitwiseEqual=True)
        del reference, current_parameters, reference_parameters; gc.collect(); mx.clear_cache()
    initial_path = args.output.parent / str(uuid.uuid4())
    save_checkpoint(initial_path, policy, kind='initial', step=0)
    report['initialCheckpoint'] = str(initial_path)
    if args.probe_before_training:
        initial_probe = args.output.with_name(args.output.stem + '-initial-probe.json')
        counterfactual_probe(initial_path, seeds=heldout, output=initial_probe, budget_seconds=min(110, deadline-time.perf_counter()-120))
        report['initialProbe'] = str(initial_probe)

    def trials(checkpoint, seeds, *, greedy=True, label):
        loaded = load_checkpoint(checkpoint); loaded.policy.eval()
        execution = _PolicyExecution(loaded.policy, greedy=greedy)
        results = []
        for seed in seeds:
            within_budget()
            env = PracticeEnvironment(environment); current = env.reset(seed=seed)
            state = None; key = mx.random.key(seed + 900000); events = []; last_input = None; step = 0; total_reward = 0
            while True:
                within_budget()
                observation = make_observation([(mx.array(current.pixels), current.metadata)], current.control_state,
                    cutoff_nanos=current.metadata['observedNanos'], elapsed_seconds=config.period_ms / 1000,
                    reset=step == 0, config=config, executed_events=events, last_input_nanos=last_input,
                    surface_preparer=prepare_metal_surface)
                output = execution(observation, state, key); mx.eval(output)
                if not output['finite'].item(): raise FloatingPointError('Nonfinite evaluation policy')
                commands = decode_commands(PacketBatch(**output['packet']), config=config, vocabulary=source.vocabulary,
                                            visual=VisualFeatures(**output['visual']), surfaces=[current.metadata['surface']])
                state, key = output['state'], output['key']
                transition = env.step(commands, episode_id=current.episode_id)
                current = transition.observation; events = transition.raw_events; step += 1; total_reward += transition.reward
                times = [event['observedNanos'] for event in events if event['origin'] in ('physical', 'agent')]
                if times: last_input = max(last_input or 0, max(times))
                if transition.outcome != 'continuing':
                    results.append(dict(seed=seed, outcome=transition.outcome, terminalReward=transition.reward,
                                        totalReward=total_reward, decisions=step,
                                        success=transition.outcome == 'terminated' and transition.reward > 0))
                    break
            if len(results) % 8 == 0:
                publish('evaluating', currentEvaluation=label, evaluated=len(results), successes=sum(row['success'] for row in results))
        del execution, loaded; gc.collect(); mx.clear_cache()
        return dict(checkpoint=str(checkpoint), actionSelection='greedy' if greedy else 'stochastic', trials=results,
                    successes=sum(row['success'] for row in results), episodes=len(results),
                    timeouts=sum(row['outcome'] == 'truncated' for row in results))

    def always_left():
        results = []
        # The baseline never reads pixels, labels, cue color or oracle actions.
        for seed in heldout:
            within_budget(); env = PracticeEnvironment(environment); current = env.reset(seed=seed)
            while True:
                commands = [] if env.elapsed_ms < environment.cue_ms + environment.delay_ms else [
                    dict(operation='keyUp', offsetMs=0, keyCode=123), dict(operation='keyDown', offsetMs=0, keyCode=123)]
                transition = env.step(commands, episode_id=current.episode_id); current = transition.observation
                if transition.outcome != 'continuing':
                    results.append(dict(seed=seed, success=transition.outcome == 'terminated' and transition.reward > 0)); break
        return dict(trials=results, successes=sum(row['success'] for row in results), episodes=len(results))

    report['memorylessBaseline'] = always_left()
    try:
        if not args.skip_baseline_policy:
            report['closedLoop']['initialGreedy'] = trials(initial_path, heldout, label='initialGreedy')
        trainer = BehaviorTrainer(policy, training, dataset_id='practice-memory-' + environment.signature)
        training_deadline = min(deadline - 120, time.perf_counter() + args.training_budget_seconds)
        last_progress = 0.
        def progress(metrics):
            nonlocal last_progress
            if time.perf_counter() - last_progress >= 10:
                last_progress = time.perf_counter(); publish('training', latestMetrics=asdict(metrics))
            else:
                report['latestMetrics'] = asdict(metrics)
        interrupted = False
        for _ in range(args.epochs):
            try:
                metrics = trainer.train_epoch(source, cancelled=lambda: time.perf_counter() >= training_deadline or (args.max_updates is not None and trainer.updates >= args.max_updates), on_metrics=progress)
                report['epochs'].append(asdict(metrics)); publish('epochCompleted', completedEpochs=trainer.epoch)
            except InterruptedError:
                interrupted = True; break
        checkpoint_path = args.output.parent / str(uuid.uuid4())
        save_checkpoint(checkpoint_path, policy, kind='behavioral', step=trainer.updates, training_state=trainer.state,
                        metrics=dict(epoch=trainer.epoch, updates=trainer.updates, decisions=trainer.decisions), training_config=asdict(training))
        publish('checkpointSaved', checkpoint=str(checkpoint_path), trainingInterrupted=interrupted,
                updates=trainer.updates, trainingDecisions=trainer.decisions, completedEpochs=trainer.epoch,
                updateTargetReached=args.max_updates is not None and trainer.updates == args.max_updates)
        del trainer, policy; gc.collect(); mx.clear_cache()
        report['closedLoop']['trainedGreedy'] = trials(checkpoint_path, heldout, label='trainedGreedy')
        publish('completed')
    except InterruptedError as error:
        publish('budgetEnded', issue=str(error))
    except BaseException as error:
        publish('failed', issue=str(error)); raise


if __name__ == '__main__': main()
