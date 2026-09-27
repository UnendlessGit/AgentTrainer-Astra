#!/usr/bin/env python3
"""One matched continuation at the original two-second wait; GRU schema 2 only."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
from scripts.gru_loss_clipping import encoded, digest, publish, read, check_reference, checkpoint_reference, schedule
from scripts.gru_zero_wait import visual_key, control_features
from astra.environments.practice import PracticeConfig, PracticeEnvironment

TRAIN = (0, 1, 2, 3)
DEV = tuple(range(1000, 1008))
DELAYS = (2000, 30000)
IMPLEMENTATION = ('scripts/gru_loss_clipping.py', 'scripts/gru_zero_wait.py', 'experiments/temporal_study.py',
    'python/astra/environments/practice.py', 'python/astra/data/history.py', 'python/astra/model/temporal.py',
    'python/astra/model/policy.py', 'python/astra/model/actions.py', 'python/astra/model/observation.py',
    'python/astra/learning/optimizers.py', 'python/astra/data/observations.py')


class PassiveWorld(PracticeEnvironment):
    """Cache only repeated renders of the same actual phase/cursor/cue.

    The first raster of each phase is produced by the real renderer. Clock,
    scheduler, control state and oracle behavior are never skipped or replaced.
    """
    def _render(self):
        phase = 0 if self.elapsed_ms < self.config.cue_ms else (1 if self.elapsed_ms < self.config.cue_ms + self.config.delay_ms else 2)
        key = (phase, self._pointer, self._answer)
        if key != getattr(self, '_render_key', None):
            self._render_key, self._render_pixels = key, super()._render()
        return self._render_pixels


def world(seed, lane, delay):
    env = PassiveWorld(PracticeConfig(task='delayed_memory', delay_ms=delay, time_limit_ms=delay + 2500))
    raw = env.reset(seed)
    if lane:
        env._answer = 1 - env._answer
        raw = env._observe()
    return env, raw


def prepare(args):
    import numpy as np
    root = args.root.resolve()
    if root.exists(): raise ValueError('Use a new output directory; never replace evidence')
    campaign, zero_root = args.campaign.resolve(), args.zero_wait.resolve()
    old, zero = read(campaign / 'plan.json'), read(zero_root / 'plan.json')
    zero_result, source = read(zero_root / 'result.json'), read(campaign / 'run-836-B.json')
    if source['phase'] != 'completed' or source['completedUpdates'] != 512 or source['checkpoint'] != zero['sourceCheckpoint']:
        raise ValueError('The matched continuation must start the same original 836-B checkpoint')
    if zero_result['planSHA256'] != digest(zero_root / 'plan.json') or zero_result['completedUpdates'] != 512:
        raise ValueError('The fixed zero-wait comparator is incomplete or changed')
    check_reference(source['checkpoint'])
    checkpoint = read(Path(source['checkpoint']['path']) / 'manifest.json')
    if checkpoint['model'].get('schema_version') != 2: raise ValueError('No new model architecture is allowed')
    groups, episodes, references = [], [], []
    for pair in old['delayed']['groups']:
        rows = []
        for entry in pair:
            if digest(entry['path']) != entry['sha256']: raise ValueError('Original training episode changed')
            row = read(entry['path'])
            if row['seed'] not in TRAIN or row['delayMS'] != 2000 or row.get('cueReplay', False) or len(row['steps']) != 27:
                raise ValueError('The original four paired 27-step layouts are required')
            rows.append(row); references.append(entry)
        if [row['counterfactual'] for row in rows] != [False, True]: raise ValueError('Cue lane order changed')
        groups.append(rows); episodes.extend(rows)
    if [pair[0]['seed'] for pair in groups] != list(TRAIN): raise ValueError('Training layout order changed')
    order = schedule(groups, 512)
    by_id = {row['id']: row for row in episodes}; zero_by_id = {row['id']: row for row in zero['episodes']}
    for left, right in zip(order, zero['order'], strict=True):
        if [(by_id[i]['seed'], by_id[i]['counterfactual']) for i in left] != [(zero_by_id[i]['seed'], zero_by_id[i]['counterfactual']) for i in right]:
            raise ValueError('Optimizer exposure/order differs from zero-wait control')
    cache_root = Path(old['visualRoot']) / old['visualDigest']
    tensors, templates, proofs = {}, [], []
    def verify_visual(raw):
        key = visual_key(raw, checkpoint['model'], old['visualDigest'])
        if key in tensors: return key
        descriptor = read(cache_root / (key + '.json')); tensor = cache_root / (key + '.safetensors')
        if descriptor['visualDigest'] != old['visualDigest'] or descriptor['bytes'] != tensor.stat().st_size or descriptor['tensorHash'] != digest(tensor):
            raise ValueError('Frozen visual features changed; do not substitute')
        tensors[key] = {'bytes': descriptor['bytes'], 'sha256': descriptor['tensorHash']}
        return key
    for seed in (*TRAIN, *DEV):
        phase_paths = list(Path(old['phaseRoot']).glob('*/' + str(seed) + '.npz'))
        if len(phase_paths) != 1: raise ValueError('Original phase fixture missing or ambiguous')
        with np.load(phase_paths[0], allow_pickle=False) as phases:
            metadata = json.loads(str(phases['metadata'].item()))
            if metadata['seed'] != seed: raise ValueError('Phase source identity changed')
            keys, controls = [], None
            for name, phase in zip(('cueA', 'cueB', 'wait', 'choice'), metadata['phases'], strict=True):
                from types import SimpleNamespace
                raw = SimpleNamespace(pixels=phases[name], metadata=phase['metadata'], control_state=phase['controlState'])
                keys.append(verify_visual(raw))
                feature = control_features(raw).tolist()
                if controls is not None and feature != controls: raise ValueError('Passive controls are not phase-invariant')
                controls = feature
            for delay in DELAYS:
                for lane in (0, 1):
                    env, raw = world(seed, lane, delay)
                    length = 5 + delay // 100 + 1
                    clock_start = raw.metadata['observedNanos']
                    for index in range(length):
                        phase_index = lane if index < 5 else (2 if index < length - 1 else 3)
                        name = ('cueA', 'cueB', 'wait', 'choice')[phase_index]
                        before = metadata['phases'][phase_index]
                        if (not np.array_equal(raw.pixels, phases[name]) or raw.metadata['surface'] != before['metadata']['surface'] or
                            control_features(raw).tolist() != controls or raw.metadata['observedNanos'] != clock_start + index * 100_000_000):
                            raise ValueError('Rendered original-delay evaluation input differs from frozen source')
                        commands = env.oracle_commands()
                        if commands != (metadata['labels'][lane] if index == length - 1 else []): raise ValueError('Actual readiness/labels changed')
                        if index < length - 1: raw = env.step([], episode_id=raw.episode_id).observation
                    proofs.append({'seed': seed, 'delayMS': delay, 'cueLane': lane, 'actualObservedPrefixLength': length,
                                   'clockStartNanos': clock_start, 'pixelControlGeometryTargetsMatch': True})
            templates.append({'seed': seed, 'visualKeys': keys, 'controls': controls, 'surface': metadata['phases'][3]['metadata']['surface'],
                              'labels': metadata['labels'], 'phasePath': str(phase_paths[0]), 'phaseSHA256': digest(phase_paths[0])})
        if seed in TRAIN:
            for lane in (0, 1):
                original = next(row for row in episodes if row['seed'] == seed and row['counterfactual'] == bool(lane))
                env, raw = world(seed, lane, 2000)
                for step in original['steps']:
                    if (verify_visual(raw), control_features(raw).tolist(), env.oracle_commands(), raw.metadata['surface'], raw.metadata['observedNanos']) != (
                        step['visual'], step['controls'], step['commands'], step['surface'], step['observedNanos']):
                        raise ValueError('Original 27-step training observations or labels changed')
                    transition = env.step(step['commands'], episode_id=raw.episode_id, provenance='oracle'); raw = transition.observation
                if transition.outcome != 'terminated' or transition.reward <= 0: raise ValueError('Original oracle episode did not complete')
    plan = {'schemaVersion': 1, 'scope': 'matched836B_extended_two_second_wait', 'productionDefaultsChanged': False,
        'sourceCheckpoint': source['checkpoint'], 'originalCampaignPlanSHA256': digest(campaign / 'plan.json'),
        'zeroWaitPlanSHA256': digest(zero_root / 'plan.json'), 'zeroWaitResultSHA256': digest(zero_root / 'result.json'),
        'trainingLayouts': TRAIN, 'developmentLayouts': DEV, 'evaluationDelaysMS': DELAYS, 'reservedTestUsed': False,
        'optimizerSeed': zero['optimizerSeed'], 'optimizer': zero['optimizer'], 'targetUpdates': 512, 'targetSupervisedPackets': 1024,
        'trainingStepsPerEpisode': 27, 'validDecisionsPerUpdate': 54, 'supervisedPacketsPerUpdate': 2, 'gradientScale': 27,
        'episodes': episodes, 'order': order, 'sourceEpisodeReferences': references, 'evaluationTemplates': templates, 'renderProofs': proofs,
        'visualRoot': old['visualRoot'], 'visualDigest': old['visualDigest'], 'originalVisualTensors': tensors,
        'runnerSHA256': digest(__file__), 'implementation': {name: digest(ROOT / name) for name in IMPLEMENTATION},
        'dependencies': {name: importlib.metadata.version(name) for name in ('mlx', 'numpy', 'Pillow')},
        'primaryCriteria': ['2s and30s training fit: extra optimization sufficient for this selected checkpoint; no initializer change',
            '2s training fit but30s failure: duration extrapolation, since30s is outside additional2s training',
            '2s training still fails: history-dependent optimization persists; evaluate saved zero-wait endpoint at2/30 before architecture changes'],
        'readout': 'one greedy and two categorical packets after a passive original-delay prefix; only empty packets follow; not autonomous waiting',
        'evaluationSampling': 'fixed71000 + original study layout/delay index*3 + draw; original2s/30s sample keys retained'}
    root.mkdir(parents=True); publish(root / 'plan.json', plan)
    print(json.dumps({'phase': 'prepared_cpu_only', 'planSHA256': digest(root / 'plan.json'), 'runnerSHA256': plan['runnerSHA256'],
                     'verifiedTrainingEpisodes': len(episodes), 'verifiedEvaluationPrefixes': len(proofs), 'visualEntries': len(tensors)}), flush=True)


def load_plan(root):
    plan = read(root / 'plan.json')
    if plan['runnerSHA256'] != digest(__file__): raise ValueError('Frozen runner changed')
    for path, expected in plan['implementation'].items():
        if digest(ROOT / path) != expected: raise ValueError('Frozen implementation changed: ' + path)
    for name, version in plan['dependencies'].items():
        if importlib.metadata.version(name) != version: raise ValueError('Frozen dependency changed')
    for entry in plan['sourceEpisodeReferences']:
        if digest(entry['path']) != entry['sha256']: raise ValueError('Original training source changed')
    check_reference(plan['sourceCheckpoint'])
    return plan


def execute_packet(commands, seed, lane, delay):
    env, raw = world(seed, lane, delay)
    while env.elapsed_ms < env.config.cue_ms + delay: raw = env.step([], episode_id=raw.episode_id).observation
    correct = env._choice_colors.index(env._answer); reward = 0.; statuses = []
    try:
        for index in range(4):
            result = env.step(commands if index == 0 else [], episode_id=raw.episode_id)
            reward += result.reward; statuses.extend(row['status'] for row in result.command_results)
            if result.outcome != 'continuing': break
        outcome = 'success' if env.outcome == 'terminated' and reward > 0 else ('wrong_choice' if env.outcome == 'terminated' else 'no_choice')
        return {'outcome': outcome, 'reward': reward, 'selectedTarget': correct if outcome == 'success' else (1 - correct if outcome == 'wrong_choice' else None), 'commandStatuses': statuses}
    except ValueError as error: return {'outcome': 'invalid_packet', 'issue': str(error), 'selectedTarget': None}
    finally:
        if env.outcome == 'continuing': env.abort('Fixed original-delay packet readout ended')


def evaluate(head, cache, plan):
    import mlx.core as mx
    import numpy as np
    from astra.data.actions import encode_commands, decode_commands
    from astra.model.actions import PacketBatch
    from astra.model.observation import ObservationBatch
    head.eval(); rows = []
    for layout_index, template in enumerate(plan['evaluationTemplates']):
        seed, keys = template['seed'], template['visualKeys']
        cue = mx.concatenate([cache.get(keys[index]).summary for index in (0, 1)])[:, None]
        wait = mx.broadcast_to(cache.get(keys[2]).summary[:, None], cue.shape)
        choice = mx.broadcast_to(cache.get(keys[3]).summary[:, None], cue.shape)
        visual = cache.batch([keys[3], keys[3]])
        labels = [encode_commands(command, config=head.config, vocabulary=head.actions.vocabulary,
                  visual=cache.get(keys[3]), surfaces=[template['surface']]) for command in template['labels']]
        correct = PacketBatch(**{name: mx.concatenate([getattr(label, name) for label in labels]) for name in PacketBatch.__dataclass_fields__})
        wrong = PacketBatch(**{name: getattr(correct, name)[mx.array([1, 0])] for name in PacketBatch.__dataclass_fields__})
        for delay in DELAYS:
            summary = mx.concatenate([mx.repeat(cue, 5, axis=1), mx.repeat(wait, delay // 100, axis=1), choice], axis=1)
            length = summary.shape[1]
            observed = ObservationBatch((), mx.broadcast_to(mx.array(template['controls']), (2, length, head.config.control_width)),
                mx.full((2, length), .1), mx.zeros((2, length, 0), dtype=mx.int32),
                mx.broadcast_to(mx.arange(length) == 0, (2, length)), mx.ones((2, length), dtype=mx.bool_))
            context = head.temporal(summary, observed).context[:, -1]
            good = head.actions.log_prob(context, visual, correct); bad = head.actions.log_prob(context, visual, wrong)
            mx.eval(good.log_probability, bad.log_probability, good.factor_log_probabilities, bad.factor_log_probabilities)
            row = {'seed': seed, 'delayMS': delay, 'split': 'train' if seed in TRAIN else 'development',
                'correctLogProbability': np.asarray(good.log_probability).tolist(), 'wrongLogProbability': np.asarray(bad.log_probability).tolist(),
                'margins': np.asarray(good.log_probability - bad.log_probability).tolist(),
                'factorMargins': {name: np.asarray(mx.sum(value - bad.factor_log_probabilities[name], axis=-1)).tolist() for name, value in good.factor_log_probabilities.items()}, 'readouts': []}
            for draw, greedy in enumerate((True, False, False)):
                sample_key = 71000 + (layout_index * 3 + (0 if delay == 2000 else 2)) * 3 + draw
                sample = head.actions.sample(context, visual, key=mx.random.key(sample_key), greedy=greedy)
                mx.eval(sample.packets.operation, sample.log_probability)
                for lane in (0, 1):
                    try:
                        commands = decode_commands(sample.packets, config=head.config, vocabulary=head.actions.vocabulary,
                            visual=visual, surfaces=[template['surface']], batch_index=lane)
                        result = {'commands': commands, 'execution': execute_packet(commands, seed, lane, delay)}
                    except ValueError as error: result = {'execution': {'outcome': 'invalid_packet', 'selectedTarget': None, 'issue': str(error)}}
                    row['readouts'].append({'cueLane': lane, 'greedy': greedy, 'draw': draw, 'sampleKey': sample_key,
                                           'firstOperation': int(sample.packets.operation[lane, 0]), **result})
            rows.append(row)
    summaries = []
    for split in ('train', 'development'):
        for delay in DELAYS:
            chosen = [row for row in rows if row['split'] == split and row['delayMS'] == delay]
            greedy = [item for row in chosen for item in row['readouts'] if item['greedy']]
            sampled = [item for row in chosen for item in row['readouts'] if not item['greedy']]
            pairs = [[item['execution'] for item in row['readouts'] if item['greedy']] for row in chosen]
            summaries.append({'split': split, 'delayMS': delay, 'layouts': len(chosen), 'greedySuccesses': sum(item['execution']['outcome'] == 'success' for item in greedy),
                'greedyTrials': len(greedy), 'sampledSuccesses': sum(item['execution']['outcome'] == 'success' for item in sampled), 'sampledTrials': len(sampled),
                'bothCuesCorrectLayouts': sum(all(item['outcome'] == 'success' for item in pair) for pair in pairs),
                'selectedTargetChangesWithCue': sum(all(item['selectedTarget'] is not None for item in pair) and pair[0]['selectedTarget'] != pair[1]['selectedTarget'] for pair in pairs),
                'centerPacketRankingCorrect': sum(margin > 0 for row in chosen for margin in row['margins']),
                'firstOperationCounts': {str(op): sum(item['firstOperation'] == op for item in greedy) for op in sorted({item['firstOperation'] for item in greedy})}})
    return {'rows': rows, 'summaries': summaries, 'scope': plan['readout']}


def run(root, plan):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_map
    from astra.checkpoints import load_checkpoint, save_checkpoint
    from astra.learning.optimizers import GroupedAdamW, finite_gradients
    from experiments.temporal_study import FrozenHead, VisualCache, episode_batch, cached_batch_gradients
    if (root / 'started.json').exists(): raise ValueError('One fixed run only; preserve any failed evidence')
    publish(root / 'started.json', {'planSHA256': digest(root / 'plan.json'), 'runnerSHA256': digest(__file__)})
    started = time.perf_counter(); mx.set_memory_limit(10 * 1024**3); mx.set_cache_limit(64 * 1024**2)
    loaded = load_checkpoint(Path(plan['sourceCheckpoint']['path'])); loaded.policy.vision.freeze()
    cache = VisualCache(Path(plan['visualRoot']), loaded.policy, expected_digest=plan['visualDigest'])
    head = FrozenHead(loaded.policy); head.train(); mx.random.seed(plan['optimizerSeed'])
    optimizer = GroupedAdamW(learning_rate=3e-4, pretrained_learning_rate=3e-5, weight_decay=.01)
    episodes = {row['id']: row for row in plan['episodes']}; metrics = []
    for index, identities in enumerate(plan['order']):
        batch = episode_batch(cache, [episodes[identity] for identity in identities])
        result = cached_batch_gradients(head, batch, horizon=512, choice_only=True)
        if (result.valid_count, result.selected_count) != (54, 2): raise ValueError('Matched exposure changed')
        scale = result.valid_count / result.selected_count
        gradients = tree_map(lambda value: value * scale, result.gradients)
        if not finite_gradients(gradients): raise FloatingPointError('Nonfinite diagnostic gradients')
        clipped, norm = optim.clip_grad_norm(gradients, 1.); mx.eval(clipped, norm)
        optimizer.update(head, clipped); mx.eval(head.parameters(), optimizer.state)
        if not finite_gradients(head.parameters()): raise FloatingPointError('Nonfinite diagnostic parameters')
        metrics.append({'update': index + 1, 'episodeIDs': identities, 'validDecisions': 54, 'supervisedPackets': 2,
            'choiceNLL': float(result.loss) * scale, 'gradientScale': scale, 'preClipNorm': float(norm),
            'clipMultiplier': min(1., 1. / (float(norm) + 1e-6)), 'cueSummaryGradientNorm': float(mx.sqrt(mx.sum(result.summary_gradient[:, :5]**2))) * scale})
        if (index + 1) % 64 == 0:
            publish(root / 'progress.json', {'updates': index + 1, 'metrics': metrics, 'seconds': time.perf_counter() - started})
            print(json.dumps({'updates': index + 1, 'choiceNLL': metrics[-1]['choiceNLL']}), flush=True)
    checkpoint = root / 'checkpoints' / str(uuid.uuid4())
    save_checkpoint(checkpoint, loaded.policy, kind='behavioral', step=loaded.manifest['step'] + 512, parent_id=loaded.manifest['id'],
        training_state={'kind': 'experimental-extended-wait-control', 'planSHA256': digest(root / 'plan.json'), 'optimizer': optimizer.state,
                        'rng': tuple(mx.random.state), 'updates': 512, 'metrics': metrics},
        training_config={'diagnostic': plan['scope'], 'planSHA256': digest(root / 'plan.json')})
    endpoint = evaluate(head, cache, plan); publish(root / 'endpoint.json', endpoint)
    result = {'schemaVersion': 1, 'scope': plan['scope'], 'planSHA256': digest(root / 'plan.json'), 'runnerSHA256': digest(__file__),
        'checkpoint': checkpoint_reference(checkpoint), 'completedUpdates': 512, 'supervisedPackets': 1024, 'validDecisions': 27648,
        'metrics': metrics, 'endpoint': endpoint['summaries'], 'wallSeconds': time.perf_counter() - started,
        'peakMLXBytes': mx.get_peak_memory(), 'reservedTestUsed': False, 'productionDefaultsChanged': False}
    publish(root / 'result.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'metrics'}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run')); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, default=ROOT / '.local/gru-loss-clipping-2026-09-27')
    parser.add_argument('--zero-wait', type=Path, default=ROOT / '.local/gru-zero-wait-2026-09-27')
    args = parser.parse_args()
    if args.mode == 'prepare': prepare(args)
    else: run(args.root.resolve(), load_plan(args.root.resolve()))


if __name__ == '__main__': main()
