#!/usr/bin/env python3
"""Resumable frozen-visual GRU loss/clipping diagnosis; never production defaults.

Audit uses CPU only. Initial preparation saves untrained heads. Every training
invocation runs one explicit seed/phase for at most six minutes, then saves a
complete optimizer boundary. Evaluation never claims autonomous behavior.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
SEEDS = (834, 835, 836)
DELAYS = (2000, 8000, 30000)
ARMS = {'A': {'perChoice': False, 'clipNorm': 1.}, 'B': {'perChoice': True, 'clipNorm': 1.},
        'C': {'perChoice': True, 'clipNorm': 27.}}
SOURCE_FILES = ('experiments/temporal_study.py', 'scripts/qualify_memory.py', 'python/astra/model/temporal.py',
    'python/astra/model/actions.py', 'python/astra/model/policy.py', 'python/astra/data/observations.py',
    'python/astra/learning/optimizers.py', 'python/astra/learning/backward.py')


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path):
    with Path(path).open('rb') as file: return hashlib.file_digest(file, 'sha256').hexdigest()


def publish(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + str(uuid.uuid4()) + '.tmp')
    temporary.write_bytes(encoded(value)); temporary.replace(path)


def read(path): return json.loads(Path(path).read_bytes())


def checkpoint_reference(path):
    path = Path(path).resolve(); manifest = read(path / 'manifest.json')
    if manifest['id'] != path.name or manifest['schemaVersion'] != 1: raise ValueError('Checkpoint identity changed')
    for name, artifact in manifest['artifacts'].items():
        if name not in ('policy.safetensors', 'training.json', 'training.safetensors'): raise ValueError('Unexpected checkpoint artifact')
        target = path / name
        if target.is_symlink() or target.stat().st_size != artifact['bytes'] or digest(target) != artifact['sha256']:
            raise ValueError('Checkpoint bytes changed: ' + str(target))
    return {'path': str(path), 'manifestSHA256': digest(path / 'manifest.json'),
            'policySHA256': manifest['artifacts']['policy.safetensors']['sha256']}


def check_reference(reference):
    if checkpoint_reference(reference['path']) != reference: raise ValueError('Pinned checkpoint changed')


def metadata(value):
    if set(value) == {'scalar'}: return value['scalar']
    if set(value) == {'dict'}: return {k: metadata(v) for k, v in value['dict'].items()}
    if set(value) == {'list'}: return [metadata(v) for v in value['list']]
    if set(value) == {'tuple'}: return tuple(metadata(v) for v in value['tuple'])
    raise ValueError('Expected scalar-only experimental metadata')


def schedule(groups, count):
    import numpy as np
    result = []
    for update in range(count):
        epoch, position = divmod(update, len(groups))
        order = np.random.default_rng(np.random.SeedSequence([834, epoch])).permutation(len(groups))
        result.append([row['id'] for row in groups[int(order[position])]])
    return result


def dataset(path, common, *, replay):
    path = Path(path).resolve(); manifest = read(path)
    if (manifest['model'] != common['model'] or manifest['actions'] != common['actions'] or
        manifest.get('paired') is not True or bool(manifest.get('cueReplay')) != replay or len(manifest['episodes']) != 192):
        raise ValueError('The retained paired dataset does not match the original study')
    groups = {}; visual_keys = set(); ids = set()
    for entry in manifest['episodes']:
        if digest(entry['path']) != entry['sha256']: raise ValueError('Source episode checksum failed')
        episode = read(entry['path'])
        if episode['id'] != entry['id'] or episode['id'] in ids or bool(episode.get('cueReplay')) != replay:
            raise ValueError('Duplicated or changed source episode')
        ids.add(episode['id']); groups.setdefault(episode['pairID'], []).append(entry)
        visual_keys.update(step['visual'] for step in episode['steps'])
    if len(groups) != 96: raise ValueError('Preparation needs exactly96 complete cue pairs')
    for pair in groups.values():
        if len(pair) != 2: raise ValueError('Incomplete opposite-cue pair')
        one, two = map(lambda entry: read(entry['path']), pair)
        if [one['counterfactual'], two['counterfactual']] != [False, True] or len(one['steps']) != len(two['steps']):
            raise ValueError('The original pair order or duration changed')
        cue_repeat = (one['delayMS'] + 500) // 100 - 1
        for index, (left, right) in enumerate(zip(one['steps'], two['steps'], strict=True)):
            if any(left[key] != right[key] for key in ('controls', 'surface', 'observedNanos')):
                raise ValueError('Cue pair changed causal controls, geometry or time')
            visible = index < 5 or (replay and index == cue_repeat)
            if (left['visual'] != right['visual']) != visible: raise ValueError('Cue pair contains an unintended visual difference')
        if any(sum(bool(row['commands']) for row in item['steps']) != 1 for item in (one, two)):
            raise ValueError('Every source episode must have one supervised choice packet')
    return {'path': str(path), 'sha256': digest(path), 'groups': list(groups.values())}, manifest['visualDigest'], visual_keys


def audit(args):
    output = args.root.resolve()
    if output.exists(): raise ValueError('Choose a new campaign directory; audit never replaces a plan')
    common_ref = checkpoint_reference(args.common)
    common = read(Path(common_ref['path']) / 'manifest.json')
    if common['model'].get('schema_version') != 2: raise ValueError('This study is fixed to the existing schema2 GRU')
    prep, visual_digest, keys = dataset(args.preparation_dataset, common, replay=True)
    arm_data, other_digest, arm_keys = dataset(args.delayed_dataset, common, replay=False)
    if visual_digest != other_digest: raise ValueError('Frozen visual maps differ')
    keys |= arm_keys
    cache_root = args.visual_root.resolve() / visual_digest
    from astra.model.config import ModelConfig
    model_signature = ModelConfig.from_dict(common['model']).signature
    visual_bytes = 0
    for key in sorted(keys):
        descriptor = read(cache_root / (key + '.json')); tensor = cache_root / (key + '.safetensors')
        if (descriptor['visualDigest'] != visual_digest or descriptor['modelSignature'] != model_signature or
            descriptor['bytes'] != tensor.stat().st_size or descriptor['tensorHash'] != digest(tensor)):
            raise ValueError('Frozen visual cache changed: ' + key)
        visual_bytes += descriptor['bytes']
    groups = [pair for pair in arm_data['groups'] if pair[0]['delayMS'] == 2000 and pair[0]['seed'] in range(4)]
    groups.sort(key=lambda pair: pair[0]['seed'])
    if [pair[0]['seed'] for pair in groups] != list(range(4)): raise ValueError('The four original paired layouts are missing')
    if any(len(read(entry['path'])['steps']) != 27 for pair in groups for entry in pair): raise ValueError('Fixed27-step scale contrast changed')
    arm_data['groups'] = groups
    prep_order, arm_order = schedule(prep['groups'], 192), schedule(groups, 512)
    warm = checkpoint_reference(args.warm834)
    warm_manifest = read(Path(warm['path']) / 'manifest.json')
    tree = read(Path(warm['path']) / 'training.json')['dict']
    params = metadata(tree['parameters']); history = metadata(tree['metricsHistory'])
    expected = {'arm': 'C', 'horizon': 512, 'choiceOnly': True, 'initialization': 'geometric', 'modelSeed': 834,
        'visualDigest': visual_digest, 'learningRate': 3e-4, 'weightDecay': .01, 'gradientNorm': 1.,
        'normalization': 'total_valid_decisions_in_two_complete_episodes', 'pairedData': True, 'cueReplay': True}
    if (metadata(tree['kind']) != 'experimental-temporal' or metadata(tree['updates']) != 192 or
        metadata(tree['datasetHash']) != prep['sha256'] or params != expected or warm_manifest['step'] != 192 or
        warm_manifest['model'] != common['model'] or warm_manifest['actions'] != common['actions'] or
        [row['episodeIDs'] for row in history] != prep_order or [row['update'] for row in history] != list(range(1, 193))):
        raise ValueError('Saved834 C192 does not authenticate the exact planned preparation history')
    plan = {'schemaVersion': 1, 'scope': 'frozen_visual_GRU_loss_clipping_diagnostic', 'productionDefaultsChanged': False,
        'headSeeds': list(SEEDS), 'dataOrderSeed': 834, 'armRNGSeed': 834, 'commonCheckpoint': common_ref,
        'reusedPreparation834': warm, 'visualRoot': str(args.visual_root.resolve()), 'visualDigest': visual_digest,
        'phaseRoot': str(args.phase_root.resolve()), 'preparation': prep, 'delayed': arm_data,
        'preparationOrder': prep_order, 'armOrder': arm_order, 'arms': ARMS,
        'trainingLayouts': [0, 1, 2, 3], 'developmentLayouts': list(range(1000, 1008)), 'reservedTestUsed': False,
        'optimizer': {'learningRate': 3e-4, 'weightDecay': .01, 'epsilon': 1e-8, 'biasCorrection': True, 'clipStabilizer': 1e-6},
        'horizon': 512, 'preparationUpdates': 192, 'armUpdates': 512, 'runnerSHA256': digest(__file__),
        'implementation': {name: digest(ROOT / name) for name in SOURCE_FILES},
        'dependencies': {name: importlib.metadata.version(name) for name in ('mlx', 'numpy')},
        'cpuAudit': {'episodes': 384, 'visualEntries': len(keys), 'visualBytesVerified': visual_bytes,
                     'preparationValidDecisions': sum(row['validDecisions'] for row in history),
                     'preparationChoicePackets': sum(row['selectedDecisions'] for row in history)}}
    output.mkdir(parents=True)
    publish(output / 'plan.json', plan)
    print(json.dumps({'phase': 'audited_cpu_only', 'plan': str(output / 'plan.json'), **plan['cpuAudit']}, sort_keys=True))


def load_plan(root):
    plan = read(root / 'plan.json')
    if plan['runnerSHA256'] != digest(__file__): raise ValueError('Runner changed; create a separately audited campaign')
    for name, expected in plan['implementation'].items():
        if digest(ROOT / name) != expected: raise ValueError('Study implementation changed: ' + name)
    for name, expected in plan['dependencies'].items():
        if importlib.metadata.version(name) != expected: raise ValueError('Study dependency changed: ' + name)
    for source in ('preparation', 'delayed'):
        if digest(plan[source]['path']) != plan[source]['sha256']: raise ValueError('Source dataset manifest changed')
    return plan


@contextmanager
def gpu_owner(root):
    with (root / '.gpu-owner.lock').open('a+b') as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def runtime(plan):
    import mlx.core as mx
    from astra.checkpoints import load_checkpoint
    from experiments.temporal_study import VisualCache
    mx.set_memory_limit(10 * 1024**3); mx.set_cache_limit(64 * 1024**2)
    check_reference(plan['commonCheckpoint'])
    common = load_checkpoint(Path(plan['commonCheckpoint']['path']))
    cache = VisualCache(Path(plan['visualRoot']), common.policy, expected_digest=plan['visualDigest'])
    return mx, common, cache


def head_digest(policy):
    import numpy as np
    from mlx.utils import tree_flatten
    result = hashlib.sha256()
    for name, value in sorted(tree_flatten(policy.parameters())):
        if name.startswith('vision.'): continue
        result.update(name.encode()); result.update(encoded([list(value.shape), str(value.dtype)])); result.update(np.asarray(value).tobytes())
    return result.hexdigest()


def prepare_initials(root, plan):
    from astra.checkpoints import save_checkpoint
    from astra.model.policy import AgentPolicy
    from experiments.temporal_study import VisualCache
    from scripts.qualify_memory import initialize_experimental_retention
    mx, common, cache = runtime(plan)
    path = root / 'initials.json'; result = read(path) if path.exists() else {}
    for seed in SEEDS:
        if str(seed) in result:
            check_reference(result[str(seed)]['checkpoint']); continue
        mx.random.seed(seed)
        policy = AgentPolicy(common.policy.config, common.policy.actions.vocabulary)
        policy.vision = common.policy.vision
        initialize_experimental_retention(policy, 'geometric'); policy.vision.freeze()
        checkpoint = root / 'checkpoints' / str(uuid.uuid4())
        save_checkpoint(checkpoint, policy, kind='initial', step=0, parent_id=common.manifest['id'],
            training_state={'kind': 'experimental-gru-clip-initial', 'modelSeed': seed, 'rng': tuple(mx.random.state)},
            metrics={'scope': plan['scope'], 'modelSeed': seed, 'completedOptimizerUpdates': 0})
        VisualCache(Path(plan['visualRoot']), policy, expected_digest=cache.digest)
        result[str(seed)] = {'checkpoint': checkpoint_reference(checkpoint), 'headSHA256': head_digest(policy)}
        publish(path, result)
        print(json.dumps({'phase': 'initial_prepared', 'modelSeed': seed, 'checkpoint': str(checkpoint), 'optimizerUpdates': 0}), flush=True)
    if len({entry['headSHA256'] for entry in result.values()}) != 3: raise ValueError('Independent head seeds produced duplicate parameters')


def phase_input(root, plan, seed, phase):
    if phase == 'preparation': return read(root / 'initials.json')[str(seed)]['checkpoint']
    if seed == 834: return plan['reusedPreparation834']
    report = read(root / f'run-{seed}-preparation.json')
    if report['completedUpdates'] != 192: raise ValueError('Finish the matching192-update preparation before any arm')
    return report['checkpoint']


def phase_spec(plan, seed, phase):
    arm = {'perChoice': False, 'clipNorm': 1.} if phase == 'preparation' else plan['arms'][phase]
    return {'planSHA256': hashlib.sha256(encoded(plan)).hexdigest(), 'modelSeed': seed, 'phase': phase,
            'targetUpdates': 192 if phase == 'preparation' else 512, 'horizon': 512, **arm}


def load_group(plan, phase, update):
    source, order_name = ('preparation', 'preparationOrder') if phase == 'preparation' else ('delayed', 'armOrder')
    entries = {row['id']: row for pair in plan[source]['groups'] for row in pair}
    episodes = []
    for identifier in plan[order_name][update]:
        entry = entries[identifier]
        if digest(entry['path']) != entry['sha256']: raise ValueError('Selected immutable episode changed')
        episodes.append(read(entry['path']))
    return episodes


def train(root, plan, seed, phase, seconds, stop_after):
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from astra.checkpoints import load_checkpoint, save_checkpoint, restore_mlx_random_state
    from astra.learning.optimizers import GroupedAdamW, finite_gradients
    from experiments.temporal_study import FrozenHead, episode_batch, cached_batch_gradients, VisualCache
    if seed == 834 and phase == 'preparation': raise ValueError('Seed834 uses the authenticated saved C192 preparation; no duplicate training')
    began = time.perf_counter(); deadline = began + seconds; interrupted = False
    def stop_signal(*_):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM, stop_signal); signal.signal(signal.SIGINT, stop_signal)
    expired = lambda: interrupted or time.perf_counter() >= deadline
    mx, common, cache = runtime(plan); spec = phase_spec(plan, seed, phase)
    report_path = root / f'run-{seed}-{phase}.json'
    prior = read(report_path) if report_path.exists() else None
    source_ref = prior['checkpoint'] if prior else phase_input(root, plan, seed, phase)
    check_reference(source_ref); loaded = load_checkpoint(Path(source_ref['path']), include_training=True)
    VisualCache(Path(plan['visualRoot']), loaded.policy, expected_digest=cache.digest)
    loaded.policy.vision.freeze(); head = FrozenHead(loaded.policy); head.train()
    optimizer = GroupedAdamW(learning_rate=3e-4, pretrained_learning_rate=3e-5, weight_decay=.01)
    completed, metrics = 0, []
    if prior:
        state = loaded.training_state
        if state['kind'] != 'experimental-gru-loss-clipping' or state['specification'] != spec: raise ValueError('Checkpoint belongs to another seed, arm or plan')
        optimizer.state = state['optimizer']; completed = state['completedUpdates']; metrics = state['metrics']
        restore_mlx_random_state(state['rng'])
        if completed != prior['completedUpdates'] or len(metrics) != completed: raise ValueError('Report and saved optimizer boundary disagree')
    elif phase == 'preparation': restore_mlx_random_state(loaded.training_state['rng'])
    else: mx.random.seed(plan['armRNGSeed'])  # Fresh identical moments/RNG; preparation optimizer is intentionally discarded.
    ceiling = min(spec['targetUpdates'], stop_after or spec['targetUpdates'])
    if ceiling <= completed:
        print(json.dumps({'phase': 'already_at_boundary', 'seed': seed, 'arm': phase, 'completedUpdates': completed})); return
    failure = None; last_saved = completed if prior else -1
    def save(status):
        nonlocal last_saved, source_ref
        if completed != last_saved:
            destination = root / 'checkpoints' / str(uuid.uuid4())
            state = {'kind': 'experimental-gru-loss-clipping', 'specification': spec, 'completedUpdates': completed,
                     'optimizer': optimizer.state, 'rng': tuple(mx.random.state), 'metrics': metrics}
            save_checkpoint(destination, loaded.policy, kind='behavioral', step=completed + (0 if phase == 'preparation' else 192), parent_id=Path(source_ref['path']).name,
                training_state=state, training_config=spec, metrics={'scope': plan['scope'], 'phase': phase, 'modelSeed': seed})
            source_ref = checkpoint_reference(destination); last_saved = completed
        report = {'schemaVersion': 1, 'scope': plan['scope'], 'specification': spec, 'phase': status, 'checkpoint': source_ref,
            'completedUpdates': completed, 'completedTotalHeadUpdates': completed + (0 if phase == 'preparation' else 192), 'validDecisions': sum(row['validDecisions'] for row in metrics),
            'supervisedPackets': sum(row['supervisedPackets'] for row in metrics), 'metrics': metrics,
            'validationPending': True, 'wallSecondsThisInvocation': time.perf_counter() - began,
            'peakMLXBytes': mx.get_peak_memory(), 'productionDefaultsChanged': False, 'issue': failure}
        publish(report_path, report)
    try:
        while completed < ceiling:
            if expired(): raise InterruptedError('Bounded study deadline or stop request')
            episodes = load_group(plan, phase, completed); batch = episode_batch(cache, episodes)
            started = time.perf_counter()
            result = cached_batch_gradients(head, batch, horizon=512, choice_only=True, cancelled=expired)
            scale = result.valid_count / result.selected_count if spec['perChoice'] else 1.
            gradients = tree_map(lambda value: value * scale, result.gradients)
            if not finite_gradients(gradients) or not bool(mx.isfinite(result.loss).item()): raise FloatingPointError('Nonfinite diagnostic gradient/loss')
            clipped, norm = optim.clip_grad_norm(gradients, spec['clipNorm']); mx.eval(clipped, norm)
            group_norms = {}
            for prefix in ('temporal.input_projection', 'temporal.layers.0', 'temporal.layers.1', 'actions'):
                leaves = [value for name, value in tree_flatten(gradients) if name.startswith(prefix)]
                group_norms[prefix] = float(mx.sqrt(sum(mx.sum(value * value) for value in leaves)).item()) if leaves else 0.
            pre_norm = float(norm.item()); post_norm = float(mx.sqrt(sum(mx.sum(value * value) for _, value in tree_flatten(clipped))).item())
            metric = {'update': completed + 1, 'episodeIDs': [episode['id'] for episode in episodes],
                'layoutSeed': episodes[0]['seed'], 'delayMS': episodes[0]['delayMS'], 'validDecisions': result.valid_count,
                'supervisedPackets': result.selected_count, 'lossDenominator': result.selected_count if spec['perChoice'] else result.valid_count,
                'choiceNLL': float(result.loss.item()) * result.valid_count / result.selected_count,
                'gradientScale': scale, 'preClipNorm': pre_norm, 'postClipNorm': post_norm,
                'clipMultiplier': min(spec['clipNorm'] / (pre_norm + 1e-6), 1.), 'groupGradientNorms': group_norms,
                'cueSummaryGradientNorm': float(mx.sqrt(mx.sum(result.summary_gradient[:, :5] ** 2)).item()) * scale,
                'updateSeconds': None}
            before, moments = head.parameters(), deepcopy(optimizer.state)
            try:
                optimizer.update(head, clipped); mx.eval(head.parameters(), optimizer.state)
                if not finite_gradients(head.parameters()): raise FloatingPointError('Nonfinite updated parameters')
            except BaseException:
                head.update(before); optimizer.state = moments; raise
            metric['updateSeconds'] = time.perf_counter() - started
            metrics.append(metric); completed += 1
            if completed % 64 == 0:
                save('training'); print(json.dumps({'phase': phase, 'seed': seed, 'completedUpdates': completed, 'choiceNLL': metrics[-1]['choiceNLL']}), flush=True)
    except InterruptedError as error: failure = str(error)
    except BaseException as error:
        failure = str(error); save('failed'); raise
    save('completed' if completed == spec['targetUpdates'] else 'paused')
    print(json.dumps({'phase': phase, 'seed': seed, 'completedUpdates': completed, 'targetReached': completed == spec['targetUpdates'], 'report': str(report_path)}), flush=True)


def evaluate(root, plan, seed, phase, seconds):
    import numpy as np
    from types import SimpleNamespace
    from astra.checkpoints import load_checkpoint
    from astra.data.actions import decode_commands
    from experiments.temporal_study import FrozenHead, paired_validation, VisualCache
    mx, _, cache = runtime(plan)
    from astra.environments.practice import PracticeConfig, PracticeEnvironment
    from scripts.qualify_memory import memory_phase_fixture
    reference = plan['reusedPreparation834'] if seed == 834 and phase == 'preparation' else read(root / f'run-{seed}-{phase}.json')['checkpoint']
    check_reference(reference); loaded = load_checkpoint(Path(reference['path'])); head = FrozenHead(loaded.policy); head.eval()
    VisualCache(Path(plan['visualRoot']), loaded.policy, expected_digest=cache.digest)
    deadline = time.perf_counter() + seconds
    class CacheView:
        surfaces = None
        def __getattr__(self, name): return getattr(cache, name)
        def batch(self, keys):
            self.surfaces = [read(cache.root / (key + '.json'))['sourceSurface'] for key in keys]
            return cache.batch(keys)
    view = CacheView()
    class PacketWorld(PracticeEnvironment):
        # Unused returned rasters need not be regenerated during passive
        # stepping or execution. Physics, admission, clocks and input cleanup
        # still run through the real adapter. The choice raster is independently
        # re-rendered and checked before any sampled packet is executed.
        omit_unused_raster = False
        retained_raster = None
        def _render(self):
            return self.retained_raster if self.omit_unused_raster else super()._render()
    bases = {}; phases_by_seed = {}
    def execute_packet(commands, layout, delay, lane):
        key = (layout, delay, lane)
        if key not in bases:
            if bases and next(iter(bases))[:2] != (layout, delay): bases.clear()
            if layout not in phases_by_seed:
                phases_by_seed[layout] = memory_phase_fixture(
                    PracticeConfig(task='delayed_memory', delay_ms=30000, time_limit_ms=32500), layout, Path(plan['phaseRoot']))
            phases, labels = phases_by_seed[layout]
            world = PacketWorld(PracticeConfig(task='delayed_memory', delay_ms=delay, time_limit_ms=delay + 2500))
            raw = world.reset(seed=layout)
            if lane: world._answer = 1 - world._answer
            world.retained_raster = phases[2].pixels; world.omit_unused_raster = True
            while world.elapsed_ms < world.config.cue_ms + delay: world.step([], episode_id=raw.episode_id)
            world.omit_unused_raster = False; actual = world._observe()
            if (not np.array_equal(actual.pixels, phases[3].pixels) or actual.metadata['surface'] != phases[3].metadata['surface'] or
                any(actual.control_state[name] != phases[3].control_state[name] for name in actual.control_state if name != 'observedNanos') or
                world.oracle_commands() != labels[lane]):
                raise ValueError('Readiness-gated execution does not match the exact ranked visual/control source')
            world.omit_unused_raster = True; bases[key] = world
        world = deepcopy(bases[key]); total_reward = 0.; statuses = []
        try:
            steps = (world.config.lead_ms + world.config.period_ms + world.config.period_ms - 1) // world.config.period_ms + 1
            for index in range(steps):
                transition = world.step(commands if index == 0 else [], episode_id=world.episode_id)
                total_reward += transition.reward; statuses.extend(row['status'] for row in transition.command_results)
                if transition.outcome != 'continuing': break
            outcome = 'success' if world.outcome == 'terminated' and total_reward > 0 else (
                'wrong_choice' if world.outcome == 'terminated' else 'no_choice')
            return {'scope': 'one_actual_packet_after_passive_readiness', 'outcome': outcome,
                    'reward': total_reward, 'commandStatuses': statuses}
        except ValueError as error: return {'scope': 'one_actual_packet_after_passive_readiness', 'outcome': 'invalid_packet', 'issue': str(error)}
        finally:
            if world.outcome == 'continuing': world.abort('Readiness-gated single-packet diagnostic ended')
    class ActionProbe:
        def __init__(self): self.vocabulary = head.actions.vocabulary; self.rows = []; self.good = None
        def log_prob(self, context, visual, labels):
            scored = head.actions.log_prob(context, visual, labels)
            mx.eval(scored.log_probability, scored.factor_log_probabilities)
            factors = {name: np.asarray(mx.sum(value, axis=-1)).tolist() for name, value in scored.factor_log_probabilities.items()}
            if self.good is None:
                layouts = [*plan['trainingLayouts'], *plan['developmentLayouts']]
                layout = layouts[len(self.rows) // 3]; delay = DELAYS[len(self.rows) % 3]
                row = {'correctLogProbability': np.asarray(scored.log_probability).tolist(), 'correctFactors': factors, 'readoutPackets': []}
                for draw, greedy in enumerate((True, False, False)):
                    sampled = head.actions.sample(context, visual, key=mx.random.key(71000 + len(self.rows) * 3 + draw), greedy=greedy)
                    mx.eval(sampled.packets.operation, sampled.log_probability)
                    for lane in (0, 1):
                        try:
                            commands = decode_commands(sampled.packets, config=head.config, vocabulary=self.vocabulary,
                                visual=visual, surfaces=[view.surfaces[lane]], batch_index=lane)
                            item = {'validWirePacket': True, 'commands': commands, 'execution': execute_packet(commands, layout, delay, lane)}
                        except ValueError as error: item = {'validWirePacket': False, 'issue': str(error)}
                        row['readoutPackets'].append({'greedy': greedy, 'draw': draw, 'cueLane': lane,
                            'logProbability': float(sampled.log_probability[lane].item()),
                            'firstOperation': int(sampled.packets.operation[lane, 0].item()), **item})
                self.good = row
            else:
                self.good['wrongLogProbability'] = np.asarray(scored.log_probability).tolist()
                self.good['factorMargins'] = {name: (np.asarray(self.good['correctFactors'][name]) - values).tolist() for name, values in factors.items()}
                self.rows.append(self.good); self.good = None
            return scored
    probe = ActionProbe(); seeds = [*plan['trainingLayouts'], *plan['developmentLayouts']]
    result = paired_validation(SimpleNamespace(config=head.config, temporal=head.temporal, actions=probe), view,
        Path(plan['phaseRoot']), seeds=seeds, cue_replay=False, cancelled=lambda: time.perf_counter() >= deadline)
    if len(result['rows']) != len(probe.rows) or probe.good is not None: raise ValueError('Scoring probe lost its exact paired validation order')
    for row, scores in zip(result['rows'], probe.rows, strict=True): row.update(scores)
    result['splitSummaries'] = [{'split': split, 'delayMS': delay, 'layouts': len(allowed),
        'pairedAccuracy': sum(row['correct'] for row in result['rows'] if row['seed'] in allowed and row['delayMS'] == delay) / (2 * len(allowed))}
        for split, allowed in [('train', plan['trainingLayouts']), ('development', plan['developmentLayouts'])] for delay in DELAYS]
    result.update(modelSeed=seed, phase=phase, checkpoint=reference, schemaVersion=1,
        readoutScope='Greedy and two sampled packets executed individually after an independently verified passive choice-window history; no further policy decisions; not autonomous success',
        reservedTestUsed=False, completedOptimizerUpdatesDuringEvaluation=0)
    destination = root / 'evaluations' / (str(uuid.uuid4()) + '.json'); publish(destination, result)
    print(json.dumps({'evaluation': str(destination), 'seed': seed, 'phase': phase, 'summaries': result['splitSummaries']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('audit', 'initials', 'train', 'evaluate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--phase', choices=('preparation', 'A', 'B', 'C'))
    parser.add_argument('--seconds', type=float, default=360)
    parser.add_argument('--stop-after', type=int, help='Pause at this absolute completed update, without changing the fixed study target')
    parser.add_argument('--common', type=Path, default=ROOT / '.local/verification/3ea5799a-2b26-4e7a-912a-ad32412ad171')
    parser.add_argument('--warm834', type=Path, default=ROOT / '.local/temporal-immediate-study/checkpoints/2531737d-4b0c-44f0-b058-66f87378d335')
    parser.add_argument('--preparation-dataset', type=Path, default=ROOT / '.local/temporal-immediate-study/dataset.json')
    parser.add_argument('--delayed-dataset', type=Path, default=ROOT / '.local/temporal-paired-study/dataset.json')
    parser.add_argument('--visual-root', type=Path, default=ROOT / '.local/temporal-study/visual')
    parser.add_argument('--phase-root', type=Path, default=ROOT / '.local/temporal-study/phase-fixtures')
    args = parser.parse_args(); args.root = args.root.resolve()
    if not 1 <= args.seconds <= 360 or args.stop_after is not None and args.stop_after < 1: parser.error('Bounded invocations need1–360 seconds and a positive optional stop boundary')
    if args.mode == 'audit': audit(args); return
    plan = load_plan(args.root)
    if args.mode in ('train', 'evaluate') and (args.seed is None or args.phase is None): parser.error('Select one explicit seed and phase')
    with gpu_owner(args.root):
        if args.mode == 'initials': prepare_initials(args.root, plan)
        elif args.mode == 'train': train(args.root, plan, args.seed, args.phase, args.seconds, args.stop_after)
        else: evaluate(args.root, plan, args.seed, args.phase, args.seconds)


if __name__ == '__main__': main()
