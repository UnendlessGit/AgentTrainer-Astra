#!/usr/bin/env python3
"""One frozen delayed readout of the saved zero-wait endpoint; no training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
from scripts.gru_loss_clipping import read, publish, digest, check_reference
from scripts.gru_extended_wait import load_plan as load_evaluation_plan, evaluate


def prepare(args):
    root = args.root.resolve()
    if root.exists(): raise ValueError('Choose a new readout directory; preserve all evidence')
    evaluation_root, zero_root = args.extended_wait.resolve(), args.zero_wait.resolve()
    evaluation_plan = load_evaluation_plan(evaluation_root)
    zero = read(zero_root / 'result.json')
    zero_plan = read(zero_root / 'plan.json')
    if (zero['completedUpdates'] != 512 or zero['supervisedPackets'] != 1024 or
            zero['planSHA256'] != digest(zero_root / 'plan.json') or
            evaluation_plan['zeroWaitResultSHA256'] != digest(zero_root / 'result.json') or
            zero_plan['sourceCheckpoint'] != evaluation_plan['sourceCheckpoint']):
        raise ValueError('The selected fixed zero-wait endpoint or common starting checkpoint changed')
    check_reference(zero['checkpoint'])
    manifest = read(Path(zero['checkpoint']['path']) / 'manifest.json')
    original = read(Path(evaluation_plan['sourceCheckpoint']['path']) / 'manifest.json')
    if manifest['model'] != original['model'] or manifest['actions'] != original['actions']:
        raise ValueError('The readout cannot change the model or action vocabulary')
    plan = {'schemaVersion': 1, 'scope': 'saved_zero_wait_endpoint_at_original_delays', 'optimizerUpdates': 0,
        'checkpoint': zero['checkpoint'], 'zeroWaitResultPath': str(zero_root / 'result.json'),
        'zeroWaitResultSHA256': digest(zero_root / 'result.json'), 'evaluationPlanPath': str(evaluation_root / 'plan.json'),
        'evaluationPlanSHA256': digest(evaluation_root / 'plan.json'), 'evaluationRunnerSHA256': digest(ROOT / 'scripts/gru_extended_wait.py'),
        'runnerSHA256': digest(__file__), 'trainingLayouts': evaluation_plan['trainingLayouts'],
        'developmentLayouts': evaluation_plan['developmentLayouts'], 'delaysMS': [2000, 30000],
        'readout': evaluation_plan['readout'], 'sampling': evaluation_plan['evaluationSampling'],
        'reservedTestUsed': False, 'productionDefaultsChanged': False,
        'interpretation': 'Readout only: distinguish transport through added waiting from optimization through that history; no endpoint retry or new training'}
    root.mkdir(parents=True); publish(root / 'plan.json', plan)
    print(json.dumps({'phase': 'prepared_cpu_only', 'planSHA256': digest(root / 'plan.json'),
                      'runnerSHA256': plan['runnerSHA256'], 'optimizerUpdates': 0}), flush=True)


def run(root):
    plan = read(root / 'plan.json')
    if plan['runnerSHA256'] != digest(__file__) or plan['evaluationRunnerSHA256'] != digest(ROOT / 'scripts/gru_extended_wait.py'):
        raise ValueError('Frozen readout implementation changed')
    for path, expected in ((plan['evaluationPlanPath'], plan['evaluationPlanSHA256']),
                           (plan['zeroWaitResultPath'], plan['zeroWaitResultSHA256'])):
        if digest(path) != expected: raise ValueError('Frozen source document changed')
    check_reference(plan['checkpoint'])
    evaluation_plan = load_evaluation_plan(Path(plan['evaluationPlanPath']).parent)
    if (root / 'started.json').exists(): raise ValueError('One readout only; do not replace its result')
    publish(root / 'started.json', {'planSHA256': digest(root / 'plan.json'), 'optimizerUpdates': 0})
    import mlx.core as mx
    from astra.checkpoints import load_checkpoint
    from experiments.temporal_study import FrozenHead, VisualCache
    started = time.perf_counter(); mx.set_memory_limit(10 * 1024**3); mx.set_cache_limit(64 * 1024**2)
    loaded = load_checkpoint(Path(plan['checkpoint']['path']))
    cache = VisualCache(Path(evaluation_plan['visualRoot']), loaded.policy, expected_digest=evaluation_plan['visualDigest'])
    endpoint = evaluate(FrozenHead(loaded.policy), cache, evaluation_plan)
    publish(root / 'endpoint.json', endpoint)
    check_reference(plan['checkpoint'])
    result = {'schemaVersion': 1, 'scope': plan['scope'], 'planSHA256': digest(root / 'plan.json'),
        'runnerSHA256': digest(__file__), 'checkpoint': plan['checkpoint'], 'optimizerUpdates': 0,
        'endpoint': endpoint['summaries'], 'checkpointBytesUnchanged': True,
        'wallSeconds': time.perf_counter() - started, 'peakMLXBytes': mx.get_peak_memory(),
        'reservedTestUsed': False, 'productionDefaultsChanged': False}
    publish(root / 'result.json', result); print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run')); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--extended-wait', type=Path, default=ROOT / '.local/gru-extended-wait-2026-09-27')
    parser.add_argument('--zero-wait', type=Path, default=ROOT / '.local/gru-zero-wait-2026-09-27')
    args = parser.parse_args()
    if args.mode == 'prepare': prepare(args)
    else: run(args.root.resolve())


if __name__ == '__main__': main()
