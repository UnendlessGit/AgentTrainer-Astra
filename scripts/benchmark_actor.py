#!/usr/bin/env python3
"""Compare actor execution on native-size synthetic frames, without TCC.

The measured interval includes an owned BGRA ingestion copy, exact Metal
preprocessing, policy, full-budget decoder, finite checks and validated wire
conversion. It excludes SCK, native ring publication and subprocess pipe waits.
"""
from pathlib import Path
import argparse
import gc
import json
import platform
from importlib.metadata import version
import resource
import statistics
import time

import mlx.core as mx
import numpy as np

from astra.data.actions import decode_commands
from astra.data.observations import make_observation
from astra.environments.practice import PracticeConfig, PracticeEnvironment
from astra.inference import _PolicyExecution, prepare_metal_surface
from astra.model.actions import ActionVocabulary, PacketBatch
from astra.model.config import ModelConfig
from astra.model.policy import AgentPolicy
from astra.model.vision import VisualFeatures


def summary(samples):
    return dict(medianMS=statistics.median(samples), p95MS=float(np.percentile(samples, 95)), maximumMS=max(samples), samplesMS=samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repetitions', type=int, default=12)
    parser.add_argument('--modes', nargs='+', choices=('eager', 'compiled'), default=['eager', 'compiled'])
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--small', action='store_true', help='Numerical development fixture; never a default-policy throughput claim')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.repetitions < 10: parser.error('At least ten warm repetitions are required')
    mx.set_memory_limit(2500 * 1024**2); mx.set_cache_limit(64 * 1024**2)
    config = ModelConfig.test_small() if args.small else ModelConfig()
    environment = PracticeEnvironment(PracticeConfig(pixel_width=1280, pixel_height=720))
    frames = [environment.reset(seed=seed) for seed in (807, 808, 809)]
    vocabulary = ActionVocabulary(key_codes=(0, 13), mouse_buttons=(0,), absolute_pointer=True, relative_pointer=True, scroll=True)
    mx.random.seed(171)
    policy = AgentPolicy(config, vocabulary)
    if not args.small:
        policy.vision.backbone.load_pretrained(Path(__file__).resolve().parents[1] / 'vendor/weights/convnext_tiny.safetensors')
    policy.eval(); mx.eval(policy.parameters())
    runners = {mode: _PolicyExecution(policy, greedy=args.greedy, compiled=mode == 'compiled') for mode in args.modes}
    results = {}; references = []
    for mode in args.modes:
        gc.collect(); mx.clear_cache(); mx.reset_peak_memory()
        state = policy.temporal.initial_state(1); key = mx.random.key(817)
        samples = []; phases = []; compared = []; cold = []
        for index in range(args.repetitions + 3):
            raw = frames[index % len(frames)]
            start = time.perf_counter()
            owned = mx.array(raw.pixels)
            observation = make_observation([(owned, raw.metadata)], raw.control_state, cutoff_nanos=raw.metadata['observedNanos'],
                                           elapsed_seconds=.1, reset=index == 0, config=config, surface_preparer=prepare_metal_surface)
            built = time.perf_counter()
            output = runners[mode](observation, state, key)
            graphed = time.perf_counter()
            mx.eval(output)
            evaluated = time.perf_counter()
            if not bool(output['finite'].item()): raise AssertionError('Nonfinite policy output')
            commands = decode_commands(PacketBatch(**output['packet']), config=config, vocabulary=vocabulary,
                                       visual=VisualFeatures(**output['visual']), surfaces=[raw.metadata['surface']])
            encoded = json.dumps(dict(commands=commands, value=float(output['value'].item()),
                                      logProbability=float(output['log_probability'].item()),
                                      conditionalEntropy=float(output['conditional_entropy'].item())), allow_nan=False)
            finished = time.perf_counter()
            state, key = output['state'], output['key']
            snapshot = (commands, np.asarray(output['log_probability']).copy(), np.asarray(output['value']).copy(),
                        [np.asarray(value).copy() for value in state])
            compared.append(snapshot)
            if index < 3: cold.append((finished - start) * 1000)
            if index >= 3:
                samples.append((finished - start) * 1000)
                phases.append(dict(prepareGraphMS=(built-start)*1000, policyGraphMS=(graphed-built)*1000,
                                   evaluateMS=(evaluated-graphed)*1000, wireMS=(finished-evaluated)*1000))
            print(f'{mode} step {index}: {(finished-start)*1000:.3f} ms, commands {len(commands)}, bytes {len(encoded)}', flush=True)
        results[mode] = {**summary(samples), 'initialSamplesMS': cold, 'phaseMediansMS': {name: statistics.median(item[name] for item in phases) for name in phases[0]},
                         'peakMLXBytes': mx.get_peak_memory(), 'processPeakRSSBytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)}
        if not references: references = compared
        else:
            errors = dict(logProbability=0., value=0., state=0.)
            for expected, actual in zip(references, compared):
                assert actual[0] == expected[0], 'Compilation changed a sampled action packet'
                for label, left, right in [('logProbability', expected[1], actual[1]), ('value', expected[2], actual[2]),
                                          *[('state', a, b) for a, b in zip(expected[3], actual[3])]]:
                    errors[label] = max(errors[label], float(np.max(np.abs(left - right))))
                    np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-5 if label != 'logProbability' else 1e-4)
            results[mode]['maximumAbsoluteErrors'] = errors
            results[mode]['packetsIdentical'] = True
    report = dict(schemaVersion=1, platform=platform.platform(), machine=platform.machine(), mlxVersion=version('mlx'),
                  device=mx.device_info(), pixelSize=[1280,720], model=config.to_dict(), parameterCount=config.parameter_count,
                  greedy=args.greedy, sourceVariants=len(frames), repetitions=args.repetitions, results=results,
                  scope='Owned BGRA ingestion through validated JSON actions; no SCK, native ring publication or IPC timing.')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))

if __name__ == '__main__': main()
