#!/usr/bin/env python3
"""Experiment-only GRU retention/gradient comparison; no production defaults."""
from pathlib import Path
import argparse
from dataclasses import replace
import json
from types import SimpleNamespace
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from qualify_memory import initialize_experimental_retention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from astra.data.observations import make_observation
    from astra.environments.practice import PracticeConfig, PracticeEnvironment
    from astra.model.config import ModelConfig
    from astra.model.observation import ObservationBatch
    from astra.model.policy import AgentPolicy

    mx.set_memory_limit(1024**3); mx.set_cache_limit(32 * 1024**2)
    began = time.perf_counter()
    # Independent analytic check of the installed gate ordering/sign and VJP.
    gate = nn.GRU(8, 8)
    initialize_experimental_retention(SimpleNamespace(temporal=SimpleNamespace(layers=[gate])), 'geometric')
    gate.Wx = mx.zeros_like(gate.Wx); gate.Wh = mx.zeros_like(gate.Wh); gate.bhn = mx.zeros_like(gate.bhn)
    gate.b = mx.concatenate((mx.zeros(8), gate.b[8:16], mx.zeros(8)))
    zero = mx.zeros((1, 305, 8)); initial = mx.ones((1, 8))
    result = gate(zero, initial)[:, -1]
    gradient = mx.grad(lambda state: mx.sum(gate(zero, state)[:, -1]))(initial)
    expected = (1 - 1 / np.geomspace(1.01, 512, 8)) ** 305
    np.testing.assert_allclose(result[0], expected, rtol=5e-5, atol=1e-7)
    np.testing.assert_allclose(gradient[0], expected, rtol=5e-5, atol=1e-7)

    config = ModelConfig.test_small()
    environment = PracticeConfig(task='delayed_memory', pixel_width=64, pixel_height=64, delay_ms=30000)
    records = []
    for seed in (834, 835, 836):
        for variant in ('native', 'zero-update', 'geometric'):
            mx.random.seed(seed)
            env = PracticeEnvironment(environment); raw = env.reset(seed=1000)
            policy = AgentPolicy(config, env.action_vocabulary); policy.eval()
            initializer = initialize_experimental_retention(policy, variant)
            def observation(value):
                return make_observation([(value.pixels, value.metadata)], value.control_state,
                    cutoff_nanos=value.metadata['observedNanos'], elapsed_seconds=.1, reset=False, config=config)
            cue_a = observation(raw)
            env._answer = 1 - env._answer
            cue_b = observation(env._observe())
            for _ in range(5): raw = env.step([], episode_id=env.episode_id).observation
            wait = observation(raw)
            summaries = [policy.encode_visual(value).summary for value in (cue_a, cue_b, wait)]
            mx.eval(summaries)
            np.testing.assert_array_equal(cue_a.controls, cue_b.controls)
            np.testing.assert_array_equal(cue_a.controls, wait.controls)
            projection = np.random.default_rng(seed).choice([-1., 1.], config.recurrent_width).astype(np.float32) / np.sqrt(config.recurrent_width)
            projection = mx.array(projection.astype(np.float32))
            for delay in (2000, 8000, 30000):
                steps = 5 + delay // 100
                timing = ObservationBatch((), mx.repeat(wait.controls, steps, axis=1), mx.full((1, steps), .1),
                    mx.zeros((1, steps, 0), dtype=mx.int32), mx.arange(steps)[None] == 0, mx.ones((1, steps), dtype=mx.bool_))
                def state(cue):
                    sequence = mx.concatenate((mx.repeat(cue, 5, axis=1), mx.repeat(summaries[2], steps-5, axis=1)), axis=1)
                    return policy.temporal(sequence, timing).state
                a, b = state(summaries[0]), state(summaries[1])
                gradient = mx.grad(lambda cue: mx.sum(state(cue)[-1] * projection))(summaries[0])
                mx.eval(a, b, gradient)
                rms = [float(mx.sqrt(mx.mean((left-right)**2)).item()) for left, right in zip(a,b)]
                norm = float(mx.sqrt(mx.sum(gradient**2)).item())
                assert np.isfinite(norm) and all(np.isfinite(rms))
                record = dict(modelSeed=seed, initialization=variant, delayMS=delay, recurrentRMSDifference=rms,
                              cueSummaryGradientNorm=norm)
                records.append(record)
                print(json.dumps(record), flush=True)
            del policy; mx.clear_cache()
    report = dict(schemaVersion=1, configuration='numerical-test-small', model=config.to_dict(), seeds=[834,835,836],
        variants=['native','zero-update','geometric'], maximumHorizon=512, minimumHorizon=1.01,
        installedGateEquation='h_next=(1-z)*candidate+z*h; b order r,z,n', analyticGateAndGradientCheck=True,
        source='https://arxiv.org/abs/1804.11188',
        scope='Untrained temporal cue sensitivity/VJP on real64px cue/wait encoder features; not learned-policy success.',
        records=records, wallSeconds=time.perf_counter()-began, peakMLXBytes=mx.get_peak_memory())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(report=str(args.output), seconds=report['wallSeconds'])), flush=True)

if __name__ == '__main__': main()
