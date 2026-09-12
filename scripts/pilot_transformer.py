#!/usr/bin/env python3
"""Bounded numerical/footprint pilot for the experimental causal transformer.

This performs two real longest-episode optimizer updates per objective, not a
learning campaign. It never writes a standard GRU checkpoint or app catalog.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common-checkpoint', type=Path, default=ROOT / '.local/verification/3ea5799a-2b26-4e7a-912a-ad32412ad171')
    parser.add_argument('--visual-root', type=Path, default=ROOT / '.local/temporal-study/visual')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=180)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 360:
        parser.error('The complete pilot process is bounded to at most six minutes')
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from dataclasses import asdict
    from mlx.utils import tree_flatten
    from astra.checkpoints import load_checkpoint
    from astra.environments.practice import PracticeConfig
    from astra.model.observation import ObservationBatch
    from astra.learning.optimizers import GroupedAdamW, finite_gradients
    from experiments.causal_transformer import ExperimentalTransformerHead, TransformerSpecification
    from experiments.temporal_study import VisualCache, prepare_episode, episode_batch, cached_batch_gradients

    started = time.perf_counter()
    deadline = started + args.seconds
    mx.set_memory_limit(10 * 1024**3)
    mx.set_cache_limit(64 * 1024**2)
    report = dict(schemaVersion=1, scope='experimental_transformer_longest_update_pilot_only',
                  commonCheckpoint=str(args.common_checkpoint), modelSeed=834, specification=asdict(TransformerSpecification()),
                  objectives=[], completed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def tensor_bytes(tree):
        return sum(value.nbytes for _, value in tree_flatten(tree))

    def digest(tree):
        result = hashlib.sha256()
        for name, value in sorted(tree_flatten(tree)):
            result.update(name.encode())
            result.update(str(value.shape).encode())
            result.update(str(value.dtype).encode())
            result.update(np.asarray(value).tobytes())
        return result.hexdigest()

    def publish(phase):
        report.update(phase=phase, wallSeconds=time.perf_counter() - started,
            peakMLXBytes=mx.get_peak_memory(), processPeakRSSBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024))
        temporary = args.output.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output)
        print(json.dumps(dict(phase=phase, wallSeconds=report['wallSeconds'], objectives=report['objectives'])), flush=True)

    def cancelled():
        return time.perf_counter() >= deadline

    try:
        common = load_checkpoint(args.common_checkpoint)
        cache = VisualCache(args.visual_root, common.policy)
        environment = PracticeConfig(task='delayed_memory', delay_ms=30000, time_limit_ms=32500)
        episodes = [prepare_episode(cache, environment, 200, counterfactual=opposite, cancelled=cancelled) for opposite in (False, True)]
        batch = episode_batch(cache, episodes)
        assert batch.observation.shape == (2, 307)
        report.update(visualDigest=cache.digest, fullPixelDimensions=[1280, 720],
            validDecisions=int(mx.sum(batch.observation.valid).item()), source='two complete opposite-cue ordinary episodes, seed 200, delay 30 s',
            sharedActionDigest=digest(common.policy.actions.parameters()),
            sharedValueDigest=digest(dict(hidden=common.policy.temporal.value_hidden.parameters(), output=common.policy.temporal.value_head.parameters())),
            originalGRUTemporalParameterBytes=tensor_bytes(common.policy.temporal.parameters()),
            originalGRUActorStateBytes=tensor_bytes(common.policy.temporal.initial_state(1)))
        publish('prepared')
        for choice_only in (True, False):
            if cancelled(): raise InterruptedError('Pilot deadline before complete objective')
            loaded = load_checkpoint(args.common_checkpoint)
            mx.random.seed(834)
            head = ExperimentalTransformerHead(loaded.policy)
            head.train()
            assert digest(head.actions.parameters()) == report['sharedActionDigest']
            assert digest(dict(hidden=head.temporal.value_hidden.parameters(), output=head.temporal.value_head.parameters())) == report['sharedValueDigest']
            assert digest(loaded.policy.vision.parameters()) == digest(common.policy.vision.parameters())
            optimizer = GroupedAdamW(learning_rate=3e-4, pretrained_learning_rate=3e-5, weight_decay=.01)
            row = dict(choiceOnly=choice_only, normalization='total valid decisions in same two complete episodes',
                       sharedInitialDecoderAndValueBitwiseEqual=True, visualWeightsBitwiseEqual=True,
                       temporalParameterBytes=tensor_bytes(head.temporal.parameters()),
                       actionParameterBytes=tensor_bytes(head.actions.parameters()),
                       actorCacheAllocatedBytes=tensor_bytes(head.temporal.initial_state(1)),
                       trainingCacheAllocatedBytes=tensor_bytes(head.temporal.initial_state(2)), updates=[])
            report['objectives'].append(row)
            mx.eval(head.parameters())
            gc.collect(); mx.clear_cache(); mx.reset_peak_memory()
            for update in range(2):
                before = head.temporal.output_projection.weight
                began = time.perf_counter()
                result = cached_batch_gradients(head, batch, horizon=512, choice_only=choice_only, cancelled=cancelled)
                if not bool(mx.isfinite(result.loss).item()) or not finite_gradients(result.gradients):
                    raise FloatingPointError('Nonfinite transformer pilot loss/gradient')
                gradient, norm = optim.clip_grad_norm(result.gradients, 1)
                optimizer.update(head, gradient)
                mx.eval(head.parameters(), optimizer.state)
                elapsed = time.perf_counter() - began
                change = mx.max(mx.abs(head.temporal.output_projection.weight - before))
                if not float(change.item()) > 0:
                    raise ValueError('Transformer optimizer did not change its output projection')
                row['updates'].append(dict(update=update + 1, seconds=elapsed,
                    loss=float(result.loss.item()), meanSelectedPacketNLL=float((result.loss * result.valid_count / result.selected_count).item()),
                    gradientNorm=float(norm.item()), earlyCueGradientNorm=float(mx.linalg.norm(result.summary_gradient[:, :5]).item()),
                    outputProjectionMaxChange=float(change.item()), peakMLXBytes=mx.get_peak_memory(),
                    optimizerTensorBytes=tensor_bytes(optimizer.state), selectedPackets=result.selected_count))
                publish('updated')
            del head, optimizer, loaded, result, gradient, before
            gc.collect(); mx.clear_cache()
        # Isolated temporal-only actor overhead with an occupied cache. This
        # excludes visual encoding, the decoder, IPC and capture; no end-to-end
        # throughput claim follows from it.
        loaded = load_checkpoint(args.common_checkpoint)
        mx.random.seed(834)
        head = ExperimentalTransformerHead(loaded.policy)
        head.eval()
        single = ObservationBatch((), batch.observation.controls[:1], batch.observation.elapsed_seconds[:1],
            batch.observation.context_ids[:1], batch.observation.reset[:1], batch.observation.valid[:1])
        prefix = head.temporal(batch.summary[:1], single)
        carried = ObservationBatch((), single.controls, single.elapsed_seconds, single.context_ids,
            mx.zeros_like(single.reset), single.valid)
        state = head.temporal(batch.summary[:1], carried, prefix.state).state
        mx.eval(state)
        if int(mx.sum(state[2]).item()) != 512: raise ValueError('Actor pilot cache was not fully occupied')
        observation = carried.slice_time(306, 307)
        times = []
        for index in range(22):
            if cancelled(): raise InterruptedError('Pilot deadline during actor timing')
            began = time.perf_counter()
            result = head.temporal(batch.summary[:1, 306:307], observation, state)
            mx.eval(result.context, result.value, result.state)
            state = tuple(mx.stop_gradient(value) for value in result.state)
            elapsed = time.perf_counter() - began
            if index >= 2: times.append(elapsed)
        report['actorTemporalOnly'] = dict(warmSamples=20, occupiedCacheObservations=512,
            medianMS=float(np.median(times) * 1000), p95MS=float(np.percentile(times, 95) * 1000),
            maxMS=max(times) * 1000, stateTensorBytes=tensor_bytes(state))
        report['completed'] = True
        publish('complete')
    except InterruptedError as error:
        report['reason'] = str(error)
        publish('paused')
    except Exception as error:
        report['reason'] = f'{type(error).__name__}: {error}'
        publish('failed')
        raise


if __name__ == '__main__':
    main()
