#!/usr/bin/env python3
"""Isolated, guarded full-policy backward memory qualification on Apple Silicon.

Repeated real practice pixels isolate sequence-memory scaling; this is neither
a learning-quality benchmark nor a recording of the user's desktop. MLX's
memory limit is only a guideline, so a worker watchdog enforces a lower stop
threshold and the parent does not launch projected-unsafe larger graphs.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
GIB = 1024**3


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def memory(mx):
    return {"activeBytes": mx.get_active_memory(), "cacheBytes": mx.get_cache_memory(),
            "peakActiveBytes": mx.get_peak_memory(),
            "maximumRSSBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}


def worker(args):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    import numpy as np
    from astra.data.batching import LearningSample, training_batch
    from astra.data.observations import make_observation
    from astra.environments.practice import PracticeConfig, PracticeEnvironment
    from astra.learning.optimizers import GroupedAdamW
    from astra.learning.backward import policy_gradients
    from astra.model.config import ModelConfig
    from astra.model.policy import AgentPolicy

    mx.set_memory_limit(int(args.memory_gib * GIB))
    mx.set_cache_limit(int(args.cache_gib * GIB))
    stop = threading.Event()
    phase = ["initialization"]
    highest = {"activeBytes": 0, "cacheBytes": 0, "maximumRSSBytes": 0}
    guard = int(args.guard_gib * GIB)

    def monitor():
        previous = 0
        while not stop.wait(0.05):
            values = memory(mx)
            for name in highest:
                highest[name] = max(highest[name], values[name])
            if values["activeBytes"] > guard or values["maximumRSSBytes"] > guard:
                emit({"kind": "guard", "phase": phase[0], "sequenceLength": args.worker_length,
                      "reason": "Memory watchdog stopped this worker before the configured ceiling", **values})
                os._exit(72)
            if values["peakActiveBytes"] >= previous + GIB:
                previous = values["peakActiveBytes"]
                emit({"kind": "memory", "phase": phase[0], **values})

    watchdog = threading.Thread(target=monitor, name="memory-watchdog", daemon=True)
    watchdog.start()
    mx.random.seed(493)
    model_config = ModelConfig()
    environment = PracticeEnvironment(PracticeConfig(pixel_width=1280, pixel_height=720))
    raw = environment.reset(seed=42)
    policy = AgentPolicy(model_config, environment.action_vocabulary)
    policy.vision.backbone.load_pretrained(ROOT / "vendor/weights/convnext_tiny.safetensors")
    policy.configure_execution(vision_microbatch=1, checkpoint_vision=True)
    policy.eval()
    prepared = make_observation([(raw.pixels, raw.metadata)], raw.control_state,
                                cutoff_nanos=raw.metadata["observedNanos"], elapsed_seconds=0.1,
                                reset=True, config=model_config)
    commands = ({"operation": "pointerAbsolute", "surfaceID": "practice", "x": .37, "y": .61, "offsetMs": 0},
                {"operation": "buttonUp", "button": 0, "offsetMs": 20},
                {"operation": "buttonDown", "button": 0, "offsetMs": 30},
                {"operation": "buttonUp", "button": 0, "offsetMs": 80})
    samples = [LearningSample(replace(prepared, reset=mx.array([[index == 0]])),
                              (raw.metadata["surface"],), commands, raw.episode_id, index)
               for index in range(args.worker_length)]
    phase[0] = "observation batching"
    observations, packets = training_batch([samples for _ in range(args.lanes)], model_config, environment.action_vocabulary)
    # Match the persistent actor, rollback parameters, and optimizer/rollback
    # slots retained by an established PPO update, rather than benchmarking
    # one model in an otherwise empty process.
    actor = copy.deepcopy(policy)
    rollback_weights = copy.deepcopy(policy.parameters())
    optimizer = GroupedAdamW(learning_rate=1e-4, pretrained_learning_rate=1e-5)
    flat = dict(tree_flatten(policy.trainable_parameters()))
    for group, instance in optimizer.optimizers.items():
        instance.init({name: value for name, value in flat.items() if optimizer.group(name, value) == group})
    mx.eval(policy.parameters(), actor.parameters(), rollback_weights, optimizer.state, observations.as_tensors(), packets.operation)
    rollback_optimizer = copy.deepcopy(optimizer.state)
    mx.eval(rollback_optimizer)
    mx.clear_cache()
    initial = memory(mx)
    mx.reset_peak_memory()
    emit({"kind": "started", "sequenceLength": args.worker_length, "lanes": args.lanes, "mode": args.mode,
          "device": mx.device_info(), "modelSignature": model_config.signature,
          "parameterCount": model_config.parameter_count,
          "inputShapes": {name: list(getattr(observations.surfaces[0], name).shape)
                          for name in ("global_image", "detail_image", "cursor_image")},
          "initialMemory": initial})

    def loss_fn(model):
        encoding = model(observations)
        scores = model.score(encoding, packets)
        # Full contiguous TBPTT loss, including real spatial pointer arguments
        # and the value branch. No frozen/detached visual parameters.
        return -mx.mean(scores.log_probability) + 0.5 * mx.mean(mx.square(encoding.temporal.value))

    phase[0] = "backward"
    started = time.perf_counter()
    def differentiate():
        if args.mode == "current":
            return nn.value_and_grad(policy, loss_fn)(policy)
        def objective(logp, values, entropy):
            return -mx.mean(logp) + 0.5 * mx.mean(values ** 2), mx.array(0)
        result = policy_gradients(policy, observations, packets, objective)
        return result.loss, result.gradients
    value, gradients = differentiate()
    mx.eval(value, gradients)
    elapsed = time.perf_counter() - started
    phase[0] = "verification"
    finite = all(bool(mx.all(mx.isfinite(leaf)).item()) for _, leaf in tree_flatten(gradients))
    if not finite or not np.isfinite(float(value)):
        raise FloatingPointError("The production recurrent backward produced nonfinite loss or gradients")
    # Deep copies initially share immutable MLX buffers. A real update creates
    # distinct learner parameters/moments while actor and rollback stay alive.
    # Retain the first gradient as an accumulation buffer across the next chunk.
    if args.mode == "staged":
        phase[0] = "optimizer update"
        clipped, norm = optim.clip_grad_norm(gradients, 0.5)
        optimizer.update(policy, clipped)
        mx.eval(policy.parameters(), optimizer.state)
        phase[0] = "second backward with actor, rollback and accumulated gradient"
        second_value, second_gradients = differentiate()
        mx.eval(second_value, second_gradients)
        if not all(bool(mx.all(mx.isfinite(leaf)).item()) for _, leaf in tree_flatten(second_gradients)):
            raise FloatingPointError("Second production gradient is not finite")
    after = memory(mx)
    gradient_norms = {name: float(mx.linalg.norm(leaf)) for name, leaf in tree_flatten(gradients)
                      if name in {"vision.backbone.downsamples.0.conv.weight", "vision.detail.convolutions.0.weight",
                                  "temporal.layers.0.Wx", "temporal.layers.1.Wh", "actions.cell_query.weight"}}
    stop.set()
    watchdog.join(timeout=1)
    emit({"kind": "result", "sequenceLength": args.worker_length, "lanes": args.lanes, "mode": args.mode,
          "seconds": elapsed, "loss": float(value), "finiteGradients": finite,
          "gradientNorms": gradient_norms, "initialMemory": initial, "finalMemory": after,
          "sampledMaxima": highest})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--mode", choices=["current", "staged"], default="staged")
    parser.add_argument("--lanes", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--memory-gib", type=float, default=20)
    parser.add_argument("--guard-gib", type=float, default=18)
    parser.add_argument("--cache-gib", type=float, default=.5)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-length", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.guard_gib < args.memory_gib <= 22 or not 0 <= args.cache_gib <= 2:
        parser.error("Memory limits must satisfy 1 <= guard < limit <= 22 GiB and cache <= 2 GiB")
    if any(length not in (1, 2, 4, 8, 16, 32, 64) for length in args.lengths) or args.lengths != sorted(set(args.lengths)):
        parser.error("Lengths must be increasing unique powers of two through 64")
    if args.worker_length:
        worker(args)
        return
    report = {"kind": "training-memory", "mode": args.mode, "lanes": args.lanes, "memoryLimitBytes": int(args.memory_gib * GIB),
              "guardBytes": int(args.guard_gib * GIB), "cacheLimitBytes": int(args.cache_gib * GIB),
              "source": "Repeated real 1280x720 practice observation; full model visual sizes, contiguous recurrent loss",
              "includesActorOptimizerAndRollback": True, "runs": []}
    previous = None
    guarded = False
    for length in args.lengths:
        projected = None if previous is None else previous["finalMemory"]["peakActiveBytes"] * length / previous["sequenceLength"]
        if guarded or (projected is not None and projected > args.guard_gib * GIB):
            item = {"sequenceLength": length, "status": "not_run", "reason": "Earlier measured memory predicts an unsafe graph",
                    "linearPeakProjectionBytes": projected}
            report["runs"].append(item)
            emit(item)
            continue
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-length", str(length), "--mode", args.mode,
                   "--lanes", str(args.lanes),
                   "--memory-gib", str(args.memory_gib), "--guard-gib", str(args.guard_gib), "--cache-gib", str(args.cache_gib)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = process.communicate(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            guarded = True
        events = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
        result = next((event for event in reversed(events) if event.get("kind") == "result"), None)
        item = result or {"sequenceLength": length, "status": "stopped", "returncode": process.returncode,
                          "events": events, "error": stderr[-4000:]}
        report["runs"].append(item)
        emit(item)
        if result is None:
            guarded = True
        else:
            previous = result
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
