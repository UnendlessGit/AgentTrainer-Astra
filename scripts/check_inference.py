#!/usr/bin/env python3
"""Qualify an actor executable with a real native ring and local checkpoint.

The development Python creates small fixtures and runs the verifier. The target
worker receives only system PATH, no venv/PYTHONPATH, and optionally no network.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time


def qualify_local(root):
    """One synthetic native-size observation; no training or screen access."""
    import mlx.core as mx
    import numpy as np
    from astra.data.actions import decode_commands
    from astra.data.observations import make_observation
    from astra.environments.practice import PracticeConfig, PracticeEnvironment
    from astra.inference import prepare_metal_surface
    from astra.model.actions import flatten_visual
    from astra.model.config import ModelConfig
    from astra.model.policy import AgentPolicy

    mx.set_memory_limit(2 * 1024**3)
    mx.set_cache_limit(64 * 1024**2)
    environment = PracticeEnvironment(PracticeConfig(pixel_width=1280, pixel_height=720))
    raw = environment.reset(seed=807)
    config = ModelConfig()
    options = dict(cutoff_nanos=raw.metadata["observedNanos"], elapsed_seconds=.1, reset=True, config=config)
    expected = make_observation([(raw.pixels, raw.metadata)], raw.control_state, **options)
    started = time.perf_counter()
    owned = mx.array(raw.pixels)
    actual = make_observation([(owned, raw.metadata)], raw.control_state, surface_preparer=prepare_metal_surface, **options)
    mx.eval(actual.as_tensors())
    elapsed = time.perf_counter() - started
    differences = {}
    for field in expected.surfaces[0].__dataclass_fields__:
        left, right = np.asarray(getattr(expected.surfaces[0], field)), np.asarray(getattr(actual.surfaces[0], field))
        differences[field] = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-6, err_msg=field)
    mx.random.seed(171)
    policy = AgentPolicy(config, environment.action_vocabulary)
    policy.vision.backbone.load_pretrained(root / "vendor/weights/convnext_tiny.safetensors")
    policy.eval()
    def evaluate(observation):
        encoding = policy(observation)
        sampled = policy.sample(encoding, key=mx.random.key(3), greedy=True)
        mx.eval(encoding.temporal.state, encoding.temporal.value, sampled.log_probability)
        commands = decode_commands(sampled.packets, config=config, vocabulary=environment.action_vocabulary,
                                   visual=flatten_visual(encoding.visual), surfaces=[raw.metadata["surface"]])
        return encoding, sampled, commands
    left, left_action, left_commands = evaluate(expected)
    right, right_action, right_commands = evaluate(actual)
    state_error = max(float(mx.max(mx.abs(a - b)).item()) for a, b in zip(left.temporal.state, right.temporal.state))
    value_error = float(mx.max(mx.abs(left.temporal.value - right.temporal.value)).item())
    logp_error = float(mx.max(mx.abs(left_action.log_probability - right_action.log_probability)).item())
    assert state_error < 2e-5 and value_error < 2e-5 and logp_error < 1e-4
    assert left_commands == right_commands
    return {"pixelSize": [1280, 720], "parameterCount": config.parameter_count,
            "preprocessingMaximumAbsoluteErrors": differences, "commandsIdentical": True,
            "stateMaximumAbsoluteError": state_error, "valueAbsoluteError": value_error, "jointLogProbabilityAbsoluteError": logp_error,
            "singleIngestionAndPreparationSeconds": elapsed, "peakMLXActiveBytes": mx.get_peak_memory(),
            "note": "One synthetic frame and pretrained visual initialization; not a latency distribution or learned behavior qualification."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path, nargs="?")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--qualify-local", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.binary is None and not args.qualify_local:
        parser.error("Provide a worker executable or --qualify-local")
    if args.binary is None and args.offline:
        parser.error("--offline applies to the supplied worker executable")
    result = {}
    if args.binary is not None:
        binary = args.binary.resolve(strict=True)
        environment = dict(os.environ)
        environment["ASTRA_COMPUTE_TEST_BINARY"] = str(binary)
        environment["ASTRA_COMPUTE_TEST_OFFLINE"] = "1" if args.offline else "0"
        subprocess.run([sys.executable, "-W", "error", "-m", "pytest",
                        "python/tests/test_inference.py::test_worker_actor_role_runs_real_ring_checkpoint_and_returns_release_on_error", "-q"],
                       cwd=root, env=environment, check=True, timeout=120)
        result["worker"] = {"binary": str(binary), "offline": args.offline, "passed": True}
    if args.qualify_local:
        result["localQualification"] = qualify_local(root)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
