#!/usr/bin/env python3
"""Actual default-policy PPO collection, recurrent replay and local update."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--full-defaults", action="store_true")
    parser.add_argument("--episode-ms", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx
    import numpy as np
    from astra.checkpoints import save_checkpoint, load_checkpoint
    from astra.model.config import ModelConfig
    from astra.model.policy import AgentPolicy
    from astra.environments.practice import PracticeEnvironment, PracticeConfig
    from astra.learning.reinforcement import ReinforcementTrainer, ReinforcementConfig

    mx.random.seed(412)
    episode_ms = args.episode_ms if args.episode_ms is not None else (10000 if args.full_defaults else 1500)
    environment = PracticeEnvironment(PracticeConfig(seed=718, time_limit_ms=episode_ms, shaping_scale=1.0))
    if args.checkpoint:
        loaded = load_checkpoint(args.checkpoint)
        policy = loaded.policy; parent_id = loaded.manifest["id"]
    else:
        policy = AgentPolicy(ModelConfig(), environment.action_vocabulary)
        policy.vision.backbone.load_pretrained(ROOT / "vendor/weights/convnext_tiny.safetensors")
        parent_id = None
    config = ReinforcementConfig() if args.full_defaults else ReinforcementConfig(rollout_decisions=args.decisions, epochs=2, sequence_length=4,
                                                                                burn_in=2, effective_batch_decisions=8)
    trainer = ReinforcementTrainer(policy, environment, config)
    before = np.asarray(policy.actions.operation_head.weight).copy()
    reports = []
    started = time.perf_counter()
    for iteration in range(args.iterations):
        def collected(decision):
            if decision % 32 == 0:
                print(json.dumps({"iteration": iteration, "collected": decision}), flush=True)
        result = trainer.run_iteration(on_decision=collected, on_update=lambda update: print(json.dumps({"optimizerUpdate": update}), flush=True))
        reports.append(asdict(result.metrics))
        print(json.dumps({"iteration": iteration, "metrics": asdict(result.metrics)}), flush=True)
    if np.array_equal(before, np.asarray(policy.actions.operation_head.weight)):
        raise RuntimeError("PPO did not update the actual policy")
    trainer.stop("Qualification finished")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.parent / str(uuid.uuid4())
    manifest = save_checkpoint(checkpoint, policy, kind="reinforcement", step=trainer.optimizer_updates, parent_id=parent_id,
                               training_state=trainer.state, training_config=asdict(config))
    restored = load_checkpoint(checkpoint, include_training=True)
    if restored.manifest["policySignature"] != manifest["policySignature"]:
        raise RuntimeError("PPO checkpoint reload changed the policy configuration")
    report = {"model": policy.config.to_dict(), "reinforcement": asdict(config), "environment": environment.config.to_dict(),
              "initialization": "behavioral" if args.checkpoint else "fresh policy with pretrained vision", "iterations": reports,
              "checkpoint": str(checkpoint), "wallSeconds": time.perf_counter() - started,
              "peakMemoryBytes": mx.get_peak_memory(), "updatedPolicy": True, "checkpointReloaded": True}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "wallSeconds": report["wallSeconds"]}, indent=2))


if __name__ == "__main__":
    main()
