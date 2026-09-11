#!/usr/bin/env python3
"""Real local BC -> checkpoint -> closed-loop practice qualification.

Practice oracle data has explicit fixture provenance. These short runs establish
integration and numerical learning, not generalization to arbitrary desktop use.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small", action="store_true", help="Numerical-test layout; does not qualify default capacity")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx
    from astra.checkpoints import save_checkpoint, load_checkpoint
    from astra.model.config import ModelConfig
    from astra.model.policy import AgentPolicy
    from astra.model.actions import flatten_visual
    from astra.data.actions import decode_commands
    from astra.data.observations import make_observation
    from astra.environments.practice import PracticeEnvironment, PracticeConfig
    from astra.environments.demonstrations import PracticeDemonstrations
    from astra.learning.behavioral import BehaviorTrainer, BehaviorConfig

    mx.random.seed(834)
    config = ModelConfig.test_small() if args.small else ModelConfig()
    environment = PracticeConfig(pixel_width=64, pixel_height=64, logical_bounds=(0, 0, 64, 64), time_limit_ms=1500) if args.small else PracticeConfig(time_limit_ms=1500)
    source = PracticeDemonstrations(environment=environment, model=config, seeds_by_split={"train": [7, 8, 9], "validation": [70], "test": [700]})
    policy = AgentPolicy(config, source.vocabulary)
    if not args.small:
        policy.vision.backbone.load_pretrained(ROOT / "vendor/weights/convnext_tiny.safetensors")
    training = BehaviorConfig(epochs=args.epochs, lanes=1, sequence_length=2, learning_rate=2e-3 if args.small else 3e-4)
    trainer = BehaviorTrainer(policy, training, dataset_id="practice-oracle-qualification")
    before = trainer.evaluate(source, split="train")
    reports = []
    started = time.perf_counter()
    for _ in range(args.epochs):
        metrics = trainer.train_epoch(source)
        reports.append(asdict(metrics))
        print(json.dumps({"epoch": metrics.epoch, "meanNLL": metrics.mean_nll, "updates": metrics.updates}), flush=True)
    after = trainer.evaluate(source, split="train")
    validation = trainer.evaluate(source)
    if not after["meanNLL"] < before["meanNLL"]:
        raise RuntimeError("Actual behavioral updates did not reduce demonstration NLL")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output.parent / str(uuid.uuid4())
    checkpoint_manifest = save_checkpoint(checkpoint_path, policy, kind="behavioral", step=trainer.updates, training_state=trainer.state,
                                          metrics={"trainMeanNLL": after["meanNLL"]}, training_config=asdict(training))
    restored = load_checkpoint(checkpoint_path, include_training=True)
    trials = []
    for seed in [7, 8, 9, 700, 701, 702]:
        env = PracticeEnvironment(environment)
        current = env.reset(seed=seed)
        state = None; previous_events = []; steps = 0
        while True:
            observation = make_observation([(current.pixels, current.metadata)], current.control_state,
                                            cutoff_nanos=current.metadata["observedNanos"], elapsed_seconds=config.period_ms / 1000,
                                            reset=steps == 0, config=config, executed_events=previous_events)
            encoding = restored.policy(observation, state)
            sampled = restored.policy.sample(encoding, key=mx.random.key(seed + steps), greedy=True)
            commands = decode_commands(sampled.packets, config=config, vocabulary=source.vocabulary,
                                       visual=flatten_visual(encoding.visual), surfaces=[current.metadata["surface"]])
            state = tuple(mx.stop_gradient(value) for value in encoding.temporal.state)
            transition = env.step(commands, episode_id=current.episode_id)
            current = transition.observation; previous_events = transition.raw_events; steps += 1
            if transition.outcome != "continuing":
                trials.append({"seed": seed, "split": "train" if seed < 10 else "heldOut", "outcome": transition.outcome,
                               "reward": transition.reward, "decisions": steps, "success": transition.outcome == "terminated" and transition.reward > 0})
                break
    report = {"configuration": "numerical-test-small" if args.small else "production-default", "provenance": source.provenance,
              "model": config.to_dict(), "training": asdict(training), "before": before, "after": after,
              "validation": validation, "epochs": reports, "checkpoint": str(checkpoint_path),
              "policySignature": checkpoint_manifest["policySignature"], "closedLoopTrials": trials,
              "wallSeconds": time.perf_counter() - started, "peakMemoryBytes": mx.get_peak_memory()}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"beforeMeanNLL": before["meanNLL"], "afterMeanNLL": after["meanNLL"], "closedLoop": trials, "report": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
