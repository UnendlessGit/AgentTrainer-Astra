#!/usr/bin/env python3
"""Qualify production PPO through the same asynchronous jobs as the native app.

Uses explicit permission-free practice rollouts. This is a worker/protocol and
checkpoint qualification, not evidence that the policy learned a successful task.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


class Worker:
    def __init__(self, binary: Path | None):
        command = [str(binary)] if binary else [sys.executable, "-m", "astra.worker"]
        environment = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"} if binary else dict(os.environ, PYTHONPATH=str(ROOT / "python"))
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        env=environment)
        self.events = queue.Queue()
        self.sequence = 0
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        hello = self.next()
        if hello["kind"] != "hello" or "train.reinforcement" not in hello["payload"]["capabilities"]:
            raise RuntimeError("Worker does not advertise the qualified reinforcement operation")

    def _read(self):
        try:
            for line in self.process.stdout:
                self.events.put(json.loads(line))
        finally:
            self.events.put(None)

    def next(self):
        event = self.events.get(timeout=1800)
        if event is None:
            raise RuntimeError("Worker exited: " + self.process.stderr.read().decode())
        return event

    def send(self, kind, payload, run=None):
        from astra.protocol import Message
        request = str(uuid.uuid4())
        self.process.stdin.write(Message(kind, self.sequence, payload, request_id=request, run_id=run).encode())
        self.process.stdin.flush()
        self.sequence += 1
        return request

    def job(self, operation, payload, *, cancel_after=None):
        run = str(uuid.uuid4())
        request = self.send(operation, payload, run)
        acknowledgement = self.next()
        if acknowledgement["kind"] != "ack" or acknowledgement.get("requestID") != request or acknowledgement.get("runID") != run:
            raise RuntimeError(f"Uncorrelated job acknowledgement: {acknowledgement}")
        job_id = acknowledgement["payload"]["jobID"]
        cancellation = None
        previous_phase = None
        while True:
            event = self.next()
            if event.get("requestID") == cancellation and cancellation is not None:
                if event["kind"] != "ack":
                    raise RuntimeError(f"Cancellation was rejected: {event}")
                continue
            if event.get("requestID") != request or event.get("runID") != run or event["payload"].get("jobID") != job_id:
                raise RuntimeError(f"Uncorrelated job event: {event}")
            if event["kind"] == "job.progress":
                progress = event["payload"]
                phase = progress.get("phase")
                count = progress.get("validation_decisions", progress.get("audit_tail_decisions", progress.get("rollout_decisions", 0)))
                if phase != previous_phase or count % 64 == 0:
                    print(json.dumps({"operation": operation, "phase": phase, "iteration": progress.get("iteration"),
                                      "collected": progress.get("rollout_decisions"), "updates": progress.get("optimizer_updates"),
                                      "validated": progress.get("validation_decisions"),
                                      "candidateStepScale": progress.get("candidate_step_scale"),
                                      "auditTailDecisions": progress.get("audit_tail_decisions")}), flush=True)
                previous_phase = phase
                if cancel_after is not None and cancellation is None and phase == "collecting" and progress.get("rollout_decisions", 0) >= cancel_after:
                    cancellation = self.send("cancel", {"jobID": job_id}, run)
                continue
            expected = "job.completed" if cancel_after is None else "job.cancelled"
            if event["kind"] != expected:
                raise RuntimeError(f"Unexpected job terminal state: {event}")
            return event["payload"]["result"]

    def close(self):
        try:
            if self.process.poll() is None:
                self.send("shutdown", {})
                self.process.wait(timeout=35)
                if self.process.returncode != 0:
                    raise RuntimeError(self.process.stderr.read().decode())
        finally:
            if self.process.poll() is None:
                self.process.kill(); self.process.wait()
            self.reader.join(timeout=5)
            self.process.stdin.close(); self.process.stdout.close(); self.process.stderr.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-decisions", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--effective-batch-decisions", type=int, default=256)
    parser.add_argument("--episode-ms", type=int, default=120000)
    args = parser.parse_args()
    from astra.environments.practice import PracticeConfig, PracticeEnvironment
    from astra.learning.reinforcement import ReinforcementConfig
    from astra.model.config import ModelConfig
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = {name: args.output.parent / str(uuid.uuid4()) for name in ("initial", "completed", "cancelled", "resumed")}
    model, training = ModelConfig(), replace(ReinforcementConfig(), rollout_decisions=args.rollout_decisions,
        epochs=args.epochs, effective_batch_decisions=args.effective_batch_decisions).validate()
    environment = PracticeConfig(seed=718, shaping_scale=.1, time_limit_ms=args.episode_ms).validate()
    vocabulary = PracticeEnvironment(environment).action_vocabulary
    worker = Worker(args.binary.resolve() if args.binary else None)
    started = time.monotonic()
    try:
        initial = worker.job("checkpoint.create", {"destination": str(paths["initial"]), "model": model.to_dict(),
            "actions": vocabulary.to_dict(), "seed": 412, "pretrainedPath": str(ROOT / "vendor/weights/convnext_tiny.safetensors")})
        def payload(source, destination, iterations, resume=False):
            return {"checkpointPath": str(paths[source]), "destination": str(paths[destination]), "environment": environment.to_dict(),
                    "training": asdict(training), "iterations": iterations, "resume": resume}
        completed = worker.job("train.reinforcement", payload("initial", "completed", 1))
        cancelled = worker.job("train.reinforcement", payload("completed", "cancelled", 2, True), cancel_after=16)
        resumed = worker.job("train.reinforcement", payload("cancelled", "resumed", 2, True))
    finally:
        worker.close()
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from astra.checkpoints import load_checkpoint
    loaded = {name: load_checkpoint(path, include_training=True) for name, path in paths.items()}
    for name, expected_iteration in (("completed", 1), ("cancelled", 1), ("resumed", 2)):
        assert loaded[name].training_state["iteration"] == expected_iteration
        assert loaded[name].training_state["requiresEnvironmentReset"] is True
    before = dict(tree_flatten(loaded["completed"].policy.parameters()))
    for name, value in tree_flatten(loaded["cancelled"].policy.parameters()):
        np.testing.assert_array_equal(np.asarray(value), np.asarray(before[name]))
    assert loaded["cancelled"].training_state["optimizerUpdates"] == loaded["completed"].training_state["optimizerUpdates"]
    assert loaded["cancelled"].training_state["decisions"] == loaded["completed"].training_state["decisions"]
    assert loaded["resumed"].training_state["decisions"] == loaded["completed"].training_state["decisions"] + sum(
        metric["decisions"] for metric in resumed["iterationMetrics"])
    before_optimizer = dict(tree_flatten(loaded["completed"].training_state["optimizer"]))
    after_optimizer = dict(tree_flatten(loaded["cancelled"].training_state["optimizer"]))
    assert before_optimizer.keys() == after_optimizer.keys()
    for name, value in before_optimizer.items():
        np.testing.assert_array_equal(np.asarray(value), np.asarray(after_optimizer[name]))
    assert loaded["completed"].training_state["optimizerUpdates"] > 0
    assert loaded["resumed"].training_state["optimizerUpdates"] > loaded["completed"].training_state["optimizerUpdates"]
    for result in (completed, resumed):
        for metric in result["iterationMetrics"]:
            assert metric["maximum_accepted_kl"] <= training.ppo.target_kl
    assert completed["iterationsThisJob"] == resumed["iterationsThisJob"] == 1
    assert cancelled["iterationsThisJob"] == 0 and cancelled["cancelled"] is True
    report = {"model": model.to_dict(), "training": asdict(training), "environment": environment.to_dict(),
              "wallSeconds": time.monotonic() - started, "paths": {key: str(value) for key, value in paths.items()},
              "completed": completed, "cancelled": cancelled, "resumed": resumed,
              "checkpointReloaded": True, "cancelledCollectionPreservedWeightsAndOptimizerCount": True,
              "cancelledCollectionPreservedEntireOptimizer": True,
              "candidateAdmissionVerified": True,
              "completeEpisodeCollector": training.schema_version == 2,
              "limitation": "Protocol/learning integration only; no claim of held-out task success or native desktop control."}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.output), "wallSeconds": report["wallSeconds"]}), flush=True)


if __name__ == "__main__":
    main()
