from __future__ import annotations

from dataclasses import replace
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

import pytest

from astra.checkpoints import load_checkpoint
from astra.environments import PracticeConfig, PracticeEnvironment
from astra.jobs import JobManager
from astra.model.actions import ActionVocabulary
from astra.model.config import ModelConfig
from astra.protocol import Message
from astra.worker import BoundedSender, OutputOverflow
from test_recordings import native_recording


def worker_command():
    binary = os.environ.get("ASTRA_COMPUTE_TEST_BINARY")
    command = [binary] if binary else [sys.executable, "-m", "astra.worker"]
    if os.environ.get("ASTRA_COMPUTE_TEST_OFFLINE") == "1":
        command = ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)", *command]
    return command


def worker_environment():
    # Frozen checks cannot accidentally import from the development venv or
    # rely on Homebrew, uv or a system Python executable discovered via PATH.
    return {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"} if os.environ.get("ASTRA_COMPUTE_TEST_BINARY") else None


class Worker:
    def __init__(self):
        self.process = subprocess.Popen(worker_command(), env=worker_environment(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.events = queue.Queue()
        self.received = []
        self.deferred = []
        self.sequence = 0
        def read():
            try:
                for line in self.process.stdout:
                    event = json.loads(line)
                    self.received.append(event); self.events.put(event)
            finally:
                self.events.put(None)
        self.reader = threading.Thread(target=read, daemon=True); self.reader.start()
        self.hello = self.until(lambda event: event["kind"] == "hello")

    def send(self, kind, payload=None, *, run_id=None):
        request_id = str(uuid.uuid4())
        request = Message(kind, self.sequence, payload or {}, request_id=request_id, run_id=run_id)
        self.sequence += 1
        self.process.stdin.write(request.encode()); self.process.stdin.flush()
        return request_id

    def until(self, predicate, timeout=60):
        for index, event in enumerate(self.deferred):
            if predicate(event): return self.deferred.pop(index)
        deadline = time.monotonic() + timeout
        while True:
            event = self.events.get(timeout=max(.001, deadline - time.monotonic()))
            if event is None:
                raise AssertionError(f"Compute exited unexpectedly: {self.process.stderr.read().decode()}")
            if predicate(event): return event
            self.deferred.append(event)

    def request(self, kind, payload=None, *, run_id=None):
        request = self.send(kind, payload, run_id=run_id)
        return self.until(lambda event: event.get("requestID") == request and event["kind"] in ("ack", "error"))

    def job(self, kind, payload):
        run = str(uuid.uuid4())
        reply = self.request(kind, payload, run_id=run)
        assert reply["kind"] == "ack", reply
        result = self.until(lambda event: event.get("runID") == run and event["kind"] in ("job.completed", "job.failed", "job.cancelled"))
        assert result["kind"] == "job.completed", result
        return result["payload"]["result"]

    def close(self):
        if self.process.poll() is None:
            try:
                reply = self.request("shutdown")
                assert reply["payload"]["stopped"]
                self.process.wait(timeout=35)
                assert self.process.returncode == 0, self.process.stderr.read().decode()
            finally:
                if self.process.poll() is None: self.process.kill(); self.process.wait()
        self.reader.join(timeout=5)
        self.process.stdin.close(); self.process.stdout.close(); self.process.stderr.close()


@pytest.fixture
def worker():
    instance = Worker()
    try: yield instance
    finally: instance.close()


def initial(worker, root, *, model=None, vocabulary=None):
    model = model or ModelConfig.test_small()
    environment = PracticeConfig(pixel_width=64, pixel_height=64, logical_bounds=(0, 0, 64, 64))
    vocabulary = vocabulary or PracticeEnvironment(environment).action_vocabulary
    destination = root / str(uuid.uuid4())
    result = worker.job("checkpoint.create", {"destination": str(destination), "model": model.to_dict(),
                                               "actions": vocabulary.to_dict(), "seed": 884})
    assert result["checkpointPublished"] and destination.is_dir()
    assert result["parameterCount"] == model.parameter_count
    return destination


def fixture_dataset():
    return {"kind": "practice_oracle", "environment": PracticeConfig(pixel_width=64, pixel_height=64,
            logical_bounds=(0, 0, 64, 64)).to_dict(), "seedsBySplit": {"train": [7, 8, 9], "validation": [70]}}


def test_real_job_cancel_resume_evaluate_and_status_remain_correlated(worker, tmp_path):
    assert "train.behavioral" in worker.hello["payload"]["capabilities"]
    assert "inference.step" not in worker.hello["payload"]["capabilities"]
    origin = initial(worker, tmp_path)
    dataset = fixture_dataset()
    before = worker.job("evaluate.behavioral", {"checkpointPath": str(origin), "dataset": dataset,
                         "verificationMode": True, "split": "train", "sequenceLength": 2})
    cancelled_path = tmp_path / str(uuid.uuid4())
    training = {"epochs": 2, "lanes": 1, "sequence_length": 1, "accumulation_chunks": 3,
                "learning_rate": .002, "pretrained_learning_rate": .0003, "seed": 884}
    run = str(uuid.uuid4())
    request_id = worker.send("train.behavioral", {"checkpointPath": str(origin), "dataset": dataset,
                            "training": training, "destination": str(cancelled_path), "verificationMode": True}, run_id=run)
    acknowledgement = worker.until(lambda event: event.get("requestID") == request_id and event["kind"] == "ack")
    job_id = acknowledgement["payload"]["jobID"]
    progress = worker.until(lambda event: event.get("runID") == run and event["kind"] == "job.progress" and event["payload"].get("decisions", 0) > 0)
    assert progress["payload"]["provenance"] == "practice_oracle"
    # Real chunk execution has occurred. Input control remains responsive while
    # the owner thread continues model work; no fake sleeping job is involved.
    began = time.monotonic()
    ping = worker.request("ping")
    assert ping["kind"] == "ack" and ping["payload"]["alive"]
    assert time.monotonic() - began < 5
    cancellation = worker.request("cancel", {"jobID": job_id}, run_id=run)
    assert cancellation["payload"]["status"] in ("cancelling", "cancelled")
    ended = worker.until(lambda event: event.get("runID") == run and event["kind"] in ("job.cancelled", "job.failed"))
    assert ended["kind"] == "job.cancelled", ended
    assert ended["payload"]["result"]["checkpointPublished"]
    checkpoint = load_checkpoint(cancelled_path, include_training=True)
    assert checkpoint.training_state["kind"] == "behavioral"
    assert checkpoint.training_state["decisions"] > 0 and checkpoint.training_state["epoch"] < 2
    assert checkpoint.training_state["sampler"] is not None
    assert checkpoint.training_state["pendingCount"] > 0
    assert checkpoint.manifest["metrics"]["provenance"] == "practice_oracle"
    status = worker.request("job.status", {"jobID": job_id})
    assert status["payload"]["status"] == "cancelled"
    assert status["payload"]["result"]["checkpointPath"] == str(cancelled_path)

    resumed_path = tmp_path / str(uuid.uuid4())
    resumed = worker.job("train.behavioral", {"checkpointPath": str(cancelled_path), "dataset": dataset,
                         "training": training, "destination": str(resumed_path), "verificationMode": True, "resume": True})
    assert resumed["manifest"]["step"] > 0 and not resumed["cancelled"]
    restored = load_checkpoint(resumed_path, include_training=True)
    assert restored.training_state["epoch"] == 2 and restored.training_state["decisions"] == 12
    assert restored.training_state["pendingCount"] == 0
    after = worker.job("evaluate.behavioral", {"checkpointPath": str(resumed_path), "dataset": dataset,
                        "verificationMode": True, "split": "train", "sequenceLength": 2})
    assert after["evaluation"]["meanNLL"] < before["evaluation"]["meanNLL"]
    inspection = worker.job("checkpoint.inspect", {"path": str(resumed_path)})
    assert inspection["integrityVerified"] and inspection["manifest"]["id"] == resumed_path.name
    assert inspection["parameterCount"] == resumed["parameterCount"] == ModelConfig.test_small().parameter_count
    assert [event["sequence"] for event in worker.received] == list(range(len(worker.received)))


def test_native_recording_dataset_job_then_training_uses_real_sealed_source(worker, tmp_path, native_recording):
    model = replace(ModelConfig.test_small(), period_ms=20, lead_ms=20)
    vocabulary = ActionVocabulary((13,), (), True, False, True)
    source = Path(native_recording["directory"])
    prepared_path = tmp_path / str(uuid.uuid4())
    prepared = worker.job("dataset.prepare", {"destination": str(prepared_path), "recordingRoot": str(source.parent),
                         "selections": [{"recording_id": source.stem, "context_ids": []}], "model": model.to_dict(),
                         "actions": vocabulary.to_dict(), "pointerMode": "absolute", "splitSeed": 0})
    assert prepared["manifest"]["steps"] == 3
    origin = initial(worker, tmp_path, model=model, vocabulary=vocabulary)
    destination = tmp_path / str(uuid.uuid4())
    result = worker.job("train.behavioral", {"checkpointPath": str(origin), "destination": str(destination),
                       "dataset": {"kind": "recordings", "path": str(prepared_path), "recordingRoot": str(source.parent)},
                       "training": {"epochs": 1, "lanes": 1, "sequence_length": 2}})
    assert result["provenance"] == "recorded_demonstrations"
    assert result["manifest"]["metrics"]["decisions"] == 3
    assert result["manifest"]["step"] == 2


def test_multi_range_dataset_job_keeps_one_session_and_trains_both_episodes(worker, tmp_path, native_recording):
    model = replace(ModelConfig.test_small(), period_ms=20, lead_ms=0)
    vocabulary = ActionVocabulary((13,), (), True, False, True)
    source = Path(native_recording["directory"])
    prepared_path = tmp_path / str(uuid.uuid4())
    prepared = worker.job("dataset.prepare", {"destination": str(prepared_path), "recordingRoot": str(source.parent),
        "selections": [{"recording_id": source.stem, "ranges": [
            {"start_nanos": 1_002_000_000, "end_nanos": 1_022_000_000},
            {"start_nanos": 1_075_000_000, "end_nanos": 1_100_000_000}], "context_ids": []}],
        "model": model.to_dict(), "actions": vocabulary.to_dict(), "pointerMode": "absolute"})
    assert prepared["manifest"]["schemaVersion"] == 2 and prepared["manifest"]["steps"] == 2
    assert len(prepared["manifest"]["sources"]) == 1 and len(prepared["manifest"]["sources"][0]["labelPartitions"]) == 2
    origin = initial(worker, tmp_path, model=model, vocabulary=vocabulary)
    result = worker.job("train.behavioral", {"checkpointPath": str(origin), "destination": str(tmp_path / str(uuid.uuid4())),
        "dataset": {"kind": "recordings", "path": str(prepared_path), "recordingRoot": str(source.parent)},
        "training": {"epochs": 1, "lanes": 1, "sequence_length": 2}})
    assert result["manifest"]["metrics"]["decisions"] == 2 and result["manifest"]["step"] == 2


@pytest.mark.parametrize("selection", [
    {"ranges": []}, {"ranges": [{"start_nanos": 20, "end_nanos": 10}]},
    {"start_nanos": None, "ranges": [{"start_nanos": 10, "end_nanos": 20}]},
    {"ranges": [{"start_nanos": 10, "end_nanos": 30}, {"start_nanos": 20, "end_nanos": 40}]},
    {"ranges": [{"start_nanos": 10, "end_nanos": 20}] * 257},
])
def test_invalid_multi_range_job_is_rejected_before_worker_admission(worker, tmp_path, selection):
    result = worker.request("dataset.prepare", {"destination": str(tmp_path / str(uuid.uuid4())), "recordingRoot": str(tmp_path),
        "selections": [{"recording_id": str(uuid.uuid4()), **selection}], "model": ModelConfig.test_small().to_dict(),
        "actions": ActionVocabulary((13,), (), True, False, True).to_dict(), "pointerMode": "absolute"}, run_id=str(uuid.uuid4()))
    assert result["kind"] == "error" and result["payload"]["code"] == "job.invalidConfiguration"
    assert not list(tmp_path.iterdir())


def test_invalid_requests_and_failed_jobs_do_not_terminate_valid_control(worker, tmp_path):
    missing_run = worker.request("checkpoint.inspect", {"path": str(tmp_path)})
    assert missing_run["kind"] == "error" and missing_run["payload"]["code"] == "job.missingRunID"
    run = str(uuid.uuid4())
    invalid = worker.request("train.behavioral", {"checkpointPath": str(tmp_path / str(uuid.uuid4())),
                             "destination": str(tmp_path / str(uuid.uuid4())), "dataset": fixture_dataset(), "training": {}}, run_id=run)
    assert invalid["kind"] == "error" and invalid["payload"]["code"] == "job.verificationRequired"
    unsupported = worker.request("inference.step", {})
    assert unsupported["kind"] == "error" and unsupported["payload"]["code"] == "protocol.unsupportedOperation"
    request_id = worker.send("checkpoint.inspect", {"path": str(tmp_path / str(uuid.uuid4()))}, run_id=run)
    acknowledged = worker.until(lambda event: event.get("requestID") == request_id and event["kind"] == "ack")
    failed = worker.until(lambda event: event.get("runID") == run and event["kind"] == "job.failed")
    assert failed["payload"]["error"]["recoverable"] is True
    status = worker.request("job.status", {"jobID": acknowledged["payload"]["jobID"]})
    assert status["payload"]["status"] == "failed"
    assert worker.request("ping")["payload"]["alive"]


class PausedOutput(io.BytesIO):
    def __init__(self):
        super().__init__(); self.entered = threading.Event(); self.release = threading.Event()
    def write(self, data):
        self.entered.set()
        assert self.release.wait(timeout=5)
        # Exercise partial writes too; one frame must remain intact.
        return super().write(bytes(data[:31]))


def test_bounded_output_coalesces_progress_and_preserves_final_replies():
    stream = PausedOutput()
    sender = BoundedSender(stream, maximum_messages=3, maximum_bytes=4096)
    request = Message("train.behavioral", 0, {}, request_id=str(uuid.uuid4()), run_id=str(uuid.uuid4()))
    sender.send("ack", {"accepted": True}, request=request)
    assert stream.entered.wait(2)
    for number in range(1000): sender.send("job.progress", {"decisions": number}, request=request)
    sender.send("job.completed", {"completed": True}, request=request)
    stream.release.set()
    assert sender.close()
    values = [Message.decode(line + b"\n") for line in stream.getvalue().splitlines()]
    assert [value.kind for value in values] == ["ack", "job.progress", "job.completed"]
    assert values[1].payload["decisions"] == 999
    assert [value.sequence for value in values] == [0, 1, 2]
    assert all(value.run_id == request.run_id and value.request_id == request.request_id for value in values)


def test_control_reply_overflow_is_explicit_channel_failure():
    stream = PausedOutput()
    sender = BoundedSender(stream, maximum_messages=1)
    sender.send("ack", {})
    assert stream.entered.wait(2)
    sender.send("ack", {})
    with pytest.raises(OutputOverflow): sender.send("job.completed", {})
    assert sender.failed.is_set()
    stream.release.set()
    assert not sender.close()


def test_duplicate_protocol_sequence_and_partial_eof_are_terminal_failures():
    for data in (Message("ping", 0, {}).encode() * 2, b'{"version":1'):
        result = subprocess.run(worker_command(), env=worker_environment(), input=data, capture_output=True, timeout=15)
        assert result.returncode == 65
        responses = [json.loads(line) for line in result.stdout.splitlines()]
        assert responses[-1]["kind"] == "error"
        assert not responses[-1]["payload"]["recoverable"]


def test_lost_progress_consumer_still_saves_completed_chunk_for_resume(worker, tmp_path):
    origin = initial(worker, tmp_path)
    destination = tmp_path / str(uuid.uuid4())
    failed = threading.Event()
    def send(kind, payload, *, request=None):
        if kind == "job.progress" and payload.get("decisions", 0):
            failed.set()
        if failed.is_set(): raise OutputOverflow("Test consumer closed")
    manager = JobManager(send)
    try:
        identifier = manager.submit(Message("train.behavioral", 0, {
            "checkpointPath": str(origin), "destination": str(destination), "dataset": fixture_dataset(),
            "verificationMode": True, "training": {"epochs": 2, "lanes": 1, "sequence_length": 1, "accumulation_chunks": 3},
        }, run_id=str(uuid.uuid4())))
        assert failed.wait(15)
        assert manager.close(15)
        status = manager.status(identifier)
        assert status["status"] == "cancelled" and status["result"]["checkpointPublished"]
        saved = load_checkpoint(destination, include_training=True)
        assert saved.training_state["pendingCount"] == 1
        assert saved.training_state["pendingChunks"] == 1
        assert saved.training_state["decisions"] == 1
    finally:
        manager.close(5)


def test_closed_output_terminates_worker_even_with_input_left_open():
    process = subprocess.Popen(worker_command(), env=worker_environment(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        process.stdout.close()
        assert process.wait(timeout=15) == 74
    finally:
        if process.poll() is None: process.kill(); process.wait()
        process.stdin.close(); process.stderr.close()


def reinforcement_payload(origin, destination, *, iterations=2):
    return {"checkpointPath": str(origin), "destination": str(destination), "iterations": iterations,
            "environment": PracticeConfig(pixel_width=64, pixel_height=64, logical_bounds=(0, 0, 64, 64),
                                           time_limit_ms=350, shaping_scale=.1).to_dict(),
            "training": {"rollout_decisions": 4, "epochs": 1, "sequence_length": 2, "burn_in": 1,
                         "effective_batch_decisions": 4, "seed": 884,
                         "ppo": {"clip_ratio": .2, "target_kl": .02},
                         "returns": {"discount_half_life_seconds": 30.0}}}


def test_real_reinforcement_job_collects_updates_and_reloads_without_oracle_mode(worker, tmp_path):
    import numpy as np
    from mlx.utils import tree_flatten
    assert "train.reinforcement" in worker.hello["payload"]["capabilities"]
    origin = initial(worker, tmp_path)
    before = {name: np.asarray(value).copy() for name, value in tree_flatten(load_checkpoint(origin).policy.parameters())}
    destination = tmp_path / str(uuid.uuid4())
    result = worker.job("train.reinforcement", reinforcement_payload(origin, destination))
    assert result["sourceKind"] == result["provenance"] == "practice_rollout"
    assert result["completedIterations"] == result["iterationsThisJob"] == 2
    assert result["checkpointPublished"] and result["resumable"] and result["requiresEnvironmentReset"]
    loaded = load_checkpoint(destination, include_training=True)
    assert loaded.manifest["kind"] == "reinforcement"
    assert loaded.training_state["kind"] == "reinforcement" and loaded.training_state["iteration"] == 2
    assert loaded.training_state["optimizerUpdates"] > 0
    assert loaded.training_state["requiresEnvironmentReset"] is True
    assert any(np.any(np.asarray(value) != before[name]) for name, value in tree_flatten(loaded.policy.parameters()))
    metrics = loaded.manifest["metrics"]
    assert metrics["decisions"] == 8 and metrics["sourceKind"] == "practice_rollout"
    assert metrics["lastIteration"]["behavior_replay_error"] < 2e-4
    assert metrics["lastIteration"]["maximum_sampled_kl"] >= 0
    progresses = [event["payload"] for event in worker.received if event["kind"] == "job.progress"
                  and event["payload"].get("sourceKind") == "practice_rollout"]
    # Progress can be coalesced by the bounded transport. Its latest completed
    # iteration and the terminal history must survive a faster following phase.
    assert {"collecting", "updating", "checkpointing"} <= {item["phase"] for item in progresses}
    assert [item["iteration"] for item in result["iterationMetrics"]] == [1, 2]
    assert progresses[-1]["last_iteration"]["iteration"] == 2
    assert any(item.get("rollout_decisions", 0) > 0 for item in progresses)
    assert progresses[-1]["last_iteration"]["mean_value_loss"] >= 0
    cumulative = [item["optimizer_updates"] for item in progresses]
    assert cumulative == sorted(cumulative)
    assert cumulative[-1] == loaded.training_state["optimizerUpdates"]
    resumed_path = tmp_path / str(uuid.uuid4())
    resumed = worker.job("train.reinforcement", {**reinforcement_payload(destination, resumed_path, iterations=3), "resume": True})
    assert resumed["completedIterations"] == 3 and resumed["iterationsThisJob"] == 1
    assert load_checkpoint(resumed_path, include_training=True).training_state["iteration"] == 3


def test_reinforcement_cancelled_collection_publishes_a_fresh_reset_resume_state(worker, tmp_path):
    import numpy as np
    from mlx.utils import tree_flatten
    origin = initial(worker, tmp_path)
    destination = tmp_path / str(uuid.uuid4())
    released = threading.Event()
    messages = []
    manager = None
    def send(kind, payload, *, request=None):
        messages.append((kind, payload))
        if kind == "job.progress" and payload.get("phase") == "collecting" and payload.get("rollout_decisions", 0) >= 2:
            # Synchronous cancellation at a real collection boundary prevents a
            # timing race with a tiny environment finishing before an IPC roundtrip.
            manager.cancel(payload["jobID"], run_id=request.run_id)
            released.set()
    manager = JobManager(send)
    try:
        identifier = manager.submit(Message("train.reinforcement", 0,
            reinforcement_payload(origin, destination), run_id=str(uuid.uuid4())))
        assert released.wait(15)
        assert manager.close(15)
        status = manager.status(identifier)
        assert status["status"] == "cancelled" and status["result"]["checkpointPublished"]
        loaded = load_checkpoint(destination, include_training=True)
        assert loaded.training_state["iteration"] == 0 and loaded.training_state["optimizerUpdates"] == 0
        assert loaded.training_state["requiresEnvironmentReset"]
        before = dict(tree_flatten(load_checkpoint(origin).policy.parameters()))
        for name, value in tree_flatten(loaded.policy.parameters()):
            np.testing.assert_array_equal(np.asarray(value), np.asarray(before[name]))
    finally:
        manager.close(5)
    resumed_path = tmp_path / str(uuid.uuid4())
    result = worker.job("train.reinforcement", {**reinforcement_payload(destination, resumed_path), "resume": True})
    assert result["completedIterations"] == 2


def test_reinforcement_cancelled_update_rolls_back_before_checkpoint_publication(worker, tmp_path):
    import numpy as np
    from mlx.utils import tree_flatten
    origin = initial(worker, tmp_path)
    destination = tmp_path / str(uuid.uuid4())
    cancelled = threading.Event()
    manager = None
    def send(kind, payload, *, request=None):
        if kind == "job.progress" and payload.get("phase") == "updating" and payload.get("optimizer_updates", 0) > 0:
            manager.cancel(payload["jobID"], run_id=request.run_id)
            cancelled.set()
    manager = JobManager(send)
    payload = reinforcement_payload(origin, destination)
    payload["training"]["epochs"] = 2
    payload["training"]["effective_batch_decisions"] = 2
    try:
        identifier = manager.submit(Message("train.reinforcement", 0, payload, run_id=str(uuid.uuid4())))
        assert cancelled.wait(15) and manager.close(15)
        status = manager.status(identifier)
        assert status["status"] == "cancelled" and status["result"]["checkpointPublished"]
        loaded = load_checkpoint(destination, include_training=True)
        assert loaded.training_state["iteration"] == loaded.training_state["optimizerUpdates"] == 0
        before = dict(tree_flatten(load_checkpoint(origin).policy.parameters()))
        for name, value in tree_flatten(loaded.policy.parameters()):
            np.testing.assert_array_equal(np.asarray(value), np.asarray(before[name]))
    finally:
        manager.close(5)


@pytest.mark.parametrize("field,value", [("ppo", {"clip_ratio": True}), ("returns", {"lambda_per_reference": 2}),
                                        ("ppo", {"unknown": 1}), ("returns", [])])
def test_reinforcement_nested_configuration_is_rejected_before_start(worker, tmp_path, field, value):
    payload = reinforcement_payload(tmp_path / str(uuid.uuid4()), tmp_path / str(uuid.uuid4()))
    payload["training"][field] = value
    response = worker.request("train.reinforcement", payload, run_id=str(uuid.uuid4()))
    assert response["kind"] == "error"
    assert not Path(payload["destination"]).exists()
    assert worker.request("ping")["payload"]["alive"]


def test_behavioral_checkpoint_can_begin_fresh_reinforcement_with_trainable_vision(worker, tmp_path):
    origin = initial(worker, tmp_path)
    behavioral = tmp_path / str(uuid.uuid4())
    worker.job("train.behavioral", {"checkpointPath": str(origin), "destination": str(behavioral),
        "dataset": fixture_dataset(), "verificationMode": True,
        "training": {"epochs": 1, "lanes": 1, "sequence_length": 2, "freeze_pretrained_epochs": 1}})
    assert load_checkpoint(behavioral).manifest["frozenParameters"]
    destination = tmp_path / str(uuid.uuid4())
    result = worker.job("train.reinforcement", reinforcement_payload(behavioral, destination, iterations=1))
    assert result["manifest"]["kind"] == "reinforcement"
    assert result["manifest"]["frozenParameters"] == []
    assert load_checkpoint(destination, include_training=True).training_state["optimizerUpdates"] > 0


def test_reinforcement_rollout_disk_is_admitted_before_environment_allocation(worker, tmp_path):
    payload = reinforcement_payload(tmp_path / str(uuid.uuid4()), tmp_path / str(uuid.uuid4()))
    payload["environment"]["pixel_width"] = 1280
    payload["environment"]["pixel_height"] = 720
    payload["training"]["rollout_decisions"] = 2048
    payload["training"]["maximum_rollout_disk_bytes"] = 4 * 1024**3
    response = worker.request("train.reinforcement", payload, run_id=str(uuid.uuid4()))
    assert response["kind"] == "error" and response["payload"]["code"] == "job.rolloutBudget"
    assert worker.request("ping")["payload"]["alive"]


def test_reinforcement_cancel_during_candidate_validation_publishes_rolled_back_checkpoint(worker, tmp_path):
    import numpy as np
    from mlx.utils import tree_flatten
    origin = initial(worker, tmp_path)
    destination = tmp_path / str(uuid.uuid4())
    observed = threading.Event()
    manager = None
    def send(kind, payload, *, request=None):
        if kind == 'job.progress' and payload.get('phase') == 'validating_update' and payload.get('validation_decisions', 0) >= 2:
            manager.cancel(payload['jobID'], run_id=request.run_id)
            observed.set()
    manager = JobManager(send)
    try:
        identifier = manager.submit(Message('train.reinforcement', 0,
            reinforcement_payload(origin, destination), run_id=str(uuid.uuid4())))
        assert observed.wait(15) and manager.close(15)
        status = manager.status(identifier)
        assert status['status'] == 'cancelled' and status['result']['checkpointPublished']
        saved = load_checkpoint(destination, include_training=True)
        assert saved.training_state['optimizerUpdates'] == saved.training_state['iteration'] == 0
        before = dict(tree_flatten(load_checkpoint(origin).policy.parameters()))
        for name, value in tree_flatten(saved.policy.parameters()):
            np.testing.assert_array_equal(np.asarray(value), np.asarray(before[name]))
    finally:
        manager.close(5)


def test_reinforcement_finishing_episode_progress_is_cancellable_without_publishing_partial_experience(worker, tmp_path):
    origin = initial(worker, tmp_path)
    destination = tmp_path / str(uuid.uuid4())
    observed = threading.Event()
    manager = None
    progress = []
    def send(kind, payload, *, request=None):
        if kind == 'job.progress':
            progress.append(payload)
            if payload.get('phase') == 'finishing_episode':
                manager.cancel(payload['jobID'], run_id=request.run_id)
                observed.set()
    payload = reinforcement_payload(origin, destination)
    payload['environment']['time_limit_ms'] = 2000
    manager = JobManager(send)
    try:
        identifier = manager.submit(Message('train.reinforcement', 0, payload, run_id=str(uuid.uuid4())))
        assert observed.wait(15) and manager.close(15)
        status = manager.status(identifier)
        assert status['status'] == 'cancelled' and status['result']['checkpointPublished']
        saved = load_checkpoint(destination, include_training=True)
        assert saved.training_state['iteration'] == 0
        assert saved.manifest['metrics']['decisions'] == 0
        waiting = next(row for row in progress if row.get('phase') == 'finishing_episode')
        assert waiting['rollout_decisions'] == 4 and waiting['rollout_target'] == 4
        assert waiting['iteration'] == 0 and waiting['decisions'] == 0
        assert not list(tmp_path.glob('.astra-rollout-*'))
        assert saved.training_state['requiresEnvironmentReset']
    finally:
        manager.close(5)


def test_reinforcement_cleanup_failure_does_not_replace_the_primary_training_error(worker, tmp_path, monkeypatch):
    from astra.learning.reinforcement import ReinforcementTrainer
    origin = initial(worker, tmp_path)
    destination = tmp_path / str(uuid.uuid4())
    failed = threading.Event()
    environments = []
    def failing_update(trainer, rollout, **kwargs):
        environments.append(trainer.environment)
        close = rollout.spool.close
        def failed_cleanup():
            close()
            raise OSError('Secondary rollout cleanup failure')
        monkeypatch.setattr(rollout.spool, 'close', failed_cleanup)
        raise ValueError('Primary PPO scoring failure')
    monkeypatch.setattr(ReinforcementTrainer, 'update', failing_update)
    manager = JobManager(lambda kind, *args, **kwargs: failed.set() if kind == 'job.failed' else None)
    payload = reinforcement_payload(origin, destination)
    payload['environment']['time_limit_ms'] = 2000
    try:
        identifier = manager.submit(Message('train.reinforcement', 0, payload, run_id=str(uuid.uuid4())))
        assert failed.wait(15) and manager.close(15)
        status = manager.status(identifier)
        assert status['status'] == 'failed' and status['error']['message'] == 'Primary PPO scoring failure'
        assert not destination.exists()
        assert environments[0].outcome == 'truncated'
        assert not list(tmp_path.glob('.astra-rollout-*'))
        assert not environments[0]._keys and not environments[0]._buttons and not environments[0]._pending
    finally:
        manager.close(5)


def test_reinforcement_jobs_save_actual_complete_episode_counts_and_retire_spools(worker, tmp_path):
    origin = initial(worker, tmp_path)
    destination = tmp_path / str(uuid.uuid4())
    payload = reinforcement_payload(origin, destination)
    payload['environment']['time_limit_ms'] = 700
    result = worker.job('train.reinforcement', payload)
    saved = load_checkpoint(destination, include_training=True)
    counts = [row['decisions'] for row in result['iterationMetrics']]
    assert counts == [7, 7]
    assert saved.training_state['decisions'] == saved.manifest['metrics']['decisions'] == 14
    assert saved.training_state['config']['schema_version'] == 2
    assert saved.training_state['schemaVersion'] == 2
    assert not list(tmp_path.glob('.astra-rollout-*'))
    progress = [row['payload'] for row in worker.received if row['kind'] == 'job.progress'
                and row['payload'].get('phase') == 'finishing_episode']
    assert progress and any(row['rollout_decisions'] > row['rollout_target'] for row in progress)
    assert all(row.get('audit_tail_decisions', 0) == 0 for row in result['iterationMetrics'])
