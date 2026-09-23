"""Serialized local compute jobs; only this worker thread owns learning state.

Job control is independent of model execution. Cancellation is observed between
complete training chunks and publishes the real resumable optimizer/sampler state.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import copy
import hashlib
import json
from pathlib import Path
import queue
import sys
import threading
import time
import uuid

import mlx.core as mx

from astra.checkpoints import load_checkpoint, save_checkpoint
from astra.data.datasets import DatasetReader, RecordingSelection, build_dataset
from astra.environments.demonstrations import PracticeDemonstrations
from astra.environments.practice import PracticeConfig, PracticeEnvironment
from astra.learning.behavioral import BehaviorConfig, BehaviorTrainer
from astra.learning.reinforcement import ReinforcementConfig, ReinforcementTrainer
from astra.learning.rl import PPOConfig, ReturnConfig
from astra.model.actions import ActionVocabulary
from astra.model.config import ModelConfig
from astra.model.policy import AgentPolicy
from astra.protocol import Message

JOB_OPERATIONS = ("feedback.inspect", "feedback.materialize", "feedback.combine", "checkpoint.inspect", "checkpoint.create", "dataset.prepare", "train.behavioral", "evaluate.behavioral", "train.reinforcement", "train.reinforcement.external", "checkpoint.externalBoundary")
TERMINAL = {"completed", "cancelled", "failed"}


class JobError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _object(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or not set(required) <= value.keys():
        raise JobError("job.invalidConfiguration", "Job configuration has missing or unknown fields")
    return value


def _integer(value, low=0, high=2**63 - 1):
    if type(value) is not int or not low <= value <= high:
        raise JobError("job.invalidConfiguration", "Job integer is outside its supported range")
    return value


def _boolean(value):
    if type(value) is not bool:
        raise JobError("job.invalidConfiguration", "Job flags must be Boolean")
    return value


def _path(value, *, destination=False):
    if not isinstance(value, str) or not 1 <= len(value.encode()) <= 4096 or "\x00" in value:
        raise JobError("job.invalidPath", "A bounded absolute local path is required")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise JobError("job.invalidPath", "A normalized absolute local path is required")
    if path.is_symlink():
        raise JobError("job.invalidPath", "Artifact packages cannot be symbolic links")
    if destination:
        if str(uuid.UUID(path.name)) != path.name:
            raise JobError("job.invalidPath", "Artifact destination must end in a canonical UUID")
        if path.exists():
            raise JobError("job.destinationExists", "Artifacts are immutable; allocate a new destination UUID")
    return path


def _dataset(value, verification):
    if not isinstance(value, dict):
        raise JobError("job.invalidConfiguration", "Dataset selection must be an object")
    kind = value.get("kind")
    if kind == "recordings":
        _object(value, ("kind", "path", "recordingRoot"), ("path", "recordingRoot"))
        _path(value["path"]); _path(value["recordingRoot"])
    elif kind == "practice_oracle":
        _object(value, ("kind", "environment", "seedsBySplit"), ("environment", "seedsBySplit"))
        if not verification:
            raise JobError("job.verificationRequired", "Oracle demonstrations require explicit verificationMode=true")
        PracticeConfig.from_dict(value["environment"])
        splits = value["seedsBySplit"]
        _object(splits, ("train", "validation", "test"))
        if not splits or any(not isinstance(seeds, list) for seeds in splits.values()):
            raise JobError("job.invalidConfiguration", "Practice splits require seed arrays")
        seeds = [seed for values in splits.values() for seed in values]
        for seed in seeds:
            _integer(seed)
        if not 1 <= len(seeds) <= 4096 or len(set(seeds)) != len(seeds):
            raise JobError("job.invalidConfiguration", "Practice seed sets must be nonempty, bounded and disjoint")
    else:
        raise JobError("job.invalidConfiguration", "Unsupported dataset kind")
    return value


def _reinforcement_config(value):
    _object(value, ReinforcementConfig.__dataclass_fields__)
    copied = dict(value)
    for name, configuration in (("ppo", PPOConfig), ("returns", ReturnConfig)):
        nested = copied.get(name, {})
        _object(nested, configuration.__dataclass_fields__)
        copied[name] = configuration(**nested)
    return ReinforcementConfig(**copied).validate()


def validate_request(request: Message) -> dict:
    if request.run_id is None:
        raise JobError("job.missingRunID", "Asynchronous compute jobs require a runID")
    value = copy.deepcopy(request.payload)
    if request.kind == "checkpoint.inspect":
        _object(value, ("path",), ("path",)); _path(value["path"])
    elif request.kind == "checkpoint.create":
        _object(value, ("destination", "model", "actions", "seed", "pretrainedPath"), ("destination", "model", "actions"))
        _path(value["destination"], destination=True)
        ModelConfig.from_dict(value["model"]); ActionVocabulary.from_dict(value["actions"])
        _integer(value.get("seed", 0))
        if value.get("pretrainedPath") is not None:
            _path(value["pretrainedPath"])
    elif request.kind == "dataset.prepare":
        _object(value, ("destination", "recordingRoot", "selections", "model", "actions", "pointerMode", "splitSeed"),
                ("destination", "recordingRoot", "selections", "model", "actions", "pointerMode"))
        _path(value["destination"], destination=True); _path(value["recordingRoot"])
        ModelConfig.from_dict(value["model"]); ActionVocabulary.from_dict(value["actions"])
        _integer(value.get("splitSeed", 0))
        if value["pointerMode"] not in ("absolute", "relative"):
            raise JobError("job.invalidConfiguration", "Dataset pointerMode must be absolute or relative")
        if not isinstance(value["selections"], list) or not 1 <= len(value["selections"]) <= 4096:
            raise JobError("job.invalidConfiguration", "Dataset requires a bounded recording selection list")
        total_ranges = 0
        for selection in value["selections"]:
            _object(selection, ("recording_id", "start_nanos", "end_nanos", "ranges", "context_ids"), ("recording_id",))
            uuid.UUID(selection["recording_id"])
            for name in ("start_nanos", "end_nanos"):
                if selection.get(name) is not None: _integer(selection[name])
            if selection.get("start_nanos") is not None and selection.get("end_nanos") is not None and selection["start_nanos"] >= selection["end_nanos"]:
                raise JobError("job.invalidConfiguration", "Recording selection must have a positive duration")
            if "ranges" in selection:
                ranges = selection["ranges"]
                if "start_nanos" in selection or "end_nanos" in selection or not isinstance(ranges, list) or not 1 <= len(ranges) <= 256:
                    raise JobError("job.invalidConfiguration", "Choose 1–256 ranges without mixing legacy selection bounds")
                previous_end = 0
                for interval in ranges:
                    _object(interval, ("start_nanos", "end_nanos"), ("start_nanos", "end_nanos"))
                    start, end = _integer(interval["start_nanos"]), _integer(interval["end_nanos"])
                    if start < previous_end or start >= end:
                        raise JobError("job.invalidConfiguration", "Recording ranges must be ordered, non-overlapping and nonempty")
                    previous_end = end
                total_ranges += len(ranges)
            else:
                total_ranges += 1
            if not isinstance(selection.get("context_ids", []), list):
                raise JobError("job.invalidConfiguration", "Context choices must be an integer array")
            for choice in selection.get("context_ids", []): _integer(choice, high=65535)
        if total_ranges > 100_000:
            raise JobError("job.invalidConfiguration", "Dataset selection exceeds its range budget")
    elif request.kind=="feedback.combine":
        _object(value,("fragments","destination"),("fragments","destination"));_path(value["destination"],destination=True)
        if type(value["fragments"]) is not list or not 1<=len(value["fragments"])<=128:raise JobError("job.invalidConfiguration","Batch requires bounded fragment references")
        from astra.retrospective_feedback import _digest
        for reference in value["fragments"]:
            _object(reference,("path","manifestSHA256"),("path","manifestSHA256"));_path(reference["path"]);_digest(reference["manifestSHA256"])
    elif request.kind in ("feedback.inspect", "feedback.materialize"):
        allowed=("sourcePath","manifestSHA256") if request.kind=="feedback.inspect" else ("sourcePath","manifestSHA256","revisionDirectory","revisionChain","destination")
        _object(value,allowed,allowed)
        _path(value["sourcePath"])
        from astra.retrospective_feedback import _digest, _fields, _id
        _digest(value["manifestSHA256"])
        if request.kind=="feedback.materialize":
            _path(value["revisionDirectory"]);_path(value["destination"],destination=True)
            if type(value["revisionChain"]) is not list or not 1<=len(value["revisionChain"])<=4096:
                raise JobError("job.invalidConfiguration","Review requires a bounded immutable revision chain")
            for reference in value["revisionChain"]:
                _fields(reference,("id","sha256"));_id(reference["id"]);_digest(reference["sha256"])
    elif request.kind == "checkpoint.externalBoundary":
        _object(value,("checkpointPath","auditPath","destination","resume"),("checkpointPath","auditPath","destination"))
        _path(value["checkpointPath"]);_path(value["auditPath"])
        destination=_path(value["destination"])
        if str(uuid.UUID(destination.name))!=destination.name:
            raise JobError("job.invalidPath","Boundary checkpoint destinations require a canonical UUID")
        _boolean(value.get("resume",False))
    elif request.kind == "train.reinforcement.external":
        _object(value,("checkpointPath","rolloutPath","destination","resume","boundaryTimeoutSeconds"),
                ("checkpointPath","rolloutPath","destination"))
        _path(value["checkpointPath"]);_path(value["rolloutPath"]);_path(value["destination"],destination=True)
        _boolean(value.get("resume",False));_integer(value.get("boundaryTimeoutSeconds",60),1,600)
    elif request.kind == "train.reinforcement":
        _object(value, ("checkpointPath", "environment", "training", "iterations", "destination", "resume", "contextIDs"),
                ("checkpointPath", "environment", "training", "iterations", "destination"))
        _path(value["checkpointPath"]); _path(value["destination"], destination=True)
        environment = PracticeConfig.from_dict(value["environment"])
        training = _reinforcement_config(value["training"])
        frame_bytes = environment.pixel_width * environment.pixel_height * 4
        if frame_bytes > training.maximum_rollout_bytes:
            raise JobError("job.rolloutBudget", "One practice frame exceeds the rollout RAM budget")
        if frame_bytes * (training.rollout_decisions + 1) > training.maximum_rollout_disk_bytes:
            raise JobError("job.rolloutBudget", "The minimum practice rollout exceeds its raw-image disk budget")
        _integer(value["iterations"], 1, 100_000)
        _boolean(value.get("resume", False))
        if not isinstance(value.get("contextIDs", []), list) or len(value.get("contextIDs", [])) > 32:
            raise JobError("job.invalidConfiguration", "Reinforcement contexts require a bounded integer array")
        for choice in value.get("contextIDs", []):
            _integer(choice, 0, 65535)
    elif request.kind in ("train.behavioral", "evaluate.behavioral"):
        training = request.kind == "train.behavioral"
        allowed = ("checkpointPath", "dataset", "verificationMode", "training", "destination", "resume") if training else (
            "checkpointPath", "dataset", "verificationMode", "split", "sequenceLength")
        required = ("checkpointPath", "dataset", "training", "destination") if training else ("checkpointPath", "dataset")
        _object(value, allowed, required)
        _path(value["checkpointPath"])
        verification = _boolean(value.get("verificationMode", False))
        _dataset(value["dataset"], verification)
        if training:
            _path(value["destination"], destination=True)
            _boolean(value.get("resume", False))
            _object(value["training"], BehaviorConfig.__dataclass_fields__)
            BehaviorConfig(**value["training"]).validate()
        else:
            if value.get("split", "validation") not in ("train", "validation", "test"):
                raise JobError("job.invalidConfiguration", "Unknown evaluation split")
            _integer(value.get("sequenceLength", 64), 1, 512)
    else:
        raise JobError("protocol.unsupportedOperation", "Unsupported compute job operation")
    return value


@dataclass
class _Job:
    identifier: str
    request: Message
    configuration: dict
    status: str = "queued"
    cancel: threading.Event = field(default_factory=threading.Event)
    metrics: dict = field(default_factory=dict)
    result: dict | None = None
    error: dict | None = None
    external_boundary_path: str | None = None
    external_boundary_event: threading.Event = field(default_factory=threading.Event)

    def snapshot(self):
        return copy.deepcopy({"jobID": self.identifier, "runID": self.request.run_id, "operation": self.request.kind,
                              "status": self.status, "metrics": self.metrics, "result": self.result, "error": self.error})


class JobManager:
    """One active job per compute role, bounded status history, no model sharing."""
    def __init__(self, send):
        self._send = send
        self._lock = threading.RLock()
        self._jobs: OrderedDict[str, _Job] = OrderedDict()
        self._active: _Job | None = None
        self._queue = queue.Queue(maxsize=1)
        self._closing = False
        self._thread = threading.Thread(target=self._work, name="Astra learning owner", daemon=True)
        self._thread.start()

    @property
    def busy(self):
        with self._lock:
            return self._active is not None

    def submit(self, request: Message):
        configuration = validate_request(request)
        with self._lock:
            if self._closing:
                raise JobError("job.closing", "Compute worker is closing")
            if self._active is not None:
                raise JobError("job.busy", "This compute role already owns an active job")
            if any(job.request.run_id == request.run_id for job in self._jobs.values()):
                raise JobError("job.duplicateRun", "This runID already has a job; inspect its status")
            job = _Job(str(uuid.uuid4()), request, configuration)
            # Ack is queued before execution can emit any progress/completion.
            self._send("ack", {"jobID": job.identifier, "status": "queued"}, request=request)
            self._jobs[job.identifier] = job
            while len(self._jobs) > 32: self._jobs.popitem(last=False)
            self._active = job
            self._queue.put_nowait(job)
        return job.identifier

    def status(self, identifier: str):
        with self._lock:
            if identifier not in self._jobs:
                raise JobError("job.notFound", "Job is no longer in this worker's bounded status history")
            return self._jobs[identifier].snapshot()

    def external_boundary(self,request):
        _object(request.payload,("jobID","auditPath"),("jobID","auditPath"))
        _path(request.payload["auditPath"])
        with self._lock:
            job=self._jobs.get(str(uuid.UUID(request.payload["jobID"])))
            if job is None or job.request.kind!="train.reinforcement.external" or request.run_id is None or uuid.UUID(request.run_id)!=uuid.UUID(job.request.run_id):
                raise JobError("job.boundaryMismatch","External boundary belongs to another job/run")
            if job.status in TERMINAL or job.metrics.get("phase")!="waiting_for_actor_boundary" or job.external_boundary_path is not None:
                raise JobError("job.boundaryState","The learner is not waiting for its first actor boundary proof")
            job.external_boundary_path=request.payload["auditPath"]
            self._send("ack",{"jobID":job.identifier,"status":"boundary_queued"},request=request)
            job.external_boundary_event.set()

    def cancel(self, identifier: str, *, run_id: str | None = None):
        with self._lock:
            if identifier not in self._jobs:
                raise JobError("job.notFound", "Job is not known to this compute role")
            job = self._jobs[identifier]
            if run_id is not None and run_id != job.request.run_id:
                raise JobError("job.runMismatch", "Cancellation belongs to a different run")
            if job.status not in TERMINAL:
                job.cancel.set(); job.status = "cancelling"
            return job.snapshot()

    def close(self, timeout=30.0):
        with self._lock:
            self._closing = True
            if self._active is not None:
                self._active.cancel.set()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _progress(self, job, metrics):
        with self._lock:
            job.metrics = copy.deepcopy(metrics)
        try:
            self._send("job.progress", {"jobID": job.identifier, **metrics}, request=job.request)
        except Exception:
            # Losing the consumer requests orderly cancellation; it must not
            # throw through a completed gradient chunk and skip its checkpoint.
            with self._lock:
                job.cancel.set(); self._closing = True

    def _work(self):
        while True:
            try:
                job = self._queue.get(timeout=.1)
            except queue.Empty:
                with self._lock:
                    if self._closing: return
                continue
            with self._lock:
                if job.status != "cancelling": job.status = "running"
            try:
                if job.cancel.is_set(): raise InterruptedError("Job cancelled before execution")
                self._progress(job, {"phase": "preparing"})
                result = self._execute(job)
                with self._lock:
                    job.result = result
                    job.status = "cancelled" if result.get("cancelled") else "completed"
            except InterruptedError as error:
                with self._lock:
                    job.status = "cancelled"
                    job.result = {"cancelled": True, "reason": str(error), "checkpointPublished": False}
            except Exception as error:
                with self._lock:
                    job.status = "failed"
                    job.error = {"code": getattr(error, "code", "job.executionFailed"),
                                 "message": str(error)[:2048] or type(error).__name__, "recoverable": True}
            finally:
                with self._lock:
                    snapshot = job.snapshot()
                    self._active = None
                try:
                    self._send(f"job.{job.status}", snapshot, request=job.request)
                except Exception:
                    # The transport owns its fatal flag. The control loop will
                    # cancel/close rather than accumulating undeliverable data.
                    with self._lock: self._closing = True
                self._queue.task_done()

    @contextmanager
    def _source(self, specification, model, vocabulary, cancelled):
        if cancelled(): raise InterruptedError("Job cancelled before dataset preparation")
        if specification["kind"] == "recordings":
            with DatasetReader(Path(specification["path"]), recording_root=Path(specification["recordingRoot"])) as source:
                if source.config != model or source.vocabulary != vocabulary:
                    raise JobError("job.datasetMismatch", "Dataset revision does not match checkpoint configuration and action vocabulary")
                yield source, source.manifest["id"], "recorded_demonstrations"
        else:
            environment = PracticeConfig.from_dict(specification["environment"])
            identity = {"provenance": "practice_oracle", "environment": environment.to_dict(),
                        "model": model.to_dict(), "seedsBySplit": specification["seedsBySplit"]}
            encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
            identifier = str(uuid.uuid5(uuid.NAMESPACE_URL, "astra.practice.dataset.v1:" + hashlib.sha256(encoded.encode()).hexdigest()))
            source = PracticeDemonstrations(environment=environment, model=model, seeds_by_split=specification["seedsBySplit"], cancelled=cancelled)
            if source.vocabulary != vocabulary:
                raise JobError("job.datasetMismatch", "Practice environment and checkpoint expose different action vocabularies")
            if cancelled(): raise InterruptedError("Job cancelled during fixture preparation")
            yield source, identifier, "practice_oracle"

    def _execute(self, job):
        value, operation = job.configuration, job.request.kind
        if operation=="feedback.combine":
            from astra.learning.review_batches import combine
            return combine(value["fragments"],value["destination"],job.cancel.is_set)
        if operation in ("feedback.inspect","feedback.materialize"):
            from astra.learning.review_pipeline import inspect_source,materialize
            if operation=="feedback.inspect":return inspect_source(value["sourcePath"],value["manifestSHA256"])
            return materialize(value["sourcePath"],value["manifestSHA256"],value["revisionDirectory"],value["revisionChain"],value["destination"],job.cancel.is_set)
        if operation == "checkpoint.inspect":
            loaded = load_checkpoint(Path(value["path"]))
            return {"manifest": loaded.manifest, "path": value["path"], "integrityVerified": True,
                    "parameterCount": loaded.policy.config.parameter_count}
        if operation == "checkpoint.create":
            mx.random.seed(value.get("seed", 0))
            model = ModelConfig.from_dict(value["model"])
            policy = AgentPolicy(model, ActionVocabulary.from_dict(value["actions"]))
            if value.get("pretrainedPath") is not None:
                policy.vision.backbone.load_pretrained(Path(value["pretrainedPath"]))
            if job.cancel.is_set(): raise InterruptedError("Checkpoint initialization cancelled before publication")
            manifest = save_checkpoint(Path(value["destination"]), policy, kind="initial", step=0,
                                       training_config={"initialSeed": value.get("seed", 0),
                                                        "pretrained": value.get("pretrainedPath") is not None})
            return {"checkpointPath": value["destination"], "manifest": manifest, "checkpointPublished": True,
                    "parameterCount": policy.config.parameter_count}
        if operation == "dataset.prepare":
            selections = [RecordingSelection.from_payload(item) for item in value["selections"]]
            manifest = build_dataset(Path(value["destination"]), recording_root=Path(value["recordingRoot"]), selections=selections,
                                     config=ModelConfig.from_dict(value["model"]), vocabulary=ActionVocabulary.from_dict(value["actions"]),
                                     pointer_mode=value["pointerMode"], split_seed=value.get("splitSeed", 0), cancelled=job.cancel.is_set)
            return {"datasetPath": value["destination"], "manifest": manifest, "provenance": "recorded_demonstrations"}
        loaded = load_checkpoint(Path(value["checkpointPath"]), include_training=value.get("resume", False))
        if operation == "checkpoint.externalBoundary":
            from astra.learning.external_boundary import preserve_external_boundary
            return preserve_external_boundary(self,job,loaded)
        if operation == "train.reinforcement.external":
            from astra.learning.external_job import run_external_job
            return run_external_job(self,job,loaded)
        if operation == "train.reinforcement":
            return self._reinforcement(job, loaded)
        if operation == "train.behavioral":
            training = BehaviorConfig(**value["training"]).validate()
        else:
            training = BehaviorConfig(epochs=1, lanes=1, sequence_length=value.get("sequenceLength", 64))
        with self._source(value["dataset"], loaded.policy.config, loaded.policy.actions.vocabulary, job.cancel.is_set) as (source, dataset_id, provenance):
            if value.get("resume") and loaded.training_state is None:
                raise JobError("job.resumeUnavailable", "Checkpoint has no resumable training state")
            if not value.get("resume"):
                mx.random.seed(training.seed)
            trainer = BehaviorTrainer(loaded.policy, training, dataset_id=dataset_id,
                                      restored_state=loaded.training_state if value.get("resume") else None)
            if operation == "evaluate.behavioral":
                result = trainer.evaluate(source, split=value.get("split", "validation"), cancelled=job.cancel.is_set)
                return {"evaluation": result, "datasetID": dataset_id, "checkpointID": loaded.manifest["id"], "provenance": provenance}
            self._progress(job, {"phase": "training", "datasetID": dataset_id, "provenance": provenance})
            cancelled = False
            try:
                while trainer.epoch < training.epochs:
                    trainer.train_epoch(source, cancelled=job.cancel.is_set,
                                        on_metrics=lambda metrics: self._progress(job, {"phase": "training", "provenance": provenance, **asdict(metrics)}))
                # A cancellation arriving at the final chunk still receives an
                # accurate completed checkpoint; no completed work is undone.
            except InterruptedError:
                cancelled = True
            self._progress(job, {"phase": "checkpointing", "provenance": provenance, "cancelled": cancelled})
            metrics = {"epoch": trainer.epoch, "updates": trainer.updates, "decisions": trainer.decisions,
                       "cancelled": cancelled, "provenance": provenance}
            manifest = save_checkpoint(Path(value["destination"]), trainer.policy, kind="behavioral", step=trainer.updates,
                                       training_state=trainer.state, parent_id=loaded.manifest["id"], dataset_id=dataset_id,
                                       metrics=metrics, training_config=asdict(training))
            return {"checkpointPath": value["destination"], "manifest": manifest, "checkpointPublished": True,
                    "cancelled": cancelled, "resumable": True, "provenance": provenance,
                    "parameterCount": trainer.policy.config.parameter_count}


    def _reinforcement(self, job, loaded):
        value = job.configuration
        training = _reinforcement_config(value["training"])
        resume = value.get("resume", False)
        if resume and (loaded.training_state is None or loaded.training_state.get("kind") != "reinforcement"):
            raise JobError("job.resumeUnavailable", "Checkpoint has no resumable reinforcement state; start a new reinforcement run instead")
        if not resume:
            # BC may have saved a temporarily frozen pretrained backbone. A
            # fresh PPO run trains every group at its configured learning rate.
            loaded.policy.unfreeze()
            mx.random.seed(training.seed)
        environment = PracticeEnvironment(PracticeConfig.from_dict(value["environment"]))
        trainer = ReinforcementTrainer(loaded.policy, environment, training, policy_id=loaded.manifest["id"],
            context_ids=tuple(value.get("contextIDs", [])), restored_state=loaded.training_state if resume else None,
            scratch_directory=Path(value["destination"]).parent)
        if trainer.iteration > value["iterations"]:
            trainer.stop()
            raise JobError("job.invalidConfiguration", "The iteration target precedes this checkpoint's completed reinforcement work")
        started = time.monotonic()
        completed_metrics = None
        iteration_metrics = []
        iteration_metric_bytes = 0
        interrupted = False
        iterations_at_start = trainer.iteration
        def progress(phase, **fields):
            self._progress(job, {"phase": phase, "sourceKind": "practice_rollout", "provenance": "practice_rollout",
                "iteration": trainer.iteration, "iteration_target": value["iterations"],
                "decisions": trainer.decisions, "optimizer_updates": trainer.optimizer_updates,
                "elapsed_seconds": time.monotonic() - started, "actor_policy_id": trainer.actor_policy_id,
                "pending_policy_id": trainer.pending_policy_id, "last_iteration": completed_metrics, **fields})
        def stop_preserving_failure(reason):
            failure = sys.exception()
            try:
                trainer.stop(reason)
            except BaseException as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(f"Practice cleanup also failed ({type(cleanup_error).__name__}).")
        try:
            try:
                while trainer.iteration < value["iterations"]:
                    if job.cancel.is_set():
                        raise InterruptedError("Reinforcement training cancelled before the next rollout")
                    progress("collecting", rollout_decisions=0, rollout_target=training.rollout_decisions)
                    def collected(count):
                        progress("collecting", rollout_decisions=count, rollout_target=training.rollout_decisions)
                    def finishing_episode(count):
                        progress("finishing_episode", rollout_decisions=count,
                                 rollout_target=training.rollout_decisions)
                    def updated(count):
                        progress("updating", optimizer_updates=count)
                    # Separate callbacks expose actual work. Pending actor
                    # weights activate only inside the trainer's confirmed reset.
                    rollout = trainer.collect(cancelled=job.cancel.is_set, on_decision=collected,
                                              on_finishing_episode=finishing_episode)
                    progress("updating", rollout_decisions=len(rollout.decisions), rollout_target=training.rollout_decisions)
                    try:
                        metrics = trainer.update(rollout, cancelled=job.cancel.is_set, on_update=updated,
                            on_validation=lambda fields: progress("validating_update", **fields))
                    except InterruptedError:
                        # update restores the pre-update learner/optimizer and
                        # keeps its sealed rollout until explicitly discarded.
                        trainer.discard_rollout(rollout)
                        raise
                    completed_metrics = {**asdict(metrics), "elapsed_seconds": time.monotonic() - started}
                    iteration_metrics.append(completed_metrics)
                    metric_bytes = lambda item: len(json.dumps(item, separators=(",", ":"), allow_nan=False).encode())
                    iteration_metric_bytes += metric_bytes(completed_metrics)
                    # A final job reply shares the 1 MiB protocol budget with its
                    # checkpoint manifest and status. Bound by bytes as well as rows.
                    while len(iteration_metrics) > 1000 or iteration_metric_bytes > 512 * 1024:
                        iteration_metric_bytes -= metric_bytes(iteration_metrics.pop(0))
                    # Iteration metrics contain per-iteration update counts;
                    # the envelope retains cumulative run counters in every phase.
                    progress("iteration")
                    del rollout
            except InterruptedError:
                interrupted = True
            finally:
                # No live practice world or outstanding rollout is serialized.
                # This also clears partially collected/discarded rollout state.
                stop_preserving_failure("Reinforcement job stopped" if interrupted else "Reinforcement job finished")
            progress("checkpointing", cancelled=interrupted)
            metrics = {"iteration": trainer.iteration, "optimizer_updates": trainer.optimizer_updates,
                "decisions": trainer.decisions, "cancelled": interrupted,
                "sourceKind": "practice_rollout", "provenance": "practice_rollout",
                "elapsed_seconds": time.monotonic() - started, "requiresEnvironmentReset": True}
            if completed_metrics is not None:
                metrics["lastIteration"] = completed_metrics
            manifest = save_checkpoint(Path(value["destination"]), trainer.policy, kind="reinforcement",
                step=trainer.optimizer_updates, training_state=trainer.state, parent_id=loaded.manifest["id"],
                metrics=metrics, training_config=asdict(training))
            return {"checkpointPath": value["destination"], "manifest": manifest, "checkpointPublished": True,
                "cancelled": interrupted, "resumable": True, "requiresEnvironmentReset": True,
                "provenance": "practice_rollout", "sourceKind": "practice_rollout",
                "completedIterations": trainer.iteration, "iterationsThisJob": trainer.iteration - iterations_at_start,
                "iterationMetrics": iteration_metrics, "metricsHistoryLimit": 1000,
                "metricsHistoryByteLimit": 512 * 1024,
                "metricsHistoryDropped": trainer.iteration - iterations_at_start - len(iteration_metrics),
                "parameterCount": trainer.policy.config.parameter_count}
        finally:
            stop_preserving_failure("Reinforcement job released its environment")
