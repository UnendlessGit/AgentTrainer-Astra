"""Immutable demonstration revisions with causal timing and session splits."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import OrderedDict
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile
import uuid
from typing import Iterator

import mlx.core as mx
import numpy as np

from astra.checkpoints import _publish, _sync
from astra.model.config import ModelConfig
from astra.model.actions import ActionVocabulary
from astra.model.observation import ObservationBatch
from astra.recordings import RecordingReader, RecordingError, _json, _integer, _regular, validate_frame
from .history import ControlHistory
from .preprocessing import prepare_surface
from .actions import (canonicalize, quantized_event_nanos, ActionEncodingError, CANONICALIZER_VERSION,
                      DISCRETE_EVENT_KINDS)
from .batching import LearningSample, pointing_layout
from .actions import encode_commands

MAXIMUM_DATASET_STEPS = 100_000_000
MAXIMUM_DATASET_EPISODES = 100_000
MAXIMUM_SOURCE_READERS = 4
MAXIMUM_INTERVAL_EVENTS = 100_000
MAXIMUM_INTERVAL_EVENT_BYTES = 16 * 1024**2
MAXIMUM_SELECTION_RANGES = 256


def _selection(selection: RecordingSelection, config: ModelConfig) -> None:
    if not isinstance(selection, RecordingSelection):
        raise RecordingError("Invalid recording selection")
    uuid.UUID(selection.recording_id)
    if not isinstance(selection.context_ids, tuple) or len(selection.context_ids) != len(config.context_sizes) or any(type(value) is not int or not 0 <= value < size for value, size in zip(selection.context_ids, config.context_sizes)):
        raise RecordingError("Dataset context choices do not match model vocabularies")
    for value in (selection.start_nanos, selection.end_nanos):
        if value is not None:
            _integer(value)
    if selection.start_nanos is not None and selection.end_nanos is not None and selection.start_nanos >= selection.end_nanos:
        raise RecordingError("Source selection must have a nonempty forward interval")
    if selection.ranges is not None:
        if (selection.start_nanos is not None or selection.end_nanos is not None or
            not isinstance(selection.ranges, tuple) or not 1 <= len(selection.ranges) <= MAXIMUM_SELECTION_RANGES):
            raise RecordingError("Explicit ranges cannot mix with legacy bounds and must contain 1–256 intervals")
        previous_end = 0
        for interval in selection.ranges:
            if not isinstance(interval, RecordingRange):
                raise RecordingError("Invalid recording range")
            _integer(interval.start_nanos); _integer(interval.end_nanos)
            if interval.start_nanos >= interval.end_nanos or interval.start_nanos < previous_end:
                raise RecordingError("Recording ranges must be ordered, non-overlapping and nonempty")
            previous_end = interval.end_nanos


def _source_paths(root: Path) -> dict[str, Path]:
    paths = {}
    for path in Path(root).glob("*.astrarecord"):
        try:
            identifier = str(uuid.UUID(path.stem))
        except ValueError:
            continue
        if identifier in paths:
            raise RecordingError("Recording identities are duplicated in the library")
        paths[identifier] = path
    return paths


def _interval(manifest: dict, selection: RecordingSelection) -> tuple[int, int]:
    start = max(manifest["firstObservedNanos"], 0 if selection.start_nanos is None else selection.start_nanos)
    end = min(manifest["stoppedNanos"], manifest["stoppedNanos"] if selection.end_nanos is None else selection.end_nanos,
              manifest.get("firstInvalidObservedNanos", manifest["stoppedNanos"]))
    return start, end


@dataclass(frozen=True)
class RecordingRange:
    start_nanos: int
    end_nanos: int


@dataclass(frozen=True)
class RecordingSelection:
    recording_id: str
    start_nanos: int | None = None
    end_nanos: int | None = None
    context_ids: tuple[int, ...] = ()
    ranges: tuple[RecordingRange, ...] | None = None

    @classmethod
    def from_payload(cls, value: dict) -> RecordingSelection:
        fields = {**value, "context_ids": tuple(value.get("context_ids", []))}
        if "ranges" in fields:
            if not isinstance(fields["ranges"], list) or not 1 <= len(fields["ranges"]) <= MAXIMUM_SELECTION_RANGES:
                raise RecordingError("Recording selection needs 1–256 ranges")
            fields["ranges"] = tuple(RecordingRange(**item) for item in fields["ranges"])
        return cls(**fields)


def _ranges(manifest: dict, selection: RecordingSelection) -> tuple[RecordingRange, ...]:
    first, last = _interval(manifest, RecordingSelection(selection.recording_id))
    if selection.ranges is None:
        start, end = _interval(manifest, selection)
        return (RecordingRange(start, end),)
    if any(item.start_nanos < first or item.end_nanos > last for item in selection.ranges):
        raise RecordingError("Recording range extends beyond verified source coverage")
    return selection.ranges


def _dump(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def session_splits(identifiers: list[str], *, seed: int) -> tuple[dict[str, str], list[str]]:
    if type(seed) is not int or not -(2**63) <= seed < 2**63:
        raise ValueError("Split seed must be a signed 64-bit integer")
    if len(set(identifiers)) != len(identifiers) or not identifiers or any(type(value) is not str or not value for value in identifiers):
        raise ValueError("Dataset sources need unique nonempty session identifiers")
    ordered = sorted(identifiers, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    count = len(ordered)
    validation = max(1, round(count * 0.1)) if count >= 2 else 0
    test = max(1, round(count * 0.1)) if count >= 3 else 0
    training = count - validation - test
    split = {value: "train" if index < training else "validation" if index < training + validation else "test" for index, value in enumerate(ordered)}
    warnings = []
    if count == 1:
        warnings.append("Only one independent session: all samples train; held-out evaluation is unavailable.")
    elif count == 2:
        warnings.append("Two independent sessions: one trains and one validates; an independent test set is unavailable.")
    return split, warnings


def _validate_partition(value: dict, config: ModelConfig) -> None:
    expected = {"gridOriginNanos", "executionEndNanos", "sourceStartNanos", "sourceEndNanos",
                "assignedPhysicalEvents", "assignedDiscreteEvents", "excludedPhysicalEvents",
                "excludedDiscreteEvents", "exclusionExamples"}
    if not isinstance(value, dict) or set(value) != expected:
        raise RecordingError("Invalid dataset label partition")
    for name in ("gridOriginNanos", "executionEndNanos", "sourceStartNanos", "sourceEndNanos",
                 "assignedPhysicalEvents", "assignedDiscreteEvents"):
        _integer(value[name])
    origin, end = value["gridOriginNanos"], value["executionEndNanos"]
    period = config.period_ms * 1_000_000
    if (origin != value["sourceStartNanos"] + config.lead_ms * 1_000_000 or end <= origin or
            end != origin + ((value["sourceEndNanos"] - origin) // period) * period or
            value["assignedDiscreteEvents"] > value["assignedPhysicalEvents"]):
        raise RecordingError("Dataset label partition does not match its execution grid")
    for name in ("excludedPhysicalEvents", "excludedDiscreteEvents"):
        counts = value[name]
        if not isinstance(counts, dict) or set(counts) != {"before", "after"}:
            raise RecordingError("Invalid dataset label exclusion counts")
        for count in counts.values():
            _integer(count)
    if any(value["excludedDiscreteEvents"][side] > value["excludedPhysicalEvents"][side] for side in ("before", "after")):
        raise RecordingError("Invalid discrete-event exclusion count")
    examples = value["exclusionExamples"]
    if not isinstance(examples, list) or len(examples) > min(8, sum(value["excludedPhysicalEvents"].values())):
        raise RecordingError("Invalid dataset label exclusion examples")
    for sample in examples:
        if not isinstance(sample, dict) or set(sample) != {"sequence", "eventNanos", "quantizedNanos", "side"}:
            raise RecordingError("Invalid dataset label exclusion example")
        for name in ("sequence", "eventNanos", "quantizedNanos"):
            _integer(sample[name])
        quantized = quantized_event_nanos(sample["eventNanos"], grid_origin_nanos=origin)
        if (not value["sourceStartNanos"] <= sample["eventNanos"] < value["sourceEndNanos"] or
                sample["quantizedNanos"] != quantized or sample["side"] not in ("before", "after") or
                (sample["side"] == "before" and quantized >= origin) or (sample["side"] == "after" and quantized < end)):
            raise RecordingError("Dataset label exclusion is inconsistent with its source/grid")


def build_dataset(destination: Path, *, recording_root: Path, selections: list[RecordingSelection],
                  config: ModelConfig, vocabulary: ActionVocabulary, pointer_mode: str,
                  split_seed: int = 0, cancelled=lambda: False) -> dict:
    config.validate(); vocabulary.validate()
    if config.control_width != 178 or pointer_mode not in ("absolute", "relative", "disabled"):
        raise RecordingError("Dataset control schema or pointer mode is unsupported")
    destination = Path(destination)
    identifier = str(uuid.UUID(destination.name))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Dataset revisions are immutable")
    for selection in selections:
        _selection(selection, config)
    if sum(len(selection.ranges) if selection.ranges is not None else 1 for selection in selections) > MAXIMUM_DATASET_EPISODES:
        raise RecordingError("Dataset selection exceeds its range budget")
    ids = [str(uuid.UUID(selection.recording_id)) for selection in selections]
    if len(ids) > 65536:
        raise RecordingError("Dataset source count exceeds its supported bounds")
    splits, warnings = session_splits(ids, seed=split_seed)
    paths = _source_paths(recording_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dataset-", dir=destination.parent))
    database = None
    try:
        database = sqlite3.connect(staging / "index.sqlite")
        database.execute("PRAGMA synchronous=FULL")
        database.execute("CREATE TABLE episodes (id TEXT PRIMARY KEY, recording_id TEXT NOT NULL, split TEXT NOT NULL, steps INTEGER NOT NULL, selection_index INTEGER NOT NULL)")
        database.execute("CREATE TABLE steps (episode_id TEXT NOT NULL, step INTEGER NOT NULL, cutoff INTEGER NOT NULL, frame BLOB NOT NULL, controls BLOB NOT NULL, pointer BLOB, commands BLOB NOT NULL, context BLOB NOT NULL, PRIMARY KEY(episode_id,step))")
        sources = []
        errors = []
        total = 0
        episode_count = 0
        for selection, recording_id in zip(selections, ids):
            if cancelled():
                raise InterruptedError("Dataset creation cancelled")
            if recording_id not in paths:
                raise RecordingError("Selected recording is missing or duplicated in the library")
            with RecordingReader(paths[recording_id]) as reader, ExitStack() as streams:
                manifest = reader.manifest
                if manifest["frameCount"] == 0:
                    raise RecordingError("Selected recording has no complete frames")
                roles = list(reader.connection.execute("SELECT DISTINCT json_extract(CAST(block AS TEXT),'$.metadata.surface.id') FROM frames LIMIT 2"))
                if len(roles) != 1:
                    raise RecordingError("This recording needs aligned multi-surface observations; single-surface dataset construction cannot discard its other surfaces")
                chosen_ranges = _ranges(manifest, selection)
                source_spec = {"id": recording_id, "manifestSHA256": hashlib.sha256(_dump(manifest).encode()).hexdigest(),
                               "split": splits[recording_id], "selection": {"recording_id": recording_id,
                               "ranges": [asdict(item) for item in chosen_ranges], "context_ids": list(selection.context_ids)},
                               "labelPartitions": []}
                sources.append(source_spec)
                # Two independent event cursors prevent the shifted label stream
                # from contaminating causal control history at t.
                observations = iter(reader.events())
                labels = iter(reader.events(time_axis="source_time"))
                streams.callback(observations.close)
                streams.callback(labels.close)
                observed = next(observations, None)
                labeled = next(labels, None)
                history = ControlHistory()
                frame_stream = iter(reader.frames())
                streams.callback(frame_stream.close)
                future_frame = next(frame_stream, None)
                current_frame = None
                verified_frame = None
                prepared_layout = None
                label_pointer = None
                # Sorted ranges share monotonic source cursors, but each range
                # starts a new recurrent episode and discards transient gap deltas.
                for selection_index, chosen_range in enumerate(chosen_ranges):
                    start, end = chosen_range.start_nanos, chosen_range.end_nanos
                    period = config.period_ms * 1_000_000
                    lead = config.lead_ms * 1_000_000
                    if start + lead + period > end:
                        raise RecordingError("Selected recording has no complete causal observation/action interval")
                    if total + (end - start - lead) // period > MAXIMUM_DATASET_STEPS:
                        raise RecordingError("Dataset exceeds its supported decision-count budget")
                    execution_origin = start + lead
                    execution_end = execution_origin + ((end - execution_origin) // period) * period
                    partition = {"gridOriginNanos": execution_origin, "executionEndNanos": execution_end,
                                 "sourceStartNanos": start, "sourceEndNanos": end,
                                 "assignedPhysicalEvents": 0, "assignedDiscreteEvents": 0,
                                 "excludedPhysicalEvents": {"before": 0, "after": 0},
                                 "excludedDiscreteEvents": {"before": 0, "after": 0}, "exclusionExamples": []}
                    source_spec["labelPartitions"].append(partition)
                    # A mid-session selection retains held state and age, but its
                    # transient deltas cover only the immediately preceding period.
                    while observed is not None and observed["observedNanos"] <= max(0, start - period):
                        history.apply(observed)
                        observed = next(observations, None)
                    history.advance_interval()
                    def consume_label():
                        nonlocal labeled, label_pointer
                        event = labeled
                        quantized = quantized_event_nanos(event["eventNanos"], grid_origin_nanos=execution_origin)
                        if execution_origin <= event["eventNanos"] < execution_end and (event["origin"] == "boundary" or event["kind"] == "gap"):
                            raise ActionEncodingError("Selected action coverage contains an input discontinuity, including at a quantized boundary")
                        if event["origin"] == "physical" and start <= event["eventNanos"] < end:
                            discrete = event["kind"] in DISCRETE_EVENT_KINDS
                            if execution_origin <= quantized < execution_end:
                                partition["assignedPhysicalEvents"] += 1
                                partition["assignedDiscreteEvents"] += int(discrete)
                            else:
                                side = "before" if quantized < execution_origin else "after"
                                partition["excludedPhysicalEvents"][side] += 1
                                partition["excludedDiscreteEvents"][side] += int(discrete)
                                if len(partition["exclusionExamples"]) < 8:
                                    partition["exclusionExamples"].append({"sequence": event["sequence"],
                                        "eventNanos": event["eventNanos"], "quantizedNanos": quantized, "side": side})
                        if event.get("x") is not None and event.get("y") is not None and event["origin"] in ("physical", "reconciliation"):
                            label_pointer = (event["x"], event["y"])
                        labeled = next(labels, None)
                        return event
                    current_geometry = None
                    episode = str(uuid.uuid4()); episode_step = 0
                    def seal_episode():
                        nonlocal episode_count
                        if episode_step:
                            episode_count += 1
                            if episode_count > MAXIMUM_DATASET_EPISODES:
                                raise RecordingError("Dataset episode count exceeds its supported bounds")
                            database.execute("INSERT INTO episodes VALUES(?,?,?,?,?)", (episode, recording_id, splits[recording_id], episode_step, selection_index))
                    for cutoff in range(start, end - lead - period + 1, period):
                        if cancelled():
                            raise InterruptedError("Dataset creation cancelled")
                        while observed is not None and observed["observedNanos"] <= cutoff:
                            if observed["eventNanos"] > cutoff:
                                raise RecordingError("Source input time is beyond the causal observation cutoff")
                            history.apply(observed)
                            observed = next(observations, None)
                        while future_frame is not None and future_frame["block"]["metadata"]["observedNanos"] <= cutoff:
                            candidate = future_frame
                            future_frame = next(frame_stream, None)
                            if candidate["block"]["metadata"]["eventNanos"] <= cutoff:
                                current_frame = candidate
                        if current_frame is None:
                            raise RecordingError("Selected interval has no causal observation; action labels cannot be discarded")
                        metadata = current_frame["block"]["metadata"]
                        surface = metadata["surface"]
                        geometry = _dump(surface)
                        if current_geometry is not None and geometry != current_geometry:
                            seal_episode(); episode = str(uuid.uuid4()); episode_step = 0
                        current_geometry = geometry
                        execution = cutoff + lead
                        # One source-time lookahead is also the boundary carry:
                        # events rounding onto the next packet remain unconsumed.
                        # Pre-selection state can seed a pointer, but never a label.
                        while labeled is not None and (labeled["eventNanos"] < start or
                                quantized_event_nanos(labeled["eventNanos"], grid_origin_nanos=execution_origin) < execution):
                            consume_label()
                        interval = []
                        interval_bytes = 0
                        initial_pointer = label_pointer
                        while labeled is not None and labeled["eventNanos"] < end and (
                                quantized_event_nanos(labeled["eventNanos"], grid_origin_nanos=execution_origin) < execution + period):
                            interval_bytes += len(_dump(labeled).encode())
                            if len(interval) >= MAXIMUM_INTERVAL_EVENTS or interval_bytes > MAXIMUM_INTERVAL_EVENT_BYTES:
                                raise ActionEncodingError("Action interval exceeds its bounded raw-input budget; select a faster cadence")
                            interval.append(consume_label())
                        if not history.valid:
                            raise RecordingError("Selected interval has no valid causal initial input state")
                        try:
                            packet = canonicalize(interval, start_nanos=execution, config=config, vocabulary=vocabulary,
                                                  surfaces=[surface], pointer_mode=pointer_mode, initial_pointer=initial_pointer)
                            if verified_frame != metadata["id"]:
                                prepared = prepare_surface(reader.pixels(current_frame), metadata, config, pointer=None)
                                observation = ObservationBatch((prepared.batch(),), mx.zeros((1, 1, config.control_width)), mx.zeros((1, 1)),
                                                                mx.zeros((1, 1, len(config.context_sizes)), dtype=mx.int32), mx.array([[True]]), mx.array([[True]]))
                                prepared_layout = pointing_layout(observation)
                                verified_frame = metadata["id"]
                            encode_commands(packet.commands, config=config, vocabulary=vocabulary, visual=prepared_layout, surfaces=[surface])
                        except ActionEncodingError as error:
                            if len(errors) < 100:
                                errors.append({"recordingID": recording_id, "cutoffNanos": cutoff, "message": str(error)})
                            continue
                        # Validate pixels/checksum at construction. Readers recheck
                        # again on use so changed source can never silently train.
                        controls = history.features(cutoff, [surface], interval_covered=True)
                        database.execute("INSERT INTO steps VALUES(?,?,?,?,?,?,?,?)", (episode, episode_step, cutoff,
                                         _dump(current_frame).encode(), controls.tobytes(), _dump(history.pointer).encode(),
                                         _dump(packet.commands).encode(), _dump(selection.context_ids).encode()))
                        history.advance_interval(); episode_step += 1; total += 1
                    seal_episode()
                    # The incomplete tail and half-ms boundary carry are explicit
                    # exclusions, never silently lost or moved into the last packet.
                    while labeled is not None and labeled["eventNanos"] < end:
                        if cancelled():
                            raise InterruptedError("Dataset creation cancelled")
                        consume_label()
                    database.commit()
        if errors:
            preview = "\n".join(f"{item['recordingID']} at {item['cutoffNanos']}: {item['message']}" for item in errors[:5])
            raise ActionEncodingError("Dataset action fit failed; no partial revision was published.\n" + preview)
        if not total:
            raise RecordingError("Dataset contains no usable control intervals")
        excluded = sum(sum(partition["excludedPhysicalEvents"].values()) for source in sources for partition in source["labelPartitions"])
        excluded_discrete = sum(sum(partition["excludedDiscreteEvents"].values()) for source in sources for partition in source["labelPartitions"])
        if excluded:
            warnings.append(f"Excluded {excluded} selected physical events ({excluded_discrete} control transitions) outside complete action windows. "
                            "Each source's label partition records exact counts and up to eight boundary examples; the raw recording is unchanged.")
        database.close(); database = None
        _sync(staging / "index.sqlite")
        if (staging / "index.sqlite").stat().st_size > 8 * 1024**3:
            raise RecordingError("Dataset index exceeds its supported size")
        with (staging / "index.sqlite").open("rb") as stream:
            index_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest = {"schemaVersion": 2, "id": identifier, "model": config.to_dict(), "actions": vocabulary.to_dict(),
                    "canonicalizerVersion": CANONICALIZER_VERSION, "pointerMode": pointer_mode, "splitSeed": split_seed,
                    "sources": sources, "steps": total, "warnings": warnings, "indexSHA256": index_digest}
        description = _dump(manifest).encode()
        if len(description) > 8 * 1024**2:
            raise RecordingError("Dataset manifest exceeds its supported size")
        (staging / "manifest.json").write_bytes(description)
        if cancelled():
            raise InterruptedError("Dataset creation cancelled")
        _sync(staging / "manifest.json"); _sync(staging); _publish(staging, destination)
        return manifest
    finally:
        if database is not None:
            database.close()
        if staging.exists():
            shutil.rmtree(staging)


class DatasetReader:
    def __init__(self, directory: Path, *, recording_root: Path):
        self.directory = Path(directory)
        self.recording_root = Path(recording_root)
        self._sources = OrderedDict()
        self._database = None
        try:
            if self.directory.is_symlink() or not self.directory.is_dir():
                raise RecordingError("A dataset package must be a local directory")
            path = self.directory / "manifest.json"
            _regular(path)
            if path.stat().st_size > 8 * 1024**2:
                raise RecordingError("Dataset manifest exceeds its supported size")
            self.manifest = _json(path.read_bytes(), limit=8 * 1024**2)
            manifest = self.manifest
            required = {"schemaVersion", "id", "model", "actions", "canonicalizerVersion", "pointerMode", "splitSeed", "sources", "steps", "warnings", "indexSHA256"}
            if (not isinstance(manifest, dict) or set(manifest) != required
                or type(manifest["schemaVersion"]) is not int or manifest["schemaVersion"] not in (1, 2)
                or manifest["id"] != str(uuid.UUID(self.directory.name))
                or type(manifest["canonicalizerVersion"]) is not int or manifest["canonicalizerVersion"] != CANONICALIZER_VERSION):
                raise RecordingError("Unsupported dataset identity/version")
            self.config = ModelConfig.from_dict(manifest["model"])
            self.vocabulary = ActionVocabulary.from_dict(manifest["actions"])
            if self.config.control_width != 178 or manifest["pointerMode"] not in ("absolute", "relative", "disabled"):
                raise RecordingError("Dataset control schema or pointer mode is unsupported")
            _integer(manifest["steps"], 1, MAXIMUM_DATASET_STEPS)
            if not isinstance(manifest["warnings"], list) or len(manifest["warnings"]) > 100 or any(type(value) is not str or len(value) > 4096 for value in manifest["warnings"]):
                raise RecordingError("Dataset warnings are invalid")
            if not isinstance(manifest["sources"], list) or not 1 <= len(manifest["sources"]) <= 65536:
                raise RecordingError("Dataset sources exceed their supported bounds")
            self._source_specs = {}
            for source in manifest["sources"]:
                partition_key = "labelPartition" if manifest["schemaVersion"] == 1 else "labelPartitions"
                if not isinstance(source, dict) or set(source) != {"id", "manifestSHA256", "split", "selection", partition_key}:
                    raise RecordingError("Invalid dataset source description")
                identifier = str(uuid.UUID(source["id"]))
                selection = source["selection"]
                if (identifier != source["id"] or identifier in self._source_specs or not isinstance(selection, dict)
                    or set(selection) != ({"recording_id", "start_nanos", "end_nanos", "context_ids"} if manifest["schemaVersion"] == 1
                                          else {"recording_id", "ranges", "context_ids"})
                    or not isinstance(selection["context_ids"], list)):
                    raise RecordingError("Invalid or duplicated dataset source identity")
                if manifest["schemaVersion"] == 2 and (not isinstance(selection["ranges"], list) or
                        any(not isinstance(item, dict) or set(item) != {"start_nanos", "end_nanos"} for item in selection["ranges"])):
                    raise RecordingError("Invalid dataset recording ranges")
                chosen = RecordingSelection.from_payload(selection)
                _selection(chosen, self.config)
                if str(uuid.UUID(chosen.recording_id)) != identifier:
                    raise RecordingError("Dataset selection belongs to a different recording")
                self._digest(source["manifestSHA256"])
                partitions = [source["labelPartition"]] if manifest["schemaVersion"] == 1 else source["labelPartitions"]
                if not isinstance(partitions, list) or len(partitions) != (len(chosen.ranges) if chosen.ranges is not None else 1):
                    raise RecordingError("Dataset range partitions do not match its selection")
                for partition in partitions:
                    _validate_partition(partition, self.config)
                self._source_specs[identifier] = {**source, "chosen": chosen, "partitions": partitions}
            if sum(len(source["partitions"]) for source in self._source_specs.values()) > MAXIMUM_DATASET_EPISODES:
                raise RecordingError("Dataset exceeds its supported selection range budget")
            splits, _ = session_splits(list(self._source_specs), seed=manifest["splitSeed"])
            if any(source["split"] != splits[identifier] for identifier, source in self._source_specs.items()):
                raise RecordingError("Dataset splits are inconsistent with independent source sessions")
            self._paths = _source_paths(self.recording_root)
            index = self.directory / "index.sqlite"
            _regular(index)
            if index.stat().st_size > 8 * 1024**3:
                raise RecordingError("Dataset index exceeds its supported size")
            self._digest(manifest["indexSHA256"])
            if Path(str(index) + "-wal").exists() and Path(str(index) + "-wal").stat().st_size:
                raise RecordingError("Dataset index contains an unpublished WAL")
            with index.open("rb") as stream:
                index_digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if index_digest != manifest["indexSHA256"]:
                raise RecordingError("Dataset index integrity check failed")
            self._database = sqlite3.connect(index.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
            self._database.row_factory = sqlite3.Row
            self._database.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1024**2)
            self._database.execute("PRAGMA query_only=ON")
            self._database.execute("PRAGMA trusted_schema=OFF")
            if self._database.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RecordingError("Dataset index failed its SQLite integrity check")
            self._validate_index()
            # Verify every source up front without retaining one descriptor set
            # per recording. Reopened cache entries repeat their identity check.
            for identifier in self._source_specs:
                self._source(identifier)
        except BaseException:
            self.close(); raise

    @staticmethod
    def _digest(value):
        if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise RecordingError("Invalid dataset artifact digest")

    def _source(self, identifier: str) -> RecordingReader:
        reader = self._sources.pop(identifier, None)
        if reader is None:
            if identifier not in self._paths:
                raise RecordingError("Dataset recording is missing from the selected library root")
            # Evict before opening so even construction never exceeds the cap.
            while len(self._sources) >= MAXIMUM_SOURCE_READERS:
                _, oldest = self._sources.popitem(last=False)
                oldest.close()
            reader = RecordingReader(self._paths[identifier])
            try:
                if hashlib.sha256(_dump(reader.manifest).encode()).hexdigest() != self._source_specs[identifier]["manifestSHA256"]:
                    raise RecordingError("Dataset source changed; create a new revision for the recovered or edited source")
                if not reader.manifest["frameCount"] or "firstObservedNanos" not in reader.manifest:
                    raise RecordingError("Dataset source has no complete observation")
                spec = self._source_specs[identifier]
                chosen_ranges = _ranges(reader.manifest, spec["chosen"])
                if any((partition["sourceStartNanos"], partition["sourceEndNanos"]) != (interval.start_nanos, interval.end_nanos)
                       for partition, interval in zip(spec["partitions"], chosen_ranges)):
                    raise RecordingError("Dataset label partition does not match the source selection")
            except BaseException:
                reader.close(); raise
        self._sources[identifier] = reader
        return reader

    def _validate_index(self):
        range_names = ["selection_index"] if self.manifest["schemaVersion"] == 2 else []
        for table, names, keys in (
            ("episodes", ["id", "recording_id", "split", "steps"] + range_names, [1, 0, 0, 0] + [0] * len(range_names)),
            ("steps", ["episode_id", "step", "cutoff", "frame", "controls", "pointer", "commands", "context"], [1, 2, 0, 0, 0, 0, 0, 0]),
        ):
            columns = list(self._database.execute(f"PRAGMA table_info({table})"))
            if [row["name"] for row in columns] != names or [row["pk"] for row in columns] != keys:
                raise RecordingError("Dataset index schema is incompatible")
        counts = self._database.execute("SELECT (SELECT COUNT(*) FROM episodes), (SELECT COUNT(*) FROM steps)").fetchone()
        if not 1 <= counts[0] <= MAXIMUM_DATASET_EPISODES or counts[1] != self.manifest["steps"]:
            raise RecordingError("Dataset index counts do not match its manifest")
        total = 0
        cursor = self._database.execute("SELECT e.*, COUNT(s.step) AS actual, MIN(s.step) AS first, MAX(s.step) AS last, MIN(s.cutoff) AS first_cutoff, MAX(s.cutoff) AS last_cutoff FROM episodes e LEFT JOIN steps s ON s.episode_id=e.id GROUP BY e.id")
        try:
            for row in cursor:
                if str(uuid.UUID(row["id"])) != row["id"] or row["recording_id"] not in self._source_specs:
                    raise RecordingError("Dataset episode has an unknown source or identity")
                size = _integer(row["steps"], 1, MAXIMUM_DATASET_STEPS)
                partitions = self._source_specs[row["recording_id"]]["partitions"]
                index = _integer(row["selection_index"], 0, len(partitions) - 1) if range_names else 0
                partition = partitions[index]
                first_cutoff, last_cutoff = _integer(row["first_cutoff"]), _integer(row["last_cutoff"])
                if (first_cutoff < partition["sourceStartNanos"] or
                        last_cutoff + (self.config.lead_ms + self.config.period_ms) * 1_000_000 > partition["sourceEndNanos"]):
                    raise RecordingError("Dataset episode escapes its selected recording range")
                if row["split"] != self._source_specs[row["recording_id"]]["split"] or (row["actual"], row["first"], row["last"]) != (size, 0, size - 1):
                    raise RecordingError("Dataset episode steps or source split are inconsistent")
                total += size
        finally:
            cursor.close()
        if total != self.manifest["steps"]:
            raise RecordingError("Dataset contains orphaned steps")
        fault = self._database.execute("SELECT 1 FROM steps WHERE typeof(step)!='integer' OR typeof(cutoff)!='integer' OR cutoff<0 LIMIT 1").fetchone()
        gap = self._database.execute("SELECT 1 FROM steps s JOIN steps p ON s.episode_id=p.episode_id AND s.step=p.step+1 WHERE s.cutoff-p.cutoff!=? LIMIT 1", (self.config.period_ms * 1_000_000,)).fetchone()
        if fault is not None or gap is not None:
            raise RecordingError("Dataset temporal sequence is invalid or discontinuous")

    def close(self):
        for reader in self._sources.values():
            reader.close()
        self._sources.clear()
        if self._database is not None:
            self._database.close(); self._database = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    def episodes(self, split: str = "train") -> list[dict]:
        if self._database is None:
            raise RecordingError("Dataset is closed")
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown dataset split")
        return [dict(row) for row in self._database.execute("SELECT * FROM episodes WHERE split=? ORDER BY id", (split,))]

    def samples(self, episode_id: str, start: int = 0, count: int | None = None) -> Iterator[LearningSample]:
        if self._database is None:
            raise RecordingError("Dataset is closed")
        episode = self._database.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()
        if episode is None or type(start) is not int or not 0 <= start <= MAXIMUM_DATASET_STEPS or (count is not None and (type(count) is not int or not 1 <= count <= MAXIMUM_DATASET_STEPS)):
            raise ValueError("Invalid dataset episode range")
        source = self._source_specs[episode["recording_id"]]
        selection = source["chosen"]
        selection_index = episode["selection_index"] if self.manifest["schemaVersion"] == 2 else 0
        cursor = self._database.execute("SELECT * FROM steps WHERE episode_id=? AND step>=? AND step<? ORDER BY step", (episode_id, start, start + count if count is not None else episode["steps"]))
        try:
            for row in cursor:
                if self._database is None:
                    raise RecordingError("Dataset is closed")
                if any(not isinstance(row[field], bytes) for field in ("frame", "pointer", "commands", "context")):
                    raise RecordingError("Dataset row metadata must use bounded JSON blobs")
                reader = self._source(episode["recording_id"])
                chosen_range = _ranges(reader.manifest, selection)[selection_index]
                first, end = chosen_range.start_nanos, chosen_range.end_nanos
                cutoff = _integer(row["cutoff"])
                if not first <= cutoff or cutoff + (self.config.lead_ms + self.config.period_ms) * 1_000_000 > end:
                    raise RecordingError("Dataset action interval escapes its source selection or valid coverage")
                reference = _json(row["frame"])
                if not isinstance(reference, dict) or set(reference) != {"shard", "block"} or not isinstance(reference["block"], dict):
                    raise RecordingError("Invalid dataset frame reference")
                metadata = validate_frame(reference["block"].get("metadata"))
                if max(metadata["eventNanos"], metadata["observedNanos"]) > cutoff:
                    raise RecordingError("Dataset frame would leak beyond the observation cutoff")
                pointer = _json(row["pointer"])
                if pointer is not None and (not isinstance(pointer, list) or len(pointer) != 2 or any(type(value) not in (int, float) or not math.isfinite(value) for value in pointer)):
                    raise RecordingError("Dataset pointer state is invalid")
                prepared = prepare_surface(reader.pixels(reference), metadata, self.config, pointer=pointer)
                if not isinstance(row["controls"], bytes) or len(row["controls"]) != self.config.control_width * 4:
                    raise RecordingError("Dataset control feature storage is invalid")
                controls = np.frombuffer(row["controls"], dtype=np.float32)
                if controls.shape != (self.config.control_width,) or not np.isfinite(controls).all():
                    raise RecordingError("Dataset control features are invalid")
                context = _json(row["context"])
                if context != list(selection.context_ids) or any(type(value) is not int for value in context):
                    raise RecordingError("Dataset context differs from its immutable source selection")
                observation = ObservationBatch((prepared.batch(),), mx.array(controls)[None, None], mx.array([[self.config.period_ms / 1000]]),
                                                mx.array(context, dtype=mx.int32)[None, None], mx.array([[row["step"] == 0]]), mx.array([[True]]))
                commands = _json(row["commands"])
                if not isinstance(commands, list):
                    raise RecordingError("Dataset commands must be an ordered packet")
                encode_commands(commands, config=self.config, vocabulary=self.vocabulary, visual=pointing_layout(observation), surfaces=[metadata["surface"]])
                yield LearningSample(observation, (metadata["surface"],), tuple(commands), episode_id, row["step"])
        finally:
            if self._database is not None:
                cursor.close()
