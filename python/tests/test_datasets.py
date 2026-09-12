from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import uuid

import numpy as np
import pytest

import astra.data.datasets as datasets
from astra.data.actions import ActionEncodingError
from astra.data.batching import training_batch
from astra.data.datasets import DatasetReader, RecordingSelection, RecordingRange, build_dataset, session_splits
from astra.model.actions import ActionVocabulary
from astra.model.config import ModelConfig
from astra.recordings import RecordingError
from test_recordings import native_recording  # Runs AstraFixture, never capture/input APIs.


def _configuration(lead=0):
    return replace(ModelConfig.test_small(), period_ms=20, lead_ms=lead)


def _vocabulary():
    return ActionVocabulary((13,), (), True, False, True)


def _build(tmp_path, native_recording, *, config=None, vocabulary=None, selection=None):
    source = Path(native_recording["directory"])
    destination = tmp_path / str(uuid.uuid4())
    manifest = build_dataset(destination, recording_root=source.parent,
                             selections=[selection or RecordingSelection(str(uuid.UUID(source.stem)))],
                             config=config or _configuration(), vocabulary=vocabulary or _vocabulary(), pointer_mode="absolute")
    return destination, manifest


def _clone_source(native_recording, root):
    original = Path(native_recording["directory"])
    identifier = str(uuid.uuid4())
    destination = root / (identifier + ".astrarecord")
    shutil.copytree(original, destination)
    path = destination / "manifest.json"
    manifest = json.loads(path.read_bytes()); manifest["id"] = identifier
    path.write_text(json.dumps(manifest))
    return destination, manifest


def _rehash_index(directory):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["indexSHA256"] = hashlib.sha256((directory / "index.sqlite").read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))


def _write_legacy_dataset_format(directory):
    """Explicit archived schema-1 layout, retaining the same immutable episode IDs."""
    path = directory / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["schemaVersion"] = 1
    for source in manifest["sources"]:
        ranges = source["selection"].pop("ranges")
        assert len(ranges) == 1
        source["selection"].update(ranges[0])
        source["labelPartition"] = source.pop("labelPartitions")[0]
    with sqlite3.connect(directory / "index.sqlite") as database:
        database.execute("ALTER TABLE episodes DROP COLUMN selection_index")
    manifest["indexSHA256"] = hashlib.sha256((directory / "index.sqlite").read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))


def test_multiple_ranges_share_one_session_split_and_preserve_causal_state_across_gaps(tmp_path, native_recording):
    source, source_manifest = _clone_source(native_recording, tmp_path / "sources")
    _append_events_to_fixture(source, source_manifest, [
        {"sequence": 6, "eventNanos": 1_060_000_000, "observedNanos": 1_065_000_000, "kind": "keyDown", "keyCode": 13, "origin": "physical"},
        {"sequence": 7, "eventNanos": 1_090_000_000, "observedNanos": 1_095_000_000, "kind": "keyUp", "keyCode": 13, "origin": "physical"},
    ])
    other, _ = _clone_source({"directory": str(source)}, source.parent)
    original = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir() if path.is_file()}
    ranges = (RecordingRange(1_002_000_000, 1_022_000_000), RecordingRange(1_075_000_000, 1_100_000_000))
    destination = tmp_path / str(uuid.uuid4())
    manifest = build_dataset(destination, recording_root=source.parent,
        selections=[RecordingSelection(item.stem, ranges=ranges) for item in (source, other)],
        config=_configuration(), vocabulary=_vocabulary(), pointer_mode="absolute")
    assert manifest["schemaVersion"] == 2 and len(manifest["sources"]) == 2
    assert {item["split"] for item in manifest["sources"]} == {"train", "validation"}
    with DatasetReader(destination, recording_root=source.parent) as reader:
        for split in ("train", "validation"):
            episodes = sorted(reader.episodes(split), key=lambda item: item["selection_index"])
            assert len(episodes) == 2 and len({item["recording_id"] for item in episodes}) == 1
            first, second = [list(reader.samples(item["id"])) for item in episodes]
            assert first[0].observation.reset.item() and second[0].observation.reset.item()
            assert second[0].observation.controls[0, 0, 13].item() == 1  # Gap press is prior state, not a label.
            np.testing.assert_array_equal(np.asarray(second[0].observation.controls)[0, 0, 170:174], np.zeros(4))
            assert not any(command["operation"] == "keyDown" for row in second for command in row.commands)
            assert second[0].commands == ({"operation": "keyUp", "keyCode": 13, "offsetMs": 15},)
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir() if path.is_file()} == original


def test_adjacent_ranges_remain_separate_recurrent_episodes(tmp_path, native_recording):
    source = Path(native_recording["directory"])
    ranges = (RecordingRange(1_002_000_000, 1_022_000_000), RecordingRange(1_022_000_000, 1_042_000_000))
    destination, _ = _build(tmp_path, native_recording, selection=RecordingSelection(source.stem, ranges=ranges))
    with DatasetReader(destination, recording_root=source.parent) as reader:
        episodes = reader.episodes()
        assert {item["selection_index"] for item in episodes} == {0, 1}
        assert all(next(reader.samples(item["id"])).observation.reset.item() for item in episodes)


@pytest.mark.parametrize("ranges", [(), (RecordingRange(0, 1),),
    (RecordingRange(1_002_000_000, 1_042_000_000), RecordingRange(1_022_000_000, 1_062_000_000)),
    (RecordingRange(1_075_000_000, 1_100_000_000), RecordingRange(1_002_000_000, 1_022_000_000))])
def test_invalid_multi_ranges_never_publish_a_partial_dataset(tmp_path, native_recording, ranges):
    source = Path(native_recording["directory"])
    with pytest.raises(RecordingError, match="range|interval|coverage"):
        _build(tmp_path, native_recording, selection=RecordingSelection(source.stem, ranges=ranges))
    assert not list(tmp_path.iterdir())


def test_range_index_cannot_move_an_episode_to_another_selected_interval(tmp_path, native_recording):
    source = Path(native_recording["directory"])
    ranges = (RecordingRange(1_002_000_000, 1_022_000_000), RecordingRange(1_075_000_000, 1_100_000_000))
    destination, _ = _build(tmp_path, native_recording, selection=RecordingSelection(source.stem, ranges=ranges))
    with sqlite3.connect(destination / "index.sqlite") as database:
        database.execute("UPDATE episodes SET selection_index=0 WHERE selection_index=1")
    _rehash_index(destination)
    with pytest.raises(RecordingError, match="range"):
        DatasetReader(destination, recording_root=source.parent)


def test_schema_one_dataset_checkpoint_resumes_the_exact_optimizer_and_sampler(tmp_path, native_recording):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from astra.checkpoints import save_checkpoint, load_checkpoint
    from astra.learning.behavioral import BehaviorConfig, BehaviorTrainer
    from astra.model.policy import AgentPolicy
    destination, _ = _build(tmp_path, native_recording)
    _write_legacy_dataset_format(destination)
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in destination.iterdir()}
    with DatasetReader(destination, recording_root=Path(native_recording["directory"]).parent) as reader:
        assert reader.manifest["schemaVersion"] == 1
        mx.random.seed(731)
        policy = AgentPolicy(reader.config, reader.vocabulary)
        training = BehaviorConfig(epochs=1, lanes=1, sequence_length=1, accumulation_chunks=2)
        trainer = BehaviorTrainer(policy, training, dataset_id=destination.name)
        with pytest.raises(InterruptedError):
            trainer.train_epoch(reader, cancelled=lambda: trainer.pending_chunks == 1)
        checkpoint_path = tmp_path / str(uuid.uuid4())
        save_checkpoint(checkpoint_path, policy, kind="behavioral", step=trainer.updates, training_state=trainer.state)
        loaded = load_checkpoint(checkpoint_path, include_training=True)
        resumed = BehaviorTrainer(loaded.policy, training, dataset_id=destination.name, restored_state=loaded.training_state)
        trainer.train_epoch(reader); resumed.train_epoch(reader)
        assert trainer.decisions == resumed.decisions and trainer.updates == resumed.updates
        expected = dict(tree_flatten(policy.parameters()))
        for name, value in tree_flatten(resumed.policy.parameters()):
            np.testing.assert_allclose(np.asarray(value), np.asarray(expected[name]), atol=2e-6, rtol=2e-5, err_msg=name)
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in destination.iterdir()} == before


def test_native_recording_dataset_reader_and_batch_preserve_causality_and_geometry(tmp_path, native_recording):
    destination, manifest = _build(tmp_path, native_recording, config=_configuration(lead=20))
    source_root = Path(native_recording["directory"]).parent
    assert manifest["steps"] == 3
    with DatasetReader(destination, recording_root=source_root) as reader:
        episodes = reader.episodes()
        assert len(episodes) == 2  # Native fixture changes from 4×4 to 64×64.
        sequences = [list(reader.samples(episode["id"])) for episode in episodes]
        earlier = next(samples for samples in sequences if samples[0].surfaces[0]["pixelWidth"] == 4)
        later = next(samples for samples in sequences if samples[0].surfaces[0]["pixelWidth"] == 64)
        assert len(earlier) == 2 and len(later) == 1
        assert bool(earlier[0].observation.reset.item()) and not bool(earlier[1].observation.reset.item())
        assert bool(later[0].observation.reset.item())
        # First action target begins 20 ms after the observation. Its pointer is
        # already moved in source time, while the observation still sees the seed.
        assert earlier[0].commands[0] == {"operation": "keyUp", "offsetMs": 3, "keyCode": 13}
        assert any(command["operation"] == "scroll" for command in earlier[0].commands)
        np.testing.assert_allclose(np.asarray(earlier[0].observation.controls)[0, 0, 168:170], [0.625, 0.625])
        assert earlier[0].observation.controls[0, 0, 13].item() == 0
        assert earlier[1].observation.controls[0, 0, 13].item() == 1
        np.testing.assert_allclose(np.asarray(earlier[1].observation.controls)[0, 0, 168:170], [0.625, 0.625])
        observation, packets = training_batch([earlier, [later[0], None]], reader.config, reader.vocabulary)
        assert observation.shape == (2, 2)
        assert packets.operation.shape == (4, reader.config.packet_capacity + 1)
        assert observation.surfaces[0].detail_image.shape == (2, 2, 64, 64, 3)
        np.testing.assert_array_equal(np.asarray(observation.valid), [[True, True], [True, False]])
        assert not np.asarray(packets.operation[3]).any()
    with pytest.raises(RecordingError, match="closed"):
        list(reader.samples(episodes[0]["id"]))


def test_selection_bounds_and_first_window_transients_are_not_silently_changed(tmp_path, native_recording):
    identifier = str(uuid.UUID(Path(native_recording["directory"]).stem))
    with pytest.raises(RecordingError, match="complete causal"):
        _build(tmp_path, native_recording, selection=RecordingSelection(identifier, end_nanos=0))
    with pytest.raises(RecordingError, match="forward interval"):
        _build(tmp_path, native_recording, selection=RecordingSelection(identifier, 20, 10))
    destination, _ = _build(tmp_path, native_recording, selection=RecordingSelection(identifier, start_nanos=1_075_000_000))
    with DatasetReader(destination, recording_root=Path(native_recording["directory"]).parent) as reader:
        sample = next(reader.samples(reader.episodes()[0]["id"]))
        # Pointer and scroll arrived at 1030/1042 ms, outside the selected first
        # observation's preceding 20 ms interval. Held pointer state remains.
        np.testing.assert_array_equal(np.asarray(sample.observation.controls)[0, 0, 170:174], np.zeros(4))
        np.testing.assert_allclose(np.asarray(sample.observation.controls)[0, 0, 168:170], [0.9375, 0.75])
    assert not list(tmp_path.glob(".dataset-*"))


def test_all_actions_must_fit_before_any_revision_is_published(tmp_path, native_recording):
    source, manifest = _clone_source(native_recording, tmp_path / "sources")
    database = sqlite3.connect(source / "index.sqlite")
    for index in range(20):
        time = 1_062_000_000 + index * 500_000
        event = {"sequence": 6 + index, "eventNanos": time, "observedNanos": time + 100,
                 "origin": "physical", "kind": "keyRepeat", "keyCode": 13, "isDown": True}
        database.execute("INSERT INTO events VALUES(?,?,?,?)", (event["sequence"], event["observedNanos"], event["eventNanos"], json.dumps(event).encode()))
    database.commit(); database.close()
    manifest["eventCount"] += 20
    (source / "manifest.json").write_text(json.dumps(manifest))
    destination = tmp_path / "revisions" / str(uuid.uuid4())
    with pytest.raises(ActionEncodingError, match="no partial revision"):
        build_dataset(destination, recording_root=source.parent, selections=[RecordingSelection(source.stem)],
                      config=_configuration(), vocabulary=_vocabulary(), pointer_mode="absolute")
    assert not destination.exists()
    assert not list(destination.parent.glob(".dataset-*"))
    # Semantic commands may fit capacity yet fail dense coordinate precision.
    with pytest.raises(ActionEncodingError, match="quantization"):
        _build(tmp_path, native_recording, config=replace(_configuration(), coordinate_bins=2))


def test_session_splits_and_reader_descriptors_remain_bounded(tmp_path, native_recording, monkeypatch):
    root = tmp_path / "sources"
    identifiers = [_clone_source(native_recording, root)[0].stem for _ in range(7)]
    expected, warnings = session_splits(identifiers, seed=319)
    assert not warnings
    assert set(expected.values()) == {"train", "validation", "test"}
    assert session_splits(list(reversed(identifiers)), seed=319)[0] == expected
    destination = tmp_path / str(uuid.uuid4())
    build_dataset(destination, recording_root=root, selections=[RecordingSelection(value) for value in identifiers],
                  config=_configuration(), vocabulary=_vocabulary(), pointer_mode="absolute", split_seed=319)
    active, peak = set(), []
    original = datasets.RecordingReader
    class TrackedReader(original):
        def __init__(self, directory):
            super().__init__(directory)
            active.add(id(self)); peak.append(len(active))
        def close(self):
            active.discard(id(self))
            super().close()
    monkeypatch.setattr(datasets, "RecordingReader", TrackedReader)
    with DatasetReader(destination, recording_root=root) as reader:
        consumed = set()
        episode_by_source = {}
        for split in ("test", "train", "validation"):
            for episode in reader.episodes(split):
                assert expected[episode["recording_id"]] == split
                assert list(reader.samples(episode["id"]))
                consumed.add(episode["recording_id"])
                episode_by_source[episode["recording_id"]] = episode
        assert consumed == set(identifiers)
        # An active sample iterator must survive its source being evicted while
        # another training lane loads a different session.
        iterators = [reader.samples(episode["id"]) for episode in episode_by_source.values()]
        try:
            for iterator in iterators:
                assert next(iterator).step == 0
            assert next(iterators[0]).step == 1
        finally:
            for iterator in iterators:
                iterator.close()
        assert max(peak) <= datasets.MAXIMUM_SOURCE_READERS
    assert not active


@pytest.mark.parametrize("corruption", ["count", "gap", "split", "frame_future", "context", "command"])
def test_malformed_rehashed_revisions_are_rejected_before_learning(tmp_path, native_recording, corruption):
    destination, manifest = _build(tmp_path, native_recording)
    root = Path(native_recording["directory"]).parent
    if corruption == "count":
        manifest["steps"] += 1
        (destination / "manifest.json").write_text(json.dumps(manifest))
    else:
        database = sqlite3.connect(destination / "index.sqlite")
        if corruption == "gap":
            database.execute("UPDATE steps SET cutoff=cutoff+1 WHERE step=1")
        elif corruption == "split":
            database.execute("UPDATE episodes SET split='validation'")
        elif corruption == "frame_future":
            database.execute("UPDATE steps SET cutoff=cutoff-10000000")
        elif corruption == "context":
            database.execute("UPDATE steps SET context=?", (b"[7]",))
        else:
            database.execute("UPDATE steps SET commands=?", (b'[{"operation":"scroll","offsetMs":0,"dx":999999,"dy":0}]',))
        database.commit(); database.close(); _rehash_index(destination)
    with pytest.raises((RecordingError, ActionEncodingError)):
        with DatasetReader(destination, recording_root=root) as reader:
            for episode in reader.episodes():
                list(reader.samples(episode["id"]))


def test_multiple_surface_roles_are_never_silently_reduced_to_the_latest_frame(tmp_path, native_recording):
    source, _ = _clone_source(native_recording, tmp_path / "sources")
    # Distinct surface metadata is detected before any potentially mismatching
    # block is decoded. Aligned multi-surface dataset support remains explicit.
    database = sqlite3.connect(source / "index.sqlite")
    row = database.execute("SELECT id,block FROM frames ORDER BY observed DESC LIMIT 1").fetchone()
    block = json.loads(row[1]); block["metadata"]["surface"]["id"] = "surface:1"
    database.execute("UPDATE frames SET block=? WHERE id=?", (json.dumps(block).encode(), row[0]))
    database.commit(); database.close()
    destination = tmp_path / str(uuid.uuid4())
    with pytest.raises(RecordingError, match="aligned multi-surface"):
        build_dataset(destination, recording_root=source.parent, selections=[RecordingSelection(source.stem)],
                      config=_configuration(), vocabulary=_vocabulary(), pointer_mode="absolute")
    assert not destination.exists()


def test_cancellation_removes_only_unpublished_derived_work(tmp_path, native_recording):
    source = Path(native_recording["directory"])
    original = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir() if path.is_file()}
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 3
    destination = tmp_path / str(uuid.uuid4())
    with pytest.raises(InterruptedError, match="cancelled"):
        build_dataset(destination, recording_root=source.parent, selections=[RecordingSelection(source.stem)],
                      config=_configuration(), vocabulary=_vocabulary(), pointer_mode="absolute", cancelled=cancelled)
    assert not destination.exists() and not list(tmp_path.glob(".dataset-*"))
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir() if path.is_file()} == original


def test_manifest_size_is_checked_before_reading_its_contents(tmp_path, native_recording, monkeypatch):
    destination, _ = _build(tmp_path, native_recording)
    path = destination / "manifest.json"
    with path.open("wb") as stream:
        stream.truncate(8 * 1024**2 + 1)
    original = Path.read_bytes
    def bounded_read(candidate):
        if candidate == path:
            raise AssertionError("Oversized manifest was read before resource admission")
        return original(candidate)
    monkeypatch.setattr(Path, "read_bytes", bounded_read)
    with pytest.raises(RecordingError, match="size"):
        DatasetReader(destination, recording_root=Path(native_recording["directory"]).parent)


def test_raw_interval_budget_is_enforced_before_retaining_an_unbounded_batch(tmp_path, native_recording, monkeypatch):
    monkeypatch.setattr(datasets, "MAXIMUM_INTERVAL_EVENTS", 2)
    with pytest.raises(ActionEncodingError, match="raw-input budget"):
        _build(tmp_path, native_recording)
    assert not list(tmp_path.iterdir())


def _append_events_to_fixture(source, manifest, events):
    with sqlite3.connect(source / "index.sqlite") as database:
        for item in events:
            database.execute("INSERT INTO events VALUES(?,?,?,?)", (item["sequence"], item["observedNanos"], item["eventNanos"], json.dumps(item).encode()))
    manifest["eventCount"] += len(events)
    (source / "manifest.json").write_text(json.dumps(manifest))


def test_native_dataset_stream_carries_fractional_boundary_events_without_future_history(tmp_path, native_recording):
    source, source_manifest = _clone_source(native_recording, tmp_path / "sources")
    samples = [
        {"sequence": 6, "eventNanos": 1_061_800_000, "observedNanos": 1_070_000_000, "kind": "keyDown", "isDown": True},
        {"sequence": 7, "eventNanos": 1_062_200_000, "observedNanos": 1_071_000_000, "kind": "keyUp", "isDown": False},
        {"sequence": 8, "eventNanos": 1_081_510_000, "observedNanos": 1_090_000_000, "kind": "keyRepeat", "isDown": True},
    ]
    events = [{**item, "keyCode": 13, "origin": "physical"} for item in samples]
    _append_events_to_fixture(source, source_manifest, events)
    original = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir() if path.is_file()}
    destination = tmp_path / str(uuid.uuid4())
    manifest = build_dataset(destination, recording_root=source.parent, selections=[RecordingSelection(source.stem)],
                             config=_configuration(lead=20), vocabulary=_vocabulary(), pointer_mode="absolute")
    assert manifest["canonicalizerVersion"] == 2
    with DatasetReader(destination, recording_root=source.parent) as reader:
        rows = [item for episode in reader.episodes() for item in reader.samples(episode["id"])]
        later = next(item for item in rows if item.surfaces[0]["pixelWidth"] == 64)
        # Both transitions occupy the next execution packet at offset zero,
        # in original source order; they are unavailable at its observation t.
        assert [item for item in later.commands if item["operation"].startswith("key")] == [
            {"operation": "keyDown", "offsetMs": 0, "keyCode": 13},
            {"operation": "keyUp", "offsetMs": 0, "keyCode": 13}]
        assert later.observation.controls[0, 0, 13].item() == 0
        assert sum(item["operation"] == "keyDown" for row in rows for item in row.commands) == 1
        assert sum(item["operation"] == "keyRepeat" for row in rows for item in row.commands) == 0
    partition = manifest["sources"][0]["labelPartitions"][0]
    assert partition["gridOriginNanos"] == 1_022_000_000
    assert partition["executionEndNanos"] == 1_082_000_000
    assert partition["assignedPhysicalEvents"] == partition["assignedDiscreteEvents"] == 4
    assert partition["excludedPhysicalEvents"] == {"before": 3, "after": 1}
    assert partition["excludedDiscreteEvents"] == {"before": 2, "after": 1}
    assert partition["exclusionExamples"][-1] == {"sequence": 8, "eventNanos": 1_081_510_000,
                                                    "quantizedNanos": 1_082_000_000, "side": "after"}
    assert any("Excluded 4 selected physical events" in warning for warning in manifest["warnings"])
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir() if path.is_file()} == original


def test_selection_does_not_pull_a_rounded_event_from_outside_raw_selection(tmp_path, native_recording):
    source, source_manifest = _clone_source(native_recording, tmp_path / "sources")
    events = [{"sequence": index + 6, "eventNanos": time, "observedNanos": time + 100_000,
               "kind": kind, "keyCode": 13, "origin": "physical"}
              for index, (time, kind) in enumerate(((1_061_800_000, "keyDown"), (1_062_200_000, "keyUp"), (1_081_800_000, "keyRepeat")))]
    _append_events_to_fixture(source, source_manifest, events)
    destination = tmp_path / str(uuid.uuid4())
    selection = RecordingSelection(source.stem, 1_062_000_000, 1_082_000_000)
    manifest = build_dataset(destination, recording_root=source.parent, selections=[selection],
                             config=_configuration(), vocabulary=_vocabulary(), pointer_mode="absolute")
    with DatasetReader(destination, recording_root=source.parent) as reader:
        row = next(reader.samples(reader.episodes()[0]["id"]))
        assert row.commands == ({"operation": "keyUp", "offsetMs": 0, "keyCode": 13},)
        assert row.observation.controls[0, 0, 13].item() == 1  # Real prior state, not an expert press label.
    partition = manifest["sources"][0]["labelPartitions"][0]
    assert partition["assignedDiscreteEvents"] == 1
    assert partition["excludedPhysicalEvents"] == {"before": 0, "after": 1}
    assert partition["exclusionExamples"] == [{"sequence": 8, "eventNanos": 1_081_800_000,
                                              "quantizedNanos": 1_082_000_000, "side": "after"}]


@pytest.mark.parametrize("corruption", ["version", "origin", "counts", "selection"])
def test_dataset_partition_report_is_versioned_and_validated(tmp_path, native_recording, corruption):
    destination, manifest = _build(tmp_path, native_recording)
    partition = manifest["sources"][0]["labelPartitions"][0]
    if corruption == "version":
        manifest["canonicalizerVersion"] = 1
    elif corruption == "origin":
        partition["gridOriginNanos"] += 1
    elif corruption == "counts":
        partition["assignedDiscreteEvents"] = partition["assignedPhysicalEvents"] + 1
    else:
        # Keep an internally coherent report while changing its source range.
        for name in ("gridOriginNanos", "executionEndNanos", "sourceStartNanos", "sourceEndNanos"):
            partition[name] += 1
    (destination / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RecordingError, match="version|partition|count|range"):
        DatasetReader(destination, recording_root=Path(native_recording["directory"]).parent)


def test_discontinuity_in_the_final_rounding_tail_still_rejects_the_revision(tmp_path, native_recording):
    source, manifest = _clone_source(native_recording, tmp_path / "sources")
    _append_events_to_fixture(source, manifest, [{"sequence": 6, "eventNanos": 1_081_800_000,
        "observedNanos": 1_090_000_000, "kind": "gap", "origin": "boundary"}])
    destination = tmp_path / str(uuid.uuid4())
    with pytest.raises(ActionEncodingError, match="discontinuity"):
        build_dataset(destination, recording_root=source.parent, selections=[RecordingSelection(source.stem)],
                      config=_configuration(), vocabulary=_vocabulary(), pointer_mode="absolute")
    assert not destination.exists() and not list(tmp_path.glob(".dataset-*"))
