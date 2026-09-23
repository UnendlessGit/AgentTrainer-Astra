"""Causal multi-source recording -> real visual policy, without screen capture."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
import sqlite3
import struct
import uuid

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

from astra.data.actions import ActionEncodingError, canonicalize
from astra.data.batching import training_batch
from astra.data.datasets import DatasetReader, RecordingSelection, RecordingRange, build_dataset
from astra.model.config import ModelConfig
from astra.model.actions import ActionVocabulary, Operation
from astra.model.policy import AgentPolicy
from astra.recordings import RecordingError

MS = 1_000_000
CONFIG = replace(ModelConfig.test_small(), period_ms=20, lead_ms=0)
VOCABULARY = ActionVocabulary(mouse_buttons=(0,), absolute_pointer=True)


def _frame(role, source_ms, available_ms, revision=1, width=32, x=0):
    return {"id": str(uuid.uuid4()), "eventNanos": source_ms * MS, "observedNanos": available_ms * MS,
            "surface": {"id": role, "pixelWidth": width, "pixelHeight": 32, "geometryRevision": revision,
                        "nativeWindowID": 101 if role == "front" else 102,
                        "globalBounds": {"x": x, "y": 0, "width": width, "height": 32},
                        "contentBounds": {"x": 0, "y": 0, "width": width, "height": 32}},
            "byteCount": width * 32 * 4, "codec": "raw", "pixelFormat": "bgra8-srgb"}


def _recording(root, frames, *, roles=None, stopped_ms=100, events=None, proofs=()):
    identifier = str(uuid.uuid4()); folder = root / (identifier + ".astrarecord")
    folder.mkdir(parents=True); (folder / ".writer.lock").touch()
    if events is None:
        events = [{"sequence": 0, "eventNanos": 0, "observedNanos": 0, "origin": "reconciliation",
                   "kind": "pointer", "x": 10, "y": 10, "surfaceID": "front"}]
    with sqlite3.connect(folder / "index.sqlite") as database, (folder / "frames-00000.astraframes").open("wb") as archive:
        database.executescript("CREATE TABLE frames(id TEXT PRIMARY KEY, observed INTEGER, source_time INTEGER, shard TEXT, offset INTEGER, length INTEGER, block BLOB); CREATE TABLE events(sequence INTEGER PRIMARY KEY, observed INTEGER, source_time INTEGER, event BLOB); CREATE TABLE coverage(observed INTEGER, surface_id TEXT, frame_id TEXT, proof BLOB);")
        for index, metadata in enumerate(frames):
            encoded = json.dumps(metadata).encode()
            pixels = np.zeros((metadata["surface"]["pixelHeight"], metadata["surface"]["pixelWidth"], 4), np.uint8)
            pixels[..., 0] = 30 + index * 20; pixels[..., 3] = 255
            payload = pixels.tobytes(); prefix = b"ASTRAF01" + struct.pack("<IQ", len(encoded), len(payload))
            raw = prefix + encoded + payload; digest = hashlib.sha256(raw).digest()
            offset = archive.tell(); archive.write(raw + digest)
            block = {"offset": offset, "length": len(raw) + 32, "metadata": metadata, "digest": digest.hex()}
            database.execute("INSERT INTO frames VALUES(?,?,?,?,?,?,?)", (metadata["id"], metadata["observedNanos"], metadata["eventNanos"], "frames-00000.astraframes", offset, len(raw) + 32, json.dumps(block).encode()))
        for event in events:
            database.execute("INSERT INTO events VALUES(?,?,?,?)", (event["sequence"], event["observedNanos"], event["eventNanos"], json.dumps(event).encode()))
        for proof in proofs:
            database.execute("INSERT INTO coverage VALUES(?,?,?,?)", (proof["verifiedAtNanos"], proof["surface"]["id"], proof["frameID"], json.dumps(proof).encode()))
    manifest = {"schemaVersion": 1, "id": identifier, "status": "complete", "frameCount": len(frames),
                "eventCount": len(events), "storedBytes": (folder / "frames-00000.astraframes").stat().st_size,
                "firstObservedNanos": min(frame["observedNanos"] for frame in frames), "stoppedNanos": stopped_ms * MS}
    if roles is not None: manifest["surfaceIDs"] = roles
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


def _dataset(source, destination, selection=None):
    return build_dataset(destination, recording_root=source.parent, selections=[selection or RecordingSelection(source.stem)],
                         config=CONFIG, vocabulary=VOCABULARY, pointer_mode="absolute")


def test_ordered_latest_causal_frames_geometry_and_mixed_batches_train(tmp_path):
    rear = _frame("rear", 0, 0, revision=7, width=64, x=-32)
    front = _frame("front", 18, 20, revision=90)
    moved = _frame("rear", 55, 65, revision=8, width=64, x=-30)
    seed = {"sequence": 0, "eventNanos": 0, "observedNanos": 0, "origin": "reconciliation", "kind": "pointer", "x": 10, "y": 10, "surfaceID": "front"}
    click = {"sequence": 1, "eventNanos": 25 * MS, "observedNanos": 26 * MS, "origin": "physical", "kind": "buttonDown", "button": 0, "x": 11, "y": 10, "surfaceID": "front"}
    release = {"sequence": 2, "eventNanos": 45 * MS, "observedNanos": 46 * MS, "origin": "physical", "kind": "buttonUp", "button": 0, "x": -20, "y": 10, "surfaceID": "rear"}
    source = _recording(tmp_path / "sources", [rear, front, moved], roles=["front", "rear"], events=[seed, click, release])
    original = {path.name: path.read_bytes() for path in source.iterdir()}
    revision = tmp_path / str(uuid.uuid4()); manifest = _dataset(source, revision)
    assert manifest["schemaVersion"] == 3 and manifest["steps"] == 4
    assert any("all required surfaces" in warning for warning in manifest["warnings"])
    with sqlite3.connect(revision / "index.sqlite") as db:
        rows = [(cutoff, json.loads(raw)) for cutoff, raw in db.execute("SELECT cutoff,frame FROM steps ORDER BY cutoff")]
    assert [[item["block"]["metadata"]["id"] for item in row["frames"]] for _, row in rows] == [[front["id"], rear["id"]]] * 3 + [[front["id"], moved["id"]]]
    assert [row["geometryRevision"] for _, row in rows] == [0, 0, 0, 1]
    assert rows[0][1]["frames"][0]["block"]["metadata"] == front
    assert rows[0][1]["frames"][1]["block"]["metadata"] == rear
    with DatasetReader(revision, recording_root=source.parent) as reader:
        episodes = reader.episodes(); assert sorted(episode["steps"] for episode in episodes) == [1, 3]
        multi = next(list(reader.samples(episode["id"])) for episode in episodes if episode["steps"] == 3)
    single = _recording(source.parent, [_frame("front", 0, 0)])
    single_revision = tmp_path / str(uuid.uuid4()); _dataset(single, single_revision)
    with DatasetReader(single_revision, recording_root=source.parent) as reader:
        mono = list(reader.samples(reader.episodes()[0]["id"]))
    observed, labels = training_batch([multi[:2], mono[:2]], CONFIG, VOCABULARY)
    assert len(observed.surfaces) == 2
    np.testing.assert_array_equal(np.asarray(observed.surfaces[1].available), [[True, True], [False, False]])
    assert set(np.asarray(labels.surface)[np.asarray(labels.operation) == int(Operation.ABSOLUTE)]) == {0, 1}
    assert any(command.get("surfaceID") == "rear" and command["x"] == .1875 for command in multi[1].commands)
    policy = AgentPolicy(CONFIG, VOCABULARY)
    loss, gradients = nn.value_and_grad(policy, lambda model: -mx.mean(model.score(model(observed), labels).log_probability))(policy)
    mx.eval(loss, gradients)
    assert np.isfinite(float(loss)) and all(np.isfinite(np.asarray(value)).all() for _, value in tree_flatten(gradients))
    assert np.any(np.asarray(gradients["actions"]["surface_query"]["weight"]) != 0)
    assert {path.name: path.read_bytes() for path in source.iterdir()} == original


def test_missing_roles_and_explicit_early_ranges_are_never_silently_trimmed(tmp_path):
    source = _recording(tmp_path / "sources", [_frame("front", 0, 0)], roles=["front", "rear"])
    with pytest.raises(RecordingError, match="never produced"):
        _dataset(source, tmp_path / str(uuid.uuid4()))
    source = _recording(source.parent, [_frame("front", 0, 0), _frame("rear", 18, 20)], roles=["front", "rear"])
    with pytest.raises(RecordingError, match="every required surface"):
        _dataset(source, tmp_path / str(uuid.uuid4()), RecordingSelection(source.stem, ranges=(RecordingRange(0, 100 * MS),)))


def test_static_sources_require_actual_causal_unchanged_proofs(tmp_path):
    frame = _frame("front", 0, 0)
    def proof(at):
        return {"streamID": str(uuid.uuid4()), "frameID": frame["id"], "surface": frame["surface"], "eventNanos": 0,
                "observedNanos": 0, "throughNanos": (at - 10) * MS, "verifiedAtNanos": at * MS, "kind": "unchanged"}
    proofs = [proof(200), proof(400)]
    source = _recording(tmp_path / "sources", [frame], roles=["front"], stopped_ms=600, proofs=proofs)
    revision = tmp_path / str(uuid.uuid4()); _dataset(source, revision)
    with DatasetReader(revision, recording_root=source.parent) as reader:
        samples = list(reader.samples(reader.episodes()[0]["id"]))
        assert len(samples) == 30
    for variant in ([], [proof(400)]):
        missing = _recording(source.parent, [frame], roles=["front"], stopped_ms=600, proofs=variant)
        with pytest.raises(RecordingError, match="stale"):
            _dataset(missing, tmp_path / str(uuid.uuid4()))
    with sqlite3.connect(source / "index.sqlite") as db: db.execute("DELETE FROM coverage")
    with DatasetReader(revision, recording_root=source.parent) as reader:
        with pytest.raises(RecordingError, match="evidence is missing"):
            list(reader.samples(reader.episodes()[0]["id"]))


@pytest.mark.parametrize("hint", [None, "missing", "outside"])
def test_ambiguous_or_invalid_pointer_hints_cannot_be_retargeted(hint):
    front, rear, outside = _frame("front", 0, 0)["surface"], _frame("rear", 0, 0)["surface"], _frame("outside", 0, 0, x=100)["surface"]
    event = {"sequence": 1, "eventNanos": 5 * MS, "observedNanos": 6 * MS, "origin": "physical", "kind": "buttonDown", "button": 0, "x": 10, "y": 10}
    if hint is not None: event["surfaceID"] = hint
    with pytest.raises(ActionEncodingError, match="unambiguous|hint"):
        canonicalize([event], start_nanos=0, config=CONFIG, vocabulary=VOCABULARY, surfaces=[front, rear, outside],
                     pointer_mode="absolute", initial_pointer=(10, 10), initial_surface_id="front")


def _ring(path, metadata, run_id):
    from astra.frame_ring import _metadata_fingerprint
    ring_id, lease_id = str(uuid.uuid4()), str(uuid.uuid4())
    capacity = metadata["byteCount"]; stride = ((128 + capacity + 63) // 64) * 64; size = 128 + stride
    header = struct.pack("<8sII16s16sIIQQQ", b"ASTRAR01", 1, 128, uuid.UUID(ring_id).bytes, uuid.UUID(run_id).bytes, 1, 128, capacity, stride, size).ljust(128, b"\0")
    slot = struct.pack("<IIII16s16sQQ32s32x", 2, 1, 0, 0, uuid.UUID(lease_id).bytes, uuid.UUID(metadata["id"]).bytes, 1, capacity, _metadata_fingerprint(metadata))
    pixels = bytes([32, 64, 128, 255]) * (capacity // 4)
    path.write_bytes((header + slot + pixels).ljust(size, b"\0")); os.chmod(path, 0o600)
    reference = {"version": 1, "runID": run_id, "ringID": ring_id, "slot": 0, "leaseID": lease_id, "sequence": 1,
                 "offset": 256, "size": capacity, "metadata": metadata}
    return {"path": str(path), "ringID": ring_id}, reference


def test_actor_maps_independent_sized_rings_and_joins_every_reader(tmp_path):
    from astra.checkpoints import save_checkpoint
    from astra.inference import InferenceSession
    checkpoint = tmp_path / str(uuid.uuid4()); save_checkpoint(checkpoint, AgentPolicy(CONFIG, VOCABULARY), kind="initial", step=0)
    run = str(uuid.uuid4()); episode = str(uuid.uuid4())
    primary, first = _ring(tmp_path / "one.ring", _frame("front", 18, 20, revision=90), run)
    secondary, second = _ring(tmp_path / "two.ring", _frame("rear", 0, 0, revision=7, width=64, x=-32), run)
    actor = InferenceSession()
    try:
        prepared = actor.prepare({"checkpointPath": str(checkpoint), "ring": primary, "additionalRings": [secondary]}, run_id=run)
        assert prepared["ringIDs"] == [primary["ringID"], secondary["ringID"]]
        readers = list(actor._readers.values())
        assert [reader.slot_capacity for reader in readers] == [4096, 8192]
        reset = actor.reset({"confirmed": True, "episodeID": episode, "contextIDs": []}, run_id=run)
        response = actor.step({"observationID": str(uuid.uuid4()), "episodeID": episode, "previousStateID": reset["stateID"],
            "cutoffNanos": 30 * MS, "geometryRevision": 123, "frames": [first, second], "contextIDs": [],
            "controlState": {"keys": [], "buttons": [], "modifiers": 0, "pointer": {"x": 10, "y": 10},
                             "observedNanos": 30 * MS, "revision": 1, "valid": True},
            "executedEvents": [], "intervalCovered": True}, run_id=run)
        assert response["packet"]["geometryRevision"] == 123
        assert {ack["ringID"] for ack in response["releasedFrames"]} == {primary["ringID"], secondary["ringID"]}
    finally: actor.close()
    assert all(reader._descriptor is None and reader._mapping is None for reader in readers)


def test_actor_prepare_failure_closes_previously_mapped_sources(tmp_path, monkeypatch):
    import astra.inference as inference
    from astra.checkpoints import save_checkpoint
    checkpoint = tmp_path / str(uuid.uuid4()); save_checkpoint(checkpoint, AgentPolicy(CONFIG, VOCABULARY), kind="initial", step=0)
    run = str(uuid.uuid4()); primary, _ = _ring(tmp_path / "one.ring", _frame("front", 0, 0), run)
    readers = []; original = inference.FrameRingReader
    def open_reader(*args, **kwargs):
        reader = original(*args, **kwargs); readers.append(reader); return reader
    monkeypatch.setattr(inference, "FrameRingReader", open_reader)
    actor = inference.InferenceSession()
    with pytest.raises(FileNotFoundError):
        actor.prepare({"checkpointPath": str(checkpoint), "ring": primary,
                       "additionalRings": [{"path": str(tmp_path / "missing"), "ringID": str(uuid.uuid4())}]}, run_id=run)
    assert len(readers) == 1 and readers[0]._descriptor is None and not actor._readers
    actor.close()


def test_archived_schema_two_rows_remain_readable_without_new_coverage_rules(tmp_path):
    source = _recording(tmp_path / "sources", [_frame("front", 0, 0)])
    revision = tmp_path / str(uuid.uuid4()); manifest = _dataset(source, revision)
    manifest["schemaVersion"] = 2; manifest.pop("maximumFrameAgeMs")
    for specification in manifest["sources"]: specification.pop("surfaceIDs")
    with sqlite3.connect(revision / "index.sqlite") as db:
        for episode, step, raw in list(db.execute("SELECT episode_id,step,frame FROM steps")):
            db.execute("UPDATE steps SET frame=? WHERE episode_id=? AND step=?", (json.dumps(json.loads(raw)["frames"][0]).encode(), episode, step))
    manifest["indexSHA256"] = hashlib.sha256((revision / "index.sqlite").read_bytes()).hexdigest()
    (revision / "manifest.json").write_text(json.dumps(manifest))
    with DatasetReader(revision, recording_root=source.parent) as reader:
        assert len(list(reader.samples(reader.episodes()[0]["id"]))) == 5


def test_ambiguous_synthetic_seed_does_not_invent_a_route_or_hide_a_real_event():
    surfaces = [_frame("front", 0, 0)["surface"], _frame("rear", 0, 0)["surface"]]
    event = {"sequence": 1, "eventNanos": 5 * MS, "observedNanos": 6 * MS, "origin": "physical", "kind": "pointer", "x": 11, "y": 10, "surfaceID": "front"}
    packet = canonicalize([event], start_nanos=0, config=CONFIG, vocabulary=VOCABULARY, surfaces=surfaces,
                          pointer_mode="absolute", initial_pointer=(10, 10))
    assert packet.commands[0]["offsetMs"] == 5
    assert all(command["surfaceID"] == "front" for command in packet.commands)
    event.update(eventNanos=0, observedNanos=0, x=10);event.pop("surfaceID")
    with pytest.raises(ActionEncodingError, match="unambiguous"):
        canonicalize([event], start_nanos=0, config=CONFIG, vocabulary=VOCABULARY, surfaces=surfaces,
                     pointer_mode="absolute", initial_pointer=(10, 10))
