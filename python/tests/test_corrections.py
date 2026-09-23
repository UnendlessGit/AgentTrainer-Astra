"""Correction labels start at the explicit gate; agent lead-up stays review-only."""
import hashlib
import json
import struct
import uuid
import numpy as np
import pytest

from astra.recordings import RecordingReader, RecordingError
from astra.data.actions import ActionEncodingError
from astra.data.datasets import DatasetReader, RecordingSelection, RecordingRange
from test_multi_surface_data import _frame, _recording, _dataset, MS


def add_correction(source, *, supervised_ms=40):
    folder = source / 'correction'; folder.mkdir()
    metadata = _frame('front', 10, 10)
    pixels = bytes([23]) * metadata['byteCount']
    raw = json.dumps(metadata).encode()
    header = b'ASTRAF01' + struct.pack('<IQ', len(raw), len(pixels))
    payload = header + raw + pixels
    digest = hashlib.sha256(payload).hexdigest()
    (folder / 'frames.astraframes').write_bytes(payload + bytes.fromhex(digest))
    cutoff = 10 * MS
    actor_input = {'observationID': str(uuid.uuid4()), 'episodeID': str(uuid.uuid4()), 'previousStateID': str(uuid.uuid4()),
        'cutoffNanos': cutoff, 'geometryRevision': 1, 'controlState': {'keys': [], 'buttons': [], 'modifiers': 0,
            'pointer': {'x': 10, 'y': 10}, 'observedNanos': cutoff, 'revision': 0, 'valid': True},
        'executedEvents': [], 'intervalCovered': True, 'contextIDs': []}
    prelude = {'schemaVersion': 1, 'sourceRunID': str(uuid.uuid4()), 'sourceCheckpointID': str(uuid.uuid4()),
        'sourcePolicySignature': 'a' * 64, 'contextIDs': [], 'requestedAtNanos': 20 * MS, 'controlJoinedAtNanos': 20 * MS,
        'supervisionStartNanos': supervised_ms * MS, 'maximumDurationNanos': 2_000_000_000, 'maximumBytes': 268435456,
        'continuityProven': False, 'observations': [{'actorInput': actor_input, 'frames': [{'shard': 'correction/frames.astraframes',
        'block': {'offset': 0, 'length': len(payload) + 32, 'metadata': metadata, 'digest': digest}}]}]}
    write_prelude(source, prelude)
    return prelude


def write_prelude(source, prelude):
    data = json.dumps(prelude).encode()
    (source / 'correction/prelude.json').write_bytes(data)
    manifest = json.loads((source / 'manifest.json').read_bytes())
    manifest['correction'] = {'schemaVersion': 1, 'path': 'correction/prelude.json', 'sha256': hashlib.sha256(data).hexdigest()}
    (source / 'manifest.json').write_text(json.dumps(manifest))


def test_correction_gate_applies_to_dataset_and_prefix_is_review_only(tmp_path):
    events = [dict(sequence=0, eventNanos=0, observedNanos=0, origin='reconciliation', kind='pointer', x=10, y=10),
              dict(sequence=1, eventNanos=30*MS, observedNanos=30*MS, origin='physical', kind='buttonDown', button=0, x=10, y=10),
              dict(sequence=2, eventNanos=50*MS, observedNanos=50*MS, origin='physical', kind='buttonUp', button=0, x=10, y=10)]
    source = _recording(tmp_path / 'sources', [_frame('front', time, time) for time in (0, 20, 40, 60, 80)], roles=['front'], events=events)
    prelude = add_correction(source)
    revision = tmp_path / str(uuid.uuid4())
    manifest = _dataset(source, revision)
    assert manifest['steps'] == 3
    assert manifest['sources'][0]['correction']['sourceCheckpointID'] == prelude['sourceCheckpointID']
    assert manifest['sources'][0]['correction']['preRollUse'] == 'review_only'
    with RecordingReader(source) as recording:
        assert recording.supervision_start_nanos == 40 * MS
        assert np.all(recording.correction_pixels(0, 0) == 23)
    with DatasetReader(revision, recording_root=source.parent) as dataset:
        samples = list(dataset.samples(dataset.episodes()[0]['id']))
        assert len(samples) == 3 and samples[0].observation.reset.item()
        assert not any(command['operation'] == 'buttonDown' for sample in samples for command in sample.commands)
        assert any(command['operation'] == 'buttonUp' for sample in samples for command in sample.commands)
    with pytest.raises(RecordingError, match='before expert supervision'):
        _dataset(source, tmp_path / str(uuid.uuid4()), RecordingSelection(source.stem, ranges=(RecordingRange(0, 100 * MS),)))


def test_correction_integrity_and_unproven_continuity_cannot_be_bypassed(tmp_path):
    source = _recording(tmp_path / 'sources', [_frame('front', time, time) for time in (0, 20, 40, 60, 80)], roles=['front'])
    prelude = add_correction(source)
    prelude['continuityProven'] = True; write_prelude(source, prelude)
    with pytest.raises(RecordingError, match='continuous handoff'):
        RecordingReader(source)
    prelude['continuityProven'] = False; write_prelude(source, prelude)
    path = source / 'correction/frames.astraframes'
    data = bytearray(path.read_bytes()); data[-40] ^= 1; path.write_bytes(data)
    with pytest.raises(RecordingError, match='checksum'):
        RecordingReader(source)


def test_agent_activity_cannot_become_empty_expert_targets(tmp_path):
    events = [dict(sequence=0, eventNanos=0, observedNanos=0, origin='reconciliation', kind='pointer', x=10, y=10),
              dict(sequence=1, eventNanos=50*MS, observedNanos=50*MS, origin='agent', kind='keyDown', keyCode=12)]
    source = _recording(tmp_path / 'sources', [_frame('front', time, time) for time in (0, 20, 40, 60, 80)], roles=['front'], events=events)
    add_correction(source)
    with pytest.raises(ActionEncodingError, match='cannot be treated as human supervision'):
        _dataset(source, tmp_path / str(uuid.uuid4()))
