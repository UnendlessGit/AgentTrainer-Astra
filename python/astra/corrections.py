"""Read-only provenance for explicitly requested expert correction recordings.

Current handoff joins the actor/control/capture owners. Its retained lead-up is
review evidence, not a recurrent training prefix: no continuity Boolean can
turn policy behavior into an expert label or invent coverage across the join.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import struct
import uuid

from .recordings import RecordingError, _integer, _json, validate_frame, validate_event, validate_frame_coverage

MAXIMUM_PRELUDE_BYTES = 16 * 1024**2
MAXIMUM_PIXELS = 256 * 1024**2
MAXIMUM_DURATION = 2_000_000_000
MAXIMUM_OBSERVATIONS = 256


def _digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise RecordingError('Correction provenance requires a SHA-256 digest')
    return value


def _uuid(value):
    if type(value) is not str:
        raise RecordingError('Correction provenance requires UUID identities')
    try:
        return uuid.UUID(value)
    except ValueError as error:
        raise RecordingError('Correction provenance requires UUID identities') from error


def _open(directory: Path, name: str, maximum: int):
    folder = directory / 'correction'
    if folder.is_symlink() or not folder.is_dir():
        raise RecordingError('Correction evidence must remain inside its recording package')
    descriptor = os.open(folder / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= maximum:
            raise RecordingError('Correction evidence is not a bounded regular file')
        return descriptor, info.st_size
    except BaseException:
        os.close(descriptor)
        raise


def _block(fd: int, size: int, reference: dict):
    if type(reference) is not dict or not {'shard', 'block'} <= set(reference) or set(reference) - {'shard', 'block', 'coverage'} or reference['shard'] != 'correction/frames.astraframes':
        raise RecordingError('Correction frame reference escapes its archive')
    value = reference['block']
    if type(value) is not dict or set(value) != {'offset', 'length', 'metadata', 'digest'}:
        raise RecordingError('Invalid correction frame block')
    offset = _integer(value['offset']); length = _integer(value['length'], 53, MAXIMUM_PIXELS + 65588)
    if offset + length > size:
        raise RecordingError('Correction frame extends beyond its archive')
    prefix = os.pread(fd, 20, offset)
    if len(prefix) != 20 or prefix[:8] != b'ASTRAF01':
        raise RecordingError('Correction frame has an invalid archive header')
    metadata_size, pixel_size = struct.unpack('<IQ', prefix[8:])
    if not 1 <= metadata_size <= 65536 or not 1 <= pixel_size <= MAXIMUM_PIXELS or length != 52 + metadata_size + pixel_size:
        raise RecordingError('Correction frame has invalid byte bounds')
    metadata = validate_frame(_json(os.pread(fd, metadata_size, offset + 20), limit=65536))
    if metadata != value['metadata']:
        raise RecordingError('Correction frame metadata differs from its saved block')
    digest = hashlib.sha256(prefix)
    remaining, cursor = length - 52, offset + 20
    while remaining:
        chunk = os.pread(fd, min(1024**2, remaining), cursor)
        if not chunk:
            raise RecordingError('Correction frame became truncated')
        digest.update(chunk); cursor += len(chunk); remaining -= len(chunk)
    checksum = os.pread(fd, 32, cursor)
    if digest.hexdigest() != _digest(value['digest']) or checksum.hex() != value['digest']:
        raise RecordingError('Correction frame checksum does not match')
    return metadata


def read_prelude(directory: Path, descriptor: dict) -> dict:
    if (type(descriptor) is not dict or set(descriptor) != {'schemaVersion', 'path', 'sha256'} or
            type(descriptor['schemaVersion']) is not int or descriptor['schemaVersion'] != 1 or descriptor['path'] != 'correction/prelude.json'):
        raise RecordingError('Unsupported correction recording descriptor')
    fd, size = _open(directory, 'prelude.json', MAXIMUM_PRELUDE_BYTES)
    try:
        data = os.pread(fd, size, 0)
    finally:
        os.close(fd)
    if len(data) != size or hashlib.sha256(data).hexdigest() != _digest(descriptor['sha256']):
        raise RecordingError('Correction provenance checksum does not match')
    value = _json(data, limit=MAXIMUM_PRELUDE_BYTES)
    fields = {'schemaVersion', 'sourceRunID', 'sourceCheckpointID', 'sourcePolicySignature', 'contextIDs',
              'requestedAtNanos', 'controlJoinedAtNanos', 'supervisionStartNanos', 'maximumDurationNanos',
              'maximumBytes', 'continuityProven', 'observations'}
    if type(value) is not dict or set(value) != fields or type(value['schemaVersion']) is not int or value['schemaVersion'] != 1:
        raise RecordingError('Unsupported correction prelude schema')
    _uuid(value['sourceRunID']); _uuid(value['sourceCheckpointID']); _digest(value['sourcePolicySignature'])
    requested, joined, supervised = (_integer(value[name]) for name in ('requestedAtNanos', 'controlJoinedAtNanos', 'supervisionStartNanos'))
    if supervised < max(requested, joined) or value['continuityProven'] is not False:
        raise RecordingError('Correction requires joined controls and an explicit supervision gate; continuous handoff is not supported')
    if (_integer(value['maximumDurationNanos']) != MAXIMUM_DURATION or _integer(value['maximumBytes']) != MAXIMUM_PIXELS):
        raise RecordingError('Correction pre-roll exceeds its supported retention contract')
    contexts = value['contextIDs']
    if type(contexts) is not list or len(contexts) > 32:
        raise RecordingError('Correction contexts must be bounded categorical choices')
    for choice in contexts:
        _integer(choice, 0, 65535)
    rows = value['observations']
    if type(rows) is not list or len(rows) > MAXIMUM_OBSERVATIONS:
        raise RecordingError('Correction has too many pre-roll observations')
    archive, archive_size = _open(directory, 'frames.astraframes', MAXIMUM_PIXELS + MAXIMUM_PRELUDE_BYTES)
    try:
        seen, blocks = set(), {}
        first = previous = None
        surfaces = None
        retained = 0
        from .environments.interface import control_observation, EnvironmentError
        for row in rows:
            if type(row) is not dict or set(row) != {'actorInput', 'frames'}:
                raise RecordingError('Invalid retained correction observation')
            observed = row['actorInput']
            required = {'observationID', 'episodeID', 'previousStateID', 'cutoffNanos', 'geometryRevision',
                        'controlState', 'executedEvents', 'intervalCovered', 'contextIDs'}
            if type(observed) is not dict or not required <= set(observed) or set(observed) - required - {'controlCoverageNanos'}:
                raise RecordingError('Correction observation does not contain original actor input')
            identifier = _uuid(observed['observationID']); _uuid(observed['episodeID']); _uuid(observed['previousStateID'])
            cutoff = _integer(observed['cutoffNanos']); _integer(observed['geometryRevision'])
            if (identifier in seen or cutoff > min(requested, joined) or previous is not None and cutoff <= previous or
                    observed['contextIDs'] != contexts or type(observed['contextIDs']) is not list or
                    any(type(choice) is not int for choice in observed['contextIDs'])):
                raise RecordingError('Correction observation identity, order, time or contexts changed')
            seen.add(identifier); first = cutoff if first is None else first
            if cutoff - first > MAXIMUM_DURATION:
                raise RecordingError('Correction pre-roll duration exceeds its bound')
            try:
                control_observation(observed['controlState'], cutoff, observed.get('controlCoverageNanos'), interval_covered=observed['intervalCovered'])
            except EnvironmentError as error:
                raise RecordingError(str(error)) from error
            events = observed['executedEvents']
            if type(events) is not list or len(events) > 2048:
                raise RecordingError('Correction input history exceeds its bound')
            last_event = None
            for event in events:
                validate_event(event)
                if (event['eventNanos'] > event['observedNanos'] or event['observedNanos'] > cutoff or
                        previous is not None and event['observedNanos'] <= previous or
                        last_event is not None and event['sequence'] <= last_event):
                    raise RecordingError('Correction input history is not causal and ordered')
                last_event = event['sequence']
            frames = row['frames']
            if type(frames) is not list or not 1 <= len(frames) <= 16:
                raise RecordingError('Correction observation requires a complete bounded source group')
            row_surfaces = []
            for reference in frames:
                metadata = _block(archive, archive_size, reference)
                block = reference['block']; key = block['offset']
                if key in blocks and blocks[key] != block:
                    raise RecordingError('Correction archive offset was reused with another identity')
                blocks[key] = block
                retained += metadata['byteCount']
                if retained > MAXIMUM_PIXELS or not metadata['eventNanos'] <= metadata['observedNanos'] <= cutoff:
                    raise RecordingError('Correction pixels exceed their retention or causal bound')
                proof = reference.get('coverage')
                if proof is not None:
                    if type(proof) is not dict:
                        raise RecordingError('Invalid correction frame coverage')
                    for name in ('eventNanos', 'observedNanos', 'throughNanos', 'verifiedAtNanos'):
                        _integer(proof.get(name))
                    if proof.get('kind') == 'frame':
                        if (set(proof) != {'streamID', 'frameID', 'surface', 'eventNanos', 'observedNanos', 'throughNanos', 'verifiedAtNanos', 'kind'} or
                                _uuid(proof['frameID']) != _uuid(metadata['id']) or proof['surface'] != metadata['surface'] or
                                proof['eventNanos'] != metadata['eventNanos'] or proof['observedNanos'] != metadata['observedNanos'] or
                                proof['throughNanos'] != metadata['observedNanos'] or proof['verifiedAtNanos'] != metadata['observedNanos']):
                            raise RecordingError('Correction frame coverage changed its original evidence')
                        _uuid(proof['streamID'])
                    else:
                        validate_frame_coverage(proof, frame=metadata, cutoff=cutoff)
                age_from = proof['throughNanos'] if proof is not None and proof['kind'] == 'unchanged' else metadata['eventNanos']
                if cutoff - age_from > 250_000_000:
                    raise RecordingError('Correction pre-roll contains a stale source')
                row_surfaces.append(metadata['surface'])
            if len({surface['id'] for surface in row_surfaces}) != len(row_surfaces) or surfaces is not None and row_surfaces != surfaces:
                raise RecordingError('Correction pre-roll source membership or geometry changed')
            surfaces = row_surfaces; previous = cutoff
        position = 0
        for block in sorted(blocks.values(), key=lambda block: block['offset']):
            if block['offset'] != position:
                raise RecordingError('Correction frame archive has gaps or overlaps')
            position += block['length']
        if position != archive_size:
            raise RecordingError('Correction frame archive contains unclaimed bytes')
    finally:
        os.close(archive)
    return value
