"""Bounded, causal executor evidence. This never infers a queue from targets.

The wire object is preserved exactly (including absent optional fields). IDs and
clock magnitudes authenticate evidence; neural feature construction is separate.
"""
from __future__ import annotations

import copy
import json
import math
import uuid

MAX_PACKETS = 64
MAX_CHANGES = 128
MAX_BYTES = 262_144
UINT64_MAX = 2**64 - 1
_UNSPECIFIED = object()


class ControlFeedbackError(ValueError):
    pass


def _require(condition, message='Invalid or noncausal queued-control evidence'):
    if not condition:
        raise ControlFeedbackError(message)


def _fields(value, required, optional=()):
    _require(type(value) is dict and set(required) <= value.keys() and not value.keys() - set(required) - set(optional),
             'Unknown or missing queued-control fields')


def _uint(value, maximum=UINT64_MAX, minimum=0):
    _require(type(value) is int and minimum <= value <= maximum)
    return value


def _optional_uint(value):
    return None if value is None else _uint(value)


def _id(value):
    _require(type(value) is str and len(value) == 36)
    try:
        return str(uuid.UUID(value))
    except ValueError as error:
        raise ControlFeedbackError('Invalid queued-control identity') from error


def canonical_feedback(value):
    try:
        result = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False, ensure_ascii=False).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise ControlFeedbackError('Malformed queued-control object') from error
    _require(len(result) <= MAX_BYTES, 'Queued-control evidence exceeds its byte limit')
    return result


def _command(command, duration, previous_offset, surfaces):
    arguments = {'keyDown': ('keyCode',), 'keyUp': ('keyCode',), 'keyRepeat': ('keyCode',),
                 'buttonDown': ('button',), 'buttonUp': ('button',),
                 'pointerAbsolute': ('surfaceID', 'x', 'y'), 'pointerRelative': ('dx', 'dy'), 'scroll': ('dx', 'dy')}
    _fields(command, ('operation', 'offsetMs'), ('keyCode', 'button', 'surfaceID', 'x', 'y', 'dx', 'dy'))
    op = command['operation']
    _require(type(op) is str and op in arguments)
    _require(all(command.get(name) is not None for name in arguments[op]))
    _require(all(value is None for name, value in command.items() if name not in arguments[op] + ('operation', 'offsetMs')))
    motion = op in ('pointerAbsolute', 'pointerRelative')
    _uint(command['offsetMs'], duration if motion else duration - 1, previous_offset)
    if op.startswith('key'):
        _uint(command['keyCode'], 127)
    elif op.startswith('button'):
        _uint(command['button'], 31)
    elif op == 'pointerAbsolute':
        _require(type(command['surfaceID']) is str and 1 <= len(command['surfaceID'].encode()) <= 256)
        _require(surfaces is None or command['surfaceID'] in surfaces)
        _require(all(type(command[name]) in (int, float) and math.isfinite(command[name]) and 0 <= command[name] < 1 for name in ('x', 'y')))
    else:
        for name in ('dx', 'dy'):
            number = command[name]
            _require(type(number) in (int, float) and math.isfinite(number) and -32768 <= number <= 32767)
            if op == 'pointerRelative':
                _require(number == round(number))
    return motion


def validate_control_feedback(value, *, cutoff_nanos, geometry_revision=None, run_id=None,
                              control_epoch_id=None, acknowledged_through=_UNSPECIFIED,
                              surfaces=None, require_available=False):
    """Validate a complete original v1 snapshot without normalizing its bytes.

    ``require_available`` is mandatory at live/PPO admission. Offline BC may
    represent missing evidence separately; a malformed supplied object fails.
    """
    _uint(cutoff_nanos)
    _fields(value, ('version', 'controlEpochID', 'runID', 'geometryRevision', 'cutoffNanos', 'changes', 'packets'),
            ('coverageNanos', 'unavailableReason', 'acknowledgedThrough', 'throughSequence'))
    _require(type(value['version']) is int and value['version'] == 1)
    epoch, run = _id(value['controlEpochID']), _id(value['runID'])
    geometry = _uint(value['geometryRevision'])
    _require(_uint(value['cutoffNanos']) == cutoff_nanos)
    _require(geometry_revision is None or geometry == geometry_revision)
    _require(run_id is None or run == _id(run_id))
    _require(control_epoch_id is None or epoch == _id(control_epoch_id))
    reason = value.get('unavailableReason')
    _require(reason in (None, 'postInFlight', 'arming', 'stopping', 'untrustedState', 'historyGap', 'overflow'))
    _require(not require_available or reason is None, 'Queued-control snapshot is unavailable')
    coverage = _optional_uint(value.get('coverageNanos'))
    _require(coverage == cutoff_nanos if reason is None else coverage is None)
    acknowledged = _optional_uint(value.get('acknowledgedThrough'))
    through = _optional_uint(value.get('throughSequence'))
    _require(acknowledged_through is _UNSPECIFIED or acknowledged == acknowledged_through,
             'Queued-control cursor did not continue the consumed observation')
    changes, rows = value['changes'], value['packets']
    _require(type(changes) is list and len(changes) <= MAX_CHANGES and type(rows) is list and len(rows) <= MAX_PACKETS)
    surface_ids = None if surfaces is None else {surface['id'] for surface in surfaces}
    by_id, previous_packet, previous_admission, previous_time = {}, None, None, None
    for row in rows:
        _fields(row, ('packet', 'admissionSequence', 'admittedNanos', 'progress'), ('terminal',))
        packet = row['packet']
        _fields(packet, ('id', 'runID', 'sequence', 'observationID', 'geometryRevision', 'executeAtNanos', 'durationMs', 'commands'))
        packet_id, packet_sequence = _id(packet['id']), _uint(packet['sequence'])
        _id(packet['observationID'])
        _require(packet_id not in by_id and _id(packet['runID']) == run and _uint(packet['geometryRevision']) == geometry)
        admission, admitted = _uint(row['admissionSequence']), _uint(row['admittedNanos'], cutoff_nanos)
        _require(through is not None and admission <= through)
        _require(previous_packet is None or (packet_sequence > previous_packet and admission > previous_admission and admitted >= previous_time))
        previous_packet, previous_admission, previous_time = packet_sequence, admission, admitted
        duration, start = _uint(packet['durationMs'], 1000, 1), _uint(packet['executeAtNanos'])
        _require(start + duration * 1_000_000 <= UINT64_MAX)
        commands, progress = packet['commands'], row['progress']
        _require(type(commands) is list and len(commands) <= 64 and type(progress) is list and len(progress) == len(commands))
        terminal = row.get('terminal')
        if terminal is not None:
            _fields(terminal, ('sequence', 'status', 'availableNanos'))
            _require(_uint(terminal['sequence']) > admission and terminal['sequence'] <= through)
            _uint(terminal['availableNanos'], cutoff_nanos, admitted)
            _require(terminal['status'] in ('executed', 'cancelled', 'late'))
            _require(terminal['status'] == 'executed' or reason is not None)
            if terminal['status'] == 'executed':
                _require(terminal['availableNanos'] >= start + duration * 1_000_000)
        previous_motion, offset = None, 0
        for index, (command, state) in enumerate(zip(commands, progress)):
            motion = _command(command, duration, offset, surface_ids)
            offset = command['offsetMs']
            if motion and previous_motion is not None:
                _require(command['operation'] == previous_motion['operation'])
            interpolated = motion and previous_motion is not None and offset > previous_motion['offsetMs'] and (
                command['operation'] == 'pointerRelative' or command['surfaceID'] == previous_motion['surfaceID'])
            sample_limit = offset - previous_motion['offsetMs'] if interpolated else 1
            _fields(state, ('commandIndex', 'status', 'completedSampleCount'),
                    ('lastCompletedOffsetMs', 'lastCompletedAvailableNanos', 'lastPostedNanos', 'emittedDx', 'emittedDy'))
            _require(_uint(state['commandIndex'], 63) == index)
            count = _uint(state['completedSampleCount'], sample_limit)
            last_offset = state.get('lastCompletedOffsetMs')
            available, posted = state.get('lastCompletedAvailableNanos'), state.get('lastPostedNanos')
            _require((count > 0) == (last_offset is not None) == (available is not None))
            if count:
                _uint(last_offset, 1000)
                _require(last_offset == (previous_motion['offsetMs'] + count if interpolated else offset))
                scheduled = start + last_offset * 1_000_000
                _uint(available, cutoff_nanos, max(scheduled, admitted))
                if posted is not None:
                    _uint(posted, available, scheduled)
                if terminal is not None:
                    _require(available <= terminal['availableNanos'])
            else:
                _require(posted is None)
            status = state['status']
            _require(status in ('pending', 'partial', 'posted', 'noOp', 'cancelled', 'failed'))
            if status == 'pending': _require(count == 0 and posted is None)
            elif status == 'partial': _require(motion and count > 0 and last_offset < offset and posted is not None)
            elif status == 'posted': _require(count > 0 and last_offset == offset and posted is not None)
            elif status == 'noOp': _require(not motion and count == 1 and last_offset == offset and posted is None)
            if reason is None: _require(status not in ('cancelled', 'failed'))
            if terminal is not None and terminal['status'] == 'executed': _require(status in ('posted', 'noOp'))
            if command['operation'] == 'pointerRelative':
                for component, name in (('dx', 'emittedDx'), ('dy', 'emittedDy')):
                    _require(type(state.get(name)) is int and -32768 <= state[name] <= 32767)
                    _require(state[name] == round(command[component] * (count / sample_limit)))
            else:
                _require(state.get('emittedDx') is None and state.get('emittedDy') is None)
            if motion: previous_motion = command
        by_id[packet_id] = row
    preceding, preceding_time, listed = acknowledged, 0, set()
    for change in changes:
        _fields(change, ('sequence', 'packetID', 'kind', 'availableNanos'))
        sequence = _uint(change['sequence'])
        _require(sequence == (0 if preceding is None else preceding + 1))
        available = _uint(change['availableNanos'], cutoff_nanos, preceding_time)
        packet_id = _id(change['packetID'])
        _require(packet_id in by_id and change['kind'] in ('admitted', 'terminal'))
        row = by_id[packet_id]
        if change['kind'] == 'admitted':
            _require((sequence, available) == (row['admissionSequence'], row['admittedNanos']))
        else:
            terminal = row.get('terminal')
            _require(terminal is not None and (sequence, available) == (terminal['sequence'], terminal['availableNanos']))
        listed.add((packet_id, change['kind']))
        preceding, preceding_time = sequence, available
    _require(preceding == through)
    for packet_id, row in by_id.items():
        if acknowledged is None or row['admissionSequence'] > acknowledged:
            _require((packet_id, 'admitted') in listed)
        if row.get('terminal') is not None:
            _require((packet_id, 'terminal') in listed)
    canonical_feedback(value)
    return value


def validate_feedback_continuation(previous, current):
    """Require original outstanding plans to survive until a real terminal."""
    if previous is None:
        _require(current.get('acknowledgedThrough') is None)
        return
    _require(_id(previous['controlEpochID']) == _id(current['controlEpochID']))
    _require(current.get('acknowledgedThrough') == previous.get('throughSequence'))
    _require(current['cutoffNanos'] > previous['cutoffNanos'])
    current_rows = {_id(row['packet']['id']): row for row in current['packets']}
    old_ids = {_id(row['packet']['id']) for row in previous['packets']}
    for identity, row in current_rows.items():
        if identity not in old_ids:
            _require(previous.get('throughSequence') is None or row['admissionSequence'] > previous['throughSequence'],
                     'An earlier unobserved packet appeared after its admission cursor')
    for old in previous['packets']:
        if old.get('terminal') is not None:
            continue
        new = current_rows.get(_id(old['packet']['id']))
        _require(new is not None and new['packet'] == old['packet'] and new['admissionSequence'] == old['admissionSequence'] and new['admittedNanos'] == old['admittedNanos'],
                 'An outstanding original packet disappeared or changed')
        for before, after in zip(old['progress'], new['progress']):
            _require(after['completedSampleCount'] >= before['completedSampleCount'])
            if before['status'] in ('posted', 'noOp') or before['completedSampleCount'] == after['completedSampleCount']:
                _require(before == after, 'Committed queued-control progress changed')


def validate_control_exclusion(value, *, recording_id):
    """Offline integrity proof, never an executor epoch or a neural timestamp."""
    _fields(value, ('schemaVersion', 'ownershipID', 'recordingID', 'startedNanos'), ('throughNanos', 'producersJoinedNanos'))
    _require(type(value['schemaVersion']) is int and value['schemaVersion'] == 1)
    _id(value['ownershipID']); _require(_id(value['recordingID']) == _id(recording_id))
    start = _uint(value['startedNanos'], 2**63 - 1)
    through, joined = value.get('throughNanos'), value.get('producersJoinedNanos')
    _require((through is None) == (joined is None), 'Recording exclusion needs a complete producer-join seal')
    if through is not None:
        _uint(through, 2**63 - 1, start)
        _uint(joined, 2**63 - 1, through)
    return value


def exclusion_covers(value, cutoff_nanos):
    return (value is not None and value.get('throughNanos') is not None and value.get('producersJoinedNanos') is not None
            and value['startedNanos'] <= cutoff_nanos <= value['throughNanos'])
