"""One shared semantic builder for live, practice, BC and rollout replay.

Only actual admitted-control evidence is accepted. Known-empty human recordings
must be certified by the recording loader; shifted teacher packets are not input.
"""
from __future__ import annotations

import math
import uuid
import numpy as np

from astra.control_feedback import validate_control_feedback
from astra.model.queued_control import QueuedControlBatch
from astra.model.queue_layout import (MAXIMUM_PACKETS, CATEGORICAL_NAMES, COMMAND_FEATURE_NAMES, PACKET_FEATURE_NAMES,
                                     OPERATIONS, PROGRESS, PACKET_STATUS)


def _signed_log(value):
    return math.copysign(math.log1p(abs(value)), value) if value else 0.0


def _time(delta_nanos):
    seconds = delta_nanos / 1_000_000_000
    return max(-4.0, min(4.0, seconds)) / 4.0, _signed_log(seconds)


def _delta(value):
    # Native semantic deltas span [-32768,32767]. Unclipped binary scaling
    # preserves the whole range; the log term retains sensitivity near zero.
    return value / 4096.0, _signed_log(value) / math.log1p(32768)


def _arguments(command):
    if command is None: return (0.0,) * 6
    x, y = command.get('x') or 0.0, command.get('y') or 0.0
    dx, dy = command.get('dx') or 0.0, command.get('dy') or 0.0
    scaled_x, log_x = _delta(dx); scaled_y, log_y = _delta(dy)
    return x, y, scaled_x, scaled_y, log_x, log_y


def prepare_queued_controls(feedback, surfaces, config, *, cutoff_nanos, known_empty=False):
    if type(known_empty) is not bool or known_empty and feedback is not None:
        raise ValueError('Queued-control availability needs one unambiguous provenance')
    if config.schema_version == 2:
        return None
    if config.schema_version != 3:
        raise ValueError('Queued-control tensors require model schema 3')
    if feedback is None:
        return QueuedControlBatch.empty(1, 1, config.packet_capacity, available=known_empty)
    validate_control_feedback(feedback, cutoff_nanos=cutoff_nanos, surfaces=surfaces)
    if any(len(row['packet']['commands']) > config.packet_capacity for row in feedback['packets']):
        raise ValueError('Queued-control packet exceeds this model command capacity')
    available = feedback.get('coverageNanos') == cutoff_nanos and feedback.get('unavailableReason') is None
    if not available:
        return QueuedControlBatch.empty(1, 1, config.packet_capacity, available=False)
    command_shape = (1, 1, MAXIMUM_PACKETS, config.packet_capacity)
    packet_shape = command_shape[:-1]
    arrays = dict(categorical=np.zeros((*command_shape, len(CATEGORICAL_NAMES)), dtype=np.int32),
                  command_features=np.zeros((*command_shape, len(COMMAND_FEATURE_NAMES)), dtype=np.float32),
                  command_mask=np.zeros(command_shape, dtype=np.bool_),
                  packet_features=np.zeros((*packet_shape, len(PACKET_FEATURE_NAMES)), dtype=np.float32),
                  packet_status=np.zeros(packet_shape, dtype=np.int32), packet_mask=np.zeros(packet_shape, dtype=np.bool_),
                  available=np.ones((1, 1), dtype=np.bool_))
    roles = {surface['id']: index + 1 for index, surface in enumerate(surfaces)}
    changes = {(str(uuid.UUID(change['packetID'])), change['kind']) for change in feedback['changes']}
    for row_index, row in enumerate(feedback['packets']):
        packet = row['packet']; commands = packet['commands']; duration = packet['durationMs']
        start = packet['executeAtNanos']; packet_id = str(uuid.UUID(packet['id']))
        terminal = row.get('terminal')
        arrays['packet_mask'][0, 0, row_index] = True
        arrays['packet_status'][0, 0, row_index] = 0 if terminal is None else PACKET_STATUS[terminal['status']]
        arrays['packet_features'][0, 0, row_index] = (
            *_time(cutoff_nanos - row['admittedNanos']), *_time(start - cutoff_nanos),
            *_time(start + duration * 1_000_000 - cutoff_nanos), duration / 1000.0,
            len(commands) / config.packet_capacity, float((packet_id, 'admitted') in changes),
            float((packet_id, 'terminal') in changes),
            0.0 if terminal is None else math.log1p((cutoff_nanos - terminal['availableNanos']) / 1e9))
        previous_motion = None
        for command_index, (command, progress) in enumerate(zip(commands, row['progress'])):
            operation = command['operation']; offset = command['offsetMs']
            motion = operation in ('pointerAbsolute', 'pointerRelative')
            anchor = previous_motion if motion else None
            samples = 1
            if (anchor is not None and offset > anchor['offsetMs'] and operation == anchor['operation'] and
                    (operation == 'pointerRelative' or command['surfaceID'] == anchor['surfaceID'])):
                samples = offset - anchor['offsetMs']
            arrays['categorical'][0, 0, row_index, command_index] = (
                OPERATIONS[operation], 0 if command.get('keyCode') is None else command['keyCode'] + 1,
                0 if command.get('button') is None else command['button'] + 1,
                roles.get(command.get('surfaceID'), 0), PROGRESS[progress['status']],
                0 if anchor is None else OPERATIONS[anchor['operation']],
                0 if anchor is None else roles.get(anchor.get('surfaceID'), 0))
            completed = progress['completedSampleCount']; last_offset = progress.get('lastCompletedOffsetMs')
            completed_at = progress.get('lastCompletedAvailableNanos'); posted_at = progress.get('lastPostedNanos')
            anchor_values = _arguments(anchor)
            arrays['command_features'][0, 0, row_index, command_index] = (
                offset / duration, *_time(start + offset * 1_000_000 - cutoff_nanos), *_arguments(command),
                float(anchor is not None), 0.0 if anchor is None else anchor['offsetMs'] / duration, *anchor_values,
                completed / samples, math.log1p(completed) / math.log1p(1000), float(last_offset is not None),
                0.0 if last_offset is None else last_offset / duration,
                0.0 if completed_at is None else math.log1p((cutoff_nanos - completed_at) / 1e9),
                float(posted_at is not None), 0.0 if posted_at is None else math.log1p((cutoff_nanos - posted_at) / 1e9),
                (progress.get('emittedDx') or 0) / 4096.0, (progress.get('emittedDy') or 0) / 4096.0)
            arrays['command_mask'][0, 0, row_index, command_index] = True
            if motion: previous_motion = command
    return QueuedControlBatch.from_numpy(maximum_surfaces=config.maximum_surfaces, **arrays)
