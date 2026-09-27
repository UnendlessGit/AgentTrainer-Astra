"""Simulator-owned admitted-packet ledger, separate from its command heap."""
from __future__ import annotations

import copy
import uuid
from astra.control_feedback import (MAX_BYTES, MAX_PACKETS, MAX_CHANGES, UINT64_MAX,
                                    ControlFeedbackError, canonical_feedback, validate_control_feedback)


class PracticePacketLedger:
    def __init__(self, run_id, geometry_revision=0):
        self.run_id = run_id
        self.geometry_revision = geometry_revision
        self.epoch_id = str(uuid.uuid4())
        self.rows = {}
        self.changes = []
        self._reservations = {}
        self._bytes = 4096
        self._next_sequence = 0
        self._acknowledged = None
        self._delivered = None
        self._unavailable = None

    def acknowledge(self, sequence):
        if sequence is None:
            if self._acknowledged is not None:
                raise ControlFeedbackError('Practice feedback acknowledgement regressed')
            return
        if (type(sequence) is not int or self._delivered is None or sequence > self._delivered or
                (self._acknowledged is not None and sequence < self._acknowledged)):
            raise ControlFeedbackError('Practice feedback cursor is outside delivered history')
        self._acknowledged = sequence
        self.changes = [item for item in self.changes if item['sequence'] > sequence]
        for identity, row in tuple(self.rows.items()):
            if row.get('terminal', {}).get('sequence', UINT64_MAX) <= sequence:
                del self.rows[identity]
                self._bytes -= self._reservations.pop(identity)

    def admit(self, packet, now):
        identity = packet['id']
        reservation = len(canonical_feedback(packet)) + 2048 + 512 * len(packet['commands'])
        outstanding = sum(row.get('terminal') is None for row in self.rows.values())
        if (identity in self.rows or len(self.rows) >= MAX_PACKETS or outstanding >= 32 or
                reservation > MAX_BYTES - self._bytes or len(self.changes) + outstanding + 2 > MAX_CHANGES or
                UINT64_MAX - self._next_sequence < outstanding + 2):
            raise ControlFeedbackError('Practice feedback admission capacity is exhausted')
        row = {'packet': copy.deepcopy(packet), 'admissionSequence': self._next_sequence, 'admittedNanos': now,
               'progress': [{'commandIndex': index, 'status': 'pending', 'completedSampleCount': 0,
                             **({'emittedDx': 0, 'emittedDy': 0} if command['operation'] == 'pointerRelative' else {})}
                            for index, command in enumerate(packet['commands'])]}
        self.rows[identity] = row
        self._reservations[identity] = reservation
        self._bytes += reservation
        self.changes.append({'sequence': self._next_sequence, 'packetID': identity, 'kind': 'admitted', 'availableNanos': now})
        self._next_sequence += 1

    def complete(self, identity, command_index, sample, *, endpoint, status, now):
        row = self.rows[identity]
        state = row['progress'][command_index]
        state['completedSampleCount'] += 1
        state['lastCompletedOffsetMs'] = sample['offsetMs']
        state['lastCompletedAvailableNanos'] = now
        if status == 'posted':
            state['lastPostedNanos'] = now
            if sample['operation'] == 'pointerRelative':
                state['emittedDx'] += sample['dx']
                state['emittedDy'] += sample['dy']
        state['status'] = status if endpoint else 'partial'

    def finish(self, identity, status, now):
        row = self.rows[identity]
        if row.get('terminal') is not None:
            return
        if status != 'executed':
            self._unavailable = 'stopping'
        for state in row['progress']:
            if state['status'] in ('pending', 'partial'):
                state['status'] = 'cancelled'
        row['terminal'] = {'sequence': self._next_sequence, 'status': status, 'availableNanos': now}
        self.changes.append({'sequence': self._next_sequence, 'packetID': identity, 'kind': 'terminal', 'availableNanos': now})
        self._next_sequence += 1

    def snapshot(self, cutoff):
        value = {'version': 1, 'controlEpochID': self.epoch_id, 'runID': self.run_id,
                 'geometryRevision': self.geometry_revision, 'cutoffNanos': cutoff,
                 'changes': self.changes, 'packets': sorted(self.rows.values(), key=lambda row: row['packet']['sequence'])}
        if self._acknowledged is not None:
            value['acknowledgedThrough'] = self._acknowledged
        if self._next_sequence:
            value['throughSequence'] = self._next_sequence - 1
        if self._unavailable is None:
            value['coverageNanos'] = cutoff
            self._delivered = value.get('throughSequence')
        else:
            value['unavailableReason'] = self._unavailable
        validate_control_feedback(value, cutoff_nanos=cutoff)
        return copy.deepcopy(value)
