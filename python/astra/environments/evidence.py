"""Shared delayed-label and execution-evidence validation for rollout consumers."""
from __future__ import annotations
from dataclasses import dataclass
import copy
import math
from .interface import DecisionContext, EnvironmentError, controls, identifier, integer, fields, same_id

OUTCOMES = {'continuing': 'continuing', 'succeeded': 'terminated', 'failed': 'terminated',
            'terminated': 'terminated', 'truncated': 'truncated', 'aborted': 'aborted'}

@dataclass
class DecisionEvidence:
    context: DecisionContext
    sequence: int
    commands: tuple[dict, ...]
    observation: dict | None = None
    reward: float | None = None
    outcome: str | None = None
    end: int | None = None
    reason: str | None = None
    admitted: bool = False
    admission_seen: bool = False
    receipt: dict | None = None
    assembled: bool = False


    def accept_label(self, kind, value, *, period_ms, watermark, maximum_interval_nanos=None):
        if kind not in ('environment.reward', 'environment.outcome'):
            raise EnvironmentError('Unsupported label evidence kind')
        expected = ('clockID', 'episodeID', 'startNanos', 'endNanos', 'value') if kind == 'environment.reward' else ('clockID', 'episodeID', 'endNanos', 'outcome')
        fields(value, expected, ('reason',) if kind == 'environment.outcome' else ())
        end = integer(value['endNanos'])
        start = self.context.cutoff_nanos
        maximum = period_ms * 1000000 if maximum_interval_nanos is None else maximum_interval_nanos
        if not start < end <= start + maximum or end <= watermark or (self.end is not None and self.end != end):
            raise EnvironmentError('Reward/outcome interval is shifted, revised after sealing or inconsistent')
        self.end = end
        if kind == 'environment.reward':
            if integer(value['startNanos']) != start: raise EnvironmentError('Reward must begin at the decision cutoff, not execution time')
            reward = value['value']
            if reward is not None and (type(reward) not in (int, float) or not math.isfinite(reward)):
                raise EnvironmentError('Reward must be finite or explicitly unknown')
            self.reward = reward
        else:
            outcome = value['outcome']
            if outcome is not None and outcome not in (*OUTCOMES, 'unknown'):
                raise EnvironmentError('Unsupported episode outcome')
            self.outcome = outcome
            self.reason = value.get('reason')
            if self.reason is not None and (type(self.reason) is not str or len(self.reason.encode()) > 2048):
                raise EnvironmentError('Outcome reason must be bounded text')
            if outcome == 'continuing' and (end < start + period_ms * 1000000 or
                    (maximum_interval_nanos is None and end != start + period_ms * 1000000)):
                raise EnvironmentError('A continuing decision must cover the negotiated minimum period')
            if outcome in ('succeeded', 'failed', 'terminated', 'truncated', 'aborted'):
                return end
        return None

    def accept_receipt(self, value, *, lead_ms, retain_failure=False):
        fields(value, ('clockID', 'episodeID', 'receipt'), ('cancellationCause',))
        receipt = value['receipt']
        fields(receipt, ('packetID', 'runID', 'sequence', 'status', 'observedNanos', 'commandResults', 'resultingState'))
        if not same_id(receipt['packetID'], self.context.packet_id) or not same_id(receipt['runID'], self.context.run_id) or integer(receipt['sequence']) != self.sequence:
            raise EnvironmentError('Execution receipt identity does not match its policy packet')
        integer(receipt['observedNanos']); controls(receipt['resultingState'], receipt['observedNanos'])
        if receipt['resultingState']['valid'] is not True:
            raise EnvironmentError('Executor reported unavailable controls')
        status = receipt['status']
        if status in ('late', 'rejected') and not retain_failure:
            raise EnvironmentError('A policy packet was late or rejected; rollout admission stopped')
        if status == 'admitted':
            if self.admission_seen: raise EnvironmentError('Packet admission receipt was duplicated')
            if receipt['commandResults']: raise EnvironmentError('Admission cannot claim execution effects')
            self.admitted = self.admission_seen = True
        elif status in ('executed', 'cancelled', 'late', 'rejected'):
            if self.receipt is not None: raise EnvironmentError('Final packet execution receipt was duplicated')
            results = receipt['commandResults']
            if type(results) is not list or len(results) > len(self.commands) or (status != 'rejected' and len(results) != len(self.commands)):
                raise EnvironmentError('Final receipt omits or duplicates policy commands')
            indices = set()
            for result in results:
                fields(result, ('commandIndex', 'scheduledNanos', 'status'), ('postedNanos', 'message'))
                index = integer(result['commandIndex'], max(len(self.commands) - 1, 0))
                scheduled = self.context.cutoff_nanos + lead_ms * 1000000 + self.commands[index]['offsetMs'] * 1000000
                accepted_statuses = ('posted', 'noOp', 'cancelled', 'failed') if retain_failure else ('posted', 'noOp', 'cancelled')
                if index in indices or result['scheduledNanos'] != scheduled or result['status'] not in accepted_statuses:
                    raise EnvironmentError('Final command receipt is invalid or reports execution failure')
                indices.add(index)
                if result['status'] == 'posted':
                    posted = integer(result.get('postedNanos'))
                    if not scheduled <= posted <= receipt['observedNanos']:
                        raise EnvironmentError('Posted command time is outside its execution/receipt interval')
                elif result.get('postedNanos') is not None:
                    raise EnvironmentError('An unposted command cannot claim a posting timestamp')
                elif result['status'] == 'noOp' and receipt['observedNanos'] < scheduled:
                    raise EnvironmentError('No-op execution was reported before its scheduled time')
            self.receipt = {**copy.deepcopy(receipt), 'cancellationCause': value.get('cancellationCause')}
            if status == 'executed': self.admitted = True
        else: raise EnvironmentError('Unsupported execution receipt status')

    def execution_complete(self, boundary_nanos):
        receipt = self.receipt
        if receipt is None: return False
        if receipt['status'] in ('late', 'rejected') or any(result['status'] == 'failed' for result in receipt['commandResults']):
            raise EnvironmentError('Failed or late execution cannot enter the on-policy prefix')
        if receipt['status'] == 'cancelled':
            if receipt['cancellationCause'] != 'episodeBoundary':
                raise EnvironmentError('Administrative cancellation cannot be on-policy experience')
            if boundary_nanos is None: return False
            if receipt['observedNanos'] < boundary_nanos:
                raise EnvironmentError('Cancellation predates the confirmed episode boundary')
        for result in receipt['commandResults']:
            if result['status'] == 'cancelled':
                if receipt['status'] != 'cancelled' or receipt['cancellationCause'] != 'episodeBoundary' or boundary_nanos is None or result['scheduledNanos'] < boundary_nanos:
                    raise EnvironmentError('Cancellation of a policy command is not explained by a future episode boundary')
        return True
