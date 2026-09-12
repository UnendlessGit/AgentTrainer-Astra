"""Asynchronous external observations/rewards/receipts behind a cancellable actor API.

The control owner feeds Message envelopes into a bounded mailbox. The learning
thread resolves out-of-band frames and waits for sealed reward windows; it never
reads the control pipe or invokes OCR, input posting, or reset automation itself.
"""
from __future__ import annotations

from collections import deque, OrderedDict
from dataclasses import dataclass
import copy
import math
import threading
import time
import uuid
from typing import Callable

import numpy as np

from astra.protocol import Message
from astra.recordings import validate_frame, validate_event
from .interface import (DecisionContext, EnvironmentError, EnvironmentObservation, EnvironmentSpec,
                        EnvironmentTransition, SurfaceObservation, controls, identifier, integer)
from .interface import owned_bgra, fields, uuid_key, same_id
from .evidence import DecisionEvidence as _Window, OUTCOMES as _OUTCOMES
from .observation_transport import ResolvedFrame, FrameRingResolver, SnapshotDecoder, json_geometry

_KINDS = {'environment.ready', 'environment.observation', 'environment.reward', 'environment.outcome',
          'environment.watermark', 'environment.receipt', 'environment.stopped', 'environment.fault'}



class ExternalEnvironment:
    """One run/clock, one active decision, bounded delayed execution receipts.

    emit(kind, payload, request_id, run_id) MUST enqueue into the owner's bounded
    nonblocking transport; that owner assigns its global sender sequence. A
    frame resolver must return an owned compact BGRA array and exact lease ack.
    All OS activity and target/reward/reset semantics remain in the peer.
    """
    def __init__(self, spec: EnvironmentSpec, *, run_id: str, clock_id: str,
                 emit: Callable, resolve_frame: Callable[[dict, dict], ResolvedFrame],
                 timeout_seconds: float = 5, reset_timeout_seconds: float = 300,
                 maximum_mailbox_bytes: int = 8 * 1024**2):
        self.spec = spec.validate()
        if not spec.reward_signature or not spec.reset_signature:
            raise EnvironmentError('External reward and reset programs require immutable fingerprints')
        self.run_id, self.clock_id = identifier(run_id), identifier(clock_id)
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 3600
               for value in (timeout_seconds, reset_timeout_seconds)):
            raise EnvironmentError('Environment transport waits require finite bounded timeouts')
        integer(maximum_mailbox_bytes, 64 * 1024**2, 1024)
        self.timeout_seconds, self.reset_timeout_seconds = timeout_seconds, reset_timeout_seconds
        self._emit_callback, self._resolve_frame = emit, resolve_frame
        self._snapshot_decoder=SnapshotDecoder(spec,resolve_frame,lambda obs,acks: self._emit(
            'environment.framesConsumed',{'observationID':obs,'acknowledgements':acks},obs))
        self._condition = threading.Condition()
        self._mailbox = deque()
        self._mailbox_bytes = 0
        self._maximum_mailbox_bytes = maximum_mailbox_bytes
        self._last_sequence = -1
        self._failure = None
        self._windows: dict[str, _Window] = {}
        self._completed = OrderedDict()
        self._ready_request = self._stop_request = None
        self._ready = None
        self._current = None
        self._episode = None
        self._episode_start = self._watermark = 0
        self._episode_step = self._packet_sequence = 0
        self._last_event_sequence = None
        self._observations = set()
        self._outcome = 'aborted'
        self._cleanup_confirmed = True
        self._boundary_nanos = None
        self._geometry = None
        self._maximum_pending = math.ceil(spec.lead_ms / spec.period_ms) + 3

    @property
    def signature(self):
        return self.spec.signature

    @property
    def outcome(self):
        return self._outcome

    @property
    def cleanup_confirmed(self):
        return self._cleanup_confirmed

    def receive(self, message: Message) -> bool:
        """Nonblocking control-thread delivery. Rejection wakes the actor waiter."""
        try:
            size = len(message.encode())
            if message.kind not in _KINDS or not same_id(message.run_id, self.run_id):
                raise EnvironmentError('External event has a different run or unsupported kind')
            with self._condition:
                if self._failure is not None and message.kind != 'environment.stopped': return False
                if message.sequence <= self._last_sequence:
                    raise EnvironmentError('External sender sequence repeated or moved backwards')
                if len(self._mailbox) >= 128 or self._mailbox_bytes + size > self._maximum_mailbox_bytes:
                    raise EnvironmentError('External environment exceeded its bounded mailbox')
                self._last_sequence = message.sequence
                self._mailbox.append((copy.deepcopy(message), size))
                self._mailbox_bytes += size
                self._condition.notify_all()
            return True
        except Exception as error:
            self.disconnect(str(error))
            return False

    def disconnect(self, reason='External environment disconnected'):
        with self._condition:
            if self._failure is None:
                self._failure = EnvironmentError(reason)
                self._mailbox.clear(); self._mailbox_bytes = 0
            self._condition.notify_all()

    def _emit(self, kind, payload, request):
        self._emit_callback(kind, {'clockID': self.clock_id, **payload}, request, self.run_id)

    def _wait(self, predicate, cancelled, timeout, *, cleanup=False):
        deadline = time.monotonic() + timeout
        while True:
            if cancelled(): raise InterruptedError('External environment wait cancelled')
            if time.monotonic() >= deadline:
                raise EnvironmentError('External environment timed out before its required evidence arrived')
            with self._condition:
                if self._failure is not None and not cleanup: raise self._failure
                item = self._mailbox.popleft() if self._mailbox else None
                if item is not None: self._mailbox_bytes -= item[1]
            if item is not None:
                try:
                    self._dispatch(item[0])
                except Exception as error:
                    self.disconnect(str(error))
                    raise
                continue
            if predicate(): return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EnvironmentError('External environment timed out before its required evidence arrived')
            with self._condition:
                self._condition.wait(min(remaining, .05))

    def _observation(self, value, *, episode, expected_cutoff=None):
        return self._snapshot_decoder.decode(value, episode=episode, expected_cutoff=expected_cutoff)

    def _dispatch(self, message):
        value = message.payload
        if not same_id(value.get('clockID'), self.clock_id):
            raise EnvironmentError('External event uses a different monotonic clock identity')
        if message.kind == 'environment.fault':
            raise EnvironmentError(str(value.get('reason', 'External environment reported a fault'))[:2048])
        if message.kind == 'environment.stopped':
            fields(value, ('clockID', 'episodeID', 'controlsReleased', 'pendingPackets'))
            if self._stop_request is None or not same_id(message.request_id, self._stop_request) or not same_id(value['episodeID'], self._episode) or value['controlsReleased'] is not True or type(value['pendingPackets']) is not int or value['pendingPackets'] != 0:
                raise EnvironmentError('Stop acknowledgement did not prove cleanup of this episode')
            self._cleanup_confirmed = True
            self._stop_request = None
            return
        if message.kind == 'environment.ready':
            fields(value, ('clockID', 'episodeID', 'environmentSignature', 'controlsReleased', 'pendingPackets', 'observation'))
            if self._ready_request is None or not same_id(message.request_id, self._ready_request) or self._ready is not None:
                raise EnvironmentError('Reset readiness is unsolicited, duplicated or stale')
            episode = identifier(value['episodeID'])
            if same_id(episode, self._episode) or value['environmentSignature'] != self.signature or value['controlsReleased'] is not True or type(value['pendingPackets']) is not int or value['pendingPackets'] != 0:
                raise EnvironmentError('Reset readiness lacks a fresh identity, stable configuration or cleared ledger')
            observation, events = self._observation(value['observation'], episode=episode)
            if observation.cutoff_nanos < self._watermark:
                raise EnvironmentError('Reset readiness moved the host monotonic clock backwards')
            if events or observation.control_state['keys'] or observation.control_state['buttons']:
                raise EnvironmentError('Reset readiness requires cleared controls and empty episode input history')
            self._ready = observation
            return
        if not same_id(value.get('episodeID'), self._episode):
            raise EnvironmentError('External event belongs to a stale episode')
        if message.kind == 'environment.watermark':
            fields(value, ('clockID', 'episodeID', 'throughNanos'))
            through = integer(value['throughNanos'])
            if through < self._watermark:
                raise EnvironmentError('Reward/outcome watermark moved backwards')
            self._watermark = through
            for window in self._windows.values():
                if window.end is None and window.context.cutoff_nanos + self.spec.period_ms * 1000000 <= through:
                    raise EnvironmentError('Watermark sealed a missing reward/outcome window')
                if window.end is not None and window.end <= through and (window.reward is None or window.outcome is None or window.outcome == 'unknown'):
                    raise EnvironmentError('Sealed reward/outcome window is unknown; it cannot enter PPO')
            return
        key = uuid_key(message.request_id)
        window = self._windows.get(key) or self._completed.get(key)
        if window is None:
            raise EnvironmentError('External event has no matching admitted decision identity')
        if window.assembled and message.kind != 'environment.receipt':
            raise EnvironmentError('A sealed decision cannot be revised or observed twice')
        if message.kind == 'environment.observation':
            fields(value, ('clockID', 'episodeID', 'observation'))
            if window.observation is not None: raise EnvironmentError('Decision observation was duplicated')
            window.observation = copy.deepcopy(value['observation'])
        elif message.kind in ('environment.reward', 'environment.outcome'):
            boundary = window.accept_label(message.kind, value, period_ms=self.spec.period_ms, watermark=self._watermark)
            if boundary is not None:
                if self._boundary_nanos is not None and self._boundary_nanos != boundary:
                    raise EnvironmentError('Episode outcome cutoffs disagree')
                self._boundary_nanos = boundary
        elif message.kind == 'environment.receipt':
            window.accept_receipt(value, lead_ms=self.spec.lead_ms)

    def _final_receipt(self, window):
        return window.execution_complete(self._boundary_nanos)

    def reset(self, *, seed, cancelled):
        if cancelled(): raise InterruptedError('External reset cancelled before request')
        if self._failure is not None: raise self._failure
        integer(seed, 2**63 - 1)
        if self._outcome == 'continuing': raise EnvironmentError('Reset requires a confirmed episode boundary')
        self.seal_episode(cancelled=cancelled)
        self._ready_request, self._ready = str(uuid.uuid4()), None
        self._snapshot_decoder.reset()
        self._emit('environment.reset', {'environmentSignature': self.signature, 'previousEpisodeID': self._episode,
                   'seed': seed if self.spec.seeded_reset else None}, self._ready_request)
        self._cleanup_confirmed = False
        self._wait(lambda: self._ready is not None, cancelled, self.reset_timeout_seconds)
        self._current = self._ready
        self._ready = None; self._ready_request = None
        self._episode = self._current.episode_id
        self._episode_start = self._watermark = self._current.cutoff_nanos
        self._episode_step = 0
        self._completed.clear()
        self._boundary_nanos = None
        self._outcome = 'continuing'
        self._cleanup_confirmed = True
        return self._current

    def step(self, commands, *, context: DecisionContext, cancelled):
        if cancelled(): raise InterruptedError('External decision cancelled before action admission')
        if self._failure is not None: raise self._failure
        context.validate()
        if self._outcome != 'continuing' or self._current is None or not all((
            same_id(context.run_id, self.run_id), same_id(context.episode_id, self._episode),
            same_id(context.observation_id, self._current.id), context.cutoff_nanos == self._current.cutoff_nanos,
            context.geometry_revision == self._current.geometry_revision, context.episode_step == self._episode_step)):
            raise EnvironmentError('Policy action does not describe the current episode observation')
        validate_commands(commands, self.spec, self._current)
        if self._episode_step >= 65536:
            raise EnvironmentError('External episode exceeded the bounded observation identity history')
        for key in list(self._windows):
            window = self._windows[key]
            if window.assembled and self._final_receipt(window):
                self._completed[key] = self._windows.pop(key)
                while len(self._completed) > self._maximum_pending + 16:
                    _, retired = self._completed.popitem(last=False)
                    if not retired.admission_seen:
                        raise EnvironmentError('Admission acknowledgement exceeded bounded reordering history')
        if len(self._windows) >= self._maximum_pending:
            raise EnvironmentError('Execution receipts fell behind the bounded action lookahead')
        window = _Window(context, self._packet_sequence, tuple(copy.deepcopy(commands)))
        self._packet_sequence += 1
        if uuid_key(context.packet_id) in self._windows or uuid_key(context.packet_id) in self._completed:
            raise EnvironmentError('Policy packet identity was reused')
        self._windows[uuid_key(context.packet_id)] = window
        self._cleanup_confirmed = False
        self._emit('environment.action', {'episodeID': self._episode, 'episodeStep': context.episode_step,
            'policyID': context.policy_id, 'decisionNanos': context.cutoff_nanos,
            'packet': {'id': context.packet_id, 'runID': self.run_id, 'sequence': window.sequence,
                'observationID': context.observation_id, 'geometryRevision': context.geometry_revision,
                'executeAtNanos': context.cutoff_nanos + self.spec.lead_ms * 1000000,
                'durationMs': self.spec.period_ms, 'commands': list(window.commands)}}, context.packet_id)
        def complete():
            return window.observation is not None and window.end is not None and window.end <= self._watermark and (
                window.admitted or (window.receipt is not None and self._final_receipt(window)))
        self._wait(complete, cancelled, self.timeout_seconds)
        if window.reward is None or window.outcome not in _OUTCOMES:
            raise EnvironmentError('Unknown sealed reward/outcome cannot enter the learner')
        if window.outcome == 'aborted': raise EnvironmentError(window.reason or 'External episode was aborted')
        observation, events = self._observation(window.observation, episode=self._episode, expected_cutoff=window.end)
        if window.end - context.cutoff_nanos <= 0 or (window.end - context.cutoff_nanos) % 1000000:
            raise EnvironmentError('Decision duration is not a positive integer number of milliseconds')
        if window.end - self._episode_start >= self.spec.maximum_episode_ms * 1000000 and window.outcome == 'continuing':
            raise EnvironmentError('The peer exceeded its configured episode bound without an outcome')
        self._current = observation
        self._outcome = _OUTCOMES[window.outcome]
        self._episode_step += 1
        window.assembled = True
        window.observation = None  # Retain only bounded receipt/identity state.
        return EnvironmentTransition(observation, window.reward, (window.end - context.cutoff_nanos) // 1000000,
            self._outcome, events, reason=window.reason, outcome_detail=window.outcome)

    def seal_episode(self, *, cancelled):
        if not self._windows and not self._completed: return
        if self._outcome == 'continuing': raise EnvironmentError('Cannot seal an ongoing external episode')
        self._wait(lambda: all(window.assembled and window.admission_seen and self._final_receipt(window)
                              for window in (*self._windows.values(), *self._completed.values())),
                   cancelled, self.timeout_seconds)
        self._windows.clear()
        self._completed.clear()

    def abort(self, reason):
        active = self._outcome == 'continuing'
        self._outcome = 'aborted'
        if not active and self._cleanup_confirmed and not self._windows and self._ready_request is None: return
        self._stop_request = str(uuid.uuid4())
        self._cleanup_confirmed = False
        try:
            self._emit('environment.abort', {'episodeID': self._episode, 'reason': str(reason)[:2048]}, self._stop_request)
            self._wait(lambda: self._cleanup_confirmed, lambda: False, self.timeout_seconds, cleanup=True)
        finally:
            self._windows.clear(); self._completed.clear(); self._ready_request = None; self._current = None
            self._ready = None; self._snapshot_decoder.reset()


def validate_commands(commands, spec, observation):
    if type(commands) is not list or len(commands) > 64:
        raise EnvironmentError('Policy action exceeds the bounded command capacity')
    previous = 0
    modes = set()
    surfaces = {frame.metadata['surface']['id'] for frame in observation.frames}
    for command in commands:
        operation = command.get('operation') if type(command) is dict else None
        argument_names = {'keyDown': ('keyCode',), 'keyUp': ('keyCode',), 'keyRepeat': ('keyCode',),
                          'buttonDown': ('button',), 'buttonUp': ('button',),
                          'pointerAbsolute': ('surfaceID', 'x', 'y'), 'pointerRelative': ('dx', 'dy'), 'scroll': ('dx', 'dy')}
        if operation not in argument_names: raise EnvironmentError('Unsupported policy action')
        fields(command, ('operation', 'offsetMs', *argument_names[operation]))
        offset = integer(command['offsetMs'], spec.period_ms)
        if offset < previous or (offset == spec.period_ms and operation not in ('pointerAbsolute', 'pointerRelative')):
            raise EnvironmentError('Policy command timing violates packet grammar')
        previous = offset
        if operation.startswith('key'):
            if integer(command['keyCode'], 127) not in spec.action_vocabulary.key_codes: raise EnvironmentError('Keyboard capability is unavailable')
        elif operation.startswith('button'):
            if integer(command['button'], 31) not in spec.action_vocabulary.mouse_buttons: raise EnvironmentError('Button capability is unavailable')
        elif operation == 'pointerAbsolute':
            if not spec.action_vocabulary.absolute_pointer or command['surfaceID'] not in surfaces or any(type(command[name]) not in (int, float) or not math.isfinite(command[name]) or not 0 <= command[name] < 1 for name in ('x', 'y')):
                raise EnvironmentError('Absolute pointer command is outside its observed surface')
            modes.add('absolute')
        else:
            if not (spec.action_vocabulary.relative_pointer if operation == 'pointerRelative' else spec.action_vocabulary.scroll):
                raise EnvironmentError('Motion capability is unavailable')
            scale = 1 if operation == 'pointerRelative' else spec.action_vocabulary.scroll_units_per_point
            if any(type(command[name]) not in (int, float) or not math.isfinite(command[name])
                   or not -32768 <= command[name] <= 32767 or not float(command[name] * scale).is_integer()
                   for name in ('dx', 'dy')):
                raise EnvironmentError('Motion command is outside its exact action representation')
            if operation == 'pointerRelative': modes.add('relative')
        if len(modes) > 1: raise EnvironmentError('Absolute and relative trajectories cannot share a packet')
