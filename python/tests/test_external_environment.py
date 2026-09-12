from __future__ import annotations

import copy
from dataclasses import replace
import queue
import threading
import uuid

import mlx.core as mx
import numpy as np
import pytest

from astra.environments.external import ExternalEnvironment, ResolvedFrame
from astra.environments.interface import EnvironmentSpec, DecisionContext, EnvironmentError
from astra.environments.practice import PracticeConfig, PracticeEnvironment
from astra.learning.reinforcement import ReinforcementConfig, ReinforcementTrainer, ObservationRecord
from astra.model.config import ModelConfig
from astra.model.policy import AgentPolicy
from astra.protocol import Message


class Peer:
    """Actual virtual actions on a separate thread, serialized control messages.

    This fixture models producer ownership and delayed execution receipts. No
    oracle labels or macOS APIs are used; pixels stay outside JSON envelopes.
    """
    def __init__(self, *, mode='', surfaces=1, timeout=1):
        self.mode = mode
        self.environment = PracticeEnvironment(PracticeConfig(pixel_width=64, pixel_height=64,
            logical_bounds=(0, 0, 64, 64), time_limit_ms=350, task='delayed_memory'))
        self.spec = EnvironmentSpec('threaded-fixture', self.environment.action_vocabulary,
            maximum_episode_ms=350, maximum_observation_bytes=surfaces * 64 * 64 * 4,
            maximum_surfaces=surfaces, seeded_reset=True, reward_signature='1' * 64, reset_signature='2' * 64,
            maximum_frame_age_ms=50 if mode == 'stale_frame' else 250)
        self.surface_count = surfaces
        self.run, self.clock = str(uuid.uuid4()), str(uuid.uuid4())
        self.queue = queue.Queue(maxsize=128)
        self.frames = {}
        self.actions = []
        self.receipts = []
        self.acknowledgements = []
        self.failures = []
        self.sequence = 0
        self.pending = {}
        self.reset_requested = threading.Event()
        self.adapter = ExternalEnvironment(self.spec, run_id=self.run, clock_id=self.clock,
            emit=lambda *message: self.queue.put_nowait(message), resolve_frame=self.resolve,
            timeout_seconds=timeout, reset_timeout_seconds=timeout)
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def resolve(self, reference, metadata):
        assert set(reference) == {'frameID'}
        original, pixels = self.frames[reference['frameID']]
        assert metadata == original
        return ResolvedFrame(pixels, {'frameID': reference['frameID']})

    def observation(self, observed, events=()):
        frames = []
        cutoff = observed.metadata['observedNanos']
        for index in range(self.surface_count):
            metadata = copy.deepcopy(observed.metadata)
            pixels = observed.pixels
            if index:
                metadata['id'] = str(uuid.uuid4())
                metadata['surface']['id'] = 'secondary'
                metadata['surface']['globalBounds']['x'] += 64
                pixels = np.ascontiguousarray(pixels[:, ::-1])
            coverage, kind = cutoff, 'frame'
            if self.mode == 'future_frame' and self.actions:
                metadata['eventNanos'] = metadata['observedNanos'] = cutoff + 1
            if self.mode in ('stale_frame', 'idle_frame') and self.actions:
                metadata['eventNanos'] = metadata['observedNanos'] = 1
                coverage = cutoff if self.mode == 'idle_frame' else 1
                kind = 'unchanged' if self.mode == 'idle_frame' else 'frame'
            if self.mode == 'lagged_frame' and self.actions:
                metadata['eventNanos'] -= 10000000
                metadata['observedNanos'] -= 10000000
                coverage = metadata['observedNanos']
            self.frames[metadata['id']] = (metadata, pixels)
            frames.append({'metadata': metadata, 'reference': {'frameID': metadata['id']},
                           'coverageNanos': coverage, 'coverageKind': kind})
        snapshot = {'id': str(uuid.uuid4()), 'episodeID': observed.episode_id, 'cutoffNanos': cutoff,
            'geometryRevision': 0, 'frames': frames, 'controlState': copy.deepcopy(observed.control_state), 'events': list(events)}
        if self.mode == 'wrong_cutoff' and self.actions:
            snapshot['cutoffNanos'] += 1
        return snapshot

    def send(self, kind, payload, request):
        payload = copy.deepcopy(payload)
        clock, run = self.clock, self.run
        if self.mode == 'native_uuid_case':
            clock, run, request = clock.upper(), run.upper(), request.upper()
            if payload.get('episodeID') is not None: payload['episodeID'] = payload['episodeID'].upper()
            if 'receipt' in payload:
                for key in ('packetID', 'runID'): payload['receipt'][key] = payload['receipt'][key].upper()
            if 'observation' in payload:
                for key in ('id', 'episodeID'): payload['observation'][key] = payload['observation'][key].upper()
        message = Message(kind, self.sequence, {'clockID': clock, **payload}, request_id=request, run_id=run)
        self.sequence += 1
        # Exercise the same finite JSON/integer round-trip as real pipes.
        return self.adapter.receive(Message.decode(message.encode()))

    def receipt(self, action, status, results=(), *, now=None, cause=None):
        packet = action['packet']
        control = self.environment._observe().control_state
        payload = {'episodeID': action['episodeID'], 'receipt': {'packetID': packet['id'], 'runID': self.run,
            'sequence': packet['sequence'], 'status': status, 'observedNanos': control['observedNanos'] if now is None else now,
            'commandResults': list(results), 'resultingState': control}}
        if cause is not None: payload['cancellationCause'] = cause
        self.receipts.append((packet['id'], status))
        self.send('environment.receipt', payload, packet['id'])

    def work(self):
        while True:
            item = self.queue.get()
            if item is None: return
            kind, payload, request, run = item
            try:
                assert payload['clockID'] == self.clock and run == self.run
                if kind == 'environment.framesConsumed':
                    for ack in payload['acknowledgements']:
                        self.acknowledgements.append(ack['frameID'])
                        del self.frames[ack['frameID']]
                elif kind == 'environment.reset':
                    self.reset_requested.set()
                    observed = self.environment.reset(seed=payload['seed'])
                    self.pending.clear()
                    if self.mode == 'hold_ready': continue
                    ready = {'episodeID': observed.episode_id, 'environmentSignature': self.adapter.signature,
                        'controlsReleased': True, 'pendingPackets': 0, 'observation': self.observation(observed)}
                    if self.mode == 'bad_ready': ready['controlsReleased'] = False
                    self.send('environment.ready', ready, request)
                elif kind == 'environment.action':
                    self.actions.append(copy.deepcopy(payload))
                    local_sequence = self.environment._packet_sequence
                    self.pending[local_sequence] = [payload, []]
                    self.receipt(payload, 'admitted')
                    if self.mode == 'hold_window': continue
                    assert uuid.UUID(payload['episodeID']) == uuid.UUID(self.environment.episode_id)
                    result = self.environment.step(payload['packet']['commands'], episode_id=self.environment.episode_id)
                    for command in result.command_results:
                        self.pending[command['packetSequence']][1].append({key: value for key, value in command.items()
                            if key in {'commandIndex', 'scheduledNanos', 'postedNanos', 'status', 'message'}})
                    observed = self.observation(result.observation, result.raw_events)
                    episode, start, end = result.observation.episode_id, payload['decisionNanos'], result.observation.metadata['observedNanos']
                    self.send('environment.observation', {'episodeID': episode, 'observation': observed}, request)
                    outcome = result.outcome if self.mode != 'unknown_outcome' else 'unknown'
                    self.send('environment.outcome', {'episodeID': episode, 'endNanos': end, 'outcome': outcome}, request)
                    reward = result.reward if self.mode != 'unknown_reward' else None
                    reward_start = start + 100000000 if self.mode == 'shifted_reward' else start
                    # Pending unknown and later known value are distinct events.
                    self.send('environment.reward', {'episodeID': episode, 'startNanos': reward_start, 'endNanos': end, 'value': None}, request)
                    self.send('environment.reward', {'episodeID': episode, 'startNanos': reward_start, 'endNanos': end, 'value': reward}, request)
                    if self.mode != 'no_watermark':
                        self.send('environment.watermark', {'episodeID': episode, 'throughNanos': end}, request)
                    for key, (action, results) in list(self.pending.items()):
                        packet = action['packet']
                        done = len(results) == len(packet['commands'])
                        ended = result.outcome != 'continuing'
                        if done and (packet['executeAtNanos'] + packet['durationMs'] * 1000000 <= end or ended):
                            cancelled = any(item['status'] == 'cancelled' for item in results)
                            status = 'cancelled' if cancelled else 'executed'
                            cause = 'episodeBoundary' if cancelled else None
                            if self.mode == 'late_receipt': status = 'late'
                            self.receipt(action, status, sorted(results, key=lambda value: value['commandIndex']), cause=cause)
                            del self.pending[key]
                elif kind == 'environment.abort':
                    if self.environment.outcome == 'continuing': self.environment.abort('Fixture peer stopped')
                    self.pending.clear(); self.frames.clear()
                    self.send('environment.stopped', {'episodeID': payload['episodeID'], 'controlsReleased': True,
                                                     'pendingPackets': 0}, request)
                else:
                    raise AssertionError(kind)
            except Exception as error:
                self.failures.append(error)
                self.adapter.disconnect('Fixture peer failed: ' + str(error))

    def close(self):
        try:
            self.adapter.abort('Fixture finished')
        finally:
            self.queue.put(None); self.thread.join(3)
            assert not self.thread.is_alive()


def context(adapter, observed, step=0):
    return DecisionContext(adapter.run_id, observed.episode_id, 'fixture-policy', observed.id,
                           str(uuid.uuid4()), step, observed.cutoff_nanos, observed.geometry_revision)


@pytest.mark.parametrize('mode', ['', 'native_uuid_case'])
def test_external_peer_runs_real_ppo_and_complete_episodes_with_delayed_receipts(tmp_path, mode):
    peer = Peer(mode=mode)
    try:
        mx.random.seed(92)
        policy = AgentPolicy(ModelConfig.test_small(), peer.spec.action_vocabulary)
        trainer = ReinforcementTrainer(policy, peer.adapter,
            ReinforcementConfig(rollout_decisions=3, sequence_length=2, burn_in=1, epochs=1, effective_batch_decisions=8),
            scratch_directory=tmp_path)
        collected = trainer.collect()
        assert len(collected.decisions) == 4 and collected.rollout.run_id == peer.run
        assert [item.transition.packet_id for item in collected.decisions] == [item['packet']['id'] for item in peer.actions]
        assert all(item.observation.observation_id != item.observation.metadata['id'] for item in collected.decisions)
        assert trainer.verify_behavior(collected)[0] < 2e-4
        result = trainer.update(collected)
        assert result.optimizer_updates > 0 and result.maximum_accepted_kl <= .02
        assert result.decisions == 4 and not list(tmp_path.glob('.astra-rollout-*'))
        pending = trainer.pending_policy_id
        next_rollout = trainer.collect()
        assert next_rollout.rollout.policy_id == pending
        trainer.discard_rollout(next_rollout); trainer.stop()
        assert peer.adapter.cleanup_confirmed and not peer.pending
        assert not peer.failures
    finally:
        peer.close()


def test_next_decision_does_not_wait_for_execution_of_its_future_packet():
    peer = Peer()
    try:
        observed = peer.adapter.reset(seed=0, cancelled=lambda: False)
        first = context(peer.adapter, observed)
        result = peer.adapter.step([], context=first, cancelled=lambda: False)
        assert (first.packet_id, 'executed') not in peer.receipts
        assert result.observation.cutoff_nanos - observed.cutoff_nanos == 100000000
        for step in (1, 2, 3):
            result = peer.adapter.step([], context=context(peer.adapter, result.observation, step), cancelled=lambda: False)
        peer.adapter.seal_episode(cancelled=lambda: False)
        assert (first.packet_id, 'executed') in peer.receipts
    finally:
        peer.close()


@pytest.mark.parametrize('mode,reason', [('unknown_reward', 'unknown'), ('unknown_outcome', 'unknown'),
    ('future_frame', 'clocks'), ('stale_frame', 'stale'), ('wrong_cutoff', 'cutoff'), ('shifted_reward', 'begin'), ('late_receipt', 'late')])
def test_bad_or_unknown_transition_evidence_never_becomes_on_policy_data(mode, reason, tmp_path):
    peer = Peer(mode=mode)
    try:
        trainer = ReinforcementTrainer(AgentPolicy(ModelConfig.test_small(), peer.spec.action_vocabulary), peer.adapter,
            ReinforcementConfig(rollout_decisions=2, sequence_length=2, burn_in=1, epochs=1), scratch_directory=tmp_path)
        with pytest.raises(EnvironmentError, match=reason): trainer.collect()
        assert trainer.iteration == trainer.decisions == trainer.optimizer_updates == 0
        assert peer.adapter.cleanup_confirmed and not list(tmp_path.glob('.astra-rollout-*'))
    finally:
        peer.close()


def test_independent_cutoff_preserves_lagged_frame_source_time_and_multiple_surfaces():
    peer = Peer(mode='lagged_frame', surfaces=2)
    try:
        initial = peer.adapter.reset(seed=0, cancelled=lambda: False)
        result = peer.adapter.step([], context=context(peer.adapter, initial), cancelled=lambda: False)
        assert result.observation.cutoff_nanos > result.observation.frames[0].metadata['observedNanos']
        record = ObservationRecord.capture(result.observation, elapsed_seconds=.1, reset=False)
        batch = record.prepare(ModelConfig.test_small(), ())
        assert len(batch.surfaces) == 2
        assert record.cutoff_nanos == result.observation.cutoff_nanos
        assert len(record.images) == 2
    finally:
        peer.close()


def test_cancellation_while_waiting_for_readiness_joins_peer_cleanup(tmp_path):
    peer = Peer(mode='hold_ready')
    cancel = threading.Event()
    try:
        def trigger():
            assert peer.reset_requested.wait(2)
            cancel.set()
        helper = threading.Thread(target=trigger); helper.start()
        trainer = ReinforcementTrainer(AgentPolicy(ModelConfig.test_small(), peer.spec.action_vocabulary), peer.adapter,
            ReinforcementConfig(rollout_decisions=2), scratch_directory=tmp_path)
        with pytest.raises(InterruptedError): trainer.collect(cancelled=cancel.is_set)
        helper.join(2)
        assert peer.adapter.cleanup_confirmed and peer.environment.outcome == 'aborted'
        assert not peer.actions and not list(tmp_path.glob('.astra-rollout-*'))
    finally:
        peer.close()


def test_protocol_sequence_and_mailbox_limits_fail_without_accepting_actions():
    spec = EnvironmentSpec('bounded', PracticeEnvironment(PracticeConfig()).action_vocabulary, reward_signature='1' * 64, reset_signature='2' * 64)
    adapter = ExternalEnvironment(spec, run_id=str(uuid.uuid4()), clock_id=str(uuid.uuid4()),
        emit=lambda *_: None, resolve_frame=lambda *_: None, timeout_seconds=.01, maximum_mailbox_bytes=1024)
    message = Message('environment.fault', 0, {'clockID': adapter.clock_id, 'reason': 'x'}, run_id=adapter.run_id)
    assert adapter.receive(message)
    assert not adapter.receive(message)
    with pytest.raises(EnvironmentError, match='sequence'):
        adapter.reset(seed=0, cancelled=lambda: False)


def test_known_labels_without_producer_watermark_do_not_complete_a_transition(tmp_path):
    peer = Peer(mode='no_watermark', timeout=.15)
    try:
        trainer = ReinforcementTrainer(AgentPolicy(ModelConfig.test_small(), peer.spec.action_vocabulary), peer.adapter,
            ReinforcementConfig(rollout_decisions=2), scratch_directory=tmp_path)
        with pytest.raises(EnvironmentError, match='timed out'): trainer.collect()
        assert trainer.optimizer_updates == trainer.decisions == 0
        assert peer.adapter.cleanup_confirmed and not list(tmp_path.glob('.astra-rollout-*'))
    finally:
        peer.close()


def test_verified_unchanged_coverage_keeps_source_timestamps_instead_of_faking_new_frames():
    peer = Peer(mode='idle_frame')
    try:
        observed = peer.adapter.reset(seed=0, cancelled=lambda: False)
        for step in range(3):
            observed = peer.adapter.step([], context=context(peer.adapter, observed, step), cancelled=lambda: False).observation
        assert observed.cutoff_nanos > 250000000
        assert observed.frames[0].metadata['eventNanos'] == 1
        assert observed.frames[0].coverage_nanos == observed.cutoff_nanos
    finally:
        peer.close()


def test_cancel_before_send_does_not_submit_another_policy_packet():
    peer = Peer()
    try:
        observed = peer.adapter.reset(seed=0, cancelled=lambda: False)
        with pytest.raises(InterruptedError):
            peer.adapter.step([], context=context(peer.adapter, observed), cancelled=lambda: True)
        assert not peer.actions
    finally:
        peer.close()


def test_external_spec_identity_pins_programs_and_rejects_silent_default_changes():
    peer = Peer()
    try:
        value = peer.spec.to_dict()
        assert EnvironmentSpec.from_dict(value).signature == peer.adapter.signature
        changed = {**value, 'reward_signature': '3' * 64}
        assert EnvironmentSpec.from_dict(changed).signature != peer.adapter.signature
        del value['maximum_frame_age_ms']
        with pytest.raises(EnvironmentError, match='every versioned field'):
            EnvironmentSpec.from_dict(value)
    finally:
        peer.close()
