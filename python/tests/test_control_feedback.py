from __future__ import annotations

import copy
from dataclasses import replace
import json
import uuid

import pytest

from astra.control_feedback import (ControlFeedbackError, validate_control_feedback,
    validate_feedback_continuation, validate_control_exclusion, exclusion_covers)
from astra.environments.practice import PracticeConfig, PracticeEnvironment, PracticeError
from astra.environments.practice_adapter import PracticeAdapter
from astra.environments.interface import DecisionContext, EnvironmentError
from astra.environments.observation_transport import SnapshotDecoder, ResolvedFrame
from astra.learning.reinforcement import ObservationRecord
from astra.learning.rollout_artifacts import _observation
from astra.learning.rollout_store import FrameSpool


def world(**changes):
    return PracticeEnvironment(PracticeConfig(pixel_width=32, pixel_height=32,
        task='delayed_memory', **changes)).enable_control_feedback()


def relative(dx, dy, offset):
    return {'operation': 'pointerRelative', 'offsetMs': offset, 'dx': dx, 'dy': dy}


def test_empty_packet_lifetime_is_half_open_and_does_not_block_oracle():
    env = world(lead_ms=100, cue_ms=1, delay_ms=1)
    initial = env.reset(42)
    assert initial.control_feedback['packets'] == []
    first = env.step([]).observation.control_feedback
    assert len(first['packets']) == 1 and 'terminal' not in first['packets'][0]
    assert env._pending == [] and env.oracle_commands()  # Ends are not pending commands.
    second = env.step([]).observation.control_feedback
    assert 'terminal' not in second['packets'][0]  # End exactly at cutoff is still pending.
    third = env.step([]).observation.control_feedback
    assert third['packets'][0]['terminal']['status'] == 'executed'
    fourth = env.step([]).observation.control_feedback
    assert all(row['packet']['sequence'] != 0 for row in fourth['packets'])
    assert 'terminal' not in first['packets'][0]  # Later commits cannot rewrite saved input.
    for before, after in zip((initial.control_feedback, first, second, third), (first, second, third, fourth)):
        validate_feedback_continuation(before, after)


def test_partial_relative_progress_keeps_original_anchor_and_pre_cleanup_bootstrap():
    env = world(lead_ms=150, time_limit_ms=200, relative_pointer=True)
    initial = env.reset(4)
    commands = [relative(0, 0, 0), {'operation': 'buttonUp', 'offsetMs': 20, 'button': 0}, relative(101, -55, 100)]
    first = env.step(commands).observation.control_feedback
    assert all(row['completedSampleCount'] == 0 for row in first['packets'][0]['progress'])
    result = env.step([])
    feedback = result.observation.control_feedback
    assert result.outcome == 'truncated' and not env._pending and not env._feedback_ends
    row = feedback['packets'][0]
    assert row['packet']['commands'] == commands
    assert row['progress'][0]['status'] == 'posted'
    assert row['progress'][1]['status'] == 'noOp'
    progress = row['progress'][2]
    assert progress['status'] == 'partial' and progress['completedSampleCount'] == 49
    assert progress['lastCompletedOffsetMs'] == 49
    assert (progress['emittedDx'], progress['emittedDy']) == (round(101 * .49), round(-55 * .49))
    assert feedback['coverageNanos'] == result.observation.metadata['observedNanos']
    assert row.get('terminal') is None  # Cleanup cancellation is later, never bootstrap input.
    assert env._feedback.snapshot(env._now)['unavailableReason'] == 'stopping'
    validate_control_feedback(feedback, cutoff_nanos=feedback['cutoffNanos'], require_available=True)
    validate_feedback_continuation(first, feedback)


def test_actual_context_packet_and_observation_ids_survive_admission():
    adapter = PracticeAdapter(PracticeEnvironment(PracticeConfig(pixel_width=32, pixel_height=32)), control_feedback=True)
    before = adapter.reset(seed=3, cancelled=lambda: False)
    context = DecisionContext(adapter.run_id, before.episode_id, str(uuid.uuid4()), before.id,
                              str(uuid.uuid4()), 0, before.cutoff_nanos, before.geometry_revision)
    # Frame identity is also the adapter observation identity, and context is checked before admission.
    wrong = replace(context, observation_id=str(uuid.uuid4()))
    with pytest.raises(PracticeError, match='bind'):
        adapter.step([], context=wrong, cancelled=lambda: False)
    assert adapter.environment._packet_sequence == 0
    result = adapter.step([], context=context, cancelled=lambda: False)
    packet = result.observation.control_feedback['packets'][0]['packet']
    assert packet['id'] == context.packet_id and packet['observationID'] == before.id
    assert packet['runID'] == adapter.run_id and result.observation.observation_schema_version == 2


@pytest.mark.parametrize('corrupt', [
    lambda value: value.update(coverageNanos=value['cutoffNanos'] + 1),
    lambda value: value['packets'][0]['progress'][0].update(completedSampleCount=True),
    lambda value: value['changes'][0].update(sequence=1),
    lambda value: value['packets'][0].update(admittedNanos=value['cutoffNanos'] + 1),
    lambda value: value['packets'][0]['progress'][0].update(status='posted'),
    lambda value: value['packets'][0]['packet']['commands'][0].update(keyCode=12),
    lambda value: value.update(extra='not part of the protocol'),
])
def test_supplied_corrupt_feedback_never_becomes_empty(corrupt):
    env = world()
    env.reset(1)
    value = env.step([{'operation': 'buttonUp', 'offsetMs': 0, 'button': 0}]).observation.control_feedback
    corrupt(value)
    with pytest.raises(ControlFeedbackError):
        validate_control_feedback(value, cutoff_nanos=value['cutoffNanos'], require_available=True)


def test_unavailable_and_known_empty_are_distinct_and_dropped_outstanding_plan_fails():
    env = world()
    empty = env.reset(1).control_feedback
    unavailable = {**empty, 'unavailableReason': 'postInFlight'}
    unavailable.pop('coverageNanos')
    validate_control_feedback(unavailable, cutoff_nanos=empty['cutoffNanos'])
    with pytest.raises(ControlFeedbackError, match='unavailable'):
        validate_control_feedback(unavailable, cutoff_nanos=empty['cutoffNanos'], require_available=True)
    before = env.step([]).observation.control_feedback
    after = env.step([]).observation.control_feedback
    corrupt = copy.deepcopy(after)
    corrupt['packets'].pop(0)
    validate_control_feedback(corrupt, cutoff_nanos=corrupt['cutoffNanos'])
    with pytest.raises(ControlFeedbackError, match='disappeared'):
        validate_feedback_continuation(before, corrupt)
    assert env._feedback.snapshot(env._now) == env._feedback.snapshot(env._now)  # Polling does not acknowledge.
    with pytest.raises(ControlFeedbackError):
        env._feedback.acknowledge(after['throughSequence'] + 1)


def test_snapshot_rejects_feedback_mismatch_before_pixel_acquisition():
    adapter = PracticeAdapter(PracticeEnvironment(PracticeConfig(pixel_width=32, pixel_height=32)), control_feedback=True)
    initial = adapter.reset(seed=7, cancelled=lambda: False)
    frame = initial.frames[0]
    wire = {'id': initial.id, 'episodeID': initial.episode_id, 'cutoffNanos': initial.cutoff_nanos,
            'geometryRevision': 0, 'controlState': initial.control_state, 'events': [],
            'observationSchemaVersion': 2, 'controlFeedback': initial.control_feedback,
            'frames': [{'metadata': frame.metadata, 'reference': {}, 'coverageNanos': frame.coverage_nanos, 'coverageKind': 'frame'}]}
    copied = []
    def resolve(*_):
        copied.append(True)
        return ResolvedFrame(frame.pixels, {})
    bad = copy.deepcopy(wire)
    bad['controlFeedback']['cutoffNanos'] += 1
    with pytest.raises(ControlFeedbackError):
        SnapshotDecoder(adapter.spec, resolve, lambda *_: None).decode(bad, episode=initial.episode_id)
    assert not copied
    decoded, _ = SnapshotDecoder(adapter.spec, resolve, lambda *_: None).decode(wire, episode=initial.episode_id)
    assert decoded.control_feedback == wire['controlFeedback'] and decoded.observation_schema_version == 2
    wire['controlFeedback']['packets'].append({})
    assert decoded.control_feedback['packets'] == []


def test_recording_exclusion_requires_full_join_and_exact_identity():
    recording = str(uuid.uuid4())
    proof = {'schemaVersion': 1, 'ownershipID': str(uuid.uuid4()), 'recordingID': recording, 'startedNanos': 100}
    validate_control_exclusion(proof, recording_id=recording)
    assert not exclusion_covers(proof, 200)
    with pytest.raises(ControlFeedbackError):
        validate_control_exclusion({**proof, 'throughNanos': 300}, recording_id=recording)
    with pytest.raises(ControlFeedbackError):
        validate_control_exclusion({**proof, 'startedNanos': 2**63}, recording_id=recording)
    proof.update(throughNanos=300, producersJoinedNanos=350)
    validate_control_exclusion(proof, recording_id=recording)
    assert exclusion_covers(proof, 100) and exclusion_covers(proof, 300)
    assert not exclusion_covers(proof, 99) and not exclusion_covers(proof, 301)
    with pytest.raises(ControlFeedbackError):
        validate_control_exclusion(proof, recording_id=str(uuid.uuid4()))


def test_immutable_observation_record_retains_original_evidence_and_legacy_fields_stay_absent(tmp_path):
    adapter = PracticeAdapter(PracticeEnvironment(PracticeConfig(pixel_width=32, pixel_height=32)), control_feedback=True)
    initial = adapter.reset(seed=7, cancelled=lambda: False)
    spool = FrameSpool(directory=tmp_path, memory_bytes=2**20, disk_bytes=2**20)
    try:
        record = ObservationRecord.capture(initial, elapsed_seconds=.1, reset=True, spool=spool)
        archived = _observation(record)
        initial.control_feedback['changes'].append({'not': 'original'})
        assert json.loads(record.control_feedback_json)['changes'] == []
        assert archived['control_feedback']['changes'] == [] and archived['observation_schema_version'] == 2
        legacy = replace(record, control_feedback_json=None, observation_schema_version=1)
        assert 'control_feedback' not in _observation(legacy) and 'observation_schema_version' not in _observation(legacy)
    finally:
        spool.close()
