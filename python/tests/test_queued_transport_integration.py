"""Small real-model contracts, without a learning-quality or timing claim."""
from __future__ import annotations

from dataclasses import asdict, replace
import copy
import json
from pathlib import Path
import subprocess
import uuid

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest

from astra.checkpoints import save_checkpoint
from astra.environments.practice import PracticeConfig, PracticeEnvironment
from astra.learning.reinforcement import ReinforcementConfig, ReinforcementTrainer
from astra.learning.rollout_store import FrameSpool
from astra.learning.rollout_artifacts import publish_package, inspect_package, load_rollout
from astra.model.config import ModelConfig
from astra.model.policy import AgentPolicy
from astra.model.actions import ActionVocabulary
from astra.inference import InferenceSession
from test_frame_ring import _message
from test_inference import _prepare, _step, _acknowledge


def test_queue_aware_practice_archive_replay_and_one_ppo_update(tmp_path):
    mx.random.seed(181)
    env = PracticeEnvironment(PracticeConfig(task='delayed_memory', pixel_width=32, pixel_height=32,
        logical_bounds=(0, 0, 32, 32), time_limit_ms=250, lead_ms=150))
    model = replace(ModelConfig.test_small(), schema_version=3, lead_ms=150)
    policy = AgentPolicy(model, env.action_vocabulary)
    config = ReinforcementConfig(rollout_decisions=2, epochs=1, sequence_length=2, burn_in=1,
        effective_batch_decisions=4, checkpoint_vision=True)
    worker = ReinforcementTrainer(policy, env, config, policy_id=str(uuid.uuid4()), scratch_directory=tmp_path)
    collected = imported = None
    try:
        collected = worker.collect()
        assert len(collected.decisions) == 3
        assert all(item.observation.observation_schema_version == 2 for item in collected.decisions)
        prior_packet = json.loads(collected.decisions[1].observation.control_feedback_json)['packets'][0]['packet']
        assert prior_packet['id'] == collected.decisions[0].transition.packet_id
        assert prior_packet['commands'] == json.loads(collected.decisions[0].commands_json)
        final = collected.decisions[-1].bootstrap_observation
        assert final is not None and json.loads(final.control_feedback_json)['packets']
        assert json.loads(final.control_feedback_json).get('unavailableReason') is None
        assert env._feedback.snapshot(env._now)['unavailableReason'] == 'stopping'
        error, low, high = worker.verify_behavior(collected)
        assert error < 2e-4
        np.testing.assert_allclose([low, high], [1, 1], atol=2e-5)

        # Copy original immutable storage for the external archive codec; the
        # original practice rollout remains owned by its learner for one update.
        archive_store = FrameSpool(directory=tmp_path, memory_bytes=config.maximum_rollout_bytes,
                                   disk_bytes=config.maximum_rollout_disk_bytes)
        rows = tuple(replace(item, observation=item.observation.with_spool(archive_store),
            bootstrap_observation=None if item.bootstrap_observation is None else item.bootstrap_observation.with_spool(archive_store))
            for item in collected.decisions)
        archive_store.seal()
        archive = replace(collected, decisions=rows, spool=archive_store, actor_sampling=('categorical', 1.0, 'none', 1))
        binding = {'runID': worker._run_id, 'clockID': str(uuid.uuid4()), 'policyID': worker.actor_policy_id,
            'policySignature': '1' * 64, 'actorSourceID': str(uuid.uuid4()), 'environmentSourceID': str(uuid.uuid4()),
            'environment': worker._environment_adapter.spec.to_dict(), 'model': model.to_dict(), 'training': asdict(config),
            'contextIDs': [], 'purpose': 'learning', 'previousActorProgress': None}
        working = tmp_path / 'working'; working.mkdir()
        (working / 'journal.ndjson').write_bytes(b'')
        (working / 'audit.json').write_text('{}')
        destination = tmp_path / str(uuid.uuid4())
        publish_package(working, destination, binding=binding, status='sealed', actor_progress=None,
                        control_closure_known=True, rollout=archive)
        manifest = inspect_package(destination)
        assert manifest['schemaVersion'] == 1 and manifest['observationSchemaVersion'] == 2
        imported = load_rollout(destination, manifest, config)
        for before, after in zip(collected.decisions, imported.decisions):
            assert before.observation.control_feedback_json == after.observation.control_feedback_json
            assert before.observation.images[0].metadata_json == after.observation.images[0].metadata_json
        assert imported.decisions[-1].bootstrap_observation.control_feedback_json == final.control_feedback_json
        replay_error, replay_low, replay_high = worker.verify_behavior(imported)
        assert replay_error == pytest.approx(error, abs=1e-7)
        np.testing.assert_allclose([replay_low, replay_high], [low, high], atol=1e-7)
        parameters = {name: np.asarray(value).copy() for name, value in tree_flatten(policy.parameters())}
        metrics = worker.update(collected)
        assert metrics.optimizer_updates > 0 and np.isfinite(metrics.mean_loss)
        assert any(np.any(np.asarray(value) != parameters[name]) for name, value in tree_flatten(policy.parameters()))
    finally:
        if imported is not None: imported.close()
        if collected is not None and not collected.spool.closed: worker.discard_rollout(collected)
        worker.stop()


def _fixture_binary():
    root = Path(__file__).resolve().parents[2]
    binary = root / '.build/debug/AstraFixture'
    if not binary.is_file():
        subprocess.run([str(root / 'script/swift.sh'), 'build', '--product', 'AstraFixture'], cwd=root, check=True,
                       capture_output=True, timeout=120)
    return binary


def test_native_ring_compiled_queue_actor_warmup_and_exact_collection_echo(tmp_path):
    producer = subprocess.Popen([str(_fixture_binary()), '--frame-ring', str(tmp_path)],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    actor = InferenceSession()
    try:
        report = _message(producer); run = report['reference']['runID']
        checkpoint = tmp_path / str(uuid.uuid4())
        mx.random.seed(182)
        policy = AgentPolicy(replace(ModelConfig.test_small(), schema_version=3), ActionVocabulary())
        save_checkpoint(checkpoint, policy, kind='initial', step=0)
        ready = actor.prepare({**_prepare(checkpoint, report, deterministic=False), 'collection': True}, run_id=run)
        assert ready['collectionVersion'] == 2
        state = actor.reset({'confirmed': True, 'episodeID': str(uuid.uuid4()), 'contextIDs': []}, run_id=run)
        warm = actor.warmup({**_step(report, state), 'controlFeedback': None}, run_id=run)
        assert warm['warmup'] is True and 'collectionRecord' not in warm
        assert actor._feedback is None and actor._episode_step == actor._draw_index == 0
        assert actor._state is None and actor._state_id == state['stateID']
        _acknowledge(producer, warm['releasedFrames'][0]); report = _message(producer)
        request = _step(report, state)
        feedback = {'version': 1, 'controlEpochID': str(uuid.uuid4()), 'runID': run,
                    'geometryRevision': request['geometryRevision'], 'cutoffNanos': request['cutoffNanos'],
                    'coverageNanos': request['cutoffNanos'], 'changes': [], 'packets': []}
        request['controlFeedback'] = feedback
        result = actor.step(request, run_id=run)
        collection = result['collectionRecord']
        assert collection['schemaVersion'] == 2 and collection['controlFeedback'] == feedback
        assert collection['episodeStep'] == collection['sampler']['drawIndex'] == 0
        assert result['packet']['sequence'] == 0
        feedback['changes'].append({'not': 'original'})
        assert not collection['controlFeedback']['changes'] and not actor._feedback['changes']
        _acknowledge(producer, result['releasedFrames'][0])
        assert _message(producer)['closed']
        assert producer.wait(timeout=10) == 0
    finally:
        actor.close()
        if producer.poll() is None: producer.kill()
        producer.communicate(timeout=10)


def test_native_recording_queue_proof_is_independent_of_future_teacher_labels(tmp_path):
    import sqlite3
    from astra.data.datasets import build_dataset, DatasetReader, RecordingSelection
    from astra.model.queued_control import QueuedControlBatch
    generated = json.loads(subprocess.check_output([str(_fixture_binary()), str(tmp_path)]))
    source = Path(generated['directory'])
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    model = replace(ModelConfig.test_small(), schema_version=3, period_ms=20, lead_ms=20)
    vocabulary = ActionVocabulary((13, 14), (), True, False, True)

    def first_sample():
        destination = tmp_path / str(uuid.uuid4())
        result = build_dataset(destination, recording_root=source.parent,
            selections=[RecordingSelection(str(uuid.UUID(source.stem)))], config=model,
            vocabulary=vocabulary, pointer_mode='absolute')
        assert result['observationSchemaVersion'] == 2
        with DatasetReader(destination, recording_root=source.parent) as reader:
            # Native geometry changes split episodes; UUID sorting is not temporal order.
            episode_id = reader._database.execute('SELECT episode_id FROM steps ORDER BY cutoff LIMIT 1').fetchone()[0]
            sample = next(reader.samples(episode_id, count=1))
            queue = sample.observation.queued_control
            return (sample.commands, np.asarray(sample.observation.controls).copy(),
                    {name: np.asarray(getattr(queue, name)).copy() for name in QueuedControlBatch.__dataclass_fields__})

    original_commands, original_controls, unavailable = first_sample()
    assert not unavailable['available'].any() and not unavailable['packet_mask'].any()
    # Generated provenance fixture only; no user recording or control ownership
    # is modified. The real recorder seals this proof while holding its lock.
    manifest['controlExclusion'] = {'schemaVersion': 1, 'ownershipID': str(uuid.uuid4()),
        'recordingID': manifest['id'], 'startedNanos': 900_000_000,
        'throughNanos': manifest['stoppedNanos'], 'producersJoinedNanos': manifest['stoppedNanos'] + 1}
    manifest_path.write_text(json.dumps(manifest))
    commands, controls, known_empty = first_sample()
    assert commands == original_commands
    assert known_empty['available'].all() and not known_empty['packet_mask'].any()
    np.testing.assert_array_equal(controls, original_controls)
    with sqlite3.connect(source / 'index.sqlite') as database:
        event = json.loads(database.execute('SELECT event FROM events WHERE sequence=3').fetchone()[0])
        event['keyCode'] = 14  # At25ms, after the2ms observation but inside its lead-shifted target.
        database.execute('UPDATE events SET event=? WHERE sequence=3', (json.dumps(event).encode(),))
    changed_commands, changed_controls, changed_queue = first_sample()
    assert changed_commands != commands
    np.testing.assert_array_equal(changed_controls, controls)
    for name, before in known_empty.items():
        np.testing.assert_array_equal(changed_queue[name], before)
