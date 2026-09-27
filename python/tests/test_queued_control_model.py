from copy import deepcopy
from dataclasses import replace
import uuid

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

from astra.model.config import ModelConfig
from astra.model.policy import AgentPolicy
from astra.model.actions import ActionVocabulary
from astra.model.queued_control import QueuedControlEncoder, QueuedControlBatch
from astra.model.queue_layout import COMMAND_FEATURES
from astra.data.queued_controls import prepare_queued_controls
from astra.data.batching import stack_observations
from test_policy_core import observations


def feedback_fixture(*, empty_commands=False):
    run, epoch, packet = (str(uuid.uuid4()) for _ in range(3))
    surface = dict(id='source', globalBounds=dict(x=-800, y=0, width=800, height=600),
        pixelWidth=800, pixelHeight=600, contentBounds=dict(x=0, y=0, width=800, height=600), geometryRevision=7)
    start, cutoff = 1_050_000_000, 1_055_000_000
    commands = [] if empty_commands else [dict(operation='pointerAbsolute', offsetMs=0, surfaceID='source', x=.2, y=.3),
        dict(operation='keyDown', offsetMs=2, keyCode=0), dict(operation='pointerAbsolute', offsetMs=10, surfaceID='source', x=.8, y=.9)]
    progress = [] if empty_commands else [dict(commandIndex=index, status='partial' if index == 2 else 'posted',
        completedSampleCount=5 if index == 2 else 1, lastCompletedOffsetMs=offset,
        lastCompletedAvailableNanos=start + offset*1_000_000, lastPostedNanos=start + offset*1_000_000)
        for index, offset in enumerate((0, 2, 5))]
    feedback = dict(version=1, controlEpochID=epoch, runID=run, geometryRevision=7, cutoffNanos=cutoff,
        coverageNanos=cutoff, unavailableReason=None, acknowledgedThrough=None, throughSequence=0,
        changes=[dict(sequence=0, packetID=packet, kind='admitted', availableNanos=950_000_000)],
        packets=[dict(packet=dict(id=packet, runID=run, sequence=10, observationID=str(uuid.uuid4()),
            geometryRevision=7, executeAtNanos=start, durationMs=100, commands=commands),
            admissionSequence=0, admittedNanos=950_000_000, progress=progress)])
    return feedback, [surface]


def build(feedback=None, *, empty_commands=False):
    config = replace(ModelConfig.test_small(), schema_version=3)
    default, surfaces = feedback_fixture(empty_commands=empty_commands)
    return prepare_queued_controls(default if feedback is None else feedback, surfaces, config, cutoff_nanos=default['cutoffNanos'])


def test_schema_two_identity_and_schema_three_parameter_count_zero_residual_parity():
    config = ModelConfig()
    assert config.signature == 'c90055feaea46483be73f4cd5a48f1dd71867d9ad496ba6ae908b8a83ec94ff3'
    assert config.parameter_count == 34_638_639
    assert 'queued_command_width' not in config.to_dict() and 'queued_packet_width' not in config.to_dict()
    assert replace(config, schema_version=3).parameter_count - config.parameter_count == 208_360
    with pytest.raises(ValueError, match='schema 3'):
        replace(config, queued_command_width=32).validate()
    old = ModelConfig.test_small(); new = replace(old, schema_version=3)
    vocabulary = ActionVocabulary(key_codes=(0,), mouse_buttons=(0,), absolute_pointer=True)
    legacy = AgentPolicy(old, vocabulary); queued = AgentPolicy(new, vocabulary)
    queued.update(legacy.parameters())
    assert sum(value.size for _, value in tree_flatten(legacy.parameters())) == old.parameter_count
    assert sum(value.size for _, value in tree_flatten(queued.parameters())) == new.parameter_count
    obs = observations(batch=1, time=1, surfaces=1)
    current = replace(obs, queued_control=build())
    original, added = legacy(obs), queued(current)
    np.testing.assert_array_equal(np.asarray(original.temporal.context), np.asarray(added.temporal.context))
    np.testing.assert_array_equal(np.asarray(original.temporal.value), np.asarray(added.temporal.value))
    assert 'queued_control' not in obs.as_tensors()
    assert 'queued_control' in current.as_tensors()
    assert 'queued_control' not in current.as_tensors(include_queued_control=False)


def test_semantic_builder_retains_interleaved_motion_anchor_progress_and_excludes_ids():
    feedback, surfaces = feedback_fixture()
    config = replace(ModelConfig.test_small(), schema_version=3)
    first = prepare_queued_controls(feedback, surfaces, config, cutoff_nanos=feedback['cutoffNanos'])
    categories = np.asarray(first.categorical)
    features = np.asarray(first.command_features)
    assert categories[0, 0, 0, 2, 5] == 6  # The preceding pointer remains the anchor across a key command.
    assert features[0, 0, 0, 2, COMMAND_FEATURES['anchor_x']] == np.float32(.2)
    assert features[0, 0, 0, 2, COMMAND_FEATURES['x']] == np.float32(.8)
    assert features[0, 0, 0, 2, COMMAND_FEATURES['completed_fraction']] == .5
    changed = deepcopy(feedback)
    changed['controlEpochID'] = str(uuid.uuid4()); changed['runID'] = str(uuid.uuid4())
    item = changed['packets'][0]['packet']; item['runID'] = changed['runID']; item['id'] = str(uuid.uuid4())
    item['observationID'] = str(uuid.uuid4()); item['sequence'] = 999
    changed['changes'][0]['packetID'] = item['id']
    second = prepare_queued_controls(changed, surfaces, config, cutoff_nanos=changed['cutoffNanos'])
    for name in first.__dataclass_fields__:
        np.testing.assert_array_equal(np.asarray(getattr(first, name)), np.asarray(getattr(second, name)))
    arrays = {name: np.array(value) for name, value in first.as_tensors().items()}
    arrays['command_mask'][0, 0, 0, 1] = False
    with pytest.raises(ValueError, match='prefix'):
        QueuedControlBatch.from_numpy(maximum_surfaces=config.maximum_surfaces, **arrays)


def test_zero_output_learns_projection_then_encoder_and_keeps_inactive_branch_exact():
    config = replace(ModelConfig.test_small(), schema_version=3)
    encoder = QueuedControlEncoder(config); feedback = build(); active = mx.ones((1, 1), dtype=mx.bool_)
    def loss(module): return mx.sum(module(feedback, active))
    value, gradients = nn.value_and_grad(encoder, loss)(encoder)
    mx.eval(value, gradients)
    flat = dict(tree_flatten(gradients))
    assert float(value) == 0 and np.any(np.asarray(flat['output.weight']) != 0)
    assert all(not np.any(np.asarray(gradient) != 0) for name, gradient in flat.items() if name != 'output.weight')
    encoder.output.weight = encoder.output.weight - .01 * gradients['output']['weight']
    _, learned = nn.value_and_grad(encoder, loss)(encoder); mx.eval(learned)
    for name in ('commands.Wx', 'packets.Wx', 'embeddings.0.weight'):
        assert np.any(np.asarray(dict(tree_flatten(learned))[name]) != 0), name
    for available in (False, True):
        empty = QueuedControlBatch.empty(1, 1, config.packet_capacity, available=available)
        np.testing.assert_array_equal(np.asarray(encoder(empty, active)), np.zeros((1, 1, config.recurrent_width)))
    np.testing.assert_array_equal(np.asarray(encoder(feedback, mx.zeros((1, 1), dtype=mx.bool_))), np.zeros((1, 1, config.recurrent_width)))
    assert np.any(np.asarray(encoder(build(empty_commands=True), active)) != 0)  # A real empty packet has a lifetime.


def test_queue_batch_time_slicing_preserves_recurrence_reset_and_padding():
    config = replace(ModelConfig.test_small(), schema_version=3)
    model = AgentPolicy(config, ActionVocabulary(absolute_pointer=True))
    model.temporal.queued_control.output.weight = mx.random.normal(model.temporal.queued_control.output.weight.shape) * .01
    source = observations(batch=1, time=3, surfaces=1)
    rows = [[replace(source.slice_time(i, i + 1), queued_control=build()) for i in range(3)]]
    observation = stack_observations(rows)
    complete = model(observation)
    state, result = None, []
    for index in range(3):
        output = model(observation.slice_time(index, index + 1), state)
        state = output.temporal.state; result.append(output.temporal.context)
    np.testing.assert_allclose(np.asarray(complete.temporal.context), np.asarray(mx.concatenate(result, axis=1)), atol=2e-5, rtol=2e-5)
    padded = replace(observation.slice_time(1, 3), valid=mx.zeros((1, 2), dtype=mx.bool_), reset=mx.ones((1, 2), dtype=mx.bool_))
    after_padding = model(padded, state)
    for before, after in zip(state, after_padding.temporal.state): np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    reset = replace(observation, reset=mx.array([[False, False, True]]))
    np.testing.assert_allclose(np.asarray(model(reset).temporal.context[:, 2:]),
                               np.asarray(model(observation.slice_time(2, 3)).temporal.context), atol=2e-5, rtol=2e-5)


def test_staged_backward_matches_complete_policy_with_learned_queue_residual():
    from test_staged_backward import example, compare_gradients
    old, observation, packets, state = example()
    config = replace(old.config, schema_version=3)
    model = AgentPolicy(config, old.actions.vocabulary); model.update(old.parameters())
    model.configure_execution(vision_microbatch=2)
    model.temporal.queued_control.output.weight = mx.random.normal(model.temporal.queued_control.output.weight.shape) * .01
    queue = build()
    queued = QueuedControlBatch.from_tensors({name: mx.broadcast_to(value, (*observation.shape, *value.shape[2:]))
                                             for name, value in queue.as_tensors().items()})
    result = compare_gradients(model, replace(observation, queued_control=queued), packets, state, 'ppo', action_microbatch=2)
    assert np.any(np.asarray(dict(tree_flatten(result.gradients))['temporal.queued_control.commands.Wx']) != 0)
