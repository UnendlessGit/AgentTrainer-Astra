"""Experimental attention semantics and independent derivative checks."""
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

from astra.model.config import ModelConfig
from astra.model.observation import ObservationBatch
from astra.model.actions import VisualFeatures
from experiments.causal_transformer import (
    TransformerSpecification, CausalObservationTransformer, ExperimentalTransformerHead,
)
from experiments.temporal_study import cached_batch_gradients
from experiments.test_temporal_study import fixture


def inputs(length=13, lanes=2, *, history=8, layers=2):
    mx.random.seed(909)
    config = ModelConfig.test_small()
    core = CausalObservationTransformer(config, TransformerSpecification(width=24, heads=3, layers=layers, history=history))
    summary = mx.random.normal((lanes, length, config.query_count * config.spatial_width))
    observation = ObservationBatch((), mx.random.normal((lanes, length, config.control_width)),
        mx.full((lanes, length), .1), mx.zeros((lanes, length, 0), dtype=mx.int32),
        mx.broadcast_to(mx.arange(length) == 0, (lanes, length)), mx.ones((lanes, length), dtype=mx.bool_))
    return core, summary, observation


def close(left, right, *, atol=2e-5):
    for a, b in zip(left, right):
        np.testing.assert_allclose(a, b, rtol=3e-5, atol=atol)


def test_step_sequence_equivalence_with_resets_padding_and_cache_eviction():
    core, summary, observation = inputs(length=23)
    valid = np.ones((2, 23), dtype=np.bool_); valid[0, [2, 4, 13]] = False; valid[1, 20:] = False
    resets = np.zeros_like(valid); resets[:, 0] = True; resets[0, [4, 9]] = True; resets[1, 18] = True
    observation = replace(observation, valid=mx.array(valid), reset=mx.array(resets))
    full = core(summary, observation)
    state = None; contexts = []; values = []
    for step in range(23):
        result = core(summary[:, step:step+1], observation.slice_time(step, step+1), state)
        state = result.state; contexts.append(result.context); values.append(result.value)
    close([full.context, full.value], [mx.concatenate(contexts, axis=1), mx.concatenate(values, axis=1)])
    close(full.state, state)
    np.testing.assert_array_equal(full.state[0], [13, 2])
    np.testing.assert_array_equal(mx.sum(full.state[2], axis=1), [8, 2])
    np.testing.assert_array_equal(full.state[1][0], np.arange(5, 13))
    assert all(value.shape[2] == 8 for value in full.state[3:])


def test_padded_reset_preserves_every_cache_tensor_exactly():
    core, summary, observation = inputs(length=5)
    state = core(summary, observation).state
    inactive = replace(observation, valid=mx.zeros((2, 5), dtype=mx.bool_), reset=mx.ones((2, 5), dtype=mx.bool_))
    result = core(summary * 1000, inactive, state)
    for before, after in zip(state, result.state): np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(result.context, 0)
    np.testing.assert_array_equal(result.value, 0)


def test_reset_removes_all_previous_episode_influence_and_gradients():
    core, summary, observation = inputs(length=9, lanes=1)
    reset = mx.array([[True, False, False, False, True, False, False, False, False]])
    observation = replace(observation, reset=reset)
    full = core(summary, observation)
    fresh = core(summary[:, 4:], observation.slice_time(4, 9))
    close([full.context[:, 4:], full.value[:, 4:]], [fresh.context, fresh.value])
    close(full.state, fresh.state)
    grad = mx.grad(lambda x: mx.sum(core(x, observation).context[:, -1, :3]))(summary)
    np.testing.assert_array_equal(grad[:, :4], 0)
    assert float(mx.linalg.norm(grad[:, 4:]).item()) > 1e-5


def test_future_observations_and_keys_have_zero_gradient_to_earlier_output():
    core, summary, observation = inputs(length=8, lanes=1)
    changed = mx.concatenate((summary[:, :4], summary[:, 4:] + 77), axis=1)
    other = replace(observation, controls=mx.concatenate((observation.controls[:, :4], observation.controls[:, 4:] - 39), axis=1))
    np.testing.assert_array_equal(core(summary, observation).context[:, :4], core(changed, other).context[:, :4])
    gradient = mx.grad(lambda x: mx.sum(core(x, observation).context[:, 3, :3]))(summary)
    np.testing.assert_array_equal(gradient[:, 4:], 0)
    assert float(mx.linalg.norm(gradient[:, :4]).item()) > 1e-5


def test_one_layer_direct_key_window_is_enforced_beyond_cache_size():
    core, summary, observation = inputs(length=12, lanes=1, history=4, layers=1)
    changed = summary.at[:, 0].add(10)
    original, different = core(summary, observation), core(changed, observation)
    np.testing.assert_array_equal(original.context[:, 4:], different.context[:, 4:])
    assert float(mx.linalg.norm(original.context[:, 3] - different.context[:, 3]).item()) > 1e-5
    # A multi-layer cached representation can indirectly contain older history;
    # only this one-layer control should have a strict raw-input horizon of four.


def test_attached_chunk_cache_matches_full_gradients_and_detach_cuts_credit():
    core, summary, observation = inputs(length=11, lanes=1, history=16)
    def run(value, detached=False):
        first = core(value[:, :5], observation.slice_time(0, 5))
        state = tuple(mx.stop_gradient(x) for x in first.state) if detached else first.state
        return core(value[:, 5:], observation.slice_time(5, 11), state).context
    expected = mx.grad(lambda x: mx.sum(core(x, observation).context[:, -1, :3]))(summary)
    actual = mx.grad(lambda x: mx.sum(run(x)[:, -1, :3]))(summary)
    detached = mx.grad(lambda x: mx.sum(run(x, True)[:, -1, :3]))(summary)
    np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=2e-5)
    assert float(mx.linalg.norm(actual[:, :5]).item()) > 1e-5
    np.testing.assert_array_equal(detached[:, :5], 0)
    np.testing.assert_allclose(run(summary), core(summary, observation).context[:, 5:], rtol=3e-5, atol=2e-5)


def test_input_derivative_agrees_with_finite_difference():
    core, summary, observation = inputs(length=6, lanes=1)
    direction = mx.random.normal(summary.shape); direction /= mx.linalg.norm(direction)
    objective = lambda x: mx.sum(core(x, observation).context[:, -1, :3])
    analytic = mx.sum(mx.grad(objective)(summary) * direction)
    epsilon = .01
    numerical = (objective(summary + epsilon * direction) - objective(summary - epsilon * direction)) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=3e-3, atol=1e-4)


def test_irregular_attached_chunks_match_parameter_gradients_through_eviction_and_padding():
    core, summary, observation = inputs(length=19, lanes=1, history=8)
    valid = mx.array([[True] * 5 + [False] * 3 + [True] * 11])
    observation = replace(observation, valid=valid, reset=observation.reset | ~valid)

    def chunked(model):
        state = None; contexts = []
        for start, end in ((0, 5), (5, 8), (8, 13), (13, 19)):
            result = model(summary[:, start:end], observation.slice_time(start, end), state)
            state = result.state; contexts.append(result.context)
        return mx.concatenate(contexts, axis=1)

    np.testing.assert_allclose(chunked(core), core(summary, observation).context, rtol=3e-5, atol=2e-5)
    _, whole = nn.value_and_grad(core, lambda model: mx.sum(model(summary, observation).context[:, -1, :3]))(core)
    _, parts = nn.value_and_grad(core, lambda model: mx.sum(chunked(model)[:, -1, :3]))(core)
    left, right = dict(tree_flatten(whole)), dict(tree_flatten(parts))
    assert left.keys() == right.keys()
    for name in left: np.testing.assert_allclose(left[name], right[name], rtol=4e-4, atol=2e-5, err_msg=name)


@pytest.mark.parametrize('choice_only', [False, True])
def test_shared_decoder_and_staged_gradients_match_monolithic_objective(choice_only):
    policy, batch = fixture(length=4)
    original = dict(tree_flatten(policy.actions.parameters()))
    head = ExperimentalTransformerHead(policy, TransformerSpecification(width=24, heads=3, layers=2, history=8))
    assert head.actions is policy.actions
    assert head.temporal.value_hidden is policy.temporal.value_hidden
    assert head.temporal.value_head is policy.temporal.value_head
    for name, value in tree_flatten(head.actions.parameters()): np.testing.assert_array_equal(value, original[name])
    valid = batch.observation.valid.reshape(-1)
    if choice_only: valid = valid & (batch.packets.operation[:, 0] != 0)
    visual = batch.visual_at(list(range(batch.summary.shape[0] * batch.summary.shape[1])))
    def objective(model):
        temporal = model.temporal(batch.summary, batch.observation)
        scores = model.actions.log_prob(temporal.context.reshape(-1, policy.config.recurrent_width), visual, batch.packets)
        return -mx.sum(mx.where(valid, scores.log_probability, 0)) / mx.sum(batch.observation.valid)
    expected_loss, expected_gradient = nn.value_and_grad(head, objective)(head)
    mx.eval(expected_loss, expected_gradient)
    actual = cached_batch_gradients(head, batch, horizon=512, choice_only=choice_only)
    np.testing.assert_allclose(actual.loss, expected_loss, rtol=2e-5, atol=1e-5)
    left, right = dict(tree_flatten(actual.gradients)), dict(tree_flatten(expected_gradient))
    assert left.keys() == right.keys()
    for name in left: np.testing.assert_allclose(left[name], right[name], rtol=4e-4, atol=2e-5, err_msg=name)
    assert all(bool(mx.all(mx.isfinite(value)).item()) for value in left.values())


def test_invalid_state_and_half_precision_are_rejected():
    core, summary, observation = inputs(length=2)
    with pytest.raises(ValueError, match='cache shape'): core(summary, observation, core.initial_state(1))
    with pytest.raises(ValueError, match='FP32'): core(summary.astype(mx.float16), observation)
    with pytest.raises(ValueError, match='even integral'): TransformerSpecification(width=24, heads=8).validate()
    with pytest.raises(ValueError, match='bounded'): TransformerSpecification(history=513).validate()
