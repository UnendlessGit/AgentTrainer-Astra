"""Compare bounded execution with the ordinary complete policy gradient.

Tiny visual dimensions exercise the actual policy, packet likelihood and loss
math. Production memory/learning qualification lives in separate benchmarks.
"""
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

from astra.learning.backward import policy_gradients
from astra.learning.rl import ppo_loss
from astra.model.actions import ActionVocabulary, PacketBatch
from astra.model.config import ModelConfig
from astra.model.observation import ObservationBatch, SurfaceBatch
from astra.model.policy import AgentPolicy


def example(*, frozen=False, invalid_padding=False):
    mx.random.seed(19)
    config = replace(ModelConfig.test_small(), context_sizes=(3, 4))
    policy = AgentPolicy(config, ActionVocabulary((4,), (0,), True, True, True))
    policy.configure_execution(vision_microbatch=2, checkpoint_vision=False)
    if frozen:
        policy.vision.backbone.freeze()
    surfaces = []
    shape = (2, 3)
    for role, height in enumerate((64, 32)):
        rect = mx.broadcast_to(mx.array([0., 0., 1., 1.]), (*shape, 4))
        surfaces.append(SurfaceBatch(
            mx.random.normal((*shape, height, 64, 3)),
            mx.random.normal((*shape, height, 64, 3)),
            mx.random.normal((*shape, 32, 32, 3)), rect, rect,
            mx.broadcast_to(mx.array([-.1, .2, .4, .4]), (*shape, 4)),
            mx.broadcast_to(mx.array([-200. + role * 1024, 0., 1024., 768.]), (*shape, 4)),
            mx.ones(shape, dtype=mx.bool_)))
    observation = ObservationBatch(
        tuple(surfaces), mx.random.normal((*shape, config.control_width)),
        mx.array([[.1, .12, .04], [.08, .1, 0.]]),
        mx.array([[[0, 3], [1, 2], [2, 1]], [[2, 0], [1, 1], [0, 2]]]),
        mx.array([[False, False, True], [False, False, True]]),
        mx.array([[True, True, True], [True, True, False]]))
    state = tuple(mx.random.normal((2, config.recurrent_width)) for _ in range(config.recurrent_layers))
    # Distinct lane-major labels detect accidental T,B transposition. Every
    # action family is represented, including pointing on the shorter surface.
    fields = {
        'operation': [[6, 4, 5], [1, 6, 2], [7, 8], [6, 3], [8, 7], [0]],
        'offset': [[0, 20, 30], [0, 10, 40], [20, 21], [1, 2], [9, 90], []],
        'key': [[], [4, 0, 4], [], [0, 4]],
        'surface': [[0], [0, 1], [], [1]],
        'cell': [[7], [0, 17], [], [6]],
        'within_x': [[31], [0, 21], [], [12]],
        'within_y': [[9], [0, 44], [], [3]],
        'dx': [[], [], [2, -3], [], [4, 6]],
        'dy': [[], [], [3, -2], [], [5, 8]],
    }
    arrays = {}
    for name in PacketBatch.__dataclass_fields__:
        values = np.zeros((6, config.packet_capacity + 1), dtype=np.int32)
        for row, entries in enumerate(fields.get(name, [])):
            values[row, :len(entries)] = entries
        arrays[name] = values
    if invalid_padding:
        # This key is outside the vocabulary. Its log likelihood really is
        # -inf; only the objective's invalid-decision mask removes it.
        arrays['operation'][5, 0] = 1
        arrays['key'][5, 0] = 127
    packets = PacketBatch(**{name: mx.array(value) for name, value in arrays.items()})
    return policy, observation, packets, state


def loss_objective(policy, observation, packets, state, kind):
    valid = observation.valid.reshape(-1)
    if kind == 'behavioral':
        def objective(logp, value, entropy):
            return -mx.sum(mx.where(valid, logp, 0)), {'validCount': mx.sum(valid)}
        return objective
    initial = policy(observation, state)
    scored = policy.score(initial, packets)
    # Exercise both clipped and unclipped ratios with both advantage signs.
    # The invalid row is excluded before every PPO arithmetic operation.
    old = mx.stop_gradient(scored.log_probability + mx.array([.4, -.4, 0., .1, -.1, 0.]))
    mx.eval(old)
    def objective(logp, value, entropy):
        result = ppo_loss(logp, value, entropy, old_log_probabilities=old,
                          advantages=mx.array([1., -1., .2, -.3, 2., 0.]),
                          returns=mx.array([.2, -.1, .7, -.3, .9, 0.]), valid=valid)
        return result.total * result.valid_count, {
            'metrics': mx.stack((result.policy, result.value, result.entropy, result.sampled_kl, result.clip_fraction)),
            'validCount': result.valid_count,
            'ratio': (result.ratio_min, result.ratio_max),
            'finite': result.finite,
        }
    return objective


def assert_trees_close(actual, expected):
    actual, expected = dict(tree_flatten(actual)), dict(tree_flatten(expected))
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        assert np.isfinite(np.asarray(value)).all(), name
        np.testing.assert_allclose(np.asarray(value), np.asarray(expected[name]),
                                   atol=2e-5, rtol=2e-4, err_msg=name)


def compare_gradients(policy, observation, packets, state, kind, *, action_microbatch):
    objective = loss_objective(policy, observation, packets, state, kind)
    def ordinary(model):
        encoding = model(observation, state)
        score = model.score(encoding, packets)
        loss, auxiliary = objective(score.log_probability, encoding.temporal.value.reshape(-1), score.conditional_entropy)
        return loss, (auxiliary, encoding.temporal.state)
    (reference_loss, (reference_aux, reference_state)), reference = nn.value_and_grad(policy, ordinary)(policy)
    mx.eval(reference_loss, reference_aux, reference_state, reference)
    bindings = dict(tree_flatten(policy.parameters()))
    result = policy_gradients(policy, observation, packets, objective, state, action_microbatch=action_microbatch)
    mx.eval(result.loss, result.auxiliary, result.next_state, result.gradients)
    np.testing.assert_allclose(float(result.loss), float(reference_loss), atol=2e-5, rtol=2e-5)
    assert_trees_close(result.auxiliary, reference_aux)
    assert_trees_close(result.next_state, reference_state)
    assert_trees_close(result.gradients, reference)
    # Recomputing gradients cannot rebind the caller's parameters to tracers.
    for name, value in tree_flatten(policy.parameters()):
        assert value is bindings[name], name
    assert dict(tree_flatten(result.gradients)).keys() == dict(tree_flatten(policy.trainable_parameters())).keys()
    return result


@pytest.mark.parametrize('kind', ['behavioral', 'ppo'])
@pytest.mark.parametrize('frozen', [False, True])
def test_staged_gradient_matches_complete_policy_with_carries_resets_and_padding(kind, frozen):
    policy, observation, packets, state = example(frozen=frozen)
    result = compare_gradients(policy, observation, packets, state, kind, action_microbatch=4)
    flat = dict(tree_flatten(result.gradients))
    # The fixture must actually exercise both direct pointing and recurrent
    # visual paths, rather than passing on an accidentally constant objective.
    for name in ('vision.detail.convolutions.0.weight', 'vision.crop_position.weight',
                 'vision.pool.query_proj.weight', 'temporal.layers.0.Wx', 'temporal.layers.1.Wh',
                 'temporal.context_embeddings.0.weight', 'actions.cell_query.weight',
                 'actions.x_head.weight', 'actions.delta_heads.0.weight'):
        assert np.any(np.asarray(flat[name]) != 0), name
    assert any(name.startswith('vision.backbone.') for name in flat) is not frozen
    if kind == 'ppo':
        assert np.any(np.asarray(flat['temporal.value_head.weight']) != 0)
        assert bool(result.auxiliary['finite'])
        low, high = result.auxiliary['ratio']
        assert float(low) < .8 and float(high) > 1.2


@pytest.mark.parametrize('kind', ['behavioral', 'ppo'])
def test_zero_cotangent_for_masked_invalid_likelihood_does_not_contaminate_gradients(kind):
    policy, observation, packets, state = example(invalid_padding=True)
    score = policy.score(policy(observation, state), packets)
    assert np.isneginf(float(score.log_probability[5]))
    assert not bool(observation.valid.reshape(-1)[5])
    compare_gradients(policy, observation, packets, state, kind, action_microbatch=1)


@pytest.mark.parametrize('cancel_check', [1, 8, 12])
def test_cancelled_stages_preserve_all_parameter_bindings_and_trainable_mask(cancel_check):
    policy, observation, packets, state = example(frozen=True)
    objective = loss_objective(policy, observation, packets, state, 'behavioral')
    mx.eval(policy.parameters(), state)
    bindings = dict(tree_flatten(policy.parameters()))
    trainable = set(dict(tree_flatten(policy.trainable_parameters())))
    original_state = [np.asarray(value).copy() for value in state]
    checks = 0
    def cancelled():
        nonlocal checks
        checks += 1
        return checks == cancel_check
    with pytest.raises(InterruptedError, match='cancelled'):
        policy_gradients(policy, observation, packets, objective, state, action_microbatch=2, cancelled=cancelled)
    assert checks == cancel_check
    for name, value in tree_flatten(policy.parameters()):
        assert value is bindings[name], name
    assert set(dict(tree_flatten(policy.trainable_parameters()))) == trainable
    for actual, expected in zip(state, original_state, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), expected)
