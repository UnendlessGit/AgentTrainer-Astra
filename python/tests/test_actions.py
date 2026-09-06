from dataclasses import replace
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_flatten
import numpy as np
import pytest

from astra.model.actions import ActionVocabulary, PacketBatch, PacketDecoder, Operation
from astra.model.config import ModelConfig
from astra.model.vision import VisualFeatures


def visual(batch=2):
    cells = mx.random.normal((batch, 2, 12, 24))
    valid = mx.broadcast_to(mx.array([[True] * 12, [True] * 8 + [False] * 4]), (batch, 2, 12))
    return VisualFeatures(mx.zeros((batch, 96)), cells, mx.zeros((batch, 2, 12, 4)), valid,
                          mx.ones((batch, 2), dtype=mx.bool_))


def make_packets(config, operations, offsets=None, **fields):
    result = {}
    batch = len(operations)
    for name in PacketBatch.__dataclass_fields__:
        values = operations if name == "operation" else (offsets if name == "offset" else fields.get(name))
        matrix = np.zeros((batch, config.packet_capacity + 1), dtype=np.int32)
        if values is not None:
            for index, row in enumerate(values):
                matrix[index, :len(row)] = row
        result[name] = mx.array(matrix)
    return PacketBatch(**result)


def test_sampled_packet_matches_teacher_scoring_and_capability_grammar():
    mx.random.seed(832)
    config = ModelConfig.test_small()
    vocabulary = ActionVocabulary((4, 13, 123), (0, 1), True, True, True)
    decoder = PacketDecoder(config, vocabulary)
    decoder.operation_head.bias = mx.array([-2.0] + [0.0] * 8)
    context = mx.random.normal((8, config.recurrent_width))
    features = visual(8)
    sampled = decoder.sample(context, features, key=mx.random.key(112))
    sampled.packets.validate(config, vocabulary, features)
    scored = decoder.log_prob(context, features, sampled.packets)
    np.testing.assert_allclose(np.asarray(sampled.log_probability), np.asarray(scored.log_probability), atol=2e-4, rtol=2e-5)
    for factor in sampled.factor_log_probabilities:
        np.testing.assert_allclose(np.asarray(sampled.factor_log_probabilities[factor]), np.asarray(scored.factor_log_probabilities[factor]), atol=2e-5, rtol=2e-5)
    assert np.isfinite(np.asarray(sampled.conditional_entropy)).all()
    assert (np.asarray(sampled.log_probability) < 0).all()


def test_exact_joint_probability_counts_end_ordered_time_and_active_arguments():
    config = replace(ModelConfig.test_small(), period_ms=4)
    vocabulary = ActionVocabulary((4, 13), (0,))
    decoder = PacketDecoder(config, vocabulary)
    decoder.update(tree_map(mx.zeros_like, decoder.parameters()))
    features = visual(1)
    context = mx.zeros((1, config.recurrent_width))
    packets = make_packets(config, [[Operation.KEY_DOWN, Operation.BUTTON_UP, Operation.END]], [[1, 2]], key=[[13]])
    packets.validate(config, vocabulary, features)
    output = decoder.log_prob(context, features, packets)
    expected = -3 * math.log(6) - math.log(4) - math.log(3) - math.log(2)
    np.testing.assert_allclose(float(output.log_probability[0]), expected, atol=2e-6)
    # Inactive arguments, including everything after END, have no probability
    # and cannot influence any later active prediction.
    noisy = replace(packets, cell=mx.full(packets.cell.shape, 9999), dx=mx.full(packets.dx.shape, 4000))
    np.testing.assert_allclose(np.asarray(decoder.log_prob(context, features, noisy).log_probability), expected, atol=2e-6)


def test_forced_end_is_probability_one_and_never_repeated_input_state_masking():
    config = replace(ModelConfig.test_small(), period_ms=1)
    vocabulary = ActionVocabulary((4,))
    decoder = PacketDecoder(config, vocabulary)
    decoder.update(tree_map(mx.zeros_like, decoder.parameters()))
    features = visual(1)
    context = mx.zeros((1, config.recurrent_width))
    packets = make_packets(config, [[Operation.KEY_DOWN] * config.packet_capacity], key=[[4] * config.packet_capacity])
    packets.validate(config, vocabulary, features)
    output = decoder.log_prob(context, features, packets)
    np.testing.assert_allclose(float(output.log_probability[0]), -config.packet_capacity * math.log(4), atol=3e-6)
    assert float(output.factor_log_probabilities["operation"][0, -1]) == 0


def test_disabled_capabilities_and_unavailable_surfaces_only_emit_end():
    config = ModelConfig.test_small()
    features = visual(2)
    features = replace(features, cell_valid=mx.zeros_like(features.cell_valid), surface_valid=mx.zeros_like(features.surface_valid))
    decoder = PacketDecoder(config, ActionVocabulary(absolute_pointer=True))
    output = decoder.sample(mx.zeros((2, config.recurrent_width)), features, key=mx.random.key(1))
    assert not np.asarray(output.packets.operation).any()
    np.testing.assert_array_equal(np.asarray(output.log_probability), np.zeros(2))
    np.testing.assert_array_equal(np.asarray(output.conditional_entropy), np.zeros(2))


def test_packet_nll_gradients_are_finite_with_inactive_conditional_heads():
    config = ModelConfig.test_small()
    decoder = PacketDecoder(config, ActionVocabulary((4,), (0,), True, True, True))
    features = visual(2)
    context = mx.random.normal((2, config.recurrent_width))
    packets = make_packets(config, [[Operation.KEY_DOWN, Operation.ABSOLUTE], [Operation.RELATIVE, Operation.SCROLL]],
                           [[0, 10], [1, 20]], key=[[4], []], surface=[[0, 1], []], cell=[[0, 7], []],
                           within_x=[[0, 31], []], within_y=[[0, 10], []], dx=[[], [-4095, 4095]], dy=[[], [18, -20]])
    packets.validate(config, decoder.vocabulary, features)
    def loss_fn(model):
        return -mx.mean(model.log_prob(context, features, packets).log_probability)
    loss, grads = nn.value_and_grad(decoder, loss_fn)(decoder)
    mx.eval(loss, grads)
    assert np.isfinite(float(loss))
    for name, value in tree_flatten(grads):
        assert np.isfinite(np.asarray(value)).all(), name
    assert np.any(np.asarray(grads["operation_head"]["weight"]) != 0)
    assert np.any(np.asarray(grads["cell_query"]["weight"]) != 0)
    assert np.any(np.asarray(grads["delta_heads"][0]["weight"]) != 0)


@pytest.mark.parametrize("kind", ["after_end", "mixed_pointer", "time", "unavailable_cell", "delta"])
def test_invalid_teacher_packets_are_rejected_before_training(kind):
    config = ModelConfig.test_small()
    vocabulary = ActionVocabulary((4,), (0,), True, True, True)
    features = visual(1)
    if kind == "after_end":
        packet = make_packets(config, [[Operation.END, Operation.KEY_DOWN]], key=[[0, 4]])
    elif kind == "mixed_pointer":
        packet = make_packets(config, [[Operation.RELATIVE, Operation.ABSOLUTE]])
    elif kind == "time":
        packet = make_packets(config, [[Operation.KEY_DOWN]], [[100]], key=[[4]])
    elif kind == "unavailable_cell":
        packet = make_packets(config, [[Operation.ABSOLUTE]], surface=[[1]], cell=[[11]])
    else:
        packet = make_packets(config, [[Operation.RELATIVE]], dx=[[4096]])
    with pytest.raises(ValueError):
        packet.validate(config, vocabulary, features)
