from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

from astra.model.config import ModelConfig
from astra.model.observation import ObservationBatch, SurfaceBatch
from astra.model.policy import PolicyCore


def observations(batch=2, time=3, *, surfaces=2, contexts=()):
    result = []
    for surface in range(surfaces):
        height, width = (64, 64) if surface == 0 else (32, 64)
        shape = (batch, time)
        result.append(SurfaceBatch(
            global_image=mx.random.normal((*shape, height, width, 3)),
            detail_image=mx.random.normal((*shape, height, width, 3)),
            cursor_image=mx.random.normal((*shape, 32, 32, 3)),
            global_content_rect=mx.broadcast_to(mx.array([0.0, 0.0, 1.0, 1.0]), (*shape, 4)),
            content_rect=mx.broadcast_to(mx.array([0.0, 0.0, 1.0, 1.0]), (*shape, 4)),
            cursor_rect=mx.broadcast_to(mx.array([0.2, 0.3, 0.3, 0.3]), (*shape, 4)),
            global_bounds=mx.broadcast_to(mx.array([-200.0, 0.0, 1024.0, 768.0]), (*shape, 4)),
            available=mx.ones(shape, dtype=mx.bool_),
        ))
    return ObservationBatch(tuple(result), mx.random.normal((batch, time, 178)),
                            mx.full((batch, time), 0.1), mx.zeros((batch, time, len(contexts)), dtype=mx.int32),
                            mx.zeros((batch, time), dtype=mx.bool_), mx.ones((batch, time), dtype=mx.bool_))


def test_sequence_matches_streaming_and_preserves_dense_surfaces():
    mx.random.seed(712)
    config = replace(ModelConfig.test_small(), context_sizes=(3, 4))
    model = PolicyCore(config)
    observation = observations(contexts=config.context_sizes)
    complete = model(observation)
    state = None
    chunks = []
    for index in range(observation.shape[1]):
        chunk = model(observation.slice_time(index, index + 1), state)
        state = chunk.temporal.state
        chunks.append(chunk.temporal.context)
    np.testing.assert_allclose(np.asarray(complete.temporal.context), np.asarray(mx.concatenate(chunks, axis=1)), atol=2e-5, rtol=2e-5)
    for left, right in zip(complete.temporal.state, state):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), atol=2e-5, rtol=2e-5)
    assert complete.visual.cells.shape == (2, 3, 2, 64, config.spatial_width)
    assert np.asarray(complete.visual.cell_valid)[..., 1, 32:].sum() == 0
    assert np.asarray(complete.visual.cell_valid)[..., 0, :].all()


def test_reset_isolates_history_and_padding_does_not_advance_state():
    mx.random.seed(81)
    model = PolicyCore(ModelConfig.test_small())
    original = observations(batch=1, time=3, surfaces=1)
    reset = replace(original, reset=mx.array([[False, False, True]]))
    encoded = model(reset)
    fresh = model(original.slice_time(2, 3))
    np.testing.assert_allclose(np.asarray(encoded.temporal.context[:, 2:]), np.asarray(fresh.temporal.context), atol=2e-5, rtol=2e-5)
    first = model(original.slice_time(0, 1))
    padding = replace(original.slice_time(1, 3), valid=mx.zeros((1, 2), dtype=mx.bool_), reset=mx.ones((1, 2), dtype=mx.bool_))
    padded = model(padding, first.temporal.state)
    for left, right in zip(first.temporal.state, padded.temporal.state):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    assert np.asarray(padded.temporal.context).sum() == 0
    assert np.asarray(padded.temporal.value).sum() == 0


def test_gradients_reach_every_visual_branch_and_recurrent_layer():
    mx.random.seed(199)
    model = PolicyCore(ModelConfig.test_small())
    observation = observations(batch=1, time=2, surfaces=1)

    def loss_fn(module, data):
        result = module(data)
        return mx.mean(result.temporal.value ** 2) + mx.mean(result.temporal.context[..., :3] ** 2)

    loss, grads = nn.value_and_grad(model, loss_fn)(model, observation)
    mx.eval(loss, grads)
    assert np.isfinite(float(loss))
    flat = dict(tree_flatten(grads))
    for name, value in flat.items():
        assert np.all(np.isfinite(np.asarray(value))), name
    for name in ("vision.backbone.downsamples.0.conv.weight", "vision.backbone.stages.3.0.contract.weight",
                 "vision.detail.convolutions.0.weight", "vision.crop_position.weight", "vision.pool.query_proj.weight",
                 "temporal.layers.0.Wx", "temporal.layers.1.Wh", "temporal.value_head.weight"):
        assert np.any(np.asarray(flat[name]) != 0), name


def test_unavailable_images_have_finite_attention_and_no_pointing_cells():
    model = PolicyCore(ModelConfig.test_small())
    observation = observations(batch=1, time=1, surfaces=1)
    surface = replace(observation.surfaces[0], available=mx.zeros((1, 1), dtype=mx.bool_))
    output = model(replace(observation, surfaces=(surface,)))
    assert np.isfinite(np.asarray(output.temporal.context)).all()
    assert not np.asarray(output.visual.cell_valid).any()


def test_visual_microbatch_and_checkpointing_preserve_values_and_parameter_gradients():
    mx.random.seed(615)
    model = PolicyCore(ModelConfig.test_small())
    observation = observations(batch=1, time=3, surfaces=1)
    def loss_fn(module):
        return mx.mean(module(observation).temporal.value ** 2)
    model.configure_execution(vision_microbatch=4, checkpoint_vision=False)
    original, gradients = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(original, gradients)
    model.configure_execution(vision_microbatch=1, checkpoint_vision=True)
    checkpointed, restored = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(checkpointed, restored)
    np.testing.assert_allclose(float(original), float(checkpointed), atol=3e-6, rtol=3e-5)
    reference = dict(tree_flatten(gradients))
    for name, actual in tree_flatten(restored):
        assert np.isfinite(np.asarray(actual)).all(), name
        np.testing.assert_allclose(np.asarray(reference[name]), np.asarray(actual), atol=2e-5, rtol=2e-4, err_msg=name)


@pytest.mark.parametrize("change", [{"period_ms": True}, {"coordinate_bins": 1}, {"spatial_width": 25}, {"global_long_edge": 16}])
def test_invalid_configurations_are_rejected(change):
    with pytest.raises(ValueError):
        replace(ModelConfig.test_small(), **change).validate()
