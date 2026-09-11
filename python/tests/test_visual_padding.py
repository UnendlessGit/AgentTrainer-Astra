from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

from astra.model.config import ModelConfig
from astra.model.convnext import ConvNeXtEncoder, content_mask
from astra.model.vision import DetailEncoder, VisualEncoder


def test_internal_masks_preserve_unpadded_backbone_reference_path():
    mx.random.seed(802)
    encoder = ConvNeXtEncoder(ModelConfig.test_small())
    image = mx.random.normal((2, 64, 64, 3))
    plain = encoder(image)
    masked = encoder(image, mx.array([[0.0, 0.0, 1.0, 1.0]] * 2))
    for actual, expected in zip(masked, plain):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("branch", ["backbone", "detail"])
def test_padding_values_cannot_influence_valid_features_or_input_gradients(branch):
    mx.random.seed(82)
    encoder = ConvNeXtEncoder(ModelConfig.test_small()) if branch == "backbone" else DetailEncoder((8, 16, 24))
    rect = mx.array([[0.0, 10 / 64, 1.0, 43 / 64]])
    mask = content_mask(rect, 64, 64)
    image = mx.random.normal((1, 64, 64, 3))
    clean = mx.where(mask, image, 0)
    contaminated = mx.where(mask, image, mx.random.normal(image.shape) * 100)
    def features(value):
        result = encoder(value, rect)
        return result[-1] if isinstance(result, list) else result
    np.testing.assert_array_equal(np.asarray(features(clean)), np.asarray(features(contaminated)))
    gradient = mx.grad(lambda value: mx.mean(mx.square(features(value))))(contaminated)
    assert np.isfinite(np.asarray(gradient)).all()
    np.testing.assert_array_equal(np.asarray(mx.where(mask, 0, gradient)), 0)
    assert bool(mx.any(gradient != 0))


def test_random_backbone_padding_no_longer_amplifies_zero_stem_bias_gradient():
    mx.random.seed(1)
    encoder = ConvNeXtEncoder(ModelConfig.test_small())
    rect = mx.array([[0.0, 10 / 64, 1.0, 43 / 64]])
    image = mx.where(content_mask(rect, 64, 64), mx.random.normal((1, 64, 64, 3)), 0)
    def loss(module):
        return mx.mean(mx.square(module(image, rect)[-1]))
    value, gradient = nn.value_and_grad(encoder, loss)(encoder)
    mx.eval(value, gradient)
    assert all(np.isfinite(np.asarray(item)).all() for _, item in tree_flatten(gradient))
    bias = np.linalg.norm(np.asarray(gradient["downsamples"][0]["conv"]["bias"], dtype=np.float64))
    weight = np.linalg.norm(np.asarray(gradient["downsamples"][0]["conv"]["weight"], dtype=np.float64))
    # The unmasked random backbone amplified the zero stem bias by millions
    # through padded LayerNorm sites while ordinary weight gradients stayed small.
    assert bias < 100 * max(weight, 1e-6)


def test_empty_content_masks_stay_finite_and_have_no_parameter_gradient():
    mx.random.seed(813)
    encoder = ConvNeXtEncoder(ModelConfig.test_small())
    image = mx.random.normal((1, 64, 64, 3))
    rect = mx.array([[0.0, 0.0, 0.0, 0.0]])
    value, gradient = nn.value_and_grad(encoder, lambda model: mx.sum(model(image, rect)[-1]))(encoder)
    assert float(value) == 0
    for _, leaf in tree_flatten(gradient):
        np.testing.assert_array_equal(np.asarray(leaf), 0)


def test_old_development_model_schema_is_rejected_explicitly():
    assert ModelConfig().schema_version == 2
    with pytest.raises(ValueError, match="version 2"):
        replace(ModelConfig(), schema_version=1).validate()


@pytest.mark.parametrize("shape", [(96, 48), (48, 96), (96, 64)])
def test_visual_features_match_single_inference_when_batch_adds_spatial_padding(shape):
    from astra.data.batching import stack_observations
    from astra.data.observations import make_observation
    from astra.environments.practice import PracticeConfig, PracticeEnvironment

    mx.random.seed(614)
    config = ModelConfig.test_small()
    def observation(width, height):
        world = PracticeEnvironment(PracticeConfig(pixel_width=width, pixel_height=height,
                                                    logical_bounds=(0, 0, width, height)))
        value = world.reset(seed=9)
        return make_observation([(value.pixels, value.metadata)], value.control_state,
                                cutoff_nanos=value.metadata["observedNanos"], elapsed_seconds=0.1,
                                reset=True, config=config)
    original = observation(*shape)
    larger = observation(96, 96)
    model = VisualEncoder(config)
    single = model(original)
    batched = model(stack_observations([[original], [larger]]))
    mx.eval(single.summary, single.cells, batched.summary, batched.cells)
    original_valid = np.asarray(single.cell_valid[0, 0, 0])
    batch_valid = np.asarray(batched.cell_valid[0, 0, 0])
    assert original_valid.sum() == batch_valid.sum()
    np.testing.assert_allclose(np.asarray(single.cell_bounds[0, 0, 0])[original_valid],
                               np.asarray(batched.cell_bounds[0, 0, 0])[batch_valid], atol=1e-6)
    np.testing.assert_allclose(np.asarray(single.cells[0, 0, 0])[original_valid],
                               np.asarray(batched.cells[0, 0, 0])[batch_valid], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(np.asarray(single.summary[0, 0]), np.asarray(batched.summary[0, 0]), atol=2e-5, rtol=2e-5)
