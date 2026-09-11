from dataclasses import replace
import uuid

import mlx.core as mx
import numpy as np
import pytest

from astra.data.history import ControlHistory
from astra.data.preprocessing import MEAN, STD, prepare_surface
from astra.model.config import ModelConfig
from astra.model.vision import align_features
from astra.recordings import RecordingError


def surface(width=120, height=60):
    return {"id": "primary", "globalBounds": {"x": -300, "y": 100, "width": width / 2, "height": height / 2},
            "pixelWidth": width, "pixelHeight": height, "contentBounds": {"x": 0, "y": 0, "width": width, "height": height}, "geometryRevision": 0}


def event(sequence, observed, kind, **fields):
    return {"sequence": sequence, "eventNanos": observed - 1, "observedNanos": observed,
            "kind": kind, "origin": "physical", **fields}


def test_control_history_does_not_backdate_late_input_or_teacher_actions():
    history = ControlHistory()
    history.apply(event(0, 10, "pointer", origin="reconciliation", x=-285, y=107.5, modifiers=0))
    before = history.features(20, [surface()], interval_covered=True)
    assert before[13] == 0 and before[176] == 1
    late = event(1, 30, "keyDown", keyCode=13, eventNanos=15)
    history.apply(late)
    with pytest.raises(RecordingError, match="leak"):
        history.features(20, [surface()], interval_covered=True)
    after = history.features(30, [surface()], interval_covered=True)
    assert after[13] == 1
    np.testing.assert_allclose(after[168:170], [0.25, 0.25])
    history.apply(event(2, 31, "keyRepeat", keyCode=13))
    history.apply(event(3, 32, "keyUp", keyCode=13))
    assert history.features(32, [surface()], interval_covered=True)[13] == 0


def test_missing_input_invalidates_state_and_interval_accumulators_are_transient():
    history = ControlHistory()
    history.apply(event(0, 10, "pointer", origin="reconciliation", x=-280, y=110))
    history.apply(event(1, 20, "pointer", x=-275, y=112, dx=5, dy=2))
    history.apply(event(2, 21, "scroll", scrollX=0.5, scrollY=-1.25))
    first = history.features(30, [surface()], interval_covered=True)
    assert first[170] > 0 and first[173] < 0
    history.advance_interval()
    np.testing.assert_array_equal(history.features(31, [surface()], interval_covered=True)[170:174], np.zeros(4))
    history.apply(event(4, 40, "keyUp", keyCode=13))
    assert history.features(40, [surface()], interval_covered=True)[176] == 0


def test_global_detail_padding_and_native_cursor_coordinates_are_explicit():
    config = replace(ModelConfig.test_small(), global_long_edge=64, detail_long_edge=128)
    pixels = np.zeros((60, 120, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    pixels[28:32, 58:62, :3] = [10, 20, 250]
    descriptor = surface()
    metadata = {"id": str(uuid.uuid4()), "eventNanos": 1, "observedNanos": 2, "surface": descriptor,
                "byteCount": pixels.nbytes, "codec": "raw", "pixelFormat": "bgra8-srgb"}
    prepared = prepare_surface(pixels, metadata, config, pointer=(-270, 115))
    assert prepared.global_image.shape == (32, 64, 3)
    assert prepared.detail_image.shape == (64, 120, 3)
    np.testing.assert_allclose(prepared.global_content_rect, [0, 0, 1, 1])
    np.testing.assert_allclose(prepared.content_rect, [0, 2 / 64, 1, 60 / 64])
    expected_color = (np.array([250, 20, 10]) / 255 - MEAN) / STD
    np.testing.assert_allclose(prepared.cursor_image[16, 16], expected_color, atol=1e-6)
    np.testing.assert_allclose(prepared.cursor_rect, [(60 - 16) / 120, (30 - 16) / 60, 32 / 120, 32 / 60])
    assert prepared.batch().global_content_rect.shape == (1, 1, 4)
    unknown = prepare_surface(pixels, metadata, config, pointer=None)
    assert np.all(unknown.cursor_image == 0)
    assert np.all(unknown.cursor_rect[:2] > 1)


def test_feature_fusion_aligns_surface_coordinates_across_different_padding():
    # Source content occupies the middle two cells. Destination has no padding.
    # Their pixel centers should align exactly, not stretch the padded edges.
    values = mx.array([[[[100.0], [1.0], [2.0], [100.0]]]])
    aligned = align_features(values, 1, 2, mx.array([[0.25, 0, 0.5, 1]]), mx.array([[0, 0, 1, 1]]))
    np.testing.assert_allclose(np.asarray(aligned).ravel(), [1, 2], atol=1e-6)
    gradient = mx.grad(lambda x: mx.sum(align_features(x, 1, 2, mx.array([[0.25, 0, 0.5, 1]]), mx.array([[0, 0, 1, 1]]))))(values)
    np.testing.assert_array_equal(np.asarray(gradient).ravel(), [0, 1, 1, 0])


def test_future_source_input_cannot_enter_available_control_history():
    history = ControlHistory()
    with pytest.raises(RecordingError, match="availability"):
        history.apply(event(0, 10, "pointer", origin="reconciliation", eventNanos=11, x=0, y=0))
    assert not history.valid and history.sequence is None


def test_training_lanes_reject_gaps_and_missing_episode_resets():
    from astra.data.batching import LearningSample, training_batch
    from astra.model.actions import ActionVocabulary
    from test_policy_core import observations
    config = ModelConfig.test_small()
    first = LearningSample(replace(observations(batch=1, time=1, surfaces=1), reset=mx.array([[True]])),
                           (surface(),), (), "episode-one", 0)
    second = replace(first, observation=replace(first.observation, reset=mx.array([[False]])), step=1)
    bad_rows = ([first, None, second], [second, first], [first, replace(second, episode_id="episode-two")],
                [first, replace(second, surfaces=(surface(width=122),))], [replace(first, observation=second.observation)])
    for row in bad_rows:
        with pytest.raises(ValueError):
            training_batch([row], config, ActionVocabulary())
    observation, _ = training_batch([[second, None]], config, ActionVocabulary())
    assert observation.shape == (1, 1)  # A TBPTT cut can start after zero; common padding is trimmed.
    mixed, labels = training_batch([[first, second, None], [first, None, None]], config, ActionVocabulary())
    assert mixed.shape == (2, 2)
    np.testing.assert_array_equal(np.asarray(mixed.valid), [[True, True], [True, False]])
    assert labels.operation.shape == (4, config.packet_capacity + 1)


@pytest.mark.parametrize("field,value", [("x", 1e300), ("width", 1e-300)])
def test_geometry_must_fit_the_models_finite_nonempty_representation(field, value):
    descriptor = surface()
    descriptor["globalBounds"][field] = value
    pixels = np.zeros((60, 120, 4), dtype=np.uint8)
    metadata = {"id": str(uuid.uuid4()), "eventNanos": 1, "observedNanos": 2, "surface": descriptor,
                "byteCount": pixels.nbytes, "codec": "raw", "pixelFormat": "bgra8-srgb"}
    with pytest.raises(RecordingError, match="numerical range"):
        prepare_surface(pixels, metadata, ModelConfig.test_small(), pointer=None)
