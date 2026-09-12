from copy import deepcopy
from dataclasses import replace
import json
import queue
import threading
import uuid

import mlx.core as mx
import numpy as np
import pytest

from astra.data.preprocessing import prepare_surface
from astra.checkpoints import load_checkpoint, save_checkpoint
from astra.data.actions import decode_commands
from astra.data.observations import make_observation
from astra.inference import InferenceError, InferenceManager, InferenceSession, prepare_metal_surface
from astra.model.actions import ActionVocabulary, PacketBatch, flatten_visual
from astra.model.config import ModelConfig
from astra.model.policy import AgentPolicy
from astra.protocol import Message
from test_frame_ring import native_ring, ring_fixture_executable, _message
from test_jobs import Worker, worker_command


def _image(pattern, width=63, height=49):
    y, x = np.mgrid[:height, :width]
    if pattern == "uniform":
        rgb = np.broadcast_to(np.array([43, 111, 219], np.uint8), (height, width, 3)).copy()
    elif pattern == "checkerboard":
        rgb = np.repeat((((x + y) % 2) * 255)[..., None], 3, -1).astype(np.uint8)
    elif pattern == "text":
        foreground = ((x % 7 == 0) | (y % 11 == 0) | ((x % 13 < 3) & (y % 9 < 7)))
        rgb = np.where(foreground[..., None], np.array([8, 23, 241]), np.array([246, 221, 193])).astype(np.uint8)
    else:
        rgb = np.random.default_rng(1729).integers(0, 256, (height, width, 3), dtype=np.uint8)
    alpha = np.full((height, width, 1), 255, dtype=np.uint8)
    if pattern == "alpha":
        alpha = ((x * 17 + y * 31) % 256).astype(np.uint8)[..., None]
        rgb = (rgb.astype(np.float32) * alpha / 255).round().astype(np.uint8)
    return np.concatenate((rgb[..., ::-1], alpha), -1)


def _metadata(pixels, fractional=False):
    height, width = pixels.shape[:2]
    return {"id": str(uuid.uuid4()), "eventNanos": 10, "observedNanos": 11,
            "byteCount": pixels.nbytes, "codec": "raw", "pixelFormat": "bgra8-srgb",
            "surface": {"id": "display:α", "pixelWidth": width, "pixelHeight": height, "geometryRevision": 7,
                        "globalBounds": {"x": -127.25, "y": 31.125, "width": width / 2.5, "height": height / 1.75},
                        "contentBounds": {"x": .375 if fractional else 0, "y": 1.125 if fractional else 0,
                                          "width": width - .875 if fractional else width,
                                          "height": height - 1.375 if fractional else height}}}


@pytest.mark.parametrize("pattern", ["uniform", "checkerboard", "text", "alpha"])
@pytest.mark.parametrize("fractional", [False, True])
def test_metal_preprocessing_matches_training_pixels_and_geometry(pattern, fractional, monkeypatch):
    pixels = _image(pattern)
    metadata = _metadata(pixels, fractional)
    config = replace(ModelConfig.test_small(), global_long_edge=32, detail_long_edge=64)
    pointer = (-126.75, 33.125)  # Partial native crop at a negative-origin surface edge.
    reference = prepare_surface(pixels, metadata, config, pointer=pointer)
    owned = mx.array(pixels)
    original_asarray = np.asarray
    def no_image_readback(value, *args, **kwargs):
        if isinstance(value, mx.array):
            pytest.fail("Actor preprocessing attempted a GPU-to-CPU image transfer")
        return original_asarray(value, *args, **kwargs)
    with monkeypatch.context() as checks:
        checks.setattr(np, "asarray", no_image_readback)
        prepared = prepare_metal_surface(owned, metadata, config, pointer=pointer).batch()
        mx.eval(tuple(getattr(prepared, field) for field in prepared.__dataclass_fields__))
    for field in reference.__dataclass_fields__:
        actual = np.asarray(getattr(prepared, field))[0, 0]
        np.testing.assert_allclose(actual, getattr(reference, field), rtol=0, atol=1e-6, err_msg=field)


def test_unknown_and_off_surface_cursor_crops_preserve_training_masks():
    pixels = _image("text", width=31, height=23)
    metadata = _metadata(pixels)
    config = ModelConfig.test_small()
    for pointer in (None, (-1000, -900), (-123, 40)):
        expected = prepare_surface(pixels, metadata, config, pointer=pointer)
        result = prepare_metal_surface(mx.array(pixels), metadata, config, pointer=pointer).batch()
        for name in ("cursor_image", "cursor_rect", "global_content_rect", "content_rect"):
            np.testing.assert_allclose(np.asarray(getattr(result, name))[0, 0], getattr(expected, name), rtol=0, atol=1e-6)


@pytest.fixture
def checkpoint(tmp_path):
    mx.random.seed(751)
    model = AgentPolicy(ModelConfig.test_small(), ActionVocabulary(key_codes=(0, 13), mouse_buttons=(0,), absolute_pointer=True, scroll=True))
    destination = tmp_path / str(uuid.uuid4())
    save_checkpoint(destination, model, kind="initial", step=0)
    return destination


def _prepare(checkpoint, report, *, deterministic=True):
    return {"checkpointPath": str(checkpoint), "ring": {"path": report["path"], "ringID": report["reference"]["ringID"]},
            "seed": 72, "deterministic": deterministic}


def _step(report, state, *, cutoff=None):
    reference = report["reference"]
    cutoff = reference["metadata"]["observedNanos"] if cutoff is None else cutoff
    return {"observationID": str(uuid.uuid4()), "episodeID": state["episodeID"], "previousStateID": state["stateID"],
            "cutoffNanos": cutoff, "geometryRevision": 17, "frames": [reference], "contextIDs": [],
            "controlState": {"keys": [], "buttons": [], "modifiers": 0, "pointer": {"x": -1919, "y": 1},
                             "observedNanos": cutoff, "revision": 1, "valid": True},
            "executedEvents": [], "intervalCovered": True}


def _acknowledge(process, acknowledgement):
    process.stdin.write(json.dumps({"release": acknowledgement}).encode() + b"\n"); process.stdin.flush()


def test_collection_record_replays_exact_packet_anchor_and_rng_without_resampling(native_ring, checkpoint):
    producer, report = native_ring
    run_id = report["reference"]["runID"]
    actor = InferenceSession()
    try:
        ready = actor.prepare({**_prepare(checkpoint, report, deterministic=False), "collection": True}, run_id=run_id)
        assert ready["collectionVersion"] == 1 and ready["collection"]
        state = actor.reset({"confirmed": True, "episodeID": str(uuid.uuid4()), "contextIDs": []}, run_id=run_id)
        request = _step(report, state)
        first = actor.step(request, run_id=run_id)
        record = first["collectionRecord"]
        assert record["frameIDs"] == [str(uuid.UUID(report["reference"]["metadata"]["id"]))]
        assert record["previousStateID"] == state["stateID"] and record["nextStateID"] == first["stateID"]
        assert record["episodeStep"] == record["sampler"]["drawIndex"] == 0
        assert record["environmentResets"] == 1 and record["recurrentReset"] is True
        assert record["elapsedSeconds"] == .1
        assert not np.any(record["stateBefore"])
        assert record["sampler"]["rngStreamID"] == ready["rngStreamID"]
        np.testing.assert_array_equal(record["sampler"]["stateBefore"], np.asarray(mx.random.key(72)))
        keys = np.asarray(mx.random.split(mx.random.key(72)))
        np.testing.assert_array_equal(record["sampler"]["sampleKey"], keys[1])
        np.testing.assert_array_equal(record["sampler"]["stateAfter"], keys[0])

        policy = load_checkpoint(checkpoint).policy
        pixels = np.array([np.uint8((value * 17) % 256) for value in range(140)], np.uint8).reshape(5, 7, 4)
        observation = make_observation([(pixels, report["reference"]["metadata"])], request["controlState"],
            cutoff_nanos=record["cutoffNanos"], elapsed_seconds=record["elapsedSeconds"], reset=record["recurrentReset"],
            config=policy.config, context_ids=record["contextIDs"], executed_events=[], maximum_timestamp=2**64 - 1)
        anchor = tuple(mx.array([layer], dtype=mx.float32) for layer in record["stateBefore"])
        encoding = policy(observation, anchor)
        packet = PacketBatch(**{name: mx.array([values], dtype=mx.int32) for name, values in record["packetFields"].items()})
        decoded = decode_commands(packet, config=policy.config, vocabulary=policy.actions.vocabulary,
            visual=flatten_visual(encoding.visual), surfaces=[report["reference"]["metadata"]["surface"]])
        assert decoded == first["packet"]["commands"]
        scored = policy.score(encoding, packet)
        assert float(scored.log_probability.item()) == pytest.approx(record["logProbability"], abs=1e-5)
        assert float(encoding.temporal.value.item()) == pytest.approx(record["value"], abs=1e-6)
        json.dumps(first, allow_nan=False)
        saved_anchor = [np.asarray(value)[0].copy() for value in actor._state]
        _acknowledge(producer, first["releasedFrames"][0])
        second_report = _message(producer)
        second = actor.step(_step(second_report, first, cutoff=request["cutoffNanos"] + 100_123_456), run_id=run_id)
        followup = second["collectionRecord"]
        assert followup["elapsedSeconds"] == .100123456 and followup["recurrentReset"] is False
        np.testing.assert_array_equal(followup["stateBefore"], saved_anchor)
        assert followup["episodeStep"] == followup["sampler"]["drawIndex"] == 1
        assert followup["sampler"]["stateBefore"] == record["sampler"]["stateAfter"]
        _acknowledge(producer, second["releasedFrames"][0]); assert _message(producer)["closed"]
        assert producer.wait(timeout=10) == 0
    finally:
        actor.close()


def test_collection_warmup_has_no_record_and_restores_all_stream_counters(native_ring, checkpoint):
    producer, report = native_ring
    run_id = report["reference"]["runID"]
    actor = InferenceSession()
    try:
        with pytest.raises(InferenceError, match="categorical"):
            actor.prepare({**_prepare(checkpoint, report), "collection": True}, run_id=run_id)
        actor.prepare({**_prepare(checkpoint, report, deterministic=False), "collection": True}, run_id=run_id)
        state = actor.reset({"confirmed": True, "episodeID": str(uuid.uuid4()), "contextIDs": []}, run_id=run_id)
        key = np.asarray(actor._key).copy()
        warmed = actor.warmup(_step(report, state), run_id=run_id)
        assert "collectionRecord" not in warmed
        assert actor._draw_index == actor._episode_step == actor._sequence == 0
        assert actor._environment_resets == 1 and not actor._warming
        np.testing.assert_array_equal(np.asarray(actor._key), key)
        _acknowledge(producer, warmed["releasedFrames"][0])
        second_report = _message(producer)
        actual = actor.step(_step(second_report, state), run_id=run_id)
        assert actual["collectionRecord"]["sampler"]["drawIndex"] == 0
        _acknowledge(producer, actual["releasedFrames"][0]); assert _message(producer)["closed"]
        assert producer.wait(timeout=10) == 0
    finally:
        actor.close()


def test_collection_episode_reset_preserves_random_stream_and_zeroes_only_recurrence(native_ring, checkpoint):
    producer, report = native_ring
    run_id = report["reference"]["runID"]
    actor = InferenceSession()
    try:
        actor.prepare({**_prepare(checkpoint, report, deterministic=False), "collection": True}, run_id=run_id)
        state = actor.reset({"confirmed": True, "episodeID": str(uuid.uuid4()), "contextIDs": []}, run_id=run_id)
        first_request = _step(report, state)
        first = actor.step(first_request, run_id=run_id)
        reset_request = {"confirmed": True, "episodeID": str(uuid.uuid4()), "contextIDs": []}
        with pytest.raises(InferenceError, match="random stream"):
            actor.reset({**reset_request, "seed": 72}, run_id=run_id)
        assert actor._needs_reset
        state = actor.reset(reset_request, run_id=run_id)
        _acknowledge(producer, first["releasedFrames"][0]); second_report = _message(producer)
        second = actor.step(_step(second_report, state, cutoff=first_request["cutoffNanos"] + 100_000_000), run_id=run_id)
        record = second["collectionRecord"]
        assert record["episodeStep"] == 0 and record["environmentResets"] == 2 and record["recurrentReset"]
        assert not np.any(record["stateBefore"])
        assert record["sampler"]["drawIndex"] == 1
        assert record["sampler"]["stateBefore"] == first["collectionRecord"]["sampler"]["stateAfter"]
        assert record["sampler"]["rngStreamID"] == first["collectionRecord"]["sampler"]["rngStreamID"]
        _acknowledge(producer, second["releasedFrames"][0]); assert _message(producer)["closed"]
        assert producer.wait(timeout=10) == 0
    finally:
        actor.close()


def test_warmup_preserves_actor_state_and_rng_but_consumes_its_frame_lease(native_ring, checkpoint):
    producer, report = native_ring
    run_id = report["reference"]["runID"]
    actor = InferenceSession()
    try:
        actor.prepare(_prepare(checkpoint, report, deterministic=False), run_id=run_id)
        state = actor.reset({"confirmed": True, "episodeID": str(uuid.uuid4()), "contextIDs": [], "seed": 72}, run_id=run_id)
        initial_key = np.asarray(actor._key).copy()
        request = _step(report, state)
        warmed = actor.warmup(request, run_id=run_id)
        assert warmed["warmup"] and warmed["packet"]["sequence"] == 0
        assert actor._state is None and actor._state_id == state["stateID"]
        assert actor._sequence == 0 and actor._last_cutoff is None and not actor._observations
        np.testing.assert_array_equal(np.asarray(actor._key), initial_key)
        # A discarded prediction does not permit consuming/acknowledging the
        # same lease again. Only a freshly published frame can advance a run.
        with pytest.raises(InferenceError, match="stale") as failure:
            actor.warmup(request, run_id=run_id)
        assert failure.value.released_frames == []
        assert actor._needs_reset is False
        _acknowledge(producer, warmed["releasedFrames"][0])
        second = _message(producer)
        real = actor.step(_step(second, state), run_id=run_id)
        assert real["packet"]["sequence"] == 0
        expected_key = mx.random.split(mx.random.key(72))[0]
        np.testing.assert_array_equal(np.asarray(actor._key), np.asarray(expected_key))
        with pytest.raises(InferenceError, match="before the first"):
            actor.warmup(_step(second, real), run_id=run_id)
        _acknowledge(producer, real["releasedFrames"][0])
        assert _message(producer)["closed"]
        assert producer.wait(timeout=10) == 0
    finally:
        actor.close()


def test_warmup_processing_failure_keeps_reset_state_and_releases_owned_copy(native_ring, checkpoint, monkeypatch):
    import astra.inference as implementation
    producer, report = native_ring
    run_id = report["reference"]["runID"]
    actor = InferenceSession()
    try:
        actor.prepare(_prepare(checkpoint, report), run_id=run_id)
        state = actor.reset({"confirmed": True, "episodeID": str(uuid.uuid4()), "contextIDs": []}, run_id=run_id)
        def failure(*args, **kwargs):
            raise RuntimeError("controlled preparation failure")
        with monkeypatch.context() as patch:
            patch.setattr(implementation, "prepare_metal_surface", failure)
            with pytest.raises(InferenceError, match="controlled preparation") as error:
                actor.warmup(_step(report, state), run_id=run_id)
        assert actor._state_id == state["stateID"] and actor._state is None and not actor._needs_reset
        assert actor._sequence == 0 and actor._last_cutoff is None
        assert len(error.value.released_frames) == 1
        _acknowledge(producer, error.value.released_frames[0])
        second = _message(producer)
        actual = actor.step(_step(second, state), run_id=run_id)
        assert actual["packet"]["sequence"] == 0
        _acknowledge(producer, actual["releasedFrames"][0])
        assert _message(producer)["closed"]
        assert producer.wait(timeout=10) == 0
    finally:
        actor.close()


@pytest.mark.parametrize("deterministic", [True, False])
def test_real_checkpoint_ring_step_matches_training_policy_and_carries_recurrent_state(native_ring, checkpoint, deterministic):
    producer, report = native_ring
    run_id = report["reference"]["runID"]
    actor = InferenceSession()
    try:
        ready = actor.prepare(_prepare(checkpoint, report, deterministic=deterministic), run_id=run_id)
        assert ready["needsReset"]
        with pytest.raises(InferenceError, match="confirmed"):
            actor.step({}, run_id=run_id)
        state = actor.reset({"confirmed": True, "episodeID": str(uuid.uuid4()), "contextIDs": [], "seed": 72}, run_id=run_id)
        request = _step(report, state)
        input_time = request["cutoffNanos"] - 3
        request["controlState"]["keys"] = [13]
        request["executedEvents"] = [{"sequence": 1, "eventNanos": input_time - 1, "observedNanos": input_time,
                                      "origin": "agent", "kind": "keyDown", "keyCode": 13}]
        first = actor.step(request, run_id=run_id)
        assert first["packet"]["sequence"] == 0
        assert first["packet"]["executeAtNanos"] == request["cutoffNanos"] + 100_000_000
        assert first["packet"]["durationMs"] == 100
        assert first["stateID"] != state["stateID"] and first["checkpointID"] == checkpoint.name
        with pytest.raises(InferenceError, match="state"):
            actor.step(request, run_id=run_id)

        # Independent CPU training preparation and the same loaded policy must
        # produce identical commands, exact joint score and persistent state.
        reference_policy = load_checkpoint(checkpoint).policy
        pixels = np.array([np.uint8((value * 17) % 256) for value in range(140)], np.uint8).reshape(5, 7, 4)
        observation = make_observation([(pixels, report["reference"]["metadata"])], request["controlState"],
                                       cutoff_nanos=request["cutoffNanos"], elapsed_seconds=.1, reset=True,
                                       config=reference_policy.config, maximum_timestamp=2**64 - 1,
                                       executed_events=request["executedEvents"])
        encoding = reference_policy(observation)
        expected = reference_policy.sample(encoding, key=mx.random.split(mx.random.key(72))[1], greedy=deterministic)
        expected_commands = decode_commands(expected.packets, config=reference_policy.config, vocabulary=reference_policy.actions.vocabulary,
                                            visual=flatten_visual(encoding.visual), surfaces=[report["reference"]["metadata"]["surface"]])
        assert first["packet"]["commands"] == expected_commands
        assert first["logProbability"] == pytest.approx(float(expected.log_probability.item()), abs=1e-5)
        assert first["value"] == pytest.approx(float(encoding.temporal.value.item()), abs=1e-6)
        for actual, expected_state in zip(actor._state, encoding.temporal.state):
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected_state), rtol=0, atol=1e-6)

        _acknowledge(producer, first["releasedFrames"][0])
        second_report = _message(producer)
        second_request = _step(second_report, first, cutoff=request["cutoffNanos"] + 100_000_000)
        second_request["controlState"]["keys"] = [13]
        second = actor.step(second_request, run_id=run_id)
        assert actor._last_input_nanos == input_time
        assert second["packet"]["sequence"] == 1 and second["stateID"] != first["stateID"]
        second_pixels = np.bitwise_xor(pixels, 0xA5)
        second_observation = make_observation([(second_pixels, second_report["reference"]["metadata"])], second_request["controlState"],
                                             cutoff_nanos=second_request["cutoffNanos"], elapsed_seconds=.1, reset=False,
                                             config=reference_policy.config, maximum_timestamp=2**64 - 1, last_input_nanos=input_time)
        second_encoding = reference_policy(second_observation, encoding.temporal.state)
        for actual, expected_state in zip(actor._state, second_encoding.temporal.state):
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected_state), rtol=0, atol=1e-6)
        _acknowledge(producer, second["releasedFrames"][0])
        assert _message(producer) == {"closed": True, "pathExists": False}
        assert producer.wait(timeout=10) == 0
    finally:
        actor.close()


def test_last_executed_input_age_survives_two_idle_intervals_and_rejects_future_history():
    pixels = _image("uniform", width=32, height=32)
    metadata = _metadata(pixels)
    control = {"keys": [13], "buttons": [], "modifiers": 0, "pointer": {"x": -126, "y": 32},
               "observedNanos": 20, "revision": 1, "valid": True}
    event = {"sequence": 1, "eventNanos": 18, "observedNanos": 19, "origin": "agent", "kind": "keyDown", "keyCode": 13}
    ages = []
    for index, cutoff in enumerate((20, 100_000_020, 200_000_020)):
        observation = make_observation([(pixels, metadata)], control, cutoff_nanos=cutoff, elapsed_seconds=.1,
                                       reset=index == 0, config=ModelConfig.test_small(),
                                       executed_events=[event] if index == 0 else [], last_input_nanos=None if index == 0 else 19)
        ages.append(float(observation.controls[0, 0, 174].item()))
    assert 0 <= ages[0] < ages[1] < ages[2] < 1
    with pytest.raises(ValueError, match="Last executed input"):
        make_observation([(pixels, metadata)], control, cutoff_nanos=20, elapsed_seconds=.1, reset=False,
                         config=ModelConfig.test_small(), last_input_nanos=21)


def test_actor_releases_owned_frames_on_processing_failure_and_requires_reset(native_ring, checkpoint):
    producer, report = native_ring
    run_id = report["reference"]["runID"]
    actor = InferenceSession()
    try:
        actor.prepare(_prepare(checkpoint, report), run_id=run_id)
        state = actor.reset({"confirmed": True, "episodeID": str(uuid.uuid4()), "contextIDs": []}, run_id=run_id)
        request = _step(report, state)
        request["controlState"]["observedNanos"] += 1  # Future physical state cannot reach the policy.
        with pytest.raises(InferenceError) as failure:
            actor.step(request, run_id=run_id)
        assert len(failure.value.released_frames) == 1 and actor._needs_reset
        assert actor._state is None
        _acknowledge(producer, failure.value.released_frames[0])
        next_report = _message(producer)
        with pytest.raises(InferenceError, match="reset"):
            actor.step(_step(next_report, state), run_id=run_id)
        with pytest.raises(InferenceError, match="confirmed"):
            actor.reset({"confirmed": False, "episodeID": str(uuid.uuid4()), "contextIDs": []}, run_id=run_id)
        state = actor.reset({"confirmed": True, "episodeID": str(uuid.uuid4()), "contextIDs": []}, run_id=run_id)
        invalid = _step(next_report, state)
        invalid["frames"] = deepcopy(invalid["frames"])
        invalid["frames"][0]["runID"] = str(uuid.uuid4())
        with pytest.raises(InferenceError) as rejected:
            actor.step(invalid, run_id=run_id)
        assert rejected.value.released_frames == [] and actor._needs_reset
        state = actor.reset({"confirmed": True, "episodeID": str(uuid.uuid4()), "contextIDs": []}, run_id=run_id)
        result = actor.step(_step(next_report, state), run_id=run_id)
        _acknowledge(producer, result["releasedFrames"][0])
        assert _message(producer)["closed"]
        assert producer.wait(timeout=10) == 0
    finally:
        actor.close()


def test_checkpoint_activation_is_reset_only_and_causal_events_are_not_replayed(native_ring, checkpoint, tmp_path):
    _, report = native_ring
    run_id = report["reference"]["runID"]
    actor = InferenceSession()
    try:
        actor.prepare(_prepare(checkpoint, report), run_id=run_id)
        with pytest.raises(InferenceError, match="another stream"):
            actor.prepare(_prepare(checkpoint, report), run_id=run_id)
        with pytest.raises(InferenceError, match="run"):
            actor.reset({}, run_id=str(uuid.uuid4()))
        state = actor.reset({"confirmed": True, "episodeID": str(uuid.uuid4()), "contextIDs": []}, run_id=run_id)
        request = _step(report, state)
        request["executedEvents"] = [{"sequence": 1, "eventNanos": request["cutoffNanos"] + 1,
                                      "observedNanos": request["cutoffNanos"], "origin": "agent", "kind": "keyDown", "keyCode": 13}]
        with pytest.raises(InferenceError, match="causal"):
            actor.step(request, run_id=run_id)
        assert actor._state_id == state["stateID"] and not actor._needs_reset
        changed = load_checkpoint(checkpoint).policy
        changed.temporal.value_head.bias = changed.temporal.value_head.bias + 1
        destination = tmp_path / str(uuid.uuid4())
        save_checkpoint(destination, changed, kind="reinforcement", step=1, parent_id=checkpoint.name)
        activated = actor.reset({"confirmed": True, "episodeID": str(uuid.uuid4()), "contextIDs": [], "checkpointPath": str(destination)}, run_id=run_id)
        assert activated["checkpointID"] == destination.name and activated["stateID"] != state["stateID"]
        assert actor._state is None
        changed.actions.vocabulary = ActionVocabulary(key_codes=(0,))
        incompatible = tmp_path / str(uuid.uuid4())
        save_checkpoint(incompatible, changed, kind="initial", step=0)
        with pytest.raises(InferenceError, match="new actor run"):
            actor.reset({"confirmed": True, "episodeID": str(uuid.uuid4()), "contextIDs": [], "checkpointPath": str(incompatible)}, run_id=run_id)
        assert actor._identity()["checkpointID"] == activated["checkpointID"]
        assert actor._identity()["stateID"] == activated["stateID"] and actor._needs_reset
    finally:
        actor.close()


def test_actor_owner_thread_rejects_overlap_and_joins_before_close(native_ring, checkpoint, monkeypatch):
    _, report = native_ring
    entered, resume = threading.Event(), threading.Event()
    delivered = queue.Queue()
    original = InferenceSession.prepare
    owner = threading.get_ident()
    def delayed(self, payload, *, run_id):
        assert threading.get_ident() != owner
        entered.set()
        assert resume.wait(10)
        return original(self, payload, run_id=run_id)
    monkeypatch.setattr(InferenceSession, "prepare", delayed)
    manager = InferenceManager(lambda kind, payload, request=None: delivered.put((kind, payload, request)))
    request = Message("inference.prepare", 0, _prepare(checkpoint, report), request_id=str(uuid.uuid4()), run_id=report["reference"]["runID"])
    try:
        manager.submit(request)
        assert entered.wait(10) and manager.busy
        with pytest.raises(InferenceError, match="already owns"):
            manager.submit(request)
        assert not manager.close(timeout=.01)
        resume.set()
        assert manager.close(timeout=10)
        kind, result, replied = delivered.get(timeout=10)
        assert kind == "error" and result["code"] == "inference.cancelled"
        assert replied == request and not manager.busy
    finally:
        resume.set()
        assert manager.close(timeout=10)


@pytest.fixture
def actor_worker(monkeypatch):
    monkeypatch.setattr("test_jobs.worker_command", lambda: [*worker_command(), "--role", "actor"])
    instance = Worker()
    try:
        yield instance
    finally:
        instance.close()


def test_worker_actor_role_runs_real_ring_checkpoint_and_returns_release_on_error(actor_worker, native_ring, checkpoint):
    producer, report = native_ring
    run_id = report["reference"]["runID"]
    assert actor_worker.hello["payload"]["role"] == "actor"
    assert "inference.step" in actor_worker.hello["payload"]["capabilities"]
    assert "train.behavioral" not in actor_worker.hello["payload"]["capabilities"]
    assert actor_worker.request("train.behavioral", {}, run_id=run_id)["kind"] == "error"
    preparing = actor_worker.send("inference.prepare", _prepare(checkpoint, report), run_id=run_id)
    assert actor_worker.request("ping")["payload"]["alive"]
    prepared = actor_worker.until(lambda event: event.get("requestID") == preparing)
    assert prepared["kind"] == "ack", prepared
    reset = actor_worker.request("inference.reset", {"confirmed": True, "episodeID": str(uuid.uuid4()), "contextIDs": []}, run_id=run_id)
    assert reset["kind"] == "ack", reset
    request = _step(report, reset["payload"])
    request["controlState"]["observedNanos"] += 1
    failed = actor_worker.request("inference.step", request, run_id=run_id)
    assert failed["kind"] == "error" and failed["payload"]["needsReset"]
    assert len(failed["payload"]["releasedFrames"]) == 1
    _acknowledge(producer, failed["payload"]["releasedFrames"][0])
    second = _message(producer)
    reset = actor_worker.request("inference.reset", {"confirmed": True, "episodeID": str(uuid.uuid4()), "contextIDs": []}, run_id=run_id)
    result = actor_worker.request("inference.step", _step(second, reset["payload"]), run_id=run_id)
    assert result["kind"] == "ack", result
    assert result["payload"]["packet"]["observationID"] != request["observationID"]
    assert np.isfinite(result["payload"]["logProbability"]) and np.isfinite(result["payload"]["value"])
    _acknowledge(producer, result["payload"]["releasedFrames"][0])
    assert _message(producer)["closed"]
    assert producer.wait(timeout=10) == 0
    closed = actor_worker.request("inference.close", run_id=run_id)
    assert closed["kind"] == "ack" and closed["payload"] == {"closed": True}
    sequences = [message["sequence"] for message in actor_worker.received]
    assert sequences == list(range(len(sequences)))


@pytest.mark.parametrize('greedy', [False, True])
def test_compiled_actor_matches_eager_packets_probabilities_and_recurrent_contexts(greedy):
    from astra.inference import _PolicyExecution
    from astra.model.actions import PacketBatch
    from astra.model.vision import VisualFeatures

    mx.random.seed(913)
    config = replace(ModelConfig.test_small(), context_sizes=(3,))
    vocabulary = ActionVocabulary(key_codes=(0, 13), mouse_buttons=(0, 2), absolute_pointer=True, relative_pointer=True, scroll=True)
    policy = AgentPolicy(config, vocabulary); policy.eval()
    eager, compiled = _PolicyExecution(policy, greedy=greedy, compiled=False), _PolicyExecution(policy, greedy=greedy)
    states, keys = [None, None], [mx.random.key(72), mx.random.key(72)]
    for index, pattern in enumerate(('text', 'checkerboard', 'alpha', 'uniform')):
        pixels = _image(pattern); metadata = _metadata(pixels, fractional=True)
        control = dict(keys=[13] if index == 2 else [], buttons=[], modifiers=0, pointer=dict(x=-126 + index, y=34),
                       observedNanos=20, revision=index, valid=True)
        observation = make_observation([(mx.array(pixels), metadata)], control, cutoff_nanos=20 + index * 100_000_000,
                                       elapsed_seconds=.1, reset=index in (0, 2), config=config, context_ids=(index % 3,),
                                       surface_preparer=prepare_metal_surface)
        outputs = [runner(observation, state, key) for runner, state, key in zip((eager, compiled), states, keys)]
        mx.eval(outputs)
        for name in PacketBatch.__dataclass_fields__:
            np.testing.assert_array_equal(outputs[0]['packet'][name], outputs[1]['packet'][name])
        for name in ('value', 'log_probability', 'conditional_entropy'):
            np.testing.assert_allclose(outputs[0][name], outputs[1][name], rtol=1e-5, atol=1e-5)
        for before, after in zip(outputs[0]['state'], outputs[1]['state']):
            np.testing.assert_allclose(before, after, rtol=1e-5, atol=2e-6)
        for name in ('cell_bounds', 'cell_valid', 'surface_valid'):
            np.testing.assert_array_equal(outputs[0]['visual'][name], outputs[1]['visual'][name])
        np.testing.assert_array_equal(outputs[0]['key'], outputs[1]['key'])
        for output in outputs:
            assert output['finite'].item()
            decode_commands(PacketBatch(**output['packet']), config=config, vocabulary=vocabulary,
                            visual=VisualFeatures(**output['visual']), surfaces=[metadata['surface']])
        states, keys = [output['state'] for output in outputs], [output['key'] for output in outputs]
    assert len(compiled._cache) == 1  # Pixels, context IDs and reset values are runtime inputs.


def test_compiled_geometry_cache_is_bounded_and_parameter_values_are_not_frozen():
    from astra.inference import _PolicyExecution

    mx.random.seed(813)
    config = ModelConfig.test_small()
    policy = AgentPolicy(config, ActionVocabulary(key_codes=(0,), absolute_pointer=True)); policy.eval()
    runner = _PolicyExecution(policy, greedy=True)
    runner.maximum_buckets = 2
    def observation(width):
        pixels = _image('text', width=width, height=32); metadata = _metadata(pixels)
        control = dict(keys=[], buttons=[], modifiers=0, pointer=dict(x=-126, y=32), observedNanos=20, revision=0, valid=True)
        return make_observation([(mx.array(pixels), metadata)], control, cutoff_nanos=20, elapsed_seconds=.1,
                                reset=True, config=config, surface_preparer=prepare_metal_surface)
    for width in (32, 48, 64):
        output = runner(observation(width), None, mx.random.key(1)); mx.eval(output)
        assert output['finite'].item()
        assert len(runner._cache) <= 2
    original = runner(observation(64), None, mx.random.key(1)); mx.eval(original)
    policy.temporal.value_head.bias = policy.temporal.value_head.bias + 1
    changed = runner(observation(64), None, mx.random.key(1)); mx.eval(changed)
    np.testing.assert_allclose(changed['value'] - original['value'], 1, rtol=0, atol=1e-6)
    # Revisiting an evicted geometry retraces with the current parameters.
    revisited = runner(observation(32), None, mx.random.key(1)); mx.eval(revisited)
    eager = _PolicyExecution(policy, greedy=True, compiled=False)(observation(32), None, mx.random.key(1)); mx.eval(eager)
    np.testing.assert_allclose(revisited['value'], eager['value'], rtol=1e-5, atol=1e-6)
    assert len(runner._cache) == 2


def test_checkpoint_activation_discards_compiled_old_policy(native_ring, checkpoint, tmp_path):
    producer, report = native_ring
    actor = InferenceSession(); run_id = report['reference']['runID']
    try:
        actor.prepare(_prepare(checkpoint, report), run_id=run_id)
        initial = actor.reset(dict(confirmed=True, episodeID=str(uuid.uuid4()), contextIDs=[]), run_id=run_id)
        first = actor.step(_step(report, initial), run_id=run_id)
        previous_execution = actor._execution
        assert len(previous_execution._cache) == 1
        _acknowledge(producer, first['releasedFrames'][0]); next_report = _message(producer)
        replacement = load_checkpoint(checkpoint).policy
        replacement.temporal.value_head.bias = replacement.temporal.value_head.bias + 2
        destination = tmp_path / str(uuid.uuid4())
        save_checkpoint(destination, replacement, kind='reinforcement', step=1, parent_id=checkpoint.name)
        state = actor.reset(dict(confirmed=True, episodeID=str(uuid.uuid4()), contextIDs=[], checkpointPath=str(destination)), run_id=run_id)
        assert actor._execution is not previous_execution and len(actor._execution._cache) == 0
        request = _step(next_report, state, cutoff=report['reference']['metadata']['observedNanos'] + 100_000_000)
        second = actor.step(request, run_id=run_id)
        assert second['checkpointID'] == destination.name and len(actor._execution._cache) == 1
        pixels = np.bitwise_xor(np.array([(value * 17) % 256 for value in range(140)], np.uint8).reshape(5, 7, 4), 0xA5)
        observation = make_observation([(pixels, next_report['reference']['metadata'])], request['controlState'],
                                       cutoff_nanos=request['cutoffNanos'], elapsed_seconds=.1, reset=True,
                                       config=replacement.config, maximum_timestamp=2**64 - 1)
        expected = replacement(observation)
        np.testing.assert_allclose(second['value'], expected.temporal.value.item(), rtol=1e-5, atol=1e-6)
        _acknowledge(producer, second['releasedFrames'][0]); assert _message(producer)['closed']
    finally:
        actor.close()
    assert actor._execution is None


def test_compiled_nonfinite_result_releases_frame_without_committing_actor_state(native_ring, checkpoint):
    producer, report = native_ring
    actor = InferenceSession(); run_id = report['reference']['runID']
    try:
        actor.prepare(_prepare(checkpoint, report), run_id=run_id)
        state = actor.reset(dict(confirmed=True, episodeID=str(uuid.uuid4()), contextIDs=[]), run_id=run_id)
        bias = actor._checkpoint.policy.temporal.value_head.bias
        key = np.asarray(actor._key).copy()
        actor._checkpoint.policy.temporal.value_head.bias = mx.full_like(bias, float('nan'))
        with pytest.raises(InferenceError, match='nonfinite') as failure:
            actor.step(_step(report, state), run_id=run_id)
        expected_ack = {name: report['reference'][name] for name in ('version', 'runID', 'ringID', 'slot', 'leaseID', 'sequence')}
        for name in ('runID', 'ringID', 'leaseID'): expected_ack[name] = str(uuid.UUID(expected_ack[name]))
        assert failure.value.released_frames == [expected_ack]
        assert actor._needs_reset and actor._state is None and actor._sequence == 0 and actor._state_id == state['stateID']
        np.testing.assert_array_equal(actor._key, key)
        _acknowledge(producer, failure.value.released_frames[0]); next_report = _message(producer)
        actor._checkpoint.policy.temporal.value_head.bias = bias
        state = actor.reset(dict(confirmed=True, episodeID=str(uuid.uuid4()), contextIDs=[]), run_id=run_id)
        result = actor.step(_step(next_report, state), run_id=run_id)
        assert np.isfinite(result['value']) and result['packet']['sequence'] == 0
        _acknowledge(producer, result['releasedFrames'][0]); assert _message(producer)['closed']
    finally:
        actor.close()
