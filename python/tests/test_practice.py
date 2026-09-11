from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from astra.environments import PracticeConfig, PracticeEnvironment, PracticeError
from astra.recordings import validate_event, validate_frame


def environment(**kwargs):
    return PracticeEnvironment(PracticeConfig(**{"pixel_width": 160, "pixel_height": 90, **kwargs}))


def absolute(x, y, offset=0):
    return {"operation": "pointerAbsolute", "offsetMs": offset, "surfaceID": "practice", "x": x, "y": y}


def button(operation="buttonDown", offset=0):
    return {"operation": operation, "offsetMs": offset, "button": 0}


def key(operation, code=123, offset=0):
    return {"operation": operation, "offsetMs": offset, "keyCode": code}


def relative(dx, dy, offset=0):
    return {"operation": "pointerRelative", "offsetMs": offset, "dx": dx, "dy": dy}


@pytest.mark.parametrize("task,delay", [("pointing", 2000), ("delayed_memory", 2000),
                                       ("delayed_memory", 8000), ("delayed_memory", 30000)])
def test_oracle_completes_heldout_trials_at_real_delays(task, delay):
    env = environment(task=task, delay_ms=delay, lead_ms=150)
    # These trials use the full stated delays; smaller images only reduce test
    # rendering cost, not temporal memory or execution-lead requirements.
    for seed in (1001, 2003, 3007):
        initial = env.reset(seed)
        elapsed, reward = 0, 0
        for _ in range(400):
            transition = env.step(env.oracle_commands(), initial.episode_id, provenance="oracle")
            elapsed += transition.duration_ms
            reward += transition.reward
            for event in transition.raw_events + transition.cleanup_events:
                validate_event(event)
                assert event["origin"] != "physical"
            if transition.outcome != "continuing":
                break
        assert transition.outcome == "terminated"
        assert reward == 1
        assert elapsed > 150 + (500 + delay if task == "delayed_memory" else 0)
        assert {entry["provenance"] for entry in transition.command_results} == {"oracle"}
        with pytest.raises(PracticeError, match="Reset"):
            env.step([])


def test_pixels_are_native_bgra_and_coordinates_use_logical_negative_origin():
    config = PracticeConfig(pixel_width=640, pixel_height=240, logical_bounds=(-512, -120, 320, 240), lead_ms=0)
    env = PracticeEnvironment(config)
    observation = env.reset(2)
    validate_frame(observation.metadata)
    assert observation.pixels.shape == (240, 640, 4)
    assert observation.pixels.dtype == np.uint8
    assert not observation.pixels.flags.writeable
    assert np.all(observation.pixels[..., 3] == 255)
    target = env.oracle_commands()[0]
    px, py = int(target["x"] * 640), int(target["y"] * 240)
    np.testing.assert_array_equal(observation.pixels[py, px], [158, 140, 20, 255])
    moved = env.step([absolute(.25, .75)]).observation
    assert moved.control_state["pointer"] == {"x": -432.0, "y": 60.0}
    assert moved.metadata["surface"]["globalBounds"] == {"x": -512, "y": -120, "width": 320, "height": 240}
    assert set(observation.metadata) == {"id", "eventNanos", "observedNanos", "surface", "byteCount", "pixelFormat", "codec"}
    assert set(observation.control_state) == {"keys", "buttons", "modifiers", "pointer", "observedNanos", "revision", "valid"}


def test_seeded_world_replays_and_unseen_layouts_vary():
    env = environment()
    first = env.reset(17)
    replay = env.reset(17)
    np.testing.assert_array_equal(first.pixels, replay.pixels)
    assert first.episode_id != replay.episode_id
    assert replay.metadata["observedNanos"] > first.metadata["observedNanos"]
    targets = []
    for seed in range(101, 121):
        env.reset(seed)
        command = env.oracle_commands()[0]
        targets.append((command["x"], command["y"]))
    assert len(set(targets)) == 20
    assert max(x for x, _ in targets) - min(x for x, _ in targets) > .5
    assert max(y for _, y in targets) - min(y for _, y in targets) > .4


def test_default_reset_varies_trials_reproducibly_and_configuration_roundtrips():
    config = PracticeConfig(seed=42, pixel_width=160, pixel_height=90, logical_bounds=(-20, 0, 640, 360))
    decoded = PracticeConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert config == decoded and config.signature == decoded.signature
    first, second = PracticeEnvironment(config), PracticeEnvironment(decoded)
    frames = []
    for seed in (42, 43, 44):
        a, b = first.reset(), second.reset()
        np.testing.assert_array_equal(a.pixels, b.pixels)
        assert first.current_seed == second.current_seed == seed
        frames.append(a.pixels)
    assert not np.array_equal(frames[0], frames[1])
    with pytest.raises(PracticeError):
        PracticeConfig.from_dict({"hidden_answer": 1})


def test_erased_cue_has_no_counterfactual_leak_in_pixels_controls_or_reward():
    left = environment(task="delayed_memory", lead_ms=0)
    first = left.reset(17)
    right = copy.deepcopy(left)
    # Counterfactual twins differ ONLY in the hidden cue answer. This is stronger
    # than comparing arbitrary seeds, whose different layouts hide subtle leaks.
    right._answer = 1 - left._answer
    assert not np.array_equal(first.pixels, right._observe().pixels)
    for _ in range(5):
        a, b = left.step([]), right.step([])
    np.testing.assert_array_equal(a.observation.pixels, b.observation.pixels)
    assert a.observation.control_state == b.observation.control_state
    for _ in range(21):
        a, b = left.step([]), right.step([])
        np.testing.assert_array_equal(a.observation.pixels, b.observation.pixels)
        assert a.reward == b.reward == 0
        assert a.observation.control_state == b.observation.control_state
    a, b = left.step([key("keyDown")]), right.step([key("keyDown")])
    assert a.outcome == b.outcome == "terminated"
    assert sorted([a.reward, b.reward]) == [-1, 1]


def test_execution_lead_does_not_shift_reward_intervals_or_execute_at_end_boundary():
    env = environment(lead_ms=100)
    initial = env.reset(4)
    commands = env.oracle_commands()
    a = env.step(commands, provenance="oracle")
    assert a.duration_ms == 100 and a.reward == 0 and not a.raw_events and not a.command_results
    assert a.observation.control_state["pointer"] == initial.control_state["pointer"]
    b = env.step([], provenance="oracle")
    assert b.reward == 1 and b.outcome == "terminated"
    assert all(event["eventNanos"] == a.observation.metadata["observedNanos"] for event in b.raw_events)
    assert b.observation.metadata["observedNanos"] - initial.metadata["observedNanos"] == 200_000_000


def test_final_motion_knot_at_period_belongs_to_next_transition():
    env = environment(lead_ms=0)
    env.reset(0)
    a = env.step([absolute(.1, .2), absolute(.9, .2, 100)])
    assert len(a.command_results) == 1
    assert a.observation.control_state["pointer"]["x"] == pytest.approx((.1 + .8 * .99) * 1280)
    b = env.step([])
    assert len(b.command_results) == 1
    assert b.command_results[0]["commandIndex"] == 1
    assert b.observation.control_state["pointer"]["x"] == pytest.approx(.9 * 1280)


def test_interpolation_precedes_click_at_intermediate_time():
    env = environment(lead_ms=0)
    env.reset(19)
    target = env.oracle_commands()[0]
    x, y = target["x"], target["y"]
    # At t=50 ms the trajectory crosses the target. Array order places click
    # before the later t=100 endpoint, so instant-knot execution would miss it.
    transition = env.step([absolute(x - .05, y), button(offset=50), absolute(x + .05, y, 100)])
    assert transition.outcome == "terminated" and transition.reward == 1
    assert transition.raw_events[-1]["kind"] == "buttonDown"
    assert transition.raw_events[-1]["x"] == pytest.approx(x * 1280)
    assert transition.command_results[-1]["status"] == "cancelled"


def test_relative_interpolation_preserves_signed_integer_total_and_zero_anchor():
    env = environment(lead_ms=0, relative_pointer=True)
    initial = env.reset(8)
    a = env.step([relative(0, 0), relative(17, -9, 13), relative(-6, 4, 29)])
    assert a.command_results[0]["status"] == "noOp"
    assert sum(event["dx"] for event in a.raw_events) == 11
    assert sum(event["dy"] for event in a.raw_events) == -5
    assert all(type(event["dx"]) is int and type(event["dy"]) is int for event in a.raw_events)
    assert a.observation.control_state["pointer"] == {
        "x": initial.control_state["pointer"]["x"] + 11,
        "y": initial.control_state["pointer"]["y"] - 5,
    }


@pytest.mark.parametrize("commands", [
    [absolute(.2, .2), button(offset=100)],
    [absolute(.2, .2), {"operation": "buttonDown", "offsetMs": 0, "button": 1}],
    [absolute(.2, .2), absolute(float("nan"), .3)],
    [absolute(.2, .2, 80), absolute(.3, .3, 20)],
    [absolute(.2, .2), {**button(), "keyCode": 1}],
    [absolute(.2, .2), relative(1, 1)],
    [absolute(.2, .2), {**button(), "offsetMs": True}],
    [absolute(.2, .2)] * 65,
])
def test_invalid_packet_is_atomic_even_with_an_existing_lead_queue(commands):
    env = environment(lead_ms=200)
    before = env.reset(8)
    env.step([absolute(.1, .1)])
    now, pending, sequence = env._now, copy.deepcopy(env._pending), env._packet_sequence
    with pytest.raises(PracticeError):
        env.step(commands)
    assert env._now == now and env._pending == pending and env._packet_sequence == sequence
    env.step([])
    result = env.step([])
    assert result.observation.control_state["pointer"] == {"x": 128.0, "y": 72.0}
    assert result.observation.episode_id == before.episode_id


def test_idempotent_holds_repeat_and_reset_cancel_old_episode_commands():
    env = environment(task="delayed_memory", lead_ms=100)
    old = env.reset(11)
    env.step([key("keyRepeat"), key("keyDown"), key("keyDown"), key("keyRepeat"), button(), button()])
    held = env.step([key("keyDown", 124)])
    assert [result["status"] for result in held.command_results] == ["noOp", "posted", "noOp", "posted", "posted", "noOp"]
    assert held.observation.control_state["keys"] == [123]
    assert held.observation.control_state["buttons"] == [0]
    new = env.reset(12)
    assert not new.control_state["keys"] and not new.control_state["buttons"]
    with pytest.raises(PracticeError, match="another episode"):
        env.step([], episode_id=old.episode_id)
    after = env.step([])
    assert not after.raw_events and not after.command_results
    assert after.observation.control_state["keys"] == []


def test_truncation_preserves_precleanup_observation_for_bootstrap():
    env = environment(task="delayed_memory", lead_ms=0, time_limit_ms=150)
    env.reset(9)
    env.step([key("keyDown")])
    ended = env.step([key("keyDown", 124, 70)])
    assert ended.outcome == "truncated" and ended.duration_ms == 50
    assert ended.reward == 0
    assert ended.observation.control_state["keys"] == [123]
    assert ended.cleanup_events[0]["kind"] == "keyUp"
    assert ended.cleanup_events[0]["origin"] == "boundary"
    assert ended.command_results[0]["status"] == "cancelled"
    assert not env._keys and not env._pending


def test_abort_is_not_a_fabricated_learning_transition():
    env = environment(task="delayed_memory", lead_ms=0)
    env.reset(7)
    env.step([key("keyDown")])
    aborted = env.abort("Operator stopped the trial")
    assert aborted.outcome == "aborted" and aborted.duration_ms == 0 and aborted.reward == 0
    assert aborted.observation.control_state["keys"] == [123]
    assert aborted.cleanup_events and not env._keys
    with pytest.raises(PracticeError):
        env.step([])


def test_potential_shaping_telescopes_using_duration_discount_and_terminal_zero():
    env = environment(lead_ms=0, shaping_scale=.7, discount_half_life_ms=15000)
    env.reset(31)
    initial_potential = env._potential()
    a = env.step([absolute(.05, .05)])
    potential_a = env._potential()
    gamma = 2 ** (-100 / 15000)
    assert a.reward == pytest.approx(.7 * (gamma * potential_a - initial_potential))
    b = env.step(env.oracle_commands(), provenance="oracle")
    assert b.outcome == "terminated"
    # Discounted shaped return differs from sparse return by only -scale*Phi0.
    assert a.reward + gamma * b.reward == pytest.approx(gamma - .7 * initial_potential)


def test_truncated_potential_is_not_incorrectly_forced_to_terminal_zero():
    env = environment(lead_ms=0, shaping_scale=1, time_limit_ms=50)
    env.reset(22)
    phi = env._potential()
    result = env.step([])
    assert result.outcome == "truncated"
    assert result.reward == pytest.approx((2 ** (-50 / 30000) - 1) * phi)


@pytest.mark.parametrize("changes", [
    {"task": "browser"}, {"seed": True}, {"pixel_width": 0}, {"lead_ms": -1},
    {"logical_bounds": (0, 0, float("inf"), 10)}, {"logical_bounds": (0, 0, -1, 10)},
    {"task": "delayed_memory", "shaping_scale": .1}, {"discount_half_life_ms": 0},
    {"relative_pointer": 1},
])
def test_invalid_configuration_rejected(changes):
    with pytest.raises(PracticeError):
        environment(**changes)


def test_default_vocabulary_avoids_equivalent_relative_exploration():
    env = environment()
    env.reset(0)
    assert env.action_vocabulary.key_codes == ()
    assert env.action_vocabulary.mouse_buttons == (0,)
    assert env.action_vocabulary.absolute_pointer
    assert not env.action_vocabulary.relative_pointer
    with pytest.raises(PracticeError, match="unavailable"):
        env.step([relative(1, 1)])
