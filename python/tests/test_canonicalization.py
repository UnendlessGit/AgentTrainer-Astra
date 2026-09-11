from dataclasses import replace
import mlx.core as mx
import numpy as np
import pytest

from astra.data.actions import canonicalize, quantized_event_nanos, encode_commands, decode_commands, ActionEncodingError, PacketCapacityError
from astra.model.actions import ActionVocabulary, Operation
from astra.model.config import ModelConfig
from astra.model.vision import VisualFeatures, position_features
from test_data_preparation import surface, event


def layout():
    bounds, _, valid = position_features(mx.array([[0, 0, 1, 1]]), 8, 16)
    return VisualFeatures(mx.zeros((1, 1)), mx.zeros((1, 1, 128, 1)), bounds[:, None], valid[:, None], mx.ones((1, 1), dtype=mx.bool_))


def test_taps_modifiers_repeats_and_equal_time_order_survive():
    config = ModelConfig.test_small()
    vocabulary = ActionVocabulary((13, 56), (0,), True, False, True)
    samples = [event(0, 1001, "keyDown", eventNanos=10_000_000, keyCode=13),
               event(1, 1002, "keyRepeat", eventNanos=11_000_000, keyCode=13),
               event(2, 1003, "keyUp", eventNanos=12_000_000, keyCode=13),
               event(3, 1004, "flags", eventNanos=12_000_000, keyCode=56, isDown=True, modifiers=1 << 17)]
    packet = canonicalize(samples, start_nanos=0, config=config, vocabulary=vocabulary, surfaces=[surface()], pointer_mode="absolute", initial_pointer=(-280, 110))
    assert [command["operation"] for command in packet.commands] == ["keyDown", "keyRepeat", "keyUp", "keyDown"]
    assert [command["offsetMs"] for command in packet.commands] == [10, 11, 12, 12]
    encoded = encode_commands(packet.commands, config=config, vocabulary=vocabulary, visual=layout(), surfaces=[surface()])
    assert decode_commands(encoded, config=config, vocabulary=vocabulary, visual=layout(), surfaces=[surface()]) == list(packet.commands)


def test_time_aware_motion_preserves_stop_reversal_and_click_location():
    config = ModelConfig.test_small()
    vocabulary = ActionVocabulary(mouse_buttons=(0,), absolute_pointer=True)
    samples = []
    for time in range(1, 100):
        x = -295 + min(time, 35) * 0.4 if time < 60 else -281 - (time - 60) * 0.2
        samples.append(event(time, time * 1_000_000 + 100, "pointer", eventNanos=time * 1_000_000, x=x, y=110))
    samples.append(event(100, 50_000_100, "buttonDown", eventNanos=50_000_000, x=-281, y=110, button=0))
    result = canonicalize(samples, start_nanos=0, config=config, vocabulary=vocabulary, surfaces=[surface()], pointer_mode="absolute", initial_pointer=(-295, 110))
    assert result.maximum_motion_error <= 0.75
    assert result.raw_event_count == 100
    assert len(result.commands) < 16
    click = next(index for index, item in enumerate(result.commands) if item["operation"] == "buttonDown")
    assert result.commands[click - 1]["operation"] == "pointerAbsolute"
    assert result.commands[click - 1]["offsetMs"] == 50
    assert result.commands[-1]["offsetMs"] == 100
    # The encoded pixel-cell target adds at most .25 logical point error.
    encoded = encode_commands(result.commands, config=config, vocabulary=vocabulary, visual=layout(), surfaces=[surface()])
    decoded = decode_commands(encoded, config=config, vocabulary=vocabulary, visual=layout(), surfaces=[surface()])
    for original, restored in zip(result.commands, decoded):
        if original["operation"] == "pointerAbsolute":
            assert abs(original["x"] - restored["x"]) * 60 <= 0.25
            assert abs(original["y"] - restored["y"]) * 30 <= 0.25


def test_relative_knots_preserve_cumulative_counts_and_never_mix_absolute():
    config = ModelConfig.test_small()
    vocabulary = ActionVocabulary(relative_pointer=True)
    samples = [event(time, time * 1_000_000 + 1, "pointer", eventNanos=time * 1_000_000,
                     x=-280, y=110, dx=1 if time < 50 else -1, dy=0) for time in range(1, 100)]
    result = canonicalize(samples, start_nanos=0, config=config, vocabulary=vocabulary, surfaces=[surface()], pointer_mode="relative", initial_pointer=None)
    assert {item["operation"] for item in result.commands} == {"pointerRelative"}
    assert sum(item["dx"] for item in result.commands) == -1
    assert result.commands[0]["offsetMs"] == 0
    encoded = encode_commands(result.commands, config=config, vocabulary=vocabulary, visual=layout(), surfaces=[surface()])
    decoded = decode_commands(encoded, config=config, vocabulary=vocabulary, visual=layout(), surfaces=[surface()])
    assert decoded == list(result.commands)


def test_capacity_fit_refuses_dropping_raw_actions():
    config = ModelConfig.test_small()
    samples = [event(index, index * 1_000_000 + 1, "keyDown" if index % 2 else "keyUp", eventNanos=index * 1_000_000, keyCode=13)
               for index in range(20)]
    with pytest.raises(PacketCapacityError) as error:
        canonicalize(samples, start_nanos=0, config=config, vocabulary=ActionVocabulary((13,)), surfaces=[surface()], pointer_mode="disabled", initial_pointer=None)
    assert error.value.required == 20
    fit = canonicalize(samples, start_nanos=0, config=replace(config, packet_capacity=32), vocabulary=ActionVocabulary((13,)), surfaces=[surface()], pointer_mode="disabled", initial_pointer=None)
    assert len(fit.commands) == 20


def test_reconciliation_never_becomes_an_expert_press_and_boundaries_reject():
    config = ModelConfig.test_small()
    sample = event(0, 10, "keyDown", keyCode=13, origin="reconciliation")
    result = canonicalize([sample], start_nanos=0, config=config, vocabulary=ActionVocabulary((13,)), surfaces=[surface()], pointer_mode="disabled", initial_pointer=None)
    assert result.commands == ()
    with pytest.raises(ActionEncodingError, match="discontinuity"):
        canonicalize([event(1, 20, "gap", origin="boundary")], start_nanos=0, config=config, vocabulary=ActionVocabulary(), surfaces=[surface()], pointer_mode="disabled", initial_pointer=None)


def test_out_of_range_motion_and_ambiguous_command_arguments_are_rejected():
    config = ModelConfig.test_small()
    vocabulary = ActionVocabulary((13,), relative_pointer=True)
    with pytest.raises(ActionEncodingError, match="bounded"):
        encode_commands([{"operation": "pointerRelative", "offsetMs": 0, "dx": 5000, "dy": 0}], config=config, vocabulary=vocabulary, visual=layout(), surfaces=[surface()])
    with pytest.raises(ActionEncodingError, match="bounded"):
        encode_commands([{"operation": "scroll", "offsetMs": 0, "dx": 1e308, "dy": 0}], config=config,
                        vocabulary=ActionVocabulary(scroll=True), visual=layout(), surfaces=[surface()])
    with pytest.raises(ActionEncodingError, match="irrelevant"):
        encode_commands([{"operation": "keyDown", "offsetMs": 0, "keyCode": 13, "dx": 0}], config=config, vocabulary=vocabulary, visual=layout(), surfaces=[surface()])


def test_timestamp_rounding_carries_boundary_events_into_the_next_packet():
    config = ModelConfig.test_small()
    vocabulary = ActionVocabulary((13,), absolute_pointer=True)
    source = event(0, 99_900_100, "keyDown", eventNanos=99_900_000, keyCode=13)
    original = dict(source)
    first = canonicalize([source], start_nanos=0, config=config, vocabulary=vocabulary,
                         surfaces=[surface()], pointer_mode="absolute", initial_pointer=(-295, 110))
    second = canonicalize([source], start_nanos=100_000_000, config=config, vocabulary=vocabulary,
                          surfaces=[surface()], pointer_mode="absolute", initial_pointer=(-295, 110))
    assert first.commands == () and first.raw_event_count == 0 and first.deferred_event_count == 1
    assert second.commands == ({"offsetMs": 0, "operation": "keyDown", "keyCode": 13},)
    assert second.raw_event_count == 1 and second.maximum_time_error_ms == .1
    assert source == original  # Availability and source clocks remain unchanged.


def test_quantized_partition_preserves_ten_thousand_boundary_transitions_exactly_once():
    config = ModelConfig.test_small()
    vocabulary = ActionVocabulary((13, 56), (0,), scroll=True)
    # Nanosecond-scale jitter and a large odd clock origin exercise exact integer
    # rounding, half-ms ties, both sides of packet boundaries and original order.
    origin = 2**62 + 12345
    jitter = (-510_000, -500_000, -490_000, -1, 0, 1, 490_000, 499_999, 500_000, 510_000)
    kinds = ("keyDown", "keyUp", "keyRepeat", "flags", "scroll")
    yielded = 0
    def source_events():
        nonlocal yielded
        for boundary in range(1, 1001):
            for index, delta in enumerate(jitter):
                sequence = (boundary - 1) * len(jitter) + index
                source = origin + boundary * 100_000_000 + delta
                kind = kinds[index % len(kinds)]
                fields = ({"keyCode": 13} if kind.startswith("key") else
                          {"keyCode": 56, "isDown": bool(sequence % 2), "modifiers": 0} if kind == "flags" else
                          {"scrollX": sequence % 7, "scrollY": -(sequence % 5)})
                yielded += 1
                yield event(sequence, source + 700_000, kind, eventNanos=source, **fields)
    stream = iter(source_events())
    future = next(stream, None)
    represented = 0
    maximum_retained = 0
    for packet_index in range(1002):
        start = origin + packet_index * 100_000_000
        pending = []
        while future is not None and quantized_event_nanos(future["eventNanos"], grid_origin_nanos=origin) < start + 100_000_000:
            pending.append(future)
            future = next(stream, None)
        maximum_retained = max(maximum_retained, len(pending))
        packet = canonicalize(pending, start_nanos=start, config=config, vocabulary=vocabulary,
                              surfaces=[surface()], pointer_mode="disabled", initial_pointer=None)
        assert packet.raw_event_count == len(pending) == len(packet.commands)
        assert packet.before_event_count == packet.deferred_event_count == 0
        for command, source in zip(packet.commands, pending, strict=True):
            absolute_time = start + command["offsetMs"] * 1_000_000
            assert absolute_time == quantized_event_nanos(source["eventNanos"], grid_origin_nanos=origin)
            assert abs(absolute_time - source["eventNanos"]) <= 500_000
            expected = "keyDown" if source["kind"] == "flags" and source["isDown"] else "keyUp" if source["kind"] == "flags" else source["kind"]
            assert command["operation"] == expected
        represented += packet.raw_event_count
    assert future is None and yielded == represented == 10000
    assert maximum_retained <= 10


def test_boundary_motion_start_anchor_and_equal_time_click_order_remain_explicit():
    config = ModelConfig.test_small()
    vocabulary = ActionVocabulary(mouse_buttons=(0,), absolute_pointer=True)
    samples = [event(0, 99_800_100, "pointer", eventNanos=99_800_000, x=-294.5, y=110),
               event(1, 99_900_100, "buttonDown", eventNanos=99_900_000, x=-294.5, y=110, button=0),
               event(2, 100_100_100, "pointer", eventNanos=100_100_000, x=-294.25, y=110),
               event(3, 100_200_100, "buttonUp", eventNanos=100_200_000, x=-294.25, y=110, button=0)]
    packet = canonicalize(samples, start_nanos=100_000_000, config=config, vocabulary=vocabulary,
                          surfaces=[surface()], pointer_mode="absolute", initial_pointer=(-295, 110))
    assert packet.raw_event_count == 4 and packet.maximum_motion_error <= .75
    assert packet.commands[0]["offsetMs"] == 0
    np.testing.assert_allclose(packet.commands[0]["x"] * 60 - 300, -295)
    down = next(index for index, command in enumerate(packet.commands) if command["operation"] == "buttonDown")
    up = next(index for index, command in enumerate(packet.commands) if command["operation"] == "buttonUp")
    assert down < up
    for index, expected in ((down, -294.5), (up, -294.25)):
        assert packet.commands[index]["offsetMs"] == packet.commands[index - 1]["offsetMs"] == 0
        assert packet.commands[index - 1]["operation"] == "pointerAbsolute"
        np.testing.assert_allclose(packet.commands[index - 1]["x"] * 60 - 300, expected)
    assert packet.commands[-1]["offsetMs"] == config.period_ms


def test_motion_accuracy_is_checked_at_source_times_after_rounding():
    config = ModelConfig.test_small()
    samples = [event(0, 49_490_100, "pointer", eventNanos=49_490_000, x=-295, y=110),
               event(1, 50_490_100, "pointer", eventNanos=50_490_000, x=-280, y=110)]
    with pytest.raises(ActionEncodingError, match="reconstruction bound"):
        canonicalize(samples, start_nanos=0, config=config, vocabulary=ActionVocabulary(absolute_pointer=True),
                     surfaces=[surface()], pointer_mode="absolute", initial_pointer=(-295, 110))


def test_key_and_scroll_anchors_preserve_their_pointer_position():
    config = ModelConfig.test_small()
    vocabulary = ActionVocabulary((13,), absolute_pointer=True, scroll=True)
    samples = [event(0, 10_000_100, "pointer", eventNanos=10_000_000, x=-290, y=110),
               event(1, 20_000_100, "keyDown", eventNanos=20_000_000, keyCode=13),
               event(2, 30_000_100, "scroll", eventNanos=30_000_000, x=-280, y=110, scrollX=0, scrollY=1),
               event(3, 40_000_100, "pointer", eventNanos=40_000_000, x=-275, y=110)]
    result = canonicalize(samples, start_nanos=0, config=config, vocabulary=vocabulary, surfaces=[surface()],
                          pointer_mode="absolute", initial_pointer=(-295, 110))
    for operation, time, x in (("keyDown", 20, -290), ("scroll", 30, -280)):
        index = next(index for index, command in enumerate(result.commands) if command["operation"] == operation)
        anchor = result.commands[index - 1]
        assert anchor["operation"] == "pointerAbsolute" and anchor["offsetMs"] == time
        np.testing.assert_allclose(anchor["x"] * 60 - 300, x)


def test_relative_reconstruction_matches_every_raw_cumulative_sample():
    config = replace(ModelConfig.test_small(), packet_capacity=64)
    vocabulary = ActionVocabulary(relative_pointer=True)
    previous = np.zeros(2)
    samples, targets = [], []
    for time in range(1, 100):
        cumulative = np.rint([15 * np.sin(time / 20), 4 * np.cos(time / 10) - 4])
        delta = cumulative - previous; previous = cumulative
        targets.append(cumulative.copy())
        samples.append(event(time, time * 1_000_000 + 100, "pointer", eventNanos=time * 1_000_000,
                             x=-280, y=110, dx=int(delta[0]), dy=int(delta[1])))
    result = canonicalize(samples, start_nanos=0, config=config, vocabulary=vocabulary, surfaces=[surface()], pointer_mode="relative", initial_pointer=None)
    encoded = encode_commands(result.commands, config=config, vocabulary=vocabulary, visual=layout(), surfaces=[surface()])
    decoded = decode_commands(encoded, config=config, vocabulary=vocabulary, visual=layout(), surfaces=[surface()])
    times = [command["offsetMs"] for command in decoded]
    positions = np.cumsum([[command["dx"], command["dy"]] for command in decoded], axis=0)
    restored = np.stack([np.interp(np.arange(1, 100), times, positions[:, axis]) for axis in range(2)], axis=-1)
    assert np.max(np.linalg.norm(restored - targets, axis=-1)) <= 1


@pytest.mark.parametrize("point", [(1., .5), (.5, 1.)])
def test_absolute_commands_cannot_target_the_exclusive_surface_edge(point):
    with pytest.raises(ActionEncodingError, match="outside"):
        encode_commands([{"operation": "pointerAbsolute", "offsetMs": 0, "surfaceID": "primary", "x": point[0], "y": point[1]}],
                        config=ModelConfig.test_small(), vocabulary=ActionVocabulary(absolute_pointer=True),
                        visual=layout(), surfaces=[surface()])
