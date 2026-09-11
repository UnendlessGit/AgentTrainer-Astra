"""Versioned raw-event canonicalization and packet/neural/wire conversion.

Raw demonstrations stay immutable. Capacity and reconstruction failures reject
derived intervals with actionable reports; input is never silently truncated.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
import mlx.core as mx
import numpy as np

from astra.model.actions import ActionVocabulary, Operation, PacketBatch
from astra.model.config import ModelConfig
from astra.model.vision import VisualFeatures
from astra.recordings import RecordingError, validate_event, validate_surface
from astra.versions import CANONICALIZER_VERSION

NANOS_PER_MILLISECOND = 1_000_000
DISCRETE_EVENT_KINDS = frozenset(("keyDown", "keyUp", "keyRepeat", "flags", "buttonDown", "buttonUp", "scroll"))
OPERATIONS = {"keyDown": Operation.KEY_DOWN, "keyUp": Operation.KEY_UP, "keyRepeat": Operation.KEY_REPEAT,
              "buttonDown": Operation.BUTTON_DOWN, "buttonUp": Operation.BUTTON_UP,
              "pointerAbsolute": Operation.ABSOLUTE, "pointerRelative": Operation.RELATIVE, "scroll": Operation.SCROLL}
WIRE_OPERATIONS = {value: key for key, value in OPERATIONS.items()}


class ActionEncodingError(ValueError):
    pass


class PacketCapacityError(ActionEncodingError):
    def __init__(self, required: int, capacity: int):
        self.required, self.capacity = required, capacity
        super().__init__(f"This control interval needs {required} commands; the model supports {capacity}. Select a larger packet capacity or faster decision cadence.")


@dataclass(frozen=True)
class CanonicalPacket:
    commands: tuple[dict, ...]
    raw_event_count: int
    maximum_motion_error: float
    maximum_time_error_ms: float
    # Physical events supplied outside this quantized packet are reported so a
    # single-packet caller cannot mistake deferred events for represented ones.
    before_event_count: int = 0
    deferred_event_count: int = 0


def quantized_event_nanos(event_nanos: int, *, grid_origin_nanos: int) -> int:
    """Nearest integer millisecond on one fixed execution grid; ties go later.

    Integer subtraction/rounding preserves sub-ms precision even for large
    monotonic clocks. Adjacent packet origins differ by whole milliseconds and
    therefore produce the same absolute quantized timestamp.
    """
    if any(type(value) is not int or not 0 <= value <= 2**64 - 1 for value in (event_nanos, grid_origin_nanos)):
        raise ActionEncodingError("Invalid source timestamp or quantization grid")
    ticks = (event_nanos - grid_origin_nanos + NANOS_PER_MILLISECOND // 2) // NANOS_PER_MILLISECOND
    return grid_origin_nanos + ticks * NANOS_PER_MILLISECOND


def _simplify(points: list[tuple[int, float, float]], tolerance: float, forced: set[int]) -> tuple[list[int], float]:
    if len(points) < 3:
        return list(range(len(points))), 0.0
    # Time is the independent variable; geometric perpendicular distance would
    # incorrectly remove a stop, acceleration or reversal along a straight line.
    data = np.asarray(points, dtype=np.float64)
    selected = {0, len(points) - 1, *forced}
    boundaries = sorted(selected)
    pending = list(zip(boundaries, boundaries[1:]))
    while pending:
        first, last = pending.pop()
        if last - first <= 1:
            continue
        start, end = data[first], data[last]
        duration = end[0] - start[0]
        if duration == 0:
            # Equal-time commands preserve original order; no interpolation can
            # reconstruct an intermediate position from a zero-width interval.
            selected.update(range(first + 1, last))
            continue
        ratio = (data[first + 1:last, 0] - start[0]) / duration
        expected = start[None, 1:] + ratio[:, None] * (end[1:] - start[1:])
        error = np.linalg.norm(data[first + 1:last, 1:] - expected, axis=-1)
        position = int(np.argmax(error))
        if error[position] > tolerance:
            index = first + 1 + position
            selected.add(index); pending.extend(((first, index), (index, last)))
    indices = sorted(selected)
    maximum = 0.0
    for first, last in zip(indices, indices[1:]):
        if last - first < 2 or data[last, 0] == data[first, 0]:
            continue
        ratio = (data[first + 1:last, 0] - data[first, 0]) / (data[last, 0] - data[first, 0])
        expected = data[first, None, 1:] + ratio[:, None] * (data[last, 1:] - data[first, 1:])
        maximum = max(maximum, float(np.linalg.norm(data[first + 1:last, 1:] - expected, axis=-1).max()))
    return indices, maximum


def canonicalize(events: Sequence[dict], *, start_nanos: int, config: ModelConfig,
                 vocabulary: ActionVocabulary, surfaces: Sequence[dict],
                 pointer_mode: str, initial_pointer: tuple[float, float] | None) -> CanonicalPacket:
    config.validate(); vocabulary.validate()
    surfaces = [validate_surface(surface) for surface in surfaces]
    if pointer_mode not in ("absolute", "relative", "disabled") or type(start_nanos) is not int or start_nanos < 0:
        raise ActionEncodingError("Invalid canonicalization mode or execution start")
    if len(events) > 100_000:
        raise ActionEncodingError("Input interval exceeds the supported source event budget")
    end_nanos = start_nanos + config.period_ms * 1_000_000
    eligible = []
    before_count = deferred_count = 0
    for event in events:
        validate_event(event)
        quantized = quantized_event_nanos(event["eventNanos"], grid_origin_nanos=start_nanos)
        in_packet = start_nanos <= quantized < end_nanos
        if (in_packet or start_nanos <= event["eventNanos"] < end_nanos) and (event["origin"] == "boundary" or event["kind"] == "gap"):
            raise ActionEncodingError("This action interval contains an input discontinuity")
        if event["origin"] == "physical":
            if in_packet:
                eligible.append(event)
            elif quantized < start_nanos:
                before_count += 1
            else:
                deferred_count += 1
    eligible.sort(key=lambda event: (event["eventNanos"], event["sequence"]))
    if len({event["sequence"] for event in eligible}) != len(eligible):
        raise ActionEncodingError("Duplicate raw input sequence in derived interval")
    commands: list[tuple[int, int, dict]] = []
    points: list[tuple[int, float, float]] = []
    source_points: list[tuple[float, float, float, int]] = []
    point_order = []
    forced = set()
    maximum_time_error = 0.0
    accumulated = np.zeros(2, dtype=np.float64)
    if pointer_mode == "absolute" and initial_pointer is not None:
        if len(initial_pointer) != 2 or not all(type(value) in (int, float) and math.isfinite(value) for value in initial_pointer):
            raise ActionEncodingError("Invalid initial pointer")
        points.append((0, *initial_pointer)); point_order.append(-2)
    elif pointer_mode == "relative":
        points.append((0, 0, 0)); point_order.append(-2)
    motion_evidence = False

    def add_point(time, order, x, y, *, preserve=False, source_time=None):
        source_time = time if source_time is None else source_time
        if points and points[-1] == (time, x, y):
            if preserve:
                forced.add(len(points) - 1)
            source_points.append((source_time, x, y, len(points) - 1))
            return
        points.append((time, x, y)); point_order.append(order)
        source_points.append((source_time, x, y, len(points) - 1))
        if preserve:
            forced.add(len(points) - 1)

    for index, event in enumerate(eligible):
        source_ms = (event["eventNanos"] - start_nanos) / 1_000_000
        offset = (quantized_event_nanos(event["eventNanos"], grid_origin_nanos=start_nanos) - start_nanos) // NANOS_PER_MILLISECOND
        kind = event["kind"]
        maximum_time_error = max(maximum_time_error, abs(offset - source_ms))
        order = 2 * index
        if kind == "pointer":
            if pointer_mode == "absolute":
                if not vocabulary.absolute_pointer:
                    raise ActionEncodingError("Demonstration pointer motion exceeds the selected capabilities")
                if initial_pointer is None and not points:
                    raise ActionEncodingError("Pointer motion has no causal boundary position")
                add_point(offset, order, event["x"], event["y"], source_time=source_ms)
                motion_evidence = True
            elif pointer_mode == "relative":
                if not vocabulary.relative_pointer or event.get("dx") is None or event.get("dy") is None:
                    raise ActionEncodingError("Relative control needs captured raw movement and capability")
                position = (float(accumulated[0]) + event["dx"], float(accumulated[1]) + event["dy"])
                if not all(math.isfinite(value) for value in position):
                    raise ActionEncodingError("Cumulative relative movement exceeds its finite numerical range")
                accumulated[:] = position
                add_point(offset, order, *accumulated, source_time=source_ms)
                motion_evidence = True
            continue
        command = {"offsetMs": offset}
        if kind in ("keyDown", "keyUp", "keyRepeat"):
            if event["keyCode"] not in vocabulary.key_codes:
                raise ActionEncodingError(f"Recorded key {event['keyCode']} is outside the action vocabulary")
            command.update(operation=kind, keyCode=event["keyCode"])
        elif kind == "flags":
            if event.get("keyCode") is None or event.get("isDown") is None:
                raise ActionEncodingError("Modifier transition has no side-specific captured state")
            if event["keyCode"] not in vocabulary.key_codes:
                raise ActionEncodingError("Modifier key is outside the action vocabulary")
            command.update(operation="keyDown" if event["isDown"] else "keyUp", keyCode=event["keyCode"])
        elif kind in ("buttonDown", "buttonUp"):
            if event["button"] not in vocabulary.mouse_buttons:
                raise ActionEncodingError("Recorded button is outside the action vocabulary")
            if pointer_mode == "absolute":
                if event.get("x") is None or event.get("y") is None:
                    raise ActionEncodingError("Absolute button commands need their recorded pointer position")
                add_point(offset, order - 1, event["x"], event["y"], preserve=True, source_time=source_ms)
                motion_evidence = True
            elif points:
                add_point(offset, order - 1, *accumulated, preserve=True, source_time=source_ms)
            command.update(operation=kind, button=event["button"])
        elif kind == "scroll":
            if not vocabulary.scroll:
                raise ActionEncodingError("Recorded scrolling exceeds the selected capabilities")
            command.update(operation="scroll", dx=event["scrollX"], dy=event["scrollY"])
        else:
            continue
        if kind not in ("buttonDown", "buttonUp") and points:
            if pointer_mode == "absolute":
                position = (event["x"], event["y"]) if event.get("x") is not None and event.get("y") is not None else points[-1][1:]
                motion_evidence |= position != tuple(points[-1][1:])
            else:
                position = accumulated
            add_point(offset, order - 1, *position, preserve=True, source_time=source_ms)
        commands.append((offset, order, command))
    maximum_motion_error = 0.0
    if motion_evidence and points:
        # A terminal held-position knot prevents interpolation from implicitly
        # moving for the whole interval after the last actual motion sample.
        add_point(config.period_ms, 2 * len(eligible) + 1, *points[-1][1:])
        # A spatial bound at rounded timestamps alone misses fast movement
        # displaced by timestamp quantization. Qualify at original sample times
        # too, adding knots where retaining more detail can repair the error.
        while True:
            selected, maximum_motion_error = _simplify(points, 0.75, forced)
            curve = np.asarray([points[index] for index in selected], dtype=np.float64)
            worst = None
            for time, x, y, index in source_points:
                if index in selected and points[index][0] == time:
                    continue  # Equal-time ordered knots retain this exact state.
                expected = np.array([np.interp(time, curve[:, 0], curve[:, axis]) for axis in (1, 2)])
                error = float(np.linalg.norm(expected - (x, y)))
                if error > maximum_motion_error:
                    maximum_motion_error, worst = error, index
            if maximum_motion_error <= 0.75:
                break
            if worst is None or worst in selected:
                raise ActionEncodingError("Motion exceeds the 1-point/count reconstruction bound after millisecond timestamp quantization")
            forced.add(worst)
        previous = np.zeros(2)
        for index in selected:
            offset, x, y = points[index]
            if pointer_mode == "absolute":
                candidates = [surface for surface in surfaces if surface["globalBounds"]["x"] <= x < surface["globalBounds"]["x"] + surface["globalBounds"]["width"]
                              and surface["globalBounds"]["y"] <= y < surface["globalBounds"]["y"] + surface["globalBounds"]["height"]]
                if len(candidates) != 1:
                    raise ActionEncodingError("Pointer trajectory does not have one unambiguous observed surface")
                surface = candidates[0]; bounds = surface["globalBounds"]
                command = {"offsetMs": offset, "operation": "pointerAbsolute", "surfaceID": surface["id"],
                           "x": (x - bounds["x"]) / bounds["width"], "y": (y - bounds["y"]) / bounds["height"]}
            else:
                current = np.rint([x, y])
                delta = current - previous; previous = current
                if np.linalg.norm(current - [x, y]) > 0.25:
                    raise ActionEncodingError("Relative input contains sub-count movement that cannot be represented without excess error")
                command = {"offsetMs": offset, "operation": "pointerRelative", "dx": int(delta[0]), "dy": int(delta[1])}
            commands.append((offset, point_order[index], command))
    commands.sort(key=lambda item: (item[0], item[1]))
    if len(commands) > config.packet_capacity:
        raise PacketCapacityError(len(commands), config.packet_capacity)
    return CanonicalPacket(tuple(item[2] for item in commands), len(eligible), maximum_motion_error, maximum_time_error, before_count, deferred_count)


def encode_commands(commands: Sequence[dict], *, config: ModelConfig, vocabulary: ActionVocabulary,
                    visual: VisualFeatures, surfaces: Sequence[dict], batch_index: int = 0) -> PacketBatch:
    """Encode exactly one packet against observation-derived dense cell bounds."""
    if len(commands) > config.packet_capacity:
        raise PacketCapacityError(len(commands), config.packet_capacity)
    if visual.cells.ndim != 4 or not 0 <= batch_index < visual.cells.shape[0] or len(surfaces) != visual.cells.shape[1]:
        raise ActionEncodingError("Packet encoding needs flattened, matching observed surfaces")
    arrays = {name: np.zeros((1, config.packet_capacity + 1), dtype=np.int32) for name in PacketBatch.__dataclass_fields__}
    cell_bounds = np.asarray(visual.cell_bounds[batch_index])
    cell_valid = np.asarray(visual.cell_valid[batch_index])
    previous = 0
    for slot, command in enumerate(commands):
        if not isinstance(command, dict) or command.get("operation") not in OPERATIONS:
            raise ActionEncodingError("Unknown semantic command")
        operation = OPERATIONS[command["operation"]]
        offset = command.get("offsetMs")
        if type(offset) is not int or not previous <= offset <= config.period_ms:
            raise ActionEncodingError("Semantic command offsets must be ordered integer milliseconds")
        previous = offset
        arrays["operation"][0, slot] = operation
        arrays["offset"][0, slot] = offset
        allowed = {"operation", "offsetMs"}
        if operation in (Operation.KEY_DOWN, Operation.KEY_UP, Operation.KEY_REPEAT):
            allowed.add("keyCode")
            value = command.get("keyCode")
            if type(value) is not int or value not in vocabulary.key_codes:
                raise ActionEncodingError("Key exceeds the policy vocabulary")
            arrays["key"][0, slot] = value
        elif operation in (Operation.BUTTON_DOWN, Operation.BUTTON_UP):
            allowed.add("button")
            value = command.get("button")
            if type(value) is not int or value not in vocabulary.mouse_buttons:
                raise ActionEncodingError("Button exceeds the policy vocabulary")
            arrays["button"][0, slot] = value
        elif operation == Operation.ABSOLUTE:
            allowed.update(("surfaceID", "x", "y"))
            matches = [index for index, surface in enumerate(surfaces) if surface["id"] == command.get("surfaceID")]
            if len(matches) != 1:
                raise ActionEncodingError("Pointing surface is unavailable or ambiguous")
            surface = matches[0]
            point = np.array([command.get("x"), command.get("y")], dtype=np.float64)
            if not np.isfinite(point).all() or np.any(point < 0) or np.any(point >= 1):
                raise ActionEncodingError("Pointing coordinates are outside the observed surface")
            bounds = cell_bounds[surface]
            candidates = cell_valid[surface] & np.all(point >= bounds[:, :2], axis=-1) & np.all(point <= bounds[:, 2:], axis=-1)
            # Boundary points choose one deterministic valid cell. Per-cell
            # coordinate bins reconstruct its center without executable clipping.
            cells = np.flatnonzero(candidates)
            if not len(cells):
                raise ActionEncodingError("Pointing coordinates fall outside the dense valid mask")
            cell = int(cells[0]); box = bounds[cell]
            within = (point - box[:2]) / (box[2:] - box[:2])
            bins = np.minimum((within * config.coordinate_bins).astype(int), config.coordinate_bins - 1)
            reconstructed = box[:2] + (bins + 0.5) / config.coordinate_bins * (box[2:] - box[:2])
            geometry = surfaces[surface]["globalBounds"]
            if np.linalg.norm((reconstructed - point) * [geometry["width"], geometry["height"]]) > 0.25:
                raise ActionEncodingError("Pointing quantization exceeds 0.25 logical points; increase visual or coordinate resolution")
            arrays["surface"][0, slot] = surface; arrays["cell"][0, slot] = cell
            arrays["within_x"][0, slot], arrays["within_y"][0, slot] = bins
        elif operation in (Operation.RELATIVE, Operation.SCROLL):
            allowed.update(("dx", "dy"))
            scale = vocabulary.scroll_units_per_point if operation == Operation.SCROLL else 1
            for field in ("dx", "dy"):
                value = command.get(field)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ActionEncodingError("Motion argument is not finite")
                if abs(value) > (config.delta_magnitude + 0.5) / scale:
                    raise ActionEncodingError("Motion cannot fit the policy's bounded integer vocabulary")
                quantized = int(round(value * scale))
                if abs(quantized) > config.delta_magnitude or abs(quantized / scale - value) > 0.25:
                    raise ActionEncodingError("Motion cannot fit the policy's bounded integer vocabulary")
                arrays[field][0, slot] = quantized
        if set(command) != allowed:
            raise ActionEncodingError("Semantic command has missing or irrelevant arguments")
    packet = PacketBatch(**{name: mx.array(value) for name, value in arrays.items()})
    selected = VisualFeatures(**{name: getattr(visual, name)[batch_index:batch_index + 1] for name in VisualFeatures.__dataclass_fields__})
    packet.validate(config, vocabulary, selected)
    return packet


def decode_commands(packet: PacketBatch, *, config: ModelConfig, vocabulary: ActionVocabulary,
                    visual: VisualFeatures, surfaces: Sequence[dict], batch_index: int = 0) -> list[dict]:
    packet.validate(config, vocabulary, visual)
    if len(surfaces) != visual.cells.shape[1]:
        raise ActionEncodingError("Decoded action surfaces do not match its observation")
    arrays = {name: np.asarray(getattr(packet, name)[batch_index]) for name in PacketBatch.__dataclass_fields__}
    bounds = np.asarray(visual.cell_bounds[batch_index])
    commands = []
    for slot, operation in enumerate(arrays["operation"]):
        if operation == Operation.END:
            break
        command = {"operation": WIRE_OPERATIONS[operation], "offsetMs": int(arrays["offset"][slot])}
        if operation in (Operation.KEY_DOWN, Operation.KEY_UP, Operation.KEY_REPEAT):
            command["keyCode"] = int(arrays["key"][slot])
        elif operation in (Operation.BUTTON_DOWN, Operation.BUTTON_UP):
            command["button"] = int(arrays["button"][slot])
        elif operation == Operation.ABSOLUTE:
            surface = int(arrays["surface"][slot]); cell = int(arrays["cell"][slot]); box = bounds[surface, cell]
            point = box[:2] + (np.array([arrays["within_x"][slot], arrays["within_y"][slot]]) + 0.5) / config.coordinate_bins * (box[2:] - box[:2])
            command.update(surfaceID=surfaces[surface]["id"], x=float(point[0]), y=float(point[1]))
        else:
            scale = vocabulary.scroll_units_per_point if operation == Operation.SCROLL else 1
            command.update(dx=float(arrays["dx"][slot]) / scale, dy=float(arrays["dy"][slot]) / scale)
        commands.append(command)
    return commands
