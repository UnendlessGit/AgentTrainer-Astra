"""Deterministic, permission-free visual tasks with real delayed action semantics.

The adapter does not capture a screen or post operating-system input. Its pixels,
geometry, controls and executed-event history use the same contracts as native
observations. Privileged fixture answers are available only to oracle_commands.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import heapq
import hashlib
import json
import math
from typing import Literal
import uuid

import numpy as np
from PIL import Image, ImageDraw

from astra.model.actions import ActionVocabulary
from astra.recordings import validate_event, validate_frame, validate_surface


class PracticeError(ValueError):
    """An action/configuration is rejected without partially applying it."""


def _integer(value, minimum, maximum, name):
    if type(value) is not int or not minimum <= value <= maximum:
        raise PracticeError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True)
class PracticeConfig:
    task: Literal["pointing", "delayed_memory"] = "pointing"
    seed: int = 0
    pixel_width: int = 1280
    pixel_height: int = 720
    logical_bounds: tuple[float, float, float, float] = (0, 0, 1280, 720)
    period_ms: int = 100
    lead_ms: int = 100
    delay_ms: int = 2000
    cue_ms: int = 500
    time_limit_ms: int = 120000
    shaping_scale: float = 0.0
    relative_pointer: bool = False
    discount_half_life_ms: float = 30000.0
    schema_version: int = 1

    def validate(self) -> PracticeConfig:
        if self.task not in ("pointing", "delayed_memory"):
            raise PracticeError("Unknown practice task")
        _integer(self.schema_version, 1, 1, "schema_version")
        _integer(self.seed, 0, 2**63 - 1, "seed")
        _integer(self.pixel_width, 32, 32768, "pixel_width")
        _integer(self.pixel_height, 32, 32768, "pixel_height")
        _integer(self.period_ms, 1, 1000, "period_ms")
        _integer(self.lead_ms, 0, 2000, "lead_ms")
        _integer(self.delay_ms, 1, 120000, "delay_ms")
        _integer(self.cue_ms, 1, 120000, "cue_ms")
        _integer(self.time_limit_ms, 1, 600000, "time_limit_ms")
        if type(self.relative_pointer) is not bool:
            raise PracticeError("relative_pointer must be Boolean")
        if (type(self.discount_half_life_ms) not in (int, float)
                or not math.isfinite(self.discount_half_life_ms) or self.discount_half_life_ms <= 0):
            raise PracticeError("discount_half_life_ms must be positive and finite")
        if (type(self.shaping_scale) not in (int, float) or not math.isfinite(self.shaping_scale)
                or not 0 <= self.shaping_scale <= 1000):
            raise PracticeError("shaping_scale must be finite and nonnegative")
        if self.task == "delayed_memory" and self.shaping_scale:
            raise PracticeError("Memory rewards cannot reveal the hidden answer through shaping")
        if not isinstance(self.logical_bounds, tuple) or len(self.logical_bounds) != 4:
            raise PracticeError("logical_bounds must be (x, y, width, height)")
        try:
            validate_surface(self.surface())
        except (TypeError, ValueError) as error:
            raise PracticeError("Invalid practice surface geometry") from error
        return self

    def surface(self) -> dict:
        x, y, width, height = self.logical_bounds
        return {
            "id": "practice", "globalBounds": {"x": x, "y": y, "width": width, "height": height},
            "pixelWidth": self.pixel_width, "pixelHeight": self.pixel_height,
            "contentBounds": {"x": 0, "y": 0, "width": self.pixel_width, "height": self.pixel_height},
            "geometryRevision": 0,
        }

    def to_dict(self) -> dict:
        self.validate()
        value = asdict(self)
        value["logical_bounds"] = list(value["logical_bounds"])
        return value

    @classmethod
    def from_dict(cls, value: dict) -> PracticeConfig:
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise PracticeError("Unknown practice configuration fields")
        copied = dict(value)
        if "logical_bounds" in copied:
            if not isinstance(copied["logical_bounds"], (list, tuple)):
                raise PracticeError("logical_bounds must be a coordinate array")
            copied["logical_bounds"] = tuple(copied["logical_bounds"])
        return cls(**copied).validate()

    @property
    def signature(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class PracticeObservation:
    pixels: np.ndarray  # H,W,4 uint8 BGRA; owned immutable snapshot
    metadata: dict  # exact native FrameMetadata, no task answer/layout fields
    control_state: dict  # exact native ControlState
    episode_id: str

    @property
    def frame_metadata(self) -> dict:
        return self.metadata


@dataclass(frozen=True)
class PracticeTransition:
    observation: PracticeObservation
    reward: float
    duration_ms: int
    outcome: Literal["continuing", "terminated", "truncated", "aborted"]
    raw_events: list[dict]
    command_results: list[dict]
    provenance: Literal["agent", "oracle"]
    # Cleanup is outside the policy interval; truncation observation is captured
    # before cleanup so value bootstrap never observes an artificial reset.
    cleanup_events: list[dict] = field(default_factory=list)
    reason: str | None = None


@dataclass(order=True)
class _Scheduled:
    time: int
    packet: int
    order: int
    serial: int
    command: dict = field(compare=False)
    reference: tuple[int, int] | None = field(compare=False)
    provenance: str = field(compare=False)


class PracticeEnvironment:
    """One isolated visual world; independent instances can be vectorized.

    step() admits one complete packet at observation time t, executes queued
    events in [t,t+T), and returns the observation at t+T. A boundary event at
    exactly t+T belongs to the next transition and follows that observation.
    No wall-clock sleeping, input permissions, filesystem or network is needed.
    """

    def __init__(self, config: PracticeConfig | None = None):
        self.config = (config or PracticeConfig()).validate()
        self.vocabulary = ActionVocabulary(
            key_codes=(123, 124) if self.config.task == "delayed_memory" else (),
            mouse_buttons=(0,), absolute_pointer=True, relative_pointer=self.config.relative_pointer,
        ).validate()
        self._now = 1_000_000_000
        self._episode_start = self._now
        self._episode_id: str | None = None
        self._outcome = "aborted"
        self._pending: list[_Scheduled] = []
        self._commands: dict[tuple[int, int], dict] = {}
        self._keys: set[int] = set()
        self._buttons: set[int] = set()
        self._pointer = (0.0, 0.0)
        self._revision = 0
        self._event_sequence = 0
        self._packet_sequence = 0
        self._serial = 0
        self._reset_count = 0
        self.current_seed: int | None = None

    @property
    def action_vocabulary(self) -> ActionVocabulary:
        return self.vocabulary

    @property
    def episode_id(self) -> str | None:
        return self._episode_id

    @property
    def outcome(self) -> str:
        """Current administrative/episode outcome without advancing the world."""
        return self._outcome

    @property
    def elapsed_ms(self) -> int:
        return (self._now - self._episode_start) // 1_000_000

    def reset(self, seed: int | None = None) -> PracticeObservation:
        seed = ((self.config.seed + self._reset_count) % 2**63 if seed is None
                else _integer(seed, 0, 2**63 - 1, "seed"))
        self.current_seed = seed
        self._reset_count += 1
        rng = np.random.default_rng(seed)
        # The virtual monotonic clock survives reset. Episode identity, never a
        # timestamp guess, rejects a stale caller's work after this boundary.
        self._now += 1
        self._episode_start = self._now
        self._episode_id = str(uuid.uuid4())
        self._outcome = "continuing"
        self._pending.clear(); self._commands.clear()
        self._keys.clear(); self._buttons.clear()
        self._revision = 0; self._event_sequence = 0; self._packet_sequence = 0; self._serial = 0
        x, y, width, height = self.config.logical_bounds
        self._pointer = (x + width * float(rng.uniform(.15, .85)), y + height * float(rng.uniform(.25, .85)))
        # Layout and cursor randomness precede the independent answer draw.
        self._target = (x + width * float(rng.uniform(.12, .88)), y + height * float(rng.uniform(.25, .84)))
        self._radius = min(width, height) * float(rng.uniform(.018, .030))
        self._choices = [
            (x + width * float(rng.uniform(.18, .40)), y + height * float(rng.uniform(.43, .78))),
            (x + width * float(rng.uniform(.60, .82)), y + height * float(rng.uniform(.43, .78))),
        ]
        self._choice_radius = min(width, height) * .070
        self._choice_colors = list(rng.permutation(2))
        self._answer = int(rng.integers(0, 2))
        return self._observe()

    def _validate_commands(self, commands: list[dict]) -> list[dict]:
        if type(commands) is not list or len(commands) > 64:
            raise PracticeError("A practice packet must be a list with at most 64 commands")
        result, previous, motion = [], 0, None
        fields = {
            "keyDown": {"keyCode"}, "keyUp": {"keyCode"}, "keyRepeat": {"keyCode"},
            "buttonDown": {"button"}, "buttonUp": {"button"},
            "pointerAbsolute": {"surfaceID", "x", "y"}, "pointerRelative": {"dx", "dy"},
        }
        for command in commands:
            if not isinstance(command, dict) or type(command.get("operation")) is not str or command["operation"] not in fields:
                raise PracticeError("Practice command operation is unavailable")
            op = command["operation"]
            if set(command) != fields[op] | {"operation", "offsetMs"}:
                raise PracticeError("Practice command has missing or irrelevant arguments")
            is_motion = op.startswith("pointer")
            offset = _integer(command["offsetMs"], previous, self.config.period_ms - (not is_motion), "offsetMs")
            previous = offset
            if op.startswith("key"):
                if type(command["keyCode"]) is not int or command["keyCode"] not in self.vocabulary.key_codes:
                    raise PracticeError("Key exceeds the environment vocabulary")
            elif op.startswith("button"):
                if type(command["button"]) is not int or command["button"] not in self.vocabulary.mouse_buttons:
                    raise PracticeError("Button exceeds the environment vocabulary")
            else:
                if motion is not None and motion != op:
                    raise PracticeError("Absolute and relative trajectories cannot share a packet")
                motion = op
                if op == "pointerRelative" and not self.vocabulary.relative_pointer:
                    raise PracticeError("Relative pointer is unavailable in this environment")
                names = ("x", "y") if op == "pointerAbsolute" else ("dx", "dy")
                if any(type(command[name]) not in (int, float) or not math.isfinite(command[name]) for name in names):
                    raise PracticeError("Pointer arguments must be finite numbers")
                if op == "pointerAbsolute":
                    if command["surfaceID"] != "practice" or any(not 0 <= command[name] < 1 for name in names):
                        raise PracticeError("Absolute pointer exceeds the observed surface")
                elif any(abs(command[name]) > 32768 or command[name] != int(command[name]) for name in names):
                    raise PracticeError("Relative pointer requires bounded integer raw counts")
            result.append(copy.deepcopy(command))
        return result

    def _schedule(self, commands: list[dict], provenance: str) -> None:
        packet = self._packet_sequence
        self._packet_sequence += 1
        start = self._now + self.config.lead_ms * 1_000_000
        previous_motion = None
        for index, command in enumerate(commands):
            scheduled = start + command["offsetMs"] * 1_000_000
            reference = (packet, index)
            self._commands[reference] = {
                "packetSequence": packet, "commandIndex": index, "episodeID": self._episode_id,
                "scheduledNanos": scheduled, "provenance": provenance,
            }
            if command["operation"].startswith("pointer"):
                if previous_motion is not None and command["offsetMs"] > previous_motion["offsetMs"]:
                    length = command["offsetMs"] - previous_motion["offsetMs"]
                    if command["operation"] == "pointerRelative":
                        last = (0, 0)
                    for tick in range(1, length):
                        fraction = tick / length
                        sample = dict(command)
                        if command["operation"] == "pointerAbsolute":
                            for name in ("x", "y"):
                                sample[name] = previous_motion[name] + fraction * (command[name] - previous_motion[name])
                        else:
                            cumulative = tuple(round(command[name] * fraction) for name in ("dx", "dy"))
                            sample["dx"], sample["dy"] = (cumulative[axis] - last[axis] for axis in range(2))
                            last = cumulative
                        self._push(start + (previous_motion["offsetMs"] + tick) * 1_000_000,
                                   packet, -1, sample, None, provenance)
                    if command["operation"] == "pointerRelative":
                        command = dict(command)
                        command["dx"] -= last[0]; command["dy"] -= last[1]
                previous_motion = commands[index]
            self._push(scheduled, packet, index, command, reference, provenance)

    def _push(self, time, packet, order, command, reference, provenance):
        heapq.heappush(self._pending, _Scheduled(time, packet, order, self._serial, command, reference, provenance))
        self._serial += 1

    def step(self, commands: list[dict], episode_id: str | None = None,
             provenance: Literal["agent", "oracle"] = "agent") -> PracticeTransition:
        if self._episode_id is None or self._outcome != "continuing":
            raise PracticeError("Reset is required before stepping a completed environment")
        if episode_id is not None and episode_id != self._episode_id:
            raise PracticeError("Commands belong to another episode")
        if provenance not in ("agent", "oracle"):
            raise PracticeError("Unknown command provenance")
        validated = self._validate_commands(commands)  # Atomic rejection before clock/queue mutation.
        self._schedule(validated, provenance)
        duration = min(self.config.period_ms, self.config.time_limit_ms - self.elapsed_ms)
        end = self._now + duration * 1_000_000
        initial_potential = self._potential()
        events, results, reward = [], [], 0.0
        while self._pending and self._pending[0].time < end:
            item = heapq.heappop(self._pending)
            self._now = item.time
            status, task_reward = self._execute(item, events)
            reward += task_reward
            if item.reference is not None:
                result = self._commands.pop(item.reference)
                results.append({**result, "status": status, "postedNanos": self._now} if status == "posted" else {**result, "status": status})
            if self._outcome == "terminated":
                break
        self._now = end
        if self._outcome == "continuing" and self.elapsed_ms >= self.config.time_limit_ms:
            self._outcome = "truncated"
        if self.config.shaping_scale:
            next_potential = 0.0 if self._outcome == "terminated" else self._potential()
            gamma = 2.0 ** (-duration / self.config.discount_half_life_ms)
            reward += self.config.shaping_scale * (gamma * next_potential - initial_potential)
        observation = self._observe()
        cleanup = []
        if self._outcome != "continuing":
            cleanup, cancelled = self._cleanup()
            results.extend(cancelled)
        return PracticeTransition(observation, float(reward), duration, self._outcome, events, results, provenance, cleanup)

    def abort(self, reason: str = "Stopped") -> PracticeTransition:
        if self._episode_id is None or self._outcome != "continuing":
            raise PracticeError("Only a running episode can be aborted")
        if not isinstance(reason, str) or not 1 <= len(reason) <= 512:
            raise PracticeError("Abort reason must be a short nonempty string")
        failure = None
        try:
            observation = self._observe()
        except BaseException as error:
            failure = error
        finally:
            self._outcome = "aborted"
            try:
                cleanup, results = self._cleanup()
            except BaseException as cleanup_error:
                if failure is not None:
                    failure.add_note(f"Practice cleanup also failed ({type(cleanup_error).__name__}).")
                    raise failure from cleanup_error
                raise
        if failure is not None:
            raise failure
        # No elapsed transition is fabricated. This administrative boundary is
        # excluded from PPO, rather than bootstrapped or treated as a failure.
        return PracticeTransition(observation, 0.0, 0, "aborted", [], results, "agent", cleanup, reason)

    def _emit(self, events: list[dict], kind: str, provenance: str, *, origin="agent", **arguments):
        event = {
            "sequence": self._event_sequence, "eventNanos": self._now, "observedNanos": self._now,
            "origin": origin, "kind": kind, "detail": f"practice:{provenance}", **arguments,
        }
        validate_event(event)
        self._event_sequence += 1
        events.append(event)

    def _execute(self, item: _Scheduled, events: list[dict]) -> tuple[str, float]:
        command, provenance = item.command, item.provenance
        op = command["operation"]
        if op.startswith("pointer"):
            x, y, width, height = self.config.logical_bounds
            old_x, old_y = self._pointer
            if op == "pointerAbsolute":
                point = (x + width * command["x"], y + height * command["y"])
            else:
                point = (min(x + width, max(x, old_x + command["dx"])),
                         min(y + height, max(y, old_y + command["dy"])))
            if point == self._pointer and not (op == "pointerRelative" and (command["dx"] or command["dy"])):
                return "noOp", 0.0
            self._pointer = point; self._revision += 1
            dx, dy = (command["dx"], command["dy"]) if op == "pointerRelative" else (point[0] - old_x, point[1] - old_y)
            self._emit(events, "pointer", provenance, x=point[0], y=point[1], dx=dx, dy=dy)
            return "posted", 0.0
        key = "keyCode" if op.startswith("key") else "button"
        value = command[key]
        held = self._keys if key == "keyCode" else self._buttons
        if op == "keyRepeat":
            if value not in held:
                return "noOp", 0.0
        elif op.endswith("Down"):
            if value in held:
                return "noOp", 0.0
            held.add(value)
        else:
            if value not in held:
                return "noOp", 0.0
            held.remove(value)
        self._revision += 1
        self._emit(events, op, provenance, **{key: value}, x=self._pointer[0], y=self._pointer[1])
        reward = self._activate(value if key == "keyCode" else None) if op in ("buttonDown", "keyDown") else 0.0
        return "posted", reward

    def _activate(self, key: int | None) -> float:
        if self.config.task == "pointing":
            if math.dist(self._pointer, self._target) <= self._radius:
                self._outcome = "terminated"
                return 1.0
            return -.05
        if self.elapsed_ms < self.config.cue_ms + self.config.delay_ms:
            return 0.0
        if key is not None:
            choice = 0 if key == 123 else 1
        else:
            choice = next((index for index, point in enumerate(self._choices)
                           if math.dist(self._pointer, point) <= self._choice_radius), None)
        if choice is None:
            return -.05
        self._outcome = "terminated"
        return 1.0 if self._choice_colors[choice] == self._answer else -1.0

    def _potential(self) -> float:
        if self.config.task != "pointing":
            return 0.0
        _, _, width, height = self.config.logical_bounds
        scale = max(width, height)
        distance = math.hypot((self._pointer[0] - self._target[0]) / scale,
                              (self._pointer[1] - self._target[1]) / scale)
        return -distance / math.hypot(width / scale, height / scale)

    def _cleanup(self) -> tuple[list[dict], list[dict]]:
        events = []
        for key in sorted(self._keys):
            self._emit(events, "keyUp", "cleanup", origin="boundary", keyCode=key)
        for button in sorted(self._buttons):
            self._emit(events, "buttonUp", "cleanup", origin="boundary", button=button)
        results = [{**value, "status": "cancelled", "message": "Episode boundary cancelled queued work"}
                   for _, value in sorted(self._commands.items())]
        self._pending.clear(); self._commands.clear(); self._keys.clear(); self._buttons.clear()
        self._revision += bool(events)
        return events, results

    def oracle_commands(self) -> list[dict]:
        """Privileged fixture labels, never part of an actor observation.

        An oracle caller must pass provenance='oracle' to step(). It should not
        mix oracle labels with agent actions inside a single episode.
        """
        if self._episode_id is None or self._outcome != "continuing":
            raise PracticeError("The oracle requires a running episode")
        if self._pending:
            return []
        if self.config.task == "delayed_memory":
            if self.elapsed_ms < self.config.cue_ms + self.config.delay_ms:
                return []
            target = self._choices[self._choice_colors.index(self._answer)]
        else:
            target = self._target
        x, y, width, height = self.config.logical_bounds
        # Stable equal-time array ordering also supports a 1 ms decision period.
        return [
            {"operation": "pointerAbsolute", "offsetMs": 0, "surfaceID": "practice",
             "x": (target[0] - x) / width, "y": (target[1] - y) / height},
            {"operation": "buttonUp", "offsetMs": 0, "button": 0},
            {"operation": "buttonDown", "offsetMs": 0, "button": 0},
        ]

    def _observe(self) -> PracticeObservation:
        metadata = {
            "id": str(uuid.uuid4()), "eventNanos": self._now, "observedNanos": self._now,
            "surface": self.config.surface(), "byteCount": self.config.pixel_width * self.config.pixel_height * 4,
            "pixelFormat": "bgra8-srgb", "codec": "raw",
        }
        validate_frame(metadata)
        controls = {"keys": sorted(self._keys), "buttons": sorted(self._buttons), "modifiers": 0,
                    "pointer": {"x": self._pointer[0], "y": self._pointer[1]},
                    "observedNanos": self._now, "revision": self._revision, "valid": True}
        return PracticeObservation(self._render(), metadata, controls, self._episode_id)

    def _render(self) -> np.ndarray:
        width, height = self.config.pixel_width, self.config.pixel_height
        image = Image.new("RGBA", (width, height), (244, 246, 249, 255))
        draw = ImageDraw.Draw(image)
        x, y, logical_width, logical_height = self.config.logical_bounds
        sx, sy = width / logical_width, height / logical_height
        colors = ((20, 140, 158, 255), (221, 115, 38, 255))

        def circle(point, radius, color):
            px, py = (point[0] - x) * sx, (point[1] - y) * sy
            draw.ellipse((px - radius * sx, py - radius * sy, px + radius * sx, py + radius * sy), fill=color)

        if self.config.task == "pointing":
            title = "Click the teal target"
            circle(self._target, self._radius, colors[0])
        elif self.elapsed_ms < self.config.cue_ms:
            title = "Remember this color"
            circle((x + logical_width / 2, y + logical_height / 2), min(logical_width, logical_height) * .11, colors[self._answer])
        elif self.elapsed_ms < self.config.cue_ms + self.config.delay_ms:
            title = "Wait for the choices"
            # Cue region is rebuilt from the background every frame. No hidden
            # cue, color-dependent countdown, answer metadata or shaping remains.
            draw.line((width * .48, height * .5, width * .52, height * .5), fill=(110, 117, 127, 255), width=max(1, round(height / 360)))
        else:
            title = "Choose the remembered color"
            for point, color in zip(self._choices, self._choice_colors):
                circle(point, self._choice_radius, colors[color])
        draw.text((max(4, width * .035), max(4, height * .045)), title, fill=(36, 43, 54, 255))
        px, py = (self._pointer[0] - x) * sx, (self._pointer[1] - y) * sy
        cursor = max(3, min(width, height) * .010)
        draw.polygon(((px, py), (px, py + cursor * 1.5), (px + cursor, py + cursor)),
                     fill=(28, 33, 43, 255), outline=(255, 255, 255, 255))
        pixels = np.ascontiguousarray(np.asarray(image)[..., [2, 1, 0, 3]])
        pixels.setflags(write=False)
        return pixels
