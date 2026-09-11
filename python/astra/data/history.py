from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable
import numpy as np

from astra.recordings import RecordingError, validate_event, validate_surface


@dataclass
class ControlHistory:
    """Reduce only events available by the observation cutoff.

    Physical source timestamps remain untouched. Delivery time determines what
    the agent could have known; future teacher packets never enter this state.
    """
    keys: set[int] = field(default_factory=set)
    buttons: set[int] = field(default_factory=set)
    modifiers: int = 0
    pointer: tuple[float, float] | None = None
    valid: bool = False
    sequence: int | None = None
    observed_nanos: int = 0
    last_input_nanos: int | None = None
    motion: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    scroll: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))

    def apply(self, event: dict) -> None:
        validate_event(event)
        if event["eventNanos"] > event["observedNanos"]:
            raise RecordingError("Input source time cannot follow its availability time")
        if event["observedNanos"] < self.observed_nanos:
            raise RecordingError("Control history was delivered out of causal order")
        if self.sequence is not None and event["sequence"] != self.sequence + 1:
            self.valid = False
        self.sequence = event["sequence"]
        self.observed_nanos = event["observedNanos"]
        kind = event["kind"]
        if event["origin"] == "boundary" or kind == "gap":
            self.valid = False
            return
        # Native v1 seeds the complete physical state as one atomic batch:
        # sequence-zero reconciliation pointer, then all held keys/buttons at
        # that exact availability time. Empty holds are represented by absence.
        if event["sequence"] == 0 and event["origin"] == "reconciliation" and kind == "pointer":
            self.keys.clear(); self.buttons.clear(); self.valid = True
        if "modifiers" in event:
            self.modifiers = event["modifiers"]
        if event.get("x") is not None and event.get("y") is not None:
            self.pointer = (event["x"], event["y"])
        if kind == "keyDown":
            self.keys.add(event["keyCode"])
        elif kind == "keyUp":
            self.keys.discard(event["keyCode"])
        elif kind == "buttonDown":
            self.buttons.add(event["button"])
        elif kind == "buttonUp":
            self.buttons.discard(event["button"])
        elif kind == "flags" and event.get("keyCode") is not None and event.get("isDown") is not None:
            if event["isDown"]:
                self.keys.add(event["keyCode"])
            else:
                self.keys.discard(event["keyCode"])
        if kind == "pointer" and event["origin"] in ("physical", "agent"):
            self.motion += (event.get("dx", 0), event.get("dy", 0))
        if kind == "scroll":
            self.scroll += (event["scrollX"], event["scrollY"])
        if event["origin"] in ("physical", "agent"):
            self.last_input_nanos = event["observedNanos"]

    def features(self, cutoff_nanos: int, surfaces: Iterable[dict], *, interval_covered: bool) -> np.ndarray:
        if cutoff_nanos < self.observed_nanos:
            raise RecordingError("Observed control history would leak beyond the model cutoff")
        surfaces = [validate_surface(surface) for surface in surfaces]
        if not surfaces:
            raise RecordingError("Control features require observed surface geometry")
        result = np.zeros(178, dtype=np.float32)
        result[list(self.keys)] = 1
        result[[128 + button for button in self.buttons]] = 1
        for offset in range(8):
            result[160 + offset] = bool(self.modifiers & (1 << (16 + offset)))
        bounds = [surface["globalBounds"] for surface in surfaces]
        if self.pointer is not None:
            x, y = self.pointer
            left, top = min(rect["x"] for rect in bounds), min(rect["y"] for rect in bounds)
            right, bottom = max(rect["x"] + rect["width"] for rect in bounds), max(rect["y"] + rect["height"] for rect in bounds)
            result[168:170] = np.clip(((x - left) / (right - left), (y - top) / (bottom - top)), 0, 1)
            result[175] = any(rect["x"] <= x < rect["x"] + rect["width"] and rect["y"] <= y < rect["y"] + rect["height"] for rect in bounds)
        result[170:172] = np.tanh(self.motion / 2048)
        result[172:174] = np.tanh(self.scroll / 128)
        if self.last_input_nanos is not None:
            age = (cutoff_nanos - self.last_input_nanos) / 1e9
            result[174] = min(math.log1p(age) / math.log(61), 1)
        else:
            result[174] = 1
        result[176] = self.valid
        result[177] = interval_covered
        return result

    def advance_interval(self) -> None:
        self.motion.fill(0); self.scroll.fill(0)
