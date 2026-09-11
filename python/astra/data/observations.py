"""Build one causal model observation from native or virtual environment data."""
from __future__ import annotations

from typing import Sequence
import math
import mlx.core as mx

from astra.model.config import ModelConfig
from astra.model.observation import ObservationBatch
from astra.recordings import RecordingError, validate_frame, validate_event
from .history import ControlHistory
from .preprocessing import prepare_surface


def make_observation(frames: Sequence[tuple[object, dict]], control_state: dict, *, cutoff_nanos: int,
                     elapsed_seconds: float, reset: bool, config: ModelConfig,
                     context_ids: tuple[int, ...] = (), executed_events: Sequence[dict] = (),
                     interval_covered: bool = True, surface_preparer=prepare_surface,
                     maximum_timestamp: int = 2**63 - 1, last_input_nanos: int | None = None) -> ObservationBatch:
    if (type(maximum_timestamp) is not int or maximum_timestamp not in (2**63 - 1, 2**64 - 1)
        or type(cutoff_nanos) is not int or not 0 <= cutoff_nanos <= maximum_timestamp
        or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0):
        raise RecordingError("Invalid observation cutoff or elapsed time")
    if not 1 <= len(frames) <= config.maximum_surfaces or len(context_ids) != len(config.context_sizes):
        raise RecordingError("Observation surface/context layout does not match its model")
    for value, size in zip(context_ids, config.context_sizes):
        if type(value) is not int or not 0 <= value < size:
            raise RecordingError("Observation context ID is outside the checkpoint vocabulary")
    required = {"keys", "buttons", "modifiers", "pointer", "observedNanos", "revision", "valid"}
    if not isinstance(control_state, dict) or set(control_state) != required:
        raise RecordingError("Invalid observed control-state schema")
    for name, maximum in (("keys", 127), ("buttons", 31)):
        values = control_state[name]
        if not isinstance(values, list) or any(type(value) is not int or not 0 <= value <= maximum for value in values) or values != sorted(set(values)):
            raise RecordingError("Observed controls must be bounded sorted unique arrays")
    for name, maximum in (("modifiers", 2**64 - 1), ("revision", 2**64 - 1), ("observedNanos", cutoff_nanos)):
        value = control_state[name]
        if type(value) is not int or not 0 <= value <= maximum:
            raise RecordingError("Observed control state exceeds its time or value bounds")
    pointer = control_state["pointer"]
    if not isinstance(pointer, dict) or set(pointer) != {"x", "y"} or any(type(value) not in (int, float) or not math.isfinite(value) for value in pointer.values()):
        raise RecordingError("Observed cursor has invalid geometry")
    if type(control_state["valid"]) is not bool:
        raise RecordingError("Observed control validity must be Boolean")
    if last_input_nanos is not None and (type(last_input_nanos) is not int or not 0 <= last_input_nanos <= cutoff_nanos):
        raise RecordingError("Last executed input exceeds the observation cutoff")
    history = ControlHistory(keys=set(control_state["keys"]), buttons=set(control_state["buttons"]),
                             modifiers=control_state["modifiers"], pointer=(pointer["x"], pointer["y"]),
                             valid=control_state["valid"], observed_nanos=control_state["observedNanos"], last_input_nanos=last_input_nanos)
    for event in executed_events:
        validate_event(event, maximum_timestamp=maximum_timestamp)
        if event["observedNanos"] > cutoff_nanos or event["eventNanos"] > cutoff_nanos:
            raise RecordingError("Executed input would leak beyond the observation cutoff")
        if event["origin"] in ("physical", "agent"):
            history.last_input_nanos = max(history.last_input_nanos or 0, event["observedNanos"])
            if event["kind"] == "pointer":
                history.motion += (event.get("dx", 0), event.get("dy", 0))
            elif event["kind"] == "scroll":
                history.scroll += (event["scrollX"], event["scrollY"])
    surfaces, prepared = [], []
    for pixels, metadata in frames:
        validate_frame(metadata, maximum_timestamp=maximum_timestamp)
        if metadata["eventNanos"] > cutoff_nanos or metadata["observedNanos"] > cutoff_nanos:
            raise RecordingError("Frame source/availability time exceeds its observation cutoff")
        surfaces.append(metadata["surface"])
        prepared.append(surface_preparer(pixels, metadata, config, pointer=history.pointer if history.valid else None,
                                         maximum_timestamp=maximum_timestamp).batch())
    if len({surface["id"] for surface in surfaces}) != len(surfaces):
        raise RecordingError("Observed surface roles must be unique")
    controls = history.features(cutoff_nanos, surfaces, interval_covered=interval_covered)
    return ObservationBatch(tuple(prepared), mx.array(controls)[None, None], mx.array([[elapsed_seconds]], dtype=mx.float32),
                            mx.array(context_ids, dtype=mx.int32)[None, None], mx.array([[reset]], dtype=mx.bool_),
                            mx.array([[control_state["valid"] and interval_covered]], dtype=mx.bool_))
