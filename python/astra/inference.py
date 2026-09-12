"""Local actor ownership, mapped-frame ingestion and training-equivalent visuals.

The actor uses the shared policy/observation/command code. Only its owner thread
touches MLX state; the protocol thread remains available for stop and liveness.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import queue
import threading
import uuid

import mlx.core as mx
import numpy as np

from astra.checkpoints import load_checkpoint
from astra.data.actions import decode_commands
from astra.data.observations import make_observation
from astra.data.preprocessing import MEAN, STD
from astra.frame_ring import FrameRingReader
from astra.model.actions import PacketBatch, flatten_visual
from astra.model.observation import ObservationBatch, SurfaceBatch
from astra.model.vision import VisualFeatures
from astra.recordings import validate_frame

INFERENCE_OPERATIONS = ("inference.prepare", "inference.reset", "inference.warmup", "inference.step", "inference.close")


class InferenceError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _fields(value, required, optional=()):
    if type(value) is not dict or not set(required) <= value.keys() or set(value) - set(required) - set(optional):
        raise InferenceError("inference.configuration", "Inference configuration has missing or unknown fields")


def _uint(value, maximum=2**64 - 1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise InferenceError("inference.configuration", "Inference integer is outside its supported range")
    return value


def _uuid(value):
    if type(value) is not str or len(value) != 36:
        raise InferenceError("inference.identity", "Inference requires a UUID identity")
    return str(uuid.UUID(value))


def _path(value):
    if type(value) is not str or not 1 <= len(value.encode()) <= 4096 or "\0" in value:
        raise InferenceError("inference.path", "Inference requires a bounded absolute artifact path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path.is_symlink():
        raise InferenceError("inference.path", "Inference artifact paths must be absolute and cannot be symbolic links")
    return path


@lru_cache(maxsize=8)
def _lanczos_coefficients(source_size, start, end, destination_size):
    """Geometry-only coefficients; no source pixels are read on the CPU.

    Match Pillow's public RGB LANCZOS resize contract, including its float32
    crop bounds and signed 22-bit coefficient rounding. Kernel qualification
    uses Pillow 12.3.0, src/libImaging/Resample.c, as the independent reference.
    """
    start, end = np.float32(start), np.float32(end)
    scale = float(np.float32(end - start)) / destination_size
    filter_scale = max(1.0, scale)
    support = 3 * filter_scale
    taps = 2 * math.ceil(support) + 1
    starts = np.empty(destination_size, dtype=np.int32)
    counts = np.empty(destination_size, dtype=np.int32)
    coefficients = np.zeros((destination_size, taps), dtype=np.int32)
    def sinc(value):
        return 1.0 if value == 0 else math.sin(math.pi * value) / (math.pi * value)
    for output in range(destination_size):
        center = float(start) + (output + .5) * scale
        first = max(0, int(center - support + .5))
        stop = min(source_size, int(center + support + .5))
        weights = []
        for source in range(first, stop):
            distance = (source - center + .5) / filter_scale
            weights.append(sinc(distance) * sinc(distance / 3) if -3 <= distance < 3 else 0.0)
        total = sum(weights)
        if not weights or not total:
            raise InferenceError("inference.geometry", "The resize footprint does not contain source pixels")
        starts[output], counts[output] = first, len(weights)
        for tap, value in enumerate(weights):
            normalized = value / total * (1 << 22)
            coefficients[output, tap] = int(normalized + (.5 if normalized >= 0 else -.5))
    for array in (starts, counts, coefficients):
        array.setflags(write=False)
    return starts, counts, coefficients


@lru_cache(maxsize=1)
def _resize_kernel():
    # One thread owns one output channel. The int32 accumulation rounds once
    # per separable pass, matching the RGB training reference without a large
    # H*W*filter-support gather tensor or any GPU-to-CPU image transfer.
    return mx.fast.metal_kernel(
        name="astra_lanczos_rgb_v1", input_names=["pixels", "starts", "counts", "weights", "dimensions"],
        output_names=["resized"], source="""
            uint index = thread_position_in_grid.x;
            uint height = dimensions[0], width = dimensions[1];
            uint target = dimensions[2], taps = dimensions[3], vertical = dimensions[4];
            uint outputWidth = vertical ? width : target;
            uint outputHeight = vertical ? target : height;
            if (index >= outputWidth * outputHeight * 3) return;
            uint channel = index % 3;
            uint x = (index / 3) % outputWidth, y = index / (3 * outputWidth);
            uint coordinate = vertical ? y : x;
            int first = starts[coordinate];
            int accumulator = 2097152;
            for (int tap = 0; tap < counts[coordinate]; ++tap) {
                uint sourceX = vertical ? x : uint(first + tap);
                uint sourceY = vertical ? uint(first + tap) : y;
                accumulator += int(pixels[(sourceY * width + sourceX) * 3 + channel])
                               * weights[coordinate * taps + tap];
            }
            resized[index] = uchar(clamp(accumulator >> 22, 0, 255));
        """)


def _resize_axis(image, start, end, size, axis):
    source_size = image.shape[axis]
    if size == source_size and start == 0 and end == size:
        return image
    starts, counts, coefficients = _lanczos_coefficients(source_size, start, end, size)
    shape = list(image.shape); shape[axis] = size
    dimensions = mx.array([image.shape[0], image.shape[1], size, coefficients.shape[1], axis == 0], dtype=mx.uint32)
    return _resize_kernel()(inputs=[image, mx.array(starts), mx.array(counts), mx.array(coefficients), dimensions],
                            grid=(math.prod(shape), 1, 1), threadgroup=(128, 1, 1),
                            output_shapes=[tuple(shape)], output_dtypes=[mx.uint8])[0]


def _normalize(image):
    return (image.astype(mx.float32) / 255 - mx.array(MEAN)) / mx.array(STD)


def _letterbox(image, box, limit, alignment):
    width, height = box[2] - box[0], box[3] - box[1]
    scale = min(1.0, (limit // alignment * alignment) / max(width, height))
    output_width, output_height = max(1, round(width * scale)), max(1, round(height * scale))
    canvas_width = max(32, math.ceil(output_width / alignment) * alignment)
    canvas_height = max(32, math.ceil(output_height / alignment) * alignment)
    left, top = (canvas_width - output_width) // 2, (canvas_height - output_height) // 2
    resized = _resize_axis(image, box[0], box[2], output_width, 1)
    resized = _resize_axis(resized, box[1], box[3], output_height, 0)
    normalized = mx.pad(_normalize(resized), [(top, canvas_height - output_height - top),
                                             (left, canvas_width - output_width - left), (0, 0)])
    rect = mx.array([left / canvas_width, top / canvas_height, output_width / canvas_width, output_height / canvas_height], dtype=mx.float32)
    return normalized, rect


@dataclass(frozen=True)
class _PreparedMetalSurface:
    values: dict

    def batch(self):
        return SurfaceBatch(**{key: value[None, None] for key, value in self.values.items()})


def prepare_metal_surface(pixels, metadata, config, *, pointer, maximum_timestamp=2**64 - 1):
    """Prepare owned MLX BGRA on Metal with the CPU training geometry contract."""
    config.validate(); validate_frame(metadata, maximum_timestamp=maximum_timestamp)
    surface = metadata["surface"]
    if not isinstance(pixels, mx.array) or pixels.dtype != mx.uint8 or pixels.shape != (surface["pixelHeight"], surface["pixelWidth"], 4):
        raise InferenceError("inference.pixels", "Owned BGRA pixels do not match their observation metadata")
    rgb = mx.take(pixels, mx.array([2, 1, 0]), axis=-1).astype(mx.float32)
    alpha = pixels[..., 3:4].astype(mx.float32) / 255
    rgb = mx.round(mx.clip(rgb + mx.array(MEAN * 255) * (1 - alpha), 0, 255)).astype(mx.uint8)
    content, bounds = surface["contentBounds"], surface["globalBounds"]
    box = (content["x"], content["y"], content["x"] + content["width"], content["y"] + content["height"])
    global_image, global_rect = _letterbox(rgb, box, config.global_long_edge, 32)
    detail_image, detail_rect = _letterbox(rgb, box, config.detail_long_edge, 8)
    geometry = [bounds[key] for key in ("x", "y", "width", "height")]
    if any(abs(value) > float(np.finfo(np.float32).max) for value in geometry) or any(np.float32(value) <= 0 for value in geometry[2:]):
        raise InferenceError("inference.geometry", "Surface bounds exceed the model's numerical range")
    size = config.cursor_size
    crop = mx.zeros((size, size, 3), dtype=mx.float32)
    crop_rect = [2, 2, 1, 1]
    if pointer is not None:
        if len(pointer) != 2 or any(type(value) not in (int, float) or not math.isfinite(value) for value in pointer):
            raise InferenceError("inference.geometry", "The observed pointer is invalid")
        px = content["x"] + (pointer[0] - bounds["x"]) / bounds["width"] * content["width"]
        py = content["y"] + (pointer[1] - bounds["y"]) / bounds["height"] * content["height"]
        if not math.isfinite(px) or not math.isfinite(py):
            raise InferenceError("inference.geometry", "The observed pointer cannot be mapped into source pixels")
        left, top = math.floor(px - size / 2), math.floor(py - size / 2)
        x0, y0 = max(left, math.floor(box[0])), max(top, math.floor(box[1]))
        x1, y1 = min(left + size, math.ceil(box[2])), min(top + size, math.ceil(box[3]))
        if x1 > x0 and y1 > y0:
            crop = mx.pad(_normalize(rgb[y0:y1, x0:x1]), [(y0 - top, top + size - y1), (x0 - left, left + size - x1), (0, 0)])
        crop_rect = [(left - content["x"]) / content["width"], (top - content["y"]) / content["height"], size / content["width"], size / content["height"]]
        if any(not math.isfinite(value) or abs(value) > float(np.finfo(np.float32).max) for value in crop_rect):
            raise InferenceError("inference.geometry", "Cursor geometry exceeds the model's numerical range")
    return _PreparedMetalSurface(dict(global_image=global_image, detail_image=detail_image, cursor_image=crop,
                                      global_content_rect=global_rect, content_rect=detail_rect,
                                      cursor_rect=mx.array(crop_rect, dtype=mx.float32),
                                      global_bounds=mx.array(geometry, dtype=mx.float32), available=mx.array(True)))


def _policy_arrays(policy, tensors, state, key, *, greedy):
    """The shared policy/decoder expressed as array trees for MLX compilation.

    Parameters remain explicit captured inputs via policy.state. No model,
    probability precision, RNG splitting or packet factor changes happen here.
    """
    encoding = policy(ObservationBatch.from_tensors(tensors), state)
    next_key, action_key = mx.random.split(key)
    sampled = policy.sample(encoding, key=action_key, greedy=greedy)
    state = tuple(mx.stop_gradient(value) for value in encoding.temporal.state)
    finite = mx.all(mx.stack([mx.all(mx.isfinite(value)) for value in
                             (*state, sampled.log_probability, sampled.conditional_entropy, encoding.temporal.value)]))
    visual = flatten_visual(encoding.visual)
    return dict(state=state, key=next_key, sample_key=action_key, finite=finite, value=encoding.temporal.value,
                log_probability=sampled.log_probability, conditional_entropy=sampled.conditional_entropy,
                packet={name: getattr(sampled.packets, name) for name in PacketBatch.__dataclass_fields__},
                visual={name: getattr(visual, name) for name in VisualFeatures.__dataclass_fields__})


class _PolicyExecution:
    """One immutable actor policy, with a bounded shape-specific compile cache.

    The cache keys contain shapes/dtypes only, never pixels, array references,
    cursor positions or observations. Runtime checkpoint activation creates a
    new owner so no graph can retain a previous policy's parameter identity.
    """
    maximum_buckets = 4

    def __init__(self, policy, *, greedy, compiled=True):
        self.policy, self.greedy, self.compiled = policy, greedy, compiled
        self._cache = OrderedDict()

    @staticmethod
    def _shape(value):
        if isinstance(value, mx.array):
            return (value.shape, str(value.dtype))
        if isinstance(value, dict):
            return tuple((name, _PolicyExecution._shape(child)) for name, child in sorted(value.items()))
        if isinstance(value, (tuple, list)):
            return tuple(_PolicyExecution._shape(child) for child in value)
        raise InferenceError("inference.tensors", "Actor compilation requires array-only observation/state trees")

    def __call__(self, observation, state, key):
        tensors = observation.as_tensors()
        # None and an explicit zero state have identical model semantics. A
        # fixed state signature lets warmup compile the first and later steps.
        state = self.policy.temporal.initial_state(observation.shape[0]) if state is None else state
        bucket = self._shape((tensors, state, key))
        runner = self._cache.pop(bucket, None)
        if runner is None:
            policy, greedy = self.policy, self.greedy
            def run(tensors, state, key):
                return _policy_arrays(policy, tensors, state, key, greedy=greedy)
            runner = mx.compile(run, inputs=policy.state) if self.compiled else run
            # Drop the old callable before tracing another geometry. The MLX
            # callable owns its shape trace and captured parameter references.
            if len(self._cache) >= self.maximum_buckets:
                self._cache.popitem(last=False)
        self._cache[bucket] = runner
        return runner(tensors, state, key)


class InferenceSession:
    """One immutable-policy actor stream, used only from one MLX owner thread."""
    def __init__(self):
        self._owner = threading.get_ident()
        self._reader = self._checkpoint = self._run_id = None
        self._execution = None
        self._state = self._state_id = self._episode_id = self._key = None
        self._last_cutoff = self._last_event_sequence = self._geometry_revision = None
        self._last_input_nanos = None
        self._surfaces = None
        self._sequence = 0
        self._observations = deque(maxlen=1024)
        self._contexts = ()
        self._needs_reset = True
        self._collection = self._warming = False
        self._rng_stream_id = None
        self._draw_index = self._episode_step = self._environment_resets = 0

    def _assert_owner(self):
        if threading.get_ident() != self._owner:
            raise InferenceError("inference.owner", "An actor session belongs to its single MLX owner thread")

    def _identity(self):
        manifest = self._checkpoint.manifest
        return {"runID": self._run_id, "checkpointID": manifest["id"], "policySignature": manifest["policySignature"],
                "stateID": self._state_id, "episodeID": self._episode_id, "needsReset": self._needs_reset}

    def prepare(self, payload, *, run_id):
        self._assert_owner()
        if self._reader is not None:
            raise InferenceError("inference.active", "Close this actor run before preparing another stream")
        _fields(payload, ("checkpointPath", "ring"), ("seed", "deterministic", "collection"))
        _fields(payload["ring"], ("path", "ringID"))
        run_id = _uuid(run_id)
        ring_id = _uuid(payload["ring"]["ringID"])
        seed = _uint(payload.get("seed", 0))
        deterministic = payload.get("deterministic", True)
        collection = payload.get("collection", False)
        if type(deterministic) is not bool or type(collection) is not bool:
            raise InferenceError("inference.configuration", "Inference mode must be Boolean")
        if collection and deterministic:
            raise InferenceError("inference.collectionMode", "On-policy collection requires categorical sampling, not greedy actions")
        checkpoint = load_checkpoint(_path(payload["checkpointPath"]))
        if checkpoint.policy.config.control_width != 178:
            raise InferenceError("inference.configuration", "This actor requires the version-one observed-control feature layout")
        reader = FrameRingReader(_path(payload["ring"]["path"]), run_id=run_id, ring_id=ring_id)
        try:
            checkpoint.policy.eval()
            key = mx.random.key(seed)
        except BaseException:
            reader.close()
            raise
        self._reader, self._checkpoint, self._run_id, self._key = reader, checkpoint, run_id, key
        self._deterministic = deterministic
        self._collection = collection
        self._rng_stream_id = str(uuid.uuid4())
        self._execution = _PolicyExecution(checkpoint.policy, greedy=deterministic)
        return {**self._identity(), "model": checkpoint.manifest["model"], "actions": checkpoint.manifest["actions"],
                "ringID": ring_id, "deterministic": deterministic, "collection": collection,
                "collectionVersion": 1 if collection else None, "rngStreamID": self._rng_stream_id if collection else None}

    def _run(self, run_id):
        self._assert_owner()
        if self._reader is None or _uuid(run_id) != self._run_id:
            raise InferenceError("inference.run", "Inference request does not belong to the prepared actor run")

    def reset(self, payload, *, run_id):
        self._run(run_id)
        _fields(payload, ("confirmed", "episodeID", "contextIDs"), ("checkpointPath", "seed"))
        if payload["confirmed"] is not True:
            raise InferenceError("inference.resetRequired", "Checkpoint activation requires a confirmed environment reset")
        episode_id = _uuid(payload["episodeID"])
        if episode_id == self._episode_id:
            raise InferenceError("inference.episode", "Each confirmed environment reset requires a new episode identity")
        # The coordinator says the environment has reset. If replacement
        # configuration/loading fails, old recurrent state must stay unusable
        # until a valid reset is committed, even though old weights are retained.
        self._needs_reset = True
        if self._collection and "seed" in payload:
            raise InferenceError("inference.collectionReseed", "A collecting actor retains its random stream across episode resets")
        contexts = payload["contextIDs"]
        config = self._checkpoint.policy.config
        if type(contexts) is not list or len(contexts) != len(config.context_sizes):
            raise InferenceError("inference.context", "Reset context choices must match the checkpoint vocabulary")
        for value, size in zip(contexts, config.context_sizes):
            _uint(value, size - 1)
        key = self._key if "seed" not in payload else mx.random.key(_uint(payload["seed"]))
        checkpoint = self._checkpoint
        if "checkpointPath" in payload:
            checkpoint = load_checkpoint(_path(payload["checkpointPath"]))
            if checkpoint.manifest["policySignature"] != self._checkpoint.manifest["policySignature"]:
                raise InferenceError("inference.policyMismatch", "A new model or action vocabulary requires a new actor run")
            checkpoint.policy.eval()
        # Commit only after the entire new checkpoint/context/reset validates.
        if checkpoint is not self._checkpoint:
            self._execution = _PolicyExecution(checkpoint.policy, greedy=self._deterministic)
        self._checkpoint, self._episode_id, self._contexts, self._key = checkpoint, episode_id, tuple(contexts), key
        self._state, self._state_id, self._needs_reset = None, str(uuid.uuid4()), False
        self._last_input_nanos = None
        self._episode_step = 0
        self._environment_resets += 1
        return self._identity()

    def warmup(self, payload, *, run_id):
        """Exercise the exact observation/action path without advancing actor state.

        Frame leases really are ingested and consumed; they are never restored
        or acknowledged twice. This operation is available only before a run's
        first real decision, while the native coordinator has no armed controls.
        """
        self._run(run_id)
        already_started = (self._episode_step != 0 or self._state is not None) if self._collection else (self._sequence != 0 or self._last_cutoff is not None)
        if already_started:
            raise InferenceError("inference.warmupActive", "Warmup must finish before the first real actor decision")
        fields = ("_state", "_state_id", "_episode_id", "_key", "_last_cutoff", "_last_event_sequence",
                  "_geometry_revision", "_last_input_nanos", "_surfaces", "_sequence", "_contexts", "_needs_reset",
                  "_draw_index", "_episode_step", "_environment_resets", "_warming")
        snapshot = {field: getattr(self, field) for field in fields}
        observations = deque(self._observations, maxlen=self._observations.maxlen)
        try:
            self._warming = True
            result = self.step(payload, run_id=run_id)
            result["warmup"] = True
            return result
        finally:
            for field, value in snapshot.items():
                setattr(self, field, value)
            self._observations = observations

    def step(self, payload, *, run_id):
        self._run(run_id)
        if self._needs_reset:
            raise InferenceError("inference.resetRequired", "A confirmed environment reset is required before inference")
        _fields(payload, ("observationID", "episodeID", "previousStateID", "cutoffNanos", "geometryRevision",
                          "frames", "controlState", "executedEvents", "intervalCovered", "contextIDs"))
        if _uuid(payload["previousStateID"]) != self._state_id or _uuid(payload["episodeID"]) != self._episode_id:
            raise InferenceError("inference.staleState", "This observation does not continue the current actor state")
        observation_id = _uuid(payload["observationID"])
        cutoff, geometry = _uint(payload["cutoffNanos"]), _uint(payload["geometryRevision"])
        if observation_id in self._observations or (self._last_cutoff is not None and cutoff <= self._last_cutoff):
            raise InferenceError("inference.staleObservation", "Observation identity/time has already been consumed")
        config = self._checkpoint.policy.config
        if self._last_cutoff is not None and cutoff < self._last_cutoff + config.period_ms * 1_000_000:
            raise InferenceError("inference.timing", "Decision intervals cannot overlap the immutable policy cadence")
        _uint(self._sequence)
        _uint(self._draw_index); _uint(self._episode_step); _uint(self._environment_resets)
        if cutoff + (config.lead_ms + config.period_ms) * 1_000_000 > 2**64 - 1:
            raise InferenceError("inference.timing", "The action interval exceeds the monotonic clock range")
        if self._geometry_revision is not None and geometry < self._geometry_revision:
            raise InferenceError("inference.geometry", "Observation geometry revision moved backwards")
        if (type(payload["contextIDs"]) is not list or any(type(value) is not int for value in payload["contextIDs"])
            or payload["contextIDs"] != list(self._contexts)):
            raise InferenceError("inference.context", "Context choices may change only at a confirmed episode reset")
        if payload["intervalCovered"] is not True or type(payload["controlState"]) is not dict or payload["controlState"].get("valid") is not True:
            self._needs_reset = True
            raise InferenceError("inference.discontinuity", "Input state/coverage is unavailable; restore it and confirm an environment reset")
        frames, events = payload["frames"], payload["executedEvents"]
        if type(frames) is not list or not 1 <= len(frames) <= config.maximum_surfaces:
            raise InferenceError("inference.frames", "Inference requires bounded observed frame references")
        if type(events) is not list or len(events) > 4096:
            raise InferenceError("inference.events", "Executed input history exceeds the actor interval budget")
        last_event = self._last_event_sequence
        last_input_nanos = self._last_input_nanos
        for event in events:
            if type(event) is not dict:
                raise InferenceError("inference.events", "Executed inputs require structured causal events")
            sequence = _uint(event.get("sequence"))
            if last_event is not None and sequence <= last_event:
                raise InferenceError("inference.events", "Executed input history repeated or reversed an event")
            source, observed = _uint(event.get("eventNanos")), _uint(event.get("observedNanos"))
            if source > observed or observed > cutoff or (self._last_cutoff is not None and observed <= self._last_cutoff):
                raise InferenceError("inference.causality", "Executed input falls outside this causal observation interval")
            last_event = sequence
            if event.get("origin") in ("physical", "agent"):
                last_input_nanos = max(last_input_nanos or 0, observed)
        owned, acknowledgements = [], []
        try:
            for reference in frames:
                frame = self._reader.copy_frame(reference)
                owned.append((frame.pixels, frame.metadata)); acknowledgements.append(frame.acknowledgement)
            surfaces = [metadata["surface"] for _, metadata in owned]
            if any(metadata["eventNanos"] > metadata["observedNanos"] for _, metadata in owned):
                raise InferenceError("inference.causality", "A frame cannot be available before its source timestamp")
            if self._surfaces is not None and surfaces != self._surfaces and geometry == self._geometry_revision:
                raise InferenceError("inference.geometry", "Changed surface geometry requires a new aggregate revision")
            first = self._state is None
            elapsed = config.period_ms / 1000 if first else (cutoff - self._last_cutoff) / 1e9
            observation = make_observation(owned, payload["controlState"], cutoff_nanos=cutoff,
                                          elapsed_seconds=elapsed,
                                          reset=first, config=config, context_ids=self._contexts, executed_events=events,
                                          interval_covered=True, surface_preparer=prepare_metal_surface,
                                          maximum_timestamp=2**64 - 1, last_input_nanos=self._last_input_nanos)
            policy = self._checkpoint.policy
            output = self._execution(observation, self._state, self._key)
            # Materialize one complete result tree, including packet fields and
            # finite checks, before any CPU wire conversion or state commit.
            mx.eval(output)
            if not bool(output["finite"].item()):
                raise InferenceError("inference.nonfinite", "The policy produced a nonfinite action score, value or recurrent state")
            commands = decode_commands(PacketBatch(**output["packet"]), config=config, vocabulary=policy.actions.vocabulary,
                                       visual=VisualFeatures(**output["visual"]), surfaces=surfaces)
            packet = {"id": str(uuid.uuid4()), "runID": self._run_id, "sequence": self._sequence,
                      "observationID": observation_id, "geometryRevision": geometry,
                      "executeAtNanos": cutoff + config.lead_ms * 1_000_000, "durationMs": config.period_ms, "commands": commands}
            result = {"packet": packet, "releasedFrames": acknowledgements, "surfaces": surfaces,
                      "logProbability": float(output["log_probability"].item()), "value": float(output["value"].item()),
                      "conditionalEntropy": float(output["conditional_entropy"].item())}
            next_state_id = str(uuid.uuid4())
            if self._collection and not self._warming:
                before = policy.temporal.initial_state(1) if first else self._state
                state_before = []
                for value in before:
                    array = np.asarray(value)
                    if array.ndim != 2 or array.shape[0] != 1 or array.dtype != np.float32 or not np.isfinite(array).all():
                        raise InferenceError("inference.collectionState", "Collection requires finite single-actor FP32 recurrent anchors")
                    state_before.append(array[0].tolist())
                result["collectionRecord"] = {
                    "schemaVersion": 1, "checkpointID": self._checkpoint.manifest["id"],
                    "policySignature": self._checkpoint.manifest["policySignature"], "modelSignature": config.signature,
                    "episodeID": self._episode_id, "episodeStep": self._episode_step,
                    "observationID": observation_id, "cutoffNanos": cutoff, "geometryRevision": geometry,
                    "frameIDs": [_uuid(metadata["id"]) for _, metadata in owned], "contextIDs": list(self._contexts),
                    "previousStateID": self._state_id, "nextStateID": next_state_id,
                    "recurrentReset": first, "elapsedSeconds": elapsed, "stateBefore": state_before,
                    "packetFields": {name: np.asarray(value)[0].tolist() for name, value in output["packet"].items()},
                    "logProbability": result["logProbability"], "value": result["value"],
                    "sampler": {"kind": "categorical", "temperature": 1, "mixture": "none", "version": 1,
                        "rngStreamID": self._rng_stream_id, "drawIndex": self._draw_index,
                        "stateBefore": np.asarray(self._key).tolist(), "sampleKey": np.asarray(output["sample_key"]).tolist(),
                        "stateAfter": np.asarray(output["key"]).tolist()},
                    "environmentResets": self._environment_resets,
                }
            self._state, self._key, self._state_id = output["state"], output["key"], next_state_id
            self._last_cutoff, self._last_event_sequence, self._geometry_revision = cutoff, last_event, geometry
            self._last_input_nanos = last_input_nanos
            self._surfaces = deepcopy(surfaces)
            self._observations.append(observation_id)
            self._sequence += 1
            self._draw_index += 1; self._episode_step += 1
            return {**result, **self._identity()}
        except Exception as error:
            # A copied lease is safe to release even if preprocessing/policy
            # validation later fails. A failed/partial copy supplies no ack.
            self._needs_reset = True
            failure = InferenceError(getattr(error, "code", "inference.stepFailed"), str(error))
            failure.released_frames = acknowledgements
            raise failure from error

    def close(self):
        self._assert_owner()
        if self._reader is not None:
            self._reader.close()
        self._reader = self._checkpoint = self._state = self._key = None
        self._execution = None
        self._run_id = self._state_id = self._episode_id = None
        self._last_cutoff = self._last_event_sequence = self._geometry_revision = None
        self._last_input_nanos = None
        self._surfaces = None
        self._sequence = 0
        self._contexts = ()
        self._observations.clear()
        self._needs_reset = True
        self._collection = self._warming = False
        self._rng_stream_id = None
        self._draw_index = self._episode_step = self._environment_resets = 0


class InferenceManager:
    """One bounded request in flight; MLX execution never runs on stdin."""
    def __init__(self, send):
        self._send = send
        self._lock = threading.RLock()
        self._queue = queue.Queue(maxsize=1)
        self._active = False
        self._closing = False
        self._thread = threading.Thread(target=self._work, name="Astra inference owner", daemon=True)
        self._thread.start()

    @property
    def busy(self):
        with self._lock:
            return self._active

    def submit(self, request):
        if request.kind not in INFERENCE_OPERATIONS or request.run_id is None:
            raise InferenceError("inference.request", "Actor operations require a supported operation and runID")
        owned = deepcopy(request)
        with self._lock:
            if self._closing or self._active:
                raise InferenceError("inference.busy", "This actor already owns an inference operation or is closing")
            self._active = True
            self._queue.put_nowait(owned)

    def close(self, timeout=30):
        with self._lock:
            self._closing = True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _work(self):
        session = InferenceSession()
        try:
            while True:
                try:
                    request = self._queue.get(timeout=.05)
                except queue.Empty:
                    with self._lock:
                        if self._closing: return
                    continue
                try:
                    with self._lock:
                        if self._closing:
                            raise InferenceError("inference.cancelled", "Actor is closing")
                    if request.kind == "inference.close":
                        if request.payload:
                            raise InferenceError("inference.configuration", "Actor close requires an empty payload")
                        session._run(request.run_id)
                        session.close()
                        result = {"closed": True}
                    else:
                        result = getattr(session, request.kind.split(".")[1])(request.payload, run_id=request.run_id)
                    with self._lock:
                        if self._closing:
                            error = InferenceError("inference.cancelled", "Actor closed before this result could be used")
                            error.released_frames = result.get("releasedFrames", [])
                            raise error
                    kind, reply = "ack", result
                except Exception as error:
                    kind, reply = "error", {"code": getattr(error, "code", "inference.failed"),
                                             "message": str(error)[:2048] or type(error).__name__, "recoverable": True,
                                             "releasedFrames": getattr(error, "released_frames", []),
                                             "needsReset": session._needs_reset}
                finally:
                    with self._lock:
                        self._active = False
                    self._queue.task_done()
                try:
                    self._send(kind, reply, request=request)
                except Exception:
                    with self._lock:
                        self._closing = True
                    return  # Broken output retires the reader on this thread.
        finally:
            session.close()
