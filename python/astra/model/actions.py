"""Timed command packet distribution shared by imitation, PPO, and inference.

Every active factor is scored, including END. Masks depend on capability,
availability and packet grammar; they never depend on predicted held state.
The packet GRU is reset for each observation and cannot modify persistent state.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelConfig
from .vision import VisualFeatures


class Operation(IntEnum):
    END = 0
    KEY_DOWN = 1
    KEY_UP = 2
    KEY_REPEAT = 3
    BUTTON_DOWN = 4
    BUTTON_UP = 5
    ABSOLUTE = 6
    RELATIVE = 7
    SCROLL = 8


@dataclass(frozen=True)
class ActionVocabulary:
    key_codes: tuple[int, ...] = ()
    mouse_buttons: tuple[int, ...] = ()
    absolute_pointer: bool = False
    relative_pointer: bool = False
    scroll: bool = False
    scroll_units_per_point: int = 8

    def validate(self) -> ActionVocabulary:
        for values, maximum in ((self.key_codes, 127), (self.mouse_buttons, 31)):
            if not isinstance(values, tuple) or any(type(value) is not int or not 0 <= value <= maximum for value in values):
                raise ValueError("Invalid action vocabulary")
            if tuple(sorted(set(values))) != values:
                raise ValueError("Action vocabularies must be sorted and unique")
        if any(type(value) is not bool for value in (self.absolute_pointer, self.relative_pointer, self.scroll)):
            raise ValueError("Capabilities must be Boolean")
        if type(self.scroll_units_per_point) is not int or self.scroll_units_per_point not in (1, 2, 4, 8, 16):
            raise ValueError("Unsupported scroll quantization")
        return self

    def to_dict(self) -> dict:
        return {"keyCodes": list(self.key_codes), "mouseButtons": list(self.mouse_buttons),
                "absolutePointer": self.absolute_pointer, "relativePointer": self.relative_pointer,
                "scroll": self.scroll, "scrollUnitsPerPoint": self.scroll_units_per_point}

    @classmethod
    def from_dict(cls, value: dict) -> ActionVocabulary:
        keys = {"keyCodes", "mouseButtons", "absolutePointer", "relativePointer", "scroll", "scrollUnitsPerPoint"}
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError("Unsupported action vocabulary fields")
        return cls(tuple(value["keyCodes"]), tuple(value["mouseButtons"]), value["absolutePointer"],
                   value["relativePointer"], value["scroll"], value["scrollUnitsPerPoint"]).validate()


@dataclass(frozen=True)
class PacketBatch:
    operation: mx.array  # B,K+1; K commands plus terminal END
    offset: mx.array  # milliseconds in [0,T]; nondecreasing while active
    key: mx.array
    button: mx.array
    surface: mx.array
    cell: mx.array
    within_x: mx.array
    within_y: mx.array
    dx: mx.array  # signed integer; raw relative counts or quantized scroll
    dy: mx.array

    @classmethod
    def zeros(cls, batch: int, length: int) -> PacketBatch:
        return cls(**{name: mx.zeros((batch, length), dtype=mx.int32) for name in cls.__dataclass_fields__})

    def validate(self, config: ModelConfig, vocabulary: ActionVocabulary, visual: VisualFeatures) -> None:
        """Host-side dataset/IPC boundary validation, never inside a gradient."""
        arrays = {name: np.asarray(getattr(self, name)) for name in self.__dataclass_fields__}
        shape = arrays["operation"].shape
        if len(shape) != 2 or shape[1] != config.packet_capacity + 1 or shape[0] != visual.cells.shape[0]:
            raise ValueError("Packet dimensions do not match the policy configuration")
        if any(value.shape != shape or not np.issubdtype(value.dtype, np.integer) for value in arrays.values()):
            raise ValueError("Packet fields must be equally shaped integer tensors")
        cell_valid = np.asarray(visual.cell_valid)
        surface_valid = np.asarray(visual.surface_valid)
        if cell_valid.ndim != 3 or surface_valid.shape != cell_valid.shape[:2]:
            raise ValueError("Packet scoring needs flattened B,S,N visual fields")
        for batch in range(shape[0]):
            ended = False
            previous = 0
            motion = None
            for slot in range(shape[1]):
                operation = int(arrays["operation"][batch, slot])
                if ended or slot == config.packet_capacity:
                    if operation != Operation.END:
                        raise ValueError("Commands cannot occur after END or beyond packet capacity")
                if operation == Operation.END:
                    ended = True
                    continue
                if not 1 <= operation <= 8:
                    raise ValueError("Unknown command operation")
                offset = int(arrays["offset"][batch, slot])
                limit = config.period_ms if operation in (Operation.ABSOLUTE, Operation.RELATIVE) else config.period_ms - 1
                if not previous <= offset <= limit:
                    raise ValueError("Packet timing is outside its causal ordered interval")
                previous = offset
                if operation in (Operation.KEY_DOWN, Operation.KEY_UP, Operation.KEY_REPEAT):
                    if arrays["key"][batch, slot] not in vocabulary.key_codes:
                        raise ValueError("Key command exceeds the checkpoint's vocabulary")
                if operation in (Operation.BUTTON_DOWN, Operation.BUTTON_UP):
                    if arrays["button"][batch, slot] not in vocabulary.mouse_buttons:
                        raise ValueError("Button command exceeds the checkpoint's vocabulary")
                if operation == Operation.SCROLL and not vocabulary.scroll:
                    raise ValueError("Scroll is unavailable")
                if operation in (Operation.ABSOLUTE, Operation.RELATIVE):
                    if motion is not None and motion != operation:
                        raise ValueError("Absolute and relative trajectories cannot share a packet")
                    motion = operation
                    if not (vocabulary.absolute_pointer if operation == Operation.ABSOLUTE else vocabulary.relative_pointer):
                        raise ValueError("Pointer operation is unavailable")
                if operation == Operation.ABSOLUTE:
                    surface, cell = arrays["surface"][batch, slot], arrays["cell"][batch, slot]
                    if not 0 <= surface < cell_valid.shape[1] or not 0 <= cell < cell_valid.shape[2] or not surface_valid[batch, surface] or not cell_valid[batch, surface, cell]:
                        raise ValueError("Pointing target is outside an observed surface")
                    if any(not 0 <= arrays[field][batch, slot] < config.coordinate_bins for field in ("within_x", "within_y")):
                        raise ValueError("Within-cell coordinate is outside its vocabulary")
                if operation in (Operation.RELATIVE, Operation.SCROLL):
                    if any(abs(int(arrays[field][batch, slot])) > config.delta_magnitude for field in ("dx", "dy")):
                        raise ValueError("Motion component exceeds its representable range")


@dataclass(frozen=True)
class PacketDistributionResult:
    packets: PacketBatch
    log_probability: mx.array  # B; exact sum, not mean over components
    conditional_entropy: mx.array  # B; sum of categorical entropies at visited prefixes
    factor_log_probabilities: dict[str, mx.array]  # each B,K+1, already masked


def flatten_visual(visual: VisualFeatures) -> VisualFeatures:
    """B,T,... -> B*T,... without changing observation/action alignment."""
    def flatten(value):
        return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
    return VisualFeatures(**{name: flatten(getattr(visual, name)) for name in visual.__dataclass_fields__})


def _gather(values: mx.array, indices: mx.array) -> mx.array:
    return mx.take_along_axis(values, indices[..., None], axis=-1)[..., 0]


def _categorical(logits: mx.array, mask: mx.array, *, choice: mx.array | None,
                 active: mx.array, key: mx.array | None, greedy: bool):
    logits = logits.astype(mx.float32)
    mask = mx.broadcast_to(mask, logits.shape)
    # Inactive conditional heads are assigned a harmless point mass. This keeps
    # all paths finite even when a checkpoint exposes no keys or no pointer.
    fallback = mx.arange(logits.shape[-1]) == 0
    permitted = mx.where(active[..., None], mask, fallback)
    # A legal active head always has support. An invalid caller gets -inf below.
    has_support = mx.any(permitted, axis=-1)
    safe_mask = permitted | ((~has_support)[..., None] & fallback)
    masked = mx.where(safe_mask, logits, -math.inf)
    log_probs = masked - mx.logsumexp(masked, axis=-1, keepdims=True)
    safe_log_probs = mx.where(safe_mask, log_probs, 0)
    entropy = -mx.sum(mx.where(safe_mask, mx.exp(log_probs), 0) * safe_log_probs, axis=-1)
    if choice is None:
        selected = mx.argmax(masked, axis=-1) if greedy else mx.random.categorical(masked, key=key)
    else:
        selected = choice
    in_range = (selected >= 0) & (selected < logits.shape[-1])
    safe_selected = mx.clip(selected, 0, logits.shape[-1] - 1).astype(mx.int32)
    legal = in_range & _gather(permitted, safe_selected) & has_support
    score = mx.where(legal, _gather(safe_log_probs, safe_selected), -math.inf)
    return mx.where(active, selected, 0).astype(mx.int32), mx.where(active, score, 0), mx.where(active, entropy, 0)


class PacketDecoder(nn.Module):
    def __init__(self, config: ModelConfig, vocabulary: ActionVocabulary):
        super().__init__()
        self.config = config.validate()
        self.vocabulary = vocabulary.validate()
        width = config.decoder_width
        self.initial = nn.Linear(config.recurrent_width, width)
        self.recurrent = nn.GRU(width, width)
        self.hidden_norm = nn.LayerNorm(width)
        self.operation_embedding = nn.Embedding(9, width)
        self.time_embedding = nn.Embedding(config.period_ms + 1, width)
        self.key_embedding = nn.Embedding(128, width)
        self.button_embedding = nn.Embedding(32, width)
        self.surface_embedding = nn.Embedding(config.maximum_surfaces, width)
        self.x_embedding = nn.Embedding(config.coordinate_bins, width)
        self.y_embedding = nn.Embedding(config.coordinate_bins, width)
        self.delta_coarse_count = (2 * config.delta_magnitude + config.delta_radix) // config.delta_radix
        self.delta_embeddings = [nn.Embedding(size, width) for size in
                                 (self.delta_coarse_count, config.delta_radix, self.delta_coarse_count, config.delta_radix)]
        self.operation_head = nn.Linear(width, 9)
        self.time_head = nn.Linear(width, config.period_ms + 1)
        self.key_head = nn.Linear(width, 128)
        self.button_head = nn.Linear(width, 32)
        self.surface_query = nn.Linear(width, config.spatial_width)
        self.cell_query = nn.Linear(width, config.spatial_width)
        self.cell_embedding = nn.Linear(config.spatial_width, width)
        self.x_head = nn.Linear(width, config.coordinate_bins)
        self.y_head = nn.Linear(width, config.coordinate_bins)
        self.delta_heads = [nn.Linear(width, size) for size in
                            (self.delta_coarse_count, config.delta_radix, self.delta_coarse_count, config.delta_radix)]
        self.argument_norm = nn.LayerNorm(width)

    def _cell_features(self, visual: VisualFeatures, surface: mx.array, cell: mx.array) -> mx.array:
        batch = mx.arange(surface.shape[0])[:, None]
        return visual.cells[batch, mx.clip(surface, 0, visual.cells.shape[1] - 1), mx.clip(cell, 0, visual.cells.shape[2] - 1)]

    def _summary(self, packet: PacketBatch, visual: VisualFeatures) -> mx.array:
        operation = packet.operation
        active = operation != Operation.END
        keyboard = (operation >= Operation.KEY_DOWN) & (operation <= Operation.KEY_REPEAT)
        button = (operation == Operation.BUTTON_DOWN) | (operation == Operation.BUTTON_UP)
        absolute = operation == Operation.ABSOLUTE
        delta = (operation == Operation.RELATIVE) | (operation == Operation.SCROLL)
        safe = lambda value, maximum: mx.clip(value, 0, maximum - 1)
        result = self.operation_embedding(safe(operation, 9)) + self.time_embedding(safe(packet.offset, self.config.period_ms + 1))
        result = result + mx.where(keyboard[..., None], self.key_embedding(safe(packet.key, 128)), 0)
        result = result + mx.where(button[..., None], self.button_embedding(safe(packet.button, 32)), 0)
        point = self.surface_embedding(safe(packet.surface, self.config.maximum_surfaces))
        point = point + self.cell_embedding(self._cell_features(visual, packet.surface, packet.cell))
        point = point + self.x_embedding(safe(packet.within_x, self.config.coordinate_bins)) + self.y_embedding(safe(packet.within_y, self.config.coordinate_bins))
        result = result + mx.where(absolute[..., None], point, 0)
        dx = mx.clip(packet.dx + self.config.delta_magnitude, 0, 2 * self.config.delta_magnitude)
        dy = mx.clip(packet.dy + self.config.delta_magnitude, 0, 2 * self.config.delta_magnitude)
        components = (dx // self.config.delta_radix, dx % self.config.delta_radix, dy // self.config.delta_radix, dy % self.config.delta_radix)
        delta_token = sum(embedding(component) for embedding, component in zip(self.delta_embeddings, components))
        result = result + mx.where(delta[..., None], delta_token, 0)
        return mx.where(active[..., None], self.argument_norm(result), 0)

    def _factors(self, hidden: mx.array, visual: VisualFeatures, *, previous_time: mx.array,
                 has_absolute: mx.array, has_relative: mx.array, alive: mx.array, forced_end: mx.array,
                 choices: PacketBatch | None, keys, greedy: bool):
        shape = hidden.shape[:2]
        fields = {}
        scores, entropies = {}, []

        def factor(name, logits, mask, active, choice=None):
            value, score, entropy = _categorical(logits, mask, choice=choice, active=active,
                                                key=next(keys) if keys is not None else None, greedy=greedy)
            scores[name] = score; entropies.append(entropy)
            return value

        available_surfaces = visual.surface_valid & mx.any(visual.cell_valid, axis=-1)
        absolute_available = mx.any(available_surfaces, axis=-1)[:, None] & self.vocabulary.absolute_pointer & ~has_relative
        normal_time = previous_time < self.config.period_ms
        operation_mask = mx.stack((mx.ones(shape, dtype=mx.bool_),
                                   normal_time & bool(self.vocabulary.key_codes), normal_time & bool(self.vocabulary.key_codes),
                                   normal_time & bool(self.vocabulary.key_codes), normal_time & bool(self.vocabulary.mouse_buttons),
                                   normal_time & bool(self.vocabulary.mouse_buttons), mx.broadcast_to(absolute_available, shape),
                                   ~has_absolute & self.vocabulary.relative_pointer, normal_time & self.vocabulary.scroll), axis=-1)
        operation_mask = mx.where((~alive | forced_end)[..., None], mx.arange(9) == Operation.END, operation_mask)
        # Forced END still passes through the common categorical path, whose
        # singleton support gives probability one and exactly zero gradient.
        operation = factor("operation", self.operation_head(hidden), operation_mask, mx.ones(shape, dtype=mx.bool_),
                           choices.operation if choices is not None else None)
        fields["operation"] = operation
        active = alive & ~forced_end & (operation != Operation.END)
        keyboard = active & (operation >= Operation.KEY_DOWN) & (operation <= Operation.KEY_REPEAT)
        buttons = active & ((operation == Operation.BUTTON_DOWN) | (operation == Operation.BUTTON_UP))
        absolute = active & (operation == Operation.ABSOLUTE)
        delta = active & ((operation == Operation.RELATIVE) | (operation == Operation.SCROLL))
        context = hidden + self.operation_embedding(mx.clip(operation, 0, 8))
        motion = (operation == Operation.ABSOLUTE) | (operation == Operation.RELATIVE)
        times = mx.arange(self.config.period_ms + 1)
        time_mask = (times >= previous_time[..., None]) & (times <= (self.config.period_ms - 1 + motion.astype(mx.int32))[..., None])
        offset = factor("time", self.time_head(nn.gelu(context)), time_mask, active, choices.offset if choices is not None else None)
        fields["offset"] = offset
        context = context + self.time_embedding(mx.clip(offset, 0, self.config.period_ms))
        key_mask = mx.array([index in self.vocabulary.key_codes for index in range(128)])
        fields["key"] = factor("key", self.key_head(nn.gelu(context)), key_mask, keyboard, choices.key if choices is not None else None)
        button_mask = mx.array([index in self.vocabulary.mouse_buttons for index in range(32)])
        fields["button"] = factor("button", self.button_head(nn.gelu(context)), button_mask, buttons, choices.button if choices is not None else None)

        surface_summary = mx.sum(visual.cells * visual.cell_valid[..., None], axis=2) / mx.maximum(mx.sum(visual.cell_valid, axis=2)[..., None], 1)
        surface_logits = mx.einsum("bkd,bsd->bks", self.surface_query(nn.gelu(context)), surface_summary) / math.sqrt(self.config.spatial_width)
        surface = factor("surface", surface_logits, available_surfaces[:, None, :], absolute, choices.surface if choices is not None else None)
        fields["surface"] = surface
        point_context = context + self.surface_embedding(mx.clip(surface, 0, self.config.maximum_surfaces - 1))
        # Form scores directly. Materializing B,K,N,D gathered cell features for
        # teacher forcing would multiply dense-vision memory by packet length.
        all_cell_logits = mx.einsum("bkd,bsnd->bksn", self.cell_query(nn.gelu(point_context)), visual.cells) / math.sqrt(self.config.spatial_width)
        batch = mx.arange(shape[0])[:, None]
        slots = mx.arange(shape[1])[None, :]
        safe_surface = mx.clip(surface, 0, visual.cells.shape[1] - 1)
        cell_logits = all_cell_logits[batch, slots, safe_surface]
        cell_mask = visual.cell_valid[batch, safe_surface]
        cell = factor("cell", cell_logits, cell_mask, absolute, choices.cell if choices is not None else None)
        fields["cell"] = cell
        point_context = point_context + self.cell_embedding(self._cell_features(visual, surface, cell))
        within_x = factor("within_x", self.x_head(nn.gelu(point_context)), mx.array(True), absolute, choices.within_x if choices is not None else None)
        fields["within_x"] = within_x
        point_context = point_context + self.x_embedding(mx.clip(within_x, 0, self.config.coordinate_bins - 1))
        fields["within_y"] = factor("within_y", self.y_head(nn.gelu(point_context)), mx.array(True), absolute,
                                     choices.within_y if choices is not None else None)

        delta_context = context
        decoded = []
        for index, (head, embedding) in enumerate(zip(self.delta_heads, self.delta_embeddings)):
            axis = "dx" if index < 2 else "dy"
            coarse = index % 2 == 0
            choice = None
            if choices is not None:
                shifted = getattr(choices, axis) + self.config.delta_magnitude
                choice = shifted // self.config.delta_radix if coarse else shifted % self.config.delta_radix
            mask = mx.array(True) if coarse else (decoded[-1][..., None] * self.config.delta_radix + mx.arange(self.config.delta_radix) <= 2 * self.config.delta_magnitude)
            component = factor(axis + ("_coarse" if coarse else "_fine"), head(nn.gelu(delta_context)), mask, delta, choice)
            decoded.append(component)
            delta_context = delta_context + embedding(mx.clip(component, 0, head.weight.shape[0] - 1))
        fields["dx"] = mx.where(delta, decoded[0] * self.config.delta_radix + decoded[1] - self.config.delta_magnitude, 0)
        fields["dy"] = mx.where(delta, decoded[2] * self.config.delta_radix + decoded[3] - self.config.delta_magnitude, 0)
        return PacketBatch(**fields), scores, sum(entropies)

    def log_prob(self, context: mx.array, visual: VisualFeatures, packets: PacketBatch) -> PacketDistributionResult:
        if context.ndim != 2 or packets.operation.shape != (context.shape[0], self.config.packet_capacity + 1):
            raise ValueError("Action scoring expects B,H contexts and fixed B,K+1 packets")
        summary = self._summary(packets, visual)
        inputs = mx.concatenate((mx.zeros_like(summary[:, :1]), summary[:, :-1]), axis=1)
        hidden = self.hidden_norm(self.recurrent(inputs, mx.tanh(self.initial(context))))
        operation = packets.operation
        before = lambda condition: mx.cumsum(condition.astype(mx.int32), axis=1) - condition.astype(mx.int32)
        previous_time = mx.concatenate((mx.zeros_like(packets.offset[:, :1]), packets.offset[:, :-1]), axis=1)
        canonical, scores, entropy = self._factors(hidden, visual, previous_time=previous_time,
                                                  has_absolute=before(operation == Operation.ABSOLUTE) > 0,
                                                  has_relative=before(operation == Operation.RELATIVE) > 0,
                                                  alive=before(operation == Operation.END) == 0,
                                                  forced_end=mx.broadcast_to(mx.arange(operation.shape[1]) == self.config.packet_capacity, operation.shape),
                                                  choices=packets, keys=None, greedy=False)
        return PacketDistributionResult(canonical, mx.sum(sum(scores.values()), axis=1), mx.sum(entropy, axis=1), scores)

    def sample(self, context: mx.array, visual: VisualFeatures, *, key: mx.array, greedy: bool = False) -> PacketDistributionResult:
        if context.ndim != 2 or visual.cells.ndim != 4 or visual.cells.shape[0] != context.shape[0]:
            raise ValueError("Action sampling expects B,H contexts and B,S,N,D visual fields")
        batch = context.shape[0]
        length = self.config.packet_capacity + 1
        keys = iter(mx.random.split(key, num=13 * length))
        state = mx.tanh(self.initial(context))
        previous = mx.zeros((batch, 1, self.config.decoder_width))
        previous_time = mx.zeros((batch, 1), dtype=mx.int32)
        absolute = mx.zeros((batch, 1), dtype=mx.bool_)
        relative = mx.zeros((batch, 1), dtype=mx.bool_)
        alive = mx.ones((batch, 1), dtype=mx.bool_)
        packets, all_scores, entropies = [], [], []
        for slot in range(length):
            state = self.recurrent(previous, state)[:, 0]
            packet, scores, entropy = self._factors(self.hidden_norm(state)[:, None], visual, previous_time=previous_time,
                                                   has_absolute=absolute, has_relative=relative, alive=alive,
                                                   forced_end=mx.full((batch, 1), slot == self.config.packet_capacity),
                                                   choices=None, keys=keys, greedy=greedy)
            packets.append(packet); all_scores.append(scores); entropies.append(entropy)
            alive = alive & (packet.operation != Operation.END)
            previous_time = packet.offset
            absolute = absolute | (packet.operation == Operation.ABSOLUTE)
            relative = relative | (packet.operation == Operation.RELATIVE)
            previous = self._summary(packet, visual)
        combined = PacketBatch(**{name: mx.concatenate([getattr(packet, name) for packet in packets], axis=1)
                                   for name in PacketBatch.__dataclass_fields__})
        scores = {name: mx.concatenate([step[name] for step in all_scores], axis=1) for name in all_scores[0]}
        return PacketDistributionResult(combined, mx.sum(sum(scores.values()), axis=1),
                                         mx.sum(mx.concatenate(entropies, axis=1), axis=1), scores)
