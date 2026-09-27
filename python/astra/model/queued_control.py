"""Hierarchical GRU encoding of already admitted, cutoff-bound controls.

This module sees typed semantic features only. It cannot consume packet IDs,
raw clock magnitudes, future teacher targets or current-decision samples.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .queue_layout import (MAXIMUM_PACKETS, CATEGORICAL_NAMES, EMBEDDING_WIDTHS, COMMAND_FEATURE_NAMES,
                           PACKET_FEATURE_NAMES, PACKET_STATUS_WIDTH, category_sizes)


@dataclass(frozen=True)
class QueuedControlBatch:
    categorical: mx.array      # B,T,64,C,7; int32, explicit null category zero
    command_features: mx.array # B,T,64,C,26; float32, semantic positions/times/progress
    command_mask: mx.array     # B,T,64,C; bool, true prefix per packet (possibly empty)
    packet_features: mx.array  # B,T,64,11; float32
    packet_status: mx.array    # B,T,64; int32, outstanding/executed/cancelled/late
    packet_mask: mx.array      # B,T,64; bool, true prefix in original packet order
    available: mx.array        # B,T; bool; absent authority never becomes known-empty

    def validate_shapes(self, batch, time, capacity):
        command = (batch, time, MAXIMUM_PACKETS, capacity)
        packet = command[:-1]
        expected = {'categorical': (*command, len(CATEGORICAL_NAMES)),
                    'command_features': (*command, len(COMMAND_FEATURE_NAMES)), 'command_mask': command,
                    'packet_features': (*packet, len(PACKET_FEATURE_NAMES)), 'packet_status': packet,
                    'packet_mask': packet, 'available': (batch, time)}
        for name, shape in expected.items():
            value = getattr(self, name)
            dtype = mx.bool_ if name.endswith('mask') or name == 'available' else mx.int32 if name in ('categorical', 'packet_status') else mx.float32
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f'Queued-control {name} must have fixed shape {shape} and dtype {dtype}')

    @classmethod
    def from_numpy(cls, *, maximum_surfaces, **arrays):
        """Validate semantic categories and prefix masks before NN allocation.

        Batching/time slicing only stack or slice already valid prefixes. The
        tensor-only reconstruction used inside mx.compile preserves this layout.
        """
        if set(arrays) != set(cls.__dataclass_fields__):
            raise ValueError('Queued-control tensors have missing or unknown fields')
        cat, commands, packets, available = (arrays[key] for key in ('categorical', 'command_mask', 'packet_mask', 'available'))
        if (cat.ndim != 5 or cat.shape[-1] != len(CATEGORICAL_NAMES) or cat.shape[2] != MAXIMUM_PACKETS or
                cat.shape[3] not in (16, 32, 64) or commands.shape != cat.shape[:-1] or packets.shape != cat.shape[:-2] or
                available.shape != cat.shape[:2]):
            raise ValueError('Queued-control masks do not match the fixed command/packet layout')
        if any(arrays[key].dtype != np.bool_ for key in ('command_mask', 'packet_mask', 'available')):
            raise ValueError('Queued-control masks must be Boolean')
        if (np.any(commands[..., 1:] & ~commands[..., :-1]) or np.any(packets[..., 1:] & ~packets[..., :-1]) or
                np.any(commands & ~packets[..., None]) or np.any(packets & ~available[..., None])):
            raise ValueError('Queued-control rows and commands must be complete prefixes under available packet evidence')
        if cat.dtype != np.int32 or arrays['packet_status'].dtype != np.int32:
            raise ValueError('Queued-control categories must be int32')
        for index, size in enumerate(category_sizes(maximum_surfaces)):
            if np.any(cat[..., index] < 0) or np.any(cat[..., index] >= size):
                raise ValueError('Queued-control semantic category is outside its vocabulary')
        if np.any(arrays['packet_status'] < 0) or np.any(arrays['packet_status'] > 3):
            raise ValueError('Invalid queued packet lifecycle category')
        for key in ('command_features', 'packet_features'):
            if arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all():
                raise ValueError('Queued-control continuous features must be finite float32')
        result = cls(**{name: mx.array(value) for name, value in arrays.items()})
        result.validate_shapes(cat.shape[0], cat.shape[1], cat.shape[3])
        return result

    @classmethod
    def empty(cls, batch, time, capacity, *, available=False):
        if type(batch) is not int or batch < 1 or type(time) is not int or time < 1 or capacity not in (16, 32, 64) or type(available) is not bool:
            raise ValueError('Invalid queued-control empty batch dimensions')
        command = (batch, time, MAXIMUM_PACKETS, capacity); packet = command[:-1]
        return cls(mx.zeros((*command, len(CATEGORICAL_NAMES)), dtype=mx.int32),
                   mx.zeros((*command, len(COMMAND_FEATURE_NAMES)), dtype=mx.float32), mx.zeros(command, dtype=mx.bool_),
                   mx.zeros((*packet, len(PACKET_FEATURE_NAMES)), dtype=mx.float32), mx.zeros(packet, dtype=mx.int32),
                   mx.zeros(packet, dtype=mx.bool_), mx.full((batch, time), available, dtype=mx.bool_))

    def slice_time(self, start, end):
        return type(self)(**{name: getattr(self, name)[:, start:end] for name in self.__dataclass_fields__})

    def as_tensors(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_tensors(cls, values):
        return cls(**values)


def _last_prefix(states, mask):
    length = mx.sum(mask, axis=-1).astype(mx.int32)
    index = mx.maximum(length - 1, 0)[:, None, None]
    selected = mx.take_along_axis(states, index, axis=1)[:, 0]
    return mx.where(length[:, None] > 0, selected, mx.zeros_like(selected))


class QueuedControlEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.command_width = config.queued_command_width
        self.packet_width = config.queued_packet_width
        self.capacity = config.packet_capacity
        self.embeddings = [nn.Embedding(size, width) for size, width in zip(category_sizes(config.maximum_surfaces), EMBEDDING_WIDTHS)]
        self.command_input = nn.Linear(sum(EMBEDDING_WIDTHS) + len(COMMAND_FEATURE_NAMES), self.command_width)
        self.command_norm = nn.LayerNorm(self.command_width)
        self.commands = nn.GRU(self.command_width, self.command_width)
        self.status_embedding = nn.Embedding(4, PACKET_STATUS_WIDTH)
        self.packet_input = nn.Linear(self.command_width + PACKET_STATUS_WIDTH + len(PACKET_FEATURE_NAMES), self.packet_width)
        self.packet_norm = nn.LayerNorm(self.packet_width)
        self.packets = nn.GRU(self.packet_width, self.packet_width)
        self.output = nn.Linear(self.packet_width, config.recurrent_width, bias=False)
        self.output.weight = mx.zeros_like(self.output.weight)

    def __call__(self, feedback: QueuedControlBatch, valid: mx.array):
        batch, time = valid.shape
        feedback.validate_shapes(batch, time, self.capacity)
        packet_mask = feedback.packet_mask & feedback.available[..., None] & valid[..., None]
        command_mask = feedback.command_mask & packet_mask[..., None]
        categories = mx.where(command_mask[..., None], feedback.categorical, 0)
        continuous = mx.where(command_mask[..., None], feedback.command_features, 0)
        encoded = [embedding(categories[..., index]) for index, embedding in enumerate(self.embeddings)]
        commands = nn.gelu(self.command_norm(self.command_input(mx.concatenate((*encoded, continuous), axis=-1))))
        count = batch * time
        states = self.commands(commands.reshape(count * MAXIMUM_PACKETS, self.capacity, self.command_width))
        command_summary = _last_prefix(states, command_mask.reshape(count * MAXIMUM_PACKETS, self.capacity))
        command_summary = command_summary.reshape(count, MAXIMUM_PACKETS, self.command_width)
        packet_features = mx.where(packet_mask[..., None], feedback.packet_features, 0).reshape(count, MAXIMUM_PACKETS, -1)
        status = self.status_embedding(mx.where(packet_mask, feedback.packet_status, 0)).reshape(count, MAXIMUM_PACKETS, -1)
        packets = nn.gelu(self.packet_norm(self.packet_input(mx.concatenate((command_summary, status, packet_features), axis=-1))))
        summary = _last_prefix(self.packets(packets), packet_mask.reshape(count, MAXIMUM_PACKETS))
        active = mx.any(packet_mask, axis=-1)[..., None]
        residual = self.output(summary).reshape(batch, time, -1)
        return mx.where(active, residual, mx.zeros_like(residual))
