"""Experimental causal observation transformer; never a production checkpoint type.

Each layer retains at most ``history`` valid observation keys/values. Cache tensors
have fixed shapes, including for padded inputs. Explicit reset/valid/position
masks provide the same semantics for full sequences and incremental calls. An
attached cache carries gradients; callers must explicitly detach at TBPTT cuts.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn

from astra.model.config import ModelConfig
from astra.model.observation import ObservationBatch
from astra.model.temporal import TemporalOutput


@dataclass(frozen=True)
class TransformerSpecification:
    width: int = 384
    layers: int = 4
    heads: int = 6
    history: int = 512
    expansion: int = 4
    rope_base: float = 10000.0

    def validate(self):
        integers = (self.width, self.layers, self.heads, self.history, self.expansion)
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError('Transformer dimensions must be positive integers')
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError('Rotary attention needs an even integral head width')
        if self.history > 512 or self.layers > 16 or self.width > 2048 or self.expansion > 8:
            raise ValueError('Experimental transformer allocation is bounded')
        if not math.isfinite(self.rope_base) or self.rope_base <= 1:
            raise ValueError('Invalid rotary position base')
        return self


def _rotary(value, positions, base):
    """Rotate adjacent coordinate pairs; positions count only valid observations."""
    half = value.shape[-1] // 2
    frequencies = mx.exp(-math.log(base) * mx.arange(half, dtype=mx.float32) / half)
    angle = positions[:, None, :, None].astype(mx.float32) * frequencies
    even, odd = value[..., 0::2], value[..., 1::2]
    cosine, sine = mx.cos(angle), mx.sin(angle)
    return mx.stack((even * cosine - odd * sine, even * sine + odd * cosine), axis=-1).reshape(value.shape)


class ObservationBlock(nn.Module):
    def __init__(self, specification):
        super().__init__()
        self.specification = specification
        width = specification.width
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_output = nn.Linear(width, width)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp_in = nn.Linear(width, specification.expansion * width)
        self.mlp_out = nn.Linear(specification.expansion * width, width)

    def __call__(self, x, old_key, old_value, positions, allowed):
        batch, time, width = x.shape
        spec = self.specification
        qkv = self.qkv(self.attention_norm(x)).reshape(batch, time, 3, spec.heads, width // spec.heads)
        query, key, value = (qkv[:, :, index].transpose(0, 2, 1, 3) for index in range(3))
        query, key = _rotary(query, positions, spec.rope_base), _rotary(key, positions, spec.rope_base)
        keys = mx.concatenate((old_key, key), axis=2)
        values = mx.concatenate((old_value, value), axis=2)
        scores = (query @ keys.transpose(0, 1, 3, 2)) * ((width // spec.heads) ** -0.5)
        # Mask again after softmax: an all-padding row has exactly zero attention
        # and finite derivatives, without assigning mass to future/invalid keys.
        weights = mx.softmax(mx.where(allowed[:, None], scores, -1e9), axis=-1)
        weights = mx.where(allowed[:, None], weights, 0)
        attended = (weights @ values).transpose(0, 2, 1, 3).reshape(batch, time, width)
        x = x + self.attention_output(attended)
        x = x + self.mlp_out(nn.gelu(self.mlp_in(self.mlp_norm(x))))
        return x, keys, values


class CausalObservationTransformer(nn.Module):
    """Shares the GRU's causal inputs and projected decoder/value interface.

    State is (next position B, cached positions B,C, cached valid B,C,
    K0,V0,...). Cache rows are chronological and right aligned. A layer's cached
    representation may summarize older inputs; 512 is a direct-key/storage bound,
    not a claim that all information older than 512 observations disappears.
    """
    def __init__(self, config: ModelConfig, specification=None, *, value_hidden=None, value_head=None):
        super().__init__()
        self.config = config.validate()
        self.specification = (specification or TransformerSpecification()).validate()
        width = self.specification.width
        self.context_embeddings = [nn.Embedding(size, config.context_width) for size in config.context_sizes]
        inputs = config.query_count * config.spatial_width + config.control_width + 2 + len(config.context_sizes) * config.context_width
        self.input_projection = nn.Linear(inputs, width)
        self.input_norm = nn.LayerNorm(width)
        self.layers = [ObservationBlock(self.specification) for _ in range(self.specification.layers)]
        self.final_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, config.recurrent_width)
        self.output_norm = nn.LayerNorm(config.recurrent_width)
        self.value_hidden = value_hidden if value_hidden is not None else nn.Linear(config.recurrent_width, config.recurrent_width // 2)
        self.value_head = value_head if value_head is not None else nn.Linear(config.recurrent_width // 2, 1)

    def initial_state(self, batch_size):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('Transformer batch size must be positive')
        spec = self.specification
        state = [mx.zeros((batch_size,), dtype=mx.int32),
                 mx.zeros((batch_size, spec.history), dtype=mx.int32),
                 mx.zeros((batch_size, spec.history), dtype=mx.bool_)]
        state.extend(mx.zeros((batch_size, spec.heads, spec.history, spec.width // spec.heads), dtype=mx.float32)
                     for _ in range(2 * spec.layers))
        return tuple(state)

    def _validate(self, summary, observation, state):
        batch, time = observation.shape
        spec = self.specification
        if not batch or not time or summary.shape != (batch, time, self.config.query_count * self.config.spatial_width):
            raise ValueError('Transformer summaries do not match the observations')
        if observation.controls.shape != (batch, time, self.config.control_width):
            raise ValueError('Transformer control width differs from the model')
        if any(getattr(observation, name).shape != (batch, time) for name in ('elapsed_seconds', 'valid', 'reset')):
            raise ValueError('Transformer temporal masks differ from the observations')
        if observation.context_ids.shape != (batch, time, len(self.context_embeddings)):
            raise ValueError('Transformer context vocabularies differ from the model')
        if summary.dtype != mx.float32 or observation.controls.dtype != mx.float32:
            raise ValueError('This experiment requires full FP32 observation features')
        if observation.valid.dtype != mx.bool_ or observation.reset.dtype != mx.bool_:
            raise ValueError('Transformer masks must be Boolean')
        expected = [(batch,), (batch, spec.history), (batch, spec.history)]
        expected += [(batch, spec.heads, spec.history, spec.width // spec.heads)] * (2 * spec.layers)
        types = [mx.int32, mx.int32, mx.bool_] + [mx.float32] * (2 * spec.layers)
        if len(state) != len(expected) or any(value.shape != shape or value.dtype != dtype for value, shape, dtype in zip(state, expected, types)):
            raise ValueError('Transformer cache shape/type differs from this model or batch')

    def __call__(self, summary, observation: ObservationBatch, state=None):
        batch, time = observation.shape
        state = self.initial_state(batch) if state is None else state
        self._validate(summary, observation, state)
        spec = self.specification
        active = observation.valid
        reset = observation.reset & active
        # The call-local segment distinguishes old cache entries from any reset
        # within this call. Reset on padding neither increments position nor clears.
        segments = mx.cumsum(reset.astype(mx.int32), axis=1)
        ordinal = state[0][:, None] + mx.cumsum(active.astype(mx.int32), axis=1) - 1
        reset_base = mx.cummax(mx.where(reset, ordinal, 0), axis=1)
        positions = ordinal - reset_base
        next_position = state[0] + mx.sum(active.astype(mx.int32), axis=1) - reset_base[:, -1]
        key_positions = mx.concatenate((state[1], positions), axis=1)
        key_valid = mx.concatenate((state[2], active), axis=1)
        key_segments = mx.concatenate((mx.zeros_like(state[1]), segments), axis=1)
        age = positions[:, :, None] - key_positions[:, None, :]
        allowed = (active[:, :, None] & key_valid[:, None, :]
                   & (segments[:, :, None] == key_segments[:, None, :])
                   & (age >= 0) & (age < spec.history))
        # Select the latest valid keys in the final episode, with fixed-size
        # right-aligned zero padding. Ordinals make sorting independent of ties.
        keep = key_valid & (key_segments == segments[:, -1:]) & (key_positions >= next_position[:, None] - spec.history)
        ranks = mx.where(keep, mx.arange(spec.history + time)[None] + 1, 0)
        selected = mx.argsort(ranks, axis=1)[:, -spec.history:]
        retained_valid = mx.take_along_axis(keep, selected, axis=1)
        retained_positions = mx.where(retained_valid, mx.take_along_axis(key_positions, selected, axis=1), 0)
        elapsed = mx.maximum(observation.elapsed_seconds, 0)
        timing = mx.stack((mx.minimum(elapsed, 60) / 60, mx.log1p(elapsed)), axis=-1)
        contexts = [embedding(observation.context_ids[..., index]) for index, embedding in enumerate(self.context_embeddings)]
        x = self.input_norm(self.input_projection(mx.concatenate((summary, observation.controls, timing, *contexts), axis=-1)))
        retained = [next_position, retained_positions, retained_valid]
        for index, layer in enumerate(self.layers):
            x, keys, values = layer(x, state[3 + 2 * index], state[4 + 2 * index], positions, allowed)
            for value in (keys, values):
                gathered = mx.take_along_axis(value, selected[:, None, :, None], axis=2)
                retained.append(mx.where(retained_valid[:, None, :, None], gathered, 0))
        context = self.output_norm(self.output_projection(self.final_norm(x)))
        context = mx.where(active[..., None], context, 0)
        value = self.value_head(nn.gelu(self.value_hidden(context)))[..., 0]
        return TemporalOutput(context, mx.where(active, value, 0), tuple(retained))


class ExperimentalTransformerHead(nn.Module):
    """Own experimental temporal weights; alias the exact common packet/value heads."""
    def __init__(self, policy, specification=None):
        super().__init__()
        self.config = policy.config
        self.temporal = CausalObservationTransformer(policy.config, specification,
            value_hidden=policy.temporal.value_hidden, value_head=policy.temporal.value_head)
        self.actions = policy.actions
