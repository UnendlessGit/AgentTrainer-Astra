"""Persistent observation state. Packet teacher forcing never enters this module."""
from __future__ import annotations

from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .observation import ObservationBatch


@dataclass(frozen=True)
class TemporalOutput:
    context: mx.array  # B,T,H
    value: mx.array  # B,T
    state: tuple[mx.array, ...]  # L tensors B,H


class TemporalCore(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config.validate()
        hidden = config.recurrent_width
        self.context_embeddings = [nn.Embedding(size, config.context_width) for size in config.context_sizes]
        features = config.query_count * config.spatial_width + config.control_width + 2 + len(config.context_sizes) * config.context_width
        self.input_projection = nn.Linear(features, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.layers = [nn.GRU(hidden, hidden) for _ in range(config.recurrent_layers)]
        self.norms = [nn.LayerNorm(hidden) for _ in self.layers]
        self.value_hidden = nn.Linear(hidden, hidden // 2)
        self.value_head = nn.Linear(hidden // 2, 1)

    def initial_state(self, batch_size: int) -> tuple[mx.array, ...]:
        return tuple(mx.zeros((batch_size, self.config.recurrent_width)) for _ in self.layers)

    def __call__(self, visual_summary: mx.array, observation: ObservationBatch,
                 state: tuple[mx.array, ...] | None = None) -> TemporalOutput:
        batch, time = observation.shape
        state = self.initial_state(batch) if state is None else state
        if len(state) != len(self.layers) or any(value.shape != (batch, self.config.recurrent_width) for value in state):
            raise ValueError("Recurrent state shape does not match this model or batch")
        elapsed = mx.maximum(observation.elapsed_seconds, 0)
        timing = mx.stack((mx.minimum(elapsed, 60) / 60, mx.log1p(elapsed)), axis=-1)
        contexts = [embedding(observation.context_ids[..., index]) for index, embedding in enumerate(self.context_embeddings)]
        x = self.input_norm(self.input_projection(mx.concatenate((visual_summary, observation.controls, timing, *contexts), axis=-1)))
        current = list(state)
        outputs = []
        for step in range(time):
            active = observation.valid[:, step, None]
            # A reset on a padded sample must not destroy the carried state.
            reset = observation.reset[:, step, None] & active
            value = x[:, step]
            for index, (layer, norm) in enumerate(zip(self.layers, self.norms)):
                previous = mx.where(reset, mx.zeros_like(current[index]), current[index])
                candidate = layer(value[:, None, :], previous)[:, 0]
                current[index] = mx.where(active, candidate, current[index])
                value = norm(current[index])
            outputs.append(mx.where(active, value, mx.zeros_like(value)))
        context = mx.stack(outputs, axis=1)
        value = self.value_head(nn.gelu(self.value_hidden(context)))[..., 0]
        value = mx.where(observation.valid, value, 0)
        return TemporalOutput(context, value, tuple(current))
