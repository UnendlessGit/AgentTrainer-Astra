"""Global semantics, fine spatial detail, and a causal cursor crop.

The temporal core receives learned position-aware query summaries. The action
decoder retains the dense detail grid for pointing; pooling does not erase it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .convnext import ConvNeXtEncoder
from .observation import ObservationBatch, SurfaceBatch


def resize_features(value: mx.array, height: int, width: int) -> mx.array:
    if value.shape[1:3] == (height, width):
        return value
    result = nn.Upsample((height / value.shape[1], width / value.shape[2]), mode="linear")(value)
    if result.shape[1:3] != (height, width):
        raise ValueError("Feature interpolation produced an unexpected spatial shape")
    return result


def align_features(value: mx.array, height: int, width: int, source_rect: mx.array, destination_rect: mx.array) -> mx.array:
    """Align feature-cell centers through explicit source-surface transforms.

    Independent stride padding must not shift small pointing targets. Gather
    interpolation remains differentiable with respect to the feature tensors.
    """
    yy, xx = mx.meshgrid((mx.arange(height, dtype=mx.float32) + 0.5) / height,
                         (mx.arange(width, dtype=mx.float32) + 0.5) / width, indexing="ij")
    destination = mx.stack((xx, yy), axis=-1)[None]
    surface = (destination - destination_rect[:, None, None, :2]) / mx.maximum(destination_rect[:, None, None, 2:], 1e-8)
    source = source_rect[:, None, None, :2] + surface * source_rect[:, None, None, 2:]
    position = source * mx.array([value.shape[2], value.shape[1]]) - 0.5
    lower = mx.floor(position)
    fraction = position - lower
    lower = mx.stop_gradient(lower.astype(mx.int32))
    x0, y0 = mx.clip(lower[..., 0], 0, value.shape[2] - 1), mx.clip(lower[..., 1], 0, value.shape[1] - 1)
    x1, y1 = mx.clip(lower[..., 0] + 1, 0, value.shape[2] - 1), mx.clip(lower[..., 1] + 1, 0, value.shape[1] - 1)
    batch = mx.arange(value.shape[0])[:, None, None]
    wx, wy = fraction[..., 0, None], fraction[..., 1, None]
    top = value[batch, y0, x0] * (1 - wx) + value[batch, y0, x1] * wx
    bottom = value[batch, y1, x0] * (1 - wx) + value[batch, y1, x1] * wx
    return top * (1 - wy) + bottom * wy


def position_features(rect: mx.array, height: int, width: int) -> tuple[mx.array, mx.array, mx.array]:
    """Surface-space cell bounds, Fourier positions, and valid-content mask."""
    batch = rect.shape[0]
    yy, xx = mx.meshgrid(mx.arange(height, dtype=mx.float32), mx.arange(width, dtype=mx.float32), indexing="ij")
    edges = mx.stack((xx / width, yy / height, (xx + 1) / width, (yy + 1) / height), axis=-1)
    origin = rect[:, None, None, :2]
    extent = mx.maximum(rect[:, None, None, 2:], 1e-8)
    low, high = (edges[None, ..., :2] - origin) / extent, (edges[None, ..., 2:] - origin) / extent
    valid = (high[..., 0] > 0) & (high[..., 1] > 0) & (low[..., 0] < 1) & (low[..., 1] < 1)
    bounds = mx.concatenate((mx.clip(low, 0, 1), mx.clip(high, 0, 1)), axis=-1).reshape(batch, height * width, 4)
    center = (bounds[..., :2] + bounds[..., 2:]) * 0.5
    frequency = mx.array([1.0, 2.0, 4.0, 8.0])
    phase = center[..., None] * frequency * math.pi * 2
    encoding = mx.concatenate((center, mx.sin(phase).reshape(batch, -1, 8), mx.cos(phase).reshape(batch, -1, 8)), axis=-1)
    return bounds, encoding, valid.reshape(batch, -1)


class DetailEncoder(nn.Module):
    def __init__(self, channels: tuple[int, ...]):
        super().__init__()
        self.convolutions = [nn.Conv2d(3 if i == 0 else channels[i - 1], width, 3, stride=2, padding=1)
                             for i, width in enumerate(channels)]
        self.norms = [nn.LayerNorm(width) for width in channels]
        self.refine = [nn.Conv2d(width, width, 3, padding=1, groups=width) for width in channels]

    def __call__(self, image: mx.array) -> mx.array:
        x = image
        for conv, norm, refine in zip(self.convolutions, self.norms, self.refine):
            x = nn.gelu(norm(conv(x)))
            x = x + nn.gelu(refine(x))
        return x


@dataclass(frozen=True)
class VisualFeatures:
    summary: mx.array  # B,T,Q*D
    cells: mx.array  # B,T,S,N,D; N padded to largest actual grid
    cell_bounds: mx.array  # B,T,S,N,4; normalized surface x0,y0,x1,y1
    cell_valid: mx.array  # B,T,S,N bool
    surface_valid: mx.array  # B,T,S bool


class VisualEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config.validate()
        width = config.spatial_width
        self.backbone = ConvNeXtEncoder(config)
        self.detail = DetailEncoder(config.detail_channels)
        self.detail_projection = nn.Linear(config.detail_channels[-1], width)
        self.global_projection = nn.Linear(config.backbone_dims[-1], width)
        self.middle_projection = nn.Linear(config.backbone_dims[1], width)
        self.global_norm = nn.LayerNorm(config.backbone_dims[-1])
        self.middle_norm = nn.LayerNorm(config.backbone_dims[1])
        self.fusion_norm = nn.LayerNorm(width)
        self.position_projection = nn.Linear(18, width, bias=False)
        self.crop_position = nn.Linear(4, width, bias=False)
        self.geometry_projection = nn.Linear(4, width, bias=False)
        self.surface_embedding = nn.Embedding(config.maximum_surfaces, width)
        self.branch_embedding = nn.Embedding(3, width)
        self.queries = mx.random.normal((config.query_count, width)) * (width ** -0.5)
        self.pool = nn.MultiHeadAttention(width, 4)
        self.pool_norm = nn.LayerNorm(width)
        self.null_token = mx.zeros((1, 1, width))

    def _surface(self, surface: SurfaceBatch, role: int, size: tuple[int, int]):
        batch, time = size
        flatten = lambda value: value.reshape(batch * time, *value.shape[2:])
        stages = self.backbone(flatten(surface.global_image))
        detail = self.detail(flatten(surface.detail_image))
        height, width = detail.shape[1:3]
        global_rect = flatten(surface.global_content_rect)
        detail_rect = flatten(surface.content_rect)
        coarse = align_features(self.global_projection(self.global_norm(stages[-1])), height, width, global_rect, detail_rect)
        middle = align_features(self.middle_projection(self.middle_norm(stages[1])), height, width, global_rect, detail_rect)
        dense = self.fusion_norm(self.detail_projection(detail) + coarse + middle)
        bounds, positions, valid = position_features(flatten(surface.content_rect), height, width)
        role_token = self.surface_embedding(mx.array(role))
        # Geometry retains relative monitor layout and scale with bounded values.
        geometry = mx.tanh(flatten(surface.global_bounds) / 4096)
        geometry_token = self.geometry_projection(geometry)[:, None, :]
        dense = dense.reshape(batch * time, height * width, -1) + self.position_projection(positions) + role_token + geometry_token
        valid = valid & flatten(surface.available)[:, None]
        crop = self.detail(flatten(surface.cursor_image))
        crop_height, crop_width = crop.shape[1:3]
        crop_rect = flatten(surface.cursor_rect)
        yy, xx = mx.meshgrid((mx.arange(crop_height) + 0.5) / crop_height,
                             (mx.arange(crop_width) + 0.5) / crop_width, indexing="ij")
        local = mx.stack((xx, yy), axis=-1).reshape(1, -1, 2)
        cursor_xy = crop_rect[:, None, :2] + local * crop_rect[:, None, 2:]
        cursor_valid = mx.all((cursor_xy >= 0) & (cursor_xy <= 1), axis=-1) & flatten(surface.available)[:, None]
        crop_tokens = self.detail_projection(crop).reshape(batch * time, -1, self.config.spatial_width)
        crop_tokens = crop_tokens + self.crop_position(mx.concatenate((cursor_xy, mx.broadcast_to(crop_rect[:, None, 2:], cursor_xy.shape)), axis=-1))
        crop_tokens = crop_tokens + role_token + geometry_token + self.branch_embedding(mx.array(1))
        # A pretrained pooled token stabilizes semantic transfer alongside dense
        # fine cells. Cursor tokens assist state estimation, not target leakage.
        _, _, global_valid = position_features(global_rect, *stages[-1].shape[1:3])
        global_weights = global_valid.reshape(*stages[-1].shape[:3], 1)
        global_mean = mx.sum(stages[-1] * global_weights, axis=(1, 2)) / mx.maximum(mx.sum(global_weights, axis=(1, 2)), 1)
        summary = self.global_projection(self.backbone.final_norm(global_mean))[:, None, :] + role_token + geometry_token + self.branch_embedding(mx.array(2))
        tokens = mx.concatenate((dense + self.branch_embedding(mx.array(0)), crop_tokens, summary), axis=1)
        masks = mx.concatenate((valid, cursor_valid, flatten(surface.available)[:, None]), axis=1)
        return dense, bounds, valid, tokens, masks

    def __call__(self, observation: ObservationBatch) -> VisualFeatures:
        batch, time = observation.shape
        dense, bounds, valid, tokens, masks = zip(*(self._surface(surface, index, (batch, time)) for index, surface in enumerate(observation.surfaces)))
        # A learnable null token means all-unavailable/padded observations have
        # finite attention. Their RNN updates remain masked by observation.valid.
        pooled_tokens = mx.concatenate((*tokens, mx.broadcast_to(self.null_token, (batch * time, 1, self.config.spatial_width))), axis=1)
        pooled_mask = mx.concatenate((*masks, mx.ones((batch * time, 1), dtype=mx.bool_)), axis=1)
        queries = mx.broadcast_to(self.queries[None], (batch * time, *self.queries.shape))
        summary = self.pool_norm(queries + self.pool(queries, pooled_tokens, pooled_tokens, mask=pooled_mask[:, None, None, :]))
        maximum = max(field.shape[1] for field in dense)
        padded = lambda value: mx.pad(value, [(0, 0), (0, maximum - value.shape[1]), *([(0, 0)] * (value.ndim - 2))])
        def stack(values):
            value = mx.stack([padded(field) for field in values], axis=1)
            return value.reshape(batch, time, *value.shape[1:])
        return VisualFeatures(summary.reshape(batch, time, -1), stack(dense), stack(bounds), stack(valid),
                              mx.stack([surface.available for surface in observation.surfaces], axis=-1))
