"""NHWC ConvNeXt backbone with explicitly verified upstream weight mapping.

Architecture: Liu et al., A ConvNet for the 2020s (2022).
The reference model and pretrained provenance are documented in assets/weights.json.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelConfig


def content_mask(rect: mx.array, height: int, width: int) -> mx.array:
    """Cells intersecting real content, as an NHWC-broadcastable Boolean mask.

    Image-size masks use exact pixel bounds; stride masks retain partial edge
    cells. The caller passes the same normalized content rectangle throughout
    the pyramid, without stretching it to fill the padded canvas.
    """
    yy, xx = mx.meshgrid(mx.arange(height, dtype=mx.float32), mx.arange(width, dtype=mx.float32), indexing="ij")
    low = mx.stack((xx / width, yy / height), axis=-1)[None]
    high = mx.stack(((xx + 1) / width, (yy + 1) / height), axis=-1)[None]
    origin, extent = rect[:, None, None, :2], rect[:, None, None, 2:]
    return mx.all((high > origin) & (low < origin + extent) & (extent > 0), axis=-1)[..., None]


def mask_content(value: mx.array, mask: mx.array | None) -> mx.array:
    return value if mask is None else mx.where(mask, value, 0.0)


class Downsample(nn.Module):
    def __init__(self, input_width: int, output_width: int, *, stem: bool = False):
        super().__init__()
        self.stem = stem
        self.conv = nn.Conv2d(input_width, output_width, 4 if stem else 2, stride=4 if stem else 2)
        self.norm = nn.LayerNorm(output_width if stem else input_width, eps=1e-6)

    def __call__(self, x, *, input_mask=None, output_mask=None):
        x = mask_content(x, input_mask)
        if self.stem:
            return mask_content(self.norm(self.conv(x)), output_mask)
        # Mask *after* normalization too: its trainable bias must not turn
        # padding into an input to a neighboring valid convolution cell.
        return mask_content(self.conv(mask_content(self.norm(x), input_mask)), output_mask)


class ConvNeXtBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, 7, padding=3, groups=width)
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.expand = nn.Linear(width, 4 * width)
        self.contract = nn.Linear(4 * width, width)
        self.scale = mx.full((width,), 1e-6)

    def __call__(self, x, *, mask=None):
        x = mask_content(x, mask)
        residual = self.contract(nn.gelu(self.expand(self.norm(self.depthwise(x)))))
        return mask_content(x + residual * self.scale, mask)


class ConvNeXtEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.dims = config.backbone_dims
        self.depths = config.backbone_depths
        self.downsamples = [Downsample(3 if index == 0 else self.dims[index - 1], width, stem=index == 0)
                            for index, width in enumerate(self.dims)]
        self.stages = [[ConvNeXtBlock(width) for _ in range(depth)]
                       for width, depth in zip(self.dims, self.depths)]
        self.final_norm = nn.LayerNorm(self.dims[-1], eps=1e-6)

    def __call__(self, x, content_rect: mx.array | None = None):
        if x.ndim != 4 or x.shape[-1] != 3 or min(x.shape[1:3]) < 32:
            raise ValueError("ConvNeXt expects NHWC RGB images at least 32 pixels per side")
        stages = []
        mask = None
        if content_rect is not None:
            if content_rect.shape != (x.shape[0], 4):
                raise ValueError("Backbone content rectangles must match its image batch")
            mask = content_mask(content_rect, *x.shape[1:3])
        for downsample, blocks in zip(self.downsamples, self.stages):
            stride = 4 if downsample.stem else 2
            next_mask = None if content_rect is None else content_mask(content_rect, x.shape[1] // stride, x.shape[2] // stride)
            x = downsample(x, input_mask=mask, output_mask=next_mask)
            mask = next_mask
            for block in blocks:
                x = block(x, mask=mask)
            stages.append(x)
        return stages

    def summary(self, stages):
        return self.final_norm(mx.mean(stages[-1], axis=(1, 2)))

    def load_pretrained(self, path: Path) -> None:
        if self.dims != (96, 192, 384, 768) or self.depths != (3, 3, 9, 3):
            raise ValueError("The pretrained artifact requires the production ConvNeXt-Tiny layout")
        self.load_weights(str(path), strict=True)
        mx.eval(self.parameters())


def convert_facebook_weights(state: Mapping[str, np.ndarray]) -> dict[str, mx.array]:
    """Convert tensor layout/names without executing serialized model objects."""
    converted = {}
    for name, tensor in state.items():
        value = np.asarray(tensor, dtype=np.float32)
        if name.startswith("head."):
            continue  # The classification head is not part of Astra's backbone.
        match = re.fullmatch(r"downsample_layers\.(\d)\.(\d)\.(weight|bias)", name)
        if match:
            stage, slot, suffix = int(match[1]), int(match[2]), match[3]
            component = "conv" if (stage == 0 and slot == 0) or (stage > 0 and slot == 1) else "norm"
            target = f"downsamples.{stage}.{component}.{suffix}"
        else:
            match = re.fullmatch(r"stages\.(\d)\.(\d+)\.(dwconv|norm|pwconv1|pwconv2)\.(weight|bias)", name)
            if match:
                stage, block, component, suffix = match.groups()
                component = {"dwconv": "depthwise", "norm": "norm", "pwconv1": "expand", "pwconv2": "contract"}[component]
                target = f"stages.{stage}.{block}.{component}.{suffix}"
            else:
                match = re.fullmatch(r"stages\.(\d)\.(\d+)\.gamma", name)
                if match:
                    target = f"stages.{match[1]}.{match[2]}.scale"
                elif name in ("norm.weight", "norm.bias"):
                    target = "final_" + name
                else:
                    raise ValueError(f"Unexpected pretrained parameter: {name}")
        if value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)
        converted[target] = mx.array(np.ascontiguousarray(value))
    return converted
