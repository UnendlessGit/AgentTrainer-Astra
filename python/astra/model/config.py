from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

MAXIMUM_PARAMETERS = 500_000_000


@dataclass(frozen=True)
class ModelConfig:
    schema_version: int = 2
    global_long_edge: int = 768
    detail_long_edge: int = 1536
    cursor_size: int = 384
    detail_channels: tuple[int, ...] = (32, 64, 128)
    spatial_width: int = 128
    query_count: int = 32
    recurrent_width: int = 512
    recurrent_layers: int = 2
    control_width: int = 178
    decoder_width: int = 256
    packet_capacity: int = 16
    maximum_surfaces: int = 16
    period_ms: int = 100
    lead_ms: int = 100
    coordinate_bins: int = 64
    delta_magnitude: int = 4095
    delta_radix: int = 64
    backbone_dims: tuple[int, ...] = (96, 192, 384, 768)
    backbone_depths: tuple[int, ...] = (3, 3, 9, 3)
    context_sizes: tuple[int, ...] = ()
    context_width: int = 32

    def validate(self) -> ModelConfig:
        if self.schema_version != 2:
            raise ValueError("Unsupported model schema; version 2 requires internal visual padding masks")
        if any(type(value) is not tuple for value in
               (self.detail_channels, self.backbone_dims, self.backbone_depths, self.context_sizes)):
            raise ValueError("Model layouts and vocabularies must be immutable tuples")
        positive = (self.global_long_edge, self.detail_long_edge, self.cursor_size,
                    self.spatial_width, self.query_count, self.recurrent_width, self.recurrent_layers,
                    self.control_width, self.decoder_width, self.maximum_surfaces, self.coordinate_bins,
                    self.delta_magnitude, self.delta_radix,
                    self.context_width, *self.detail_channels, *self.backbone_dims, *self.backbone_depths)
        if any(type(value) is not int or not 1 <= value <= 8192 for value in positive):
            raise ValueError("Model dimensions must be bounded positive integers")
        if len(self.backbone_dims) != 4 or len(self.backbone_depths) != 4 or len(self.detail_channels) != 3:
            raise ValueError("Backbone/detail stage layouts are incompatible")
        if any(type(value) is not int for value in (self.schema_version, self.packet_capacity, self.period_ms, self.lead_ms)):
            raise ValueError("Configuration versions and timings must be integers")
        if self.packet_capacity not in (16, 32, 64) or not 1 <= self.period_ms <= 1000 or not 0 <= self.lead_ms <= 2000:
            raise ValueError("Invalid action packet cadence/capacity")
        if self.maximum_surfaces > 16 or self.recurrent_layers > 4 or self.query_count > 128:
            raise ValueError("Model configuration exceeds supported resource bounds")
        if any(type(value) is not int or not 1 <= value <= 65536 for value in self.context_sizes):
            raise ValueError("Invalid context vocabulary")
        if len(self.context_sizes) > 32 or self.spatial_width % 4 or self.coordinate_bins < 2:
            raise ValueError("Unsupported attention or context layout")
        if min(self.global_long_edge, self.detail_long_edge, self.cursor_size) < 32:
            raise ValueError("Visual branches need at least 32 pixels per side")
        if self.recurrent_width < 2:
            raise ValueError("The recurrent width must support a nonempty value head")
        # This integer-only calculation runs before any MLX module constructs
        # parameter arrays, including when loading an untrusted configuration.
        if self.parameter_count > MAXIMUM_PARAMETERS:
            raise ValueError("Policy exceeds the supported parameter budget")
        return self

    @property
    def parameter_count(self) -> int:
        """Exact AgentPolicy parameter count without allocating model arrays.

        The count includes frozen parameters and the packet decoder. Vocabularies
        mask fixed heads and therefore do not change tensor dimensions. GRU has
        Wx/Wh (three gates), a three-gate bias and the candidate recurrent bias;
        attention has four bias-free projections.
        """
        linear = lambda input_width, output_width: (input_width + 1) * output_width
        gru = lambda width: 6 * width * width + 4 * width
        dims, depths = self.backbone_dims, self.backbone_depths
        backbone = 51 * dims[0] + 2 * dims[-1]
        backbone += sum(4 * before * after + after + 2 * before for before, after in zip(dims, dims[1:]))
        backbone += sum(depth * (8 * width * width + 58 * width) for width, depth in zip(dims, depths))
        detail = sum(9 * before * after + 13 * after for before, after in
                     zip((3, *self.detail_channels[:-1]), self.detail_channels))
        spatial = self.spatial_width
        vision = backbone + detail + sum(linear(width, spatial) for width in (self.detail_channels[-1], dims[-1], dims[1]))
        vision += 2 * dims[-1] + 2 * dims[1] + 2 * spatial  # global/middle/fusion norms
        vision += (18 + 4 + 4 + self.maximum_surfaces + 3 + self.query_count + 1) * spatial
        vision += 4 * spatial * spatial + 2 * spatial  # attention and pool norm
        hidden = self.recurrent_width
        features = self.query_count * spatial + self.control_width + 2 + len(self.context_sizes) * self.context_width
        temporal = sum(self.context_sizes) * self.context_width + linear(features, hidden) + 2 * hidden
        temporal += self.recurrent_layers * (gru(hidden) + 2 * hidden)
        temporal += linear(hidden, hidden // 2) + linear(hidden // 2, 1)
        decoder = self.decoder_width
        coarse = (2 * self.delta_magnitude + self.delta_radix) // self.delta_radix
        categories = (9, self.period_ms + 1, 128, 32, self.coordinate_bins, self.coordinate_bins,
                      coarse, self.delta_radix, coarse, self.delta_radix)
        actions = linear(hidden, decoder) + gru(decoder) + 4 * decoder
        actions += (sum(categories) + self.maximum_surfaces) * decoder
        actions += sum(linear(decoder, count) for count in categories)
        actions += 2 * linear(decoder, spatial) + linear(spatial, decoder)
        return vision + temporal + actions

    def to_dict(self) -> dict:
        value = asdict(self)
        for field in ("detail_channels", "backbone_dims", "backbone_depths", "context_sizes"):
            value[field] = list(value[field])
        return value

    @classmethod
    def from_dict(cls, value: dict) -> ModelConfig:
        if not isinstance(value, dict):
            raise ValueError("Model configuration must be an object")
        fields = set(cls.__dataclass_fields__)
        if set(value) - fields:
            raise ValueError("Unknown model configuration fields")
        copied = dict(value)
        for key in ("detail_channels", "backbone_dims", "backbone_depths", "context_sizes"):
            if key in copied:
                copied[key] = tuple(copied[key])
        return cls(**copied).validate()

    @property
    def signature(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @classmethod
    def test_small(cls) -> ModelConfig:
        """Numerical-test configuration, never the production default."""
        return cls(global_long_edge=64, detail_long_edge=64, cursor_size=32,
                   detail_channels=(8, 16, 24), spatial_width=24, query_count=4,
                   recurrent_width=32, decoder_width=32, backbone_dims=(8, 16, 24, 32),
                   backbone_depths=(1, 1, 1, 1), context_width=8).validate()
