"""Causal observation tensors shared by training, actor, and evaluation.

Spatial shapes are bucketed by the loader, not stretched. All images are NHWC,
FP32 ImageNet-normalized RGB. Letterbox padding is zero after normalization.
Each surface can use its own spatial bucket; its B,T dimensions agree with the
batch. A cursor crop is centered on the *observed* pointer, never a target label.
"""
from __future__ import annotations

from dataclasses import dataclass
import mlx.core as mx


@dataclass(frozen=True)
class SurfaceBatch:
    global_image: mx.array  # B,T,H,W,3
    detail_image: mx.array  # B,T,H,W,3
    cursor_image: mx.array  # B,T,H,W,3
    # Fractional left,top,width,height of actual content in each padded image.
    # Independent rectangles allow native detail without artificial upscaling
    # merely to match the global branch's stride-32 padding.
    global_content_rect: mx.array  # B,T,4; normalized global image canvas
    content_rect: mx.array  # B,T,4; normalized detail image canvas
    # Left,top,width,height of cursor crop in normalized surface coordinates.
    cursor_rect: mx.array  # B,T,4; may extend beyond the surface
    # left,top,width,height in global logical points; explicit display mapping.
    global_bounds: mx.array  # B,T,4
    available: mx.array  # B,T bool

    def slice_time(self, start: int, end: int) -> SurfaceBatch:
        return SurfaceBatch(**{name: getattr(self, name)[:, start:end] for name in self.__dataclass_fields__})


@dataclass(frozen=True)
class ObservationBatch:
    surfaces: tuple[SurfaceBatch, ...]
    controls: mx.array  # B,T,178; versioned executed/physical state features
    elapsed_seconds: mx.array  # B,T; actual interval, not nominal cadence
    context_ids: mx.array  # B,T,number of immutable context vocabularies
    reset: mx.array  # B,T bool; reset *before* this observation
    valid: mx.array  # B,T bool; padding never advances persistent state

    @property
    def shape(self) -> tuple[int, int]:
        return self.controls.shape[:2]

    def validate_shapes(self, *, control_width: int, maximum_surfaces: int, contexts: int) -> None:
        if self.controls.ndim != 3 or self.controls.shape[-1] != control_width:
            raise ValueError("Invalid observed-control tensor shape")
        batch, time = self.shape
        if not batch or not time or not 1 <= len(self.surfaces) <= maximum_surfaces:
            raise ValueError("An observation requires a nonempty batch/time and bounded surfaces")
        for field in (self.elapsed_seconds, self.reset, self.valid):
            if field.shape != (batch, time):
                raise ValueError("Temporal masks and elapsed time must match B,T")
        if self.context_ids.shape != (batch, time, contexts):
            raise ValueError("Context vocabulary dimensions do not match the checkpoint")
        for surface in self.surfaces:
            for image in (surface.global_image, surface.detail_image, surface.cursor_image):
                if image.ndim != 5 or image.shape[:2] != (batch, time) or image.shape[-1] != 3 or min(image.shape[2:4]) < 32:
                    raise ValueError("Surface images must be B,T,H,W,RGB with at least 32 pixels per side")
            for field in (surface.global_content_rect, surface.content_rect, surface.cursor_rect, surface.global_bounds):
                if field.shape != (batch, time, 4):
                    raise ValueError("Surface transforms must have shape B,T,4")
            if surface.available.shape != (batch, time):
                raise ValueError("Surface availability must match B,T")

    def slice_time(self, start: int, end: int) -> ObservationBatch:
        return ObservationBatch(tuple(surface.slice_time(start, end) for surface in self.surfaces),
                                self.controls[:, start:end], self.elapsed_seconds[:, start:end],
                                self.context_ids[:, start:end], self.reset[:, start:end], self.valid[:, start:end])

    def as_tensors(self) -> dict:
        return {"surfaces": [{name: getattr(surface, name) for name in SurfaceBatch.__dataclass_fields__}
                             for surface in self.surfaces],
                **{name: getattr(self, name) for name in self.__dataclass_fields__ if name != "surfaces"}}

    @classmethod
    def from_tensors(cls, value: dict) -> ObservationBatch:
        return cls(surfaces=tuple(SurfaceBatch(**surface) for surface in value["surfaces"]),
                   **{name: field for name, field in value.items() if name != "surfaces"})
