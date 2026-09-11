"""Deterministic CPU reference for native BGRA observation preparation.

Images retain aspect ratio, carry independent padding transforms, and use the
actual observed cursor. Metal preprocessing must qualify against this path.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import mlx.core as mx
import numpy as np
from PIL import Image

from astra.model.config import ModelConfig
from astra.model.observation import SurfaceBatch
from astra.recordings import RecordingError, validate_frame

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class PreparedSurface:
    global_image: np.ndarray
    detail_image: np.ndarray
    cursor_image: np.ndarray
    global_content_rect: np.ndarray
    content_rect: np.ndarray
    cursor_rect: np.ndarray
    global_bounds: np.ndarray
    available: bool = True

    def batch(self) -> SurfaceBatch:
        fields = {name: mx.array(getattr(self, name))[None, None] for name in self.__dataclass_fields__}
        return SurfaceBatch(**fields)


def _normalized(image: Image.Image) -> np.ndarray:
    return (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD


def _letterbox(image: Image.Image, box, limit: int, alignment: int):
    width, height = box[2] - box[0], box[3] - box[1]
    limit = (limit // alignment) * alignment
    if limit < alignment:
        raise RecordingError("The configured visual size is below its network stride")
    scale = min(1.0, limit / max(width, height))
    resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
    canvas_width = max(32, math.ceil(resized_width / alignment) * alignment)
    canvas_height = max(32, math.ceil(resized_height / alignment) * alignment)
    x, y = (canvas_width - resized_width) // 2, (canvas_height - resized_height) // 2
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS, box=box)
    result = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    result[y:y + resized_height, x:x + resized_width] = _normalized(resized)
    rect = np.array([x / canvas_width, y / canvas_height, resized_width / canvas_width, resized_height / canvas_height], dtype=np.float32)
    return result, rect


def prepare_surface(pixels: np.ndarray, metadata: dict, config: ModelConfig,
                    *, pointer: tuple[float, float] | None, maximum_timestamp: int = 2**63 - 1) -> PreparedSurface:
    config.validate()
    validate_frame(metadata, maximum_timestamp=maximum_timestamp)
    surface = metadata["surface"]
    if pixels.dtype != np.uint8 or pixels.shape != (surface["pixelHeight"], surface["pixelWidth"], 4):
        raise RecordingError("Native frame pixels disagree with their metadata")
    # Native capture BGRA is premultiplied. Composite transparent window pixels
    # onto the same neutral backdrop used for padding, then normalize once.
    rgb = pixels[..., [2, 1, 0]].astype(np.float32)
    alpha = pixels[..., 3:4].astype(np.float32) / 255
    rgb = np.clip(rgb + MEAN * 255 * (1 - alpha), 0, 255).round().astype(np.uint8)
    image = Image.fromarray(rgb)
    content = surface["contentBounds"]
    box = (content["x"], content["y"], content["x"] + content["width"], content["y"] + content["height"])
    global_image, global_rect = _letterbox(image, box, config.global_long_edge, 32)
    detail_image, detail_rect = _letterbox(image, box, config.detail_long_edge, 8)
    bounds = surface["globalBounds"]
    geometry = [bounds[key] for key in ("x", "y", "width", "height")]
    if any(abs(value) > float(np.finfo(np.float32).max) for value in geometry):
        raise RecordingError("Surface bounds exceed the model's numerical range")
    global_bounds = np.array(geometry, dtype=np.float32)
    if np.any(global_bounds[2:] <= 0):
        raise RecordingError("Surface bounds are below the model's numerical range")
    size = config.cursor_size
    crop = np.zeros((size, size, 3), dtype=np.float32)
    # An entirely out-of-surface crop rectangle masks all cursor tokens when
    # pointer state is unavailable. It must not masquerade as a real empty crop.
    crop_rect = np.array([2, 2, 1, 1], dtype=np.float32)
    if pointer is not None:
        if len(pointer) != 2 or not all(math.isfinite(value) for value in pointer):
            raise RecordingError("The observed pointer is invalid")
        px = content["x"] + (pointer[0] - bounds["x"]) / bounds["width"] * content["width"]
        py = content["y"] + (pointer[1] - bounds["y"]) / bounds["height"] * content["height"]
        if not math.isfinite(px) or not math.isfinite(py):
            raise RecordingError("The observed pointer cannot be mapped into source pixels")
        left, top = math.floor(px - size / 2), math.floor(py - size / 2)
        x0, y0 = max(left, math.floor(box[0])), max(top, math.floor(box[1]))
        x1, y1 = min(left + size, math.ceil(box[2])), min(top + size, math.ceil(box[3]))
        if x1 > x0 and y1 > y0:
            crop[y0 - top:y1 - top, x0 - left:x1 - left] = _normalized(image.crop((x0, y0, x1, y1)))
        rectangle = [(left - content["x"]) / content["width"], (top - content["y"]) / content["height"], size / content["width"], size / content["height"]]
        if any(not math.isfinite(value) or abs(value) > float(np.finfo(np.float32).max) for value in rectangle):
            raise RecordingError("Cursor geometry exceeds the model's numerical range")
        crop_rect = np.array(rectangle, dtype=np.float32)
    return PreparedSurface(global_image, detail_image, crop, global_rect, detail_rect, crop_rect, global_bounds)
