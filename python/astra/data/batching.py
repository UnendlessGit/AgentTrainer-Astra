from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import mlx.core as mx

from astra.model.actions import PacketBatch
from astra.model.config import ModelConfig
from astra.model.observation import ObservationBatch, SurfaceBatch
from astra.model.vision import VisualFeatures, position_features
from .actions import encode_commands


@dataclass(frozen=True)
class LearningSample:
    observation: ObservationBatch  # one batch lane, one decision
    surfaces: tuple[dict, ...]
    commands: tuple[dict, ...]
    episode_id: str
    step: int


def pointing_layout(observation: ObservationBatch) -> VisualFeatures:
    """The encoder's deterministic detail-grid geometry, without neural work."""
    batch, time = observation.shape
    surfaces = []
    for surface in observation.surfaces:
        height = (surface.detail_image.shape[2] + 7) // 8
        width = (surface.detail_image.shape[3] + 7) // 8
        bounds, _, mask = position_features(surface.content_rect.reshape(batch * time, 4), height, width)
        mask = mask & surface.available.reshape(batch * time, 1)
        surfaces.append((bounds, mask))
    maximum = max(mask.shape[1] for _, mask in surfaces)
    bounds = mx.stack([mx.pad(bounds, [(0, 0), (0, maximum - bounds.shape[1]), (0, 0)]) for bounds, _ in surfaces], axis=1)
    masks = mx.stack([mx.pad(mask, [(0, 0), (0, maximum - mask.shape[1])]) for _, mask in surfaces], axis=1)
    return VisualFeatures(mx.zeros((batch * time, 0)), mx.zeros((*masks.shape, 0)), bounds, masks,
                          mx.stack([surface.available.reshape(batch * time) for surface in observation.surfaces], axis=-1))


def stack_observations(rows: Sequence[Sequence[ObservationBatch]]) -> ObservationBatch:
    """Right/bottom pad spatial buckets; update transforms instead of stretching."""
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Observation batching needs a nonempty rectangular lane/time grid")
    batch, time = len(rows), len(rows[0])
    flat = [sample for row in rows for sample in row]
    count = max(len(sample.surfaces) for sample in flat)
    if not 1 <= count <= 16 or any(sample.shape != (1, 1) or not sample.surfaces for sample in flat):
        raise ValueError("Each sample must have one step and bounded nonempty surface roles")
    result = []
    for index in range(count):
        # Slots keep the recording's declared order. A shorter source has no
        # observations for trailing slots: masked padding cannot become a target.
        template = next(sample.surfaces[index] for sample in flat if index < len(sample.surfaces))
        absent = SurfaceBatch(
            global_image=mx.zeros((1, 1, 32, 32, 3)), detail_image=mx.zeros((1, 1, 32, 32, 3)),
            cursor_image=mx.zeros_like(template.cursor_image),
            global_content_rect=mx.array([[[0., 0., 1., 1.]]]), content_rect=mx.array([[[0., 0., 1., 1.]]]),
            cursor_rect=mx.array([[[2., 2., 1., 1.]]]), global_bounds=mx.array([[[0., 0., 1., 1.]]]),
            available=mx.array([[False]]))
        surfaces = [sample.surfaces[index] if index < len(sample.surfaces) else absent for sample in flat]
        images = {}
        rectangles = {}
        for image_name, rect_name in (("global_image", "global_content_rect"), ("detail_image", "content_rect"), ("cursor_image", None)):
            values = [getattr(surface, image_name)[0, 0] for surface in surfaces]
            height, width = max(value.shape[0] for value in values), max(value.shape[1] for value in values)
            if rect_name is None and any(value.shape[:2] != (height, width) for value in values):
                raise ValueError("Cursor crop size is immutable within a model")
            padded = [mx.pad(value, [(0, height - value.shape[0]), (0, width - value.shape[1]), (0, 0)]) for value in values]
            images[image_name] = mx.stack(padded).reshape(batch, time, height, width, 3)
            if rect_name:
                adjusted = [getattr(surface, rect_name)[0, 0] * mx.array([value.shape[1] / width, value.shape[0] / height] * 2)
                            for surface, value in zip(surfaces, values)]
                rectangles[rect_name] = mx.stack(adjusted).reshape(batch, time, 4)
        other = {name: mx.stack([getattr(surface, name)[0, 0] for surface in surfaces]).reshape(batch, time, *getattr(surfaces[0], name).shape[2:])
                 for name in ("cursor_rect", "global_bounds", "available")}
        result.append(SurfaceBatch(**images, **rectangles, **other))
    fields = {name: mx.stack([getattr(sample, name)[0, 0] for sample in flat]).reshape(batch, time, *getattr(flat[0], name).shape[2:])
              for name in ("controls", "elapsed_seconds", "context_ids", "reset", "valid")}
    return ObservationBatch(tuple(result), **fields)


def training_batch(rows: Sequence[Sequence[LearningSample | None]], config: ModelConfig, vocabulary):
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Training samples need a nonempty rectangular lane/time grid")
    for row in rows:
        previous = None
        padded = False
        for sample in row:
            if sample is None:
                padded = True
                continue
            if padded:
                raise ValueError("Training lanes may only have padding after their final sample")
            if (not isinstance(sample, LearningSample) or type(sample.step) is not int or sample.step < 0
                or type(sample.episode_id) is not str or not sample.episode_id
                or sample.observation.shape != (1, 1) or not bool(sample.observation.valid.item())):
                raise ValueError("Invalid learning sample identity, step or observation")
            reset = bool(sample.observation.reset.item())
            if sample.step == 0 and not reset:
                raise ValueError("The first episode sample must reset recurrent state")
            if previous is not None:
                if previous.episode_id == sample.episode_id:
                    if sample.step != previous.step + 1:
                        raise ValueError("A training lane skipped or repeated an episode step")
                elif sample.step != 0 or not reset:
                    raise ValueError("A changed episode must begin with a recurrent reset")
                if sample.surfaces != previous.surfaces and not reset:
                    raise ValueError("Changed observation geometry requires a recurrent reset")
            previous = sample
    examples = [sample for row in rows for sample in row if sample is not None]
    if not examples:
        raise ValueError("A training batch has no valid samples")
    # Remove only columns that are padding in every lane. A two-decision
    # episode does not require sixty-two full-resolution blank forward passes.
    last_active = max(index for row in rows for index, sample in enumerate(row) if sample is not None)
    rows = [row[:last_active + 1] for row in rows]
    from dataclasses import replace
    template = examples[0].observation
    padding = replace(template, valid=mx.zeros((1, 1), dtype=mx.bool_), reset=mx.zeros((1, 1), dtype=mx.bool_))
    observation = stack_observations([[sample.observation if sample is not None else padding for sample in row] for row in rows])
    layout = pointing_layout(observation)
    packets = []
    for index, sample in enumerate(sample for row in rows for sample in row):
        if sample is None:
            packets.append(PacketBatch.zeros(1, config.packet_capacity + 1))
        else:
            packets.append(encode_commands(sample.commands, config=config, vocabulary=vocabulary, visual=layout,
                                           surfaces=sample.surfaces, batch_index=index))
    labels = PacketBatch(**{field: mx.concatenate([getattr(packet, field) for packet in packets], axis=0) for field in PacketBatch.__dataclass_fields__})
    return observation, labels
