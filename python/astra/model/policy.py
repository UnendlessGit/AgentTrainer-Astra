from __future__ import annotations

from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from .config import ModelConfig
from .observation import ObservationBatch
from .temporal import TemporalCore, TemporalOutput
from .vision import VisualEncoder, VisualFeatures
from .actions import ActionVocabulary, PacketBatch, PacketDecoder, PacketDistributionResult, flatten_visual
from .checkpointing import checkpoint_call


@dataclass(frozen=True)
class PolicyEncoding:
    visual: VisualFeatures
    temporal: TemporalOutput


class PolicyCore(nn.Module):
    """The exact same causal visual/recurrent path is used by BC, PPO and actors."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config.validate()
        self.vision = VisualEncoder(config)
        self.temporal = TemporalCore(config)
        self._vision_microbatch = 4
        self._checkpoint_vision = False

    def configure_execution(self, *, vision_microbatch: int = 4, checkpoint_vision: bool = False) -> None:
        """Memory controls change execution, never the learned representation."""
        if type(vision_microbatch) is not int or not 1 <= vision_microbatch <= 64 or type(checkpoint_vision) is not bool:
            raise ValueError("Invalid visual execution settings")
        self._vision_microbatch = vision_microbatch
        self._checkpoint_vision = checkpoint_vision

    def encode_visual(self, observation: ObservationBatch) -> VisualFeatures:
        batch, time = observation.shape
        flattened = tree_map(lambda value: value.reshape(batch * time, 1, *value.shape[2:]), observation.as_tensors())
        chunks = []
        def encode(module, tensors):
            result = module(ObservationBatch.from_tensors(tensors))
            return tuple(getattr(result, field) for field in VisualFeatures.__dataclass_fields__)
        for start in range(0, batch * time, self._vision_microbatch):
            chunk = tree_map(lambda value: value[start:start + self._vision_microbatch], flattened)
            if self._checkpoint_vision:
                result = checkpoint_call(self.vision, encode, chunk)
            else:
                result = encode(self.vision, chunk)
            chunks.append(result)
        results = [mx.concatenate([chunk[index] for chunk in chunks], axis=0) for index in range(len(chunks[0]))]
        return VisualFeatures(*(value.reshape(batch, time, *value.shape[2:]) for value in results))

    def __call__(self, observation: ObservationBatch, state: tuple[mx.array, ...] | None = None) -> PolicyEncoding:
        observation.validate_shapes(control_width=self.config.control_width,
                                    maximum_surfaces=self.config.maximum_surfaces,
                                    contexts=len(self.config.context_sizes))
        visual = self.encode_visual(observation)
        temporal = self.temporal(visual.summary, observation, state)
        return PolicyEncoding(visual, temporal)


class AgentPolicy(PolicyCore):
    def __init__(self, config: ModelConfig, vocabulary: ActionVocabulary):
        super().__init__(config)
        self.actions = PacketDecoder(config, vocabulary)

    def score(self, encoding: PolicyEncoding, packets: PacketBatch) -> PacketDistributionResult:
        context = encoding.temporal.context
        return self.actions.log_prob(context.reshape(-1, context.shape[-1]), flatten_visual(encoding.visual), packets)

    def sample(self, encoding: PolicyEncoding, *, key: mx.array, greedy: bool = False) -> PacketDistributionResult:
        context = encoding.temporal.context
        return self.actions.sample(context.reshape(-1, context.shape[-1]), flatten_visual(encoding.visual), key=key, greedy=greedy)
