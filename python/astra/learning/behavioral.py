"""Contiguous recurrent behavioral learning, evaluation, and resumable state.

The model, categorical packet factors, and checkpoint format are identical to
the RL actor's. Only loss construction and episode sampling differ.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import time
from typing import Callable, Iterator

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
import numpy as np

from astra.data.batching import LearningSample, training_batch
from astra.model.actions import PacketBatch
from astra.model.policy import AgentPolicy
from astra.checkpoints import restore_mlx_random_state
from .optimizers import GroupedAdamW, finite_gradients
from .backward import policy_gradients


@dataclass(frozen=True)
class BehaviorConfig:
    schema_version: int = 1
    epochs: int = 20
    lanes: int = 2
    sequence_length: int = 64
    accumulation_chunks: int = 1
    learning_rate: float = 3e-4
    pretrained_learning_rate: float = 3e-5
    weight_decay: float = 0.01
    gradient_norm: float = 1.0
    freeze_pretrained_epochs: int = 1
    seed: int = 0
    vision_microbatch: int = 1
    checkpoint_vision: bool = True

    def validate(self):
        bounds = {"epochs": (1, 100000), "lanes": (1, 32), "sequence_length": (1, 512), "accumulation_chunks": (1, 256),
                  "freeze_pretrained_epochs": (0, 100000), "seed": (0, 2**63 - 1), "vision_microbatch": (1, 64)}
        if self.schema_version != 1 or type(self.schema_version) is not int or type(self.checkpoint_vision) is not bool:
            raise ValueError("Unsupported behavioral configuration")
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid behavioral {name}")
        for name in ("learning_rate", "pretrained_learning_rate", "weight_decay", "gradient_norm"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 10 or (name != "weight_decay" and value == 0):
                raise ValueError(f"Invalid behavioral {name}")
        return self


@dataclass(frozen=True)
class BehaviorMetrics:
    epoch: int
    updates: int
    decisions: int
    mean_nll: float
    gradient_norm: float
    decisions_per_second: float
    peak_memory_bytes: int


class EpisodeLanes:
    """Replayable shuffled episodes; lanes remain contiguous across TBPTT cuts."""
    def __init__(self, source, *, split: str, lanes: int, sequence_length: int, seed: int, epoch: int, state: dict | None = None):
        self.source = source
        episodes = source.episodes(split)
        self.lengths = {episode["id"]: episode["steps"] for episode in episodes}
        self.lanes, self.sequence_length = lanes, sequence_length
        if state is None:
            self.order = list(self.lengths)
            np.random.default_rng(np.random.SeedSequence([seed, epoch])).shuffle(self.order)
            self.next_episode = 0
            self.active = [None] * lanes
            self.offsets = [0] * lanes
        else:
            if set(state) != {"order", "next_episode", "active", "offsets"}:
                raise ValueError("Invalid episode sampler checkpoint")
            self.order = list(state["order"])
            self.next_episode = state["next_episode"]
            self.active = list(state["active"])
            self.offsets = list(state["offsets"])
            if set(self.order) != self.lengths.keys() or len(self.order) != len(self.lengths) or len(self.active) != lanes or len(self.offsets) != lanes:
                raise ValueError("Episode sampler checkpoint does not match this dataset")
            if type(self.next_episode) is not int or not 0 <= self.next_episode <= len(self.order):
                raise ValueError("Invalid saved episode cursor")
            for episode, offset in zip(self.active, self.offsets):
                if type(offset) is not int or offset < 0 or (episode is not None and (episode not in self.lengths or offset > self.lengths[episode])):
                    raise ValueError("Invalid saved episode position")

    @property
    def state(self):
        return {"order": list(self.order), "next_episode": self.next_episode, "active": list(self.active), "offsets": list(self.offsets)}

    def next(self) -> list[list[LearningSample | None]] | None:
        rows = []
        for lane in range(self.lanes):
            current = self.active[lane]
            if current is None or self.offsets[lane] == self.lengths[current]:
                if self.next_episode < len(self.order):
                    current = self.order[self.next_episode]; self.next_episode += 1
                    self.active[lane] = current; self.offsets[lane] = 0
                else:
                    self.active[lane] = None; rows.append([None] * self.sequence_length); continue
            samples = list(self.source.samples(current, self.offsets[lane], self.sequence_length))
            expected = min(self.sequence_length, self.lengths[current] - self.offsets[lane])
            if len(samples) != expected or any(sample.episode_id != current or sample.step != self.offsets[lane] + index for index, sample in enumerate(samples)):
                raise ValueError("Dataset returned a noncontiguous or incomplete episode chunk")
            self.offsets[lane] += len(samples)
            rows.append(samples + [None] * (self.sequence_length - len(samples)))
        return rows if any(sample is not None for row in rows for sample in row) else None


def behavior_loss(policy: AgentPolicy, observation, labels: PacketBatch, state=None):
    encoding = policy(observation, state)
    result = policy.score(encoding, labels)
    valid = observation.valid.reshape(-1)
    loss = -mx.sum(mx.where(valid, result.log_probability, 0))
    return loss, (encoding.temporal.state, mx.sum(valid))


class BehaviorTrainer:
    def __init__(self, policy: AgentPolicy, config: BehaviorConfig, *, dataset_id: str, restored_state: dict | None = None):
        self.policy = policy
        self.config = config.validate()
        self.dataset_id = dataset_id
        self.optimizer = GroupedAdamW(learning_rate=config.learning_rate, pretrained_learning_rate=config.pretrained_learning_rate, weight_decay=config.weight_decay)
        self.epoch = 0; self.updates = 0; self.decisions = 0
        self.sampler_state = None
        self.recurrent_state = None
        self.pending_gradients = None
        self.pending_count = 0
        self.pending_chunks = 0
        if restored_state is not None:
            if restored_state.get("kind") != "behavioral" or restored_state.get("datasetID") != dataset_id or restored_state.get("config") != asdict(config):
                raise ValueError("Training resume requires the same dataset revision and training configuration")
            self.optimizer.state = restored_state["optimizer"]
            self.epoch = restored_state["epoch"]; self.updates = restored_state["updates"]; self.decisions = restored_state["decisions"]
            self.sampler_state = restored_state["sampler"]
            self.recurrent_state = restored_state["recurrent"]
            self.pending_gradients = restored_state["pendingGradients"]
            self.pending_count = restored_state["pendingCount"]; self.pending_chunks = restored_state["pendingChunks"]
            restore_mlx_random_state(restored_state["rng"])
        self.policy.configure_execution(vision_microbatch=config.vision_microbatch, checkpoint_vision=config.checkpoint_vision)
        self._configure_freeze()

    def _configure_freeze(self):
        if self.epoch < self.config.freeze_pretrained_epochs:
            self.policy.vision.backbone.freeze()
        else:
            self.policy.vision.backbone.unfreeze()

    @property
    def state(self) -> dict:
        return {"kind": "behavioral", "datasetID": self.dataset_id, "config": asdict(self.config), "epoch": self.epoch,
                "updates": self.updates, "decisions": self.decisions, "sampler": self.sampler_state,
                "recurrent": self.recurrent_state, "pendingGradients": self.pending_gradients,
                "pendingCount": self.pending_count, "pendingChunks": self.pending_chunks,
                "optimizer": self.optimizer.state, "rng": tuple(mx.random.state)}

    def _update(self) -> float:
        if self.pending_gradients is None or not self.pending_count:
            return 0
        gradients = tree_map(lambda value: value / self.pending_count, self.pending_gradients)
        if not finite_gradients(gradients):
            raise FloatingPointError("Behavioral training produced nonfinite gradients; optimizer update was rejected")
        clipped, norm = optim.clip_grad_norm(gradients, self.config.gradient_norm)
        mx.eval(clipped, norm)
        if not math.isfinite(float(norm)):
            raise FloatingPointError("Behavioral gradient norm is nonfinite")
        self.optimizer.update(self.policy, clipped)
        mx.eval(self.policy.parameters(), self.optimizer.state)
        self.updates += 1
        self.pending_gradients = None; self.pending_count = 0; self.pending_chunks = 0
        return float(norm)

    def train_epoch(self, source, *, cancelled: Callable[[], bool] = lambda: False,
                    on_metrics: Callable[[BehaviorMetrics], None] = lambda _: None) -> BehaviorMetrics:
        if self.epoch >= self.config.epochs:
            raise ValueError("The configured behavioral run has already finished")
        self._configure_freeze(); self.policy.train()
        sampler = EpisodeLanes(source, split="train", lanes=self.config.lanes, sequence_length=self.config.sequence_length,
                               seed=self.config.seed, epoch=self.epoch, state=self.sampler_state)
        loss_sum = 0; decisions = 0; gradient_norm = 0.0
        start = time.perf_counter()
        # MLX differentiates explicit parameter pytrees, while recurrent state
        # remains an auxiliary output and is detached between contiguous chunks.
        while True:
            if cancelled():
                # Pending accumulation and sampler/recurrent state are included
                # in checkpoint state. Resume neither drops nor repeats samples.
                raise InterruptedError("Behavioral training cancelled at a complete chunk boundary")
            previous_sampler = sampler.state
            rows = sampler.next()
            if rows is None:
                break
            observation, labels = training_batch(rows, self.policy.config, self.policy.actions.vocabulary)
            validity = observation.valid.reshape(-1)
            def objective(logp, value, entropy):
                return -mx.sum(mx.where(validity, logp, 0)), mx.sum(validity)
            try:
                result = policy_gradients(self.policy, observation, labels, objective, self.recurrent_state, cancelled=cancelled)
            except BaseException:
                self.sampler_state = previous_sampler
                raise
            loss, count, next_state, gradients = result.loss, result.auxiliary, result.next_state, result.gradients
            mx.eval(loss, next_state, count, gradients)
            if not math.isfinite(float(loss)) or not finite_gradients(gradients):
                self.sampler_state = previous_sampler
                raise FloatingPointError("Behavioral batch produced nonfinite loss or gradients; training stopped before update")
            valid_count = int(count)
            self.pending_gradients = gradients if self.pending_gradients is None else tree_map(lambda previous, current: previous + current, self.pending_gradients, gradients)
            mx.eval(self.pending_gradients)
            self.pending_count += valid_count; self.pending_chunks += 1
            self.recurrent_state = tuple(mx.stop_gradient(value) for value in next_state)
            self.sampler_state = sampler.state
            self.decisions += valid_count; decisions += valid_count; loss_sum += float(loss)
            if self.pending_chunks >= self.config.accumulation_chunks:
                gradient_norm = self._update()
            on_metrics(BehaviorMetrics(self.epoch, self.updates, self.decisions, loss_sum / max(decisions, 1), gradient_norm,
                                       decisions / max(time.perf_counter() - start, 1e-6), mx.get_peak_memory()))
        gradient_norm = self._update() or gradient_norm
        if decisions == 0 and self.sampler_state is None:
            raise ValueError("Training requires at least one valid demonstration episode")
        result = BehaviorMetrics(self.epoch, self.updates, self.decisions, loss_sum / max(decisions, 1), gradient_norm,
                                 decisions / max(time.perf_counter() - start, 1e-6), mx.get_peak_memory())
        self.epoch += 1; self.sampler_state = None; self.recurrent_state = None
        return result

    def evaluate(self, source, *, split: str = "validation", cancelled=lambda: False) -> dict:
        self.policy.eval()
        episodes = source.episodes(split)
        if not episodes:
            return {"available": False, "split": split, "reason": "No independent sessions in this split"}
        losses = 0.0; count = 0
        factors = {}
        for episode in episodes:
            state = None
            for start in range(0, episode["steps"], self.config.sequence_length):
                if cancelled():
                    raise InterruptedError("Evaluation cancelled")
                samples = list(source.samples(episode["id"], start, self.config.sequence_length))
                observation, labels = training_batch([samples], self.policy.config, self.policy.actions.vocabulary)
                encoding = self.policy(observation, state)
                output = self.policy.score(encoding, labels)
                mx.eval(output.log_probability, output.factor_log_probabilities, encoding.temporal.state)
                values = np.asarray(output.log_probability)
                if not np.isfinite(values).all():
                    raise FloatingPointError("Evaluation produced nonfinite packet probabilities")
                losses -= float(values.sum()); count += values.size
                for name, value in output.factor_log_probabilities.items():
                    factors[name] = factors.get(name, 0.0) - float(mx.sum(value))
                state = tuple(mx.stop_gradient(value) for value in encoding.temporal.state)
        return {"available": True, "split": split, "decisions": count, "meanNLL": losses / count,
                "factorNLLPerDecision": {name: value / count for name, value in factors.items()}}
