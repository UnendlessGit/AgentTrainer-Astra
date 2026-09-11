"""Recurrent PPO over the real policy and an isolated virtual-time environment.

The actor is a private immutable weight snapshot. Learner weights wait for a
confirmed environment reset before activation. This synchronous practice
coordinator is not the asynchronous wall-clock desktop actor coordinator.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Callable
import uuid

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np

from astra.data.actions import decode_commands
from astra.data.batching import stack_observations
from astra.data.observations import make_observation
from astra.environments.practice import PracticeEnvironment, PracticeObservation
from astra.model.actions import PacketBatch, flatten_visual
from astra.model.observation import ObservationBatch
from astra.model.policy import AgentPolicy
from .optimizers import GroupedAdamW, finite_gradients
from .backward import policy_gradients
from .rollout_store import FrameSpool, StoredFrame
from .rl import (BootstrapObservation, Outcome, PPOConfig, ReturnConfig, RewardWindow, Rollout,
                 Transition, duration_aware_gae, normalize_advantages, ppo_loss,
                 verify_behavior_log_probabilities)


@dataclass(frozen=True)
class ReinforcementConfig:
    schema_version: int = 2
    rollout_decisions: int = 512
    epochs: int = 4
    sequence_length: int = 64
    burn_in: int = 32
    effective_batch_decisions: int = 256
    learning_rate: float = 1e-4
    pretrained_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    gradient_norm: float = 0.5
    maximum_kl_backtracks: int = 8
    kl_backtrack_ratio: float = 0.5
    vision_microbatch: int = 1
    checkpoint_vision: bool = True
    state_replay: str = "burn_in"
    seed: int = 0
    maximum_rollout_bytes: int = 4 * 1024**3
    maximum_rollout_disk_bytes: int = 32 * 1024**3
    maximum_rollout_decisions: int = 65536
    ppo: PPOConfig = PPOConfig()
    returns: ReturnConfig = ReturnConfig()

    def validate(self):
        bounds = {"schema_version": (2, 2), "rollout_decisions": (1, 65536), "epochs": (1, 100),
                  "sequence_length": (1, 512), "burn_in": (0, 4096), "effective_batch_decisions": (1, 65536),
                  "vision_microbatch": (1, 64), "seed": (0, 2**63 - 1),
                  "maximum_kl_backtracks": (0, 16),
                  "maximum_rollout_bytes": (1024, 16 * 1024**3),
                  "maximum_rollout_disk_bytes": (1024, 1024**4),
                  "maximum_rollout_decisions": (1, 65536)}
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid reinforcement {name}")
        for name in ("learning_rate", "pretrained_learning_rate", "weight_decay", "gradient_norm"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 10 or (name != "weight_decay" and value == 0):
                raise ValueError(f"Invalid reinforcement {name}")
        if type(self.checkpoint_vision) is not bool or self.state_replay not in ("burn_in", "full_prefix"):
            raise ValueError("Invalid recurrent replay/execution configuration")
        if type(self.kl_backtrack_ratio) not in (int, float) or not math.isfinite(self.kl_backtrack_ratio) or not 0 < self.kl_backtrack_ratio < 1:
            raise ValueError("The KL backtracking ratio must be strictly between zero and one")
        if not isinstance(self.ppo, PPOConfig) or not isinstance(self.returns, ReturnConfig):
            raise ValueError("Reinforcement losses and returns require validated configurations")
        if self.maximum_rollout_decisions < self.rollout_decisions:
            raise ValueError("The complete-episode decision limit cannot precede the rollout minimum")
        return self


def _frozen_array(value, dtype=None):
    value = np.asarray(value, dtype=dtype)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class ObservationRecord:
    """Raw owned pixels and immutable causal metadata; normalized only on use."""
    storage: np.ndarray | StoredFrame
    metadata_json: bytes
    controls_json: bytes
    events_json: bytes
    elapsed_seconds: float
    reset: bool
    last_input_nanos: int | None = None

    @classmethod
    def capture(cls, observation: PracticeObservation, *, events=(), elapsed_seconds: float, reset: bool,
                last_input_nanos: int | None = None, spool: FrameSpool | None = None):
        record = cls(spool.append(observation.pixels) if spool is not None else _frozen_array(observation.pixels),
                     _json(observation.metadata), _json(observation.control_state),
                     _json(events), elapsed_seconds, reset, last_input_nanos)
        if spool is not None:
            spool.reserve_metadata(record.metadata_byte_count)
        return record

    @property
    def pixels(self):
        return self.storage.read() if isinstance(self.storage, StoredFrame) else self.storage

    @property
    def metadata(self):
        return json.loads(self.metadata_json)

    @property
    def byte_count(self):
        return self.storage.nbytes + self.metadata_byte_count

    @property
    def metadata_byte_count(self):
        return len(self.metadata_json) + len(self.controls_json) + len(self.events_json) + 512

    def prepare(self, model, context_ids) -> ObservationBatch:
        metadata = self.metadata
        return make_observation([(self.pixels, metadata)], json.loads(self.controls_json),
                                cutoff_nanos=metadata["observedNanos"], elapsed_seconds=self.elapsed_seconds,
                                reset=self.reset, config=model, context_ids=context_ids,
                                executed_events=json.loads(self.events_json), last_input_nanos=self.last_input_nanos)


@dataclass(frozen=True)
class CollectedDecision:
    transition: Transition
    observation: ObservationRecord
    packet_fields: tuple[tuple[int, ...], ...]
    state_before: tuple[np.ndarray, ...]
    bootstrap_observation: ObservationRecord | None

    def packet(self):
        return PacketBatch(**{name: mx.array(values, dtype=mx.int32)[None]
                              for name, values in zip(PacketBatch.__dataclass_fields__, self.packet_fields, strict=True)})


@dataclass(frozen=True)
class CollectedRollout:
    id: str
    rollout: Rollout
    decisions: tuple[CollectedDecision, ...]
    model_signature: str
    environment_signature: str
    context_ids: tuple[int, ...]
    raw_bytes: int
    collection_seconds: float
    audit_tail_decisions: int
    spool: FrameSpool

    def __post_init__(self):
        if not isinstance(self.decisions, tuple) or tuple(item.transition for item in self.decisions) != self.rollout.transitions:
            raise ValueError("Rollout payloads must match their immutable transition records")

    def close(self):
        self.spool.close()


@dataclass(frozen=True)
class ReinforcementMetrics:
    iteration: int
    decisions: int
    optimizer_updates: int
    completed_epochs: int
    mean_reward: float
    terminated_episodes: int
    truncated_episodes: int
    mean_loss: float
    mean_policy_loss: float
    mean_value_loss: float
    mean_entropy_surrogate: float
    maximum_gradient_norm: float
    maximum_sampled_kl: float
    maximum_candidate_kl: float | None
    maximum_accepted_kl: float
    backtrack_count: int
    minimum_step_scale: float
    rejected_optimizer_steps: int
    ratio_min: float
    ratio_max: float
    clip_fraction: float
    behavior_replay_error: float
    behavior_replay_ratio_min: float
    behavior_replay_ratio_max: float
    stopped_for_kl: bool
    decisions_per_second: float
    peak_memory_bytes: int
    actor_policy_id: str
    pending_policy_id: str | None
    audit_tail_decisions: int
    rollout_disk_bytes: int
    rollout_peak_cache_bytes: int


@dataclass(frozen=True)
class IterationResult:
    metrics: ReinforcementMetrics
    checkpoint_state: dict


@dataclass(frozen=True)
class PolicyShift:
    """Exact recurrent replay, sampled whole-packet divergence on the rollout.

    This bounds only the sealed sample, not unobserved actions or environments.
    Nonfinite candidate scores are rejected by admission, never published.
    """
    sampled_kl: float
    ratio_min: float
    ratio_max: float
    clip_fraction: float
    finite: bool


def _detached_state(state):
    result = tuple(mx.stop_gradient(value) for value in state)
    mx.eval(result)
    return result


def _finite_tree(values):
    arrays = [value for _, value in tree_flatten(values) if isinstance(value, mx.array)]
    return bool(mx.all(mx.stack([mx.all(mx.isfinite(value)) for value in arrays])).item()) if arrays else True


class ReinforcementTrainer:
    """One practice actor, one learner, and at most one pending weight snapshot.

    The caller's `policy` is the learner/checkpoint policy. It must not be
    mutated concurrently. Actor and pending modules are private copies.
    Checkpoint state resumes learning at a fresh world reset, never restores
    a live world from recurrent state alone.
    """
    def __init__(self, policy: AgentPolicy, environment: PracticeEnvironment,
                 config: ReinforcementConfig = ReinforcementConfig(), *, policy_id: str | None = None,
                 context_ids: tuple[int, ...] = (), restored_state: dict | None = None,
                 scratch_directory: Path | None = None):
        if not isinstance(policy, AgentPolicy) or not isinstance(environment, PracticeEnvironment):
            raise ValueError("This collector requires AgentPolicy and an isolated PracticeEnvironment")
        self.config = config.validate()
        self.policy, self.environment = policy, environment
        if (policy.config.period_ms, policy.config.lead_ms) != (environment.config.period_ms, environment.config.lead_ms):
            raise ValueError("Actor and environment must share immutable period/lead timing")
        if policy.actions.vocabulary != environment.action_vocabulary:
            raise ValueError("Policy and practice action vocabularies must agree")
        if not math.isclose(config.returns.discount_half_life_seconds * 1000, environment.config.discount_half_life_ms):
            raise ValueError("Potential shaping and PPO must use the same discount half-life")
        if type(context_ids) is not tuple or len(context_ids) != len(policy.config.context_sizes) or any(
            type(value) is not int or not 0 <= value < size for value, size in zip(context_ids, policy.config.context_sizes)
        ):
            raise ValueError("Actor contexts must match the policy's immutable vocabulary")
        self.context_ids = context_ids
        self.optimizer = GroupedAdamW(learning_rate=config.learning_rate,
                                     pretrained_learning_rate=config.pretrained_learning_rate, weight_decay=config.weight_decay)
        self.iteration = 0
        self.decisions = 0
        self.optimizer_updates = 0
        self.environment_resets = 0
        self._rng = mx.random.key(config.seed)
        self._policy_id = policy_id or str(uuid.uuid4())
        if not isinstance(self._policy_id, str) or not self._policy_id.strip() or len(self._policy_id) > 256:
            raise ValueError("Behavior policy identity must be a bounded nonempty string")
        if restored_state is not None:
            self._restore(restored_state)
        self._run_id = str(uuid.uuid4())
        self._actor = copy.deepcopy(policy)
        self._actor.eval()
        self._actor.configure_execution(vision_microbatch=config.vision_microbatch, checkpoint_vision=False)
        self.policy.configure_execution(vision_microbatch=config.vision_microbatch, checkpoint_vision=config.checkpoint_vision)
        self._pending = None
        self._current = None
        self._last_input_nanos = None
        self._encoding = None
        self._state = None
        self._boundary = True
        self._step = 0
        self._episode_id = None
        self._outstanding_rollout_id = None
        self._busy = False
        self._spool = None
        self._scratch_directory = scratch_directory

    def _restore(self, state):
        expected = {"kind", "schemaVersion", "config", "environmentSignature", "modelSignature", "contextIDs",
                    "iteration", "decisions", "optimizerUpdates", "environmentResets", "rng", "optimizer", "policyID", "requiresEnvironmentReset"}
        if not isinstance(state, dict) or set(state) != expected or state["kind"] != "reinforcement" or state["schemaVersion"] != 2:
            raise ValueError("Unsupported reinforcement checkpoint state")
        if state["config"] != asdict(self.config) or state["environmentSignature"] != self.environment.config.signature or state["modelSignature"] != self.policy.config.signature or tuple(state["contextIDs"]) != self.context_ids:
            raise ValueError("Reinforcement resume requires the same model, environment and training configuration")
        if state["requiresEnvironmentReset"] is not True:
            raise ValueError("A checkpoint cannot restore a live practice world implicitly")
        for name in ("iteration", "decisions", "optimizerUpdates", "environmentResets"):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError("Invalid saved reinforcement progress")
        rng = state["rng"]
        if not isinstance(rng, mx.array) or rng.shape != (2,) or rng.dtype != mx.uint32:
            raise ValueError("Invalid saved actor random state")
        if not isinstance(state["policyID"], str) or not state["policyID"].strip() or len(state["policyID"]) > 256:
            raise ValueError("Invalid saved policy identity")
        self.iteration, self.optimizer_updates = state["iteration"], state["optimizerUpdates"]
        self.decisions = state["decisions"]
        self.environment_resets = state["environmentResets"]
        self._rng = mx.array(rng)
        self.optimizer.state = copy.deepcopy(state["optimizer"])
        self._policy_id = state["policyID"]

    @property
    def actor_policy_id(self):
        return self._policy_id

    @property
    def pending_policy_id(self):
        return None if self._pending is None else self._pending[0]

    @property
    def at_episode_boundary(self):
        return self._boundary

    @property
    def state(self):
        if self._busy or self._outstanding_rollout_id is not None:
            raise RuntimeError("Checkpoint reinforcement state only after completing or discarding the current rollout")
        return {"kind": "reinforcement", "schemaVersion": 2, "config": asdict(self.config),
                "environmentSignature": self.environment.config.signature, "modelSignature": self.policy.config.signature,
                "contextIDs": self.context_ids, "iteration": self.iteration, "decisions": self.decisions,
                "optimizerUpdates": self.optimizer_updates,
                "environmentResets": self.environment_resets, "rng": mx.array(self._rng),
                "optimizer": copy.deepcopy(self.optimizer.state),
                "policyID": self.pending_policy_id or self.actor_policy_id, "requiresEnvironmentReset": True}

    def activate_pending_at_reset(self):
        """Confirm/reset the environment before exposing pending actor weights."""
        if self._busy:
            raise RuntimeError("Cannot reset an actor during an active collection/update")
        self._reset()

    def _reset(self):
        if not self._boundary:
            raise RuntimeError("Pending policy activation requires a confirmed episode boundary")
        if self._outstanding_rollout_id is not None:
            raise RuntimeError("Finish or discard the sealed rollout before resetting the actor")
        observed = self.environment.reset(seed=(self.environment.config.seed + self.environment_resets) % 2**63)
        if observed.episode_id == self._episode_id or not observed.control_state["valid"] or observed.control_state["keys"] or observed.control_state["buttons"]:
            raise RuntimeError("Environment reset failed readiness/identity confirmation")
        self.environment_resets += 1
        self._last_input_nanos = None
        if self._pending is not None:
            self._policy_id, self._actor = self._pending
            self._pending = None
        self._episode_id = observed.episode_id
        self._step = 0
        self._state = None
        self._encoding = None
        self._current = ObservationRecord.capture(observed, elapsed_seconds=self.policy.config.period_ms / 1000,
                                                 reset=True, spool=self._spool)
        self._boundary = False

    def stop(self, reason="Practice learning stopped"):
        """Administrative cancellation has no fabricated zero-duration PPO row."""
        if self._busy:
            raise RuntimeError("Use the cancellation callback while a collection/update is active")
        self._stop(reason)

    def _stop(self, reason):
        try:
            if self.environment.outcome == "continuing":
                self.environment.abort(reason)
        finally:
            self._boundary = self.environment.outcome != "continuing"
            self._current = None
            self._encoding = None
            self._state = None
            self._outstanding_rollout_id = None
            if self._spool is not None:
                spool, self._spool = self._spool, None
                spool.close()

    def _stop_after_failure(self, failure, reason):
        try:
            self._stop(reason)
        except BaseException as cleanup_error:
            failure.add_note(f"Practice cleanup also failed ({type(cleanup_error).__name__}).")

    def discard_rollout(self, rollout: CollectedRollout):
        if self._busy or self._outstanding_rollout_id != rollout.id:
            raise ValueError("Only the currently sealed rollout can be discarded")
        rollout.close()
        self._spool = None
        self._outstanding_rollout_id = None

    def _decision(self) -> CollectedDecision:
        if self._boundary or self._current is None:
            raise RuntimeError("A decision requires a confirmed running environment")
        current = self._current
        metadata = current.metadata
        if not json.loads(current.controls_json)["valid"]:
            raise RuntimeError("Actor controls are unavailable; no policy packet may be admitted")
        before = self._state if self._state is not None else self._actor.temporal.initial_state(1)
        state_record = tuple(_frozen_array(value, np.float32) for value in before)
        if self._encoding is None:
            self._encoding = self._actor(current.prepare(self.policy.config, self.context_ids), before)
        encoding = self._encoding
        self._rng, key = tuple(mx.random.split(self._rng))
        sampled = self._actor.sample(encoding, key=key)
        mx.eval(sampled.log_probability, sampled.conditional_entropy, encoding.temporal.value,
                encoding.temporal.state, *(getattr(sampled.packets, name) for name in PacketBatch.__dataclass_fields__))
        old_log, value = float(sampled.log_probability[0]), float(encoding.temporal.value[0, 0])
        if not math.isfinite(old_log) or not math.isfinite(value):
            raise FloatingPointError("The actor produced nonfinite behavior likelihood/value")
        commands = decode_commands(sampled.packets, config=self.policy.config, vocabulary=self._actor.actions.vocabulary,
                                   visual=flatten_visual(encoding.visual), surfaces=(metadata["surface"],))
        packet_fields = tuple(tuple(int(value) for value in np.asarray(getattr(sampled.packets, name)[0]))
                              for name in PacketBatch.__dataclass_fields__)
        result = self.environment.step(commands, episode_id=self._episode_id, provenance="agent")
        self._boundary = result.outcome != "continuing"
        if result.provenance != "agent" or result.duration_ms <= 0 or result.observation.episode_id != self._episode_id:
            raise RuntimeError("Environment returned invalid or non-agent transition provenance")
        if any(item["status"] in ("failed", "late", "rejected") for item in result.command_results):
            raise RuntimeError("Environment execution failed; this rollout cannot enter PPO")
        end = result.observation.metadata["observedNanos"]
        if end - metadata["observedNanos"] != result.duration_ms * 1_000_000:
            raise RuntimeError("Environment duration disagrees with its decision clock")
        input_times = [event["observedNanos"] for event in result.raw_events if event["origin"] in ("physical", "agent")]
        if input_times:
            self._last_input_nanos = max(self._last_input_nanos or 0, max(input_times))
        next_observation = ObservationRecord.capture(result.observation, events=result.raw_events,
                                                     elapsed_seconds=result.duration_ms / 1000, reset=False,
                                                     last_input_nanos=self._last_input_nanos, spool=self._spool)
        self._state = _detached_state(encoding.temporal.state)
        outcome = Outcome(result.outcome)
        bootstrap = None
        self._encoding = None
        if outcome in (Outcome.CONTINUING, Outcome.TRUNCATED):
            self._encoding = self._actor(next_observation.prepare(self.policy.config, self.context_ids), self._state)
            mx.eval(self._encoding.temporal.value, self._encoding.temporal.state)
            bootstrap = BootstrapObservation(self._episode_id, self._policy_id,
                                             result.observation.metadata["id"], end, float(self._encoding.temporal.value[0, 0]))
        transition = Transition(self._run_id, self._episode_id, self._policy_id, self._step, metadata["id"], str(uuid.uuid4()),
                                metadata["observedNanos"], end, RewardWindow(metadata["observedNanos"], end, result.reward),
                                old_log, value, bootstrap, outcome, current.reset,
                                valid=outcome != Outcome.ABORTED, invalid_reason=result.reason if outcome == Outcome.ABORTED else None)
        self._current = next_observation
        self._step += 1
        self._boundary = outcome != Outcome.CONTINUING
        if self._boundary:
            self._encoding = None
            self._state = None
        return CollectedDecision(transition, current, packet_fields, state_record,
                                 next_observation if bootstrap is not None else None)

    def advance_to_reset(self, *, cancelled: Callable[[], bool] = lambda: False,
                         on_decision: Callable[[int], None] = lambda _: None) -> int:
        """Run the frozen actor to a real boundary; audit tail is not PPO data."""
        if self._busy:
            raise RuntimeError("Cannot independently advance an actor during collection/update")
        return self._advance_to_reset(cancelled=cancelled, on_decision=on_decision)

    def _advance_to_reset(self, *, cancelled, on_decision=lambda _: None):
        if self._outstanding_rollout_id is not None:
            raise RuntimeError("Finish the sealed rollout before advancing its actor")
        count = 0
        maximum = math.ceil(self.environment.config.time_limit_ms / self.environment.config.period_ms) + 1
        try:
            while not self._boundary:
                if cancelled():
                    raise InterruptedError("Reinforcement learning cancelled while waiting for reset")
                self._decision()
                count += 1
                on_decision(count)
                if count > maximum:
                    raise RuntimeError("Practice actor exceeded its configured episode time limit")
        except BaseException as failure:
            self._stop_after_failure(failure, "Practice reset wait interrupted")
            raise
        return count

    def collect(self, *, cancelled: Callable[[], bool] = lambda: False,
                on_decision: Callable[[int], None] = lambda _: None,
                on_finishing_episode: Callable[[int], None] = lambda _: None) -> CollectedRollout:
        """Reach the minimum size, then finish the episode without losing its tail."""
        if self._busy or self._outstanding_rollout_id is not None:
            raise RuntimeError("Only one collection or sealed rollout may be outstanding")
        self._busy = True
        started = time.perf_counter()
        decisions = []
        try:
            if cancelled():
                raise InterruptedError("Reinforcement rollout collection cancelled")
            frame_bytes = self.environment.config.pixel_width * self.environment.config.pixel_height * 4
            if frame_bytes > self.config.maximum_rollout_bytes:
                raise MemoryError("One practice frame exceeds the rollout RAM budget")
            self._spool = FrameSpool(memory_bytes=self.config.maximum_rollout_bytes,
                                     disk_bytes=self.config.maximum_rollout_disk_bytes,
                                     directory=self._scratch_directory)
            if self._boundary:
                self._reset()
            elif self._step == 0 and self._current is not None and self._current.reset:
                # An explicit activate_pending_at_reset() can prepare episode
                # zero before collection has acquired its temporary storage.
                self._spool.reserve_metadata(self._current.metadata_byte_count)
                self._current = replace(self._current, storage=self._spool.append(self._current.pixels))
            else:
                raise RuntimeError("Complete-episode collection must start at confirmed episode zero")
            while len(decisions) < self.config.rollout_decisions or not self._boundary:
                if cancelled():
                    raise InterruptedError("Reinforcement rollout collection cancelled")
                if len(decisions) >= self.config.maximum_rollout_decisions:
                    raise MemoryError("Episode did not finish within the configured rollout decision limit")
                if self._boundary:
                    self._reset()
                item = self._decision()
                decisions.append(item)
                self._spool.reserve_metadata(sum(value.nbytes for value in item.state_before)
                    + sum(len(field) * 4 for field in item.packet_fields) + 512)
                if len(decisions) >= self.config.rollout_decisions and not self._boundary:
                    on_finishing_episode(len(decisions))
                else:
                    on_decision(len(decisions))
            self._spool.seal()
            rollout = Rollout(tuple(item.transition for item in decisions))
            identifier = str(uuid.uuid4())
            self._outstanding_rollout_id = identifier
            return CollectedRollout(identifier, rollout, tuple(decisions), self.policy.config.signature,
                                    self.environment.config.signature, self.context_ids,
                                    self._spool.disk_bytes + self._spool.metadata_bytes,
                                    time.perf_counter() - started, 0, self._spool)
        except BaseException as failure:
            self._stop_after_failure(failure, "Practice collection interrupted")
            raise
        finally:
            self._busy = False

    def _batch(self, decisions):
        observations = []
        for item in decisions:
            value = item.observation.prepare(self.policy.config, self.context_ids)
            observations.append(replace(value, valid=value.valid & item.transition.valid))
        observation = stack_observations([observations])
        packets = PacketBatch(**{name: mx.array([item.packet_fields[index] for item in decisions], dtype=mx.int32)
                                  for index, name in enumerate(PacketBatch.__dataclass_fields__)})
        return observation, packets

    def verify_behavior(self, collected: CollectedRollout, *, cancelled=lambda: False):
        """Exact old-policy recurrent replay from episode zero, before updates."""
        state = None
        errors, ratios = [], []
        self.policy.eval()
        for start in range(0, len(collected.decisions), self.config.sequence_length):
            if cancelled():
                raise InterruptedError("Behavior-policy replay cancelled")
            items = collected.decisions[start:start + self.config.sequence_length]
            observation, packets = self._batch(items)
            encoding = self.policy(observation, state)
            scores = self.policy.score(encoding, packets)
            mx.eval(scores.log_probability, encoding.temporal.value, encoding.temporal.state)
            old = np.array([item.transition.old_log_probability for item in items], np.float32)
            valid = np.array([item.transition.valid for item in items], bool)
            actual = np.asarray(scores.log_probability)
            errors.append(verify_behavior_log_probabilities(old, actual, valid))
            recorded_values = np.array([item.transition.value for item in items], np.float32)
            if not np.allclose(np.asarray(encoding.temporal.value).reshape(-1)[valid], recorded_values[valid], atol=2e-5, rtol=2e-5):
                raise ValueError("Behavior value replay disagrees before the first PPO update")
            ratios.extend(np.exp(actual[valid].astype(np.float64) - old[valid]).tolist())
            state = _detached_state(encoding.temporal.state)
        return max(errors), min(ratios), max(ratios)

    def _chunks(self, collected):
        chunks = []
        episode_start = 0
        for index, item in enumerate(collected.decisions):
            if item.observation.reset:
                episode_start = index
            if index == episode_start or (index - episode_start) % self.config.sequence_length == 0:
                stop = min(index + self.config.sequence_length, len(collected.decisions))
                for candidate in range(index + 1, stop):
                    if collected.decisions[candidate].observation.reset:
                        stop = candidate
                        break
                if any(item.transition.valid for item in collected.decisions[index:stop]):
                    chunks.append((episode_start, index, stop))
        return chunks

    def replay_state(self, collected, episode_start, start, *, mode=None, cancelled=lambda: False):
        """Replay a detached prefix; burn-in uses a saved behavior-state anchor."""
        mode = self.config.state_replay if mode is None else mode
        if mode not in ("burn_in", "full_prefix") or not 0 <= episode_start <= start < len(collected.decisions):
            raise ValueError("Invalid recurrent replay prefix")
        first = episode_start if mode == "full_prefix" else max(episode_start, start - self.config.burn_in)
        state = None if collected.decisions[first].observation.reset else tuple(mx.array(value) for value in collected.decisions[first].state_before)
        for cursor in range(first, start, self.config.sequence_length):
            if cancelled():
                raise InterruptedError("Recurrent prefix replay cancelled")
            items = collected.decisions[cursor:min(cursor + self.config.sequence_length, start)]
            observation = stack_observations([[item.observation.prepare(self.policy.config, self.context_ids) for item in items]])
            encoding = self.policy(observation, state)
            state = _detached_state(encoding.temporal.state)
        return state

    def evaluate_policy_shift(self, collected: CollectedRollout, *, cancelled=lambda: False,
                              on_decision=lambda _: None) -> PolicyShift:
        """Score the sealed sample from episode zero using current weights.

        Carry exact current-policy recurrence through bounded forward chunks.
        Saved behavior-state anchors and the training burn-in approximation are
        deliberately absent from this policy-admission measurement.
        """
        state = None
        count = clipped = 0
        total = 0.0
        low, high = math.inf, -math.inf
        for start in range(0, len(collected.decisions), self.config.sequence_length):
            if cancelled():
                raise InterruptedError("Candidate-policy validation cancelled")
            items = collected.decisions[start:start + self.config.sequence_length]
            observation, packets = self._batch(items)
            encoding = self.policy(observation, state)
            scores = self.policy.score(encoding, packets)
            mx.eval(scores.log_probability, encoding.temporal.value, encoding.temporal.state)
            state = _detached_state(encoding.temporal.state)
            valid = np.array([item.transition.valid for item in items], bool)
            values = np.asarray(encoding.temporal.value).reshape(-1)[valid]
            if not np.isfinite(values).all() or not _finite_tree(state):
                return PolicyShift(math.inf, 0.0, math.inf, 1.0, False)
            original = np.array([item.transition.old_log_probability or 0 for item in items], np.float64)
            delta = np.asarray(scores.log_probability, np.float64)[valid] - original[valid]
            with np.errstate(over="ignore", invalid="ignore"):
                ratios = np.exp(delta)
                divergences = np.maximum(np.expm1(delta) - delta, 0.0)
            if not np.isfinite(divergences).all() or not np.isfinite(ratios).all():
                return PolicyShift(math.inf, 0.0, math.inf, 1.0, False)
            count += len(delta)
            total += float(divergences.sum())
            if not math.isfinite(total):
                return PolicyShift(math.inf, 0.0, math.inf, 1.0, False)
            if len(delta):
                low, high = min(low, float(ratios.min())), max(high, float(ratios.max()))
                clipped += int((np.abs(ratios - 1.0) > self.config.ppo.clip_ratio).sum())
            on_decision(start + len(items))
        if not count:
            raise ValueError("Candidate policy admission requires valid sampled decisions")
        return PolicyShift(total / count, low, high, clipped / count, True)

    def update(self, collected: CollectedRollout, *, cancelled: Callable[[], bool] = lambda: False,
               on_update: Callable[[int], None] = lambda _: None,
               on_validation: Callable[[dict], None] = lambda _: None) -> ReinforcementMetrics:
        if self._busy or self._outstanding_rollout_id != collected.id or self._pending is not None:
            raise RuntimeError("PPO needs the one outstanding rollout and no pending policy update")
        if collected.rollout.policy_id != self.actor_policy_id or collected.rollout.run_id != self._run_id or collected.model_signature != self.policy.config.signature or collected.environment_signature != self.environment.config.signature or collected.context_ids != self.context_ids:
            raise ValueError("Rollout identity/configuration does not match this actor and learner")
        started = time.perf_counter()
        previous_weights = copy.deepcopy(self.policy.parameters())
        previous_optimizer = copy.deepcopy(self.optimizer.state)
        previous_updates = self.optimizer_updates
        self._busy = True
        try:
            replay_error, replay_min, replay_max = self.verify_behavior(collected, cancelled=cancelled)
            returns = duration_aware_gae(collected.rollout, self.config.returns)
            advantages = normalize_advantages(returns.advantages, returns.valid)
            old = np.array([item.transition.old_log_probability or 0 for item in collected.decisions], np.float32)
            chunks = self._chunks(collected)
            totals = np.zeros(6, np.float64)  # total, policy, value, entropy, KL, clipped fraction, all weighted
            total_count = 0
            maximum_norm = maximum_kl = 0.0
            maximum_candidate_kl = 0.0
            nonfinite_candidate = False
            backtrack_count = rejected_steps = 0
            minimum_step_scale = 1.0
            accepted_shift = PolicyShift(0.0, 1.0, 1.0, 0.0, True)
            ratio_min, ratio_max = math.inf, -math.inf
            stopped = False
            completed_epochs = 0
            # Evaluation mode preserves the behavior policy's deterministic
            # scoring path; MLX value_and_grad still differentiates parameters.
            self.policy.eval()

            for epoch in range(self.config.epochs):
                order = list(chunks)
                np.random.default_rng(np.random.SeedSequence([self.config.seed, self.iteration, epoch])).shuffle(order)
                accumulated = None
                count = 0
                batch_metrics = np.zeros(6, np.float64)
                for chunk_number, (episode_start, start, end) in enumerate(order):
                    if cancelled():
                        raise InterruptedError("Recurrent PPO update cancelled at a chunk boundary")
                    state = self.replay_state(collected, episode_start, start, cancelled=cancelled)
                    observation, packets = self._batch(collected.decisions[start:end])
                    validity = returns.valid[start:end]
                    valid_count = int(validity.sum())
                    if not valid_count:
                        continue
                    def objective(logp, value, entropy):
                        result = ppo_loss(logp, value, entropy, old_log_probabilities=mx.array(old[start:end]),
                                          advantages=mx.array(advantages[start:end]), returns=mx.array(returns.returns[start:end]),
                                          valid=mx.array(validity), config=self.config.ppo)
                        metrics = mx.stack((result.total, result.policy, result.value, result.entropy,
                                            result.sampled_kl, result.clip_fraction)) * result.valid_count
                        return result.total * result.valid_count, (metrics, result.ratio_min, result.ratio_max, result.finite)
                    result = policy_gradients(self.policy, observation, packets, objective, state, cancelled=cancelled)
                    loss, (diagnostics, low, high, finite), gradients = result.loss, result.auxiliary, result.gradients
                    mx.eval(loss, diagnostics, gradients, low, high, finite)
                    if not bool(finite) or not finite_gradients(gradients):
                        raise FloatingPointError("PPO produced nonfinite loss or gradients before optimizer admission")
                    accumulated = gradients if accumulated is None else tree_map(lambda before, after: before + after, accumulated, gradients)
                    mx.eval(accumulated)
                    count += valid_count
                    batch_metrics += np.asarray(diagnostics, np.float64)
                    if count < self.config.effective_batch_decisions and chunk_number + 1 < len(order):
                        continue
                    sampled_kl = batch_metrics[4] / count
                    if sampled_kl > self.config.ppo.target_kl:
                        stopped = True
                        break
                    gradients = tree_map(lambda value: value / count, accumulated)
                    clipped, norm = optim.clip_grad_norm(gradients, self.config.gradient_norm)
                    mx.eval(clipped, norm)
                    if not math.isfinite(float(norm)) or not finite_gradients(clipped):
                        raise FloatingPointError("PPO clipping produced nonfinite gradients")
                    step_weights = copy.deepcopy(self.policy.parameters())
                    step_optimizer = copy.deepcopy(self.optimizer.state)
                    self.optimizer.update(self.policy, clipped)
                    mx.eval(self.policy.parameters(), self.optimizer.state)
                    if not _finite_tree(self.policy.parameters()) or not _finite_tree(self.optimizer.state):
                        raise FloatingPointError("PPO optimizer produced nonfinite parameters/state")
                    proposed_weights = copy.deepcopy(self.policy.parameters())
                    admitted = False
                    # Moments are computed once. Scaling the entire AdamW
                    # parameter delta is a per-step learning-rate multiplier,
                    # including decay; trials must not advance moment clocks.
                    for attempt in range(self.config.maximum_kl_backtracks + 1):
                        scale = self.config.kl_backtrack_ratio ** attempt
                        if attempt:
                            backtrack_count += 1
                            self.policy.update(tree_map(lambda before, after: before + scale * (after - before),
                                                        step_weights, proposed_weights))
                            mx.eval(self.policy.parameters())
                        def validation_progress(decisions):
                            on_validation({"candidate_index": attempt, "candidate_step_scale": scale,
                                           "validation_decisions": decisions,
                                           "validation_target": len(collected.decisions)})
                        validation_progress(0)
                        shift = self.evaluate_policy_shift(collected, cancelled=cancelled,
                                                           on_decision=validation_progress)
                        if shift.finite:
                            maximum_candidate_kl = max(maximum_candidate_kl, shift.sampled_kl)
                        else:
                            nonfinite_candidate = True
                        if cancelled():
                            raise InterruptedError("PPO cancelled before candidate admission")
                        if shift.finite and shift.sampled_kl <= self.config.ppo.target_kl:
                            accepted_shift = shift
                            maximum_kl = max(maximum_kl, shift.sampled_kl)
                            ratio_min = min(ratio_min, shift.ratio_min)
                            ratio_max = max(ratio_max, shift.ratio_max)
                            minimum_step_scale = min(minimum_step_scale, scale)
                            admitted = True
                            break
                    if not admitted:
                        self.policy.update(step_weights)
                        self.optimizer.state = step_optimizer
                        mx.eval(self.policy.parameters(), self.optimizer.state)
                        rejected_steps += 1
                        stopped = True
                        break
                    del step_weights, proposed_weights, step_optimizer
                    self.optimizer_updates += 1
                    maximum_norm = max(maximum_norm, float(norm))
                    totals += batch_metrics
                    total_count += count
                    accumulated, count, batch_metrics = None, 0, np.zeros(6, np.float64)
                    on_update(self.optimizer_updates)
                if stopped:
                    break
                completed_epochs += 1
            updates = self.optimizer_updates - previous_updates
            if updates:
                pending = copy.deepcopy(self.policy)
                pending.eval()
                pending.configure_execution(vision_microbatch=self.config.vision_microbatch, checkpoint_vision=False)
                self._pending = (str(uuid.uuid4()), pending)
            mean = totals / max(total_count, 1)
            items = collected.rollout.transitions
            metrics = ReinforcementMetrics(self.iteration + 1, int(returns.valid.sum()), updates, completed_epochs,
                                         float(np.mean([item.reward.value for item in items if item.valid])),
                                         sum(item.outcome == Outcome.TERMINATED for item in items),
                                         sum(item.outcome == Outcome.TRUNCATED for item in items),
                                         *map(float, mean[:4]), maximum_norm, float(maximum_kl),
                                         None if nonfinite_candidate else float(maximum_candidate_kl),
                                         float(maximum_kl), backtrack_count, minimum_step_scale, rejected_steps,
                                         ratio_min if math.isfinite(ratio_min) else 1.0,
                                         ratio_max if math.isfinite(ratio_max) else 1.0, accepted_shift.clip_fraction,
                                         replay_error, replay_min, replay_max, stopped,
                                         int(returns.valid.sum()) / max(time.perf_counter() - started, 1e-6),
                                         mx.get_peak_memory(), self.actor_policy_id, self.pending_policy_id,
                                         0, collected.spool.disk_bytes, collected.spool.peak_memory_bytes)
            collected.close()
            self._spool = None
            self.iteration += 1
            self.decisions += int(returns.valid.sum())
            self._outstanding_rollout_id = None
            return metrics
        except BaseException:
            self.policy.update(previous_weights)
            self.optimizer.state = previous_optimizer
            self.optimizer_updates = previous_updates
            self._pending = None
            mx.eval(self.policy.parameters(), self.optimizer.state)
            # The sealed rollout remains available for a deliberate retry or
            # discard; no partially updated learner can be published as final.
            raise
        finally:
            self._busy = False

    def run_iteration(self, *, cancelled=lambda: False, on_decision=lambda _: None, on_update=lambda _: None,
                      on_validation=lambda _: None, on_finishing_episode=lambda _: None):
        try:
            collected = self.collect(cancelled=cancelled, on_decision=on_decision,
                                     on_finishing_episode=on_finishing_episode)
            metrics = self.update(collected, cancelled=cancelled, on_update=on_update,
                                  on_validation=on_validation)
            return IterationResult(metrics, self.state)
        except InterruptedError as failure:
            self._stop_after_failure(failure, "Practice learning cancelled")
            raise
