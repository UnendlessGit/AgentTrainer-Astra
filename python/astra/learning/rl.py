"""Auditable rollout boundaries and recurrent PPO objectives.

There is one scalar likelihood per sampled packet: callers must use
PacketDistributionResult.log_probability, never average its component scores.
This module does not schedule actors, carry model state, or apply optimizer
updates. Those operations belong to the trainer and environment coordinator.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from numbers import Real

import mlx.core as mx
import numpy as np


def _identity(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty identity of at most 256 characters")


def _nanos(value: int, name: str) -> None:
    if type(value) is not int or not 0 <= value <= 2**64 - 1:
        raise ValueError(f"{name} must be unsigned 64-bit integer nanoseconds")


def _finite(value: float, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite real number")


class Outcome(StrEnum):
    CONTINUING = "continuing"
    TERMINATED = "terminated"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class RewardWindow:
    """Reward accumulated on the decision interval, not its delayed actions."""

    start_nanos: int
    end_nanos: int
    value: float

    def __post_init__(self) -> None:
        _nanos(self.start_nanos, "Reward start")
        _nanos(self.end_nanos, "Reward end")
        if self.end_nanos <= self.start_nanos:
            raise ValueError("Reward windows must have positive duration")
        _finite(self.value, "Reward")


@dataclass(frozen=True, slots=True)
class BootstrapObservation:
    """Value at the next cutoff using the behavior policy's same-episode state.

    In particular a truncation requires the final pre-reset observation. A
    newly reset observation can never be used as this bootstrap.
    """

    episode_id: str
    policy_id: str
    observation_id: str
    cutoff_nanos: int
    value: float
    recurrent_reset: bool = False

    def __post_init__(self) -> None:
        for name in ("episode_id", "policy_id", "observation_id"):
            _identity(getattr(self, name), name)
        _nanos(self.cutoff_nanos, "Bootstrap cutoff")
        _finite(self.value, "Bootstrap value")
        if self.recurrent_reset is not False:
            raise ValueError("A bootstrap must use same-episode, pre-reset recurrent state")


@dataclass(frozen=True, slots=True)
class Transition:
    run_id: str
    episode_id: str
    policy_id: str
    episode_step: int
    observation_id: str
    packet_id: str
    decision_nanos: int
    next_decision_nanos: int
    reward: RewardWindow | None
    old_log_probability: float | None
    value: float | None
    bootstrap: BootstrapObservation | None
    outcome: Outcome = Outcome.CONTINUING
    recurrent_reset: bool = False
    valid: bool = True
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "episode_id", "policy_id", "observation_id", "packet_id"):
            _identity(getattr(self, name), name)
        if type(self.episode_step) is not int or self.episode_step < 0:
            raise ValueError("Episode step must be a nonnegative integer")
        _nanos(self.decision_nanos, "Decision cutoff")
        _nanos(self.next_decision_nanos, "Next decision cutoff")
        if self.next_decision_nanos <= self.decision_nanos:
            raise ValueError("Transitions must have positive duration")
        if not isinstance(self.outcome, Outcome):
            raise ValueError("Outcome must explicitly distinguish termination, truncation and abort")
        if type(self.valid) is not bool or type(self.recurrent_reset) is not bool:
            raise ValueError("Transition validity and recurrence flags must be Boolean")
        if self.episode_step == 0 and not self.recurrent_reset:
            raise ValueError("Episode zero must reset recurrent state")
        if self.outcome == Outcome.ABORTED and self.valid:
            raise ValueError("Aborted transitions cannot be on-policy training examples")
        if self.valid == (self.invalid_reason is not None):
            raise ValueError("Invalid transitions require a reason; valid transitions cannot have one")
        if self.invalid_reason is not None:
            _identity(self.invalid_reason, "Invalid transition reason")
        if self.reward is not None:
            if not isinstance(self.reward, RewardWindow):
                raise ValueError("Reward must identify its decision window")
            if (self.reward.start_nanos, self.reward.end_nanos) != (self.decision_nanos, self.next_decision_nanos):
                raise ValueError("Rewards must cover [decision, next decision), never the execution window")
        if self.old_log_probability is not None:
            _finite(self.old_log_probability, "Old joint packet log probability")
            if self.old_log_probability > 1e-6:
                raise ValueError("Categorical packet log probabilities cannot be positive")
        if self.value is not None:
            _finite(self.value, "Behavior value")
        if self.valid and (self.reward is None or self.old_log_probability is None or self.value is None):
            raise ValueError("A valid transition needs reward, behavior value and exact old packet likelihood")
        if self.bootstrap is not None:
            if not isinstance(self.bootstrap, BootstrapObservation):
                raise ValueError("Bootstrap must identify its observation and policy")
            if (self.bootstrap.episode_id, self.bootstrap.policy_id, self.bootstrap.cutoff_nanos) != (
                self.episode_id, self.policy_id, self.next_decision_nanos
            ):
                raise ValueError("Bootstrap must belong to the same episode/policy at the next decision cutoff")
        if self.outcome in (Outcome.TERMINATED, Outcome.ABORTED) and self.bootstrap is not None:
            raise ValueError("Terminated and aborted transitions must not bootstrap")
        if self.valid and self.outcome in (Outcome.CONTINUING, Outcome.TRUNCATED) and self.bootstrap is None:
            raise ValueError("Continuing and truncated valid transitions require a pre-reset bootstrap")

    @property
    def duration_seconds(self) -> float:
        return (self.next_decision_nanos - self.decision_nanos) / 1_000_000_000


@dataclass(frozen=True, slots=True)
class Rollout:
    """One sealed behavior-policy stream; observations/packets live in its store.

    initial_state_id refers to the stored behavior state before the first
    observation. It is required when collection starts mid-episode without a
    reset. The trainer must load it and retain the identity for burn-in replay.
    """

    transitions: tuple[Transition, ...]
    initial_state_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transitions, tuple) or not self.transitions:
            raise ValueError("A sealed rollout requires a nonempty immutable tuple of transitions")
        if any(not isinstance(item, Transition) for item in self.transitions):
            raise ValueError("Rollout entries must be validated transitions")
        first = self.transitions[0]
        if self.initial_state_id is not None:
            _identity(self.initial_state_id, "Initial recurrent state")
        if not first.recurrent_reset and self.initial_state_id is None:
            raise ValueError("A rollout starting mid-episode requires its saved behavior state")
        if first.recurrent_reset and self.initial_state_id is not None:
            raise ValueError("A reset rollout must not restore stale behavior state")
        closed_episodes: set[str] = set()
        packets: set[str] = set()
        observations: set[str] = set()
        previous = None
        for item in self.transitions:
            if (item.run_id, item.policy_id) != (first.run_id, first.policy_id):
                raise ValueError("A PPO rollout cannot mix runs or behavior policies")
            if item.packet_id in packets or item.observation_id in observations:
                raise ValueError("Rollout packet and decision-observation identities must be unique")
            packets.add(item.packet_id)
            observations.add(item.observation_id)
            if previous is not None:
                if item.decision_nanos < previous.next_decision_nanos:
                    raise ValueError("Rollout decision windows cannot overlap or run backwards")
                if item.episode_id != previous.episode_id:
                    closed_episodes.add(previous.episode_id)
                    if previous.valid and previous.outcome == Outcome.CONTINUING:
                        raise ValueError("Episode changes require an explicit terminal, truncation or invalid boundary")
                    if item.episode_id in closed_episodes or item.episode_step != 0 or not item.recurrent_reset:
                        raise ValueError("Each new episode needs a fresh identity, step zero and recurrent reset")
                else:
                    if previous.outcome != Outcome.CONTINUING:
                        raise ValueError("A completed or aborted episode cannot continue")
                    if item.episode_step != previous.episode_step + 1:
                        raise ValueError("Episode steps must be contiguous")
                    if item.decision_nanos != previous.next_decision_nanos:
                        raise ValueError("Same-episode decisions need an explicit transition for every gap")
                    if item.recurrent_reset != (not previous.valid):
                        raise ValueError("Recurrence must continue valid trajectories and reset after invalid intervals")
                    bootstrap = previous.bootstrap
                    if previous.valid and bootstrap is not None:
                        if bootstrap.observation_id != item.observation_id:
                            raise ValueError("Adjacent decisions must use the recorded bootstrap observation")
                        if item.value is not None and not math.isclose(bootstrap.value, item.value, rel_tol=1e-6, abs_tol=1e-6):
                            raise ValueError("Adjacent behavior values disagree at the shared observation")
            previous = item

    @property
    def run_id(self) -> str:
        return self.transitions[0].run_id

    @property
    def policy_id(self) -> str:
        return self.transitions[0].policy_id


@dataclass(frozen=True, slots=True)
class ReturnConfig:
    discount_half_life_seconds: float = 30.0
    lambda_per_reference: float = 0.95
    lambda_reference_seconds: float = 0.1

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _finite(getattr(self, name), name)
        if self.discount_half_life_seconds <= 0 or self.lambda_reference_seconds <= 0:
            raise ValueError("Discount and trace reference durations must be positive")
        if not 0 <= self.lambda_per_reference <= 1:
            raise ValueError("GAE lambda must lie in [0, 1]")


def _immutable_array(values, dtype) -> np.ndarray:
    # Immutable bytes backing prevents callers from re-enabling WRITEABLE.
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class AdvantageBatch:
    advantages: np.ndarray
    returns: np.ndarray
    valid: np.ndarray
    discounts: np.ndarray
    trace_decays: np.ndarray


def duration_aware_gae(rollout: Rollout, config: ReturnConfig = ReturnConfig()) -> AdvantageBatch:
    """Compute fixed behavior targets; invalid rows are zero and break traces.

    gamma_i = 2**(-duration_i / half_life)
    lambda_i = lambda_reference**(duration_i / reference_duration)
    delta_i = reward_i + gamma_i * bootstrap_value_i - value_i

    The trace crosses only consecutive valid decisions in one uninterrupted
    episode. Truncation includes its final bootstrap but never a reset-frame
    value or the next episode's advantage. A final continuing row naturally
    bootstraps a collection cutoff without inventing a terminal outcome.
    """
    if not isinstance(rollout, Rollout) or not isinstance(config, ReturnConfig):
        raise ValueError("GAE requires validated rollout and return configuration")
    items = rollout.transitions
    count = len(items)
    advantages = np.zeros(count, dtype=np.float64)
    returns = np.zeros(count, dtype=np.float64)
    valid = np.array([item.valid for item in items], dtype=np.bool_)
    durations = np.array([item.duration_seconds for item in items], dtype=np.float64)
    discounts = np.exp2(-durations / config.discount_half_life_seconds)
    decays = np.power(config.lambda_per_reference, durations / config.lambda_reference_seconds)
    for index in range(count - 1, -1, -1):
        item = items[index]
        if not item.valid:
            continue
        bootstrap_value = 0.0 if item.bootstrap is None else item.bootstrap.value
        delta = item.reward.value + discounts[index] * bootstrap_value - item.value
        if index + 1 < count:
            following = items[index + 1]
            if item.outcome == Outcome.CONTINUING and following.valid and following.episode_id == item.episode_id and not following.recurrent_reset:
                delta += discounts[index] * decays[index] * advantages[index + 1]
        advantages[index] = delta
        returns[index] = delta + item.value
    if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
        raise ValueError("Rollout rewards/values overflowed return computation")
    maximum = np.finfo(np.float32).max
    if np.any(np.abs(advantages) > maximum) or np.any(np.abs(returns) > maximum):
        raise ValueError("Rollout return targets exceed FP32 training range")
    result = AdvantageBatch(*(_immutable_array(value, dtype) for value, dtype in (
        (advantages, np.float32), (returns, np.float32), (valid, np.bool_),
        (discounts, np.float32), (decays, np.float32)
    )))
    return result


def normalize_advantages(advantages: np.ndarray, valid: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    """Normalize once over valid rollout decisions, before minibatch splitting."""
    values, mask = np.asarray(advantages, dtype=np.float64), np.asarray(valid)
    if values.shape != mask.shape or mask.dtype != np.bool_:
        raise ValueError("Advantage normalization requires equally shaped values and Boolean validity")
    _finite(epsilon, "Normalization epsilon")
    if epsilon <= 0 or not mask.any() or not np.isfinite(values[mask]).all():
        raise ValueError("Advantage normalization needs finite valid decisions and positive epsilon")
    selected = values[mask]
    result = np.zeros_like(values)
    result[mask] = (selected - selected.mean()) / max(float(selected.std()), epsilon)
    return _immutable_array(result, np.float32)


@dataclass(frozen=True, slots=True)
class PPOConfig:
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    entropy_scale: float = 1.0
    target_kl: float = 0.02

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _finite(getattr(self, name), name)
        if not 0 < self.clip_ratio < 1 or self.value_coefficient < 0 or self.entropy_coefficient < 0:
            raise ValueError("Invalid PPO clipping or loss coefficients")
        if self.entropy_scale <= 0 or self.target_kl <= 0:
            raise ValueError("Entropy scale and KL guard must be positive")


@dataclass(frozen=True, slots=True)
class PPOLoss:
    total: mx.array
    policy: mx.array
    value: mx.array
    entropy: mx.array
    sampled_kl: mx.array
    signed_sampled_kl: mx.array
    clip_fraction: mx.array
    ratio_mean: mx.array
    ratio_min: mx.array
    ratio_max: mx.array
    valid_count: mx.array
    finite: mx.array
    should_stop: mx.array


def ppo_loss(new_log_probabilities: mx.array, new_values: mx.array, conditional_entropies: mx.array,
             *, old_log_probabilities: mx.array, advantages: mx.array, returns: mx.array,
             valid: mx.array, config: PPOConfig = PPOConfig()) -> PPOLoss:
    """Differentiable FP32 clipped PPO; each tensor entry is a whole decision.

    `conditional_entropies` is the decoder's SUM of active categorical
    entropies at the visited packet prefixes. Its behavior-prefix expectation
    is a surrogate regularizer, not the exact joint entropy of the new policy.
    `entropy_scale` is fixed per run, never divided by sampled packet length.

    MSE is the reported value loss; total weights it by value_coefficient.
    Inputs/targets are masked *before* arithmetic, including exp, so invalid
    rows and recurrent burn-in need not contain finite likelihoods. Targets
    and old likelihoods have stopped gradients. No valid ratio is silently
    clamped for numerical convenience: nonfinite values set should_stop.
    Evaluate should_stop before applying gradients; it also rejects an empty
    valid minibatch. The trainer must reject nonfinite gradients separately.
    """
    if not isinstance(config, PPOConfig):
        raise ValueError("PPO requires a validated loss configuration")
    arrays = (new_log_probabilities, new_values, conditional_entropies, old_log_probabilities, advantages, returns)
    if not isinstance(valid, mx.array) or valid.dtype != mx.bool_ or valid.size == 0:
        raise ValueError("PPO validity must be a nonempty Boolean tensor")
    if any(not isinstance(value, mx.array) or value.shape != valid.shape for value in arrays):
        raise ValueError("PPO tensors must have one matching entry per decision")
    if any(not mx.issubdtype(value.dtype, mx.floating) for value in arrays):
        raise ValueError("PPO likelihoods, values, entropies and targets must use floating-point tensors")
    mask = mx.stop_gradient(valid)
    clean = lambda value: mx.where(mask, value.astype(mx.float32), 0.0)
    new_log, predicted, entropies = (clean(value) for value in arrays[:3])
    old_log, advantage, target = (mx.stop_gradient(clean(value)) for value in arrays[3:])
    count = mx.sum(mask.astype(mx.float32))
    denominator = mx.maximum(count, 1.0)
    mean = lambda value: mx.sum(mx.where(mask, value, 0.0)) / denominator
    log_ratio = new_log - old_log
    ratio = mx.exp(log_ratio)
    clipped_ratio = mx.clip(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
    policy = -mean(mx.minimum(ratio * advantage, clipped_ratio * advantage))
    value_loss = mean(mx.square(predicted - target))
    entropy = mean(entropies)
    total = policy + config.value_coefficient * value_loss - config.entropy_coefficient * config.entropy_scale * entropy
    # E_old[ratio - 1 - log(ratio)] estimates KL(old || new), with each
    # summand nonnegative. expm1 avoids cancellation near ratio one.
    sampled_kl = mean(mx.maximum(mx.expm1(log_ratio) - log_ratio, 0.0))
    signed_kl = -mean(log_ratio)
    clip_fraction = mean((mx.abs(ratio - 1.0) > config.clip_ratio).astype(mx.float32))
    ratio_min = mx.where(count > 0, mx.min(mx.where(mask, ratio, math.inf)), 1.0)
    ratio_max = mx.where(count > 0, mx.max(mx.where(mask, ratio, -math.inf)), 1.0)
    finite = mx.all(mx.stack([mx.all(mx.isfinite(item)) for item in (
        new_log, predicted, entropies, old_log, advantage, target, ratio, total, sampled_kl
    )]))
    should_stop = (count == 0) | (~finite) | (sampled_kl > config.target_kl)
    return PPOLoss(total, policy, value_loss, entropy, sampled_kl, signed_kl, clip_fraction,
                   mean(ratio), ratio_min, ratio_max, count, finite, should_stop)


def verify_behavior_log_probabilities(recorded: np.ndarray, recomputed: np.ndarray,
                                     valid: np.ndarray, *, atol: float = 2e-4, rtol: float = 2e-5) -> float:
    """Reject stale recurrence/augmentation/scoring before the first update.

    Returns maximum absolute disagreement for reporting. Tolerances cover
    normal FP32 accumulation, not a changed behavior policy or packet.
    """
    original, current, mask = np.asarray(recorded), np.asarray(recomputed), np.asarray(valid)
    if original.shape != current.shape or current.shape != mask.shape or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("Behavior replay requires matching tensors and valid decisions")
    for name, tolerance in (("atol", atol), ("rtol", rtol)):
        _finite(tolerance, name)
        if tolerance < 0:
            raise ValueError("Behavior replay tolerances must be nonnegative")
    original, current = original[mask].astype(np.float64), current[mask].astype(np.float64)
    if not np.isfinite(original).all() or not np.isfinite(current).all():
        raise ValueError("Valid behavior packet likelihoods must be finite")
    if (original > 1e-6).any() or (current > 1e-6).any():
        raise ValueError("Categorical packet log probabilities cannot be positive")
    difference = float(np.max(np.abs(original - current)))
    if not np.allclose(current, original, atol=atol, rtol=rtol):
        raise ValueError(f"Behavior packet likelihood replay disagrees (maximum absolute error {difference:.6g})")
    return difference
