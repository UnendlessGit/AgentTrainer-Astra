from dataclasses import FrozenInstanceError, replace
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
import numpy as np
import pytest

from astra.learning.rl import (
    BootstrapObservation, Outcome, PPOConfig, ReturnConfig, RewardWindow, Rollout,
    Transition, duration_aware_gae, normalize_advantages, ppo_loss,
    verify_behavior_log_probabilities,
)


def transition(step=0, *, episode="episode-a", start=None, duration=100_000_000,
               reward=1.0, value=0.5, next_value=0.75, outcome=Outcome.CONTINUING,
               valid=True, reset=None):
    start = step * 100_000_000 if start is None else start
    end = start + duration
    return Transition(
        run_id="run-a", episode_id=episode, policy_id="weights-a", episode_step=step,
        observation_id=f"{episode}-obs-{step}", packet_id=f"{episode}-packet-{step}",
        decision_nanos=start, next_decision_nanos=end,
        reward=RewardWindow(start, end, reward), old_log_probability=-4.0, value=value,
        bootstrap=None if outcome in (Outcome.TERMINATED, Outcome.ABORTED) else BootstrapObservation(
            episode, "weights-a", f"{episode}-obs-{step + 1}", end, next_value
        ), outcome=outcome, recurrent_reset=step == 0 if reset is None else reset,
        valid=valid, invalid_reason=None if valid else "Missing observation",
    )


def loss_inputs(logs=(-2.0, -3.0), advantages=(1.0, -1.0), values=(0.0, 0.0), returns=(0.0, 0.0), entropies=(0.0, 0.0)):
    return dict(new_log_probabilities=mx.array(logs, dtype=mx.float32), new_values=mx.array(values, dtype=mx.float32),
                conditional_entropies=mx.array(entropies, dtype=mx.float32), old_log_probabilities=mx.array([-2.0, -3.0]),
                advantages=mx.array(advantages, dtype=mx.float32), returns=mx.array(returns, dtype=mx.float32), valid=mx.array([True, True]))


def test_irregular_duration_gae_matches_hand_computed_terminal_returns():
    items = (
        transition(0, start=1_000_000_000, duration=100_000_000, reward=2, value=1, next_value=3),
        transition(1, start=1_100_000_000, duration=250_000_000, reward=-1, value=3, next_value=2),
        transition(2, start=1_350_000_000, duration=50_000_000, reward=5, value=2, outcome=Outcome.TERMINATED),
    )
    result = duration_aware_gae(Rollout(items))
    gamma0, gamma1 = 2 ** (-0.1 / 30), 2 ** (-0.25 / 30)
    a2 = 5 - 2
    a1 = -1 + gamma1 * 2 - 3 + gamma1 * 0.95**2.5 * a2
    a0 = 2 + gamma0 * 3 - 1 + gamma0 * 0.95 * a1
    np.testing.assert_allclose(result.advantages, [a0, a1, a2], rtol=1e-6)
    np.testing.assert_allclose(result.returns, np.array([a0, a1, a2]) + [1, 3, 2], rtol=1e-6)
    np.testing.assert_allclose(result.discounts, 2 ** (-np.array([0.1, 0.25, 0.05]) / 30), rtol=1e-6)
    np.testing.assert_allclose(result.trace_decays, 0.95 ** np.array([1, 2.5, 0.5]), rtol=1e-6)
    assert result.valid.all()


def test_truncation_bootstraps_final_observation_but_breaks_next_episode_trace():
    cutoff = transition(reward=2, value=3, next_value=11, outcome=Outcome.TRUNCATED)
    next_episode = transition(episode="episode-b", start=5_000_000_000, reward=999,
                              value=-21, outcome=Outcome.TERMINATED)
    result = duration_aware_gae(Rollout((cutoff, next_episode)))
    np.testing.assert_allclose(result.advantages, [2 + 2**(-0.1 / 30) * 11 - 3, 1020], rtol=1e-6)
    with pytest.raises(ValueError, match="same episode/policy"):
        replace(cutoff, bootstrap=replace(cutoff.bootstrap, episode_id="episode-b"))
    with pytest.raises(ValueError, match="pre-reset"):
        replace(cutoff.bootstrap, recurrent_reset=True)
    with pytest.raises(ValueError, match="require a pre-reset bootstrap"):
        replace(cutoff, bootstrap=None)


def test_termination_and_rollout_cutoff_have_distinct_bootstraps():
    continuing = transition(reward=4, value=2, next_value=7)
    terminated = replace(continuing, bootstrap=None, outcome=Outcome.TERMINATED)
    np.testing.assert_allclose(duration_aware_gae(Rollout((continuing,))).returns, [4 + 2**(-0.1 / 30) * 7])
    np.testing.assert_array_equal(duration_aware_gae(Rollout((terminated,))).returns, [4])
    with pytest.raises(ValueError, match="must not bootstrap"):
        replace(continuing, outcome=Outcome.TERMINATED)


def test_abort_excluded_and_invalid_intervals_break_both_return_traces():
    items = (
        transition(0, reward=2, value=1, next_value=3),
        transition(1, reward=10_000, value=3, next_value=5, valid=False),
        transition(2, reward=4, value=5, next_value=2, reset=True),
        transition(3, reward=50_000, value=2, outcome=Outcome.ABORTED, valid=False),
        transition(episode="episode-b", start=900_000_000, reward=6, value=1, outcome=Outcome.TERMINATED),
    )
    result = duration_aware_gae(Rollout(items))
    gamma = 2**(-0.1 / 30)
    np.testing.assert_allclose(result.advantages, [2 + gamma * 3 - 1, 0, 4 + gamma * 2 - 5, 0, 5], rtol=1e-6)
    np.testing.assert_array_equal(result.valid, [True, False, True, False, True])
    assert result.returns[1] == result.returns[3] == 0
    incomplete = replace(items[3], reward=None, old_log_probability=None, value=None)
    assert not duration_aware_gae(Rollout((incomplete,), initial_state_id="state-3")).valid[0]


def test_reward_windows_do_not_move_with_execution_lead():
    item = transition(start=10_000_000_000)
    with pytest.raises(ValueError, match="never the execution window"):
        replace(item, reward=RewardWindow(item.decision_nanos + 100_000_000, item.next_decision_nanos + 100_000_000, 1))


@pytest.mark.parametrize("mutation", [
    lambda item: replace(item, policy_id="weights-b", bootstrap=replace(item.bootstrap, policy_id="weights-b")),
    lambda item: replace(item, run_id="other-run"),
    lambda item: replace(item, recurrent_reset=True),
    lambda item: replace(item, observation_id="wrong-observation"),
    lambda item: replace(item, packet_id="episode-a-packet-0"),
    lambda item: replace(item, value=1000),
])
def test_rollout_rejects_policy_state_and_observation_inconsistency(mutation):
    first, second = transition(0), transition(1, value=0.75)
    with pytest.raises(ValueError):
        Rollout((first, mutation(second)))


def test_mid_episode_collection_requires_saved_state_and_new_episode_requires_reset():
    item = transition(4)
    with pytest.raises(ValueError, match="saved behavior state"):
        Rollout((item,))
    assert Rollout((item,), initial_state_id="state-4").policy_id == "weights-a"
    with pytest.raises(ValueError, match="stale behavior state"):
        Rollout((transition(),), initial_state_id="state-4")
    with pytest.raises(ValueError, match="Episode zero"):
        transition(reset=False)
    with pytest.raises(ValueError, match="Episode changes"):
        Rollout((transition(), transition(episode="episode-b", start=200_000_000)))


def test_saved_targets_and_transitions_are_immutable():
    item = transition()
    with pytest.raises(FrozenInstanceError):
        item.value = 20
    with pytest.raises(ValueError, match="immutable tuple"):
        Rollout([item])
    result = duration_aware_gae(Rollout((item,)))
    with pytest.raises(ValueError):
        result.advantages[0] = 0
    with pytest.raises(ValueError):
        result.returns.flags.writeable = True


def test_lambda_zero_is_td_one_and_one_propagates_discounted_returns():
    items = (transition(value=2, reward=3, next_value=5),
             transition(1, value=5, reward=7, outcome=Outcome.TERMINATED))
    rollout = Rollout(items)
    gamma = 2**(-0.1 / 30)
    np.testing.assert_allclose(duration_aware_gae(rollout, ReturnConfig(lambda_per_reference=0)).advantages,
                               [3 + gamma * 5 - 2, 2])
    np.testing.assert_allclose(duration_aware_gae(rollout, ReturnConfig(lambda_per_reference=1)).returns,
                               [3 + gamma * 7, 7])


def test_ratio_one_and_value_and_entropy_conventions():
    result = ppo_loss(**loss_inputs(advantages=(2, 4), values=(1, 3), returns=(3, 5), entropies=(5, 9)))
    np.testing.assert_allclose(float(result.total), -3 + 0.5 * 4 - 0.01 * 7, atol=1e-6)
    assert float(result.policy) == -3
    assert float(result.value) == 4
    assert float(result.entropy) == 7
    assert float(result.ratio_mean) == float(result.ratio_min) == float(result.ratio_max) == 1
    assert float(result.sampled_kl) == float(result.signed_sampled_kl) == float(result.clip_fraction) == 0
    assert bool(result.finite) and not bool(result.should_stop)


@pytest.mark.parametrize("advantage,ratio,expected", [
    (2, 1.5, -2.4), (2, 0.5, -1), (-2, 0.5, 1.6), (-2, 1.5, 3),
])
def test_clipping_has_correct_sign_for_positive_and_negative_advantages(advantage, ratio, expected):
    inputs = loss_inputs(logs=(-2 + math.log(ratio), -3 + math.log(ratio)),
                         advantages=(advantage, advantage))
    result = ppo_loss(**inputs)
    np.testing.assert_allclose(float(result.policy), expected, atol=2e-6)
    np.testing.assert_allclose(float(result.ratio_mean), ratio, atol=2e-6)
    assert float(result.clip_fraction) == 1
    gradients = mx.grad(lambda logs: ppo_loss(**(inputs | dict(new_log_probabilities=logs))).policy)(
        inputs["new_log_probabilities"])
    saturated = (advantage > 0 and ratio > 1.2) or (advantage < 0 and ratio < 0.8)
    np.testing.assert_allclose(np.asarray(gradients), [0 if saturated else -advantage * ratio / 2] * 2, atol=2e-6)


def test_joint_packet_ratio_is_not_average_of_factor_ratios():
    old_factors = mx.array([-2.0, -3.0, -1.0])
    new_factors = old_factors + mx.array([0.15, 0.15, 0.15])
    result = ppo_loss(mx.sum(new_factors, keepdims=True), mx.zeros((1,)), mx.array([4.0]),
                      old_log_probabilities=mx.sum(old_factors, keepdims=True), advantages=mx.array([1.0]),
                      returns=mx.zeros((1,)), valid=mx.array([True]))
    np.testing.assert_allclose(float(result.ratio_mean), math.exp(0.45), atol=2e-6)
    np.testing.assert_allclose(float(result.policy), -1.2, atol=1e-6)
    assert float(result.entropy) == 4


def test_kl_guard_uses_joint_distribution_and_is_stable_near_ratio_one():
    values = loss_inputs(logs=(-2.5, -3.5))
    result = ppo_loss(**values)
    np.testing.assert_allclose(float(result.sampled_kl), math.exp(-0.5) - 1 + 0.5, atol=1e-6)
    assert bool(result.should_stop)
    tiny = ppo_loss(**loss_inputs(logs=(-2.0 + 1e-4, -3.0 - 1e-4)))
    assert 0 <= float(tiny.sampled_kl) < 1e-7
    assert not bool(tiny.should_stop)


def test_invalid_and_burnin_rows_are_masked_before_exp_and_do_not_dilute_loss():
    values = loss_inputs(logs=(-2, float("nan")), advantages=(3, float("nan")),
                         values=(1, float("inf")), returns=(3, float("nan")), entropies=(4, float("nan")))
    values.update(valid=mx.array([True, False]), old_log_probabilities=mx.array([-2, float("inf")]))
    result = ppo_loss(**values)
    np.testing.assert_allclose(float(result.total), -3 + 0.5 * 4 - 0.01 * 4, atol=1e-6)
    assert bool(result.finite) and int(result.valid_count) == 1
    assert float(result.ratio_min) == float(result.ratio_max) == 1
    def loss(logs, predictions, entropies):
        return ppo_loss(**(values | dict(new_log_probabilities=logs, new_values=predictions,
                                         conditional_entropies=entropies))).total
    grads = mx.grad(loss, argnums=(0, 1, 2))(values["new_log_probabilities"], values["new_values"], values["conditional_entropies"])
    for grad in grads:
        assert np.isfinite(np.asarray(grad)).all()
        assert float(grad[1]) == 0


def test_ppo_gradients_and_stopped_behavior_targets():
    values = loss_inputs(advantages=(2, -1), values=(1, 3), returns=(2, 1), entropies=(4, 8))
    def loss(logs, predictions, entropies, old, advantages, targets):
        return ppo_loss(logs, predictions, entropies, old_log_probabilities=old,
                         advantages=advantages, returns=targets, valid=values["valid"]).total
    gradients = mx.grad(loss, argnums=(0, 1, 2, 3, 4, 5))(
        values["new_log_probabilities"], values["new_values"], values["conditional_entropies"],
        values["old_log_probabilities"], values["advantages"], values["returns"])
    expected = ([-1, 0.5], [-0.5, 1], [-0.005, -0.005], [0, 0], [0, 0], [0, 0])
    for actual, reference in zip(gradients, expected):
        assert np.isfinite(np.asarray(actual)).all()
        np.testing.assert_allclose(np.asarray(actual), reference, atol=1e-6)


def test_sampled_packet_decoder_replay_and_ppo_gradient_update_agree():
    from astra.model.actions import ActionVocabulary, PacketDecoder
    from astra.model.config import ModelConfig
    from astra.model.vision import VisualFeatures

    mx.random.seed(482)
    config = ModelConfig.test_small()
    decoder = PacketDecoder(config, ActionVocabulary((4, 13), (0,), True, True, True))
    features = VisualFeatures(
        mx.zeros((3, config.query_count * config.spatial_width)),
        mx.random.normal((3, 1, 8, config.spatial_width)),
        mx.zeros((3, 1, 8, 4)), mx.ones((3, 1, 8), dtype=mx.bool_),
        mx.ones((3, 1), dtype=mx.bool_),
    )
    context = mx.random.normal((3, config.recurrent_width))
    sampled = decoder.sample(context, features, key=mx.random.key(911))
    sampled.packets.validate(config, decoder.vocabulary, features)
    old = mx.stop_gradient(sampled.log_probability)
    mx.eval(old, sampled.packets.operation)
    mask = mx.array([True, True, False])
    loss_config = PPOConfig(entropy_coefficient=0)

    def objective(model):
        scored = model.log_prob(context, features, sampled.packets)
        return ppo_loss(scored.log_probability, mx.zeros((3,)), scored.conditional_entropy,
                        old_log_probabilities=old, advantages=mx.array([1.0, -0.5, 100.0]),
                        returns=mx.zeros((3,)), valid=mask, config=loss_config).total

    replay = decoder.log_prob(context, features, sampled.packets)
    verify_behavior_log_probabilities(np.asarray(old), np.asarray(replay.log_probability), np.asarray(mask))
    initial, gradients = nn.value_and_grad(decoder, objective)(decoder)
    mx.eval(initial, gradients)
    assert all(np.isfinite(np.asarray(value)).all() for _, value in tree_flatten(gradients))
    assert any(np.any(np.asarray(value) != 0) for _, value in tree_flatten(gradients))
    decoder.update(tree_map(lambda parameter, gradient: parameter - 1e-5 * gradient,
                            decoder.parameters(), gradients))
    updated = objective(decoder)
    mx.eval(updated, decoder.parameters())
    assert float(updated) < float(initial)


def test_empty_valid_batch_and_nonfinite_ratios_stop_instead_of_optimizing():
    values = loss_inputs()
    empty = ppo_loss(**(values | dict(valid=mx.array([False, False]))))
    assert float(empty.total) == 0 and bool(empty.should_stop) and int(empty.valid_count) == 0
    invalid = ppo_loss(**(values | dict(new_log_probabilities=mx.array([float("nan"), -3]))))
    assert not bool(invalid.finite) and bool(invalid.should_stop)
    overflow = ppo_loss(**(values | dict(old_log_probabilities=mx.array([-1000.0, -3]))))
    assert not bool(overflow.finite) and bool(overflow.should_stop)


def test_advantage_normalization_and_behavior_replay_ignore_invalid_rows():
    normalized = normalize_advantages(np.array([1.0, 5.0, float("nan")]), np.array([True, True, False]))
    np.testing.assert_array_equal(normalized, [-1, 1, 0])
    assert verify_behavior_log_probabilities(np.array([-2, -4, float("nan")]), np.array([-2, -4, 10]),
                                              np.array([True, True, False])) == 0
    with pytest.raises(ValueError, match="disagrees"):
        verify_behavior_log_probabilities(np.array([-2]), np.array([-1.5]), np.array([True]))
    with pytest.raises(ValueError, match="valid decisions"):
        normalize_advantages(np.array([1]), np.array([False]))


@pytest.mark.parametrize("config", [
    lambda: ReturnConfig(lambda_per_reference=1.1), lambda: ReturnConfig(discount_half_life_seconds=0),
    lambda: ReturnConfig(lambda_reference_seconds=float("nan")), lambda: PPOConfig(clip_ratio=0),
    lambda: PPOConfig(entropy_coefficient=-1), lambda: PPOConfig(target_kl=False),
])
def test_invalid_return_and_ppo_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        config()
