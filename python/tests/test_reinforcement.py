from dataclasses import replace
import json
import math
import uuid

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest

from astra.checkpoints import load_checkpoint, save_checkpoint
from astra.environments.practice import PracticeConfig, PracticeEnvironment
from astra.learning.reinforcement import ReinforcementConfig, ReinforcementTrainer
from astra.learning.rl import Outcome, Rollout
from astra.model.config import ModelConfig
from astra.model.policy import AgentPolicy


def trainer(*, decisions=4, epochs=1, sequence=2, batch=4, limit=700, task="pointing", frozen=False):
    mx.random.seed(1)
    environment = PracticeEnvironment(PracticeConfig(
        task=task, pixel_width=96, pixel_height=64, logical_bounds=(0, 0, 96, 64),
        time_limit_ms=limit, shaping_scale=0.1 if task == "pointing" else 0))
    policy = AgentPolicy(ModelConfig.test_small(), environment.action_vocabulary)
    if frozen:
        policy.vision.backbone.freeze()
    config = ReinforcementConfig(rollout_decisions=decisions, epochs=epochs, sequence_length=sequence,
                                 burn_in=1, effective_batch_decisions=batch, checkpoint_vision=True)
    return ReinforcementTrainer(policy, environment, config, policy_id="initial-policy")


def parameters(model):
    return {name: np.asarray(value).copy() for name, value in tree_flatten(model.parameters())}


def assert_parameters_equal(model, expected):
    for name, value in tree_flatten(model.parameters()):
        np.testing.assert_array_equal(np.asarray(value), expected[name], err_msg=name)


def test_collection_preserves_delayed_actions_decision_windows_and_prereset_bootstrap():
    worker = trainer(decisions=6, limit=350, task="delayed_memory")
    collected = worker.collect()
    items = collected.decisions
    assert [item.transition.duration_seconds for item in items] == [0.1, 0.1, 0.1, 0.05] * 2
    assert items[3].transition.outcome == Outcome.TRUNCATED
    assert items[3].transition.bootstrap.episode_id != items[4].transition.episode_id
    assert items[4].observation.reset and items[4].transition.episode_step == 0
    # With a full-period lead, nothing from the first sampled packet has
    # executed before the second observation cutoff.
    assert json.loads(items[1].observation.events_json) == []
    for item in items:
        transition = item.transition
        assert transition.reward.start_nanos == transition.decision_nanos
        assert transition.reward.end_nanos == transition.next_decision_nanos
        for event in json.loads(item.observation.events_json):
            assert event["observedNanos"] <= transition.decision_nanos
        assert item.packet().operation.shape == (1, worker.policy.config.packet_capacity + 1)
        if transition.bootstrap is not None:
            assert item.bootstrap_observation.metadata["id"] == transition.bootstrap.observation_id
            assert not item.bootstrap_observation.reset
    terminal = items[3]
    state = tuple(mx.array(value) for value in terminal.state_before)
    encoding = worker.policy(terminal.observation.prepare(worker.policy.config, ()), state)
    bootstrap = worker.policy(terminal.bootstrap_observation.prepare(worker.policy.config, ()), encoding.temporal.state)
    np.testing.assert_allclose(float(bootstrap.temporal.value[0, 0]), terminal.transition.bootstrap.value, atol=2e-6)
    error, low, high = worker.verify_behavior(collected)
    assert error < 2e-4
    np.testing.assert_allclose([low, high], [1, 1], atol=2e-5)
    with pytest.raises(ValueError):
        items[0].observation.pixels.flags.writeable = True
    worker.discard_rollout(collected)
    worker.stop()


def test_fresh_policy_ppo_changes_learner_and_only_activates_after_confirmed_reset():
    worker = trainer(epochs=2)
    original = parameters(worker.policy)
    actor_original = parameters(worker._actor)
    collected = worker.collect()
    metrics = worker.update(collected)
    assert metrics.optimizer_updates > 0 and metrics.maximum_gradient_norm > 0
    assert all(math.isfinite(value) for value in (metrics.mean_loss, metrics.mean_value_loss,
                                                 metrics.mean_entropy_surrogate, metrics.maximum_sampled_kl))
    assert any(np.any(np.asarray(value) != original[name]) for name, value in tree_flatten(worker.policy.parameters()))
    assert_parameters_equal(worker._actor, actor_original)
    assert worker.actor_policy_id == "initial-policy"
    pending = worker.pending_policy_id
    assert pending and pending != worker.actor_policy_id
    assert worker.at_episode_boundary
    worker.activate_pending_at_reset()
    with pytest.raises(RuntimeError, match="episode boundary"):
        worker.activate_pending_at_reset()
    assert worker.actor_policy_id == pending and worker.pending_policy_id is None
    assert worker._state is None and worker._current.reset
    assert_parameters_equal(worker._actor, parameters(worker.policy))
    worker.stop()


def test_real_recurrent_prefix_and_burnin_replay_match_behavior_then_reveal_staleness():
    worker = trainer(decisions=6, sequence=2, batch=6, limit=900)
    collected = worker.collect()
    complete = worker.replay_state(collected, 0, 4, mode="full_prefix")
    burnin = worker.replay_state(collected, 0, 4, mode="burn_in")
    for exact, approximate, saved in zip(complete, burnin, collected.decisions[4].state_before):
        np.testing.assert_allclose(np.asarray(exact), saved, atol=2e-6)
        np.testing.assert_allclose(np.asarray(approximate), saved, atol=2e-6)
    differences = []
    def inspect_replay(_):
        complete = worker.replay_state(collected, 0, 4, mode="full_prefix")
        burnin = worker.replay_state(collected, 0, 4, mode="burn_in")
        differences.extend(float(mx.max(mx.abs(exact - approximate))) for exact, approximate in zip(complete, burnin))
    worker.update(collected, on_update=inspect_replay)
    assert all(math.isfinite(value) for value in differences)
    assert max(differences) > 1e-7  # Burn-in is an explicitly measured approximation after updates.
    worker.stop()


def test_frozen_pretrained_optimizer_groups_do_not_advance():
    worker = trainer(frozen=True)
    original = parameters(worker.policy.vision.backbone)
    result = worker.run_iteration()
    assert result.metrics.optimizer_updates > 0
    assert_parameters_equal(worker.policy.vision.backbone, original)
    assert int(worker.optimizer.state["pretrained_decay"]["step"]) == 0
    assert int(worker.optimizer.state["pretrained_no_decay"]["step"]) == 0
    assert int(worker.optimizer.state["new_decay"]["step"]) > 0
    worker.stop()


def test_update_cancellation_rolls_back_weights_and_optimizer_then_allows_retry():
    worker = trainer(epochs=2, batch=2)
    collected = worker.collect()
    original = parameters(worker.policy)
    cancelled = False
    def cancel_after_update(_):
        nonlocal cancelled
        cancelled = True
    with pytest.raises(InterruptedError):
        worker.update(collected, cancelled=lambda: cancelled, on_update=cancel_after_update)
    assert_parameters_equal(worker.policy, original)
    assert worker.optimizer_updates == 0 and worker.pending_policy_id is None
    assert int(worker.optimizer.state["new_decay"]["step"]) == 0
    metrics = worker.update(collected)
    assert metrics.optimizer_updates > 0
    worker.stop()


def test_behavior_replay_rejects_tampered_packet_likelihood_before_updating():
    worker = trainer()
    collected = worker.collect()
    changed = replace(collected.decisions[0], transition=replace(collected.decisions[0].transition,
                                                                old_log_probability=collected.decisions[0].transition.old_log_probability - 1))
    decisions = (changed, *collected.decisions[1:])
    tampered = replace(collected, decisions=decisions, rollout=Rollout(tuple(item.transition for item in decisions)))
    original = parameters(worker.policy)
    with pytest.raises(ValueError, match="replay disagrees"):
        worker.update(tampered)
    assert_parameters_equal(worker.policy, original)
    assert worker.optimizer_updates == 0
    worker.discard_rollout(collected)
    worker.stop()


def test_checkpoint_roundtrip_resumes_optimizer_at_fresh_environment_reset(tmp_path):
    worker = trainer()
    result = worker.run_iteration()
    destination = tmp_path / str(uuid.uuid4())
    save_checkpoint(destination, worker.policy, kind="reinforcement", step=worker.optimizer_updates,
                    training_state=result.checkpoint_state)
    loaded = load_checkpoint(destination, include_training=True)
    resumed = ReinforcementTrainer(loaded.policy, PracticeEnvironment(worker.environment.config), worker.config,
                                   restored_state=loaded.training_state)
    assert resumed.actor_policy_id == worker.pending_policy_id
    assert resumed.at_episode_boundary and resumed._state is None
    assert resumed.optimizer_updates == worker.optimizer_updates
    next_iteration = resumed.run_iteration()
    assert next_iteration.metrics.iteration == 2
    assert next_iteration.checkpoint_state["requiresEnvironmentReset"]
    worker.stop()
    resumed.stop()


def test_memory_admission_and_cancel_do_not_invent_abort_training_rows():
    worker = trainer()
    worker.config = replace(worker.config, maximum_rollout_bytes=1024).validate()
    with pytest.raises(MemoryError):
        worker.collect()
    assert worker.at_episode_boundary and worker.environment.elapsed_ms == 0
    assert worker.state["optimizerUpdates"] == 0
    worker = trainer()
    with pytest.raises(InterruptedError):
        worker.run_iteration(cancelled=lambda: True)
    assert worker.at_episode_boundary and worker.pending_policy_id is None
    assert worker.state["iteration"] == 0


def test_second_iteration_starts_at_reset_and_does_not_mix_policy_versions():
    worker = trainer()
    first = worker.run_iteration()
    pending = first.metrics.pending_policy_id
    second = worker.collect()
    assert second.audit_tail_decisions == 0 and worker.at_episode_boundary
    assert second.rollout.policy_id == pending
    assert second.decisions[0].transition.episode_step == 0
    assert all(item.transition.policy_id == pending for item in second.decisions)
    worker.discard_rollout(second)
    worker.stop()


def test_environment_outcome_is_readonly_and_cleanup_preserves_completed_world_error(monkeypatch):
    worker = trainer(decisions=2, limit=100)
    environment = worker.environment
    assert environment.outcome == "aborted"
    with pytest.raises(AttributeError):
        environment.outcome = "continuing"
    original = environment.step
    def fail_after_boundary(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("Rendering failed after the environment boundary")
    monkeypatch.setattr(environment, "step", fail_after_boundary)
    with pytest.raises(RuntimeError, match="Rendering failed"):
        worker.collect()
    assert environment.outcome == "truncated"
    assert worker.at_episode_boundary


def test_persistent_renderer_failure_releases_held_controls_and_preserves_primary_error(monkeypatch):
    worker = trainer(task="delayed_memory")
    environment = worker.environment
    def renderer_failed():
        raise RuntimeError("Secondary renderer failure during abort snapshot")
    def fail_before_policy_decision():
        if environment.outcome != 'continuing':
            return False
        # Exercise actual delayed virtual control ownership without running a
        # model gradient or posting any operating-system input.
        environment.step([{"operation": "buttonDown", "offsetMs": 0, "button": 0}])
        observed = environment.step([]).observation
        assert observed.control_state["buttons"] == [0]
        monkeypatch.setattr(environment, "_render", renderer_failed)
        raise RuntimeError("Primary collection failure")
    with pytest.raises(RuntimeError, match="Primary collection failure") as caught:
        worker.collect(cancelled=fail_before_policy_decision)
    assert any("cleanup also failed" in note for note in caught.value.__notes__)
    assert environment.outcome == "aborted" and worker.at_episode_boundary
    assert not environment._buttons and not environment._keys and not environment._pending
    assert worker.state["optimizerUpdates"] == 0


def test_overshooting_first_step_is_rejected_with_optimizer_and_actor_unchanged():
    from astra.learning.optimizers import GroupedAdamW
    worker = trainer(batch=64)
    worker.config = replace(worker.config, learning_rate=.01, pretrained_learning_rate=.001,
                            maximum_kl_backtracks=0).validate()
    worker.optimizer = GroupedAdamW(learning_rate=.01, pretrained_learning_rate=.001)
    original = parameters(worker.policy)
    collected = worker.collect()
    metrics = worker.update(collected)
    assert metrics.optimizer_updates == 0 and worker.optimizer_updates == 0
    assert metrics.stopped_for_kl and metrics.rejected_optimizer_steps == 1
    assert metrics.maximum_candidate_kl > worker.config.ppo.target_kl
    assert metrics.maximum_accepted_kl == metrics.maximum_sampled_kl == 0
    assert worker.pending_policy_id is None and worker.actor_policy_id == 'initial-policy'
    assert_parameters_equal(worker.policy, original)
    assert_parameters_equal(worker._actor, original)
    assert int(worker.optimizer.state['new_decay']['step']) == 0
    worker.stop()


def test_backtracking_admits_only_sample_bound_policy_without_repeated_optimizer_steps():
    from astra.learning.optimizers import GroupedAdamW
    worker = trainer(batch=64)
    worker.config = replace(worker.config, learning_rate=.01, pretrained_learning_rate=.001).validate()
    worker.optimizer = GroupedAdamW(learning_rate=.01, pretrained_learning_rate=.001)
    original = parameters(worker.policy)
    collected = worker.collect()
    seen, measurements = [], []
    metrics = worker.update(collected, on_validation=seen.append,
                            on_update=lambda _: measurements.append(worker.evaluate_policy_shift(collected)))
    assert metrics.optimizer_updates == 1 and metrics.backtrack_count > 0
    assert metrics.maximum_candidate_kl > worker.config.ppo.target_kl
    assert 0 < metrics.minimum_step_scale < 1
    assert metrics.maximum_accepted_kl <= worker.config.ppo.target_kl
    measured = measurements[-1]
    assert measured.finite and measured.sampled_kl == pytest.approx(metrics.maximum_accepted_kl)
    assert measured.clip_fraction == pytest.approx(metrics.clip_fraction)
    assert int(worker.optimizer.state['new_decay']['step']) == 1
    assert seen[-1]['validation_decisions'] == len(collected.decisions)
    assert_parameters_equal(worker._actor, original)
    assert any(np.any(np.asarray(value) != original[name]) for name, value in tree_flatten(worker.policy.parameters()))
    assert worker.pending_policy_id is not None
    worker.stop()


def test_cancel_during_candidate_replay_restores_the_entire_iteration_before_retry():
    worker = trainer()
    collected = worker.collect()
    original = parameters(worker.policy)
    cancel = False
    def validating(fields):
        nonlocal cancel
        cancel = fields['validation_decisions'] >= 2
    with pytest.raises(InterruptedError, match='validation cancelled'):
        worker.update(collected, cancelled=lambda: cancel, on_validation=validating)
    assert_parameters_equal(worker.policy, original)
    assert worker.optimizer_updates == worker.iteration == 0 and worker.pending_policy_id is None
    assert int(worker.optimizer.state['new_decay']['step']) == 0
    assert worker.update(collected).optimizer_updates > 0
    worker.stop()


def test_finishing_episode_progress_retains_every_decision_until_the_real_boundary():
    worker = trainer(limit=900)
    finishing, learning = [], []
    collected = worker.collect(on_decision=learning.append, on_finishing_episode=finishing.append)
    assert finishing == [4, 5, 6, 7, 8]
    assert learning == [1, 2, 3, 9]
    assert len(collected.decisions) == 9 and collected.audit_tail_decisions == 0
    assert collected.decisions[-1].transition.outcome == Outcome.TRUNCATED
    assert all(item.transition.policy_id == 'initial-policy' for item in collected.decisions)
    worker.discard_rollout(collected)
    worker.stop()


def test_late_sparse_reward_reaches_gae_and_ppo_beyond_rollout_minimum_and_ram_cache(tmp_path, monkeypatch):
    from astra.learning.rl import duration_aware_gae
    worker = trainer(decisions=3, batch=64, limit=900, task='delayed_memory')
    worker._scratch_directory = tmp_path
    worker.config = replace(worker.config, maximum_rollout_bytes=96 * 1024).validate()
    step = worker.environment.step
    def sparse_survival_reward(*args, **kwargs):
        result = step(*args, **kwargs)
        return replace(result, reward=1.0 if result.outcome == 'truncated' else 0.0)
    monkeypatch.setattr(worker.environment, 'step', sparse_survival_reward)
    collected = worker.collect()
    assert len(collected.decisions) == 9 > worker.config.rollout_decisions
    assert collected.spool.disk_bytes > worker.config.maximum_rollout_bytes
    assert collected.spool.peak_memory_bytes <= worker.config.maximum_rollout_bytes
    rewards = [item.transition.reward.value for item in collected.decisions]
    assert rewards == [0] * 8 + [1]
    targets = duration_aware_gae(collected.rollout, worker.config.returns)
    no_reward = Rollout(tuple(replace(item, reward=replace(item.reward, value=0)) for item in collected.rollout.transitions))
    baseline = duration_aware_gae(no_reward, worker.config.returns)
    assert targets.returns[-1] - baseline.returns[-1] == pytest.approx(1)
    assert targets.returns[0] - baseline.returns[0] > .5
    assert worker.verify_behavior(collected)[0] < 2e-4
    directory = collected.spool.directory
    result = worker.update(collected)
    assert result.decisions == worker.decisions == 9 and result.optimizer_updates > 0
    assert result.maximum_accepted_kl <= worker.config.ppo.target_kl
    assert not directory.exists() and collected.spool.closed
    assert worker.state['decisions'] == 9 and worker.state['schemaVersion'] == 2
    worker.stop()


@pytest.mark.parametrize('budget', ['disk', 'decisions', 'metadata'])
def test_incomplete_episode_resource_failure_cleans_spool_without_inventing_training_rows(tmp_path, budget):
    worker = trainer(decisions=3, limit=900, task='delayed_memory')
    worker._scratch_directory = tmp_path
    fields = {'disk': {'maximum_rollout_disk_bytes': 4 * 96 * 64 * 4},
              'decisions': {'maximum_rollout_decisions': 4},
              'metadata': {'maximum_rollout_bytes': 96 * 64 * 4 + 2048}}[budget]
    worker.config = replace(worker.config, **fields).validate()
    with pytest.raises((OSError, MemoryError)):
        worker.collect()
    assert worker.at_episode_boundary and worker.state['decisions'] == worker.state['iteration'] == 0
    assert not list(tmp_path.iterdir())
    worker.stop()


def test_cancelled_update_preserves_retry_storage_then_discard_releases_it(tmp_path):
    worker = trainer()
    worker._scratch_directory = tmp_path
    collected = worker.collect()
    with pytest.raises(InterruptedError):
        worker.update(collected, cancelled=lambda: True)
    assert collected.spool.directory.exists() and not collected.spool.closed
    assert worker.verify_behavior(collected)[0] < 2e-4
    worker.discard_rollout(collected)
    assert not list(tmp_path.iterdir())
    worker.stop()
