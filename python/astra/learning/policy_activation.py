"""Coordinator-owned policy lifecycle; this class does not load or run a model."""
from __future__ import annotations
from astra.environments.interface import EnvironmentError, uuid_key, same_id


class PolicyActivationGate:
    """One learning update, one pending snapshot, activation only after reset.

    The native coordinator must verify model-load acknowledgements and actual
    stop/reset evidence before calling these methods. Episodes started while
    learning or a pending update exists are continuity/audit experience.
    """
    def __init__(self, policy_id: str):
        self.policy_id=uuid_key(policy_id)
        self.pending_policy_id=None
        self.learning_rollout_id=None
        self.episode_id=None
        self.episode_learning_eligible=False
        self._episodes=set()
        self._resets=set()
        self._rollouts=set()

    @staticmethod
    def _released(controls_released, pending_packets):
        if controls_released is not True or type(pending_packets) is not int or pending_packets!=0:
            raise EnvironmentError('Policy lifecycle requires explicit cleared-control evidence')

    def confirm_reset(self, *, episode_id, reset_id, policy_id, controls_released, pending_packets):
        self._released(controls_released,pending_packets)
        episode_id,reset_id,policy_id=uuid_key(episode_id),uuid_key(reset_id),uuid_key(policy_id)
        if self.episode_id is not None or episode_id in self._episodes or reset_id in self._resets:
            raise EnvironmentError('Activation requires a fresh reset after joined episode stop')
        if len(self._episodes)>=65536: raise EnvironmentError('Policy lifecycle exceeded its bounded episode history')
        expected=self.pending_policy_id or self.policy_id
        if policy_id!=expected: raise EnvironmentError('Actor reset did not activate the expected immutable policy')
        if self.pending_policy_id is not None:
            self.policy_id=self.pending_policy_id; self.pending_policy_id=None
        self.episode_id=episode_id
        self._episodes.add(episode_id); self._resets.add(reset_id)
        self.episode_learning_eligible=self.learning_rollout_id is None
        return self.episode_learning_eligible

    def validate_actor(self, *, episode_id, policy_id):
        if not same_id(episode_id,self.episode_id) or not same_id(policy_id,self.policy_id):
            raise EnvironmentError('Actor policy/episode changed outside a confirmed reset')
        return self.episode_learning_eligible

    def confirm_stop(self, *, episode_id, controls_released, pending_packets):
        self._released(controls_released,pending_packets)
        if self.episode_id is None or not same_id(episode_id,self.episode_id):
            raise EnvironmentError('Stop evidence belongs to another episode')
        self.episode_id=None; self.episode_learning_eligible=False

    def begin_learning(self, *, rollout_id, policy_id):
        if self.learning_rollout_id is not None or self.pending_policy_id is not None or not same_id(policy_id,self.policy_id):
            raise EnvironmentError('Only the current behavior policy may begin one learning update')
        if self.episode_id is not None and self.episode_learning_eligible:
            raise EnvironmentError('Finish the selected learning episode before starting its update')
        rollout_id=uuid_key(rollout_id)
        if rollout_id in self._rollouts or len(self._rollouts)>=65536:
            raise EnvironmentError('A sealed rollout cannot be admitted twice or exceed lifecycle history capacity')
        self._rollouts.add(rollout_id); self.learning_rollout_id=rollout_id

    def complete_learning(self, *, rollout_id, proposed_policy_id):
        if self.learning_rollout_id is None or not same_id(rollout_id,self.learning_rollout_id):
            raise EnvironmentError('Learning completion belongs to another rollout')
        proposed=None if proposed_policy_id is None else uuid_key(proposed_policy_id)
        if proposed==self.policy_id: raise EnvironmentError('Changed weights require a new policy identity')
        self.learning_rollout_id=None; self.pending_policy_id=proposed

    def cancel_learning(self, *, rollout_id):
        if self.learning_rollout_id is None or not same_id(rollout_id,self.learning_rollout_id):
            raise EnvironmentError('Learning cancellation belongs to another rollout')
        self.learning_rollout_id=None
