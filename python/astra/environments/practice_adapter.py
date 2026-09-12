"""The virtual environment uses the same actor boundary as external sources."""
from __future__ import annotations
import uuid
from .interface import EnvironmentSpec, EnvironmentObservation, EnvironmentTransition, SurfaceObservation
from .practice import PracticeEnvironment


class PracticeAdapter:
    def __init__(self, environment: PracticeEnvironment):
        self.environment = environment
        config = environment.config
        self.run_id = str(uuid.uuid4())
        self.spec = EnvironmentSpec(identity='practice:' + config.task, action_vocabulary=environment.action_vocabulary,
            period_ms=config.period_ms, lead_ms=config.lead_ms, maximum_episode_ms=config.time_limit_ms,
            maximum_observation_bytes=config.pixel_width * config.pixel_height * 4, maximum_surfaces=1,
            maximum_frame_age_ms=max(config.period_ms, 1), discount_half_life_ms=config.discount_half_life_ms,
            seed=config.seed, seeded_reset=True).validate()

    @property
    def signature(self):
        # The complete existing practice config already fingerprints every
        # derived adapter setting; unchanged runs keep their resume identity.
        return self.environment.config.signature

    @property
    def outcome(self):
        return self.environment.outcome

    def observation(self, value):
        metadata = value.metadata
        return EnvironmentObservation(metadata['id'], value.episode_id, metadata['observedNanos'],
            metadata['surface']['geometryRevision'], (SurfaceObservation(value.pixels, metadata, metadata['observedNanos']),),
            value.control_state).validate(self.spec)

    def reset(self, *, seed, cancelled):
        if cancelled(): raise InterruptedError('Environment reset cancelled')
        return self.observation(self.environment.reset(seed=seed))

    def step(self, commands, *, context, cancelled):
        if cancelled(): raise InterruptedError('Environment decision cancelled')
        context.validate()
        result = self.environment.step(commands, episode_id=context.episode_id, provenance='agent')
        return EnvironmentTransition(self.observation(result.observation), result.reward, result.duration_ms,
            result.outcome, tuple(result.raw_events), tuple(result.command_results), result.provenance, result.reason,
            result.outcome)

    def seal_episode(self, *, cancelled):
        if cancelled(): raise InterruptedError('Episode sealing cancelled')
        if self.outcome == 'continuing': raise RuntimeError('Cannot seal a running episode')

    def abort(self, reason):
        if self.environment.outcome == 'continuing': self.environment.abort(reason)
