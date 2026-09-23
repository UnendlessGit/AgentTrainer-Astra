"""Explicit oracle fixture demonstrations; never presented as human recordings."""
from __future__ import annotations

import uuid

from astra.data.batching import LearningSample
from astra.data.observations import make_observation
from astra.model.config import ModelConfig
from .practice import PracticeEnvironment, PracticeConfig


class PracticeDemonstrations:
    provenance = "practice_oracle"

    def __init__(self, *, environment: PracticeConfig, model: ModelConfig, seeds_by_split: dict[str, list[int]], cancelled=lambda: False):
        if (model.period_ms, model.lead_ms) != (environment.period_ms, environment.lead_ms):
            raise ValueError("Practice demonstrations and model must share timing")
        if set(seeds_by_split) - {"train", "validation", "test"}:
            raise ValueError("Unknown practice demonstration split")
        all_seeds = [seed for values in seeds_by_split.values() for seed in values]
        if len(all_seeds) != len(set(all_seeds)):
            raise ValueError("Practice seeds must be independent across splits")
        self._episodes = {}
        self._cancelled = cancelled
        self.config = model
        self.environment_config = environment
        env = PracticeEnvironment(environment)
        self.vocabulary = env.action_vocabulary
        self.seeds_by_split = seeds_by_split
        for split, seeds in seeds_by_split.items():
            for seed in seeds:
                current = env.reset(seed=seed)
                episode_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"astra:practice_oracle:{environment.signature}:{model.signature}:{seed}"))
                steps = 0
                for step in range(environment.time_limit_ms // environment.period_ms + 2):
                    if cancelled():
                        raise InterruptedError("Practice demonstration preparation cancelled")
                    commands = tuple(env.oracle_commands())
                    transition = env.step(list(commands), episode_id=current.episode_id, provenance="oracle")
                    current = transition.observation
                    steps += 1
                    if transition.outcome != "continuing":
                        if transition.outcome != "terminated" or transition.reward <= 0:
                            raise RuntimeError("Practice oracle failed; its demonstration cannot be used")
                        break
                else:
                    raise RuntimeError("Practice oracle exceeded its episode bound")
                self._episodes[episode_id] = {"id": episode_id, "steps": steps, "split": split, "seed": seed,
                                                       "provenance": self.provenance}

    def episodes(self, split="train"):
        return [dict(value) for value in self._episodes.values() if value["split"] == split]

    def samples(self, episode_id, start=0, count=None):
        episode = self._episodes[episode_id]
        if type(start) is not int or start < 0 or (count is not None and (type(count) is not int or count < 1)):
            raise ValueError("Invalid practice episode range")
        end = min(episode["steps"], start + count if count is not None else episode["steps"])
        # Regenerate deterministic raw observations instead of retaining every
        # normalized image of a thirty-second memory trial on the GPU. Seeking
        # runs virtual time without neural preparation before the requested cut.
        env = PracticeEnvironment(self.environment_config)
        current = env.reset(seed=episode["seed"])
        previous_events = []
        last_input_nanos = None
        for step in range(end):
            if self._cancelled():
                raise InterruptedError("Practice demonstration loading cancelled")
            commands = tuple(env.oracle_commands())
            if step >= start:
                observation = make_observation([(current.pixels, current.metadata)], current.control_state,
                                                cutoff_nanos=current.metadata["observedNanos"], elapsed_seconds=self.config.period_ms / 1000,
                                                reset=step == 0, config=self.config, context_ids=(0,) * len(self.config.context_sizes), executed_events=previous_events,
                                                last_input_nanos=last_input_nanos)
                yield LearningSample(observation, (current.metadata["surface"],), commands, episode_id, step)
            transition = env.step(list(commands), episode_id=current.episode_id, provenance="oracle")
            current = transition.observation; previous_events = transition.raw_events
            times = [event["observedNanos"] for event in previous_events if event["origin"] in ("physical", "agent")]
            if times:
                last_input_nanos = max(last_input_nanos or 0, max(times))
