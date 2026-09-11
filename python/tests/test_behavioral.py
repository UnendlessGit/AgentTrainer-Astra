from dataclasses import replace
import uuid

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest

from astra.checkpoints import save_checkpoint, load_checkpoint
from astra.environments.practice import PracticeConfig
from astra.environments.demonstrations import PracticeDemonstrations
from astra.learning.behavioral import BehaviorConfig, BehaviorTrainer, EpisodeLanes
from astra.learning.optimizers import GroupedAdamW
from astra.model.config import ModelConfig
from astra.model.policy import AgentPolicy


def fixture(seeds=(7, 8)):
    config = ModelConfig.test_small()
    source = PracticeDemonstrations(environment=PracticeConfig(pixel_width=64, pixel_height=64, logical_bounds=(0, 0, 64, 64)),
                                    model=config, seeds_by_split={"train": list(seeds), "validation": [70]})
    return config, source


def test_contiguous_lanes_and_sampler_resume_do_not_repeat_or_skip():
    config, source = fixture((1, 2, 3, 4))
    lanes = EpisodeLanes(source, split="train", lanes=2, sequence_length=1, seed=3, epoch=0)
    first = lanes.next()
    restored = EpisodeLanes(source, split="train", lanes=2, sequence_length=1, seed=3, epoch=0, state=lanes.state)
    seen = [(sample.episode_id, sample.step) for row in first for sample in row if sample]
    while (original := lanes.next()) is not None:
        resumed = restored.next()
        for left_row, right_row in zip(original, resumed):
            for left, right in zip(left_row, right_row):
                assert (None if left is None else (left.episode_id, left.step)) == (None if right is None else (right.episode_id, right.step))
                if left is not None:
                    seen.append((left.episode_id, left.step))
    assert restored.next() is None
    expected = {(episode["id"], step) for episode in source.episodes() for step in range(episode["steps"])}
    assert len(seen) == len(expected) and set(seen) == expected


def test_actual_recurrent_behavioral_updates_lower_packet_nll():
    mx.random.seed(884)
    config, source = fixture((7,))
    policy = AgentPolicy(config, source.vocabulary)
    trainer = BehaviorTrainer(policy, BehaviorConfig(epochs=25, lanes=1, sequence_length=2, learning_rate=2e-3,
                                                     pretrained_learning_rate=3e-4, freeze_pretrained_epochs=1), dataset_id="fixture")
    before = trainer.evaluate(source, split="train")["meanNLL"]
    losses = [trainer.train_epoch(source).mean_nll for _ in range(25)]
    after = trainer.evaluate(source, split="train")["meanNLL"]
    assert np.isfinite(losses).all()
    assert after < before * 0.55, (before, after)
    assert trainer.updates >= 25
    assert trainer.evaluate(source, split="test")["available"] is False


def test_cancelled_accumulation_checkpoint_resumes_exactly(tmp_path):
    mx.random.seed(812)
    config, source = fixture((1, 2, 3))
    policy = AgentPolicy(config, source.vocabulary)
    training = BehaviorConfig(epochs=2, lanes=1, sequence_length=1, accumulation_chunks=3)
    trainer = BehaviorTrainer(policy, training, dataset_id="fixture")
    def cancelled():
        return trainer.pending_chunks == 1
    with pytest.raises(InterruptedError):
        trainer.train_epoch(source, cancelled=cancelled)
    assert trainer.pending_chunks == 1 and trainer.updates == 0
    directory = tmp_path / str(uuid.uuid4())
    save_checkpoint(directory, policy, kind="behavioral", step=trainer.updates, training_state=trainer.state)
    checkpoint = load_checkpoint(directory, include_training=True)
    resumed = BehaviorTrainer(checkpoint.policy, training, dataset_id="fixture", restored_state=checkpoint.training_state)
    trainer.train_epoch(source)
    resumed.train_epoch(source)
    expected = dict(tree_flatten(policy.parameters()))
    for name, value in tree_flatten(resumed.policy.parameters()):
        np.testing.assert_allclose(np.asarray(value), np.asarray(expected[name]), atol=2e-6, rtol=2e-5, err_msg=name)
    assert trainer.decisions == resumed.decisions and trainer.updates == resumed.updates
    assert trainer.pending_chunks == resumed.pending_chunks == 0


def test_frozen_backbone_optimizer_does_not_age_before_unfreeze():
    config, source = fixture((7,))
    policy = AgentPolicy(config, source.vocabulary)
    trainer = BehaviorTrainer(policy, BehaviorConfig(epochs=2, lanes=1, sequence_length=2), dataset_id="fixture")
    trainer.train_epoch(source)
    for name, optimizer in trainer.optimizer.optimizers.items():
        if name.startswith("pretrained"):
            assert int(optimizer.step) == 0
    trainer.train_epoch(source)
    for name, optimizer in trainer.optimizer.optimizers.items():
        if name.startswith("pretrained"):
            assert int(optimizer.step) > 0
