"""Recovery regressions for optimizer, trainability, PRNG and load admission."""
from dataclasses import replace
import hashlib
import json
import uuid

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
import pytest

import astra.checkpoints as checkpoints
from astra.checkpoints import CheckpointError, load_checkpoint, restore_mlx_random_state, save_checkpoint
from astra.model.actions import ActionVocabulary
from astra.model.config import ModelConfig
from astra.model.policy import AgentPolicy
from test_policy_core import observations


def _policy():
    return AgentPolicy(ModelConfig.test_small(), ActionVocabulary((4, 13), (0,), True, True, True))


def _step(policy, optimizer, observation):
    def objective(module):
        return mx.mean((module(observation).temporal.value - 1.5) ** 2)
    _, gradient = nn.value_and_grad(policy, objective)(policy)
    optimizer.update(policy, gradient)
    mx.eval(policy.parameters(), optimizer.state)


def _random_draws():
    draws = (mx.random.normal((17,)), mx.random.uniform(shape=(9,)), mx.random.randint(0, 37, (23,)))
    mx.eval(draws)
    return tuple(np.asarray(value).copy() for value in draws)


def test_freeze_then_unfreeze_survives_optimizer_resume(tmp_path):
    mx.random.seed(409)
    policy = _policy()
    observation = observations(batch=1, time=1, surfaces=1)
    optimizer = optim.AdamW(learning_rate=3e-4, bias_correction=True)
    policy.vision.backbone.freeze()
    # Matching leaf names elsewhere must remain trainable after restoration.
    policy.temporal.value_head.freeze(keys="bias", recurse=False, strict=True)
    for step in (1, 2):
        if step == 2:
            policy.vision.backbone.unfreeze()
            # MLX requires new optimizer slots when the trainable tree expands;
            # explicit initialization retains the already trained moments.
            optimizer.init(policy.trainable_parameters())
        _step(policy, optimizer, observation)
        destination = tmp_path / str(uuid.uuid4())
        save_checkpoint(destination, policy, kind="behavioral", step=step,
                        training_state={"optimizer": optimizer.state})
        restored = load_checkpoint(destination, include_training=True)
        expected_trainable = set(dict(tree_flatten(policy.trainable_parameters())))
        assert set(dict(tree_flatten(restored.policy.trainable_parameters()))) == expected_trainable
        assert "temporal.value_head.bias" not in expected_trainable
        assert "actions.operation_head.bias" in expected_trainable
        backbone_key = "vision.backbone.downsamples.0.conv.weight"
        assert (backbone_key in expected_trainable) == (step == 2)
        before = {name: np.asarray(value).copy() for name, value in tree_flatten(policy.parameters())}
        resumed = optim.AdamW(learning_rate=3e-4, bias_correction=True)
        resumed.state = restored.training_state["optimizer"]
        _step(policy, optimizer, observation)
        _step(restored.policy, resumed, observation)
        reference = dict(tree_flatten(policy.parameters()))
        for name, actual in tree_flatten(restored.policy.parameters()):
            np.testing.assert_allclose(np.asarray(actual), np.asarray(reference[name]), atol=1e-7, rtol=1e-6, err_msg=name)
            if name not in expected_trainable:
                np.testing.assert_array_equal(np.asarray(actual), before[name], err_msg=name)
        if step == 2:
            assert np.any(np.asarray(reference[backbone_key]) != before[backbone_key])


@pytest.mark.parametrize("stage", ["save", "load"])
def test_nonfinite_optimizer_state_is_rejected_even_with_valid_artifact_hash(tmp_path, stage):
    policy = _policy()
    optimizer = optim.AdamW(learning_rate=3e-4, bias_correction=True)
    _step(policy, optimizer, observations(batch=1, time=1, surfaces=1))
    destination = tmp_path / str(uuid.uuid4())
    if stage == "save":
        moment = optimizer.state["temporal"]["value_head"]["weight"]["m"]
        optimizer.state["temporal"]["value_head"]["weight"]["m"] = mx.full(moment.shape, float("nan"))
        with pytest.raises(CheckpointError, match="Nonfinite training state"):
            save_checkpoint(destination, policy, kind="behavioral", step=1,
                            training_state={"optimizer": optimizer.state})
        assert not destination.exists()
        assert not list(tmp_path.glob(".checkpoint-*"))
    else:
        manifest = save_checkpoint(destination, policy, kind="behavioral", step=1,
                                   training_state={"optimizer": optimizer.state})
        path = destination / "training.safetensors"
        tensors = mx.load(str(path))
        mx.eval(tensors)  # Materialize before overwriting the lazily read file.
        name = next(name for name, tensor in tensors.items() if tensor.dtype == mx.float32 and tensor.size > 1)
        tensors[name] = mx.full(tensors[name].shape, float("inf"))
        mx.save_safetensors(str(path), tensors)
        manifest["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (destination / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(CheckpointError, match="nonfinite training state"):
            load_checkpoint(destination, include_training=True)


def test_parameter_budget_rejects_before_constructing_any_model(tmp_path, monkeypatch):
    destination = tmp_path / str(uuid.uuid4())
    manifest = save_checkpoint(destination, _policy(), kind="initial", step=0)
    manifest["model"] = replace(ModelConfig.test_small(), recurrent_width=8192).to_dict()
    identity = {"model": manifest["model"], "actions": manifest["actions"], "canonicalizerVersion": manifest["canonicalizerVersion"]}
    manifest["policySignature"] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest))
    constructions = []
    def forbidden_construction(*args, **kwargs):
        constructions.append(True)
        raise AssertionError("Untrusted oversized model reached array allocation")
    monkeypatch.setattr(checkpoints, "AgentPolicy", forbidden_construction)
    with pytest.raises(CheckpointError, match="parameter budget"):
        load_checkpoint(destination)
    assert not constructions


def test_restored_rng_reproduces_subsequent_draws_exactly(tmp_path):
    destination = tmp_path / str(uuid.uuid4())
    policy = _policy()
    mx.random.seed(0xA17C_F08D_38E4_061B)
    _random_draws()  # Save an advanced generator key rather than only a seed.
    save_checkpoint(destination, policy, kind="behavioral", step=12,
                    training_state={"rng": tuple(mx.random.state)})
    expected = _random_draws()
    loaded = load_checkpoint(destination, include_training=True)
    restore_mlx_random_state(loaded.training_state["rng"])
    for reference, actual in zip(expected, _random_draws()):
        np.testing.assert_array_equal(reference, actual)


@pytest.mark.parametrize("construction_fails", [False, True])
def test_loading_does_not_advance_callers_random_generator(tmp_path, monkeypatch, construction_fails):
    destination = tmp_path / str(uuid.uuid4())
    save_checkpoint(destination, _policy(), kind="initial", step=0)
    mx.random.seed(0xCAB9_A3D8_6107_C019)
    _random_draws()
    state = tuple(mx.random.state)
    expected = _random_draws()
    restore_mlx_random_state(state)
    if construction_fails:
        def failing_construction(*args, **kwargs):
            _random_draws()
            raise RuntimeError("injected construction failure")
        monkeypatch.setattr(checkpoints, "AgentPolicy", failing_construction)
        with pytest.raises(RuntimeError, match="injected construction failure"):
            load_checkpoint(destination)
    else:
        load_checkpoint(destination)
    for reference, actual in zip(expected, _random_draws()):
        np.testing.assert_array_equal(reference, actual)
