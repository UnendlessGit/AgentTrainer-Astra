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

from astra.checkpoints import CheckpointError, load_checkpoint, save_checkpoint, _publish
from astra.model.actions import ActionVocabulary
from astra.model.config import ModelConfig
from astra.model.policy import AgentPolicy
from test_policy_core import observations


def policy():
    return AgentPolicy(ModelConfig.test_small(), ActionVocabulary((4, 13), (0,), True, True, True))


def test_checkpoint_round_trip_preserves_policy_vocabulary_and_rng(tmp_path):
    mx.random.seed(982)
    model = policy()
    observation = observations(batch=1, time=1, surfaces=1)
    encoding = model(observation)
    expected = model.sample(encoding, key=mx.random.key(72))
    state = {"rng": tuple(mx.random.state), "lanes": [{"episode": str(uuid.uuid4()), "offset": 64}],
             "empty": [], "nil": None, "array": np.array([1, 4], dtype=np.int32)}
    destination = tmp_path / str(uuid.uuid4())
    manifest = save_checkpoint(destination, model, kind="behavioral", step=31, training_state=state)
    loaded = load_checkpoint(destination, include_training=True)
    actual = loaded.policy.sample(loaded.policy(observation), key=mx.random.key(72))
    assert loaded.manifest == manifest
    assert loaded.policy.config == model.config
    assert loaded.policy.actions.vocabulary == model.actions.vocabulary
    np.testing.assert_array_equal(np.asarray(expected.packets.operation), np.asarray(actual.packets.operation))
    np.testing.assert_allclose(np.asarray(expected.log_probability), np.asarray(actual.log_probability), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(state["rng"][0]), np.asarray(loaded.training_state["rng"][0]))
    np.testing.assert_array_equal(state["array"], np.asarray(loaded.training_state["array"]))
    assert loaded.training_state["lanes"] == state["lanes"]
    assert load_checkpoint(destination).training_state is None
    assert not list(tmp_path.glob(".checkpoint-*"))
    with pytest.raises(FileExistsError):
        save_checkpoint(destination, model, kind="behavioral", step=32)


def test_adamw_resume_matches_uninterrupted_parameter_update(tmp_path):
    mx.random.seed(389)
    model = policy()
    observation = observations(batch=1, time=2, surfaces=1)
    optimizer = optim.AdamW(learning_rate=3e-4, bias_correction=True)
    def loss_fn(module):
        value = module(observation).temporal.value
        return mx.mean((value - mx.array([[1.0, -0.5]])) ** 2)
    _, gradients = nn.value_and_grad(model, loss_fn)(model)
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state)
    destination = tmp_path / str(uuid.uuid4())
    save_checkpoint(destination, model, kind="behavioral", step=1, training_state={"optimizer": optimizer.state})
    restored = load_checkpoint(destination, include_training=True)
    resumed = optim.AdamW(learning_rate=3e-4, bias_correction=True)
    resumed.state = restored.training_state["optimizer"]
    _, next_gradient = nn.value_and_grad(model, loss_fn)(model)
    _, restored_gradient = nn.value_and_grad(restored.policy, loss_fn)(restored.policy)
    optimizer.update(model, next_gradient)
    resumed.update(restored.policy, restored_gradient)
    mx.eval(model.parameters(), restored.policy.parameters())
    reference = dict(tree_flatten(model.parameters()))
    for name, value in tree_flatten(restored.policy.parameters()):
        np.testing.assert_allclose(np.asarray(value), np.asarray(reference[name]), atol=1e-7, rtol=1e-6, err_msg=name)


def test_integrity_error_precedes_model_use(tmp_path):
    directory = tmp_path / str(uuid.uuid4())
    save_checkpoint(directory, policy(), kind="initial", step=0)
    path = directory / "policy.safetensors"
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(CheckpointError, match="integrity"):
        load_checkpoint(directory)


def test_untrusted_configuration_and_artifact_paths_are_rejected(tmp_path):
    directory = tmp_path / str(uuid.uuid4())
    original = save_checkpoint(directory, policy(), kind="initial", step=0)
    path = directory / "manifest.json"
    manifest = json.loads(json.dumps(original))
    manifest["model"]["period_ms"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(CheckpointError, match="signature"):
        load_checkpoint(directory)
    manifest = json.loads(json.dumps(original))
    manifest["artifacts"]["../external"] = {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    path.write_text(json.dumps(manifest))
    with pytest.raises(CheckpointError, match="artifact set"):
        load_checkpoint(directory)


def test_atomic_publication_refuses_a_competing_empty_directory(tmp_path):
    source, target = tmp_path / "staging", tmp_path / "published"
    source.mkdir(); target.mkdir()
    (source / "manifest.json").write_text("owned staging")
    with pytest.raises(FileExistsError):
        _publish(source, target)
    assert (source / "manifest.json").read_text() == "owned staging"
    assert list(target.iterdir()) == []


def test_nonfinite_weights_never_create_a_visible_checkpoint(tmp_path):
    model = policy()
    model.temporal.value_head.bias = mx.array([float("nan")])
    target = tmp_path / str(uuid.uuid4())
    with pytest.raises(CheckpointError, match="Nonfinite"):
        save_checkpoint(target, model, kind="initial", step=0)
    assert not target.exists()
