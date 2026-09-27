from pathlib import Path
import hashlib
import uuid

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest

from astra.checkpoints import CheckpointError, add_queued_control, load_checkpoint, save_checkpoint
from astra.model.actions import ActionVocabulary
from astra.model.config import ModelConfig
from astra.model.policy import AgentPolicy
from test_jobs import Worker


def test_worker_creates_distinct_queue_warm_start_without_training_or_actor_state(tmp_path: Path):
    source = tmp_path / str(uuid.uuid4())
    policy = AgentPolicy(ModelConfig.test_small(), ActionVocabulary((0,), (), False, False, False))
    original = save_checkpoint(source, policy, kind="behavioral", step=42,
        training_state={"optimizer": {"step": mx.array(42)}, "rng": [123, 456]})
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    destination = tmp_path / str(uuid.uuid4())
    worker = Worker()
    try:
        result = worker.job("checkpoint.addQueuedControl", {
            "checkpointPath": str(source), "destination": str(destination), "seed": 17})
    finally:
        worker.close()
    loaded = load_checkpoint(destination, include_training=True)
    assert loaded.manifest == result["manifest"] and loaded.manifest["id"] != original["id"]
    assert loaded.manifest["parentID"] == original["id"] and loaded.manifest["kind"] == "initial"
    assert loaded.manifest["step"] == 0 and loaded.manifest["policySignature"] != original["policySignature"]
    assert loaded.policy.config.schema_version == 3 and loaded.training_state is None
    assert set(loaded.manifest["artifacts"]) == {"policy.safetensors"}
    tensors = dict(tree_flatten(loaded.policy.parameters()))
    for name, tensor in tree_flatten(policy.parameters()):
        np.testing.assert_array_equal(np.asarray(tensors[name]), np.asarray(tensor))
    assert np.count_nonzero(np.asarray(tensors["temporal.queued_control.output.weight"])) == 0
    provenance = loaded.manifest["trainingConfig"]["warmStart"]
    assert provenance["sourcePolicySHA256"] == original["artifacts"]["policy.safetensors"]["sha256"]
    assert provenance["optimizerAndActorStateRetained"] is False
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    cancelled = tmp_path / str(uuid.uuid4())
    with pytest.raises(InterruptedError):
        add_queued_control(source, cancelled, cancelled=lambda: True)
    assert not cancelled.exists()
    with pytest.raises(CheckpointError, match="already"):
        add_queued_control(destination, tmp_path / str(uuid.uuid4()))
