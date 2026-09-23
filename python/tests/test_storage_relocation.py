"""Independent artifact roots preserve source bytes and exact BC resume state."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest

from astra.checkpoints import load_checkpoint, save_checkpoint
from astra.data.datasets import DatasetReader
from astra.learning.behavioral import BehaviorConfig, BehaviorTrainer
from astra.model.policy import AgentPolicy
from test_jobs import worker
from test_multi_surface_data import CONFIG, VOCABULARY, _dataset, _frame, _recording


def _files(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob('*') if path.is_file()}


def _same_state(left, right, *, exact):
    one, two = dict(tree_flatten(left)), dict(tree_flatten(right))
    assert one.keys() == two.keys()
    for name in one:
        if isinstance(one[name], mx.array):
            if exact: np.testing.assert_array_equal(np.asarray(one[name]), np.asarray(two[name]), err_msg=name)
            else: np.testing.assert_allclose(np.asarray(one[name]), np.asarray(two[name]), atol=2e-6, rtol=2e-5, err_msg=name)
        else: assert one[name] == two[name], name


def test_independent_recording_and_model_moves_resume_the_same_pending_bc_update(worker, tmp_path):
    catalog = tmp_path / 'Catalog'; models = catalog / 'Models'; recordings = catalog / 'Recordings'
    dataset = catalog / 'Datasets' / str(uuid.uuid4()); models.mkdir(parents=True)
    source = _recording(recordings, [_frame('front', 0, 0)])
    _dataset(source, dataset)
    training = BehaviorConfig(epochs=1, lanes=1, sequence_length=1, accumulation_chunks=3, seed=9183)
    mx.random.seed(training.seed)
    policy = AgentPolicy(CONFIG, VOCABULARY)
    trainer = BehaviorTrainer(policy, training, dataset_id=dataset.name)
    with DatasetReader(dataset, recording_root=recordings) as reader:
        with pytest.raises(InterruptedError):
            trainer.train_epoch(reader, cancelled=lambda: trainer.pending_chunks == 1)
    assert trainer.pending_chunks == 1 and trainer.pending_count > 0 and trainer.sampler_state is not None
    paused = models / str(uuid.uuid4())
    saved = save_checkpoint(paused, policy, kind='behavioral', step=trainer.updates, dataset_id=dataset.name,
                            training_state=trainer.state, training_config=asdict(training))
    before = load_checkpoint(paused, include_training=True)
    snapshot = {'model': _files(paused), 'recordings': _files(recordings), 'dataset': _files(dataset)}
    configuration = {'checkpointPath': str(paused), 'dataset': {'kind': 'recordings', 'path': str(dataset),
                     'recordingRoot': str(recordings)}, 'training': asdict(training), 'resume': True}
    original = catalog / 'saved-configuration.json'; original.write_text(json.dumps(configuration, sort_keys=True))
    original_bytes = original.read_bytes()
    baseline_path = models / str(uuid.uuid4())
    worker.job('train.behavioral', {**configuration, 'destination': str(baseline_path)})
    baseline = load_checkpoint(baseline_path, include_training=True)

    # The two user-selected roots are independent of each other and the fixed
    # catalog/dataset location. No immutable reference or source byte is edited.
    moved_models = tmp_path / 'Model disk' / 'Models'
    moved_recordings = tmp_path / 'Recording disk' / 'Recordings'
    moved_models.parent.mkdir(); moved_recordings.parent.mkdir()
    shutil.move(models, moved_models); shutil.move(recordings, moved_recordings)
    assert not models.exists() and not recordings.exists()
    current_checkpoint = moved_models / saved['id']
    restored = load_checkpoint(current_checkpoint, include_training=True)
    assert restored.manifest == before.manifest
    _same_state(restored.training_state, before.training_state, exact=True)
    assert _files(current_checkpoint) == snapshot['model']
    assert _files(moved_recordings) == snapshot['recordings'] and _files(dataset) == snapshot['dataset']

    # This is the native resolver contract: resolve stable IDs against a frozen
    # layout, then pass existing absolute runtime fields. Historical config stays.
    runtime = deepcopy(configuration)
    runtime['checkpointPath'] = str(moved_models / saved['id'])
    runtime['dataset']['path'] = str(catalog / 'Datasets' / saved['datasetID'])
    runtime['dataset']['recordingRoot'] = str(moved_recordings)
    final_path = moved_models / str(uuid.uuid4()); runtime['destination'] = str(final_path)
    result = worker.job('train.behavioral', runtime)
    resumed = load_checkpoint(final_path, include_training=True)
    assert result['manifest']['datasetID'] == saved['datasetID'] and result['provenance'] == 'recorded_demonstrations'
    assert resumed.training_state['decisions'] == baseline.training_state['decisions'] == 5
    assert resumed.training_state['updates'] == baseline.training_state['updates'] == 2
    _same_state(resumed.training_state, baseline.training_state, exact=False)
    _same_state(resumed.policy.parameters(), baseline.policy.parameters(), exact=False)
    assert original.read_bytes() == original_bytes
    assert _files(current_checkpoint) == snapshot['model']
    assert _files(moved_recordings) == snapshot['recordings'] and _files(dataset) == snapshot['dataset']
