#!/usr/bin/env python3
"""Resume one generated BC checkpoint after its real native archive round-trip."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'python/tests')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    fixture = args.fixture.resolve()
    source = json.loads((fixture / 'fixture.json').read_text())
    result = json.loads((fixture / 'native-roundtrip-result.json').read_text())
    if source.get('generatedOnly') is not True or result.get('generatedOnly') is not True:
        parser.error('This check accepts the explicit generated fixture only')
    if (fixture / 'worker-resume-result.json').exists(): parser.error('This fixture already has preserved worker evidence')
    original, imported = Path(source['library']).resolve(), Path(result['importedLibrary']).resolve()
    if any(not path.is_relative_to(fixture) for path in (original, imported)):
        parser.error('Generated library paths must remain inside the requested fixture directory')
    for name in ('checkpointID', 'datasetID', 'runID'):
        if result[name] != source[name]: raise ValueError('Native import changed a stable artifact identity')

    import mlx.core as mx
    from mlx.utils import tree_flatten
    import numpy as np
    from astra.checkpoints import load_checkpoint
    from astra.data.datasets import DatasetReader
    from astra.learning.behavioral import BehaviorConfig, BehaviorTrainer
    from test_jobs import Worker
    checkpoint = source['checkpointID']; dataset = source['datasetID']; run = source['runID']
    baseline = load_checkpoint(original / 'Models' / checkpoint, include_training=True)
    copied = load_checkpoint(imported / 'Models' / checkpoint, include_training=True)
    assert baseline.manifest == copied.manifest and baseline.training_state['pendingChunks'] == 1
    configuration_path = imported / 'Jobs' / run / 'configuration.json'
    frozen = configuration_path.read_bytes()
    assert frozen == (original / 'Jobs' / run / 'configuration.json').read_bytes()
    configuration = json.loads(frozen)
    training = BehaviorConfig(**configuration['training'])
    trainer = BehaviorTrainer(baseline.policy, training, dataset_id=dataset, restored_state=baseline.training_state)
    with DatasetReader(original / 'Datasets' / dataset, recording_root=original / 'Recordings') as data:
        trainer.train_epoch(data)
        expected_decisions = data.manifest['steps']
    expected_state = trainer.state
    destination = imported / 'Models' / str(uuid.uuid4())
    worker = Worker()
    try:
        reply = worker.job('train.behavioral', {'checkpointPath': str(imported / 'Models' / checkpoint),
            'dataset': {'kind': 'recordings', 'path': str(imported / 'Datasets' / dataset), 'recordingRoot': str(imported / 'Recordings')},
            'training': asdict(training), 'resume': True, 'destination': str(destination)})
    finally: worker.close()
    resumed = load_checkpoint(destination, include_training=True)
    maximum_error = 0.0
    for expected, actual in [(baseline.policy.parameters(), resumed.policy.parameters()), (expected_state, resumed.training_state)]:
        left, right = dict(tree_flatten(expected)), dict(tree_flatten(actual)); assert left.keys() == right.keys()
        for name, value in left.items():
            if isinstance(value, mx.array):
                one, two = np.asarray(value), np.asarray(right[name])
                if np.issubdtype(one.dtype, np.integer): np.testing.assert_array_equal(one, two, err_msg=name)
                else:
                    np.testing.assert_allclose(one, two, atol=2e-6, rtol=2e-5, err_msg=name)
                    maximum_error = max(maximum_error, float(np.max(np.abs(one - two))) if one.size else 0)
            else: assert value == right[name], name
    assert resumed.training_state['decisions'] == expected_decisions
    assert resumed.manifest['step'] > copied.manifest['step']
    assert resumed.manifest['artifacts']['policy.safetensors'] != copied.manifest['artifacts']['policy.safetensors']
    assert configuration_path.read_bytes() == frozen
    report = {'generatedOnly': True, 'verificationModel': source['verificationModel'], 'nativeRoundTrip': True,
        'sourceCheckpointID': checkpoint, 'publishedCheckpointID': reply['manifest']['id'], 'datasetID': dataset,
        'importedLibrary': str(imported), 'optimizerUpdates': resumed.manifest['step'], 'decisions': expected_decisions,
        'pendingGradientsResumed': True, 'weightsChanged': True, 'matchesUnmovedContinuation': True,
        'integerAndRNGStateExact': True, 'maximumFloatingPointDifference': maximum_error,
        'originalConfigurationSHA256': hashlib.sha256(frozen).hexdigest(), 'workerExitCode': worker.process.returncode}
    (fixture / 'worker-resume-result.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__': main()
