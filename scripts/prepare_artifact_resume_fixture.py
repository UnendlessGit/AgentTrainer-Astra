#!/usr/bin/env python3
"""Generate a tiny native recording and real paused BC checkpoint for archive interop."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture-executable', type=Path, default=ROOT / '.build/debug/AstraFixture')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists(): parser.error('Choose a new output directory so prior evidence remains intact')
    if not args.fixture_executable.is_file(): parser.error('Build AstraFixture once before preparing this generated fixture')
    output.mkdir(parents=True)
    library = output / 'SourceLibrary'
    recording_root = library / 'Recordings'
    report = subprocess.run([str(args.fixture_executable.resolve()), str(recording_root)], check=True, capture_output=True, text=True)
    native = json.loads(report.stdout)
    recording = Path(native['directory'])

    import mlx.core as mx
    from astra.checkpoints import save_checkpoint
    from astra.data.datasets import RecordingSelection, DatasetReader, build_dataset
    from astra.learning.behavioral import BehaviorConfig, BehaviorTrainer
    from astra.model.actions import ActionVocabulary
    from astra.model.config import ModelConfig
    from astra.model.policy import AgentPolicy
    config = replace(ModelConfig.test_small(), period_ms=20, lead_ms=0)
    vocabulary = ActionVocabulary((13,), (), True, False, True)
    dataset_id, checkpoint_id, agent_id, run_id = [str(uuid.uuid4()) for _ in range(4)]
    dataset = library / 'Datasets' / dataset_id
    build_dataset(dataset, recording_root=recording_root, selections=[RecordingSelection(recording.stem)],
                  config=config, vocabulary=vocabulary, pointer_mode='absolute')
    training = BehaviorConfig(epochs=1, lanes=1, sequence_length=1, accumulation_chunks=3, seed=3511)
    mx.random.seed(training.seed)
    policy = AgentPolicy(config, vocabulary)
    trainer = BehaviorTrainer(policy, training, dataset_id=dataset_id)
    with DatasetReader(dataset, recording_root=recording_root) as reader:
        try: trainer.train_epoch(reader, cancelled=lambda: trainer.pending_chunks == 1)
        except InterruptedError: pass
    if trainer.pending_chunks != 1 or trainer.pending_count <= 0:
        raise RuntimeError('The generated source did not reach a real partially accumulated training boundary')
    checkpoint = library / 'Models' / checkpoint_id
    manifest = save_checkpoint(checkpoint, policy, kind='behavioral', step=trainer.updates, dataset_id=dataset_id,
                              training_state=trainer.state, training_config=asdict(training))
    configuration = {'schemaVersion': 1, 'runID': run_id, 'agentID': agent_id, 'operation': 'train.behavioral',
        'dataset': {'kind': 'recordings', 'path': str(dataset), 'recordingRoot': str(recording_root)},
        'training': asdict(training), 'model': config.to_dict(), 'actions': vocabulary.to_dict(),
        'resume': False, 'verificationMode': False, 'sourceRecordingIDs': [str(uuid.UUID(recording.stem))]}
    job_directory = library / 'Jobs' / run_id
    job_directory.mkdir(parents=True)
    (job_directory / 'configuration.json').write_text(json.dumps(configuration, sort_keys=True, separators=(',', ':')))
    result = {'generatedOnly': True, 'verificationModel': 'numerical_test_small', 'library': str(library),
        'agentID': agent_id, 'runID': run_id, 'recordingID': str(uuid.UUID(recording.stem)),
        'checkpointID': checkpoint_id, 'datasetID': dataset_id, 'policySignature': manifest['policySignature'],
        'parameterCount': config.parameter_count, 'trainingStep': trainer.updates, 'epoch': trainer.epoch,
        'decisions': trainer.decisions, 'pendingChunks': trainer.pending_chunks}
    (output / 'fixture.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__': main()
