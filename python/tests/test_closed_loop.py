from dataclasses import replace
import json
import os
from pathlib import Path
import uuid

from astra.closed_loop import fingerprint, validate_protocol
from astra.model.config import ModelConfig
from astra.model.contexts import vocabulary
from test_jobs import Worker, initial


def test_closed_loop_worker_uses_fixed_trials_and_native_protocol_identity(tmp_path):
    if path := os.environ.get('ASTRA_CLOSED_LOOP_PROTOCOL'):
        exported = json.loads(Path(path).read_text()); protocol = exported['protocol']
        assert fingerprint(validate_protocol(protocol)) == exported['fingerprint']
    else:
        protocol = dict(schemaVersion=1, task='pointing', periodMS=100, leadMS=100, delayMS=2000, cueMS=500,
                        timeLimitMS=200, deterministic=True, policySeed=900000, contextSizes=[], contextVocabulary=None, contextIDs=[],
                        trials=[dict(seed=100000, pixelWidth=64, pixelHeight=64, logicalBounds=[0, 0, 64, 64]),
                                dict(seed=100000, pixelWidth=96, pixelHeight=64, logicalBounds=[-96, 0, 96, 64])])
    model = replace(ModelConfig.test_small(), period_ms=protocol['periodMS'], lead_ms=protocol['leadMS'],
                    context_sizes=tuple(protocol['contextSizes']),
                    context_vocabulary=() if protocol['contextVocabulary'] is None else vocabulary(protocol['contextVocabulary'], protocol['contextSizes']))
    worker = Worker()
    try:
        checkpoint = initial(worker, tmp_path, model=model)
        result = worker.job('evaluate.closedLoop', dict(checkpointPath=str(checkpoint), protocol=protocol))
        assert result['provenance'] == 'practice_closed_loop' and result['checkpointID'] == checkpoint.name
        assert result['protocolFingerprint'] == fingerprint(protocol)
        assert len(result['trials']) == len(protocol['trials']) == 2
        assert all(row['index'] == index and row['seed'] == protocol['trials'][index]['seed'] for index, row in enumerate(result['trials']))
        assert all(row['outcome'] in ('terminated', 'truncated') for row in result['trials'])
        assert all(1 <= row['decisions'] <= 2 and row['virtualDurationMS'] <= 200 for row in result['trials'])
        assert all(type(row['success']) is bool for row in result['trials'])
        if path:
            Path(path).with_name('closed-loop-worker-result.json').write_text(json.dumps(result, indent=2))
    finally:
        worker.close()
    changed = {**protocol, 'policySeed': 123}
    assert fingerprint(changed) == fingerprint(protocol)  # Greedy behavior ignores RNG seeds.
    changed['deterministic'] = False
    assert fingerprint(changed) != fingerprint(protocol)
