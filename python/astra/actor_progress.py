"""Versioned sampled-stream cursor shared by actor, collector and checkpoints."""
from __future__ import annotations

import uuid


def validate_actor_progress(value, run_id=None):
    required = {'schemaVersion', 'runID', 'rngStreamID', 'drawIndex', 'rngState', 'actorResetGeneration'}
    if type(value) is not dict or value.keys() != required:
        raise ValueError('Actor progress has incompatible fields')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1:
        raise ValueError('Unsupported actor progress schema')
    for name in ('runID', 'rngStreamID'):
        if type(value[name]) is not str or len(value[name]) != 36:
            raise ValueError('Actor progress identity must be a UUID')
        uuid.UUID(value[name])
    if run_id is not None and uuid.UUID(value['runID']) != uuid.UUID(run_id):
        raise ValueError('Actor progress belongs to another run')
    for name in ('drawIndex', 'actorResetGeneration'):
        if type(value[name]) is not int or not 0 <= value[name] < 2**64:
            raise ValueError('Actor progress counter is outside its contract')
    words = value['rngState']
    if type(words) is not list or len(words) != 2 or any(type(word) is not int or not 0 <= word < 2**32 for word in words):
        raise ValueError('Invalid actor progress random key')
    # Schema 1 consumes exactly one stream split and one packet sequence per
    # real policy result. Warmup consumes neither; reset preserves both.
    return {**value, 'rngState': list(words)}
