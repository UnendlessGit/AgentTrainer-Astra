"""Immutable collector experience → explicit retrospective reward → PPO input.

No authored value can change source pixels, action likelihoods, recurrent state,
execution, outcome, or the actor cursor. Review artifacts contain no model input.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from astra.checkpoints import _publish, _sync
from astra.environments.interface import EnvironmentError, fields, integer, uuid_key
from astra.retrospective_feedback import (VerifiedFeedbackSource, load_revision,
    validate_revision, LoadedFeedbackRevision, _LOADED_KEY, _json, MAX_BYTES)
from .feedback_program import program_binding
from .rollout_artifacts import encoded, fingerprint, _file, MAX_JSON, MAX_PACKAGE

SOURCE_FILES = ('journal.ndjson','audit.json','frames.bgra','decisions.ndjson')
REVIEW_FILES = ('source.json','program.json','review-frames.ndjson')


def _read(path, maximum=MAX_BYTES):
    descriptor, info = _file(path, maximum)
    with os.fdopen(descriptor, 'rb') as source: data = source.read(maximum + 1)
    if len(data) != info.st_size: raise EnvironmentError('Review artifact changed while reading')
    return data


def _large_json(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(data) > MAX_BYTES: raise EnvironmentError('Review projection exceeds its byte limit')
    return data


def _lines(path):
    descriptor, _ = _file(path, MAX_PACKAGE)
    with os.fdopen(descriptor, 'rb') as source:
        while line := source.readline(MAX_JSON + 1):
            if len(line) > MAX_JSON: raise EnvironmentError('Oversized review decision row')
            yield _json(line)


def _trajectory(binding, artifacts):
    return hashlib.sha256(encoded({'binding': binding, 'artifacts': {name: artifacts[name] for name in SOURCE_FILES}})).hexdigest()


def publish_review_projection(working, destination, binding, rollout):
    from .rollout_artifacts import _observation
    program_data, program = program_binding(binding['retrospective'])
    audit = _json(_read(working/'audit.json'))
    intervals = []
    with (working/'review-frames.ndjson').open('xb') as target:
        for item in rollout.decisions:
            transition = item.transition; evidence = _json(item.retrospective_json); endpoint = item.endpoint_observation
            if endpoint is None or endpoint.reset or endpoint.episode_id != transition.episode_id or endpoint.cutoff_nanos != transition.next_decision_nanos:
                raise EnvironmentError('Review requires the exact original pre-reset endpoint observation')
            interval = {'target': {'episodeID': transition.episode_id, 'observationID': transition.observation_id,
                'packetID': transition.packet_id, 'startNanos': transition.decision_nanos, 'endNanos': transition.next_decision_nanos},
                'packetSequence': evidence['packetSequence'], 'drawIndex': evidence['drawIndex'],
                'endpointObservationID': endpoint.observation_id, 'execution': evidence['execution'],
                'outcome': transition.outcome.value, 'eligible': True}
            if transition.outcome.value == 'truncated':
                bootstrap = transition.bootstrap
                interval['bootstrap'] = {'episodeID': bootstrap.episode_id, 'policyID': bootstrap.policy_id,
                    'observationID': bootstrap.observation_id, 'cutoffNanos': bootstrap.cutoff_nanos,
                    'value': bootstrap.value, 'recurrentReset': bootstrap.recurrent_reset}
            intervals.append(interval)
            def projection(record):
                return {'id': record.observation_id, 'cutoffNanos': record.cutoff_nanos,
                        'images': _observation(record)['images']}
            target.write(encoded({'packetID': transition.packet_id, 'observation': projection(item.observation),
                                  'endpoint': projection(endpoint)}) + b'\n')
        target.flush(); os.fsync(target.fileno())
    artifacts = {name: fingerprint(working/name) for name in SOURCE_FILES}
    metadata = {'schemaVersion': 1, 'collectionID': uuid_key(destination.name), 'runID': binding['runID'],
        'clockID': binding['clockID'], 'sourceSessionID': binding['retrospective']['sourceSessionID'],
        'policyID': binding['policyID'], 'programID': program['id'], 'policySignature': binding['policySignature'],
        'trajectorySHA256': _trajectory(binding, artifacts), 'programSHA256': binding['retrospective']['programSHA256'],
        'status': 'awaiting_manual_review', 'controlClosureKnown': True,
        'closedNanos': max(row['stoppedNanos'] for row in audit['episodes']), 'closedWallTimeMS': time.time_ns()//1_000_000,
        'intervals': intervals,
        'manualRules': sorted(({'id': rule['id'], 'kind': 'manualMarker', 'amount': rule['amount']}
            for rule in program['rules'] if rule['kind']=='manualMarker'), key=lambda row: uuid_key(row['id']))}
    data = _large_json(metadata)
    VerifiedFeedbackSource(data, hashlib.sha256(data).hexdigest(), program_data)
    for name, content in (('source.json', data), ('program.json', program_data)):
        with (working/name).open('xb') as target: target.write(content); target.flush(); os.fsync(target.fileno())


def validate_manifest_extension(manifest):
    fields(manifest, ('schemaVersion','id','status','binding','actorProgress','controlClosureKnown','artifacts',
        'actorSampling','rolloutID','decisions','collectionSeconds','reason','behaviorBatchID','reviewSource','revisionChain'))
    binding = manifest['binding']
    if binding['purpose'] != 'retrospective' or uuid_key(manifest['behaviorBatchID']) != uuid_key(binding['behaviorBatchID']):
        raise EnvironmentError('Review package behavior identity mismatch')
    program_binding(binding['retrospective'])
    if manifest['status'] in ('sealed','awaiting_manual_review'):
        if manifest['rolloutID'] != manifest['behaviorBatchID'] or not manifest['controlClosureKnown'] or manifest['actorProgress'] is None or not set((*SOURCE_FILES,*REVIEW_FILES)) <= manifest['artifacts'].keys():
            raise EnvironmentError('Reviewable source requires complete behavior and joined ownership')
    if manifest['status']=='sealed':
        fields(manifest['reviewSource'], ('path','manifestSHA256'))
        if not Path(manifest['reviewSource']['path']).is_absolute() or not manifest['revisionChain'] or 'reward-revision.json' not in manifest['artifacts']:
            raise EnvironmentError('Derived reward input requires its authenticated source and immutable revision')
    elif manifest['reviewSource'] is not None or manifest['revisionChain']:
        raise EnvironmentError('Original collection cannot impersonate a derived reward revision')


def verified_source(path, expected_manifest):
    from .rollout_artifacts import inspect_package
    path = Path(path)
    manifest = inspect_package(path, expected_manifest_sha256=expected_manifest)
    if manifest['schemaVersion'] != 2 or manifest['status'] != 'awaiting_manual_review':
        raise EnvironmentError('Only joined immutable pending-review experience can be annotated')
    source = VerifiedFeedbackSource(_read(path/'source.json'), manifest['artifacts']['source.json']['sha256'], _read(path/'program.json',1_048_576))
    metadata, binding = source.metadata, manifest['binding']
    if metadata['trajectorySHA256'] != _trajectory(binding, manifest['artifacts']) or metadata['collectionID'] != manifest['id'] or any(metadata[name] != binding[name] for name in ('runID','clockID','policyID','policySignature')) or metadata['sourceSessionID'] != binding['retrospective']['sourceSessionID']:
        raise EnvironmentError('Review projection does not identify the immutable collector trajectory')
    if base64.b64encode(_read(path/'program.json')).decode() != binding['retrospective']['programBase64']:
        raise EnvironmentError('Review program bytes differ from collector binding')
    if len(metadata['intervals']) != manifest['decisions']:
        raise EnvironmentError('Review projection and source decisions disagree')
    return manifest, source


def inspect_source(path, manifest_sha256):
    path = Path(path); manifest, source = verified_source(path, manifest_sha256)
    return {'sourcePath': str(path), 'manifestSHA256': manifest_sha256, 'sourceSHA256': source.sha256,
        'metadataPath': str(path/'source.json'), 'programPath': str(path/'program.json'), 'framesPath': str(path/'frames.bgra'),
        'frameIndexPath': str(path/'review-frames.ndjson'), 'artifacts': manifest['artifacts'],
        'behaviorBatchID': manifest['behaviorBatchID'], 'actorProgress': manifest['actorProgress'],
        'decisions': manifest['decisions'], 'minimumDecisions': manifest['binding']['training'].get('rollout_decisions',512),
        'remainingDecisions': max(0,manifest['binding']['training'].get('rollout_decisions',512)-manifest['decisions']),
        'status': 'awaiting_manual_review'}


def _resolved(source, chain, references):
    if type(chain) is not list or not 1<=len(chain)<=4096 or len(chain)!=len(references):
        raise EnvironmentError('Invalid bounded reward revision chain')
    parent = None
    for data, reference in zip(chain, references, strict=True):
        fields(reference,('id','sha256'))
        if type(data) is not str or len(data)>MAX_BYTES*4//3+4:raise EnvironmentError('Oversized embedded reward revision')
        raw = base64.b64decode(data, validate=True)
        if hashlib.sha256(raw).hexdigest() != reference['sha256']: raise EnvironmentError('Embedded reward revision digest mismatch')
        document = _json(raw)
        if uuid_key(document['id']) != uuid_key(reference['id']): raise EnvironmentError('Embedded revision identity mismatch')
        validate_revision(document,source,parent)
        parent = LoadedFeedbackRevision(raw,reference,_key=_LOADED_KEY)
    resolved = parent.document['resolved']
    if any('manualReward' not in row for row in resolved):
        raise EnvironmentError('Every interval and manual rule needs explicit review before PPO')
    return {uuid_key(row['packetID']):row['manualReward'] for row in resolved}


def _rewarded(row, manual):
    result = copy.deepcopy(row)
    packet = uuid_key(row['transition']['packet_id'])
    if row['transition']['reward']['value'] is not None or packet not in manual:
        raise EnvironmentError('Original deferred reward is not explicitly unknown')
    total = row['automaticReward'] + manual[packet]
    if not __import__('math').isfinite(total): raise EnvironmentError('Resolved total reward is not finite')
    result['transition']['reward']['value'] = total
    return result


def materialize(source_path, manifest_sha256, revision_directory, references, destination, cancelled=lambda:False):
    source_path, destination = Path(source_path), Path(destination)
    manifest, source = verified_source(source_path, manifest_sha256)
    if type(references) is not list or not 1<=len(references)<=4096: raise EnvironmentError('Invalid reward revision ancestry')
    chain=[];parent=None;chain_bytes=0
    for reference in references:
        if cancelled(): raise InterruptedError('Reward materialization cancelled')
        parent=load_revision(revision_directory,reference,source,parent)
        chain_bytes+=(len(parent._data)+2)//3*4
        if chain_bytes>MAX_BYTES-1_048_576:raise EnvironmentError('Embedded revision ancestry exceeds its bounded artifact budget')
        chain.append(base64.b64encode(parent._data).decode())
    manual=_resolved(source,chain,references)
    working=destination.parent/('.review-'+str(uuid.uuid4())+'.partial')
    working.mkdir(mode=0o700)
    try:
        for name in (*SOURCE_FILES,*REVIEW_FILES):
            if name!='decisions.ndjson':os.link(source_path/name,working/name)
        with (working/'decisions.ndjson').open('xb') as target:
            for row in _lines(source_path/'decisions.ndjson'):
                if cancelled():raise InterruptedError('Reward materialization cancelled')
                target.write(encoded(_rewarded(row,manual))+b'\n')
            target.flush();os.fsync(target.fileno())
        (working/'reward-revision.json').write_bytes(_large_json({'chain':chain,'references':references}))
        _sync(working/'reward-revision.json')
        derived=copy.deepcopy(manifest)
        derived.update(id=uuid_key(destination.name),status='sealed',reviewSource={'path':str(source_path),'manifestSHA256':manifest_sha256},revisionChain=references)
        derived['artifacts']={name:fingerprint(working/name) for name in (*SOURCE_FILES,*REVIEW_FILES,'reward-revision.json')}
        (working/'manifest.json').write_bytes(encoded(derived));_sync(working/'manifest.json');_sync(working)
        if cancelled():raise InterruptedError('Reward materialization cancelled before publication')
        _publish(working,destination)
        return {'rolloutPath':str(destination),'manifest':derived,'manifestSHA256':fingerprint(destination/'manifest.json')['sha256'],
                'behaviorBatchID':derived['behaviorBatchID'],'learningEligible':derived['decisions']>=derived['binding']['training'].get('rollout_decisions',512),
                'minimumDecisions':derived['binding']['training'].get('rollout_decisions',512),
                'remainingDecisions':max(0,derived['binding']['training'].get('rollout_decisions',512)-derived['decisions'])}
    finally:
        if working.exists():shutil.rmtree(working)


def validate_derived(path, manifest):
    """Re-derive the only allowed modification before learner consumption."""
    source_ref=manifest['reviewSource'];original,source=verified_source(source_ref['path'],source_ref['manifestSHA256'])
    if any(manifest[key]!=original[key] for key in ('binding','actorProgress','controlClosureKnown','actorSampling','rolloutID','decisions','collectionSeconds','behaviorBatchID')):
        raise EnvironmentError('Reviewed input changed immutable behavior identity')
    for name in (*SOURCE_FILES,*REVIEW_FILES):
        if name!='decisions.ndjson' and manifest['artifacts'][name]!=original['artifacts'][name]:
            raise EnvironmentError('Reviewed input changed immutable behavior artifacts')
    revision=_json(_read(Path(path)/'reward-revision.json'))
    fields(revision,('chain','references'))
    if revision['references']!=manifest['revisionChain']:raise EnvironmentError('Revision manifest ancestry mismatch')
    manual=_resolved(source,revision['chain'],revision['references'])
    import itertools
    for before,after in itertools.zip_longest(_lines(Path(source_ref['path'])/'decisions.ndjson'),_lines(Path(path)/'decisions.ndjson')):
        if before is None or after is None or _rewarded(before,manual)!=after:
            raise EnvironmentError('Derived rewards changed source behavior or omit source decisions')
    return Path(source_ref['path']).parent


def collection_cursor(reference, checkpoint_manifest):
    """Resume the original actor stream, requiring the exact original policy ID."""
    fields(reference,('sourcePath','manifestSHA256'))
    manifest,_=verified_source(reference['sourcePath'],reference['manifestSHA256'])
    binding=manifest['binding']
    if (binding['policyID']!=uuid_key(checkpoint_manifest['id']) or
        binding['policySignature']!=checkpoint_manifest['policySignature'] or
        binding['model']!=checkpoint_manifest['model']):
        raise EnvironmentError('Suspended collection requires its exact original checkpoint, not a same-weight alias')
    from astra.actor_progress import validate_actor_progress
    return validate_actor_progress(manifest['actorProgress'],binding['runID'])


def continuation_binding(value):
    reference=value['continuationSource']
    fields(reference,('sourcePath','manifestSHA256'))
    original,_=verified_source(reference['sourcePath'],reference['manifestSHA256'])
    binding=original['binding']
    if uuid_key(value.get('behaviorBatchID'))!=original['behaviorBatchID'] or any(value[key]!=binding[key] for key in
        ('policyID','policySignature','environment','model','training','contextIDs')):
        raise EnvironmentError('Continuation must preserve its original behavior batch, checkpoint and task')
    if value['retrospective']['programSHA256']!=binding['retrospective']['programSHA256']:
        raise EnvironmentError('Continuation changed its frozen reward program')
    if value.get('previousActorProgress') is not None and value['previousActorProgress']!=original['actorProgress']:
        raise EnvironmentError('Continuation cursor must preserve its authenticated original run identity')
    return original['actorProgress']
