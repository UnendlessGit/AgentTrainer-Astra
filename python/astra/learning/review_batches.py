"""Complete-episode batching across suspended runs without rewriting provenance."""
from __future__ import annotations
from collections import OrderedDict
import bisect
import os
from pathlib import Path
import shutil
import uuid

from astra.checkpoints import _publish, _sync
from astra.environments.interface import EnvironmentError, fields, integer, uuid_key
from .rollout_artifacts import inspect_package, fingerprint, encoded, PackageFrames, _file, load_rollout
from .review_pipeline import validate_derived
from .rl import CompleteEpisodeBatch
from .reinforcement import CollectedRollout

BASE_FIELDS=('schemaVersion','id','status','binding','actorProgress','controlClosureKnown','artifacts',
             'actorSampling','rolloutID','decisions','collectionSeconds','reason','behaviorBatchID','fragments')
SHARED_BINDING=('policyID','policySignature','environment','model','training','contextIDs')


def _fragments(references):
    if type(references) is not list or not 1<=len(references)<=128:raise EnvironmentError('A reviewed batch requires 1–128 ordered fragments')
    manifests=[];seen=set();previous=None
    for reference in references:
        fields(reference,('path','manifestSHA256'))
        manifest=inspect_package(reference['path'],expected_manifest_sha256=reference['manifestSHA256'])
        if manifest['schemaVersion']!=2 or manifest['status']!='sealed':raise EnvironmentError('Each batch fragment must be independently reviewed and sealed')
        validate_derived(reference['path'],manifest)
        source=manifest['reviewSource']
        if source['manifestSHA256'] in seen:raise EnvironmentError('Repeated behavior fragment cannot enter a batch')
        seen.add(source['manifestSHA256'])
        if previous is not None:
            binding,prior=manifest['binding'],previous['binding']
            if manifest['behaviorBatchID']!=previous['behaviorBatchID'] or any(binding[key]!=prior[key] for key in SHARED_BINDING):
                raise EnvironmentError('Fragments changed the original batch policy or task')
            if binding['continuationSource']!={'sourcePath':previous['reviewSource']['path'],'manifestSHA256':previous['reviewSource']['manifestSHA256']}:
                raise EnvironmentError('Fragments must follow their authenticated original-source continuation chain')
            if binding['previousActorProgress']!=previous['actorProgress']:
                raise EnvironmentError('Fragment continuation changed original actor cursor provenance')
            before,after=previous['actorProgress'],manifest['actorProgress']
            if after['rngStreamID']!=before['rngStreamID'] or after['drawIndex']<=before['drawIndex'] or after['actorResetGeneration']<=before['actorResetGeneration']:
                raise EnvironmentError('Continuation must contain new draws after a fresh physical reset')
        previous=manifest;manifests.append(manifest)
    if manifests[0]['binding']['continuationSource'] is not None:
        raise EnvironmentError('A batch must include its original first fragment; a suffix cannot bypass consumed experience')
    return manifests


def _manifest(destination,references,manifests):
    from astra.jobs import _reinforcement_config
    first,last=manifests[0],manifests[-1];training=_reinforcement_config(first['binding']['training'])
    decisions=sum(row['decisions'] for row in manifests)
    integer(decisions,training.maximum_rollout_decisions,training.rollout_decisions)
    if sum(row['artifacts']['frames.bgra']['bytes'] for row in manifests)>training.maximum_rollout_disk_bytes:
        raise EnvironmentError('Complete-episode batch exceeds its original disk budget')
    return {'schemaVersion':3,'id':uuid_key(Path(destination).name),'status':'sealed','binding':first['binding'],
        'actorProgress':last['actorProgress'],'controlClosureKnown':True,'artifacts':{},'actorSampling':first['actorSampling'],
        'rolloutID':first['behaviorBatchID'],'decisions':decisions,'collectionSeconds':sum(row['collectionSeconds'] for row in manifests),
        'reason':None,'behaviorBatchID':first['behaviorBatchID'],'fragments':references}


def combine(references,destination,cancelled=lambda:False):
    destination=Path(destination);manifests=_fragments(references)
    manifest=_manifest(destination,references,manifests)
    if cancelled():raise InterruptedError('Batch composition cancelled')
    working=destination.parent/('.batch-'+str(uuid.uuid4())+'.partial');working.mkdir(mode=0o700)
    try:
        (working/'manifest.json').write_bytes(encoded(manifest));_sync(working/'manifest.json');_sync(working)
        if cancelled():raise InterruptedError('Batch composition cancelled before publication')
        _publish(working,destination)
        return {'rolloutPath':str(destination),'manifest':manifest,'manifestSHA256':fingerprint(destination/'manifest.json')['sha256'],
                'behaviorBatchID':manifest['behaviorBatchID'],'learningEligible':True}
    finally:
        if working.exists():shutil.rmtree(working)


def inspect_fragment_batch(path,manifest):
    fields(manifest,BASE_FIELDS)
    expected=_manifest(path,manifest['fragments'],_fragments(manifest['fragments']))
    if manifest!=expected:raise EnvironmentError('Reviewed batch manifest differs from its authenticated original fragments')
    return manifest


class JoinedFrames(PackageFrames):
    """One RAM budget and one LRU for independently immutable frame files."""
    def __init__(self,paths,*,memory_bytes,disk_bytes):
        self._files=[];self._starts=[];self.disk_bytes=0;self._fd=None
        self.memory_limit=memory_bytes;self.metadata_bytes=self.cache_bytes=self.peak_memory_bytes=0;self._cache=OrderedDict()
        try:
            for path in paths:
                descriptor,info=_file(path,disk_bytes)
                self._files.append(descriptor);self._starts.append(self.disk_bytes);self.disk_bytes+=info.st_size
                if self.disk_bytes>disk_bytes:raise EnvironmentError('Joined source exceeds its bounded disk budget')
            self._fd=-1 # PackageFrames uses this only as a liveness sentinel; _bytes owns actual descriptors.
        except BaseException:self.close();raise
    def _bytes(self,offset,size):
        index=bisect.bisect_right(self._starts,offset)-1
        limit=self._starts[index+1] if index+1<len(self._starts) else self.disk_bytes
        if index<0 or offset+size>limit:raise EnvironmentError('A frame cannot cross immutable source files')
        return os.pread(self._files[index],size,offset-self._starts[index])
    def close(self):
        self._fd=None
        for descriptor in self._files:os.close(descriptor)
        self._files=[];self._cache.clear();self.cache_bytes=0


def load_fragment_batch(path,manifest,training):
    manifests=_fragments(manifest['fragments'])
    store=JoinedFrames([Path(row['path'])/'frames.bgra' for row in manifest['fragments']],
        memory_bytes=training.maximum_rollout_bytes,disk_bytes=training.maximum_rollout_disk_bytes)
    try:
        rollouts=[]
        for reference,fragment,offset in zip(manifest['fragments'],manifests,store._starts,strict=True):
            rollouts.append(load_rollout(reference['path'],fragment,training,allow_partial=True,shared_store=store,frame_offset=offset))
        batch=CompleteEpisodeBatch(tuple(row.rollout for row in rollouts),tuple(row['binding']['clockID'] for row in manifests))
        first=rollouts[0]
        return CollectedRollout(manifest['rolloutID'],batch,tuple(item for row in rollouts for item in row.decisions),
            first.model_signature,first.environment_signature,first.context_ids,store.disk_bytes+store.metadata_bytes,
            manifest['collectionSeconds'],0,store,first.actor_sampling,encoded(manifest['actorProgress']))
    except BaseException:store.close();raise


def batch_admission(manifest):
    """Return all stable claim roots and the latest actual actor binding."""
    manifests=_fragments(manifest['fragments'])
    parents={Path(row['reviewSource']['path']).parent for row in manifests}
    return sorted(parents,key=str),manifests[-1]['binding']
