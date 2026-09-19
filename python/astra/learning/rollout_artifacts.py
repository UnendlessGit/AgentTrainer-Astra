"""Immutable, checked process handoff for externally sampled PPO experience."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np

from astra.checkpoints import _publish, _sync
from astra.actor_progress import validate_actor_progress
from astra.environments.interface import EnvironmentError, EnvironmentSpec, fields, integer, uuid_key
from astra.model.config import ModelConfig
from .reinforcement import (CollectedDecision, CollectedRollout, ObservationImage, ObservationRecord,
                            _frozen_array)
from .rl import Rollout, Transition, RewardWindow, BootstrapObservation, Outcome
from .rollout_store import StoredFrame

MAX_JSON = 1024 * 1024
MAX_PACKAGE = 40 * 1024**3


def encoded(value):
    result=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(result)>MAX_JSON: raise EnvironmentError('One rollout metadata record exceeds its byte limit')
    return result


def _json(data):
    if len(data)>MAX_JSON: raise EnvironmentError('Oversized rollout metadata record')
    def reject(_):raise EnvironmentError('Nonfinite rollout metadata')
    return json.loads(data,parse_constant=reject)


def _file(path,maximum):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    info=os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or not 0<=info.st_size<=maximum:
        os.close(fd); raise EnvironmentError('Invalid rollout artifact file extent')
    return fd,info


def fingerprint(path):
    fd,info=_file(path,MAX_PACKAGE)
    with os.fdopen(fd,'rb') as source: digest=hashlib.file_digest(source,'sha256').hexdigest()
    return {'bytes':info.st_size,'sha256':digest}


def progress(value,run_id):
    try:return validate_actor_progress(value,run_id)
    except (ValueError,TypeError,AttributeError) as error:raise EnvironmentError(str(error)) from error


def _observation(record):
    if record is None:return None
    images=[]
    for image in record.images:
        if not isinstance(image.storage,StoredFrame):raise EnvironmentError('External experience must own spooled source images')
        source=image.storage
        images.append({'offset':source.offset,'bytes':source.nbytes,'shape':source.shape,
                       'checksum':source.checksum.hex(),'metadata':json.loads(image.metadata_json)})
    return {'images':images,'observation_id':record.observation_id,'episode_id':record.episode_id,
        'cutoff_nanos':record.cutoff_nanos,'geometry_revision':record.geometry_revision,
        'controls':json.loads(record.controls_json),'events':json.loads(record.events_json),
        'elapsed_seconds':record.elapsed_seconds,'reset':record.reset,'last_input_nanos':record.last_input_nanos}


def publish_package(working,destination,*,binding,status,actor_progress,control_closure_known,rollout=None,reason=None):
    """Publish once. Audit/journal remain durable even when learning aborted."""
    working,destination=Path(working),Path(destination)
    if rollout is not None:
        if rollout.actor_sampling!=('categorical',1.0,'none',1) or status!='sealed' or binding['purpose']!='learning':
            raise EnvironmentError('Only explicit categorical external experience can be published for learning')
        os.link(rollout.spool.path,working/'frames.bgra')
        with (working/'decisions.ndjson').open('xb') as target:
            for item in rollout.decisions:
                value={'transition':asdict(item.transition),'observation':_observation(item.observation),
                    'packet_fields':item.packet_fields,'state_before':[row.tolist() for row in item.state_before],
                    'bootstrap_observation':_observation(item.bootstrap_observation),'outcome_detail':item.outcome_detail,
                    'commands':json.loads(item.commands_json)}
                target.write(encoded(value)+b'\n')
            target.flush();os.fsync(target.fileno())
        rollout.close()
    for name in ('journal.ndjson','audit.json','frames.bgra','decisions.ndjson'):
        if (working/name).is_file():_sync(working/name)
    artifacts={name:fingerprint(working/name) for name in ('journal.ndjson','audit.json','frames.bgra','decisions.ndjson')
               if (working/name).is_file()}
    manifest={'schemaVersion':1,'id':uuid_key(destination.name),'status':status,'binding':binding,
        'actorProgress':actor_progress,'controlClosureKnown':control_closure_known,'artifacts':artifacts,
        'actorSampling':None if rollout is None else list(rollout.actor_sampling),
        'rolloutID':None if rollout is None else rollout.id,'decisions':0 if rollout is None else len(rollout.decisions),
        'collectionSeconds':0 if rollout is None else rollout.collection_seconds,'reason':reason}
    (working/'manifest.json').write_bytes(encoded(manifest));_sync(working/'manifest.json')
    _sync(working);_publish(working,destination)
    return manifest


def inspect_package(path):
    path=Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():raise EnvironmentError('Invalid rollout package path')
    fd,_=_file(path/'manifest.json',MAX_JSON)
    with os.fdopen(fd,'rb') as source:manifest=_json(source.read())
    fields(manifest,('schemaVersion','id','status','binding','actorProgress','controlClosureKnown','artifacts',
                     'actorSampling','rolloutID','decisions','collectionSeconds','reason'))
    if integer(manifest['schemaVersion'],1,1)!=1 or uuid_key(path.name)!=manifest['id']:
        raise EnvironmentError('Rollout package identity/version mismatch')
    if manifest['status'] not in ('sealed','audited','aborted') or type(manifest['controlClosureKnown']) is not bool:
        raise EnvironmentError('Invalid rollout package completion status')
    artifacts=manifest['artifacts']
    if type(artifacts) is not dict or not {'journal.ndjson','audit.json'}<=artifacts.keys() or set(artifacts)-{'journal.ndjson','audit.json','frames.bgra','decisions.ndjson'}:
        raise EnvironmentError('Unknown or missing rollout artifacts')
    total=0
    for name,expected in artifacts.items():
        fields(expected,('bytes','sha256'));total+=integer(expected['bytes'],MAX_PACKAGE)
        if total>MAX_PACKAGE or fingerprint(path/name)!=expected:raise EnvironmentError('Rollout artifact integrity failed')
    binding=manifest['binding']
    fields(binding,('runID','clockID','policyID','policySignature','actorSourceID','environmentSourceID',
                    'environment','model','training','contextIDs','purpose','previousActorProgress'))
    for name in ('runID','clockID','policyID','actorSourceID','environmentSourceID'):uuid_key(binding[name])
    EnvironmentSpec.from_dict(binding['environment']);ModelConfig.from_dict(binding['model'])
    if manifest['actorProgress'] is not None:progress(manifest['actorProgress'],binding['runID'])
    if binding['previousActorProgress'] is not None:progress(binding['previousActorProgress'],binding['runID'])
    return manifest


class PackageFrames:
    """Read-only source with checked individual frames and a bounded LRU."""
    def __init__(self,path,*,memory_bytes,disk_bytes):
        self._fd,info=_file(path,disk_bytes);self.disk_bytes=info.st_size
        self.memory_limit=memory_bytes;self.metadata_bytes=self.cache_bytes=self.peak_memory_bytes=0
        self._cache=OrderedDict()
    @property
    def closed(self):return self._fd is None
    def reserve_metadata(self,size):
        if self.metadata_bytes+size>self.memory_limit:raise EnvironmentError('Imported rollout metadata exceeds its RAM budget')
        self.metadata_bytes+=size;self._evict(0)
    def _evict(self,size):
        while self._cache and self.cache_bytes+self.metadata_bytes+size>self.memory_limit:
            _,item=self._cache.popitem(last=False);self.cache_bytes-=item.nbytes
    def read(self,frame):
        if self.closed or frame.owner is not self or frame.offset+frame.nbytes>self.disk_bytes:
            raise EnvironmentError('Retired or invalid rollout source range')
        if frame.nbytes+self.metadata_bytes>self.memory_limit:raise EnvironmentError('Imported frame exceeds available rollout RAM')
        if frame.offset in self._cache:
            value=self._cache.pop(frame.offset);self.cache_bytes-=value.nbytes
            if value.shape!=frame.shape or value.nbytes!=frame.nbytes:
                raise EnvironmentError('Imported frame reference disagrees with its cached extent')
            data=memoryview(value).cast('B')
        else:
            self._evict(frame.nbytes);data=os.pread(self._fd,frame.nbytes,frame.offset)
            if len(data)!=frame.nbytes:raise EnvironmentError('Imported frame was truncated')
            value=np.frombuffer(data,dtype=np.uint8).reshape(frame.shape)
        if hashlib.sha256(data).digest()!=frame.checksum:raise EnvironmentError('Imported frame checksum failed')
        self._evict(frame.nbytes);self._cache[frame.offset]=value;self.cache_bytes+=frame.nbytes
        self.peak_memory_bytes=max(self.peak_memory_bytes,self.cache_bytes+self.metadata_bytes)
        return value
    def close(self):
        fd,self._fd=self._fd,None;self._cache.clear();self.cache_bytes=0
        if fd is not None:os.close(fd)


def load_rollout(path,manifest,training):
    if manifest['status']!='sealed' or not manifest['controlClosureKnown'] or not {'frames.bgra','decisions.ndjson'}<=manifest['artifacts'].keys():
        raise EnvironmentError('Only a sealed complete collection can enter PPO')
    if manifest['actorSampling']!=['categorical',1.0,'none',1]:raise EnvironmentError('Imported rollout does not certify categorical sampling')
    count=integer(manifest['decisions'],training.maximum_rollout_decisions,training.rollout_decisions)
    store=PackageFrames(Path(path)/'frames.bgra',memory_bytes=training.maximum_rollout_bytes,
                        disk_bytes=training.maximum_rollout_disk_bytes)
    model=ModelConfig.from_dict(manifest['binding']['model'])
    def observation(value):
        if value is None:return None
        fields(value,('images','observation_id','episode_id','cutoff_nanos','geometry_revision','controls','events',
                      'elapsed_seconds','reset','last_input_nanos'))
        if type(value['images']) is not list or not 1<=len(value['images'])<=model.maximum_surfaces:
            raise EnvironmentError('Invalid imported surface count')
        images=[]
        for image in value['images']:
            fields(image,('offset','bytes','shape','checksum','metadata'))
            frame=StoredFrame(store,image['offset'],image['bytes'],tuple(image['shape']),bytes.fromhex(image['checksum']))
            if frame.offset+frame.nbytes>store.disk_bytes:raise EnvironmentError('Imported frame extent exceeds artifact')
            images.append(ObservationImage(frame,encoded(image['metadata'])))
        return ObservationRecord(tuple(images),uuid_key(value['observation_id']),uuid_key(value['episode_id']),
            integer(value['cutoff_nanos']),integer(value['geometry_revision']),encoded(value['controls']),encoded(value['events']),
            value['elapsed_seconds'],value['reset'],value['last_input_nanos'])
    try:
        decisions=[]
        fd,_=_file(Path(path)/'decisions.ndjson',training.maximum_rollout_bytes)
        with os.fdopen(fd,'rb') as source:
            while line:=source.readline(MAX_JSON+1):
                if len(decisions)>=count:raise EnvironmentError('Rollout decision count is inconsistent')
                data=_json(line);store.reserve_metadata(len(line)*3)
                fields(data,('transition','observation','packet_fields','state_before','bootstrap_observation','outcome_detail','commands'))
                scalar=data['transition'];scalar['reward']=RewardWindow(**scalar['reward'])
                scalar['bootstrap']=None if scalar['bootstrap'] is None else BootstrapObservation(**scalar['bootstrap'])
                scalar['outcome']=Outcome(scalar['outcome']);scalar=Transition(**scalar)
                packet=data['packet_fields']
                from astra.model.actions import PacketBatch
                if type(packet) is not list or len(packet)!=len(PacketBatch.__dataclass_fields__) or any(type(row) is not list or len(row)!=model.packet_capacity+1 or any(type(v) is not int or not -(2**31)<=v<2**31 for v in row) for row in packet):
                    raise EnvironmentError('Invalid imported sampled packet fields')
                states=data['state_before']
                if type(states) is not list or len(states)!=model.recurrent_layers:raise EnvironmentError('Invalid imported anchor count')
                states=tuple(_frozen_array(row,np.float32) for row in states)
                if any(row.shape!=(1,model.recurrent_width) or not np.isfinite(row).all() for row in states):
                    raise EnvironmentError('Invalid imported recurrent anchor')
                decisions.append(CollectedDecision(scalar,observation(data['observation']),tuple(tuple(row) for row in packet),states,
                    observation(data['bootstrap_observation']),data['outcome_detail'],encoded(data['commands'])))
        if len(decisions)!=count:raise EnvironmentError('Imported rollout is incomplete')
        binding=manifest['binding']
        return CollectedRollout(uuid_key(manifest['rolloutID']),Rollout(tuple(row.transition for row in decisions)),tuple(decisions),
            model.signature,EnvironmentSpec.from_dict(binding['environment']).signature,tuple(binding['contextIDs']),
            store.disk_bytes+store.metadata_bytes,manifest['collectionSeconds'],0,store,tuple(manifest['actorSampling']),
            encoded(manifest['actorProgress']))
    except BaseException:
        store.close();raise
