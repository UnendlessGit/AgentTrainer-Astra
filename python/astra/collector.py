"""Dedicated CPU-owned collector role; no policy execution or control posting."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import queue
import threading
import time

from astra.environments.interface import EnvironmentError, EnvironmentSpec, fields, integer, uuid_key, same_id
from astra.environments.observation_transport import FrameRingResolver, SnapshotDecoder
from astra.frame_ring import FrameRingReader
from astra.jobs import JobError, _path, _reinforcement_config
from astra.checkpoints import _sync
from astra.learning.rollout_assembler import AsyncRolloutAssembler, AssemblyLimits, ActorRecord, _bounded_size
from astra.learning.rollout_artifacts import encoded, publish_package, progress
from astra.model.config import ModelConfig
from astra.protocol import Message, MAX_MESSAGE_BYTES

COLLECTOR_OPERATIONS=tuple('collector.'+name for name in
    ('prepare','begin','actor','evidence','bootstrap','endpoint','end','abort','finish','status'))


def _wire(message):
    return json.loads(message.encode())


def _message(value):
    fields(value,('version','kind','sequence','payload'),('runID','requestID'))
    return Message.decode(encoded(value)+b'\n')


class CollectorManager:
    """Bounded requests and one CPU I/O owner, independent from the actor."""
    def __init__(self,send):
        self._send=send;self._lock=threading.RLock();self._queue=queue.Queue(maxsize=128)
        self._closing=False;self._bytes=0;self._items=0;self._prepared=False;self._failure=None
        self._configuration=None;self._binding=None;self._assembler=None;self._rings={}
        self._journal=None;self._journal_bytes=0;self._journal_sequence=0;self._last_sequence=None;self._last_durable_sequence=None;self._finish_requested=False
        self._working=self._destination=None;self._manifest=None;self._current_progress=None
        self._episodes={};self._finishing=False;self._aborted=False;self._validated_abort=False;self._finalized=False
        self._active_request=None;self._terminal_request=None;self._status='idle';self._collection=None
        self._thread=threading.Thread(target=self._work,name='Astra collector CPU owner',daemon=True);self._thread.start()

    @property
    def busy(self):
        with self._lock:return self._prepared and not self._finalized

    def submit(self,request):
        request.validate()
        if request.run_id is None or request.request_id is None:raise JobError('collector.identity','Collector requests require run and request UUIDs')
        if request.kind=='collector.status':
            if request.payload:raise JobError('collector.configuration','Status payload must be empty')
            self._send('ack',self.status(),request=request);return
        size=_bounded_size([request.kind,request.sequence,request.payload,request.request_id,request.run_id],4*MAX_MESSAGE_BYTES)
        with self._lock:
            if self._closing or self._finalized or self._finish_requested:raise JobError('collector.closed','Collector no longer accepts source input')
            if request.kind=='collector.prepare':
                if self._prepared:raise JobError('collector.busy','One collection owns this collector process')
                self._validate_prepare(request)
                self._prepared=True;self._configuration=copy.deepcopy(request.payload)
                self._collection=uuid_key(Path(request.payload['destination']).name)
                self._run_id=uuid_key(request.run_id)
            elif not self._prepared or not same_id(request.run_id,self._run_id):
                raise JobError('collector.identity','Collector input belongs to another run or precedes preparation')
            if self._bytes+size>4*MAX_MESSAGE_BYTES or self._items>=128:
                self._failure=EnvironmentError('Collector protocol ingress capacity exceeded')
                raise JobError('collector.overflow','Collector ingress overflow; stop control and retain native unacknowledged evidence')
            self._bytes+=size;self._items+=1
            try:
                owned=Message.decode(request.encode())
                if request.kind!='collector.prepare':self._send('ack',{'collectionID':self._collection,'status':'queued'},request=request)
                self._queue.put_nowait((owned,size))
                if request.kind=='collector.finish':self._finish_requested=True
            except BaseException:
                self._bytes-=size;self._items-=1;raise

    def _validate_prepare(self,request):
        value=request.payload
        version=integer(value.get('schemaVersion'),2,1)
        extras=('retrospective','behaviorBatchID','continuationSource') if version==2 else ()
        fields(value,('schemaVersion','clockID','environment','model','training','policyID','policySignature','actorSourceID',
            'environmentSourceID','contextIDs','destination','rings','purpose'),('limits','previousActorProgress',*extras))
        if version==2:
            if value['purpose']!='retrospective' or 'retrospective' not in value:raise EnvironmentError('Collector schema2 requires explicit retrospective collection')
            from astra.learning.feedback_program import program_binding
            program_binding(value['retrospective'])
            if 'behaviorBatchID' in value:uuid_key(value['behaviorBatchID'])
            if value.get('continuationSource') is not None:
                from astra.learning.review_pipeline import continuation_binding
                continuation_binding(value)
        for name in ('clockID','policyID','actorSourceID','environmentSourceID'):uuid_key(value[name])
        spec=EnvironmentSpec.from_dict(value['environment']);model=ModelConfig.from_dict(value['model'])
        if version==2 and spec.reward_signature!=value['retrospective']['programSHA256']:
            raise EnvironmentError('Frozen program differs from environment reward identity')
        _reinforcement_config(value['training']);AssemblyLimits(**value.get('limits',{})).validate()
        _path(value['destination'],destination=True)
        if not Path(value['destination']).parent.is_dir():raise EnvironmentError('Collector destination parent must exist')
        if value['purpose'] not in (('retrospective',) if version==2 else ('learning','audit')) or type(value['contextIDs']) is not list:
            raise EnvironmentError('Invalid collector purpose or contexts')
        if type(value['rings']) is not list or not 1<=len(value['rings'])<=16:raise EnvironmentError('Collector requires bounded prebound rings')
        ids=set()
        for ring in value['rings']:
            fields(ring,('path','ringID'));_path(ring['path']);key=uuid_key(ring['ringID'])
            if key in ids:raise EnvironmentError('Duplicate collector ring identity')
            ids.add(key)
        if value.get('previousActorProgress') is not None:progress(value['previousActorProgress'],request.run_id if version==1 else None)
        if model.period_ms!=spec.period_ms or model.lead_ms!=spec.lead_ms:raise EnvironmentError('Collector model/environment timing differs')

    def status(self):
        with self._lock:return {'collectionID':self._collection,'status':self._status,
            'learningAborted':self._aborted,'journalSequence':self._journal_sequence,
            'path':None if self._destination is None else str(self._destination),'actorProgress':copy.deepcopy(self._current_progress)}

    def _event(self,kind,payload,request=None):
        scope=request or self._active_request
        if scope is None and kind in ('collector.sealed','collector.audited'):scope=self._terminal_request
        if scope is None:scope=Message('collector.event',0,{},run_id=getattr(self,'_run_id',None))
        self._send(kind,{'collectionID':self._collection,**payload},request=scope)

    def _append(self,request):
        value={'journalSequence':self._journal_sequence,'receivedMonotonicNanos':time.monotonic_ns(),'message':_wire(request)}
        data=encoded(value)+b'\n'
        maximum=self._assembler.limits.maximum_journal_bytes if self._assembler else 256*1024**2
        if self._journal_bytes+len(data)>maximum:raise EnvironmentError('Collector durable journal exhausted its explicit byte capacity')
        if os.statvfs(self._working).f_bavail*os.statvfs(self._working).f_frsize-len(data)<10*1024**3:
            raise EnvironmentError('Collector journal needs at least 10 GiB remaining disk reserve')
        self._journal.write(data);self._journal.flush();os.fsync(self._journal.fileno())
        self._journal_bytes+=len(data);self._journal_sequence+=1;self._last_durable_sequence=request.sequence

    def _prepare(self,request):
        value=self._configuration
        self._destination=Path(value['destination']);self._working=self._destination.parent/('.collector-'+self._collection+'.partial')
        self._working.mkdir(mode=0o700)
        self._journal=(self._working/'journal.ndjson').open('xb')
        _sync(self._working);_sync(self._working.parent)
        self._binding={name:copy.deepcopy(value[name]) for name in ('clockID','policyID','policySignature','actorSourceID',
            'environmentSourceID','environment','model','training','contextIDs','purpose')}
        self._binding.update(runID=self._run_id,previousActorProgress=value.get('previousActorProgress'))
        program=None
        if value['schemaVersion']==2:
            from astra.learning.feedback_program import program_binding
            _,program=program_binding(value['retrospective'])
            self._binding.update(retrospective=copy.deepcopy(value['retrospective']),behaviorBatchID=value.get('behaviorBatchID',self._collection),continuationSource=value.get('continuationSource'))
            if value.get('continuationSource') is not None:
                from astra.learning.review_pipeline import continuation_binding
                self._binding['previousActorProgress']=continuation_binding(value)
        for name in ('clockID','policyID','actorSourceID','environmentSourceID'):self._binding[name]=uuid_key(self._binding[name])
        for ring in value['rings']:
            key=uuid_key(ring['ringID']);self._rings[key]=FrameRingReader(Path(ring['path']),run_id=self._run_id,ring_id=key)
        self._resolver=FrameRingResolver(self._rings)
        self._assembler=AsyncRolloutAssembler(spec=EnvironmentSpec.from_dict(value['environment']),model=ModelConfig.from_dict(value['model']),
            training=_reinforcement_config(value['training']),run_id=self._run_id,clock_id=value['clockID'],policy_id=value['policyID'],
            policy_signature=value['policySignature'],actor_source_id=value['actorSourceID'],environment_source_id=value['environmentSourceID'],
            audit_path=self._working/'audit.json',context_ids=tuple(value['contextIDs']),limits=AssemblyLimits(**value.get('limits',{})),
            scratch_directory=self._working,audit_only=value['purpose']=='audit',previous_actor_progress=self._binding['previousActorProgress'],
            retrospective_program=program,behavior_batch_id=self._binding.get('behaviorBatchID'))
        self._append(request);self._status='collecting'
        self._send('ack',{'collectionID':self._collection,'status':'ready','journalPath':str(self._working/'journal.ndjson'),
            'destination':str(self._destination),'collectionVersion':1},request=request)

    def _consumed(self,observation_id,acknowledgements):
        self._event('collector.framesConsumed',{'observationID':observation_id,'acknowledgements':acknowledgements})

    def _apply(self,request):
        value=request.payload;kind=request.kind
        if kind=='collector.prepare':self._prepare(request);return
        self._append(request)
        if kind=='collector.abort':
            fields(value,('reason',))
            if type(value['reason']) is not str or not 1<=len(value['reason'])<=2048:raise EnvironmentError('Abort needs bounded text')
            self._abort(EnvironmentError(value['reason']),preserve_validation=True);return
        if kind=='collector.finish':
            fields(value,('throughSequence',))
            if value['throughSequence']!=self._last_sequence:raise EnvironmentError('Finish does not join the complete collector request prefix')
            if not self._episodes or not all(self._episodes.values()):raise EnvironmentError('Finish requires explicit joined stop for every begun episode')
            self._finishing=True;self._terminal_request=request
            if self._aborted and not self._validated_abort:self._publish_aborted()
            else:self._assembler.finish_collection()
            return
        if kind=='collector.begin':
            fields(value,('sourceID','episodeID','readyNanos','resetID','controlsReleased','pendingPackets'))
            if not self._aborted or self._validated_abort:self._assembler.begin_episode(episode_id=value['episodeID'],ready_nanos=value['readyNanos'],reset_id=value['resetID'],
                controls_released=value['controlsReleased'],pending_packets=value['pendingPackets'],source_id=value['sourceID'])
            if not same_id(value['sourceID'],self._binding['environmentSourceID']) or value['controlsReleased'] is not True or value['pendingPackets']!=0:
                raise EnvironmentError('Readiness does not certify cleared controls')
            episode=uuid_key(value['episodeID'])
            if episode in self._episodes:raise EnvironmentError('Duplicate physical episode')
            self._episodes[episode]=False
        elif kind=='collector.actor':
            fields(value,('sourceID','response','observation'))
            if not same_id(value['sourceID'],self._binding['actorSourceID']):raise EnvironmentError('Unbound actor source')
            if not self._aborted or self._validated_abort:
                self._assembler.submit_actor_wire(value['response'],value['observation'],source_id=value['sourceID'],
                    resolve_frame=self._resolver,on_consumed=self._consumed)
            else:
                self._assembler.wire_byte_count(value['response'],value['observation'])
                ActorRecord.from_wire(value['response'],value['observation'],spec=self._assembler.spec,
                    resolve_frame=self._resolver,on_consumed=self._consumed)
            collection=value['response']['collectionRecord'];sampler=collection['sampler']
            self._current_progress={'schemaVersion':1,'runID':self._run_id,'rngStreamID':sampler['rngStreamID'],
                'drawIndex':sampler['drawIndex'],'rngState':sampler['stateAfter'],'actorResetGeneration':collection['environmentResets']}
        elif kind=='collector.evidence':
            fields(value,('sourceID','message'))
            if not self._aborted or self._validated_abort:self._assembler.submit_evidence(_message(value['message']),source_id=value['sourceID'])
        elif kind in ('collector.bootstrap','collector.endpoint'):
            endpoint=kind=='collector.endpoint'
            if endpoint and self._binding['purpose']!='retrospective':raise EnvironmentError('Endpoint retention requires retrospective collection')
            if endpoint:fields(value,('sourceID','episodeID','policyID','precedingPacketID','observation'),('value',))
            else:fields(value,('sourceID','episodeID','policyID','precedingPacketID','value','observation'))
            if not self._aborted or self._validated_abort:
                self._assembler.submit_bootstrap_wire(episode_id=value['episodeID'],snapshot=value['observation'],value=value.get('value'),
                    policy_id=value['policyID'],preceding_packet_id=value['precedingPacketID'],source_id=value['sourceID'],
                    resolve_frame=self._resolver,on_consumed=self._consumed,endpoint=endpoint)
            else:
                self._assembler.wire_byte_count({},value['observation'])
                SnapshotDecoder(self._assembler.spec,self._resolver,self._consumed).decode(value['observation'],episode=value['episodeID'])
        elif kind=='collector.end':
            fields(value,('sourceID','episodeID','stoppedNanos','lastActorSequence','controlsReleased','pendingPackets'))
            if not same_id(value['sourceID'],self._binding['environmentSourceID']) or value['controlsReleased'] is not True or type(value['pendingPackets']) is not int or value['pendingPackets']!=0:
                raise EnvironmentError('Stop does not certify cleared controls')
            episode=uuid_key(value['episodeID'])
            if episode not in self._episodes or self._episodes[episode]:raise EnvironmentError('Unknown or repeated episode closure')
            integer(value['stoppedNanos']);integer(value['lastActorSequence'])
            self._episodes[episode]=True
            if not self._aborted or self._validated_abort:self._assembler.close_episode(episode_id=value['episodeID'],stopped_nanos=value['stoppedNanos'],
                last_actor_sequence=value['lastActorSequence'],controls_released=value['controlsReleased'],pending_packets=value['pendingPackets'],source_id=value['sourceID'])
        else:raise EnvironmentError('Unknown collector operation')

    def _abort(self,error,*,preserve_validation=False):
        if self._aborted and not self._validated_abort:return
        self._aborted=True;self._failure=error;self._status='draining_aborted'
        if preserve_validation and self._assembler is not None:
            try:self._assembler.abort_learning()
            except Exception as failure:
                self._failure=failure;self._validated_abort=False;self._assembler.close()
            else:self._validated_abort=True
        elif self._assembler is not None:
            self._validated_abort=False;self._assembler.close()
        self._event('collector.fault',{'code':'collector.admissionFailed','message':str(self._failure)[:2048],
            'learningAborted':True,'auditContinuable':self._validated_abort,
            'journalPath':None if self._working is None else str(self._working/'journal.ndjson')})

    def _publish_aborted(self):
        if self._working is None or self._finalized:return
        audit=self._working/'audit.json'
        if not audit.exists():audit.write_bytes(encoded({'schemaVersion':1,'runID':self._run_id,'aborted':True,
            'reason':str(self._failure)[:2048],'actorProgress':self._current_progress,'note':'Raw journal may contain unvalidated evidence; no cleanup is inferred.'}))
        self._finish_package(None,'aborted',self._current_progress)

    def _finish_package(self,rollout,status,actor_progress):
        if actor_progress is not None:
            try:progress(actor_progress,self._run_id)
            except (ValueError,TypeError,KeyError):actor_progress=None
        self._journal.flush();os.fsync(self._journal.fileno());self._journal.close();self._journal=None
        if self._assembler is not None:self._assembler.close()
        closure=bool(self._episodes) and all(self._episodes.values())
        if status=='sealed' and self._binding['purpose']=='retrospective':status='awaiting_manual_review'
        self._manifest=publish_package(self._working,self._destination,binding=self._binding,status=status,
            actor_progress=actor_progress,control_closure_known=closure,rollout=rollout,
            reason=None if self._failure is None else str(self._failure)[:2048])
        self._finalized=True;self._status=status
        self._event('collector.sealed' if status=='sealed' else 'collector.audited',{'path':str(self._destination),'manifest':self._manifest,
            'actorProgress':actor_progress,'learningEligible':status=='sealed','controlClosureKnown':closure,
            'manifestSHA256':__import__('hashlib').sha256((self._destination/'manifest.json').read_bytes()).hexdigest()})

    def _poll(self):
        if self._failure is not None and not self._aborted:self._abort(self._failure)
        if self._assembler is None or (self._aborted and not self._validated_abort) or self._finalized:return
        ready=self._assembler.take_ready()
        if ready is not None:
            if isinstance(ready,dict):self._finish_package(None,'audited',ready['actorProgress'])
            else:self._finish_package(ready,'sealed',json.loads(ready.actor_progress_json))

    def _work(self):
        try:
            while True:
                try:request,size=self._queue.get(timeout=.05)
                except queue.Empty:
                    if self._closing:break
                    try:self._poll()
                    except Exception as error:self._abort(error)
                    continue
                self._active_request=request
                try:
                    self._apply(request)
                    self._event('collector.applied',{'requestSequence':request.sequence,'journalSequence':self._journal_sequence-1,'status':self._status},request)
                    self._last_sequence=request.sequence
                    self._poll()
                except Exception as error:
                    self._abort(error)
                    if request.kind=='collector.prepare':
                        self._send('error',{'code':'collector.prepareFailed','message':str(error)[:2048],'recoverable':False},request=request)
                    if self._last_durable_sequence==request.sequence:
                        self._event('collector.applied',{'requestSequence':request.sequence,'journalSequence':self._journal_sequence-1,'status':'rejected'},request)
                    self._last_sequence=request.sequence
                finally:
                    with self._lock:self._bytes-=size;self._items-=1
                    self._queue.task_done();self._active_request=None
        finally:
            if not self._finalized and self._working is not None:
                try:
                    self._abort(EnvironmentError('Collector channel closed before completed source publication'))
                    self._publish_aborted()
                except Exception as error:self._failure=error
            if self._assembler is not None:self._assembler.close()
            if self._journal is not None:self._journal.close();self._journal=None
            for reader in self._rings.values():reader.close()

    def close(self):
        self._closing=True;self._thread.join(timeout=30)
        return not self._thread.is_alive() and (self._working is None or self._finalized)
