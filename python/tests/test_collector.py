from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import uuid

import mlx.core as mx
import numpy as np
import pytest

from astra.checkpoints import save_checkpoint,load_checkpoint
from astra.data.observations import make_observation
from astra.environments.interface import EnvironmentSpec
from astra.learning.reinforcement import ReinforcementConfig
from astra.learning.rollout_artifacts import inspect_package,load_rollout
from astra.model.actions import ActionVocabulary,PacketBatch
from astra.model.config import ModelConfig
from astra.model.policy import AgentPolicy
from astra.protocol import Message
from test_frame_ring import native_ring,ring_fixture_executable,_message
from test_jobs import Worker
import test_jobs


def collector_worker(monkeypatch):
    original=test_jobs.worker_command
    with monkeypatch.context() as patch:
        patch.setattr(test_jobs,'worker_command',lambda:[*original(),'--role','collector'])
        return Worker()


def collect_fixture(native_ring,tmp_path,monkeypatch,*,purpose='learning',abort=False,checkpoint_path=None,previous_actor_progress=None,retrospective=None,continuation_source=None,behavior_batch_id=None,rollout_minimum=1,outcome='terminated'):
    producer,report=native_ring; worker=collector_worker(monkeypatch)
    run=report['reference']['runID'];clock,actor_source,label_source,episode,stream,state_id=(str(uuid.uuid4()) for _ in range(6))
    model=ModelConfig.test_small();vocabulary=ActionVocabulary();mx.random.seed(991)
    policy=AgentPolicy(model,vocabulary);policy.eval();origin=tmp_path/str(uuid.uuid4())
    checkpoint=save_checkpoint(origin,policy,kind='initial',step=0)
    if checkpoint_path is not None:
        origin=checkpoint_path;loaded=load_checkpoint(origin);policy=loaded.policy;policy.eval();checkpoint=loaded.manifest
    if previous_actor_progress is not None:
        if retrospective is None:previous_actor_progress={**previous_actor_progress,'runID':str(uuid.UUID(run))}
        stream=previous_actor_progress['rngStreamID']
    training=ReinforcementConfig(rollout_decisions=rollout_minimum,sequence_length=2,burn_in=1,epochs=1,effective_batch_decisions=2)
    spec=EnvironmentSpec('native-collector-fixture',vocabulary,maximum_episode_ms=1000,
        maximum_observation_bytes=140,maximum_surfaces=1,reward_signature=retrospective['programSHA256'] if retrospective else '1'*64,reset_signature='2'*64)
    destination=tmp_path/str(uuid.uuid4());key=mx.random.key(81) if previous_actor_progress is None else mx.array(previous_actor_progress['rngState'],dtype=mx.uint32)
    generation=1 if previous_actor_progress is None else previous_actor_progress['actorResetGeneration']+1
    first_draw=0 if previous_actor_progress is None else previous_actor_progress['drawIndex']+1
    state=None;packets=[];records=[]
    def request(kind,payload):
        response=worker.request(kind,payload,run_id=run)
        assert response['kind']=='ack',response
        return response
    try:
        assert worker.hello['payload']['role']=='collector'
        request('collector.prepare',{'schemaVersion':2 if retrospective else 1,'clockID':clock,'environment':spec.to_dict(),'model':model.to_dict(),
            'training':asdict(training),'policyID':checkpoint['id'],'policySignature':checkpoint['policySignature'],
            'actorSourceID':actor_source,'environmentSourceID':label_source,'contextIDs':[],'destination':str(destination),
            'rings':[{'path':report['path'],'ringID':report['reference']['ringID']}],'purpose':purpose,
            **({} if previous_actor_progress is None else {'previousActorProgress':previous_actor_progress}),
            **({} if retrospective is None else {'retrospective':retrospective}),
            **({} if continuation_source is None else {'continuationSource':continuation_source}),
            **({} if behavior_batch_id is None else {'behaviorBatchID':behavior_batch_id})})
        start=report['reference']['metadata']['observedNanos']
        request('collector.begin',{'sourceID':label_source,'episodeID':episode,'readyNanos':start,'resetID':str(uuid.uuid4()),
            'controlsReleased':True,'pendingPackets':0})
        for step in range(2):
            reference=report['reference'];metadata=reference['metadata'];cutoff=start+step*100000000
            pixels=np.asarray([((v*17)%256)^(0 if step==0 else 0xA5) for v in range(140)],dtype=np.uint8).reshape(5,7,4)
            controls={'keys':[],'buttons':[],'modifiers':0,'pointer':{'x':-1919,'y':1},'observedNanos':cutoff,'revision':step,'valid':True}
            observed=make_observation([(pixels,metadata)],controls,cutoff_nanos=cutoff,elapsed_seconds=.1,reset=step==0,
                config=model,maximum_timestamp=2**64-1)
            before=state or policy.temporal.initial_state(1);after,sample_key=mx.random.split(key)
            encoding=policy(observed,before);sample=policy.sample(encoding,key=sample_key,greedy=False)
            mx.eval(sample.log_probability,encoding.temporal.value,encoding.temporal.state)
            observation_id,next_state=(str(uuid.uuid4()) for _ in range(2))
            packet={'id':str(uuid.uuid4()),'runID':run,'sequence':first_draw+step,'observationID':observation_id,'geometryRevision':17,
                'executeAtNanos':cutoff+100000000,'durationMs':100,'commands':[]};packets.append(packet)
            collection={'schemaVersion':1,'checkpointID':checkpoint['id'],'policySignature':checkpoint['policySignature'],
                'modelSignature':model.signature,'episodeID':episode,'episodeStep':step,'observationID':observation_id,
                'cutoffNanos':cutoff,'geometryRevision':17,'frameIDs':[metadata['id']],'contextIDs':[],
                'previousStateID':state_id,'nextStateID':next_state,'recurrentReset':step==0,'elapsedSeconds':.1,
                'stateBefore':[np.asarray(row)[0].tolist() for row in before],
                'packetFields':{name:np.asarray(getattr(sample.packets,name)[0]).tolist() for name in PacketBatch.__dataclass_fields__},
                'logProbability':float(sample.log_probability[0]),'value':float(encoding.temporal.value[0,0]),'environmentResets':generation,
                'sampler':{'kind':'categorical','temperature':1,'mixture':'none','version':1,'rngStreamID':stream,'drawIndex':first_draw+step,
                    'stateBefore':np.asarray(key).tolist(),'sampleKey':np.asarray(sample_key).tolist(),'stateAfter':np.asarray(after).tolist()}}
            records.append(collection)
            snapshot={'id':observation_id,'episodeID':episode,'cutoffNanos':cutoff,'geometryRevision':17,
                'frames':[{'metadata':metadata,'reference':reference,'coverageNanos':metadata['observedNanos'],'coverageKind':'frame'}],
                'controlState':controls,'events':[]}
            request('collector.actor',{'sourceID':actor_source,'response':{'packet':packet,'collectionRecord':collection},'observation':snapshot})
            copied=worker.until(lambda event:event['kind'] in ('collector.framesConsumed','collector.fault'))
            assert copied['kind']=='collector.framesConsumed',copied
            producer.stdin.write(json.dumps({'release':copied['payload']['acknowledgements'][0]}).encode()+b'\n');producer.stdin.flush()
            report=_message(producer)
            state,key,state_id=encoding.temporal.state,after,next_state
        assert report['closed'] and not report['pathExists']
        if abort:request('collector.abort',{'reason':'operator cancelled learning'})
        label_sequence=0
        def evidence(kind,payload,packet_id=None):
            nonlocal label_sequence
            message=Message(kind,label_sequence,{'clockID':clock,'episodeID':episode,**payload},request_id=packet_id,run_id=run)
            label_sequence+=1
            request('collector.evidence',{'sourceID':label_source,'message':json.loads(message.encode())})
        for step,packet in enumerate(packets):
            receipt={'packetID':packet['id'],'runID':run,'sequence':first_draw+step,'observedNanos':start+200000000,
                'resultingState':{**controls,'observedNanos':start+200000000},'commandResults':[]}
            evidence('environment.receipt',{'receipt':{**receipt,'status':'admitted'}},packet['id'])
            evidence('environment.receipt',{'receipt':{**receipt,'status':'executed'}},packet['id'])
        if purpose in ('learning','retrospective') and not abort:
            extras={}
            if retrospective:
                import base64
                rules=json.loads(base64.b64decode(retrospective['programBase64']))['rules']
                extras={'automaticComponents':[{'ruleID':rule['id'],'value':1.0} for rule in rules if rule['kind']!='manualMarker'],
                    'deferredManualRuleIDs':[rule['id'] for rule in rules if rule['kind']=='manualMarker']}
            evidence('environment.reward',{'startNanos':start,'endNanos':start+100000000,'value':1.0,**extras},packets[0]['id'])
            evidence('environment.outcome',{'endNanos':start+100000000,'outcome':outcome},packets[0]['id'])
            evidence('environment.watermark',{'throughNanos':start+100000000,'throughSequence':label_sequence-1,'complete':True})
        request('collector.end',{'sourceID':label_source,'episodeID':episode,'stoppedNanos':start+200000000,'lastActorSequence':first_draw+1,
            'controlsReleased':True,'pendingPackets':0})
        request('collector.finish',{'throughSequence':worker.sequence-1})
        terminal=worker.until(lambda event:event['kind'] in ('collector.sealed','collector.audited'))
        assert str(uuid.UUID(terminal['runID']))==str(uuid.UUID(run))
        worker.close()
        manifest=inspect_package(destination)
        return origin,destination,manifest,training,worker.received
    finally:
        if worker.process.poll() is None:worker.close()


@pytest.mark.parametrize('mode',['complete','cancel_at_boundary','missing_boundary','wrong_run','changed_frame'])
def test_native_collector_cpu_lease_to_immutable_rollout_and_separate_learner_job(native_ring,tmp_path,monkeypatch,mode):
    origin,path,manifest,training,events=collect_fixture(native_ring,tmp_path,monkeypatch)
    assert manifest['status']=='sealed' and manifest['decisions']==1
    assert manifest['actorProgress']['drawIndex']==1
    assert not list(path.glob('.astra-rollout-*'))
    rollout=load_rollout(path,manifest,training)
    try:
        assert rollout.decisions[0].observation.pixels.shape==(5,7,4)
        assert int(rollout.decisions[0].observation.pixels[0,0,1])==17
    finally:rollout.close()
    if mode=='changed_frame':
        frame=path/'frames.bgra';data=bytearray(frame.read_bytes());data[0]^=1;frame.write_bytes(data)
    worker=Worker();destination=tmp_path/str(uuid.uuid4());run=str(uuid.uuid4())
    try:
        ack=worker.request('train.reinforcement.external',{'checkpointPath':str(origin),'rolloutPath':str(path),'destination':str(destination),'boundaryTimeoutSeconds':1 if mode=='missing_boundary' else 60},run_id=run)
        assert ack['kind']=='ack';job=ack['payload']['jobID']
        waiting=worker.until(lambda event:event['kind'] in ('job.failed','job.cancelled') or
            event['kind']=='job.progress' and event['payload'].get('phase')=='waiting_for_actor_boundary')
        if mode=='changed_frame':
            assert waiting['kind']=='job.failed' and 'integrity' in waiting['payload']['error']['message']
            assert not destination.exists();return
        assert waiting['kind']=='job.progress',waiting
        assert not destination.exists()
        if mode=='missing_boundary':
            result=worker.until(lambda event:event['kind']=='job.cancelled')
            assert not result['payload']['result']['checkpointPublished'] and not destination.exists();return
        if mode=='wrong_run':
            rejected=worker.request('job.externalBoundary',{'jobID':job,'auditPath':str(path)},run_id=str(uuid.uuid4()))
            assert rejected['kind']=='error' and rejected['payload']['code']=='job.boundaryMismatch'
        if mode=='cancel_at_boundary':
            assert worker.request('cancel',{'jobID':job},run_id=run)['kind']=='ack'
        reply=worker.request('job.externalBoundary',{'jobID':job,'auditPath':str(path)},run_id=run)
        assert reply['kind']=='ack',reply
        result=worker.until(lambda event:event['kind'] in ('job.completed','job.failed','job.cancelled'))
        assert result['kind']==('job.cancelled' if mode=='cancel_at_boundary' else 'job.completed'),result
        saved=load_checkpoint(destination,include_training=True)
        assert result['payload']['result']['parameterCount']==saved.policy.config.parameter_count
        assert saved.training_state['kind']=='reinforcement_external'
        assert saved.training_state['actorProgress']==manifest['actorProgress']
        assert (saved.training_state['learner']['optimizerUpdates']==0) if mode=='cancel_at_boundary' else (saved.training_state['learner']['optimizerUpdates']>0)
        if mode=='cancel_at_boundary':
            from mlx.utils import tree_flatten
            for (_,before),(_,after) in zip(tree_flatten(load_checkpoint(origin).policy.parameters()),tree_flatten(saved.policy.parameters())):
                np.testing.assert_array_equal(np.asarray(before),np.asarray(after))
        assert saved.training_state['learner']['policyID']==destination.name
        assert saved.training_state['requiresEnvironmentReset']
    finally:worker.close()


@pytest.mark.parametrize('purpose,abort,status',[('audit',False,'audited'),('learning',True,'audited')])
def test_continuity_and_aborted_collections_keep_durable_actual_evidence(native_ring,tmp_path,monkeypatch,purpose,abort,status):
    _,path,manifest,_,events=collect_fixture(native_ring,tmp_path,monkeypatch,purpose=purpose,abort=abort)
    assert manifest['status']==status and manifest['controlClosureKnown']
    if abort:
        faults=[event for event in events if event['kind']=='collector.fault']
        assert faults and all(event['payload']['auditContinuable'] is True for event in faults)
    assert manifest['actorProgress']['drawIndex']==1
    assert 'frames.bgra' not in manifest['artifacts']
    entries=[json.loads(line) for line in (path/'journal.ndjson').read_text().splitlines()]
    assert sum(entry['message']['kind']=='collector.actor' for entry in entries)==2
    assert sum(entry['message']['kind']=='collector.evidence' for entry in entries)==4
    assert (path/'audit.json').exists()


def test_external_resume_preserves_rng_and_optimizer_but_requires_a_new_physical_episode(native_ring,ring_fixture_executable,tmp_path,monkeypatch):
    import subprocess
    origin,first,manifest,_,_=collect_fixture(native_ring,tmp_path,monkeypatch)
    worker=Worker()
    def learn(checkpoint,rollout,destination,resume):
        run=str(uuid.uuid4())
        reply=worker.request('train.reinforcement.external',{'checkpointPath':str(checkpoint),'rolloutPath':str(rollout),
            'destination':str(destination),'resume':resume},run_id=run)
        assert reply['kind']=='ack',reply
        phase=worker.until(lambda event:event['kind']=='job.failed' or event['kind']=='job.progress' and event['payload'].get('phase')=='waiting_for_actor_boundary')
        assert phase['kind']=='job.progress',phase
        accepted=worker.request('job.externalBoundary',{'jobID':reply['payload']['jobID'],'auditPath':str(rollout)},run_id=run)
        assert accepted['kind']=='ack',accepted
        result=worker.until(lambda event:event['kind'] in ('job.completed','job.failed'))
        assert result['kind']=='job.completed',result
        return load_checkpoint(destination,include_training=True)
    try:
        saved_path=tmp_path/str(uuid.uuid4());saved=learn(origin,first,saved_path,False)
        ring_dir=tmp_path/'resumed-ring';ring_dir.mkdir()
        producer=subprocess.Popen([str(ring_fixture_executable),'--frame-ring',str(ring_dir)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            report=_message(producer)
            _,second,second_manifest,_,_=collect_fixture((producer,report),tmp_path,monkeypatch,checkpoint_path=saved_path,
                previous_actor_progress=saved.training_state['actorProgress'])
        finally:
            if producer.poll() is None:producer.kill()
            producer.communicate(timeout=10)
        resumed=learn(saved_path,second,tmp_path/str(uuid.uuid4()),True)
        assert resumed.training_state['learner']['iteration']==2
        assert resumed.training_state['learner']['optimizerUpdates']>saved.training_state['learner']['optimizerUpdates']
        assert resumed.training_state['actorProgress']['drawIndex']==3
        assert resumed.training_state['actorProgress']['rngStreamID']==manifest['actorProgress']['rngStreamID']
        assert resumed.training_state['actorProgress']['actorResetGeneration']==2
        assert resumed.training_state['learner']['environmentResets']==0
        assert len(resumed.training_state['consumedRolloutIDs'])==2
    finally:worker.close()


def test_ingress_overflow_preserves_accepted_journal_and_join_does_not_claim_control_release(native_ring,tmp_path,monkeypatch):
    import threading
    import time
    from astra.collector import CollectorManager
    from astra.jobs import JobError
    _,report=native_ring;run=report['reference']['runID'];actor,label,clock=(str(uuid.uuid4()) for _ in range(3))
    model=ModelConfig.test_small();spec=EnvironmentSpec('overflow-audit',ActionVocabulary(),maximum_observation_bytes=140,
        maximum_surfaces=1,reward_signature='1'*64,reset_signature='2'*64)
    destination=tmp_path/str(uuid.uuid4());events=[];ready=threading.Event();entered=threading.Event();release=threading.Event()
    def send(kind,payload,request=None):
        events.append((kind,payload))
        if kind=='ack' and payload.get('status')=='ready':ready.set()
    manager=CollectorManager(send);sequence=0
    def submit(kind,payload):
        nonlocal sequence
        request=Message(kind,sequence,payload,run_id=run,request_id=str(uuid.uuid4()));sequence+=1
        manager.submit(request)
    try:
        submit('collector.prepare',{'schemaVersion':1,'clockID':clock,'environment':spec.to_dict(),'model':model.to_dict(),
            'training':asdict(ReinforcementConfig(rollout_decisions=1)),'policyID':str(uuid.uuid4()),'policySignature':'9'*64,
            'actorSourceID':actor,'environmentSourceID':label,'contextIDs':[],'destination':str(destination),
            'rings':[{'path':report['path'],'ringID':report['reference']['ringID']}],'purpose':'audit'})
        assert ready.wait(3)
        original=manager._apply
        def blocked(request):
            if request.kind=='collector.begin':entered.set();assert release.wait(3)
            return original(request)
        monkeypatch.setattr(manager,'_apply',blocked)
        submit('collector.begin',{'sourceID':label,'episodeID':str(uuid.uuid4()),'readyNanos':1,'resetID':str(uuid.uuid4()),
            'controlsReleased':True,'pendingPackets':0})
        assert entered.wait(3)
        with pytest.raises(JobError,match='overflow'):
            for _ in range(130):submit('collector.abort',{'reason':'stop after this accepted audit entry'})
        release.set();assert manager.close()
        manifest=inspect_package(destination)
        assert manifest['status']=='aborted' and manifest['controlClosureKnown'] is False
        entries=(destination/'journal.ndjson').read_text().splitlines()
        assert len(entries)==129 # prepare, one in-flight begin, 127 admitted messages
        assert any(kind=='collector.fault' and payload['auditContinuable'] is False for kind,payload in events)
        assert not any(kind=='collector.framesConsumed' for kind,_ in events)
        assert manager._bytes==manager._items==0
    finally:release.set();manager.close()


def test_later_validated_continuity_audit_publishes_the_latest_actor_rng(tmp_path):
    from astra.learning.rollout_artifacts import publish_package
    from astra.learning.rollout_assembler import AsyncRolloutAssembler
    from test_rollout_assembler import ActorFixture
    fixture=ActorFixture(empty=True);origin=tmp_path/str(uuid.uuid4())
    checkpoint=save_checkpoint(origin,fixture.policy,kind='initial',step=0)
    fixture.policy_id=checkpoint['id'];fixture.signature=checkpoint['policySignature']
    first_work=tmp_path/'first-working';first_work.mkdir();assembler=fixture.assembler(first_work)
    try:
        for _ in range(3):assembler.submit_actor(fixture.sample(),source_id=fixture.actor_source)
        fixture.receipts(assembler);fixture.labels(assembler);fixture.finish(assembler)
        rollout=assembler.seal();prior=json.loads(rollout.actor_progress_json)
        (first_work/'journal.ndjson').write_bytes(b'{"fixture":"original categorical actor records"}\n')
        binding={'runID':fixture.run,'clockID':fixture.clock,'policyID':fixture.policy_id,'policySignature':fixture.signature,
            'actorSourceID':fixture.actor_source,'environmentSourceID':fixture.label_source,'environment':fixture.spec.to_dict(),
            'model':fixture.model.to_dict(),'training':asdict(assembler.training),'contextIDs':[],'purpose':'learning','previousActorProgress':None}
        first_path=tmp_path/str(uuid.uuid4())
        publish_package(first_work,first_path,binding=binding,status='sealed',actor_progress=prior,control_closure_known=True,rollout=rollout)
    finally:assembler.close()
    worker=Worker();run=str(uuid.uuid4());destination=tmp_path/str(uuid.uuid4())
    try:
        reply=worker.request('train.reinforcement.external',{'checkpointPath':str(origin),'rolloutPath':str(first_path),'destination':str(destination)},run_id=run)
        phase=worker.until(lambda event:event['kind']=='job.failed' or event['kind']=='job.progress' and event['payload'].get('phase')=='waiting_for_actor_boundary')
        assert phase['kind']=='job.progress',phase
        fixture.observation=fixture.environment.reset(seed=1);fixture.episode=fixture.observation.episode_id
        fixture.episode_start_index=len(fixture.records);fixture.state=None;fixture.state_id=str(uuid.uuid4())
        fixture.events=();fixture.last_input=None;fixture.actor_generation+=1;fixture.local_to_global={}
        audit_work=tmp_path/'audit-working';audit_work.mkdir()
        audit=AsyncRolloutAssembler(spec=fixture.spec,model=fixture.model,training=assembler.training,run_id=fixture.run,
            clock_id=fixture.clock,policy_id=fixture.policy_id,policy_signature=fixture.signature,actor_source_id=fixture.actor_source,
            environment_source_id=fixture.label_source,audit_path=audit_work/'audit.json',scratch_directory=audit_work,
            audit_only=True,previous_actor_progress=prior)
        try:
            audit.begin_episode(episode_id=fixture.episode,ready_nanos=fixture.environment._now,reset_id=str(uuid.uuid4()),
                controls_released=True,pending_packets=0,source_id=fixture.label_source)
            for _ in range(3):audit.submit_actor(fixture.sample(),source_id=fixture.actor_source)
            fixture.receipts(audit);fixture.finish(audit)
            report=audit.seal();latest=report['actorProgress']
            assert latest['drawIndex']==5 and prior['drawIndex']==2
            (audit_work/'journal.ndjson').write_bytes(b'{"fixture":"actual continuity decisions and receipts"}\n')
            audit_path=tmp_path/str(uuid.uuid4())
            publish_package(audit_work,audit_path,binding={**binding,'purpose':'audit','previousActorProgress':prior},
                status='audited',actor_progress=latest,control_closure_known=True)
        finally:audit.close()
        accepted=worker.request('job.externalBoundary',{'jobID':reply['payload']['jobID'],'auditPath':str(audit_path)},run_id=run)
        assert accepted['kind']=='ack',accepted
        result=worker.until(lambda event:event['kind'] in ('job.completed','job.failed'))
        assert result['kind']=='job.completed',result
        saved=load_checkpoint(destination,include_training=True)
        assert saved.training_state['actorProgress']==latest
        assert saved.training_state['actorProgress']!=prior
    finally:worker.close()
