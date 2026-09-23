from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import uuid

import pytest
from astra.environments.interface import EnvironmentError
from astra.learning.feedback_program import automatic_reward
from astra.learning.review_pipeline import verified_source
from astra.learning.rollout_artifacts import fingerprint, inspect_package, load_rollout
from astra.retrospective_feedback import resolve_feedback
from astra.checkpoints import load_checkpoint
from test_collector import collect_fixture
from test_frame_ring import native_ring, ring_fixture_executable
from test_jobs import Worker


def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def program_fixture():
    program={'schemaVersion':1,'id':str(uuid.uuid4()),'name':'Reviewed reward','signals':[],'maximumEpisodeMS':1000,
        'rules':[{'id':str(uuid.uuid4()),'name':'Elapsed reward','kind':'ratePerSecond','amount':10,'maximumDelta':100},
                 {'id':str(uuid.uuid4()),'name':'Good action','kind':'manualMarker','amount':2,'maximumDelta':100}]}
    data=encoded(program)
    return program,{'programBase64':base64.b64encode(data).decode(),'programSHA256':hashlib.sha256(data).hexdigest(),
        'sourceSessionID':str(uuid.uuid4())}


def test_collector_review_to_separate_ppo_preserves_behavior_and_prevents_revision_reuse(native_ring,tmp_path,monkeypatch):
    program,configuration=program_fixture()
    checkpoint,path,manifest,training,_=collect_fixture(native_ring,tmp_path,monkeypatch,purpose='retrospective',retrospective=configuration)
    assert manifest['status']=='awaiting_manual_review' and manifest['behaviorBatchID']==manifest['rolloutID']
    digest=fingerprint(path/'manifest.json')['sha256']
    _,source=verified_source(path,digest)
    interval=source.metadata['intervals'][0];packet=interval['target']['packetID']
    frames=[json.loads(line) for line in (path/'review-frames.ndjson').read_text().splitlines()]
    assert frames[0]['endpoint']['id']==interval['endpointObservationID']
    assert frames[0]['endpoint']['cutoffNanos']==interval['target']['endNanos']
    original=json.loads((path/'decisions.ndjson').read_text())
    assert original['transition']['reward']['value'] is None and original['automaticReward']==1
    with pytest.raises(EnvironmentError,match='sealed'):load_rollout(path,manifest,training)
    authored={'sessionID':str(uuid.uuid4()),'clockID':source.metadata['clockID'],
        'observedNanos':source.metadata['closedNanos']+1,'wallTimeMS':source.metadata['closedWallTimeMS']+1}
    rule=program['rules'][1]['id']
    annotations=[{'id':str(uuid.uuid4()),'sequence':0,'target':interval['target'],'ruleID':rule,'count':1,'authored':authored}]
    reviews=[{'id':str(uuid.uuid4()),'sequence':0,'pairs':[{'packetID':packet,'ruleID':rule}],'authored':authored}]
    directory=tmp_path/'revisions';directory.mkdir()
    def revision(labels,coverage):
        document={'schemaVersion':1,'id':str(uuid.uuid4()),'revision':0,'sourceSHA256':source.sha256,'authored':authored,
            'annotations':labels,'reviews':coverage,'resolved':resolve_feedback(source,authored,labels,coverage)}
        data=encoded(document);(directory/(document['id']+'.json')).write_bytes(data)
        return {'id':document['id'],'sha256':hashlib.sha256(data).hexdigest()}
    complete=revision(annotations,reviews);unknown=revision([],[])
    worker=Worker()
    def run_job(kind,payload):
        run=str(uuid.uuid4());reply=worker.request(kind,payload,run_id=run)
        assert reply['kind']=='ack',reply
        return worker.until(lambda e:e.get('runID')==run and e['kind'] in ('job.completed','job.failed','job.cancelled'))
    try:
        inspected=run_job('feedback.inspect',{'sourcePath':str(path),'manifestSHA256':digest})
        assert inspected['kind']=='job.completed' and inspected['payload']['result']['sourceSHA256']==source.sha256
        def derive(reference):
            destination=tmp_path/str(uuid.uuid4())
            result=run_job('feedback.materialize',{'sourcePath':str(path),'manifestSHA256':digest,'revisionDirectory':str(directory),
                'revisionChain':[reference],'destination':str(destination)})
            return destination,result
        missing,rejected=derive(unknown)
        assert rejected['kind']=='job.failed' and not missing.exists()
        derived,result=derive(complete)
        assert result['kind']=='job.completed',result
        reviewed=inspect_package(derived)
        assert reviewed['rolloutID']==manifest['rolloutID'] and reviewed['actorProgress']==manifest['actorProgress']
        row=json.loads((derived/'decisions.ndjson').read_text());expected=json.loads(json.dumps(original))
        expected['transition']['reward']['value']=3
        assert row==expected
        rollout=load_rollout(derived,reviewed,training)
        try:assert rollout.decisions[0].transition.reward.value==3
        finally:rollout.close()
        destination=tmp_path/str(uuid.uuid4());run=str(uuid.uuid4())
        request=worker.request('train.reinforcement.external',{'checkpointPath':str(checkpoint),'rolloutPath':str(derived),
            'destination':str(destination)},run_id=run)
        assert request['kind']=='ack'
        waiting=worker.until(lambda e:e['kind']=='job.failed' or e['kind']=='job.progress' and e['payload'].get('phase')=='waiting_for_actor_boundary')
        assert waiting['kind']=='job.progress',waiting
        worker.request('job.externalBoundary',{'jobID':request['payload']['jobID'],'auditPath':str(path)},run_id=run)
        done=worker.until(lambda e:e['kind'] in ('job.completed','job.failed'))
        assert done['kind']=='job.completed',done
        saved=load_checkpoint(destination,include_training=True)
        assert saved.training_state['learner']['optimizerUpdates']>0
        assert saved.training_state['consumedRolloutIDs']==[manifest['behaviorBatchID']]
        assert saved.training_state['actorProgress']==manifest['actorProgress']
        other,result=derive(revision([],reviews)) # A different complete judgment cannot reuse the behavior.
        assert result['kind']=='job.completed'
        denied=run_job('train.reinforcement.external',{'checkpointPath':str(checkpoint),'rolloutPath':str(other),
            'destination':str(tmp_path/str(uuid.uuid4()))})
        assert denied['kind']=='job.failed' and denied['payload']['error']['code']=='job.rolloutConsumed'
        assert fingerprint(path/'manifest.json')['sha256']==digest
    finally:worker.close()


def test_deferred_components_are_complete_ordered_and_not_silent_zero():
    program,_=program_fixture();automatic,manual=program['rules']
    value={'clockID':str(uuid.uuid4()),'episodeID':str(uuid.uuid4()),'startNanos':1,'endNanos':100000001,'value':1,
        'automaticComponents':[{'ruleID':automatic['id'],'value':1}],'deferredManualRuleIDs':[manual['id']]}
    automatic_reward(value,program)
    for changed in ({'deferredManualRuleIDs':[]},{'automaticComponents':[]},{'value':0},
                    {'automaticComponents':[{'ruleID':automatic['id'],'value':None}]}):
        with pytest.raises(ValueError):automatic_reward({**value,**changed},program)


def test_subminimum_review_reopens_original_actor_and_combines_complete_clock_bound_fragments(native_ring,ring_fixture_executable,tmp_path,monkeypatch):
    import subprocess
    import numpy as np
    from astra.inference import InferenceSession,InferenceError
    from astra.learning.rl import Rollout,CompleteEpisodeBatch,duration_aware_gae
    from test_frame_ring import _message
    from astra.learning.review_pipeline import materialize,collection_cursor
    from astra.learning.review_batches import combine
    program,configuration=program_fixture()
    checkpoint,first,first_manifest,training,_=collect_fixture(native_ring,tmp_path,monkeypatch,
        purpose='retrospective',retrospective=configuration,rollout_minimum=2)
    reference={'sourcePath':str(first),'manifestSHA256':fingerprint(first/'manifest.json')['sha256']}
    cursor=collection_cursor(reference,load_checkpoint(checkpoint).manifest)
    assert cursor==first_manifest['actorProgress']
    alternate=load_checkpoint(checkpoint).manifest.copy();alternate['id']=str(uuid.uuid4())
    with pytest.raises(EnvironmentError,match='exact original checkpoint'):collection_cursor(reference,alternate)
    folder=tmp_path/'continued-ring';folder.mkdir()
    producer=subprocess.Popen([str(ring_fixture_executable),'--frame-ring',str(folder)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        report=_message(producer);actor=InferenceSession()
        try:
            ready=actor.prepare({'checkpointPath':str(checkpoint),'ring':{'path':report['path'],'ringID':report['reference']['ringID']},
                'deterministic':False,'collection':True,'resumeCollection':reference},run_id=report['reference']['runID'])
            assert ready['resumedActor'] and ready['resumedCollection'] and ready['needsReset']
            assert ready['nextDrawIndex']==cursor['drawIndex']+1 and ready['nextPacketSequence']==cursor['drawIndex']+1
            np.testing.assert_array_equal(np.asarray(actor._key),cursor['rngState'])
        finally:actor.close()
        _,second,second_manifest,_,_=collect_fixture((producer,report),tmp_path,monkeypatch,checkpoint_path=checkpoint,
            previous_actor_progress=cursor,purpose='retrospective',retrospective=configuration,rollout_minimum=2,
            continuation_source=reference,behavior_batch_id=first_manifest['behaviorBatchID'],outcome='truncated')
    finally:
        if producer.poll() is None:producer.kill()
        producer.communicate(timeout=10)
    assert second_manifest['binding']['previousActorProgress']['runID']==first_manifest['binding']['runID']
    assert second_manifest['binding']['runID']!=first_manifest['binding']['runID']
    references=[]
    for path,manifest in ((first,first_manifest),(second,second_manifest)):
        digest=fingerprint(path/'manifest.json')['sha256'];_,source=verified_source(path,digest)
        authored={'sessionID':str(uuid.uuid4()),'clockID':str(uuid.uuid4()),'observedNanos':1,'wallTimeMS':source.metadata['closedWallTimeMS']+1}
        coverage=[{'id':str(uuid.uuid4()),'sequence':0,'authored':authored,
            'pairs':[{'packetID':row['target']['packetID'],'ruleID':program['rules'][1]['id']} for row in source.metadata['intervals']]}]
        document={'schemaVersion':1,'id':str(uuid.uuid4()),'revision':0,'sourceSHA256':source.sha256,'authored':authored,
            'annotations':[],'reviews':coverage,'resolved':resolve_feedback(source,authored,[],coverage)}
        directory=tmp_path/str(uuid.uuid4());directory.mkdir();data=encoded(document)
        (directory/(document['id']+'.json')).write_bytes(data)
        destination=tmp_path/str(uuid.uuid4())
        result=materialize(path,digest,directory,[{'id':document['id'],'sha256':hashlib.sha256(data).hexdigest()}],destination)
        assert not result['learningEligible'] and result['remainingDecisions']==1
        references.append({'path':str(destination),'manifestSHA256':result['manifestSHA256']})
    batch_path=tmp_path/str(uuid.uuid4());combine(references,batch_path)
    batch_manifest=inspect_package(batch_path);rollout=load_rollout(batch_path,batch_manifest,training)
    try:
        assert isinstance(rollout.rollout,CompleteEpisodeBatch)
        assert [row.transition.run_id for row in rollout.decisions]==[first_manifest['binding']['runID'],second_manifest['binding']['runID']]
        with pytest.raises(ValueError,match='mix runs'):Rollout(tuple(row.transition for row in rollout.decisions))
        actual=duration_aware_gae(rollout.rollout)
        for index,fragment in enumerate(rollout.rollout.fragments):
            np.testing.assert_array_equal(actual.advantages[index:index+1],duration_aware_gae(fragment).advantages)
        assert rollout.decisions[1].bootstrap_observation.observation_id==rollout.decisions[1].transition.bootstrap.observation_id
    finally:rollout.close()
    worker=Worker();run=str(uuid.uuid4());destination=tmp_path/str(uuid.uuid4())
    try:
        reply=worker.request('train.reinforcement.external',{'checkpointPath':str(checkpoint),'rolloutPath':str(batch_path),'destination':str(destination)},run_id=run)
        assert reply['kind']=='ack'
        phase=worker.until(lambda e:e['kind']=='job.failed' or e['kind']=='job.progress' and e['payload'].get('phase')=='waiting_for_actor_boundary')
        assert phase['kind']=='job.progress',phase
        worker.request('job.externalBoundary',{'jobID':reply['payload']['jobID'],'auditPath':str(second)},run_id=run)
        result=worker.until(lambda e:e['kind'] in ('job.completed','job.failed'))
        assert result['kind']=='job.completed',result
        saved=load_checkpoint(destination,include_training=True)
        assert saved.training_state['learner']['optimizerUpdates']>0
        assert saved.training_state['actorProgress']==second_manifest['actorProgress']
        assert saved.training_state['consumedRolloutIDs']==[first_manifest['behaviorBatchID']]
    finally:worker.close()


def test_operator_stop_preserves_only_previously_completed_retrospective_episodes(tmp_path):
    from test_rollout_assembler import ActorFixture
    from astra.learning.rollout_assembler import AsyncRolloutAssembler
    from astra.learning.reinforcement import ReinforcementConfig
    program,_=program_fixture();fixture=ActorFixture(empty=True)
    assembly=AsyncRolloutAssembler(spec=fixture.spec,model=fixture.model,
        training=ReinforcementConfig(rollout_decisions=8,sequence_length=2,burn_in=1,epochs=1,effective_batch_decisions=8),
        run_id=fixture.run,clock_id=fixture.clock,policy_id=fixture.policy_id,policy_signature=fixture.signature,
        actor_source_id=fixture.actor_source,environment_source_id=fixture.label_source,audit_path=tmp_path/'audit.json',
        scratch_directory=tmp_path,retrospective_program=program)
    assembly.begin_episode(episode_id=fixture.episode,ready_nanos=fixture.observation.metadata['observedNanos'],
        reset_id=str(uuid.uuid4()),controls_released=True,pending_packets=0,source_id=fixture.label_source)
    first_episode=fixture.episode
    try:
        for _ in range(2):assembly.submit_actor(fixture.sample(),source_id=fixture.actor_source)
        fixture.receipts(assembly)
        first,endpoint=fixture.records
        fixture.evidence(assembly,'environment.reward',{'startNanos':first.observation.cutoff_nanos,'endNanos':endpoint.observation.cutoff_nanos,
            'value':1,'automaticComponents':[{'ruleID':program['rules'][0]['id'],'value':1}],
            'deferredManualRuleIDs':[program['rules'][1]['id']]},first.packet['id'])
        fixture.evidence(assembly,'environment.outcome',{'endNanos':endpoint.observation.cutoff_nanos,'outcome':'terminated'},first.packet['id'])
        fixture.evidence(assembly,'environment.watermark',{'throughNanos':endpoint.observation.cutoff_nanos,
            'throughSequence':fixture.label_sequence-1,'complete':True})
        fixture.reset_episode(assembly)
        assembly.submit_actor(fixture.sample(),source_id=fixture.actor_source)
        assembly.abort_learning();fixture.receipts(assembly);fixture.finish(assembly)
        rollout=assembly.seal()
        try:
            assert len(rollout.decisions)==1 and rollout.decisions[0].transition.episode_id==first_episode
            audit=json.loads((tmp_path/'audit.json').read_text())
            assert audit['episodes'][1]['excludedEpisode'] and audit['episodes'][1]['decisions'][0]['receipt']['status']=='executed'
        finally:rollout.close()
    finally:assembly.close()
