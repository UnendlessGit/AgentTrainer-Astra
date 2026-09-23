from copy import deepcopy
import json
from pathlib import Path
import subprocess
import threading
import uuid

import mlx.core as mx
import numpy as np
import pytest

from astra.checkpoints import load_checkpoint,save_checkpoint
from astra.jobs import JobManager
from astra.model.actions import ActionVocabulary
from astra.model.config import ModelConfig
from astra.model.policy import AgentPolicy
from astra.protocol import Message
from test_collector import collect_fixture
from test_frame_ring import native_ring,ring_fixture_executable,_message
from test_jobs import Worker


def payload(source,audit,destination,resume=False):
    return {'checkpointPath':str(source),'auditPath':str(audit),'destination':str(destination),'resume':resume}


def assert_tree_same(one,two):
    if isinstance(one,mx.array):np.testing.assert_array_equal(np.asarray(one),np.asarray(two))
    elif isinstance(one,dict):
        assert one.keys()==two.keys()
        for key in one:assert_tree_same(one[key],two[key])
    elif isinstance(one,(list,tuple)):
        assert type(one) is type(two) and len(one)==len(two)
        for left,right in zip(one,two,strict=True):assert_tree_same(left,right)
    else:assert one==two


def test_stopped_actor_checkpoint_keeps_weights_and_latest_cursor_without_a_rollout(native_ring,tmp_path,monkeypatch):
    source,audit,manifest,_,_=collect_fixture(native_ring,tmp_path,monkeypatch,purpose='audit')
    worker=Worker();destination=tmp_path/str(uuid.uuid4())
    try:
        result=worker.job('checkpoint.externalBoundary',payload(source,audit,destination))
        assert result['boundaryOnly'] and result['checkpointPublished'] and 'rolloutID' not in result
        assert result['actorProgress']==manifest['actorProgress'] and result['parameterCount']>0
        saved=load_checkpoint(destination,include_training=True);original=load_checkpoint(source)
        assert saved.manifest['artifacts']['policy.safetensors']==original.manifest['artifacts']['policy.safetensors']
        learner=saved.training_state['learner']
        assert learner['iteration']==learner['optimizerUpdates']==learner['decisions']==learner['environmentResets']==0
        assert learner['policyID']==destination.name and saved.training_state['consumedRolloutIDs']==[]
        assert saved.training_state['requiresEnvironmentReset']
        retried=worker.job('checkpoint.externalBoundary',payload(source,audit,destination))
        assert retried['manifest']==result['manifest']
        different=tmp_path/str(uuid.uuid4());run=str(uuid.uuid4())
        reply=worker.request('checkpoint.externalBoundary',payload(source,audit,different),run_id=run)
        assert reply['kind']=='ack'
        failed=worker.until(lambda event:event['kind']=='job.failed' and event.get('runID')==run)
        assert failed['payload']['error']['code']=='job.boundaryConsumed' and not different.exists()
    finally:worker.close()


def test_behavioral_source_starts_fresh_external_optimizer_without_changing_weights(native_ring,tmp_path,monkeypatch):
    policy=AgentPolicy(ModelConfig.test_small(),ActionVocabulary());policy.vision.freeze()
    source=tmp_path/str(uuid.uuid4())
    before=save_checkpoint(source,policy,kind='behavioral',step=17,
        training_state={'kind':'behavioral','epoch':3,'optimizer':{'sentinel':mx.array([3.,5.])}})
    _,audit,_,_,_=collect_fixture(native_ring,tmp_path,monkeypatch,purpose='audit',checkpoint_path=source)
    worker=Worker();destination=tmp_path/str(uuid.uuid4())
    try:
        worker.job('checkpoint.externalBoundary',payload(source,audit,destination))
        saved=load_checkpoint(destination,include_training=True)
        assert saved.manifest['artifacts']['policy.safetensors']==before['artifacts']['policy.safetensors']
        assert saved.training_state['kind']=='reinforcement_external'
        assert saved.training_state['learner']['optimizerUpdates']==0
        assert 'sentinel' not in saved.training_state['learner']['optimizer']
        assert saved.policy.trainable_parameters()['vision']
    finally:worker.close()


def test_existing_external_optimizer_is_preserved_exactly_when_new_run_stops(native_ring,ring_fixture_executable,tmp_path,monkeypatch):
    source,rollout,_,_,_=collect_fixture(native_ring,tmp_path,monkeypatch)
    worker=Worker();trained=tmp_path/str(uuid.uuid4());run=str(uuid.uuid4())
    try:
        started=worker.request('train.reinforcement.external',{'checkpointPath':str(source),'rolloutPath':str(rollout),'destination':str(trained)},run_id=run)
        waiting=worker.until(lambda event:event['kind']=='job.failed' or event['kind']=='job.progress' and event['payload'].get('phase')=='waiting_for_actor_boundary')
        assert waiting['kind']=='job.progress',waiting
        worker.request('job.externalBoundary',{'jobID':started['payload']['jobID'],'auditPath':str(rollout)},run_id=run)
        done=worker.until(lambda event:event['kind'] in ('job.failed','job.completed'))
        assert done['kind']=='job.completed',done
        previous=load_checkpoint(trained,include_training=True)
        assert previous.training_state['learner']['optimizerUpdates']>0
        ring_root=tmp_path/'new-run-ring';ring_root.mkdir()
        process=subprocess.Popen([str(ring_fixture_executable),'--frame-ring',str(ring_root)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            report=_message(process)
            _,audit,manifest,_,_=collect_fixture((process,report),tmp_path,monkeypatch,purpose='audit',checkpoint_path=trained,
                previous_actor_progress=previous.training_state['actorProgress'])
        finally:
            if process.poll() is None:process.kill()
            process.communicate(timeout=10)
        # Authenticated closure is insufficient if its saved task or prior
        # actor cursor is incompatible with the checkpoint being continued.
        original_manifest=(audit/'manifest.json').read_bytes()
        for mismatch in ('training','previousActorProgress'):
            changed=deepcopy(manifest)
            if mismatch=='training':changed['binding']['training']['learning_rate']*=2
            else:changed['binding']['previousActorProgress']['drawIndex']-=1
            (audit/'manifest.json').write_text(json.dumps(changed))
            rejected=tmp_path/str(uuid.uuid4());rejected_run=str(uuid.uuid4())
            assert worker.request('checkpoint.externalBoundary',payload(trained,audit,rejected,True),run_id=rejected_run)['kind']=='ack'
            event=worker.until(lambda value:value['kind']=='job.failed' and value.get('runID')==rejected_run)
            assert event['payload']['error']['code'] in ('job.boundaryMismatch','job.actorProgressMismatch')
            assert not rejected.exists() and not (audit.parent/('.boundary-'+audit.name+'.json')).exists()
        (audit/'manifest.json').write_bytes(original_manifest)
        destination=tmp_path/str(uuid.uuid4())
        result=worker.job('checkpoint.externalBoundary',payload(trained,audit,destination,True))
        saved=load_checkpoint(destination,include_training=True)
        assert result['actorProgress']==manifest['actorProgress']
        assert result['actorProgress']['runID']!=previous.training_state['actorProgress']['runID']
        assert result['actorProgress']['drawIndex']>previous.training_state['actorProgress']['drawIndex']
        expected=deepcopy(previous.training_state['learner']);expected['policyID']=destination.name
        assert_tree_same(expected,saved.training_state['learner'])
        assert saved.training_state['consumedRolloutIDs']==previous.training_state['consumedRolloutIDs']
        assert saved.manifest['artifacts']['policy.safetensors']==previous.manifest['artifacts']['policy.safetensors']
    finally:worker.close()


@pytest.mark.parametrize('mode',['write_failure','cancel_before_commit','cancel_after_commit','cancel_after_result'])
def test_cursor_publication_retry_is_safe_and_never_executes_policy_or_gradients(native_ring,tmp_path,monkeypatch,mode):
    from astra.learning import external_boundary
    source,audit,manifest,_,_=collect_fixture(native_ring,tmp_path,monkeypatch,purpose='audit')
    destination=tmp_path/str(uuid.uuid4());configuration=payload(source,audit,destination)
    terminal=threading.Event();first=True;events=[]
    original_save=external_boundary.save_checkpoint
    def save(*args,**kwargs):
        nonlocal first
        if first and mode=='write_failure':first=False;raise OSError('fixture publication failure')
        result=original_save(*args,**kwargs)
        if first and mode=='cancel_after_commit':
            first=False;manager.cancel(manager._active.identifier,run_id=manager._active.request.run_id)
        return result
    def no_forward(*args,**kwargs):pytest.fail('Cursor-only checkpoint invoked a model forward')
    monkeypatch.setattr(external_boundary,'save_checkpoint',save)
    monkeypatch.setattr(AgentPolicy,'__call__',no_forward)
    def send(kind,value,request=None):
        nonlocal first
        if kind=='job.progress' and value.get('phase')=='checkpointing' and first and mode=='cancel_before_commit':
            first=False;manager.cancel(value['jobID'],run_id=request.run_id)
        if kind in ('job.completed','job.failed','job.cancelled'):
            events.append((kind,value));terminal.set()
    manager=JobManager(send)
    original_execute=manager._execute
    def execute_then_cancel(job):
        nonlocal first
        result=original_execute(job)
        if first and mode=='cancel_after_result':
            first=False;manager.cancel(job.identifier,run_id=job.request.run_id)
        return result
    monkeypatch.setattr(manager,'_execute',execute_then_cancel)
    def execute(sequence):
        terminal.clear();identifier=manager.submit(Message('checkpoint.externalBoundary',sequence,configuration,run_id=str(uuid.uuid4())))
        assert terminal.wait(10);return manager.status(identifier)
    try:
        result=execute(0)
        assert result['status']==('failed' if mode=='write_failure' else 'completed' if mode=='cancel_after_result' else 'cancelled')
        if mode in ('cancel_after_commit','cancel_after_result'):
            assert destination.exists() and result['result']['checkpointPublished']
            assert result['result']['actorProgress']==manifest['actorProgress']
            assert result['result']['cancelled']==(events[-1][0]=='job.cancelled')
        else:assert not destination.exists() and not (result.get('result') or {}).get('checkpointPublished',False)
        assert (audit.parent/('.boundary-'+audit.name+'.json')).exists()
        retried=execute(1)
        assert retried['status']=='completed' and retried['result']['checkpointPublished']
        assert retried['result']['actorProgress']==manifest['actorProgress']
    finally:assert manager.close()
