from __future__ import annotations

from dataclasses import replace
import copy
import json
import threading
import uuid

import mlx.core as mx
import numpy as np
import pytest

from astra.environments.interface import EnvironmentObservation, EnvironmentSpec, SurfaceObservation, EnvironmentError
from astra.environments.external import ExternalEnvironment
from astra.environments.practice import PracticeEnvironment, PracticeConfig
from astra.data.observations import make_observation
from astra.data.actions import decode_commands
from astra.model.actions import PacketBatch, flatten_visual, ActionVocabulary
from astra.model.config import ModelConfig
from astra.model.policy import AgentPolicy
from astra.learning.reinforcement import ReinforcementConfig, ReinforcementTrainer
from astra.learning.rollout_assembler import AsyncRolloutAssembler, AssemblyLimits, ActorRecord
from astra.protocol import Message


class ActorFixture:
    def __init__(self, *, empty=False):
        mx.random.seed(92)
        self.environment=PracticeEnvironment(PracticeConfig(pixel_width=64,pixel_height=64,logical_bounds=(0,0,64,64),
            task='delayed_memory',time_limit_ms=5000))
        self.vocabulary=ActionVocabulary() if empty else self.environment.action_vocabulary
        self.model=ModelConfig.test_small(); self.policy=AgentPolicy(self.model,self.vocabulary); self.policy.eval()
        self.spec=EnvironmentSpec('delayed-evidence-fixture',self.vocabulary,maximum_episode_ms=5000,
            maximum_observation_bytes=64*64*4,maximum_surfaces=1,reward_signature='1'*64,reset_signature='2'*64)
        self.run,self.clock,self.policy_id,self.actor_source,self.label_source,self.stream=(str(uuid.uuid4()) for _ in range(6))
        self.signature='9'*64
        self.key=mx.random.key(21); self.state=None; self.state_id=str(uuid.uuid4())
        self.records=[]; self.results={}; self.label_sequence=0
        self.observation=self.environment.reset(seed=0)
        self.episode=self.observation.episode_id
        self.events=(); self.last_input=None
        self.empty=empty
        self.actor_generation=3; self.episode_start_index=0; self.local_to_global={}

    def assembler(self, path, *, limits=AssemblyLimits()):
        training=ReinforcementConfig(rollout_decisions=2,sequence_length=2,burn_in=1,epochs=1,effective_batch_decisions=8)
        assembly=AsyncRolloutAssembler(spec=self.spec,model=self.model,training=training,run_id=self.run,clock_id=self.clock,
            policy_id=self.policy_id,policy_signature=self.signature,actor_source_id=self.actor_source,
            environment_source_id=self.label_source,audit_path=path/'audit.json',scratch_directory=path,limits=limits)
        assembly.begin_episode(episode_id=self.episode,ready_nanos=self.observation.metadata['observedNanos'],
            reset_id=str(uuid.uuid4()),controls_released=True,pending_packets=0,source_id=self.label_source)
        return assembly

    def sample(self, *, jitter_ns=0):
        current=self.observation; cutoff=current.metadata['observedNanos']; step=len(self.records)
        local_step=step-self.episode_start_index
        observed=EnvironmentObservation(str(uuid.uuid4()),self.episode,cutoff,0,
            (SurfaceObservation(current.pixels,current.metadata,cutoff),),current.control_state)
        elapsed=.1 if local_step==0 else (cutoff-self.records[-1].observation.cutoff_nanos)/1e9
        observation=make_observation([(current.pixels,current.metadata)],current.control_state,cutoff_nanos=cutoff,
            elapsed_seconds=elapsed,reset=local_step==0,config=self.model,executed_events=self.events,last_input_nanos=self.last_input)
        state_before=self.state or self.policy.temporal.initial_state(1)
        before=self.key; after,key=mx.random.split(before)
        encoding=self.policy(observation,state_before); sampled=self.policy.sample(encoding,key=key,greedy=False)
        mx.eval(sampled.packets.operation,sampled.log_probability,encoding.temporal.value,encoding.temporal.state,after,key)
        commands=decode_commands(sampled.packets,config=self.model,vocabulary=self.vocabulary,
            visual=flatten_visual(encoding.visual),surfaces=(current.metadata['surface'],))
        packet={'id':str(uuid.uuid4()),'runID':self.run,'sequence':step,'observationID':observed.id,'geometryRevision':0,
            'executeAtNanos':cutoff+100000000,'durationMs':100,'commands':commands}
        next_state=str(uuid.uuid4())
        record={'schemaVersion':1,'checkpointID':self.policy_id,'policySignature':self.signature,'modelSignature':self.model.signature,
            'episodeID':self.episode,'episodeStep':local_step,'observationID':observed.id,'cutoffNanos':cutoff,'geometryRevision':0,
            'frameIDs':[current.metadata['id']],'contextIDs':[],'previousStateID':self.state_id,'nextStateID':next_state,
            'recurrentReset':local_step==0,'elapsedSeconds':elapsed,'stateBefore':[np.asarray(value)[0].tolist() for value in state_before],
            'packetFields':{name:np.asarray(getattr(sampled.packets,name)[0]).tolist() for name in PacketBatch.__dataclass_fields__},
            'logProbability':float(sampled.log_probability[0]),'value':float(encoding.temporal.value[0,0]),'environmentResets':self.actor_generation,
            'sampler':{'kind':'categorical','temperature':1,'mixture':'none','version':1,'rngStreamID':self.stream,'drawIndex':step,
                       'stateBefore':np.asarray(before).tolist(),'sampleKey':np.asarray(key).tolist(),'stateAfter':np.asarray(after).tolist()}}
        actor=ActorRecord(packet,record,observed,tuple(self.events)); self.records.append(actor)
        self.state,self.key,self.state_id=encoding.temporal.state,after,next_state
        if self.events: self.last_input=max(self.last_input or 0,max(event['observedNanos'] for event in self.events))
        self.local_to_global[self.environment._packet_sequence]=step
        result=self.environment.step(commands)
        for item in result.command_results:
            self.results.setdefault(self.local_to_global[item['packetSequence']],[]).append({key:value for key,value in item.items()
                if key in ('commandIndex','scheduledNanos','postedNanos','status','message')})
        # Empty-action timing fixtures can advance by arbitrary nanoseconds
        # without skipping a scheduled command or rewriting any source time.
        assert not jitter_ns or self.empty
        self.environment._now+=jitter_ns
        self.observation=self.environment._observe(); self.events=tuple(result.raw_events)
        return actor

    def evidence(self, assembly, kind, payload, request=None):
        message=Message(kind,self.label_sequence,{'clockID':self.clock,'episodeID':self.episode,**payload},request_id=request,run_id=self.run)
        self.label_sequence+=1
        assembly.submit_evidence(Message.decode(message.encode()),source_id=self.label_source)

    def receipts(self, assembly):
        result=self.environment.abort('Joined fixture stop')
        for item in result.command_results:
            self.results.setdefault(self.local_to_global[item['packetSequence']],[]).append({key:value for key,value in item.items()
                if key in ('commandIndex','scheduledNanos','postedNanos','status','message')})
        controls=self.environment._observe().control_state
        for index,record in enumerate(self.records[self.episode_start_index:],self.episode_start_index):
            base={'packetID':record.packet['id'],'runID':self.run,'sequence':index,'observedNanos':controls['observedNanos'],
                  'resultingState':controls}
            self.evidence(assembly,'environment.receipt',{'receipt':{**base,'status':'admitted','commandResults':[]}},record.packet['id'])
            values=sorted(self.results.get(index,[]),key=lambda item:item['commandIndex'])
            status='cancelled' if any(item['status']=='cancelled' for item in values) else 'executed'
            payload={'receipt':{**base,'status':status,'commandResults':values}}
            if status=='cancelled':payload['cancellationCause']='episodeBoundary'
            self.evidence(assembly,'environment.receipt',payload,record.packet['id'])

    def labels(self, assembly, *, boundary_index=2, outcome='terminated'):
        records=self.records[self.episode_start_index:]
        for index,record in enumerate(records[:boundary_index]):
            end=records[index+1].observation.cutoff_nanos
            self.evidence(assembly,'environment.reward',{'startNanos':record.observation.cutoff_nanos,'endNanos':end,
                'value':1.0 if index+1==boundary_index else 0.0},record.packet['id'])
            self.evidence(assembly,'environment.outcome',{'endNanos':end,'outcome':outcome if index+1==boundary_index else 'continuing'},record.packet['id'])
        self.evidence(assembly,'environment.watermark',{'throughNanos':records[boundary_index].observation.cutoff_nanos,
            'throughSequence':self.label_sequence-1,'complete':True})

    def reset_episode(self, assembly):
        assembly.close_episode(episode_id=self.episode,stopped_nanos=self.environment._now,last_actor_sequence=len(self.records)-1,
            controls_released=True,pending_packets=0,source_id=self.label_source)
        self.observation=self.environment.reset(seed=1)
        self.episode=self.observation.episode_id; self.episode_start_index=len(self.records)
        self.state=None; self.state_id=str(uuid.uuid4()); self.events=(); self.last_input=None
        self.actor_generation+=1; self.local_to_global={}
        assembly.begin_episode(episode_id=self.episode,ready_nanos=self.environment._now,reset_id=str(uuid.uuid4()),
            controls_released=True,pending_packets=0,source_id=self.label_source)

    def finish(self, assembly):
        assembly.close_episode(episode_id=self.episode,stopped_nanos=self.environment._now,last_actor_sequence=len(self.records)-1,
            controls_released=True,pending_packets=0,source_id=self.label_source)
        assembly.finish_collection()


def test_actor_advances_without_labels_and_real_executed_suffix_is_audited(tmp_path):
    fixture=ActorFixture(); assembler=fixture.assembler(tmp_path)
    try:
        for _ in range(5): assembler.submit_actor(fixture.sample(),source_id=fixture.actor_source)
        assert len(fixture.records)==5  # All actions happened before any reward was supplied.
        fixture.receipts(assembler); fixture.finish(assembler); fixture.labels(assembler)
        rollout=assembler.seal()
        assert len(rollout.decisions)==2 and rollout.actor_sampling==('categorical',1.0,'none',1)
        report=json.loads((tmp_path/'audit.json').read_text())
        suffix=report['episodes'][0]['excludedDecisions']
        assert len(suffix)==3 and report['episodes'][0]['postTerminalEffects']
        assert any(item['status']=='posted' for row in suffix for item in row['receipt']['commandResults'])
        passive=ExternalEnvironment(fixture.spec,run_id=fixture.run,clock_id=fixture.clock,
            emit=lambda *_:None,resolve_frame=lambda *_:None)
        trainer=ReinforcementTrainer(fixture.policy,passive,assembler.training,policy_id=fixture.policy_id)
        trainer.admit_external_rollout(rollout)
        result=trainer.update(rollout)
        assert result.optimizer_updates>0 and result.maximum_accepted_kl<=.02
        assert trainer.pending_policy_id is not None
        with pytest.raises(RuntimeError,match='pending'): trainer.admit_external_rollout(rollout)
        trainer.stop()
    finally: assembler.close()


def test_actual_noninteger_nanosecond_cutoffs_drive_rewards_and_gae_without_retiming(tmp_path):
    fixture=ActorFixture(empty=True); assembler=fixture.assembler(tmp_path)
    try:
        for delay in (13,71,37,19): assembler.submit_actor(fixture.sample(jitter_ns=delay),source_id=fixture.actor_source)
        fixture.receipts(assembler); fixture.finish(assembler); fixture.labels(assembler,outcome='truncated')
        rollout=assembler.seal()
        durations=[row.transition.next_decision_nanos-row.transition.decision_nanos for row in rollout.decisions]
        assert durations==[100000013,100000071]
        assert rollout.decisions[-1].bootstrap_observation.cutoff_nanos==fixture.records[2].observation.cutoff_nanos
        assert [row.transition.packet_id for row in rollout.decisions]==[record.packet['id'] for record in fixture.records[:2]]
        for row,record in zip(rollout.decisions,fixture.records):
            assert row.transition.old_log_probability==record.collection['logProbability']
        rollout.close()
    finally: assembler.close()


@pytest.mark.parametrize('change',[{'kind':'greedy'},{'temperature':.5},{'mixture':'epsilon-greedy'}])
def test_sampler_contract_rejects_off_policy_modes_even_with_matching_soft_logp(tmp_path,change):
    fixture=ActorFixture(); assembler=fixture.assembler(tmp_path)
    try:
        record=fixture.sample(); record.collection['sampler'].update(change)
        assembler.submit_actor(record,source_id=fixture.actor_source)
        with pytest.raises(EnvironmentError,match='categorical'):
            assembler.finish_collection()
    finally: assembler.close()


def test_policy_gate_keeps_continuity_episodes_out_of_training_and_activates_only_after_reset():
    from astra.learning.policy_activation import PolicyActivationGate
    old,new,episode,audit,next_episode,rollout=(str(uuid.uuid4()) for _ in range(6))
    gate=PolicyActivationGate(old)
    assert gate.confirm_reset(episode_id=episode,reset_id=str(uuid.uuid4()),policy_id=old,controls_released=True,pending_packets=0)
    gate.confirm_stop(episode_id=episode,controls_released=True,pending_packets=0)
    gate.begin_learning(rollout_id=rollout,policy_id=old)
    assert not gate.confirm_reset(episode_id=audit,reset_id=str(uuid.uuid4()),policy_id=old,controls_released=True,pending_packets=0)
    gate.complete_learning(rollout_id=rollout,proposed_policy_id=new)
    assert not gate.validate_actor(episode_id=audit,policy_id=old)
    with pytest.raises(EnvironmentError,match='outside'):
        gate.validate_actor(episode_id=audit,policy_id=new)
    with pytest.raises(EnvironmentError,match='one learning'):
        gate.begin_learning(rollout_id=str(uuid.uuid4()),policy_id=old)
    gate.confirm_stop(episode_id=audit,controls_released=True,pending_packets=0)
    assert gate.confirm_reset(episode_id=next_episode,reset_id=str(uuid.uuid4()),policy_id=new,controls_released=True,pending_packets=0)
    assert gate.pending_policy_id is None and gate.policy_id==new


def test_evidence_can_arrive_before_actor_records_without_losing_producer_order(tmp_path):
    fixture=ActorFixture(empty=True); assembler=fixture.assembler(tmp_path)
    try:
        records=[fixture.sample() for _ in range(4)]
        fixture.receipts(assembler); fixture.labels(assembler)
        for record in records: assembler.submit_actor(record,source_id=fixture.actor_source)
        fixture.finish(assembler)
        rollout=assembler.seal(); assert len(rollout.decisions)==2
        rollout.close()
    finally: assembler.close()


def test_missing_coverage_or_unknown_labels_never_become_zero_reward(tmp_path):
    fixture=ActorFixture(empty=True); assembler=fixture.assembler(tmp_path)
    try:
        for _ in range(3): assembler.submit_actor(fixture.sample(),source_id=fixture.actor_source)
        fixture.receipts(assembler); fixture.finish(assembler)
        first=fixture.records[0]; end=fixture.records[1].observation.cutoff_nanos
        fixture.evidence(assembler,'environment.reward',{'startNanos':first.observation.cutoff_nanos,'endNanos':end,'value':None},first.packet['id'])
        fixture.evidence(assembler,'environment.outcome',{'endNanos':end,'outcome':'continuing'},first.packet['id'])
        fixture.evidence(assembler,'environment.watermark',{'throughNanos':end,'throughSequence':fixture.label_sequence-1,'complete':True})
        with pytest.raises(EnvironmentError,match='unknown'): assembler.seal()
        assert not (tmp_path/'audit.json').exists()
    finally: assembler.close()


def test_label_lag_has_a_real_wall_time_bound_and_reports_a_fault(tmp_path):
    fixture=ActorFixture(empty=True); fault=threading.Event()
    assembler=fixture.assembler(tmp_path,limits=AssemblyLimits(maximum_label_lag_seconds=.1))
    assembler._on_fault=lambda _:fault.set()
    try:
        assembler.submit_actor(fixture.sample(),source_id=fixture.actor_source)
        assert fault.wait(2)
        with pytest.raises(EnvironmentError,match='lag budget'): assembler.seal()
        assert not (tmp_path/'audit.json').exists()
    finally: assembler.close()


def test_actor_rng_split_evidence_cannot_be_invented_even_when_logp_is_valid(tmp_path):
    fixture=ActorFixture(empty=True); assembler=fixture.assembler(tmp_path)
    try:
        record=fixture.sample(); record.collection['sampler']['stateAfter']=[0,0]
        assembler.submit_actor(record,source_id=fixture.actor_source)
        with pytest.raises(EnvironmentError,match='RNG evidence'): assembler.finish_collection()
    finally: assembler.close()


def test_wrong_wire_command_cannot_train_a_different_sampled_packet(tmp_path):
    fixture=ActorFixture(); assembler=fixture.assembler(tmp_path)
    try:
        for _ in range(5):
            record=fixture.sample()
            assembler.submit_actor(record,source_id=fixture.actor_source)
        fixture.receipts(assembler); fixture.finish(assembler); fixture.labels(assembler)
        rollout=assembler.seal()
        first=rollout.decisions[0]
        changed=json.loads(first.commands_json)
        assert changed
        changed[0]['offsetMs']=min(99,changed[0]['offsetMs']+1)
        if changed==json.loads(first.commands_json):changed[0]['offsetMs']=0
        tampered=replace(rollout,decisions=(replace(first,commands_json=json.dumps(changed).encode()),*rollout.decisions[1:]))
        passive=ExternalEnvironment(fixture.spec,run_id=fixture.run,clock_id=fixture.clock,emit=lambda *_:None,resolve_frame=lambda *_:None)
        trainer=ReinforcementTrainer(fixture.policy,passive,assembler.training,policy_id=fixture.policy_id)
        trainer.admit_external_rollout(tampered)
        with pytest.raises(ValueError,match='wire commands'):trainer.update(tampered)
        assert trainer.optimizer_updates==0
        trainer.discard_rollout(tampered); trainer.stop()
    finally: assembler.close()


def test_rng_and_packet_stream_continue_across_confirmed_resets(tmp_path):
    fixture=ActorFixture(empty=True); assembler=fixture.assembler(tmp_path)
    try:
        for _ in range(3): assembler.submit_actor(fixture.sample(),source_id=fixture.actor_source)
        fixture.receipts(assembler); fixture.labels(assembler)
        first_after=fixture.records[-1].collection['sampler']['stateAfter']
        fixture.reset_episode(assembler)
        for _ in range(3): assembler.submit_actor(fixture.sample(),source_id=fixture.actor_source)
        assert fixture.records[3].collection['sampler']['stateBefore']==first_after
        assert fixture.records[3].collection['sampler']['drawIndex']==3
        fixture.receipts(assembler); fixture.labels(assembler); fixture.finish(assembler)
        rollout=assembler.seal()
        assert len(rollout.decisions)==4
        assert [item.transition.episode_step for item in rollout.decisions]==[0,1,0,1]
        rollout.close()
    finally: assembler.close()


def test_ingress_reserves_capacity_before_any_concurrent_pixel_clone(tmp_path,monkeypatch):
    fixture=ActorFixture(empty=True); assembler=fixture.assembler(tmp_path)
    record=fixture.sample(); budget=record.byte_count(1024**2)+1
    assembler.limits=AssemblyLimits(maximum_queue_bytes=budget).validate()
    entered,release=threading.Event(),threading.Event(); errors=[]; copies=[]
    original=ActorRecord.owned
    def blocked_copy(item):
        copies.append(item)
        entered.set()
        assert release.wait(3)
        return original(item)
    monkeypatch.setattr(ActorRecord,'owned',blocked_copy)
    def producer():
        try:assembler.submit_actor(record,source_id=fixture.actor_source)
        except BaseException as error:errors.append(error)
    thread=threading.Thread(target=producer); thread.start()
    try:
        assert entered.wait(3)
        assert assembler._queue_bytes==budget-1 and assembler._ingress_items==1
        with pytest.raises(EnvironmentError,match='ingress byte budget'):
            assembler.submit_actor(record,source_id=fixture.actor_source)
        assert len(copies)==1
        release.set(); thread.join(3); assert not thread.is_alive() and not errors
        # The joined call is a barrier after processing the accepted actor.
        assembler.close_episode(episode_id=fixture.episode,stopped_nanos=fixture.environment._now,
            last_actor_sequence=0,controls_released=True,pending_packets=0,source_id=fixture.label_source)
        assert assembler._queue_bytes==0 and assembler._ingress_items==0
    finally:
        release.set(); thread.join(3); assembler.close()


def test_failed_clone_releases_ingress_reservation(tmp_path,monkeypatch):
    fixture=ActorFixture(empty=True); assembler=fixture.assembler(tmp_path)
    try:
        record=fixture.sample()
        def failed_copy(_):raise MemoryError('fixture copy failed')
        monkeypatch.setattr(ActorRecord,'owned',failed_copy)
        with pytest.raises(MemoryError,match='copy failed'):
            assembler.submit_actor(record,source_id=fixture.actor_source)
        assert assembler._queue_bytes==0 and assembler._ingress_items==0
    finally:assembler.close()


@pytest.mark.parametrize('completion',['cancelled','unchanged'])
def test_policy_gate_never_readmits_a_consumed_rollout(completion):
    from astra.learning.policy_activation import PolicyActivationGate
    policy,rollout=(str(uuid.uuid4()) for _ in range(2)); gate=PolicyActivationGate(policy)
    gate.begin_learning(rollout_id=rollout,policy_id=policy)
    if completion=='cancelled':gate.cancel_learning(rollout_id=rollout)
    else:gate.complete_learning(rollout_id=rollout,proposed_policy_id=None)
    with pytest.raises(EnvironmentError,match='admitted twice'):
        gate.begin_learning(rollout_id=rollout,policy_id=policy)
    gate.begin_learning(rollout_id=str(uuid.uuid4()),policy_id=policy)
