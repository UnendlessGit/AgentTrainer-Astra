"""Control authority is renewed explicitly without changing model inputs/time."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import uuid

import numpy as np
import pytest

from astra.environments.interface import EnvironmentSpec, EnvironmentError
from astra.environments.observation_transport import SnapshotDecoder, ResolvedFrame
from astra.model.actions import ActionVocabulary
from astra.model.config import ModelConfig
from astra.learning.reinforcement import ObservationRecord
from test_rollout_assembler import ActorFixture


def snapshot_fixture():
    cutoff=1_000_000_000;episode=str(uuid.uuid4());pixels=np.zeros((32,32,4),dtype=np.uint8)
    surface={'id':'test','pixelWidth':32,'pixelHeight':32,'geometryRevision':0,
        'globalBounds':{'x':0,'y':0,'width':32,'height':32},'contentBounds':{'x':0,'y':0,'width':32,'height':32}}
    metadata={'id':str(uuid.uuid4()),'surface':surface,'eventNanos':cutoff,'observedNanos':cutoff,
        'byteCount':pixels.nbytes,'pixelFormat':'bgra8-srgb','codec':'raw'}
    value={'id':str(uuid.uuid4()),'episodeID':episode,'cutoffNanos':cutoff,'geometryRevision':0,
        'controlState':{'keys':[],'buttons':[],'modifiers':0,'pointer':{'x':0,'y':0},
            'observedNanos':cutoff-300_000_000,'revision':9,'valid':True},'events':[],
        'frames':[{'metadata':metadata,'reference':{},'coverageNanos':cutoff,'coverageKind':'frame'}]}
    spec=EnvironmentSpec('control-coverage-fixture',ActionVocabulary(),maximum_observation_bytes=pixels.nbytes,maximum_surfaces=1)
    copies=[];acks=[]
    def resolve(*_):copies.append(True);return ResolvedFrame(pixels,{'owned':True})
    decoder=SnapshotDecoder(spec,resolve,lambda *value:acks.append(value))
    return value,decoder,copies,acks


@pytest.mark.parametrize('mode',['missing','future','backward','boolean','invalid','history_gap'])
def test_untrusted_control_authority_is_rejected_before_any_pixel_lease_copy(mode):
    value,decoder,copies,acks=snapshot_fixture();cutoff=value['cutoffNanos']
    if mode!='missing':value['controlCoverageNanos']={'future':cutoff+1,'backward':cutoff-1,'boolean':True}.get(mode,cutoff)
    if mode=='invalid':value['controlState']['valid']=False
    if mode=='history_gap':value['events']=[{'sequence':0,'eventNanos':cutoff,'observedNanos':cutoff,'origin':'boundary','kind':'gap'}]
    with pytest.raises(EnvironmentError):decoder.decode(value,episode=value['episodeID'])
    assert not copies and not acks


def test_control_coverage_preserves_old_state_timestamp_and_model_features():
    value,decoder,copies,acks=snapshot_fixture();state=deepcopy(value['controlState'])
    value['controlCoverageNanos']=value['cutoffNanos']
    observed,_=decoder.decode(value,episode=value['episodeID'])
    assert observed.control_state==state and observed.control_coverage_nanos==value['cutoffNanos']
    assert len(copies)==len(acks)==1
    record=ObservationRecord.capture(observed,elapsed_seconds=.1,reset=True)
    assert record.control_coverage_nanos==value['cutoffNanos']
    assert json.loads(record.controls_json)==state
    # The proof is source authority, never an added/retimestamped model feature.
    config=ModelConfig.test_small()
    covered=record.prepare(config,()).as_tensors()
    unchanged=replace(record,control_coverage_nanos=None).prepare(config,()).as_tensors()
    def compare(left,right):
        if isinstance(left,dict):
            assert left.keys()==right.keys()
            for key in left:compare(left[key],right[key])
        elif isinstance(left,(tuple,list)):
            for one,two in zip(left,right,strict=True):compare(one,two)
        else:np.testing.assert_array_equal(np.asarray(left),np.asarray(right))
    compare(covered,unchanged)


def test_fresh_legacy_snapshot_needs_no_new_field():
    value,decoder,copies,_=snapshot_fixture()
    value['controlState']['observedNanos']=value['cutoffNanos']-10_000_000
    observed,_=decoder.decode(value,episode=value['episodeID'])
    assert observed.control_coverage_nanos is None and len(copies)==1


def test_assembler_and_immutable_reload_keep_the_same_control_coverage(tmp_path):
    from astra.learning.rollout_artifacts import publish_package,inspect_package,load_rollout
    fixture=ActorFixture(empty=True)
    working=tmp_path/'working';working.mkdir();assembler=fixture.assembler(working)
    try:
        for _ in range(3):
            record=fixture.sample();cutoff=record.observation.cutoff_nanos
            old={**record.observation.control_state,'observedNanos':cutoff-300_000_000}
            record=replace(record,observation=replace(record.observation,control_state=old,control_coverage_nanos=cutoff))
            record.collection['controlCoverageNanos']=cutoff
            assembler.submit_actor(record,source_id=fixture.actor_source)
        fixture.receipts(assembler);fixture.labels(assembler);fixture.finish(assembler)
        rollout=assembler.seal()
        assert all(row.observation.control_coverage_nanos==row.transition.decision_nanos for row in rollout.decisions)
        (working/'journal.ndjson').write_text('{"fixture":"control coverage"}\n')
        binding={'runID':fixture.run,'clockID':fixture.clock,'policyID':fixture.policy_id,'policySignature':fixture.signature,
            'actorSourceID':fixture.actor_source,'environmentSourceID':fixture.label_source,'environment':fixture.spec.to_dict(),
            'model':fixture.model.to_dict(),'training':asdict(assembler.training),'contextIDs':[],
            'purpose':'learning','previousActorProgress':None}
        destination=tmp_path/str(uuid.uuid4())
        publish_package(working,destination,binding=binding,status='sealed',actor_progress=json.loads(rollout.actor_progress_json),
            control_closure_known=True,rollout=rollout)
        loaded=load_rollout(destination,inspect_package(destination),assembler.training)
        try:
            for row in loaded.decisions:
                assert row.observation.control_coverage_nanos==row.transition.decision_nanos
                assert json.loads(row.observation.controls_json)['observedNanos']==row.transition.decision_nanos-300_000_000
        finally:loaded.close()
    finally:assembler.close()


@pytest.mark.parametrize('proof_kind',['mismatch','noninteger'])
def test_actor_record_cannot_relabel_snapshot_control_authority_before_copy(proof_kind):
    from astra.learning.rollout_assembler import ActorRecord
    value,decoder,copies,acks=snapshot_fixture();cutoff=value['cutoffNanos']
    value['controlCoverageNanos']=cutoff
    response={'packet':{},'collectionRecord':{'episodeID':value['episodeID'],'cutoffNanos':cutoff,
        'controlCoverageNanos':cutoff+1 if proof_kind=='mismatch' else float(cutoff)}}
    with pytest.raises(EnvironmentError):
        ActorRecord.from_wire(response,value,spec=decoder.spec,resolve_frame=decoder._resolve_frame,on_consumed=decoder._on_consumed)
    assert not copies and not acks
