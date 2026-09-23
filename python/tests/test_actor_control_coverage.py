from copy import deepcopy

import pytest

from astra.inference import InferenceSession,InferenceError
from astra.frame_ring import FrameRingReader
from test_frame_ring import native_ring,ring_fixture_executable,_message
from test_inference import checkpoint,_prepare,_step,_acknowledge


@pytest.mark.parametrize('mode',['missing','backward','future','uncovered_history'])
def test_actor_rejects_untrusted_control_coverage_before_owned_frame_copy(native_ring,checkpoint,monkeypatch,mode):
    _,report=native_ring;run=report['reference']['runID'];actor=InferenceSession()
    try:
        actor.prepare({**_prepare(checkpoint,report,deterministic=False),'collection':True},run_id=run)
        import uuid
        ready=actor.reset({'confirmed':True,'episodeID':str(uuid.uuid4()),'contextIDs':[]},run_id=run)
        request=_step(report,ready);cutoff=request['cutoffNanos']
        request['controlState']['observedNanos']=cutoff-300_000_000
        if mode!='missing':request['controlCoverageNanos']=cutoff+({'future':1,'backward':-1}.get(mode,0))
        if mode=='uncovered_history':request['intervalCovered']=False
        def forbidden_copy(*_):pytest.fail('Invalid control coverage copied source pixels')
        monkeypatch.setattr(FrameRingReader,'copy_frame',forbidden_copy)
        with pytest.raises(InferenceError):actor.step(request,run_id=run)
        assert actor._sequence==actor._draw_index==0 and actor._state is None
    finally:actor.close()


def test_actor_echoes_valid_coverage_and_preserves_original_silent_state(native_ring,checkpoint):
    import uuid
    producer,report=native_ring;run=report['reference']['runID'];actor=InferenceSession()
    try:
        actor.prepare({**_prepare(checkpoint,report,deterministic=False),'collection':True},run_id=run)
        ready=actor.reset({'confirmed':True,'episodeID':str(uuid.uuid4()),'contextIDs':[]},run_id=run)
        request=_step(report,ready);cutoff=request['cutoffNanos']
        request['controlState']['observedNanos']=cutoff-300_000_000
        request['controlCoverageNanos']=cutoff;original=deepcopy(request['controlState'])
        result=actor.step(request,run_id=run)
        assert result['collectionRecord']['controlCoverageNanos']==cutoff
        assert request['controlState']==original and result['packet']['sequence']==0
        _acknowledge(producer,result['releasedFrames'][0]);second=_message(producer)
        # A fresh legacy snapshot continues through the same actor without the new field.
        following=actor.step(_step(second,result,cutoff=cutoff+100_000_000),run_id=run)
        assert 'controlCoverageNanos' not in following['collectionRecord']
        assert following['packet']['sequence']==1
        _acknowledge(producer,following['releasedFrames'][0]);assert _message(producer)['closed']
        assert producer.wait(timeout=10)==0
    finally:actor.close()
