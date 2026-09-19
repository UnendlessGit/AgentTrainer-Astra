"""Actual native disarm receipts remain truthful across the learner boundary."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import uuid

import pytest

from astra.environments.evidence import DecisionEvidence
from astra.environments.interface import DecisionContext, EnvironmentError
from astra.learning.policy_activation import PolicyActivationGate


@pytest.fixture(scope='module')
def native_receipts():
    root=Path(__file__).resolve().parents[2]
    subprocess.run(['swift','build','--product','AstraReceiptFixture'],cwd=root,check=True,capture_output=True,timeout=120)
    result=subprocess.run([str(root/'.build/debug/AstraReceiptFixture')],cwd=root,check=True,capture_output=True,timeout=10)
    return {case['name']:case for case in json.loads(result.stdout)}


def window(case,*,rejected=False):
    packet=case['rejectedPacket'] if rejected else case['packet']
    context=DecisionContext(packet['runID'],str(uuid.uuid4()),str(uuid.uuid4()),packet['observationID'],packet['id'],
        packet['sequence'],packet['executeAtNanos']-100_000_000,packet['geometryRevision'])
    return DecisionEvidence(context,packet['sequence'],tuple(packet['commands']))


def accept(evidence,receipt,*,cause=None):
    payload={'clockID':str(uuid.uuid4()),'episodeID':evidence.context.episode_id,'receipt':receipt}
    if cause is not None:payload['cancellationCause']=cause
    evidence.accept_receipt(payload,lead_ms=100,retain_failure=True)


@pytest.mark.parametrize('name',['futureCancellation','postedCancellation'])
def test_native_inactive_boundary_cancellation_is_retained_without_rewriting_validity(native_receipts,name):
    case=native_receipts[name];evidence=window(case)
    admitted,final,rejected=case['receipts']
    assert admitted['status']=='admitted' and admitted['resultingState']['valid']
    assert final['status']=='cancelled' and final['resultingState']['valid'] is False
    assert case['cleanupSettled'] and rejected['resultingState']['valid'] is False
    before=deepcopy(final)
    accept(evidence,admitted);accept(evidence,final,cause='episodeBoundary')
    assert final==before and evidence.receipt['resultingState']==final['resultingState']
    assert evidence.execution_complete(case['boundaryNanos'])
    if name=='postedCancellation':
        posted=[item for item in evidence.receipt['commandResults'] if item['status']=='posted']
        assert posted and posted[0]['postedNanos']>case['boundaryNanos']
        assert any(effect['cleanup'] and effect['operation']=='keyUp' for effect in case['effects'])
    suffix=window(case,rejected=True);accept(suffix,rejected)
    assert suffix.receipt['status']=='rejected' and suffix.receipt['resultingState']['valid'] is False
    with pytest.raises(EnvironmentError,match='cannot enter'):suffix.execution_complete(case['boundaryNanos'])


def test_native_inactive_cancellation_still_needs_explicit_boundary_and_future_commands(native_receipts):
    case=native_receipts['postedCancellation']
    for cause in (None,'requested','physicalTakeover'):
        evidence=window(case);accept(evidence,case['receipts'][0]);accept(evidence,case['receipts'][1],cause=cause)
        with pytest.raises(EnvironmentError,match='Administrative'):evidence.execution_complete(case['boundaryNanos'])
    overdue=native_receipts['overdueCancellation'];evidence=window(overdue)
    accept(evidence,overdue['receipts'][0]);accept(evidence,overdue['receipts'][1],cause='episodeBoundary')
    assert overdue['cleanupSettled'] and evidence.receipt['resultingState']['valid'] is False
    with pytest.raises(EnvironmentError,match='future episode boundary'):evidence.execution_complete(overdue['boundaryNanos'])


def test_native_posting_and_cleanup_failures_cannot_become_learning_or_reset_proof(native_receipts):
    failed=native_receipts['postingFailure'];evidence=window(failed)
    accept(evidence,failed['receipts'][0]);accept(evidence,failed['receipts'][1],cause='episodeBoundary')
    assert any(item['status']=='failed' for item in evidence.receipt['commandResults'])
    assert evidence.receipt['resultingState']['valid'] is False
    with pytest.raises(EnvironmentError,match='Failed'):evidence.execution_complete(failed['boundaryNanos'])
    cleanup=native_receipts['cleanupFailure'];evidence=window(cleanup)
    accept(evidence,cleanup['receipts'][0]);accept(evidence,cleanup['receipts'][1],cause='episodeBoundary')
    # Receipt retention is not an independent cleanup attestation.
    assert evidence.receipt['resultingState']['valid'] is False and not cleanup['cleanupSettled']
    assert cleanup['finalState']['keys']==[0]
    gate=PolicyActivationGate(str(uuid.uuid4()))
    with pytest.raises(EnvironmentError,match='cleared-control'):
        gate.confirm_reset(episode_id=str(uuid.uuid4()),reset_id=str(uuid.uuid4()),policy_id=gate.policy_id,
            controls_released=cleanup['cleanupSettled'],pending_packets=0)


def test_active_admission_and_executed_prefix_cannot_use_an_inactive_snapshot(native_receipts):
    case=native_receipts['completedPacket'];admitted,executed,rejected=case['receipts']
    assert executed['status']=='executed' and executed['resultingState']['valid'] is True
    assert case['finalState']['valid'] is False # A later disarm never edits past receipts.
    forged=deepcopy(admitted);forged['resultingState']['valid']=False
    with pytest.raises(EnvironmentError,match='active admission'):accept(window(case),forged)
    evidence=window(case);accept(evidence,admitted)
    forged=deepcopy(executed);forged['resultingState']['valid']=False
    accept(evidence,forged)
    assert evidence.receipt['resultingState']['valid'] is False # Auditable, not trainable.
    with pytest.raises(EnvironmentError,match='Inactive executed'):evidence.execution_complete(case['boundaryNanos'])
