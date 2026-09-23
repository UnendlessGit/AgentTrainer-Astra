#!/usr/bin/env python3
"""Real desktop host, generated pixels and virtual InputExecutor only; no TCC."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import plistlib
import shlex
import signal
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,help='Optional assembled app; defaults to locked source workers')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--feedback',action='store_true',help='Exercise original-frame review through the native UI model before PPO')
    args=parser.parse_args()
    output=(args.output or ROOT/'.local/verification'/('desktop-host-'+str(uuid.uuid4()))).resolve()
    if output.exists():parser.error('Choose a new output directory to preserve evidence')
    output.mkdir(parents=True)
    print('Evidence:',output,flush=True)
    if args.bundle:
        bundle=args.bundle.resolve()
    else:
        bundle=output/'SourceRuntime.app';contents=bundle/'Contents'
        executable=contents/'Helpers/AstraCompute.app/Contents/MacOS/AstraCompute'
        executable.parent.mkdir(parents=True)
        (contents/'Resources').mkdir()
        with (contents/'Info.plist').open('wb') as file:
            plistlib.dump({'CFBundleIdentifier':'test.astra.desktop-host.'+str(uuid.uuid4()),
                'CFBundleName':'Generated desktop qualification','CFBundlePackageType':'APPL'},file)
        executable.write_text('#!/bin/sh\nexport PYTHONPATH='+shlex.quote(str(ROOT/'python'))+'\nexec '+
            shlex.quote(str(ROOT/'.venv/bin/python'))+' -u -m astra.worker "$@"\n')
        executable.chmod(0o755)
    if not (bundle/'Contents/Helpers/AstraCompute.app/Contents/MacOS/AstraCompute').is_file():
        parser.error('The requested compute helper is unavailable')
    import mlx.core as mx
    from mlx.utils import tree_flatten
    import numpy as np
    from astra.checkpoints import save_checkpoint,load_checkpoint
    from astra.model.actions import ActionVocabulary
    from astra.model.config import ModelConfig
    from astra.model.policy import AgentPolicy
    mx.random.seed(3107)
    policy=AgentPolicy(ModelConfig.test_small(),ActionVocabulary((0,),(),False,False,False))
    destination=output/'Library/Models'/str(uuid.uuid4())
    manifest=save_checkpoint(destination,policy,kind='initial',step=0)
    (output/'initial.json').write_text(json.dumps({'id':manifest['id'],'policySignature':manifest['policySignature'],
        'parameterCount':policy.config.parameter_count}))
    environment={**os.environ,'ASTRA_HOST_QUALIFICATION_ROOT':str(output),'ASTRA_HOST_QUALIFICATION_BUNDLE':str(bundle)}
    if args.feedback:environment['ASTRA_HOST_QUALIFICATION_FEEDBACK']='1'
    suite='DesktopFeedbackQualificationTests' if args.feedback else 'DesktopHostQualificationTests'
    with (output/'swift-test.log').open('wb') as log:
        process=subprocess.Popen([str(ROOT/'script/swift.sh'),'test','--filter',suite],
            cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:returncode=process.wait(timeout=240)
        except BaseException:
            # This test owns no physical controls. Retire the whole test/process
            # family on a stalled harness, retaining its private audit directory.
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
    report_path=output/'desktop-host-report.json'
    if report_path.exists():print(report_path.read_text(),flush=True)
    if returncode:
        print((output/'swift-test.log').read_text()[-16000:],flush=True)
        raise SystemExit(returncode)
    report=json.loads(report_path.read_text())
    if not report['completed'] or report['privacyPermissionsUsed'] or not report['controlCleanupConfirmed']:
        raise RuntimeError('Real desktop host qualification did not complete')
    learned=load_checkpoint(output/'Library/Models'/report['learnedCheckpointID'],include_training=True)
    if args.feedback:
        assert learned.training_state['learner']['optimizerUpdates']>0
        assert learned.manifest['artifacts']['policy.safetensors']!=manifest['artifacts']['policy.safetensors']
        assert report['reviewPresentations']>0 and report['reviewedIntervals']>=6
        reopened=load_checkpoint(output/'Library/Models'/report['reopenedCheckpointID'],include_training=True)
        assert reopened.training_state['learner']['optimizerUpdates']>learned.training_state['learner']['optimizerUpdates']
        assert reopened.manifest['artifacts']['policy.safetensors']!=learned.manifest['artifacts']['policy.safetensors']
        continued=load_checkpoint(output/'Library/Models'/report['continuedCheckpointID'],include_training=True)
        assert continued.training_state['learner']['optimizerUpdates']>reopened.training_state['learner']['optimizerUpdates']
        assert continued.manifest['artifacts']['policy.safetensors']!=reopened.manifest['artifacts']['policy.safetensors']
        from astra.learning.rollout_artifacts import inspect_package
        batches=[path.parent for path in (output/'Library/ReviewedExperience').glob('*/manifest.json')
            if (document:=json.loads(path.read_text()))['schemaVersion']==3 and document['behaviorBatchID']==report['continuationBatchID']]
        assert len(batches)==1
        batch=inspect_package(batches[0])
        assert len(batch['fragments'])==2 and batch['decisions']>=6
        assert 0<report['prefixDecisions']<6
        source_identities=[]
        for fragment in batch['fragments']:
            derived=inspect_package(fragment['path'],expected_manifest_sha256=fragment['manifestSHA256'])
            source=json.loads((Path(derived['reviewSource']['path'])/'source.json').read_text())
            source_identities.append({key:source[key] for key in ('runID','clockID')})
            assert all(source[key]==derived['binding'][key] for key in ('runID','clockID'))
        assert source_identities==report['continuedReviewIdentities']
        report.update(bundle=str(bundle),model=policy.config.to_dict(),verificationModel='numerical_test_small',
            reviewedWeightsChanged=True,reopenedWeightsChanged=True,continuedWeightsChanged=True,
            combinedDecisions=batch['decisions'],originalRunAndClockPreserved=True,
            reviewMethod='original-frame UI-model presentation acknowledgement and explicit per-interval judgments')
        report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        return
    stopped=load_checkpoint(output/'Library/Models'/report['stoppedCheckpointID'],include_training=True)
    assert learned.training_state['learner']['optimizerUpdates']>0
    assert learned.manifest['artifacts']['policy.safetensors']!=manifest['artifacts']['policy.safetensors']
    assert learned.manifest['artifacts']['policy.safetensors']==stopped.manifest['artifacts']['policy.safetensors']
    before=dict(tree_flatten(learned.training_state['learner']['optimizer']))
    after=dict(tree_flatten(stopped.training_state['learner']['optimizer']))
    assert before.keys()==after.keys()
    for name,value in before.items():np.testing.assert_array_equal(np.asarray(value),np.asarray(after[name]))
    first,last=learned.training_state['actorProgress'],stopped.training_state['actorProgress']
    assert first['rngStreamID']==last['rngStreamID'] and first['drawIndex']<last['drawIndex']
    assert first['runID']!=last['runID'] and first['actorResetGeneration']<last['actorResetGeneration']
    sessions=[json.loads(path.read_text()) for path in (output/'Library/DesktopRuns').glob('*/results.json')]
    packages=[json.loads(path.read_text()) for path in (output/'Library/DesktopRuns').glob('*/Collections/*/manifest.json')]
    assert len(sessions)==2 and all(row['controlCleanupConfirmed'] and row['issue'] is None for row in sessions)
    complete=[row for row in sessions if not row['stopped']]
    assert len(complete)==1 and complete[0]['episodes']>=2 and complete[0]['completedUpdates']==1
    sealed=[row for row in packages if row['status']=='sealed']
    audited=[row for row in packages if row['status']=='audited']
    assert len(sealed)==len(audited)==1 and sealed[0]['decisions']>=6 and audited[0]['decisions']==0
    report.update(optimizerPreserved=True,stoppedWeightsUnchanged=True,resumedDrawIndex=last['drawIndex'],
        previousDrawIndex=first['drawIndex'],bundle=str(bundle),model=policy.config.to_dict(),
        completedEpisodes=complete[0]['episodes'],admittedDecisions=sealed[0]['decisions'],
        learningIterations=complete[0]['completedUpdates'],verificationModel='numerical_test_small',
        resetMode='explicit_fixture_confirmation_with_fresh_generated_source_proof')
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
