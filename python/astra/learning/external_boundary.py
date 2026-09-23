"""Save a joined actor cursor without applying a learning update."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import stat
import tempfile
import uuid

from astra.actor_progress import validate_actor_progress
from astra.checkpoints import load_checkpoint,load_checkpoint_actor_progress,save_checkpoint,_sync
from astra.environments.interface import EnvironmentSpec,fields,same_id
from .external_job import SealedEnvironment
from .reinforcement import ReinforcementTrainer
from .rollout_artifacts import inspect_package,fingerprint,encoded


def _cursor_equal(one,two):
    return (same_id(one['rngStreamID'],two['rngStreamID']) and
            all(one[key]==two[key] for key in ('drawIndex','rngState','actorResetGeneration')))


@contextmanager
def _publication_claim(path,expected):
    """One boundary owner; incomplete writes are retryable only to the same ID.

    A PPO boundary claim has another schema and cannot be repurposed. The lock
    serializes concurrent retries; checkpoint publication itself is exclusive.
    """
    from astra.jobs import JobError
    lock_path=path.with_name(path.name+'.lock')
    descriptor=os.open(lock_path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_NONBLOCK,0o600)
    try:
        info=os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_nlink!=1:
            raise JobError('job.boundaryClaim','Invalid actor-boundary publication lock')
        try:fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise JobError('job.boundaryBusy','This actor boundary already has a publishing owner') from error
        if not path.exists() and not path.is_symlink():
            temporary_fd,temporary=tempfile.mkstemp(prefix='.boundary-claim-',dir=path.parent)
            try:
                with os.fdopen(temporary_fd,'wb') as target:
                    target.write(encoded(expected));target.flush();os.fsync(target.fileno())
                try:os.link(temporary,path)
                except FileExistsError:pass # A concurrent PPO claim must still be authenticated below.
                _sync(path.parent)
            finally:Path(temporary).unlink(missing_ok=True)
        claim_fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(claim_fd,'rb') as source:
            info=os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_nlink!=1 or not 0<info.st_size<=1024*1024:
                raise JobError('job.boundaryClaim','Invalid actor-boundary publication claim')
            try:actual=json.loads(source.read(1024*1024+1))
            except (ValueError,UnicodeDecodeError) as error:raise JobError('job.boundaryClaim','The existing publication claim is incomplete or invalid') from error
        if actual!=expected:raise JobError('job.boundaryConsumed','This boundary is reserved for a different source, destination or operation')
        yield
    finally:os.close(descriptor)


def _result(destination,manifest,progress,collection_id,count,cancelled):
    return {'checkpointPath':str(destination),'manifest':manifest,'parameterCount':count,'checkpointPublished':True,
        'actorProgress':progress,'boundaryCollectionID':collection_id,'resumable':True,'requiresEnvironmentReset':True,
        'sourceKind':'external_rollout','provenance':'external_rollout','boundaryOnly':True,'cancelled':cancelled}


def preserve_external_boundary(manager,job,loaded):
    from astra.jobs import JobError,_reinforcement_config
    value=job.configuration;audit_path=Path(value['auditPath']);destination=Path(value['destination'])
    manager._progress(job,{'phase':'validating_boundary','sourceKind':'external_rollout','boundaryOnly':True})
    audit=inspect_package(audit_path);binding=audit['binding']
    if audit['status'] not in ('audited','sealed','awaiting_manual_review') or not audit['controlClosureKnown'] or audit['actorProgress'] is None:
        raise JobError('job.boundaryUnavailable','Only a validated complete actor audit can preserve a resume cursor')
    current=validate_actor_progress(audit['actorProgress'],binding['runID'])
    if not same_id(binding['policyID'],loaded.manifest['id']) or binding['policySignature']!=loaded.manifest['policySignature']:
        raise JobError('job.boundaryMismatch','The joined actor used another checkpoint or policy')
    environment=EnvironmentSpec.from_dict(binding['environment']);training=_reinforcement_config(binding['training'])
    if loaded.policy.config.to_dict()!=binding['model'] or loaded.policy.actions.vocabulary!=environment.action_vocabulary:
        raise JobError('job.boundaryMismatch','The actor audit changed model or action configuration')
    contexts=tuple(binding['contextIDs'])
    if len(contexts)!=len(loaded.policy.config.context_sizes) or any(type(v) is not int or not 0<=v<size for v,size in zip(contexts,loaded.policy.config.context_sizes)):
        raise JobError('job.boundaryMismatch','The actor audit has incompatible contexts')
    restored=None
    if value.get('resume',False):
        state=loaded.training_state
        if not isinstance(state,dict):raise JobError('job.resumeUnavailable','No external optimizer state is available to preserve')
        fields(state,('kind','schemaVersion','learner','actorProgress','consumedRolloutIDs','requiresEnvironmentReset'))
        if state['kind']!='reinforcement_external' or state['schemaVersion']!=1 or state['requiresEnvironmentReset'] is not True:
            raise JobError('job.resumeUnavailable','Resume needs an external reinforcement checkpoint')
        saved=load_checkpoint_actor_progress(Path(value['checkpointPath']),loaded.manifest)
        if saved!=validate_actor_progress(state['actorProgress']):
            raise JobError('job.actorProgressMismatch','Saved optimizer and verified actor metadata have different cursors')
        prior=binding['previousActorProgress']
        if prior is None or not _cursor_equal(validate_actor_progress(prior,binding['runID']),saved):
            raise JobError('job.actorProgressMismatch','The stopped actor did not continue the checkpoint random stream')
        if not same_id(current['rngStreamID'],saved['rngStreamID']) or current['drawIndex']<=saved['drawIndex'] or current['actorResetGeneration']<=saved['actorResetGeneration']:
            raise JobError('job.actorProgressMismatch','The boundary has no new real draws after a confirmed reset')
        learner=state['learner']
        required={'kind','schemaVersion','config','environmentSignature','modelSignature','contextIDs','iteration','decisions',
            'optimizerUpdates','environmentResets','rng','optimizer','policyID','requiresEnvironmentReset'}
        fields(learner,required)
        if learner['kind']!='reinforcement' or learner['schemaVersion']!=2 or learner['requiresEnvironmentReset'] is not True:
            raise JobError('job.resumeUnavailable','Unsupported external learner state')
        if (learner['config']!=asdict(training) or learner['environmentSignature']!=environment.signature or
            learner['modelSignature']!=loaded.policy.config.signature or tuple(learner['contextIDs'])!=contexts or
            not same_id(learner['policyID'],loaded.manifest['id'])):
            raise JobError('job.boundaryMismatch','Preserve the exact saved task, training configuration and contexts')
        if any(type(learner[key]) is not int or learner[key]<0 for key in ('iteration','decisions','optimizerUpdates','environmentResets')):
            raise JobError('job.resumeUnavailable','Invalid saved learner counters')
        history=state['consumedRolloutIDs']
        if not isinstance(history,(list,tuple)) or len(history)>65536 or len(set(history))!=len(history):
            raise JobError('job.resumeUnavailable','Invalid saved rollout-consumption history')
        for identifier in history:uuid.UUID(identifier)
        restored=state
    if job.cancel.is_set():raise InterruptedError('Cursor preservation cancelled before publication')
    claim={'schemaVersion':1,'operation':'checkpoint.externalBoundary','checkpointID':loaded.manifest['id'],
        'sourceManifest':fingerprint(Path(value['checkpointPath'])/'manifest.json'),
        'auditID':audit['id'],'auditManifest':fingerprint(audit_path/'manifest.json'),
        'destination':str(destination),'resume':value.get('resume',False)}
    claim_path=audit_path.parent/('.boundary-'+audit['id']+'.json')
    if destination.exists() and not claim_path.exists():
        raise JobError('job.destinationExists','The destination already exists without this boundary publication claim')
    with _publication_claim(claim_path,claim):
        if destination.exists():
            existing=load_checkpoint(destination,include_training=True)
            metrics=existing.manifest['metrics']
            if (existing.manifest['parentID']!=loaded.manifest['id'] or existing.manifest['policySignature']!=loaded.manifest['policySignature'] or
                metrics.get('operation')!='checkpoint.externalBoundary' or metrics.get('boundaryCollectionID')!=audit['id'] or
                existing.training_state is None or existing.training_state.get('actorProgress')!=current or
                existing.manifest['artifacts']['policy.safetensors']!=loaded.manifest['artifacts']['policy.safetensors']):
                raise JobError('job.destinationExists','The destination does not contain this completed boundary publication')
            return _result(destination,existing.manifest,current,audit['id'],existing.policy.config.parameter_count,job.cancel.is_set())
        trainer=None
        try:
            if restored is None:
                # This begins external PPO just as a fresh RL run would. BC's
                # optimizer is a different algorithm and is never relabelled.
                # Unfreezing changes trainability metadata, not numeric weights.
                loaded.policy.unfreeze()
                trainer=ReinforcementTrainer(loaded.policy,SealedEnvironment(environment,binding['runID']),training,
                    policy_id=loaded.manifest['id'],context_ids=contexts)
                learner=trainer.state;history=[]
            else:
                # No optimizer reconstruction, forward, reset, collection or
                # update: preserve tensor leaves and all learner counters exactly.
                learner=dict(restored['learner']);history=restored['consumedRolloutIDs']
            learner['policyID']=destination.name
            state={'kind':'reinforcement_external','schemaVersion':1,'learner':learner,'actorProgress':current,
                'consumedRolloutIDs':history,'requiresEnvironmentReset':True}
            metrics={'operation':'checkpoint.externalBoundary','boundaryOnly':True,'sourceKind':'external_rollout',
                'provenance':'external_rollout','actorProgress':current,'boundaryCollectionID':audit['id'],
                'iteration':learner['iteration'],'optimizer_updates':learner['optimizerUpdates'],'decisions':learner['decisions'],
                'requiresEnvironmentReset':True}
            manager._progress(job,{'phase':'checkpointing','sourceKind':'external_rollout','boundaryOnly':True,
                'iteration':learner['iteration'],'optimizer_updates':learner['optimizerUpdates'],'decisions':learner['decisions']})
            if job.cancel.is_set():raise InterruptedError('Cursor preservation cancelled before checkpoint commit')
            manifest=save_checkpoint(destination,loaded.policy,kind='reinforcement',step=learner['optimizerUpdates'],
                training_state=state,parent_id=loaded.manifest['id'],metrics=metrics,training_config=asdict(training))
            return _result(destination,manifest,current,audit['id'],loaded.policy.config.parameter_count,job.cancel.is_set())
        finally:
            if trainer is not None:trainer.stop('Stopped-actor cursor preserved without learning or native controls')
