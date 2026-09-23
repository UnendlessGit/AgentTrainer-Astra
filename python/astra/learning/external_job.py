"""One admitted PPO update followed by an explicit actor-boundary publication."""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import time

import mlx.core as mx

from astra.checkpoints import save_checkpoint, _sync
from mlx.utils import tree_flatten, tree_unflatten
from astra.environments.interface import EnvironmentSpec, EnvironmentError, same_id, fields, integer
from .reinforcement import ReinforcementTrainer
from .rollout_artifacts import inspect_package, load_rollout, encoded


class SealedEnvironment:
    """An immutable source binding which owns no live environment or controls."""
    def __init__(self,spec,run_id):self.spec,self.run_id=spec,run_id
    @property
    def signature(self):return self.spec.signature
    @property
    def outcome(self):return 'terminated'
    def reset(self,**_):raise EnvironmentError('External learner cannot reset a physical environment')
    def step(self,*_,**__):raise EnvironmentError('External learner cannot sample or post environment actions')
    def seal_episode(self,**_):raise EnvironmentError('Only the collector seals external episodes')
    def abort(self,reason):pass  # No local controls; does not assert native cleanup.


def _same_progress(left,right):
    return all(left[key]==right[key] for key in ('rngStreamID','drawIndex','rngState','actorResetGeneration'))


def _claim(parent,name,value):
    path=parent/name
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as target:target.write(encoded(value));target.flush();os.fsync(target.fileno())
    _sync(parent)


def run_external_job(manager,job,loaded):
    from astra.jobs import JobError,_reinforcement_config
    value=job.configuration;path=Path(value['rolloutPath']);manifest=inspect_package(path);binding=manifest['binding']
    if manifest['status']!='sealed' or not same_id(binding['policyID'],loaded.manifest['id']) or binding['policySignature']!=loaded.manifest['policySignature']:
        raise JobError('job.rolloutMismatch','Sealed behavior policy does not match the checkpoint')
    claim_parents=[path.parent];boundary_binding=binding
    if manifest['schemaVersion']==2:
        from .review_pipeline import validate_derived
        claim_parents=[validate_derived(path,manifest)]
        if binding['continuationSource'] is not None:
            raise JobError('job.fragmentNeedsBatch','Continued experience requires a complete reviewed fragment batch')
    elif manifest['schemaVersion']==3:
        from .review_batches import batch_admission
        claim_parents,boundary_binding=batch_admission(manifest)
    training=_reinforcement_config(binding['training']);spec=EnvironmentSpec.from_dict(binding['environment'])
    integer(manifest['decisions'],training.maximum_rollout_decisions,training.rollout_decisions)
    if loaded.policy.config.to_dict()!=binding['model'] or loaded.policy.actions.vocabulary!=spec.action_vocabulary:
        raise JobError('job.rolloutMismatch','External model or action vocabulary differs from its checkpoint')
    restored=None;history=[]
    if value.get('resume'):
        state=loaded.training_state
        if not isinstance(state,dict):raise JobError('job.resumeUnavailable','No external reinforcement state is available')
        fields(state,('kind','schemaVersion','learner','actorProgress','consumedRolloutIDs','requiresEnvironmentReset'))
        if state['kind']!='reinforcement_external' or state['schemaVersion']!=1 or state['requiresEnvironmentReset'] is not True:
            raise JobError('job.resumeUnavailable','This checkpoint is not resumable external reinforcement')
        prior=binding['previousActorProgress']
        if prior is None or not _same_progress(prior,state['actorProgress']):
            raise JobError('job.actorProgressMismatch','Resumed collection must continue the saved actor random stream at a fresh reset')
        restored=state['learner'];history=list(state['consumedRolloutIDs'])
    else:
        loaded.policy.unfreeze();mx.random.seed(training.seed)
    if manifest['rolloutID'] in history or len(history)>=65536:
        raise JobError('job.rolloutConsumed','This rollout was already consumed or exceeded resume history capacity')
    # Claims are adjacent to the immutable package, never mutations of it. A
    # failed/crashed consumer needs fresh experience, not silent readmission.
    try:
        for claim_parent in claim_parents:_claim(claim_parent,'.consumed-'+manifest['rolloutID']+'.json',{'jobID':job.identifier,'runID':job.request.run_id,'rolloutID':manifest['rolloutID']})
    except FileExistsError as error:raise JobError('job.rolloutConsumed','This behavior batch was already consumed; changing its reward revision cannot reuse it') from error
    history.append(manifest['rolloutID'])
    trainer=ReinforcementTrainer(loaded.policy,SealedEnvironment(spec,binding['runID']),training,
        policy_id=loaded.manifest['id'],context_ids=tuple(binding['contextIDs']),restored_state=restored)
    before_weights=tree_flatten(trainer.policy.parameters());before_state=trainer.state
    rollout=None;interrupted=False;completed=None;started=time.monotonic()
    def report(phase,**extra):
        manager._progress(job,{'phase':phase,'sourceKind':'external_rollout','provenance':'external_rollout',
            'iteration':trainer.iteration,'decisions':trainer.decisions,'optimizer_updates':trainer.optimizer_updates,
            'actor_policy_id':binding['policyID'],'actorRunID':boundary_binding['runID'],'rolloutID':manifest['rolloutID'],
            'elapsed_seconds':time.monotonic()-started,**extra})
    try:
        rollout=load_rollout(path,manifest,training);trainer.admit_external_rollout(rollout)
        report('updating',rollout_decisions=len(rollout.decisions))
        try:
            completed=asdict(trainer.update(rollout,cancelled=job.cancel.is_set,
                on_update=lambda count:report('updating',optimizer_updates=count),
                on_validation=lambda data:report('validating_update',**data)))
        except InterruptedError:
            trainer.discard_rollout(rollout);interrupted=True
        # No immutable artifact is changed; retire only the learner's open
        # source handle before waiting for the independently running actor.
        rollout.close();rollout=None
        report('waiting_for_actor_boundary',cancelled=interrupted or job.cancel.is_set(),
            rngStreamID=manifest['actorProgress']['rngStreamID'],minimumDrawIndex=manifest['actorProgress']['drawIndex'])
        deadline=time.monotonic()+value.get('boundaryTimeoutSeconds',60)
        while not job.external_boundary_event.wait(.05):
            if manager._closing or time.monotonic()>=deadline:
                return {'checkpointPublished':False,'cancelled':True,'resumable':False,'requiresEnvironmentReset':True,
                    'reason':'No verified joined actor boundary was available for checkpoint publication','rolloutID':manifest['rolloutID']}
        boundary=inspect_package(Path(job.external_boundary_path));other=boundary['binding']
        if boundary['schemaVersion']==3:
            from .review_batches import batch_admission
            _,other=batch_admission(boundary)
        if boundary['status'] not in ('sealed','audited','awaiting_manual_review') or not boundary['controlClosureKnown'] or boundary['actorProgress'] is None:
            raise JobError('job.boundaryUnavailable','Boundary package has no completely validated actor/control stream')
        if any(other[key]!=boundary_binding[key] for key in ('runID','clockID','policyID','policySignature','actorSourceID','environmentSourceID','environment','model','contextIDs')):
            raise JobError('job.boundaryMismatch','Boundary package belongs to another actor/run/policy')
        old,new=manifest['actorProgress'],boundary['actorProgress']
        if new['rngStreamID']!=old['rngStreamID'] or new['drawIndex']<old['drawIndex'] or new['actorResetGeneration']<old['actorResetGeneration']:
            raise JobError('job.boundaryMismatch','Boundary actor progress is stale or from another random stream')
        if new['drawIndex']==old['drawIndex']:
            if not _same_progress(new,old):raise JobError('job.boundaryMismatch','Unchanged draw count has inconsistent random/reset state')
        elif other['previousActorProgress'] is None or not _same_progress(other['previousActorProgress'],old):
            raise JobError('job.boundaryMismatch','Continuity audit does not begin at the sealed rollout random state')
        _claim(Path(job.external_boundary_path).parent,'.boundary-'+boundary['id']+'.json',
               {'jobID':job.identifier,'jobRunID':job.request.run_id,'actorRunID':boundary_binding['runID'],'actorProgress':new})
        interrupted=interrupted or job.cancel.is_set()
        if interrupted and completed is not None:
            # Publication is the external job's commit point. Cancellation
            # while waiting for actor closure also rolls back the proposal.
            trainer.policy.update(tree_unflatten(before_weights));trainer._restore(before_state);trainer._pending=None
            completed=None
        trainer.stop('External learner source handle retired')
        learner=trainer.state;learner['policyID']=Path(value['destination']).name
        state={'kind':'reinforcement_external','schemaVersion':1,'learner':learner,'actorProgress':new,
               'consumedRolloutIDs':history,'requiresEnvironmentReset':True}
        metrics={'sourceKind':'external_rollout','provenance':'external_rollout','cancelled':interrupted,
            'requiresEnvironmentReset':True,'actorProgress':new,'rolloutID':manifest['rolloutID'],
            'boundaryCollectionID':boundary['id'],'iteration':trainer.iteration,'optimizer_updates':trainer.optimizer_updates}
        if completed is not None:metrics['lastIteration']=completed
        report('checkpointing',cancelled=interrupted)
        saved=save_checkpoint(Path(value['destination']),trainer.policy,kind='reinforcement',step=trainer.optimizer_updates,
            training_state=state,parent_id=loaded.manifest['id'],metrics=metrics,training_config=asdict(training))
        return {'checkpointPath':value['destination'],'manifest':saved,'checkpointPublished':True,'cancelled':interrupted,
            'resumable':True,'requiresEnvironmentReset':True,'sourceKind':'external_rollout','provenance':'external_rollout',
            'actorProgress':new,'boundaryCollectionID':boundary['id'],'rolloutID':manifest['rolloutID'],'metrics':completed,
            'parameterCount':trainer.policy.config.parameter_count}
    finally:
        if rollout is not None:rollout.close()
        trainer.stop('External learner finished without owning native controls')
