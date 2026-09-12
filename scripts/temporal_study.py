#!/usr/bin/env python3
"""Experimental frozen-visual GRU study. Never changes application defaults."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common-checkpoint',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--visual-root',type=Path,help='Reuse immutable features from the common visual checkpoint')
    parser.add_argument('--base-dataset',type=Path,help='Reuse verified original-cue episodes when building a separate paired revision')
    parser.add_argument('--cue-replay',action='store_true',help='Experimental rendered cue repeat immediately before choices')
    parser.add_argument('--paired',action='store_true',help='Both visible cues per training layout; balanced paired optimizer batches')
    parser.add_argument('--mode',choices=('pilot','prepare','train'),required=True)
    parser.add_argument('--phase-cache',type=Path,help='Reuse exact paired raw phase fixtures')
    parser.add_argument('--arm',choices=('A','B','C','D'),default='B')
    parser.add_argument('--model-seed',type=int,default=834)
    parser.add_argument('--updates',type=int,default=192)
    parser.add_argument('--seconds',type=float,default=360)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if not 1<=args.seconds<=360 or args.updates<1: parser.error('Each process is bounded to at most six minutes')
    if args.cue_replay and args.base_dataset: parser.error('Cue-replay episodes must be rendered through the actual experimental environment')
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from astra.checkpoints import load_checkpoint,save_checkpoint,restore_mlx_random_state
    from astra.environments.practice import PracticeConfig
    from astra.learning.optimizers import GroupedAdamW,finite_gradients
    from astra.model.policy import AgentPolicy
    from experiments.temporal_study import VisualCache,FrozenHead,prepare_episode,episode_batch,cached_batch_gradients,paired_validation,_file_hash,_publish_json
    from scripts.qualify_memory import initialize_experimental_retention

    began=time.perf_counter();deadline=began+args.seconds
    mx.set_memory_limit(10*1024**3);mx.set_cache_limit(64*1024**2)
    args.root.mkdir(parents=True,exist_ok=True)
    common=load_checkpoint(args.common_checkpoint)
    cache=VisualCache(args.visual_root or args.root/'visual',common.policy)
    parameters=dict(arm=args.arm,horizon=64 if args.arm=='A' else 512,choiceOnly=args.arm in ('C','D'),
                    initialization='native' if args.arm=='D' else 'geometric',modelSeed=args.model_seed,
                    visualDigest=cache.digest,learningRate=3e-4,weightDecay=.01,gradientNorm=1.,
                    normalization='total_valid_decisions_in_two_complete_episodes')
    if args.paired: parameters['pairedData']=True
    if args.cue_replay: parameters['cueReplay']=True
    output=args.root/(args.mode+'-'+args.arm+'-'+str(args.model_seed)+'.json')
    report=dict(schemaVersion=1,scope='frozen_visual_temporal_diagnostic',parameters=parameters,
                commonCheckpoint=str(args.common_checkpoint.resolve()),metrics=[],checkpoints=[])
    def publish(phase,**values):
        report.update(values);report['phase']=phase;report['wallSeconds']=time.perf_counter()-began
        report['peakMLXBytes']=mx.get_peak_memory();report['processPeakRSSBytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _publish_json(output,report)
        print(json.dumps(dict(phase=phase,elapsedSeconds=report['wallSeconds'],**values),allow_nan=False),flush=True)
    def cancelled(): return time.perf_counter()>=deadline
    base_entries={}
    if args.base_dataset:
        base=json.loads(args.base_dataset.read_text())
        if (not args.paired or base['visualDigest']!=cache.digest or base['model']!=common.policy.config.to_dict()
            or base['actions']!=common.policy.actions.vocabulary.to_dict()): raise ValueError('Paired source base identity mismatch')
        base_entries={(entry['delayMS'],entry['seed']):entry for entry in base['episodes']}
    def episode(delay,seed,counterfactual=False):
        environment=PracticeConfig(task='delayed_memory',delay_ms=delay,time_limit_ms=delay+2500)
        path=args.root/'episodes'/f'{delay}-{seed}{"-opposite" if counterfactual else ""}.json'
        if path.exists():
            value=json.loads(path.read_text())
            if value['environment']!=environment.to_dict() or value['seed']!=seed or bool(value.get('counterfactual',False))!=counterfactual or bool(value.get('cueReplay',False))!=args.cue_replay: raise ValueError('Cached episode identity mismatch')
            return value
        if not counterfactual and (delay,seed) in base_entries:
            entry=base_entries[(delay,seed)]
            if _file_hash(entry['path'])!=entry['sha256']: raise ValueError('Base source episode integrity changed')
            value=json.loads(Path(entry['path']).read_text())
            if value['environment']!=environment.to_dict(): raise ValueError('Base environment changed')
            value['pairID']=value['id'];value['counterfactual']=False
        else:
            value=prepare_episode(cache,environment,seed,counterfactual=counterfactual,cue_replay=args.cue_replay,cancelled=cancelled)
        _publish_json(path,value)
        return value
    publish('preparing')
    if args.mode=='prepare':
        entries=[]
        try:
            for delay,base in ((2000,0),(8000,100),(30000,200)):
                for seed in range(base,base+32):
                    pair=[]
                    for opposite in (False,True) if args.paired else (False,):
                        item=episode(delay,seed,opposite)
                        path=args.root/'episodes'/f'{delay}-{seed}{"-opposite" if opposite else ""}.json'
                        entries.append(dict(id=item['id'],delayMS=delay,seed=seed,path=str(path.resolve()),sha256=_file_hash(path)))
                        pair.append(item)
                    if args.paired:
                        if len(pair[0]['steps'])!=len(pair[1]['steps']): raise ValueError('Paired episode lengths differ')
                        cue_steps=5
                        for index,(left,right) in enumerate(zip(pair[0]['steps'],pair[1]['steps'])):
                            if left['controls']!=right['controls'] or left['surface']!=right['surface']:
                                raise ValueError('A paired cue changed control/geometry features')
                            visible_cue=index<cue_steps or (args.cue_replay and index==(delay+500)//100-1)
                            if (left['visual']==right['visual'])==visible_cue:
                                raise ValueError('Paired inputs must differ only while the cue is visible')
                    publish('preparing',preparedEpisodes=len(entries))
            manifest=dict(schemaVersion=1,visualDigest=cache.digest,episodes=entries,model=common.policy.config.to_dict(),
                          actions=common.policy.actions.vocabulary.to_dict(),provenance='practice_oracle',paired=args.paired,cueReplay=args.cue_replay)
            _publish_json(args.root/'dataset.json',manifest)
            publish('prepared',episodes=len(entries),datasetHash=_file_hash(args.root/'dataset.json'),
                    visualBytes=sum(path.stat().st_size for path in cache.root.glob('*.safetensors')))
        except InterruptedError: publish('paused',reason='Preparation deadline; completed cache entries remain reusable')
        return

    mx.random.seed(args.model_seed)
    policy=AgentPolicy(common.policy.config,common.policy.actions.vocabulary)
    policy.vision=common.policy.vision
    initialize_experimental_retention(policy,parameters['initialization'])
    policy.vision.freeze();head=FrozenHead(policy);head.train()
    optimizer=GroupedAdamW(learning_rate=3e-4,pretrained_learning_rate=3e-5,weight_decay=.01)
    completed=0
    dataset_hash=None
    if args.mode=='train':
        path=args.root/'dataset.json';dataset_hash=_file_hash(path);manifest=json.loads(path.read_text())
        if (manifest['visualDigest']!=cache.digest or manifest['model']!=common.policy.config.to_dict()
            or manifest['actions']!=common.policy.actions.vocabulary.to_dict()): raise ValueError('Dataset model/visual/action identity changed')
        report['datasetHash']=dataset_hash
        source=[]
        for entry in manifest['episodes']:
            if _file_hash(entry['path'])!=entry['sha256']: raise ValueError('Source episode integrity changed')
            source.append(json.loads(Path(entry['path']).read_text()))
        expected=192 if args.paired else 96
        if len(source)!=expected or len({item['id'] for item in source})!=expected or bool(manifest.get('paired',False))!=args.paired or bool(manifest.get('cueReplay',False))!=args.cue_replay:
            raise ValueError('The planned study has inconsistent episode count/pairing')
        groups=None
        if args.paired:
            grouped={}
            for item in source: grouped.setdefault(item['pairID'],[]).append(item)
            groups=list(grouped.values())
            if len(groups)!=96 or any(len(pair)!=2 or {item['counterfactual'] for item in pair}!={False,True} for pair in groups):
                raise ValueError('Each training layout needs both opposite visible cues')
        if args.resume:
            loaded=load_checkpoint(args.resume,include_training=True);state=loaded.training_state
            if not state or state['kind']!='experimental-temporal' or state['parameters']!=parameters or state['datasetHash']!=dataset_hash:
                raise ValueError('Resume requires identical study parameters and data')
            # Every visual weight must still match the common immutable cache.
            VisualCache(args.visual_root or args.root/'visual',loaded.policy,expected_digest=cache.digest)
            policy=loaded.policy;policy.vision.freeze();head=FrozenHead(policy);head.train()
            optimizer.state=state['optimizer'];completed=state['updates'];restore_mlx_random_state(state['rng'])
            report['metrics']=state.get('metricsHistory',[])
            report['validation']=state.get('validationHistory',[])
            report['checkpoints']=[str(args.resume)]
            publish('resumed',startingUpdates=completed,resumedFrom=str(args.resume))
    def update(batch):
        result=cached_batch_gradients(head,batch,horizon=parameters['horizon'],choice_only=parameters['choiceOnly'],cancelled=cancelled)
        if not bool(mx.isfinite(result.loss).item()) or not finite_gradients(result.gradients): raise FloatingPointError('Nonfinite diagnostic gradients')
        clipped,norm=optim.clip_grad_norm(result.gradients,1.);mx.eval(clipped,norm)
        previous_parameters=head.parameters();previous_optimizer=deepcopy(optimizer.state)
        try:
            optimizer.update(head,clipped);mx.eval(head.parameters(),optimizer.state)
        except BaseException:
            head.update(previous_parameters);optimizer.state=previous_optimizer;raise
        choice=np.asarray(batch.observation.valid).reshape(-1)&(np.asarray(batch.packets.operation[:,0])!=0)
        waiting=np.asarray(batch.observation.valid).reshape(-1)&~choice
        logp=np.asarray(result.log_probabilities)
        return dict(loss=float(result.loss.item()),choiceNLL=float(-logp[choice].mean()),
                    waitingNLL=None if parameters['choiceOnly'] or not waiting.any() else float(-logp[waiting].mean()),
                    validDecisions=result.valid_count,selectedDecisions=result.selected_count,gradientNorm=float(norm.item()),
                    cueSummaryGradientNorm=float(mx.sqrt(mx.sum(result.summary_gradient[:,:5]**2)).item()))
    last_saved_updates=completed if args.resume else -1
    def save():
        nonlocal last_saved_updates
        if completed==last_saved_updates: return Path(report['checkpoints'][-1])
        path=args.root/'checkpoints'/str(uuid.uuid4())
        state=dict(kind='experimental-temporal',parameters=parameters,datasetHash=dataset_hash,updates=completed,
                   optimizer=optimizer.state,rng=tuple(mx.random.state),metricsHistory=report['metrics'],validationHistory=report.get('validation',[]))
        save_checkpoint(path,policy,kind='behavioral',step=completed,training_state=state,
                        parent_id=Path(report['checkpoints'][-1]).name if report['checkpoints'] else args.common_checkpoint.name,
                        metrics=dict(experimentalScope='frozen_visual_temporal_diagnostic',updates=completed),training_config=parameters)
        report['checkpoints'].append(str(path));last_saved_updates=completed;return path
    def validate_current():
        head.eval()
        try:
            validation=paired_validation(head,cache,args.phase_cache or args.root/'phase-fixtures',cue_replay=args.cue_replay,cancelled=cancelled)
        finally:
            head.train()
        report.setdefault('validation',[]).append(dict(update=completed,**validation))
        publish('validated',completedUpdates=completed,byDelay=validation['byDelay'],meanCueSummaryRMS=validation['meanCueSummaryRMS'])
    try:
        if args.mode=='pilot':
            samples=([episode(30000,200,opposite) for opposite in (False,True)] if args.paired
                     else [episode(30000,seed) for seed in (200,201)])
            batch=episode_batch(cache,samples);mx.reset_peak_memory()
            for index in range(2):
                started=time.perf_counter();metrics=update(batch);metrics['updateSeconds']=time.perf_counter()-started
                completed=index+1
                report['metrics'].append(metrics);publish('pilotUpdate',completed=completed,last=metrics)
            publish('pilotComplete',boundedMemory=mx.get_peak_memory()<10*1024**3)
            return
        if args.resume and completed in (32,64,128,192) and not any(value['update']==completed for value in report.get('validation',[])):
            validate_current()
        while completed<args.updates and not cancelled():
            epoch,position=divmod(completed,len(groups) if groups is not None else len(source)//2)
            order=np.random.default_rng(np.random.SeedSequence([args.model_seed,epoch])).permutation(len(groups) if groups is not None else len(source))
            selected=groups[int(order[position])] if groups is not None else [source[int(index)] for index in order[position*2:position*2+2]]
            batch=episode_batch(cache,selected);started=time.perf_counter();metrics=update(batch)
            completed+=1;metrics.update(update=completed,updateSeconds=time.perf_counter()-started,episodeIDs=[item['id'] for item in selected])
            report['metrics'].append(metrics);publish('training',completedUpdates=completed,last=metrics)
            if completed in (32,64,128,192):
                publish('checkpoint',completedUpdates=completed,checkpoint=str(save()))
                validate_current()
    except InterruptedError:
        if args.mode=='pilot':
            publish('pilotPaused',completed=completed,reason='Deadline; no incomplete update is reported as a pilot result')
            return
    except BaseException as error:
        publish('failed',completedUpdates=completed,issue=str(error))
        raise
    checkpoint=save()
    validated=any(value['update']==completed for value in report.get('validation',[]))
    phase=('completed' if validated else 'validationPending') if completed==args.updates else 'paused'
    publish(phase,completedUpdates=completed,targetReached=completed==args.updates,validationAtTarget=validated,checkpoint=str(checkpoint))

if __name__=='__main__':main()
