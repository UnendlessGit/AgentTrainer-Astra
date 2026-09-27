#!/usr/bin/env python3
"""One selected-checkpoint zero-wait mechanism probe; no production changes."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from scripts.gru_loss_clipping import encoded,digest,publish,read,check_reference,checkpoint_reference,schedule
from astra.environments.practice import PracticeConfig,PracticeEnvironment

TRAIN=(0,1,2,3)
DEV=tuple(range(1000,1008))
IMPLEMENTATION=('experiments/temporal_study.py','python/astra/environments/practice.py',
    'python/astra/data/history.py','python/astra/model/temporal.py','python/astra/model/policy.py',
    'python/astra/model/actions.py','python/astra/learning/optimizers.py','python/astra/data/observations.py')


class ZeroWaitConfig(PracticeConfig):
    """Diagnostic-only extension: admit an actual zero-length waiting phase.

    Production PracticeConfig continues to require a positive delay. Rendering,
    oracle readiness, physics and rewards all read this same zero-delay config.
    """
    def validate(self):
        if type(self.delay_ms) is not int or self.delay_ms!=0:raise ValueError('This control has exactly zero waiting delay')
        PracticeConfig(**{**asdict(self),'delay_ms':1}).validate()
        return self


def world(seed,lane):
    environment=PracticeEnvironment(ZeroWaitConfig(task='delayed_memory',delay_ms=0,time_limit_ms=2500))
    raw=environment.reset(seed)
    if lane:
        environment._answer=1-environment._answer
        raw=environment._observe()
    return environment,raw


def visual_key(raw,model,visual_digest):
    value=hashlib.sha256(raw.pixels.tobytes())
    value.update(encoded(dict(surface=raw.metadata['surface'],pointer=raw.control_state['pointer'],
        controlValid=raw.control_state['valid'],pixelFormat=raw.metadata['pixelFormat'],model=model,visualDigest=visual_digest)))
    return value.hexdigest()


def control_features(raw):
    from astra.data.history import ControlHistory
    value=raw.control_state
    history=ControlHistory(keys=set(value['keys']),buttons=set(value['buttons']),modifiers=value['modifiers'],
        pointer=(value['pointer']['x'],value['pointer']['y']),valid=value['valid'],observed_nanos=value['observedNanos'])
    return history.features(raw.metadata['observedNanos'],[raw.metadata['surface']],interval_covered=True)


def prepare(args):
    import numpy as np
    root=args.root.resolve()
    if root.exists():raise ValueError('Choose a new diagnostic directory; preserve all evidence')
    campaign=args.campaign.resolve(); old=read(campaign/'plan.json'); source=read(campaign/'run-836-B.json')
    if source['completedUpdates']!=512 or source['phase']!='completed':raise ValueError('The fixed836B endpoint is incomplete')
    check_reference(source['checkpoint'])
    manifest=read(Path(source['checkpoint']['path'])/'manifest.json')
    if manifest['model'].get('schema_version')!=2:raise ValueError('This diagnostic excludes queued-control architecture')
    cache_root=Path(old['visualRoot'])/old['visualDigest']
    episodes=[]; proofs=[]; tensors={}
    training_sources={entry['seed']:[read(row['path']) for row in pair] for pair in old['delayed']['groups'] for entry in pair[:1]}
    for seed in (*TRAIN,*DEV):
        phase_paths=list(Path(old['phaseRoot']).glob('*/'+str(seed)+'.npz'))
        if len(phase_paths)!=1:raise ValueError('The original phase fixture is missing or ambiguous; do not generate a substitute')
        with np.load(phase_paths[0],allow_pickle=False) as phases:
            metadata=json.loads(str(phases['metadata'].item()))
            if metadata['seed']!=seed:raise ValueError('Original phase identity changed')
            for lane in (0,1):
                environment,raw=world(seed,lane); steps=[]
                for index in range(6):
                    phase_index=lane if index<5 else 3
                    image_name=('cueA','cueB')[lane] if index<5 else 'choice'
                    original=metadata['phases'][phase_index]
                    if (not np.array_equal(raw.pixels,phases[image_name]) or raw.metadata['surface']!=original['metadata']['surface'] or
                        any(raw.control_state[name]!=original['controlState'][name] for name in raw.control_state if name!='observedNanos')):
                        raise ValueError(f'Zero-delay renderer differs from original source: seed{seed}/lane{lane}/step{index}; STOP')
                    key=visual_key(raw,manifest['model'],old['visualDigest'])
                    descriptor=read(cache_root/(key+'.json')); tensor=cache_root/(key+'.safetensors')
                    if descriptor['visualDigest']!=old['visualDigest'] or descriptor['tensorHash']!=digest(tensor) or descriptor['bytes']!=tensor.stat().st_size:
                        raise ValueError('Original visual features changed; STOP')
                    tensors[key]={'bytes':descriptor['bytes'],'sha256':descriptor['tensorHash']}
                    commands=environment.oracle_commands(); features=control_features(raw).tolist()
                    if bool(commands)!=(index==5) or (commands and commands!=metadata['labels'][lane]):raise ValueError('Genuine zero-delay readiness/labels do not match the original target')
                    if seed in TRAIN:
                        before=training_sources[seed][lane]['steps'][index if index<5 else 25]
                        if (key,features,commands)!=(before['visual'],before['controls'],before['commands']):raise ValueError('Original training feature/control/action identity mismatch; STOP')
                    steps.append({'visual':key,'controls':features,'commands':commands,'surface':raw.metadata['surface'],
                        'observedNanos':raw.metadata['observedNanos'],'pixelSHA256':hashlib.sha256(raw.pixels.tobytes()).hexdigest()})
                    if index<5:raw=environment.step([],episode_id=raw.episode_id).observation
                pair=str(uuid.uuid5(uuid.NAMESPACE_URL,'astra:zero-wait-control:'+str(seed)))
                episodes.append({'schemaVersion':1,'id':str(uuid.uuid5(uuid.UUID(pair),str(lane))), 'pairID':pair,
                    'seed':seed,'counterfactual':bool(lane),'delayMS':0,'environment':environment.config.to_dict(),
                    'steps':steps,'provenance':'diagnostic_actual_zero_delay_passive_prefix_with_oracle_readout_label'})
        proofs.append({'seed':seed,'originalPhasePath':str(phase_paths[0]),'originalPhaseSHA256':digest(phase_paths[0]),
                       'pixelControlGeometryAndTargetsMatch':True})
    groups=[[row for row in episodes if row['seed']==seed] for seed in TRAIN]
    plan={'schemaVersion':1,'scope':'selected836B_zero_wait_cue_to_choice_control','productionDefaultsChanged':False,
        'modelSchemaVersion':2,'sourceCheckpoint':source['checkpoint'],'sourceCampaignPlanSHA256':digest(campaign/'plan.json'),
        'sourceReportSHA256':digest(campaign/'run-836-B.json'),'visualRoot':old['visualRoot'],'visualDigest':old['visualDigest'],
        'trainingLayouts':TRAIN,'developmentLayouts':DEV,'reservedTestUsed':False,'headSeed':836,'optimizerSeed':834,
        'targetUpdates':512,'targetSupervisedPackets':1024,'cueFrames':5,'waitingFrames':0,'choiceFrames':1,
        'optimizer':{'learningRate':3e-4,'weightDecay':.01,'gradientNorm':1.,'epsilon':1e-8,'freshMoments':True,'normalization':'supervised_choice_packets'},
        'episodes':episodes,'order':schedule(groups,512),'renderProofs':proofs,'originalVisualTensors':tensors,
        'runnerSHA256':digest(__file__),'implementation':{name:digest(ROOT/name) for name in IMPLEMENTATION},
        'dependencies':{name:importlib.metadata.version(name) for name in ('mlx','numpy','Pillow')},
        'readout':'one greedy and two categorical packets executed separately after actual zero-delay passive readiness; no autonomous waiting',
        'primaryMetrics':['semanticTargetSuccess','bothCuesCorrectLayouts','selectedTargetChangesWithCue'],
        'secondaryMetrics':['exactCenterPacketRanking','factorMargins','choiceNLL','gradientClipping']}
    root.mkdir(parents=True);publish(root/'plan.json',plan)
    print(json.dumps({'phase':'prepared_cpu_only','plan':str(root/'plan.json'),'planSHA256':digest(root/'plan.json'),
        'runnerSHA256':plan['runnerSHA256'],'episodes':len(episodes),'matchedVisualEntries':len(tensors),'matchedOriginalPixels':True}),flush=True)


def load_plan(root):
    plan=read(root/'plan.json')
    if plan['runnerSHA256']!=digest(__file__):raise ValueError('Frozen runner changed')
    for path,expected in plan['implementation'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Frozen implementation changed: '+path)
    for name,version in plan['dependencies'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Frozen dependency changed')
    check_reference(plan['sourceCheckpoint'])
    return plan


def execute_packet(commands,seed,lane):
    environment,raw=world(seed,lane)
    for _ in range(5):raw=environment.step([],episode_id=raw.episode_id).observation
    correct_index=environment._choice_colors.index(environment._answer)
    total=0.;statuses=[]
    try:
        for index in range(4):
            result=environment.step(commands if index==0 else [],episode_id=raw.episode_id)
            total+=result.reward;statuses.extend(row['status'] for row in result.command_results)
            if result.outcome!='continuing':break
        outcome='success' if environment.outcome=='terminated' and total>0 else ('wrong_choice' if environment.outcome=='terminated' else 'no_choice')
        return {'outcome':outcome,'reward':total,'selectedTarget':correct_index if outcome=='success' else (1-correct_index if outcome=='wrong_choice' else None),'commandStatuses':statuses}
    except ValueError as error:return {'outcome':'invalid_packet','issue':str(error),'selectedTarget':None}
    finally:
        if environment.outcome=='continuing':environment.abort('Zero-wait one-packet readout complete')


def evaluate(head,cache,plan):
    import mlx.core as mx
    import numpy as np
    from astra.data.actions import encode_commands,decode_commands
    from astra.model.actions import PacketBatch
    from experiments.temporal_study import episode_batch
    rows=[];head.eval()
    for seed in (*TRAIN,*DEV):
        pair=[episode for episode in plan['episodes'] if episode['seed']==seed]
        batch=episode_batch(cache,pair);result=head.temporal(batch.summary,batch.observation)
        context=result.context[:,-1];visual=cache.batch([row['steps'][-1]['visual'] for row in pair])
        labels=[encode_commands(row['steps'][-1]['commands'],config=head.config,vocabulary=head.actions.vocabulary,
            visual=cache.get(row['steps'][-1]['visual']),surfaces=[row['steps'][-1]['surface']]) for row in pair]
        correct=PacketBatch(**{name:mx.concatenate([getattr(label,name) for label in labels]) for name in PacketBatch.__dataclass_fields__})
        wrong=PacketBatch(**{name:getattr(correct,name)[mx.array([1,0])] for name in PacketBatch.__dataclass_fields__})
        good=head.actions.log_prob(context,visual,correct);bad=head.actions.log_prob(context,visual,wrong)
        mx.eval(good.log_probability,bad.log_probability,good.factor_log_probabilities,bad.factor_log_probabilities)
        row={'seed':seed,'split':'train' if seed in TRAIN else 'development','correctLogProbability':np.asarray(good.log_probability).tolist(),
            'wrongLogProbability':np.asarray(bad.log_probability).tolist(),'margins':np.asarray(good.log_probability-bad.log_probability).tolist(),
            'factorMargins':{name:np.asarray(mx.sum(value-bad.factor_log_probabilities[name],axis=-1)).tolist() for name,value in good.factor_log_probabilities.items()},'readouts':[]}
        for draw,greedy in enumerate((True,False,False)):
            sampled=head.actions.sample(context,visual,key=mx.random.key(71900+len(rows)*3+draw),greedy=greedy)
            mx.eval(sampled.packets.operation,sampled.log_probability)
            for lane in (0,1):
                commands=decode_commands(sampled.packets,config=head.config,vocabulary=head.actions.vocabulary,visual=visual,
                    surfaces=[pair[lane]['steps'][-1]['surface']],batch_index=lane)
                row['readouts'].append({'cueLane':lane,'draw':draw,'greedy':greedy,'firstOperation':int(sampled.packets.operation[lane,0]),
                    'commands':commands,'execution':execute_packet(commands,seed,lane)})
        rows.append(row)
    summaries=[]
    for split in ('train','development'):
        selected=[row for row in rows if row['split']==split];greedy=[item for row in selected for item in row['readouts'] if item['greedy']]
        sampled=[item for row in selected for item in row['readouts'] if not item['greedy']]
        pairs=[[item['execution'] for item in row['readouts'] if item['greedy']] for row in selected]
        summaries.append({'split':split,'layouts':len(selected),'greedySuccesses':sum(item['execution']['outcome']=='success' for item in greedy),
            'greedyTrials':len(greedy),'sampledSuccesses':sum(item['execution']['outcome']=='success' for item in sampled),'sampledTrials':len(sampled),
            'bothCuesCorrectLayouts':sum(all(item['outcome']=='success' for item in pair) for pair in pairs),
            'selectedTargetChangesWithCue':sum(all(item['selectedTarget'] is not None for item in pair) and pair[0]['selectedTarget']!=pair[1]['selectedTarget'] for pair in pairs),
            'centerPacketRankingCorrect':sum(margin>0 for row in selected for margin in row['margins']),
            'firstOperationCounts':{str(op):sum(item['firstOperation']==op for item in greedy) for op in sorted({item['firstOperation'] for item in greedy})}})
    return {'rows':rows,'summaries':summaries,'scope':plan['readout']}


def run(root,plan):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_map,tree_flatten
    from astra.checkpoints import load_checkpoint,save_checkpoint,restore_mlx_random_state
    from astra.learning.optimizers import GroupedAdamW,finite_gradients
    from experiments.temporal_study import FrozenHead,VisualCache,episode_batch,cached_batch_gradients
    if (root/'result.json').exists() or (root/'started.json').exists():raise ValueError('One fixed run only; do not retry toward a positive endpoint')
    publish(root/'started.json',{'planSHA256':digest(root/'plan.json'),'runnerSHA256':digest(__file__)})
    began=time.perf_counter();mx.set_memory_limit(10*1024**3);mx.set_cache_limit(64*1024**2)
    loaded=load_checkpoint(Path(plan['sourceCheckpoint']['path']));loaded.policy.vision.freeze()
    cache=VisualCache(Path(plan['visualRoot']),loaded.policy,expected_digest=plan['visualDigest']);head=FrozenHead(loaded.policy)
    baseline=evaluate(head,cache,plan);publish(root/'baseline.json',baseline)
    head.train();mx.random.seed(plan['optimizerSeed'])
    optimizer=GroupedAdamW(learning_rate=3e-4,pretrained_learning_rate=3e-5,weight_decay=.01)
    episodes={episode['id']:episode for episode in plan['episodes']};metrics=[]
    for index,identities in enumerate(plan['order']):
        batch=episode_batch(cache,[episodes[identity] for identity in identities])
        result=cached_batch_gradients(head,batch,horizon=512,choice_only=True)
        if (result.valid_count,result.selected_count)!=(12,2):raise ValueError('Fixed exposure changed')
        scale=result.valid_count/result.selected_count;gradients=tree_map(lambda value:value*scale,result.gradients)
        if not finite_gradients(gradients):raise FloatingPointError('Nonfinite diagnostic gradients')
        gradients,norm=optim.clip_grad_norm(gradients,1.);mx.eval(gradients,norm)
        optimizer.update(head,gradients);mx.eval(head.parameters(),optimizer.state)
        if not finite_gradients(head.parameters()):raise FloatingPointError('Nonfinite diagnostic parameters')
        metrics.append({'update':index+1,'episodeIDs':identities,'validDecisions':result.valid_count,'supervisedPackets':result.selected_count,
            'choiceNLL':float(result.loss)*scale,'preClipNorm':float(norm),'clipMultiplier':min(1.,1./(float(norm)+1e-6)),
            'cueSummaryGradientNorm':float(mx.sqrt(mx.sum(result.summary_gradient[:,:5]**2)))*scale})
        if (index+1)%64==0:
            publish(root/'progress.json',{'updates':index+1,'metrics':metrics,'seconds':time.perf_counter()-began})
            print(json.dumps({'updates':index+1,'choiceNLL':metrics[-1]['choiceNLL']}),flush=True)
    checkpoint=root/'checkpoints'/str(uuid.uuid4())
    save_checkpoint(checkpoint,loaded.policy,kind='behavioral',step=loaded.manifest['step']+512,parent_id=loaded.manifest['id'],
        training_state={'kind':'experimental-zero-wait-control','planSHA256':digest(root/'plan.json'),'optimizer':optimizer.state,
            'rng':tuple(mx.random.state),'updates':512,'metrics':metrics},training_config={'diagnostic':plan['scope'],'planSHA256':digest(root/'plan.json')})
    outcome=evaluate(head,cache,plan);publish(root/'endpoint.json',outcome)
    final={'schemaVersion':1,'scope':plan['scope'],'planSHA256':digest(root/'plan.json'),'runnerSHA256':digest(__file__),
        'checkpoint':checkpoint_reference(checkpoint),'completedUpdates':512,'supervisedPackets':sum(row['supervisedPackets'] for row in metrics),
        'validDecisions':sum(row['validDecisions'] for row in metrics),'metrics':metrics,'baseline':baseline['summaries'],'endpoint':outcome['summaries'],
        'wallSeconds':time.perf_counter()-began,'peakMLXBytes':mx.get_peak_memory(),'reservedTestUsed':False,'productionDefaultsChanged':False}
    publish(root/'result.json',final);print(json.dumps({name:value for name,value in final.items() if name!='metrics'},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','run'));parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,default=ROOT/'.local/gru-loss-clipping-2026-09-27');args=parser.parse_args()
    if args.mode=='prepare':prepare(args)
    else:run(args.root.resolve(),load_plan(args.root.resolve()))


if __name__=='__main__':main()
