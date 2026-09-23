#!/usr/bin/env python3
"""Bounded GRU exposure/weighting diagnostic; never changes production defaults."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=360)
    parser.add_argument('--updates',type=int,default=512)
    args=parser.parse_args()
    if not 1<=args.seconds<=360 or not 1<=args.updates<=512:parser.error('Bounded to six minutes and512 complete updates per arm')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_map,tree_flatten
    import numpy as np
    from astra.checkpoints import load_checkpoint,save_checkpoint
    from astra.learning.optimizers import GroupedAdamW,finite_gradients
    from experiments.temporal_study import VisualCache,FrozenHead,episode_batch,cached_batch_gradients,paired_validation
    initial=ROOT/'.local/temporal-immediate-study/checkpoints/2531737d-4b0c-44f0-b058-66f87378d335'
    common=ROOT/'.local/verification/3ea5799a-2b26-4e7a-912a-ad32412ad171'
    began=time.perf_counter();deadline=began+args.seconds
    mx.set_memory_limit(10*1024**3);mx.set_cache_limit(64*1024**2)
    shared=load_checkpoint(common);cache=VisualCache(ROOT/'.local/temporal-study/visual',shared.policy)
    datasets={}
    hashes={}
    for replay,folder in [(False,'temporal-paired-study'),(True,'temporal-immediate-study')]:
        path=ROOT/'.local'/folder/'dataset.json';manifest=json.loads(path.read_text());hashes[folder]=hashlib.sha256(path.read_bytes()).hexdigest()
        pairs={seed:[] for seed in range(4)}
        for entry in manifest['episodes']:
            if entry['delayMS']!=2000 or entry['seed'] not in pairs:continue
            data=Path(entry['path']).read_bytes()
            if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Immutable source episode changed')
            pairs[entry['seed']].append(json.loads(data))
        if any(len(pair)!=2 for pair in pairs.values()):raise ValueError('Both original rendered cues are required for every layout')
        datasets[replay]={seed:sorted(pair,key=lambda row:row['counterfactual']) for seed,pair in pairs.items()}
    for seed in range(4):
        for a,b in zip(datasets[False][seed],datasets[True][seed]):
            for step,(left,right) in enumerate(zip(a['steps'],b['steps'])):
                if any(left[key]!=right[key] for key in ('controls','commands','surface','observedNanos')):
                    raise ValueError('Replay intervention changed labels, time or control inputs')
                if (left['visual']!=right['visual'])!=(step==24):raise ValueError('Unexpected replay intervention')
    report={'schemaVersion':1,'scope':'frozen_visual_GRU_exposure_weighting_mechanism_pilot',
        'initialCheckpoint':str(initial),'initialPolicySHA256':hashlib.sha256((initial/'policy.safetensors').read_bytes()).hexdigest(),
        'commonCheckpoint':str(common),'visualDigest':cache.digest,'datasetHashes':hashes,
        'initialization':'same immutable C192 checkpoint in every arm; fresh AdamW moments and identical MLX seed834',
        'optimizer':{'learningRate':3e-4,'weightDecay':.01,'clipNorm':1.,'epsilon':1e-8,'biasCorrection':True},
        'trainSeeds':list(range(4)),'heldoutDevelopmentSeeds':list(range(1000,1008)),
        'sequenceHorizon':512,'targetAdditionalUpdates':args.updates,'productionDefaultsChanged':False,'arms':[]}
    def expired():return time.perf_counter()>=deadline
    def write():
        report['wallSeconds']=time.perf_counter()-began;report['peakMLXBytes']=mx.get_peak_memory()
        temp=output/'report.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(output/'report.json')
    def validate(head,arm,step):
        head.eval()
        item={'updates':step,'scope':'forced correct-vs-wrong complete packet likelihood; not autoregressive success','conditions':[]}
        for replay in (False,True):
            result=paired_validation(head,cache,ROOT/'.local/temporal-study/phase-fixtures',seeds=[0,1,2,3,*range(1000,1008)],
                cue_replay=replay,cancelled=expired)
            summary=[]
            for split,allowed in [('train',range(4)),('development',range(1000,1008))]:
                for delay in (2000,8000,30000):
                    rows=[row for row in result['rows'] if row['seed'] in allowed and row['delayMS']==delay]
                    summary.append({'split':split,'delayMS':delay,'accuracy':sum(r['correct'] for r in rows)/(2*len(rows)),
                        'bothCuesCorrect':sum(all(m>0 for m in r['margins']) for r in rows),'layouts':len(rows),
                        'meanMargin':float(np.mean([m for r in rows for m in r['margins']]))})
            item['conditions'].append({'cueReplay':replay,'summary':summary,'rows':result['rows']})
        arm['validation'].append(item);head.train();write()
    write()
    for name,replay,per_choice in [('delayed_valid',False,False),('delayed_choice',False,True),('immediate_choice',True,True)]:
        if expired():break
        loaded=load_checkpoint(initial);VisualCache(ROOT/'.local/temporal-study/visual',loaded.policy,expected_digest=cache.digest)
        loaded.policy.vision.freeze();head=FrozenHead(loaded.policy);head.train();mx.random.seed(834)
        optimizer=GroupedAdamW(learning_rate=3e-4,pretrained_learning_rate=3e-5,weight_decay=.01)
        arm={'name':name,'cueReplay':replay,'denominator':'supervised_choices' if per_choice else 'all_valid_decisions',
            'lossCountPerBatch':2 if per_choice else 54,'updates':[],'validation':[],'phase':'training'}
        report['arms'].append(arm);count=0
        batch=result=clipped=gradients=None
        try:
            if not report.get('baseline'):
                validate(head,arm,0);report['baseline']=deepcopy(arm['validation'][0])
            else:arm['validation'].append(deepcopy(report['baseline']))
            while count<args.updates:
                if expired():raise InterruptedError('Global six-minute pilot deadline')
                epoch,position=divmod(count,4)
                order=np.random.default_rng(np.random.SeedSequence([834,epoch])).permutation(4)
                seed=int(order[position]);batch=episode_batch(cache,datasets[replay][seed])
                result=cached_batch_gradients(head,batch,horizon=512,choice_only=True,cancelled=expired)
                scale=result.valid_count/result.selected_count if per_choice else 1.
                gradients=tree_map(lambda value:value*scale,result.gradients)
                if not finite_gradients(gradients):raise FloatingPointError('Nonfinite pilot gradient')
                clipped,norm=optim.clip_grad_norm(gradients,1.);mx.eval(clipped,norm)
                optimizer.update(head,clipped);mx.eval(head.parameters(),optimizer.state)
                count+=1
                arm['updates'].append({'update':count,'layoutSeed':seed,'validDecisions':result.valid_count,
                    'supervisedChoices':result.selected_count,'lossDenominator':result.selected_count if per_choice else result.valid_count,
                    'choiceNLL':float(result.loss.item())*result.valid_count/result.selected_count,'preClipNorm':float(norm.item()),
                    'postClipNorm':float(mx.sqrt(sum(mx.sum(v*v) for _,v in tree_flatten(clipped))).item()),
                    'cueSummaryGradientNorm':float(mx.sqrt(mx.sum(result.summary_gradient[:,:5]**2)).item())*scale})
                if count%64==0:
                    write();print(json.dumps({'arm':name,'updates':count,'seconds':time.perf_counter()-began,'choiceNLL':arm['updates'][-1]['choiceNLL']}),flush=True)
            validate(head,arm,count);arm['phase']='completed'
        except InterruptedError as error:arm['phase']='paused';arm['reason']=str(error)
        checkpoint=output/'checkpoints'/str(uuid.uuid4())
        save_checkpoint(checkpoint,loaded.policy,kind='behavioral',step=loaded.manifest['step']+count,parent_id=loaded.manifest['id'],
            training_state={'kind':'experimental-gru-exposure','optimizer':optimizer.state,'rng':tuple(mx.random.state),
                'additionalUpdates':count,'arm':name},metrics={'scope':report['scope'],'arm':name,'additionalUpdates':count})
        arm['checkpoint']=str(checkpoint);arm['completedUpdates']=count;write()
        del batch,result,clipped,gradients,head,loaded,optimizer
        mx.clear_cache()
    report['phase']='completed' if len(report['arms'])==3 and all(a['phase']=='completed' for a in report['arms']) else 'paused'
    write();print(json.dumps({'phase':report['phase'],'report':str(output/'report.json'),'seconds':report['wallSeconds']}),flush=True)


if __name__=='__main__':main()
