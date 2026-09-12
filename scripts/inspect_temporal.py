#!/usr/bin/env python3
"""Training-choice fit and frozen cue-information diagnostics, not agent success."""
from pathlib import Path
import argparse
from copy import deepcopy
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--common-checkpoint',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--visual-root',type=Path,help='Shared immutable visual cache used by this dataset')
    parser.add_argument('--phase-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=360)
    args=parser.parse_args()
    if not 1<=args.seconds<=360: parser.error('Diagnostic processes are bounded to six minutes')
    import mlx.core as mx
    import numpy as np
    from astra.checkpoints import load_checkpoint
    from astra.environments.practice import PracticeConfig,PracticeEnvironment
    from astra.data.actions import encode_commands
    from astra.model.actions import PacketBatch
    from experiments.temporal_study import FrozenHead,VisualCache,episode_batch,_file_hash
    from scripts.qualify_memory import memory_phase_fixture
    began=time.perf_counter();mx.set_memory_limit(10*1024**3);mx.set_cache_limit(64*1024**2)
    loaded=load_checkpoint(args.checkpoint,include_training=True)
    manifest=json.loads((args.root/'dataset.json').read_text())
    common=load_checkpoint(args.common_checkpoint)
    cache=VisualCache(args.visual_root or args.root/'visual',common.policy,expected_digest=manifest['visualDigest'])
    VisualCache(args.visual_root or args.root/'visual',loaded.policy,expected_digest=manifest['visualDigest'])
    head=FrozenHead(loaded.policy);head.eval()
    source=[]
    for entry in manifest['episodes']:
        if _file_hash(entry['path'])!=entry['sha256']: raise ValueError('Source episode changed')
        source.append(json.loads(Path(entry['path']).read_text()))
    rows=[];training_features=[];color_labels=[]
    for first in range(0,len(source),2):
        if time.perf_counter()-began>args.seconds: raise InterruptedError('Diagnostic deadline')
        episodes=source[first:first+2]
        batch=episode_batch(cache,episodes)
        temporal=head.temporal(batch.summary,batch.observation);mx.eval(temporal.context)
        length=batch.observation.shape[1]
        indices=[index for index,value in enumerate(np.asarray(batch.packets.operation[:,0])) if value!=0]
        if len(indices)!=2: raise ValueError('Expected one choice packet per complete oracle episode')
        visual=batch.visual_at(indices)
        correct=PacketBatch(**{name:getattr(batch.packets,name)[mx.array(indices)] for name in PacketBatch.__dataclass_fields__})
        alternatives=[]
        for lane,episode in enumerate(episodes):
            world=PracticeEnvironment(PracticeConfig.from_dict(episode['environment']));world.reset(seed=episode['seed'])
            if episode.get('counterfactual',False): world._answer=1-world._answer
            selected=next(item for item in episode['steps'] if item['commands'])
            original=selected['commands'];point=original[0]
            if point['operation']!='pointerAbsolute': raise ValueError('Unexpected oracle packet')
            # Privileged fixture positions construct evaluation labels only.
            # They are never appended to observations or temporal inputs.
            x,y,w,h=world.config.logical_bounds
            target=world._choices[world._choice_colors.index(world._answer)]
            np.testing.assert_allclose([point['x'],point['y']],[(target[0]-x)/w,(target[1]-y)/h],rtol=0,atol=1e-12)
            other=world._choices[world._choice_colors.index(1-world._answer)]
            wrong=deepcopy(original);wrong[0].update(x=(other[0]-x)/w,y=(other[1]-y)/h)
            alternatives.append(encode_commands(wrong,config=head.config,vocabulary=head.actions.vocabulary,
                                                 visual=visual,surfaces=[selected['surface']],batch_index=lane))
            training_features.append(np.asarray(cache.get(episode['steps'][0]['visual']).summary[0]).copy())
            color_labels.append(world._answer)
        wrong=PacketBatch(**{name:mx.concatenate([getattr(packet,name) for packet in alternatives]) for name in PacketBatch.__dataclass_fields__})
        context=temporal.context.reshape(-1,head.config.recurrent_width)[mx.array(indices)]
        good=head.actions.log_prob(context,visual,correct).log_probability
        bad=head.actions.log_prob(context,visual,wrong).log_probability;mx.eval(good,bad)
        for lane,episode in enumerate(episodes):
            margin=float((good-bad)[lane].item())
            rows.append(dict(seed=episode['seed'],delayMS=episode['delayMS'],margin=margin,correct=margin>0,tie=margin==0))
        if len(rows)%16==0: print(json.dumps(dict(phase='trainingChoices',episodes=len(rows),elapsedSeconds=time.perf_counter()-began)),flush=True)
    # A fixed ridge linear readout tests availability of cue color in the common
    # frozen feature map. This is not a temporal model or an action policy.
    x=np.asarray(training_features,np.float64);labels=2*np.asarray(color_labels,np.float64)-1
    mean=x.mean(axis=0);label_mean=labels.mean();centered=x-mean
    dual=np.linalg.solve(centered@centered.T+1e-2*np.eye(len(x)),labels-label_mean)
    weight=centered.T@dual
    predictions=[]
    environment=PracticeConfig(task='delayed_memory',delay_ms=30000,time_limit_ms=32500)
    for seed in range(1000,1032):
        if time.perf_counter()-began>args.seconds: raise InterruptedError('Diagnostic deadline')
        phases,_=memory_phase_fixture(environment,seed,args.phase_cache)
        world=PracticeEnvironment(environment);world.reset(seed=seed)
        for member in (0,1):
            key,_=cache.observe(phases[member],first=True)
            feature=np.asarray(cache.get(key).summary[0],np.float64)
            score=float((feature-mean)@weight+label_mean)
            target=world._answer if member==0 else 1-world._answer
            predictions.append(dict(seed=seed,cueMember=member,correct=int(score>0)==target,score=score))
    report=dict(schemaVersion=1,checkpoint=str(args.checkpoint),step=loaded.manifest['step'],
                scope='training_forced_choice_and_frozen_cue_information_diagnostics',
                trainingChoices=rows,trainingByDelay=[dict(delayMS=delay,episodes=sum(row['delayMS']==delay for row in rows),
                    accuracy=sum(row['correct']+.5*row['tie'] for row in rows if row['delayMS']==delay)/sum(row['delayMS']==delay for row in rows))
                    for delay in (2000,8000,30000)],
                cueReadout=dict(type='fixed_ridge_linear',ridge=.01,input='common_frozen_visual_summary_only',
                    trainingEpisodes=len(x),heldoutCounterfactualCues=len(predictions),
                    accuracy=sum(row['correct'] for row in predictions)/len(predictions),predictions=predictions),
                wallSeconds=time.perf_counter()-began,peakMLXBytes=mx.get_peak_memory())
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(trainingByDelay=report['trainingByDelay'],cueReadoutAccuracy=report['cueReadout']['accuracy'],seconds=report['wallSeconds'])),flush=True)

if __name__=='__main__':main()
