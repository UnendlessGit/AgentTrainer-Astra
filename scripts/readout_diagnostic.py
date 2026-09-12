#!/usr/bin/env python3
"""Inspect temporal-to-packet dependence on one real immediate-cue pair."""
from pathlib import Path
import argparse,json,sys,time,uuid
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'python')]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--common-checkpoint',type=Path,default=ROOT/'.local/verification/3ea5799a-2b26-4e7a-912a-ad32412ad171')
    parser.add_argument('--visual-root',type=Path,default=ROOT/'.local/temporal-study/visual')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--overfit-updates',type=int,default=0)
    parser.add_argument('--seconds',type=float,default=180)
    parser.add_argument('--greedy-ready',action='store_true',help='Execute a greedy packet only at readiness in each virtual training world')
    args=parser.parse_args()
    if not 0<=args.overfit_updates<=1024 or not 1<=args.seconds<=180: parser.error('Tiny diagnostics are bounded to 1024 updates and 180 seconds')
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from astra.checkpoints import load_checkpoint,save_checkpoint
    from astra.environments.practice import PracticeConfig
    from astra.model.actions import PacketBatch
    from astra.data.actions import decode_commands
    from astra.learning.optimizers import GroupedAdamW,finite_gradients
    from experiments.temporal_study import FrozenHead,VisualCache,ImmediateCueEnvironment,prepare_episode,episode_batch,cached_batch_gradients
    started=time.perf_counter();mx.set_memory_limit(10*1024**3);mx.set_cache_limit(64*1024**2)
    loaded=load_checkpoint(args.checkpoint)
    common=load_checkpoint(args.common_checkpoint)
    cache=VisualCache(args.visual_root,common.policy)
    if loaded.policy.config!=common.policy.config: raise ValueError('The diagnostic and common visual configurations differ')
    # Fail closed instead of accidentally feeding another checkpoint's visual map.
    VisualCache(args.visual_root,loaded.policy,expected_digest=cache.digest)
    head=FrozenHead(loaded.policy);head.eval()
    environment=PracticeConfig(task='delayed_memory',delay_ms=2000,time_limit_ms=4500)
    pair=[prepare_episode(cache,environment,args.seed,counterfactual=value,cue_replay=True) for value in (False,True)]
    batch=episode_batch(cache,pair)
    choice_step=(environment.cue_ms+environment.delay_ms)//environment.period_ms
    flat_indices=[choice_step,batch.summary.shape[1]+choice_step]
    indices=mx.array(flat_indices);visual=batch.visual_at(flat_indices)
    correct=PacketBatch(**{name:getattr(batch.packets,name)[indices] for name in PacketBatch.__dataclass_fields__})
    wrong=PacketBatch(**{name:getattr(correct,name)[mx.array([1,0])] for name in PacketBatch.__dataclass_fields__})
    def scores(context):
        good=head.actions.log_prob(context,visual,correct)
        bad=head.actions.log_prob(context,visual,wrong)
        return good,bad
    def inspect():
        temporal=head.temporal(batch.summary,batch.observation)
        context=temporal.context.reshape(-1,head.config.recurrent_width)[indices];mx.eval(context)
        good,bad=scores(context);mx.eval(good.log_probability,bad.log_probability)
        def difference(value):
            a,b=scores(value)
            return mx.sum(a.log_probability-b.log_probability)
        gradient=mx.grad(difference)(context);mx.eval(gradient)
        factors={}
        for name in good.factor_log_probabilities:
            a=mx.sum(good.factor_log_probabilities[name],axis=1)
            b=mx.sum(bad.factor_log_probabilities[name],axis=1)
            mx.eval(a,b);factors[name]={'correct':np.asarray(a).tolist(),'wrong':np.asarray(b).tolist()}
        return {'margins':np.asarray(good.log_probability-bad.log_probability).tolist(),
            'choiceNLL':float(-mx.mean(good.log_probability).item()),
            'contextRMSDifference':float(mx.sqrt(mx.mean((context[0]-context[1])**2)).item()),
            'choiceDifferenceContextGradientNorm':float(mx.sqrt(mx.sum(gradient**2)).item()),
            'cueDirectionDerivative':float(mx.sum(gradient[0]*(context[0]-context[1])).item()),
            'factorScores':factors}
    report={'schemaVersion':1,'scope':'one_layout_immediate_cue_connection_diagnostic','seed':args.seed,
            'checkpoint':str(args.checkpoint),'commonCheckpoint':str(args.common_checkpoint),
            'visualDigest':cache.digest,'lossDenominator':int(mx.sum(batch.observation.valid).item()),
            'optimizer':'fresh AdamW; lr=3e-4, weight_decay=.01, clip_norm=1',
            'before':inspect(),'updates':[]}
    gradient=cached_batch_gradients(head,batch,horizon=512,choice_only=True)
    report['before']['earlyCueGradientNorm']=float(mx.sqrt(mx.sum(gradient.summary_gradient[:,:5]**2)).item())
    report['before']['replayedCueGradientNorm']=float(mx.sqrt(mx.sum(gradient.summary_gradient[:,24:25]**2)).item())
    print(json.dumps({'before':report['before']}),flush=True)
    optimizer=GroupedAdamW(learning_rate=3e-4,pretrained_learning_rate=3e-5,weight_decay=.01)
    head.train();completed=0
    for update in range(args.overfit_updates):
        if time.perf_counter()-started>=args.seconds: break
        try:
            result=cached_batch_gradients(head,batch,horizon=512,choice_only=True,cancelled=lambda:time.perf_counter()-started>=args.seconds)
        except InterruptedError:
            break  # The helper restores parameter bindings; no optimizer mutation occurred.
        if not bool(mx.isfinite(result.loss).item()) or not finite_gradients(result.gradients): raise FloatingPointError('Nonfinite tiny-pair loss/gradient')
        clipped,norm=optim.clip_grad_norm(result.gradients,1.);optimizer.update(head,clipped);mx.eval(head.parameters(),optimizer.state)
        completed+=1
        if completed%16==0:
            head.eval();measurement=inspect();head.train()
            report['updates'].append({'update':completed,'choiceNLL':measurement['choiceNLL'],'margins':measurement['margins']})
            print(json.dumps(report['updates'][-1]),flush=True)
    head.eval();report['after']=inspect();report['completedUpdates']=completed
    report['updateTargetReached']=completed==args.overfit_updates
    if args.greedy_ready:
        context=head.temporal(batch.summary,batch.observation).context.reshape(-1,head.config.recurrent_width)[indices]
        sampled=head.actions.sample(context,visual,key=mx.random.key(0),greedy=True)
        mx.eval(sampled.packets.operation)
        results=[]
        for member in range(2):
            world=ImmediateCueEnvironment(environment);world.reset(args.seed)
            if member: world._answer=1-world._answer
            while world.elapsed_ms<choice_step*environment.period_ms: world.step([],episode_id=world.episode_id)
            commands=decode_commands(sampled.packets,config=head.config,vocabulary=head.actions.vocabulary,
                visual=visual,surfaces=[environment.surface()],batch_index=member)
            transition=world.step(commands,episode_id=world.episode_id)
            reward=transition.reward
            while transition.outcome=='continuing':
                transition=world.step([],episode_id=world.episode_id);reward+=transition.reward
            results.append(dict(cueMember=member,commands=commands,outcome=transition.outcome,reward=reward))
        report['greedyReady']={'scope':'training_pair_only_with_forced_empty_actions_before_readiness','results':results}
    report['wallSeconds']=time.perf_counter()-started
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if completed:
        destination=args.output.parent/str(uuid.uuid4())
        save_checkpoint(destination,loaded.policy,kind='behavioral',step=loaded.manifest['step']+completed,parent_id=loaded.manifest['id'],
            training_state={'kind':'experimental-readout','optimizer':optimizer.state,'additionalUpdates':completed},
            metrics={'scope':'tiny_immediate_cue_overfit','additionalUpdates':completed,'seed':args.seed})
        report['resultCheckpoint']=str(destination)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'completedUpdates':completed,'after':report['after'],'seconds':report['wallSeconds']}),flush=True)

if __name__=='__main__':main()
