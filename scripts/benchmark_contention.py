#!/usr/bin/env python3
"""Permission-free full-GRU actor/OCR/backward contention qualification.

The learner is explicitly synthetic PPO-objective staged-backward stress, not a
completed on-policy PPO iteration. No production scheduler or model is modified.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import queue
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'scripts')]
GIB = 1024**3


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def prepare(root):
    import mlx.core as mx
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from astra.checkpoints import save_checkpoint
    from astra.environments.practice import PracticeConfig, PracticeEnvironment
    from astra.model.config import ModelConfig
    from astra.model.actions import ActionVocabulary
    from astra.model.policy import AgentPolicy
    started = time.monotonic()
    mx.set_memory_limit(3*GIB); mx.set_cache_limit(64*1024**2)
    root.mkdir(parents=True, exist_ok=True)
    env = PracticeEnvironment(PracticeConfig(pixel_width=1280, pixel_height=720))
    font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 42)
    paths=[]
    for index, seed in enumerate((807,808,809)):
        raw=env.reset(seed=seed)
        rgba=raw.pixels[..., [2,1,0,3]].copy()
        image=Image.fromarray(rgba)
        draw=ImageDraw.Draw(image)
        draw.rectangle((30,20,650,92),fill=(255,255,255,255))
        draw.text((42,28),'Score 12345',font=font,fill=(0,0,0,255))
        pixels=np.asarray(image)[..., [2,1,0,3]].copy()
        path=root/f'scene-{index}.bgra';path.write_bytes(pixels.tobytes());paths.append(str(path))
    config=ModelConfig()
    vocabulary=ActionVocabulary(key_codes=(0,13),mouse_buttons=(0,),absolute_pointer=True,relative_pointer=True,scroll=True)
    mx.random.seed(171)
    policy=AgentPolicy(config,vocabulary)
    policy.vision.backbone.load_pretrained(ROOT/'vendor/weights/convnext_tiny.safetensors')
    policy.eval();mx.eval(policy.parameters())
    destination=root/str(uuid.uuid4())
    saved=save_checkpoint(destination,policy,kind='initial',step=0,metrics={'scope':'contention_fixture_only'})
    value=dict(checkpoint=str(destination),frames=paths,model=saved['model'],actions=saved['actions'],
               parameterCount=config.parameter_count,device=mx.device_info(),setupSeconds=time.monotonic()-started)
    (root/'prepared.json').write_text(json.dumps(value,indent=2)+'\n')
    emit(value)


def learner(config_path):
    import copy
    from dataclasses import replace
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    import numpy as np
    from astra.checkpoints import load_checkpoint
    from astra.data.batching import LearningSample, training_batch
    from astra.data.observations import make_observation
    from astra.learning.backward import policy_gradients
    from astra.learning.optimizers import GroupedAdamW,finite_gradients
    from astra.learning.rl import ppo_loss,PPOConfig
    from astra.model.actions import PacketBatch
    cfg=json.loads(Path(config_path).read_text());started=time.monotonic()
    mx.set_memory_limit(12*GIB);mx.set_cache_limit(64*1024**2)
    stop_monitor=threading.Event()
    def monitor():
        while not stop_monitor.wait(.05):
            active=mx.get_active_memory()
            if active>11*GIB or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>11*GIB:
                emit(dict(kind='guard',activeMLXBytes=active,reason='11 GiB per-learner memory guard'))
                os._exit(72)
    threading.Thread(target=monitor,daemon=True,name='contention-memory-guard').start()
    loaded=load_checkpoint(Path(cfg['checkpoint']));policy=loaded.policy;policy.unfreeze();policy.eval()
    policy.configure_execution(vision_microbatch=1,checkpoint_vision=True)
    prepared=[]
    surface=dict(id='benchmark',globalBounds=dict(x=0,y=0,width=1280,height=720),pixelWidth=1280,pixelHeight=720,
                 contentBounds=dict(x=0,y=0,width=1280,height=720),geometryRevision=0)
    for index,path in enumerate(cfg['frames']):
        pixels=np.frombuffer(Path(path).read_bytes(),np.uint8).reshape(720,1280,4).copy()
        metadata=dict(id=str(uuid.uuid4()),eventNanos=1_000_000_000,observedNanos=1_000_000_000,surface=surface,
                      byteCount=pixels.nbytes,pixelFormat='bgra8-srgb',codec='raw')
        controls=dict(keys=[],buttons=[],modifiers=0,pointer=dict(x=640,y=360),observedNanos=1_000_000_000,revision=0,valid=True)
        prepared.append(make_observation([(pixels,metadata)],controls,cutoff_nanos=1_000_000_000,elapsed_seconds=.1,reset=True,config=policy.config))
    commands=({'operation':'pointerAbsolute','surfaceID':'benchmark','x':.37,'y':.61,'offsetMs':0},
              {'operation':'buttonDown','button':0,'offsetMs':20},{'operation':'buttonUp','button':0,'offsetMs':80})
    episode=str(uuid.uuid4())
    samples=[LearningSample(replace(prepared[index%3],reset=mx.array([[index==0]])),(surface,),commands,episode,index) for index in range(64)]
    observation,packets=training_batch([samples],policy.config,policy.actions.vocabulary)
    assert observation.shape==(1,64)
    if cfg.get('parityOutput'):
        observation=observation.slice_time(0,4)
        packets=PacketBatch(**{name:getattr(packets,name)[:4] for name in PacketBatch.__dataclass_fields__})
    optimizer=GroupedAdamW(learning_rate=1e-4,pretrained_learning_rate=1e-5,weight_decay=.01)
    flat=dict(tree_flatten(policy.trainable_parameters()))
    for group,instance in optimizer.optimizers.items(): instance.init({name:value for name,value in flat.items() if optimizer.group(name,value)==group})
    rollback=copy.deepcopy(policy.parameters());rollback_optimizer=copy.deepcopy(optimizer.state)
    mx.eval(policy.parameters(),optimizer.state,rollback,rollback_optimizer,observation.as_tensors(),packets.operation)
    length=observation.shape[1]
    advantages=mx.linspace(-1,1,length);returns=mx.linspace(-.5,.5,length);valid=mx.ones((length,),dtype=mx.bool_)
    def objective(logp,values,entropy):
        # Synthetic ratio-one PPO tensor stress, never asserted on-policy data.
        result=ppo_loss(logp,values,entropy,old_log_probabilities=mx.stop_gradient(logp),advantages=advantages,
                        returns=returns,valid=valid,config=PPOConfig())
        return result.total*result.valid_count,result.finite
    warm_started=time.monotonic()
    warm=policy_gradients(policy,observation,packets,objective)
    mx.eval(warm.loss,warm.gradients)
    if not finite_gradients(warm.gradients):raise FloatingPointError('Nonfinite full-GRU warmup gradients')
    gradient_branches={name:float(mx.linalg.norm(value).item()) for name,value in tree_flatten(warm.gradients)
                       if name in ('vision.backbone.downsamples.0.conv.weight','vision.detail.convolutions.0.weight','temporal.layers.0.Wx','actions.cell_query.weight')}
    if cfg.get('parityOutput'):
        output=Path(cfg['parityOutput']);output.mkdir(parents=True,exist_ok=True)
        before={name:mx.array(value) for name,value in tree_flatten(policy.parameters())}
        gradients={name:mx.array(value) for name,value in tree_flatten(warm.gradients)}
        normalized=tree_map(lambda value:value/length,warm.gradients)
        clipped,norm=optim.clip_grad_norm(normalized,.5)
        optimizer.update(policy,clipped);mx.eval(policy.parameters(),optimizer.state)
        saved={'loss':warm.loss,'gradientNorm':norm,**{'initial.'+name:value for name,value in before.items()},
               **{'gradient.'+name:value for name,value in gradients.items()},
               **{'updated.'+name:value for name,value in tree_flatten(policy.parameters())},
               **{'optimizer.'+name:value for name,value in tree_flatten(optimizer.state) if isinstance(value,mx.array)}}
        mx.eval(saved)
        mx.save_safetensors(str(output/'parity.safetensors'),saved)
        summary=dict(kind='parityComplete',shape=list(observation.shape),seconds=time.monotonic()-started,
            env={name:os.environ.get(name) for name in ('MLX_MAX_OPS_PER_BUFFER','MLX_MAX_MB_PER_BUFFER')},
            parameters=len(before),tensors=len(saved),finite=finite_gradients(saved),peakMLXBytes=mx.get_peak_memory(),
            scope='full production model B1xT4 numerical gradient/optimizer parity only')
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');emit(summary);stop_monitor.set();return
    del warm
    emit(dict(kind='ready',mode='learner' ,setupAndWarmupSeconds=time.monotonic()-started,warmBackwardSeconds=time.monotonic()-warm_started,
              scope='synthetic ratio-one PPO-objective exact staged backward; no behavior replay/GAE/KL or PPO iteration claim',
              shape=list(observation.shape),effectiveBatchDecisions=256,visionMicrobatch=1,
              inputShapes={name:list(getattr(observation.surfaces[0],name).shape) for name in ('global_image','detail_image','cursor_image')},
              gradientBranchNorms=gradient_branches,peakMLXBytes=mx.get_peak_memory()))
    window=json.loads(sys.stdin.readline());start=window['startAtNanos'];end=start+window['durationNanos']
    while time.monotonic_ns()<start:time.sleep(min(.01,(start-time.monotonic_ns())/1e9))
    accumulated=None;decisions=0;chunks=updates=0;partial=False
    mx.reset_peak_memory()
    while time.monotonic_ns()<end:
        if mx.get_active_memory()>11*GIB:raise MemoryError('Learner active memory guard')
        began=time.monotonic()
        try:result=policy_gradients(policy,observation,packets,objective,cancelled=lambda:time.monotonic_ns()>=end)
        except InterruptedError:partial=True;break
        mx.eval(result.loss,result.gradients)
        if not bool(result.auxiliary.item()) or not finite_gradients(result.gradients):raise FloatingPointError('Nonfinite exact staged gradient')
        accumulated=result.gradients if accumulated is None else tree_map(lambda a,b:a+b,accumulated,result.gradients)
        mx.eval(accumulated);decisions+=64;chunks+=1
        emit(dict(kind='chunk',chunk=chunks,seconds=time.monotonic()-began,loss=float(result.loss.item())/64,
                  activeMLXBytes=mx.get_active_memory(),peakMLXBytes=mx.get_peak_memory()))
        if decisions==256:
            began=time.monotonic()
            mean=tree_map(lambda value:value/256,accumulated)
            clipped,norm=optim.clip_grad_norm(mean,.5);optimizer.update(policy,clipped);mx.eval(policy.parameters(),optimizer.state)
            if not finite_gradients(policy.parameters()):raise FloatingPointError('Nonfinite optimizer result')
            updates+=1;accumulated=None;decisions=0
            emit(dict(kind='optimizer',update=updates,effectiveDecisions=256,seconds=time.monotonic()-began,gradientNorm=float(norm.item())))
    emit(dict(kind='complete',mode='learner',completedChunks=chunks,optimizerUpdates=updates,partialChunkDiscarded=partial,
              uncommittedDecisions=decisions,measuredEndNanos=time.monotonic_ns(),peakMLXBytes=mx.get_peak_memory(),
              activeMLXBytes=mx.get_active_memory(),peakRSSBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              scope='exact staged backward/optimizer stress only; no PPO learning claim'))
    stop_monitor.set()


class Child:
    def __init__(self,command,cwd,log,env=None):
        self.events=[];self.queue=queue.Queue();self.log=Path(log).open('w')
        self.process=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,start_new_session=True)
        def read(stream,kind):
            for line in stream:
                self.log.write(kind+line);self.log.flush()
                if kind=='':
                    try:value=json.loads(line)
                    except ValueError:continue
                    self.events.append(value);self.queue.put(value)
        self.threads=[threading.Thread(target=read,args=(self.process.stdout,''),daemon=True),threading.Thread(target=read,args=(self.process.stderr,'stderr: '),daemon=True)]
        for thread in self.threads:thread.start()
    def ready(self,timeout=180):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            try:value=self.queue.get(timeout=.2)
            except queue.Empty:
                if self.process.poll() is not None:raise RuntimeError(f'Fixture exited before readiness: {self.process.returncode}')
                continue
            if value.get('kind')=='error':raise RuntimeError(value)
            if value.get('kind')=='ready':return value
        raise TimeoutError('Bounded fixture setup/warmup timeout')
    def start(self,window):
        self.process.stdin.write(json.dumps(window)+'\n');self.process.stdin.flush()
    def join(self,timeout):
        try:self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:os.killpg(self.process.pid,signal.SIGKILL);self.process.wait();raise
        for thread in self.threads:thread.join(timeout=3)
        self.log.close()
        if self.process.returncode:raise RuntimeError(f'Fixture failed: {self.process.returncode}')
    def stop(self):
        if self.process.poll() is None:os.killpg(self.process.pid,signal.SIGTERM)
        try:self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(self.process.pid,signal.SIGKILL);self.process.wait()
        for thread in self.threads:thread.join(timeout=3)
        if not self.log.closed:self.log.close()


def distribution(samples):
    import numpy as np
    return dict(count=len(samples),medianMS=statistics.median(samples),p95MS=float(np.percentile(samples,95)),
                p99MS=float(np.percentile(samples,99)),maximumMS=max(samples))


def orchestrate(args):
    from apple_toolchain import build_environment
    began=time.monotonic();args.output.mkdir(parents=True,exist_ok=True)
    fixture=Path(tempfile.mkdtemp(prefix='AstraContention-',dir='/tmp'))
    report=dict(schemaVersion=1,scope='permission-free unchanged full-GRU contention; no screen/control',fixtureRoot=str(fixture),cases=[],completed=False)
    def save():
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    # An isolated source snapshot avoids competing with native development builds.
    source=fixture/'source';source.mkdir()
    for name in ('Sources','Tests'):shutil.copytree(ROOT/name,source/name,ignore=shutil.ignore_patterns('__pycache__'))
    package=(ROOT/'Package.swift').read_text().replace('products: [','products: [\n        .executable(name: "AstraContentionFixture", targets: ["AstraContentionFixture"]),',1)
    package=package.replace('targets: [\n','targets: [\n        .executableTarget(name: "AstraContentionFixture", dependencies: ["AstraCore", "AstraPlatform"], path: "Tests/ContentionFixture"),\n',1)
    (source/'Package.swift').write_text(package)
    native_started=time.monotonic()
    with (args.output/'build.log').open('w') as log:
        subprocess.run(['xcrun','swift','build','--product','AstraContentionFixture','--scratch-path',str(fixture/'swift-build')],cwd=source,env=build_environment(),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    report['nativeFixtureBuildSeconds']=time.monotonic()-native_started
    binary=fixture/'swift-build/debug/AstraContentionFixture'
    helper=fixture/'AstraCompute.app'
    shutil.copytree(ROOT/'build/AgentTrainer Astra.app/Contents/Helpers/AstraCompute.app',helper,symlinks=True)
    worker=helper/'Contents/MacOS/AstraCompute'
    report['helperBinarySHA256']=hashlib.sha256(worker.read_bytes()).hexdigest()
    prepared_path=fixture/'data'
    with (args.output/'prepare.log').open('w') as log:
        subprocess.run([sys.executable,str(Path(__file__)),'--worker','prepare','--worker-config',str(prepared_path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    prepared=json.loads((prepared_path/'prepared.json').read_text());report['prepared']=prepared
    report['initialSetupSeconds']=time.monotonic()-began;save()
    measured=0
    cases=[('actor-only',20,False,False),('actor-ocr',20,True,False),('actor-learner',60,False,True),('all-three',60,True,True)]
    for name,duration,use_ocr,use_learner in cases:
        children=[];case=dict(name=name,configuredSeconds=duration,setup=[])
        case_root=fixture/name;case_root.mkdir()
        config=dict(worker=str(worker),checkpoint=prepared['checkpoint'],frames=prepared['frames'],ring=str(case_root/'frames.astraring'),width=1280,height=720)
        try:
            # Prepare contenders sequentially while every already-ready owner is
            # blocked on stdin; setup is measured separately from contention.
            for mode in ['actor']+(['ocr'] if use_ocr else [])+(['learner'] if use_learner else []):
                path=case_root/(mode+'.json');path.write_text(json.dumps(dict(config,mode=mode)))
                command=([sys.executable,str(Path(__file__)),'--worker','learner','--worker-config',str(path)] if mode=='learner' else [str(binary),str(path)])
                child=Child(command,ROOT,args.output/(name+'-'+mode+'.log'));children.append(child)
                ready=child.ready();case['setup'].append(ready)
                emit(dict(phase='ready',case=name,mode=mode,setupSeconds=ready.get('setupAndWarmupSeconds')))
            remaining=180-measured
            actual_duration=min(duration,remaining-2)
            if actual_duration<=0:raise RuntimeError('Measured contention budget exhausted')
            start=time.monotonic_ns()+1_000_000_000
            window=dict(startAtNanos=start,durationNanos=int(actual_duration*1e9));case['window']=window
            for child in children:child.start(window)
            emit(dict(phase='measuring',case=name,seconds=actual_duration))
            for child in children:child.join(actual_duration+12)
            actor_events=children[0].events
            samples=[value for value in actor_events if value.get('kind')=='sample']
            complete=next(value for value in reversed(actor_events) if value.get('kind')=='complete')
            ends=[value['measuredEndNanos'] for child in children for value in child.events if value.get('kind')=='complete']
            seconds=(max(ends)-start)/1e9;measured+=seconds
            case.update(measuredSeconds=seconds,actorLatency=distribution([value['latencyMS'] for value in samples]),
                        leadMisses=sum(value['missedLead'] for value in samples),cadenceSlotsMissed=complete['missedCadenceSlots'],
                        ringPublicationLatency=distribution([value['ringPublicationMS'] for value in samples]),
                        roundTripLatency=distribution([value['roundTripMS'] for value in samples]),
                        results=[value for child in children for value in child.events if value.get('kind')=='complete'],
                        warmSamples=samples,workerEvents={str(index):[value for value in child.events if value.get('kind') in ('chunk','optimizer','ocr')] for index,child in enumerate(children)})
            report['cases'].append(case);report['totalMeasuredSeconds']=measured;save()
            emit(dict(phase='caseComplete',case=name,actorLatency=case['actorLatency'],leadMisses=case['leadMisses'],measuredSeconds=measured))
        except BaseException as error:
            if 'window' in case: case['failedMeasuredSeconds']=max(0,(time.monotonic_ns()-case['window']['startAtNanos'])/1e9)
            case['failure']=f'{type(error).__name__}: {error}'
            case['partialEvents']=[child.events for child in children]
            report['cases'].append(case);report['failed']=True;save()
            raise
        finally:
            for child in children:child.stop()
    report['completed']=True;report['totalWallSeconds']=time.monotonic()-began;save();emit(dict(phase='complete',report=str(args.output/'report.json'),measuredSeconds=measured))


def recovery(report_path):
    from apple_toolchain import build_environment
    report_path=Path(report_path);report=json.loads(report_path.read_text());output=report_path.parent
    fixture=Path(report['fixtureRoot']);source=fixture/'source';worker=fixture/'AstraCompute.app/Contents/MacOS/AstraCompute'
    if hashlib.sha256(worker.read_bytes()).hexdigest()!=report['helperBinarySHA256']:raise ValueError('Immutable actor snapshot changed')
    shutil.copyfile(ROOT/'Tests/ContentionFixture/main.swift',source/'Tests/ContentionFixture/main.swift')
    began=time.monotonic()
    with (output/'recovery-build.log').open('w') as log:
        subprocess.run(['xcrun','swift','build','--product','AstraContentionFixture','--scratch-path',str(fixture/'swift-build')],cwd=source,env=build_environment(),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    cfg=dict(mode='actor',worker=str(worker),checkpoint=report['prepared']['checkpoint'],frames=report['prepared']['frames'],
             ring=str(fixture/('recovery-'+str(uuid.uuid4())+'.astraring')),width=1280,height=720,relativePacing=True)
    path=fixture/'recovery.json';path.write_text(json.dumps(cfg))
    child=Child([str(fixture/'swift-build/debug/AstraContentionFixture'),str(path)],ROOT,output/'recovery-actor.log')
    try:
        ready=child.ready()
        emit(dict(phase='recoveryReady',setupSeconds=time.monotonic()-began))
        duration=min(15,178-report['totalMeasuredSeconds'])
        window=dict(startAtNanos=time.monotonic_ns()+1_000_000_000,durationNanos=int(duration*1e9));child.start(window)
        emit(dict(phase='measuringRecovery',seconds=duration))
        child.join(duration+12)
        samples=[row for row in child.events if row.get('kind')=='sample']
        done=next(row for row in reversed(child.events) if row.get('kind')=='complete')
        elapsed=(done['measuredEndNanos']-window['startAtNanos'])/1e9
        report['recovery']=dict(scope='actor-only; production relative-cutoff pacing; separately reported from nominal-grid cases',
            setup=ready,window=window,measuredSeconds=elapsed,actorLatency=distribution([row['latencyMS'] for row in samples]),
            leadMisses=sum(row['missedLead'] for row in samples),samples=samples,result=done)
        report['totalMeasuredSeconds']+=elapsed
        report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        emit(dict(phase='recoveryComplete',actorLatency=report['recovery']['actorLatency'],leadMisses=report['recovery']['leadMisses'],totalMeasuredSeconds=report['totalMeasuredSeconds']))
    finally:child.stop()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'.local/contention-benchmark')
    parser.add_argument('--worker',choices=('prepare','learner'))
    parser.add_argument('--worker-config',type=Path)
    parser.add_argument('--recovery-of',type=Path)
    args=parser.parse_args()
    if args.recovery_of:recovery(args.recovery_of)
    elif args.worker=='prepare':prepare(args.worker_config)
    elif args.worker=='learner':learner(args.worker_config)
    else:orchestrate(args)


if __name__=='__main__':main()
