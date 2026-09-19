#!/usr/bin/env python3
"""Learner-only MLX command-buffer heuristic comparison; no production edits."""
from pathlib import Path
import argparse
import hashlib
from importlib.metadata import version
import json
import os
import struct
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
from benchmark_contention import Child,distribution


def emit(value):print(json.dumps(value,allow_nan=False),flush=True)


def profile_environment(value):
    environment=dict(os.environ)
    if environment.get('MLX_METAL_FAST_SYNCH') not in (None,'','0') or environment.get('MLX_METAL_GPU_ARCH'):
        raise ValueError('This test forbids fast-synch and architecture overrides')
    environment['MLX_MAX_OPS_PER_BUFFER']=str(value)
    environment['MLX_MAX_MB_PER_BUFFER']=str(value)
    return environment


def tensors(path):
    import numpy as np
    with Path(path).open('rb') as file:
        size=struct.unpack('<Q',file.read(8))[0];header=json.loads(file.read(size))
    types={'F32':np.float32,'F64':np.float64,'U32':np.uint32,'U64':np.uint64,'I32':np.int32,'I64':np.int64,'BOOL':np.bool_}
    return {name:np.memmap(path,mode='r',dtype=types[value['dtype']],shape=tuple(value['shape']),offset=8+size+value['data_offsets'][0])
            for name,value in header.items() if name!='__metadata__'}


def compare(left,right):
    import numpy as np
    a,b=tensors(left),tensors(right)
    if a.keys()!=b.keys():raise ValueError('Parity tensor structure changed')
    errors={};exact={}
    for name in a:
        if a[name].shape!=b[name].shape or a[name].dtype!=b[name].dtype:raise ValueError('Parity dtype/shape changed: '+name)
        category=name.split('.')[0]
        if name.startswith('initial.'):
            np.testing.assert_array_equal(a[name],b[name],err_msg=name)
        else:
            tolerance=2e-5 if category=='gradient' else 1e-6
            relative=2e-4 if category=='gradient' else 1e-5
            np.testing.assert_allclose(a[name],b[name],rtol=relative,atol=tolerance,err_msg=name)
        difference=float(np.max(np.abs(a[name].astype(np.float64)-b[name].astype(np.float64)))) if a[name].size else 0.
        errors[category]=max(errors.get(category,0),difference)
        exact[category]=exact.get(category,True) and np.array_equal(a[name],b[name])
    return dict(tensors=len(a),maximumAbsoluteErrors=errors,bitwiseEqual=exact,passed=True,
                scope='full production GRU B1xT4 numerical gradient and one clipped AdamW update',
                gradientTolerance={'atol':2e-5,'rtol':2e-4,'reference':'python/tests/test_staged_backward.py:106'},
                otherTolerance={'atol':1e-6,'rtol':1e-5})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-report',type=Path,default=ROOT/'.local/contention-benchmark-qualified/report.json')
    parser.add_argument('--output',type=Path,default=ROOT/'.local/command-buffer-benchmark')
    parser.add_argument('--reuse-parity',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if version('mlx')!='0.32.2' or version('mlx-metal')!='0.32.2':raise ValueError('This benchmark is pinned to MLX0.32.2')
    source=json.loads(args.source_report.read_text());fixture=Path(source['fixtureRoot'])
    helper=fixture/'AstraCompute.app/Contents/MacOS/AstraCompute';binary=fixture/'swift-build/debug/AstraContentionFixture'
    if hashlib.sha256(helper.read_bytes()).hexdigest()!=source['helperBinarySHA256']:raise ValueError('Frozen helper snapshot changed')
    prepared=source['prepared'];root=fixture/('buffers-'+str(uuid.uuid4()));root.mkdir()
    common=dict(worker=str(helper),checkpoint=prepared['checkpoint'],frames=prepared['frames'],width=1280,height=720,relativePacing=True)
    report=dict(schemaVersion=1,mlxVersion=version('mlx'),helperBinarySHA256=source['helperBinarySHA256'],model=prepared['model'],
        fixtureRoot=str(fixture),scope='learner command-buffer heuristics; actual-cutoff actor pacing; synthetic exact backward stress',
        profiles=[],completed=False)
    def save():
        target=args.output/'report.json';temporary=target.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temporary.replace(target)
    # Environment is installed before each worker imports MLX, whose getter is
    # cached on first use. The actor environment is not modified.
    if not args.reuse_parity:
        for value in (50,10):
            path=root/f'parity-{value}.json';destination=args.output/f'parity-{value}'
            path.write_text(json.dumps(dict(common,parityOutput=str(destination.resolve()))))
            began=time.monotonic()
            with (args.output/f'parity-{value}.log').open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/benchmark_contention.py'),'--worker','learner','--worker-config',str(path)],
                    cwd=ROOT,env=profile_environment(value),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
            emit(dict(phase='parityPrepared',profile=value,seconds=time.monotonic()-began))
    report['parity']=compare(args.output/'parity-50/parity.safetensors',args.output/'parity-10/parity.safetensors')
    evidence=args.output/'parity-repeat-control.json'
    if evidence.exists(): report['parityRepeatControl']=json.loads(evidence.read_text())
    failed=args.output/'parity-initial-failure.json'
    if failed.exists(): report['initialStrictParityFailure']=json.loads(failed.read_text())
    save()
    emit(dict(phase='parityPassed',result=report['parity']))
    for value in (50,10):
        case=dict(learnerMaxOps=value,learnerDataSizeThreshold=value,actorEnvironment='unchanged',setup=[])
        children=[];case_root=root/str(value);case_root.mkdir()
        try:
            for mode in ('actor','learner'):
                config=case_root/(mode+'.json');config.write_text(json.dumps(dict(common,mode=mode,ring=str(case_root/'frames.astraring'))))
                command=[str(binary),str(config)] if mode=='actor' else [sys.executable,str(ROOT/'scripts/benchmark_contention.py'),'--worker','learner','--worker-config',str(config)]
                child=Child(command,ROOT,args.output/f'{value}-{mode}.log',env=None if mode=='actor' else profile_environment(value))
                children.append(child);case['setup'].append(child.ready())
            gate=args.output/f'{value}.go'
            if gate.exists():gate.unlink()
            emit(dict(phase='windowReady',profile=value,gate=str(gate)))
            deadline=time.monotonic()+180
            while not gate.exists():
                if time.monotonic()>deadline:raise TimeoutError('Coordinator did not release the measurement gate')
                time.sleep(.05)
            start=time.monotonic_ns()+1_000_000_000;window=dict(startAtNanos=start,durationNanos=60_000_000_000)
            case['window']=window
            for child in children:child.start(window)
            emit(dict(phase='measuring',profile=value,seconds=60))
            for child in children:child.join(72)
            samples=[row for row in children[0].events if row.get('kind')=='sample']
            results=[row for child in children for row in child.events if row.get('kind')=='complete']
            case.update(actorLatency=distribution([row['latencyMS'] for row in samples]),leadMisses=sum(row['missedLead'] for row in samples),
                samples=samples,results=results,learnerChunks=[row for row in children[1].events if row.get('kind')=='chunk'],
                measuredSeconds=(max(row['measuredEndNanos'] for row in results)-start)/1e9)
            report['profiles'].append(case);save()
            emit(dict(phase='profileComplete',profile=value,actorLatency=case['actorLatency'],leadMisses=case['leadMisses']))
        except BaseException as error:
            case['failure']=f'{type(error).__name__}: {error}';case['partialEvents']=[child.events for child in children]
            report['profiles'].append(case);save();raise
        finally:
            for child in children:child.stop()
    report['completed']=True;report['totalMeasuredSeconds']=sum(row['measuredSeconds'] for row in report['profiles']);save()
    emit(dict(phase='complete',report=str(args.output/'report.json')))


if __name__=='__main__':main()
