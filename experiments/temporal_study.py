"""Frozen-visual temporal diagnostics; no production schema or default changes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map

from astra.learning.backward import _bound, _sum_gradients
from astra.model.actions import PacketBatch
from astra.model.observation import ObservationBatch
from astra.model.vision import VisualFeatures


class FrozenHead(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.config = policy.config
        self.temporal, self.actions = policy.temporal, policy.actions


@dataclass(frozen=True)
class CachedBatch:
    observation: ObservationBatch
    summary: mx.array
    packets: PacketBatch
    visual_at: Callable[[list[int]], VisualFeatures]


@dataclass(frozen=True)
class CachedGradient:
    loss: mx.array
    gradients: dict
    next_state: tuple[mx.array, ...]
    summary_gradient: mx.array
    log_probabilities: mx.array
    selected_count: int
    valid_count: int


def cached_batch_gradients(head, batch: CachedBatch, *, horizon: int, choice_only: bool,
                           action_microbatch: int = 2, cancelled=lambda: False) -> CachedGradient:
    """One two-episode optimizer batch, with explicit temporal gradient cuts.

    Both loss modes divide by the TOTAL valid decisions. Choice-only masks
    waiting labels but does not silently amplify the selected gradients. Visual
    outputs are frozen; only temporal/action parameters receive an update.
    """
    if type(horizon) is not int or not 1 <= horizon <= 512 or type(choice_only) is not bool:
        raise ValueError('Invalid diagnostic horizon/loss')
    if type(action_microbatch) is not int or not 1 <= action_microbatch <= 64:
        raise ValueError('Invalid diagnostic action microbatch')
    observation, summary, packets = batch.observation, batch.summary, batch.packets
    lanes, length = observation.shape
    if summary.shape != (lanes, length, head.config.query_count * head.config.spatial_width):
        raise ValueError('Cached summaries do not match their observation/model')
    if packets.operation.shape != (lanes * length, head.config.packet_capacity + 1):
        raise ValueError('Cached labels do not match their observation')
    validity = np.asarray(observation.valid).reshape(-1)
    if validity.dtype != np.bool_ or not validity.any(): raise ValueError('Diagnostic batch has no valid decisions')
    selected = validity & (np.asarray(packets.operation[:, 0]) != 0 if choice_only else True)
    valid_count, selected_count = int(validity.sum()), int(selected.sum())
    if not selected_count: raise ValueError('The complete episode batch has no selected supervision')
    gradients = None; state = None; loss = mx.array(0., dtype=mx.float32)
    all_logp = mx.zeros((lanes * length,), dtype=mx.float32)
    summary_gradient = mx.zeros_like(summary)
    def check():
        if cancelled(): raise InterruptedError('Diagnostic cancelled before its complete optimizer boundary')
    for start in range(0, length, horizon):
        check(); end = min(start + horizon, length); span = end - start
        local_observation = observation.slice_time(start, end)
        local_summary = summary[:, start:end]
        temporal = head.temporal(local_summary, local_observation, state)
        mx.eval(temporal.context, temporal.state)
        context = mx.stop_gradient(temporal.context.reshape(lanes * span, -1))
        global_indices = [lane * length + step for lane in range(lanes) for step in range(start, end)]
        selected_local = [index for index, global_index in enumerate(global_indices) if selected[global_index]]
        dcontext = mx.zeros_like(context); action_gradients = None
        for first in range(0, len(selected_local), action_microbatch):
            check()
            local = selected_local[first:first + action_microbatch]
            global_selected = [global_indices[index] for index in local]
            local_indices, global_array = mx.array(local), mx.array(global_selected)
            visual = batch.visual_at(global_selected)
            labels = PacketBatch(**{name: getattr(packets, name)[global_array] for name in PacketBatch.__dataclass_fields__})
            def action_loss(parameters, value):
                def evaluate():
                    score = head.actions.log_prob(value, visual, labels).log_probability
                    return -mx.sum(score), score
                return _bound(head.actions, parameters, evaluate)
            (local_loss, logp), (action_gradient, context_gradient) = mx.value_and_grad(action_loss, argnums=(0, 1))(
                head.actions.trainable_parameters(), context[local_indices])
            mx.eval(local_loss, logp, action_gradient, context_gradient)
            loss = loss + local_loss
            all_logp = all_logp.at[global_array].add(logp)
            dcontext = dcontext.at[local_indices].add(mx.stop_gradient(context_gradient))
            mx.eval(loss, all_logp, dcontext)
            action_gradients = _sum_gradients(action_gradients, action_gradient)
        if selected_local:
            check()
            def temporal_loss(parameters, value):
                def evaluate():
                    result = head.temporal(value, local_observation, state)
                    return mx.sum(result.context * mx.stop_gradient(dcontext.reshape(result.context.shape)))
                return _bound(head.temporal, parameters, evaluate)
            _, (temporal_gradients, dsummaries) = mx.value_and_grad(temporal_loss, argnums=(0, 1))(
                head.temporal.trainable_parameters(), local_summary)
            mx.eval(temporal_gradients, dsummaries)
            gradients = _sum_gradients(gradients, dict(temporal=temporal_gradients, actions=action_gradients))
            summary_gradient = summary_gradient.at[:, start:end].add(dsummaries)
            mx.eval(summary_gradient)
        state = tuple(mx.stop_gradient(value) for value in temporal.state)
        mx.eval(state)
    check()
    gradients = tree_map(lambda value: value / valid_count, gradients)
    result = CachedGradient(loss / valid_count, gradients, state, summary_gradient / valid_count,
                            all_logp, selected_count, valid_count)
    mx.eval(result.loss, result.gradients, result.summary_gradient, result.log_probabilities)
    return result

# Files below belong to the experiment, never the live library catalog.
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import uuid
from mlx.utils import tree_flatten

from astra.data.actions import encode_commands
from astra.data.observations import make_observation
from astra.model.actions import flatten_visual


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while block := file.read(1024 * 1024): digest.update(block)
    return digest.hexdigest()


def _publish_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.'+str(uuid.uuid4())+'.tmp')
    temporary.write_bytes(_json(value)); temporary.replace(path)


class VisualCache:
    """Checksummed full-resolution visual features, with bounded MLX residency."""
    maximum_resident = 8

    def __init__(self, directory, policy, *, expected_digest=None):
        self.directory = Path(directory); self.policy = policy; self.config = policy.config
        digest = hashlib.sha256()
        for name, value in sorted(tree_flatten(policy.vision.parameters())):
            digest.update(name.encode()); digest.update(_json([list(value.shape), str(value.dtype)]))
            digest.update(np.asarray(value).tobytes())
        self.digest = digest.hexdigest()
        if expected_digest is not None and self.digest != expected_digest:
            raise ValueError('The visual cache belongs to different frozen weights')
        self.root = self.directory / self.digest
        self.root.mkdir(parents=True, exist_ok=True)
        self._resident = OrderedDict(); self._templates = OrderedDict()
        policy.vision.freeze(); policy.eval()

    def key(self, raw):
        digest = hashlib.sha256(raw.pixels.tobytes())
        digest.update(_json(dict(surface=raw.metadata['surface'], pointer=raw.control_state['pointer'],
                                controlValid=raw.control_state['valid'], pixelFormat=raw.metadata['pixelFormat'],
                                model=self.config.to_dict(), visualDigest=self.digest)))
        return digest.hexdigest()

    def observe(self, raw, *, first, events=(), last_input=None):
        key = self.key(raw)
        template = self._templates.pop(key, None)
        arguments = dict(cutoff_nanos=raw.metadata['observedNanos'], elapsed_seconds=self.config.period_ms/1000,
                         reset=first, config=self.config, executed_events=events, last_input_nanos=last_input)
        if template is None:
            observation = make_observation([(raw.pixels, raw.metadata)], raw.control_state, **arguments)
            template = observation.surfaces[0]
        else:
            observation = make_observation([(raw.pixels, raw.metadata)], raw.control_state,
                surface_preparer=lambda *args, **kwargs: SimpleNamespace(batch=lambda: template), **arguments)
        self._templates[key] = template
        while len(self._templates) > self.maximum_resident: self._templates.popitem(last=False)
        metadata_path = self.root / (key+'.json')
        if not metadata_path.exists():
            features = flatten_visual(self.policy.encode_visual(observation))
            tensors = dict(vars(features)); mx.eval(tensors)
            path = self.root / (key+'.safetensors'); temporary = path.with_name(key+'.'+str(uuid.uuid4())+'.tmp.safetensors')
            mx.save_safetensors(str(temporary), tensors); temporary.replace(path)
            _publish_json(metadata_path, dict(schemaVersion=1, visualDigest=self.digest, modelSignature=self.config.signature,
                tensorHash=_file_hash(path), bytes=path.stat().st_size, sourceSurface=raw.metadata['surface']))
        return key, observation

    def get(self, key):
        if type(key) is not str or len(key)!=64 or any(char not in '0123456789abcdef' for char in key):
            raise ValueError('Invalid visual cache identity')
        result = self._resident.pop(key, None)
        if result is None:
            metadata = json.loads((self.root/(key+'.json')).read_text())
            path = self.root/(key+'.safetensors')
            if (metadata.get('schemaVersion')!=1 or metadata.get('visualDigest')!=self.digest
                or metadata.get('modelSignature')!=self.config.signature or path.is_symlink()
                or path.stat().st_size!=metadata['bytes'] or _file_hash(path)!=metadata['tensorHash']):
                raise ValueError('Frozen visual cache integrity or identity mismatch')
            tensors = mx.load(str(path))
            if set(tensors)!=set(VisualFeatures.__dataclass_fields__): raise ValueError('Invalid cached visual field set')
            result = VisualFeatures(**tensors); mx.eval(tensors)
            if (result.summary.shape!=(1,self.config.query_count*self.config.spatial_width)
                or result.cells.ndim!=4 or result.cells.shape[0]!=1 or result.cells.shape[-1]!=self.config.spatial_width
                or result.cell_bounds.shape!=(*result.cells.shape[:-1],4)
                or result.cell_valid.shape!=result.cells.shape[:-1] or result.surface_valid.shape!=result.cells.shape[:2]):
                raise ValueError('Cached visual shapes do not match the model')
            for value in tensors.values():
                if value.dtype not in (mx.float32,mx.bool_): raise ValueError('Cached visual precision changed')
                if not bool(mx.all(mx.isfinite(value)).item()): raise ValueError('Nonfinite cached visual')
        self._resident[key] = result
        while len(self._resident)>self.maximum_resident: self._resident.popitem(last=False)
        return result

    def batch(self, keys):
        values = [self.get(key) for key in keys]
        return VisualFeatures(**{name:mx.concatenate([getattr(value,name) for value in values]) for name in VisualFeatures.__dataclass_fields__})



from astra.environments.practice import PracticeEnvironment


class ImmediateCueEnvironment(PracticeEnvironment):
    """Actual raster variant: repeat the cue for the last waiting frame.

    Logical time, control scheduling, reward and oracle labels stay unchanged.
    The render-only clock view selects the ordinary cue raster and its current
    cursor. No answer/category is appended to model observations.
    """
    @property
    def elapsed_ms(self):
        return 0 if getattr(self, '_rendering_repeated_cue', False) else super().elapsed_ms

    def _render(self):
        elapsed = super().elapsed_ms
        ready = self.config.cue_ms + self.config.delay_ms
        self._rendering_repeated_cue = ready - self.config.period_ms <= elapsed < ready
        try:
            return super()._render()
        finally:
            self._rendering_repeated_cue = False

def prepare_episode(cache, environment, seed, *, counterfactual=False, cue_replay=False, cancelled=lambda:False):
    """Traverse ordinary oracle episodes; retain source times, not future labels."""
    from astra.environments.practice import PracticeEnvironment
    world = (ImmediateCueEnvironment if cue_replay else PracticeEnvironment)(environment); raw = world.reset(seed=seed)
    if counterfactual:
        world._answer = 1 - world._answer
        raw = world._observe()
    steps=[]; previous_events=[]; last_input=None
    while True:
        if cancelled(): raise InterruptedError('Diagnostic cache preparation cancelled')
        key,observation=cache.observe(raw,first=not steps,events=previous_events,last_input=last_input)
        commands=world.oracle_commands()
        if raw.metadata['eventNanos']>raw.metadata['observedNanos']: raise ValueError('Noncausal fixture frame')
        steps.append(dict(visual=key,controls=np.asarray(observation.controls[0,0]).tolist(),commands=commands,
                          surface=raw.metadata['surface'],observedNanos=raw.metadata['observedNanos']))
        transition=world.step(commands,episode_id=raw.episode_id,provenance='oracle')
        raw=transition.observation;previous_events=transition.raw_events
        times=[event['observedNanos'] for event in previous_events if event['origin'] in ('physical','agent')]
        if times: last_input=max(last_input or 0,max(times))
        if transition.outcome!='continuing':
            if transition.outcome!='terminated' or transition.reward<=0: raise ValueError('An oracle episode failed')
            break
    return dict(schemaVersion=1,id=str(uuid.uuid5(uuid.NAMESPACE_URL,environment.signature+':'+str(seed)+(':opposite' if counterfactual else '')+(':cue-replay' if cue_replay else ''))),
                pairID=str(uuid.uuid5(uuid.NAMESPACE_URL,environment.signature+':'+str(seed)+(':cue-replay' if cue_replay else ''))),counterfactual=counterfactual,
                cueReplay=cue_replay,
                seed=seed,delayMS=environment.delay_ms,environment=environment.to_dict(),steps=steps,provenance='practice_oracle')


def episode_batch(cache, episodes):
    if len(episodes)!=2: raise ValueError('A matched optimizer batch contains exactly two complete episodes')
    length=max(len(episode['steps']) for episode in episodes)
    controls=np.zeros((2,length,cache.config.control_width),dtype=np.float32)
    valid=np.zeros((2,length),dtype=np.bool_); reset=np.zeros_like(valid); reset[:,0]=True
    keys=[]; labels=[]; summaries=[]
    for lane,episode in enumerate(episodes):
        previous=-1
        for step in range(length):
            if step<len(episode['steps']):
                item=episode['steps'][step]
                if item['observedNanos']<=previous: raise ValueError('Source episode observation time is not increasing')
                previous=item['observedNanos']; controls[lane,step]=item['controls'];valid[lane,step]=True
                keys.append(item['visual']);visual=cache.get(item['visual'])
                labels.append(encode_commands(item['commands'],config=cache.config,vocabulary=cache.policy.actions.vocabulary,
                                              visual=visual,surfaces=[item['surface']]))
                summaries.append(visual.summary[0])
            else:
                keys.append(None);labels.append(PacketBatch.zeros(1,cache.config.packet_capacity+1))
                summaries.append(mx.zeros((cache.config.query_count*cache.config.spatial_width,)))
    observation=ObservationBatch((),mx.array(controls),mx.full((2,length),cache.config.period_ms/1000),
        mx.zeros((2,length,len(cache.config.context_sizes)),dtype=mx.int32),mx.array(reset),mx.array(valid))
    packets=PacketBatch(**{name:mx.concatenate([getattr(packet,name) for packet in labels]) for name in PacketBatch.__dataclass_fields__})
    summary=mx.stack(summaries).reshape(2,length,-1);mx.eval(summary,packets.operation)
    def visual_at(indices):
        selected=[keys[index] for index in indices]
        if any(key is None for key in selected): raise ValueError('Padded decisions cannot be scored as supervision')
        return cache.batch(selected)
    return CachedBatch(observation,summary,packets,visual_at)


def paired_validation(head, cache, fixture_directory, *, seeds=range(1000,1032), cue_replay=False, cancelled=lambda:False):
    """Held-out paired cue likelihood, never presented as freely acting success."""
    from astra.environments.practice import PracticeConfig
    from scripts.qualify_memory import memory_phase_fixture
    environment=PracticeConfig(task='delayed_memory',delay_ms=30000,time_limit_ms=32500)
    rows=[];feature_distances=[]
    for seed in seeds:
        if cancelled(): raise InterruptedError('Diagnostic validation deadline')
        phases,commands=memory_phase_fixture(environment,seed,Path(fixture_directory))
        entries=[cache.observe(raw,first=False) for raw in phases]
        visuals=[cache.get(key) for key,_ in entries]
        controls=entries[2][1].controls
        if any(not np.array_equal(controls,entry[1].controls) for entry in entries):
            raise ValueError('Paired fixture control features differ across passive phases')
        feature_distances.append(float(mx.sqrt(mx.mean((visuals[0].summary-visuals[1].summary)**2)).item()))
        state=head.temporal.initial_state(2)
        def advance(summary,steps,reset=False):
            observation=ObservationBatch((),mx.repeat(mx.broadcast_to(controls,(2,1,head.config.control_width)),steps,axis=1),
                mx.full((2,steps),.1),mx.zeros((2,steps,0),dtype=mx.int32),
                mx.broadcast_to((mx.arange(steps)==0)&reset,(2,steps)),mx.ones((2,steps),dtype=mx.bool_))
            result=head.temporal(mx.repeat(summary,steps,axis=1),observation,state)
            mx.eval(result.context,result.state)
            return result
        cue=mx.concatenate([visuals[0].summary,visuals[1].summary])[:,None]
        state=tuple(mx.stop_gradient(value) for value in advance(cue,5,True).state)
        wait=mx.broadcast_to(visuals[2].summary[:,None],cue.shape)
        choice=mx.broadcast_to(visuals[3].summary[:,None],cue.shape)
        labels=[encode_commands(packet,config=head.config,vocabulary=head.actions.vocabulary,visual=visuals[3],
                                surfaces=[phases[3].metadata['surface']]) for packet in commands]
        correct=PacketBatch(**{name:mx.concatenate([getattr(label,name) for label in labels]) for name in PacketBatch.__dataclass_fields__})
        wrong=PacketBatch(**{name:getattr(correct,name)[mx.array([1,0])] for name in PacketBatch.__dataclass_fields__})
        visual=cache.batch([entries[3][0],entries[3][0]])
        previous=0
        for delay in (2000,8000,30000):
            if cancelled(): raise InterruptedError('Diagnostic validation deadline')
            waiting_steps=(delay-previous)//100
            if cue_replay:
                # Branch each delay before its final waiting observation. The
                # longer-delay branch must not inherit earlier cue replays.
                state=tuple(mx.stop_gradient(value) for value in advance(wait,waiting_steps-1).state)
                before_replay=state
                state=tuple(mx.stop_gradient(value) for value in advance(cue,1).state)
                result=advance(choice,1)
                state=before_replay
                state=tuple(mx.stop_gradient(value) for value in advance(wait,1).state)
            else:
                state=tuple(mx.stop_gradient(value) for value in advance(wait,waiting_steps).state)
                result=advance(choice,1)
            good=head.actions.log_prob(result.context[:,0],visual,correct).log_probability
            bad=head.actions.log_prob(result.context[:,0],visual,wrong).log_probability
            mx.eval(good,bad)
            margin=np.asarray(good-bad)
            if not np.isfinite(margin).all(): raise FloatingPointError('Nonfinite held-out choice score')
            rows.append(dict(seed=seed,delayMS=delay,margins=margin.tolist(),
                             correct=float(((margin>0)+.5*(margin==0)).sum()),
                             cueChangedChoice=bool((margin[0]>0)==(margin[1]>0))))
            previous=delay
    return dict(scope='forced_choice_diagnostic',cueReplay=cue_replay,layouts=len(feature_distances),rows=rows,
                meanCueSummaryRMS=float(np.mean(feature_distances)),minimumCueSummaryRMS=min(feature_distances),
                byDelay=[dict(delayMS=delay,accuracy=sum(row['correct'] for row in rows if row['delayMS']==delay)/(2*len(feature_distances)),
                              cueChangedChoice=sum(row['cueChangedChoice'] for row in rows if row['delayMS']==delay)) for delay in (2000,8000,30000)])
