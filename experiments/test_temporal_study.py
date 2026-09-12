"""Independent frozen-cache versus ordinary-policy gradient checks."""
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

from astra.data.actions import encode_commands
from astra.model.actions import ActionVocabulary, PacketBatch, flatten_visual
from astra.model.config import ModelConfig
from astra.model.observation import ObservationBatch, SurfaceBatch
from astra.model.policy import AgentPolicy
from astra.model.vision import VisualFeatures
from experiments.temporal_study import CachedBatch, FrozenHead, cached_batch_gradients
from scripts.qualify_memory import initialize_experimental_retention


def fixture(length=4, lanes=2):
    mx.random.seed(704)
    config = ModelConfig.test_small()
    policy = AgentPolicy(config, ActionVocabulary(key_codes=(0, 13), mouse_buttons=(0,), absolute_pointer=True))
    initialize_experimental_retention(policy, 'geometric')
    policy.vision.freeze(); policy.eval()
    rect = mx.broadcast_to(mx.array([0.,0.,1.,1.]), (lanes,length,4))
    surface = SurfaceBatch(*(mx.random.normal((lanes,length,32,32,3)) for _ in range(3)),
                           rect,rect,rect,mx.broadcast_to(mx.array([0.,0.,32.,32.]), (lanes,length,4)),
                           mx.ones((lanes,length), dtype=mx.bool_))
    valid = mx.ones((lanes,length), dtype=mx.bool_)
    if lanes>1: valid = mx.arange(length)[None] < mx.array([[length],[length-1]])
    observation = ObservationBatch((surface,), mx.random.normal((lanes,length,178)), mx.full((lanes,length),.1),
                                   mx.zeros((lanes,length,0),dtype=mx.int32), mx.broadcast_to(mx.arange(length)==0,(lanes,length)),valid)
    features = policy.encode_visual(observation); mx.eval(tuple(vars(features).values()))
    flat = flatten_visual(features)
    descriptor = dict(id='s',globalBounds=dict(x=0,y=0,width=32,height=32),pixelWidth=32,pixelHeight=32,
                      contentBounds=dict(x=0,y=0,width=32,height=32),geometryRevision=0)
    packets = []
    for index in range(lanes*length):
        commands = []
        if index == length-1 or (lanes>1 and index==2*length-2):
            commands = [dict(operation='pointerAbsolute',offsetMs=2,surfaceID='s',x=.4,y=.6),
                        dict(operation='buttonDown',offsetMs=3,button=0)]
        packets.append(encode_commands(commands,config=config,vocabulary=policy.actions.vocabulary,
                                       visual=flat,surfaces=[descriptor],batch_index=index))
    labels = PacketBatch(**{name:mx.concatenate([getattr(packet,name) for packet in packets]) for name in PacketBatch.__dataclass_fields__})
    def visual_at(indices):
        return VisualFeatures(**{name:getattr(flat,name)[mx.array(indices)] for name in VisualFeatures.__dataclass_fields__})
    return policy,CachedBatch(observation,features.summary,labels,visual_at)


@pytest.mark.parametrize('choice_only',[False,True])
def test_cached_gradients_match_ordinary_full_policy_with_frozen_vision(choice_only):
    policy,batch = fixture()
    head = FrozenHead(policy)
    mask = batch.observation.valid.reshape(-1)
    if choice_only: mask = mask & (batch.packets.operation[:,0]!=0)
    denominator = mx.sum(batch.observation.valid)
    def objective(model):
        encoding = model(batch.observation)
        score = model.score(encoding,batch.packets)
        return -mx.sum(mx.where(mask,score.log_probability,0))/denominator
    expected_loss,expected_gradient = nn.value_and_grad(policy,objective)(policy)
    mx.eval(expected_loss,expected_gradient)
    actual = cached_batch_gradients(head,batch,horizon=512,choice_only=choice_only)
    np.testing.assert_allclose(actual.loss,expected_loss,rtol=1e-5,atol=1e-5)
    left,right = dict(tree_flatten(actual.gradients)),dict(tree_flatten(expected_gradient))
    assert left.keys()==right.keys()
    for name in left:
        np.testing.assert_allclose(left[name],right[name],rtol=2e-4,atol=2e-5,err_msg=name)
    assert actual.valid_count==7 and actual.selected_count==(2 if choice_only else 7)
    assert bool(mx.all(mx.isfinite(actual.summary_gradient)).item())


def test_horizon_cut_changes_credit_but_not_forward_state_or_packet_loss():
    policy,batch = fixture(length=66,lanes=1)
    head = FrozenHead(policy)
    short = cached_batch_gradients(head,batch,horizon=64,choice_only=True)
    long = cached_batch_gradients(head,batch,horizon=512,choice_only=True)
    np.testing.assert_allclose(short.loss,long.loss,rtol=1e-5,atol=1e-5)
    for left,right in zip(short.next_state,long.next_state): np.testing.assert_allclose(left,right,rtol=1e-5,atol=2e-6)
    np.testing.assert_array_equal(short.summary_gradient[:,:64],0)
    assert float(mx.sqrt(mx.sum(long.summary_gradient[:,:5]**2)).item())>1e-10
    assert float(mx.sqrt(mx.sum(short.summary_gradient[:,64:]**2)).item())>1e-10


def test_future_observations_and_labels_never_enter_prior_temporal_state():
    policy,batch = fixture(length=5,lanes=1)
    original = policy.temporal(batch.summary,batch.observation)
    changed_observation = replace(batch.observation, controls=mx.concatenate((batch.observation.controls[:,:3],mx.full((1,2,178),99.)),axis=1))
    changed_summary = mx.concatenate((batch.summary[:,:3],mx.full((1,2,batch.summary.shape[-1]),-50.)),axis=1)
    changed = policy.temporal(changed_summary,changed_observation)
    np.testing.assert_array_equal(original.context[:,:3],changed.context[:,:3])
    head = FrozenHead(policy)
    first = cached_batch_gradients(head,batch,horizon=512,choice_only=False)
    fields = dict(vars(batch.packets)); coordinates = np.asarray(fields['within_x']).copy(); coordinates[-1,0] = 1; fields['within_x'] = mx.array(coordinates)
    different_labels = replace(batch,packets=PacketBatch(**fields))
    second = cached_batch_gradients(head,different_labels,horizon=512,choice_only=False)
    for left,right in zip(first.next_state,second.next_state): np.testing.assert_array_equal(left,right)


def test_cancellation_restores_parameter_bindings_and_never_updates_them():
    policy,batch = fixture()
    head = FrozenHead(policy); parameters = dict(tree_flatten(head.parameters()))
    checks = 0
    def cancelled():
        nonlocal checks
        checks+=1
        return checks>=3
    with pytest.raises(InterruptedError): cached_batch_gradients(head,batch,horizon=512,choice_only=False,cancelled=cancelled)
    after = dict(tree_flatten(head.parameters()))
    for name,value in parameters.items(): np.testing.assert_array_equal(value,after[name])


def test_visual_cache_matches_uncached_encoding_and_rejects_changed_weights_or_bytes(tmp_path):
    from astra.data.observations import make_observation
    from astra.environments.practice import PracticeConfig, PracticeEnvironment
    from experiments.temporal_study import VisualCache
    policy,_=fixture(length=1,lanes=1)
    env=PracticeEnvironment(PracticeConfig(task='delayed_memory',pixel_width=64,pixel_height=64))
    raw=env.reset(seed=0)
    cache=VisualCache(tmp_path,policy)
    key,observation=cache.observe(raw,first=True)
    actual=cache.get(key)
    reference_observation=make_observation([(raw.pixels,raw.metadata)],raw.control_state,
        cutoff_nanos=raw.metadata['observedNanos'],elapsed_seconds=.1,reset=True,config=policy.config)
    expected=flatten_visual(policy.encode_visual(reference_observation))
    for name in VisualFeatures.__dataclass_fields__: np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
    later=env.step([],episode_id=raw.episode_id).observation
    same_key,later_observation=cache.observe(later,first=False)
    assert key==same_key and not bool(later_observation.reset.item())
    np.testing.assert_array_equal(observation.controls,later_observation.controls)
    cache._resident.clear()
    path=cache.root/(key+'.safetensors')
    data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    with pytest.raises(ValueError,match='integrity'): cache.get(key)
    with pytest.raises(ValueError,match='different frozen weights'): VisualCache(tmp_path,policy,expected_digest='0'*64)


def test_oracle_cache_episode_matches_real_demonstration_samples(tmp_path):
    from astra.environments.practice import PracticeConfig
    from astra.environments.demonstrations import PracticeDemonstrations
    from experiments.temporal_study import VisualCache,prepare_episode,episode_batch
    environment=PracticeConfig(task='delayed_memory',pixel_width=64,pixel_height=64,logical_bounds=(0,0,64,64),time_limit_ms=4500)
    source=PracticeDemonstrations(environment=environment,model=ModelConfig.test_small(),seeds_by_split={'train':[0,1]})
    mx.random.seed(834);policy=AgentPolicy(ModelConfig.test_small(),source.vocabulary);policy.eval()
    cache=VisualCache(tmp_path,policy)
    episodes=[prepare_episode(cache,environment,seed) for seed in (0,1)]
    for prepared,original in zip(episodes,source.episodes('train')):
        assert len(prepared['steps'])==original['steps']==27
        samples=list(source.samples(original['id']))
        for item,sample in zip(prepared['steps'],samples):
            assert item['commands']==list(sample.commands)
            np.testing.assert_array_equal(item['controls'],sample.observation.controls[0,0])
        for index in (0,5,25):
            expected=flatten_visual(policy.encode_visual(samples[index].observation))
            actual=cache.get(prepared['steps'][index]['visual'])
            for name in VisualFeatures.__dataclass_fields__: np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
    batch=episode_batch(cache,episodes)
    assert batch.observation.shape==(2,27) and int(mx.sum(batch.observation.valid).item())==54
    assert int(mx.sum(batch.packets.operation[:,0]!=0).item())==2


def test_complete_episode_boundary_resume_matches_uninterrupted_updates(tmp_path):
    import uuid
    import mlx.optimizers as optim
    from astra.checkpoints import save_checkpoint,load_checkpoint
    from astra.learning.optimizers import GroupedAdamW
    policy,batch=fixture()
    initial=tmp_path/str(uuid.uuid4());save_checkpoint(initial,policy,kind='initial',step=0)
    def optimizer(): return GroupedAdamW(learning_rate=3e-4,pretrained_learning_rate=3e-5,weight_decay=.01)
    def step(policy,opt):
        head=FrozenHead(policy)
        result=cached_batch_gradients(head,batch,horizon=512,choice_only=True)
        gradient,norm=optim.clip_grad_norm(result.gradients,1.)
        opt.update(head,gradient);mx.eval(head.parameters(),opt.state)
    uninterrupted=load_checkpoint(initial).policy;full_optimizer=optimizer()
    for _ in range(3): step(uninterrupted,full_optimizer)
    partial=load_checkpoint(initial).policy;part_optimizer=optimizer()
    for _ in range(2): step(partial,part_optimizer)
    checkpoint=tmp_path/str(uuid.uuid4())
    save_checkpoint(checkpoint,partial,kind='behavioral',step=2,training_state=dict(optimizer=part_optimizer.state,updates=2))
    restored=load_checkpoint(checkpoint,include_training=True)
    resumed_optimizer=optimizer();resumed_optimizer.state=restored.training_state['optimizer']
    step(restored.policy,resumed_optimizer)
    expected,actual=dict(tree_flatten(uninterrupted.parameters())),dict(tree_flatten(restored.policy.parameters()))
    for name in expected: np.testing.assert_allclose(actual[name],expected[name],rtol=0,atol=1e-7,err_msg=name)


def test_paired_training_removes_layout_answer_shortcut_without_hidden_features(tmp_path):
    from astra.environments.practice import PracticeConfig,PracticeEnvironment
    from experiments.temporal_study import VisualCache,prepare_episode,episode_batch
    configuration=PracticeConfig(task='delayed_memory',pixel_width=64,pixel_height=64,logical_bounds=(0,0,64,64),time_limit_ms=4500)
    vocabulary=PracticeEnvironment(configuration).action_vocabulary
    mx.random.seed(834);policy=AgentPolicy(ModelConfig.test_small(),vocabulary);policy.eval()
    cache=VisualCache(tmp_path,policy)
    pair=[prepare_episode(cache,configuration,7,counterfactual=opposite) for opposite in (False,True)]
    assert pair[0]['pairID']==pair[1]['pairID'] and pair[0]['id']!=pair[1]['id']
    for index,(left,right) in enumerate(zip(pair[0]['steps'],pair[1]['steps'])):
        assert left['controls']==right['controls'] and left['surface']==right['surface']
        assert (left['visual']==right['visual'])==(index>=5)
    assert pair[0]['steps'][25]['commands']!=pair[1]['steps'][25]['commands']
    batch=episode_batch(cache,pair)
    np.testing.assert_array_equal(batch.observation.controls[0],batch.observation.controls[1])
    np.testing.assert_array_equal(batch.summary[0,5:],batch.summary[1,5:])
    assert not np.array_equal(batch.summary[0,:5],batch.summary[1,:5])
    assert int(mx.sum(batch.observation.valid).item())==54
    assert int(mx.sum(batch.packets.operation[:,0]!=0).item())==2


def test_rendered_immediate_cue_preserves_clock_controls_labels_and_all_other_frames(tmp_path):
    from astra.environments.practice import PracticeConfig,PracticeEnvironment
    from experiments.temporal_study import ImmediateCueEnvironment,VisualCache,prepare_episode
    configuration=PracticeConfig(task='delayed_memory',pixel_width=64,pixel_height=64,logical_bounds=(0,0,64,64),time_limit_ms=4500)
    normal=PracticeEnvironment(configuration);replay=ImmediateCueEnvironment(configuration)
    left=normal.reset(seed=8);right=replay.reset(seed=8)
    for step in range(27):
        assert normal.elapsed_ms==replay.elapsed_ms==step*100
        assert left.metadata['eventNanos']==right.metadata['eventNanos']
        assert left.control_state==right.control_state
        assert np.array_equal(left.pixels,right.pixels)==(step!=24)
        if step==24:
            # A cursor movement during the wait must also appear in the replayed
            # cue raster, rather than copying a stale first-frame cursor.
            expected=PracticeEnvironment(configuration);expected.reset(seed=8)
            pointer=right.control_state['pointer'];expected._pointer=(pointer['x'],pointer['y'])
            np.testing.assert_array_equal(right.pixels,expected._render())
            assert replay.elapsed_ms==2400
        commands=([dict(operation='pointerAbsolute',offsetMs=0,surfaceID='practice',x=.1,y=.1)] if step==5 else normal.oracle_commands())
        a=normal.step(commands,episode_id=left.episode_id,provenance='oracle')
        b=replay.step(commands,episode_id=right.episode_id,provenance='oracle')
        assert (a.reward,a.outcome)==(b.reward,b.outcome)
        left,right=a.observation,b.observation
        if a.outcome!='continuing':break
    policy=AgentPolicy(ModelConfig.test_small(),normal.action_vocabulary);policy.eval()
    cache=VisualCache(tmp_path,policy)
    original=prepare_episode(cache,configuration,7)
    immediate=prepare_episode(cache,configuration,7,cue_replay=True)
    assert len(original['steps'])==len(immediate['steps'])==27
    for index,(a,b) in enumerate(zip(original['steps'],immediate['steps'])):
        assert a['commands']==b['commands'] and a['controls']==b['controls']
        assert (a['visual']==b['visual'])==(index!=24)
    assert immediate['steps'][24]['visual']==immediate['steps'][0]['visual']


def test_choice_likelihood_really_depends_on_temporal_context_not_only_labels():
    policy,batch=fixture(length=4,lanes=1)
    context=policy.temporal(batch.summary,batch.observation).context[:,-1]
    visual=batch.visual_at([3])
    correct=PacketBatch(**{name:getattr(batch.packets,name)[3:4] for name in PacketBatch.__dataclass_fields__})
    fields=dict(vars(correct));cell=np.asarray(fields['cell']).copy();cell[0,0]=(cell[0,0]+1)%visual.cells.shape[2]
    fields['cell']=mx.array(cell);wrong=PacketBatch(**fields)
    def difference(value):
        return mx.sum(policy.actions.log_prob(value,visual,correct).log_probability-policy.actions.log_prob(value,visual,wrong).log_probability)
    gradient=mx.grad(difference)(context);norm=mx.sqrt(mx.sum(gradient**2));mx.eval(gradient,norm)
    assert float(norm.item())>1e-6
    direction=gradient/norm
    epsilon=1e-3
    finite_difference=(difference(context+epsilon*direction)-difference(context-epsilon*direction))/(2*epsilon)
    np.testing.assert_allclose(finite_difference,norm,rtol=.03,atol=1e-3)
