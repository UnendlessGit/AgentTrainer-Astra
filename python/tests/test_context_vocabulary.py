from dataclasses import replace
import uuid
import pytest
from astra.model.config import ModelConfig
from astra.model.contexts import vocabulary
from astra.model.policy import AgentPolicy
from astra.checkpoints import save_checkpoint, load_checkpoint
from astra.environments.demonstrations import PracticeDemonstrations
from astra.environments.practice import PracticeConfig, PracticeEnvironment


def configured():
    fields = [{'id': str(uuid.uuid4()), 'name': 'Task', 'values': [
        {'id': str(uuid.uuid4()), 'name': 'Organize'}, {'id': str(uuid.uuid4()), 'name': 'Inspect'}]}]
    return replace(ModelConfig.test_small(), context_sizes=(3,), context_vocabulary=vocabulary(fields, (3,)))


def test_context_names_do_not_change_semantics_but_ids_and_order_do():
    original = configured()
    renamed = original.to_dict()
    renamed['context_vocabulary'][0]['name'] = 'Purpose'
    renamed['context_vocabulary'][0]['values'][0]['name'] = 'Sort'
    parsed = ModelConfig.from_dict(renamed)
    assert parsed == original and parsed.signature == original.signature
    renamed['context_vocabulary'][0]['values'].reverse()
    assert ModelConfig.from_dict(renamed).signature != original.signature
    assert 'context_vocabulary' not in ModelConfig().to_dict()
    assert ModelConfig.from_dict(ModelConfig().to_dict()) == ModelConfig()
    renamed['context_sizes'] = [2]
    with pytest.raises(ValueError, match='dimensions'):
        ModelConfig.from_dict(renamed)


def test_context_checkpoint_snapshot_and_practice_unknown(tmp_path):
    config = configured()
    environment = PracticeConfig()
    policy = AgentPolicy(config, PracticeEnvironment(environment).action_vocabulary)
    destination = tmp_path / str(uuid.uuid4())
    manifest = save_checkpoint(destination, policy, kind='initial', step=0)
    loaded = load_checkpoint(destination)
    assert loaded.policy.config == config
    assert loaded.manifest['model']['context_vocabulary'] == config.to_dict()['context_vocabulary']
    source = PracticeDemonstrations(environment=environment, model=config, seeds_by_split={'train': [0]})
    row = next(source.samples(source.episodes()[0]['id'], count=1))
    assert row.observation.context_ids.tolist() == [[[0]]]
