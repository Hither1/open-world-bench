"""Observation boundary, real native actions, memory and replay accounting."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from open_world_bench.reference_tasks import ReferenceTaskEnv
from owb.bench.baselines.teyvat_model import TeyvatModelPolicy, decode, describe
from owb.scripts.teyvat_model import run, verify_run


def payload(**changes):
    out=dict(move=[0,0],look=[0,0],buttons=[],swap=0,repeat=1,memory='My observed note.')
    out.update(changes)
    return out


class Client:
    def __init__(self,outputs):
        self.outputs=iter(outputs);self.responses=self;self.requests=[]
    def create(self,**kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(next(self.outputs)),model='gpt-6.1-sol',status='completed',id='test',
            usage=SimpleNamespace(model_dump=lambda:dict(input_tokens=10,output_tokens=10)))


@pytest.mark.parametrize('changes',[
    {'move':[float('nan'),0]}, {'swap':4}, {'swap':True}, {'repeat':0}, {'repeat':31},
    {'buttons':['oracle']}, {'repeat':2,'buttons':['interact']}, {'repeat':2,'swap':1},
])
def test_invalid_actions_rejected(changes):
    with pytest.raises(ValueError): decode(payload(**changes))


def test_native_observation_allowlist_and_memory():
    with ReferenceTaskEnv('a',render_enabled=False) as env:
        obs,_=env.reset(seed=7)
        obs['evaluator_secret']='DO_NOT_SEND'
        client=Client([payload(memory='My own discovery'),payload()])
        policy=TeyvatModelPolicy(client)
        policy.decide(obs,'Restore the network.')
        policy.decide(obs,'Restore the network.')
        first,second=[json.loads(r['input'][0]['content'][0]['text']) for r in client.requests]
        assert first['memory']=='' and second['memory']=='My own discovery'
        assert len(second['recent_history'])==1
        assert 'DO_NOT_SEND' not in json.dumps(client.requests)
        assert set(describe(obs))=={'instruction','local_text','proprio','party','task'}
        assert client.requests[0]['model']=='gpt-6.1-sol'
        assert all('tools' not in r for r in client.requests)


def test_native_actions_memory_and_replay(tmp_path,monkeypatch):
    monkeypatch.setattr(ReferenceTaskEnv,'render',lambda _:np.zeros((216,384,3),np.uint8))
    client=Client([payload(move=[0,1],repeat=3),payload(buttons=['interact']),payload()])
    output=tmp_path/'episode'
    result=run('a',output,client,max_calls=3,transport='test')
    assert result['primitive_actions']==5
    assert result['milestones_completed']==0 and result['required']==12
    assert result['score_available'] and result['stop_reason']=='model_decision_limit'
    third=json.loads(client.requests[2]['input'][0]['content'][0]['text'])
    assert 'Relay lesson:' in third['observation']['local_text']
    assert json.loads((output/'verification.json').read_text())['replay_verified']
    with (output/'actions.jsonl').open('a') as stream: stream.write('{}\n')
    with pytest.raises(RuntimeError,match='hash differs'): verify_run(output)


def test_inference_failure_is_not_model_zero(tmp_path,monkeypatch):
    monkeypatch.setattr(ReferenceTaskEnv,'render',lambda _:np.zeros((216,384,3),np.uint8))
    class Broken:
        def __init__(self): self.responses=self
        def create(self,**kwargs): raise ConnectionError('test')
    result=run('b',tmp_path/'episode',Broken(),max_calls=1,transport='test')
    assert not result['score_available'] and result['primitive_actions']==0
    assert result['stop_reason']=='inference_or_interface_error'


def test_transient_transport_retry_preserves_budget_and_memory(tmp_path,monkeypatch):
    from owb.bench.baselines.codex_policy import CodexPolicyError
    monkeypatch.setattr(ReferenceTaskEnv,'render',lambda _:np.zeros((216,384,3),np.uint8))
    class Interrupted(Client):
        def create(self,**kwargs):
            if not self.requests:
                self.requests.append(kwargs)
                raise CodexPolicyError('Codex inference failed (exit 1); test')
            return super().create(**kwargs)
    client=Interrupted([payload()])
    result=run('a',tmp_path/'episode',client,max_calls=2,transport='test')
    assert result['model_decisions']==2 and result['primitive_actions']==1
    request=json.loads(client.requests[1]['input'][0]['content'][0]['text'])
    assert request['memory']=='' and request['recent_history']==[]
    assert request['decision']==2 and result['stop_reason']=='model_decision_limit'


def test_policy_image_refreshes_headless_cache():
    from owb.scripts.teyvat_model import policy_image
    with ReferenceTaskEnv('a',render_enabled=False) as env:
        env._last_frame=np.full((216,384,3),255,np.uint8)
        env._renderer.render=lambda *args,**kwargs: np.zeros((216,384,3),np.uint8)
        assert not policy_image(env).any()
