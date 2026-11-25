"""Model observation boundary, native action validation, and recorded replay."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from owb.bench.baselines.wildlands_model import WildlandsModelPolicy,describe,decode
from owb.env.wildlands_reference import WildlandsReferenceEnv
from owb.scripts.wildlands_model import run


def payload(**changes):
    out=dict(move=[0,0],look=[0,0],vector=[0,0,0],buttons=[],command='inspect',
             target=1,slot=0,ingredients=[0]*5,repeat=1,memory='Remember only what I observed.')
    out.update(changes);return out


class Client:
    def __init__(self,outputs): self.outputs=iter(outputs);self.responses=self;self.requests=[]
    def create(self,**kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(next(self.outputs)),model='gpt-6.1-sol',status='completed',id='test',
                               usage=SimpleNamespace(model_dump=lambda:dict(input_tokens=10,output_tokens=10)))


@pytest.mark.parametrize('changes',[
    {'command':'teleport'}, {'target':256}, {'move':[float('nan'),0]},
    {'repeat':True},{'repeat':2},{'vector':[1,2,0]},{'buttons':['oracle']},
])
def test_bad_or_repeated_discrete_actions_rejected(changes):
    with pytest.raises(ValueError): decode(payload(**changes))


def test_observation_allowlist_and_model_memory_only():
    with WildlandsReferenceEnv('a') as env:
        obs,_=env.reset(seed=7)
        obs['evaluator_secret']='DO_NOT_SEND'
        obs['rgb']=np.zeros((180,320,3),np.uint8)
        client=Client([payload(memory='My own learned note.'),payload(command='none')])
        policy=WildlandsModelPolicy(client)
        policy.decide(obs,'Repair the intake.')
        policy.decide(obs,'Repair the intake.')
        prompts=[json.loads(req['input'][0]['content'][0]['text']) for req in client.requests]
        assert prompts[0]['memory']==''
        assert prompts[1]['memory']=='My own learned note.'
        assert len(prompts[1]['recent_history'])==1
        assert 'DO_NOT_SEND' not in json.dumps(client.requests)
        assert 'milestones' not in describe(obs)
        assert client.requests[0]['model']=='gpt-6.1-sol'
        assert all('tools' not in req for req in client.requests)


def test_recorded_model_action_is_replayed_and_scored(tmp_path,monkeypatch):
    # Synthetic diagnostic images avoid rendering cost in the harness test.
    monkeypatch.setattr(WildlandsReferenceEnv,'render',lambda _:np.zeros((180,320,3),np.uint8))
    output=tmp_path/'episode'
    result=run('a',output,Client([payload()]),max_calls=1,max_actions=1,transport='test')
    assert result['milestones_completed']==1 and result['required']==12
    assert result['stop_reason']=='model_decision_limit'
    assert result['score_available'] and not result['success']
    assert json.loads((output/'verification.json').read_text())['replay_verified']
    assert len((output/'actions.jsonl').read_text().splitlines())==1


def test_inference_failure_is_not_a_zero_score(tmp_path,monkeypatch):
    monkeypatch.setattr(WildlandsReferenceEnv,'render',lambda _:np.zeros((180,320,3),np.uint8))
    class Broken:
        def __init__(self): self.responses=self
        def create(self,**kwargs): raise ConnectionError('test failure')
    result=run('b',tmp_path/'episode',Broken(),max_calls=1,transport='test')
    assert not result['score_available']
    assert result['stop_reason']=='inference_or_interface_error'
    assert result['primitive_actions']==0


def test_transport_resume_preserves_world_memory_and_budget(tmp_path,monkeypatch):
    from owb.bench.baselines.codex_policy import CodexPolicyError
    monkeypatch.setattr(WildlandsReferenceEnv,'render',lambda _:np.zeros((180,320,3),np.uint8))
    class Interrupted(Client):
        def create(self,**kwargs):
            if self.requests: raise CodexPolicyError('Test transport interruption')
            return super().create(**kwargs)
    first=tmp_path/'first'
    stopped=run('a',first,Interrupted([payload(memory='My observed commission.')]),max_calls=3,transport='test')
    assert stopped['model_decisions']==2 and stopped['milestones_completed']==1
    client=Client([payload(command='none')])
    resumed=run('a',tmp_path/'resumed',client,max_calls=3,transport='test',resume_source=first)
    prompt=json.loads(client.requests[0]['input'][0]['content'][0]['text'])
    assert prompt['memory']=='My observed commission.' and prompt['decision']==3
    assert len(prompt['recent_history'])==1
    assert resumed['model_decisions']==3 and resumed['primitive_actions']==2
    assert resumed['milestones_completed']==1 and resumed['stop_reason']=='model_decision_limit'


# --- checkpoint starts and the perception-oracle condition -------------------------------

def _prompt(client,k=0): return json.loads(client.requests[k]['input'][0]['content'][0]['text'])


def test_checkpoint_start_restores_verified_state_and_budgets_only_model_actions(tmp_path,monkeypatch):
    from owb.bench import checkpoints
    monkeypatch.setattr(WildlandsReferenceEnv,'render',lambda _:np.zeros((180,320,3),np.uint8))
    bank=checkpoints.bank_dir('wildlands','a',7)
    entry=checkpoints.load(bank)['checkpoints'][4]
    client=Client([payload(command='none',repeat=3)])
    output=tmp_path/'from-04'
    result=run('a',output,client,max_calls=1,max_actions=2,transport='test',start_milestone=4)
    assert result['milestones_completed']==4 and result['milestones_gained']==0
    assert result['primitive_actions']==2 and result['stop_reason']=='model_decision_limit'
    config=json.loads((output/'config.json').read_text())
    assert config['checkpoint']['replayed_actions']==entry['rows'] and config['checkpoint']['digest']==entry['digest']
    assert len((output/'actions.jsonl').read_text().splitlines())==2      # model actions only
    assert json.loads((output/'verification.json').read_text())['replay_verified']
    assert _prompt(client)['memory']=='' and 'Checkpoint start' in client.requests[0]['instructions']
    # Nothing from the reference prefix reaches the model.
    assert 'explanation' not in json.dumps(client.requests) and 'digest' not in json.dumps(client.requests)


def test_checkpoint_clue_memory_and_stop_after_count_from_the_checkpoint(tmp_path,monkeypatch):
    from owb.bench import checkpoints
    monkeypatch.setattr(WildlandsReferenceEnv,'render',lambda _:np.zeros((180,320,3),np.uint8))
    clues=checkpoints.clue_notes(checkpoints.bank_dir('wildlands','a',7),7)
    assert any('calibration mark' in c for c in clues)
    client=Client([payload(command='none')])
    result=run('a',tmp_path/'clues',client,max_calls=1,transport='test',start_milestone=7,
               checkpoint_memory='clues',stop_after=1)
    memory=_prompt(client)['memory']
    assert all(c in memory for c in clues) and result['start_milestone']==7


def test_checkpoint_start_rejects_a_stale_bank(tmp_path):
    from owb.bench import checkpoints
    with WildlandsReferenceEnv('a',7,render_enabled=False) as env:
        env.reset(seed=7)
        with pytest.raises(RuntimeError,match='different initial state'):
            checkpoints.restore(env,None,checkpoints.bank_dir('wildlands','a',7),3,'0'*64)


def test_perception_oracle_reports_visual_categories_not_roles():
    from owb.bench.baselines import perception_view
    from owb.bench.baselines.wildlands_model import WildlandsPerceptionPolicy
    with WildlandsReferenceEnv('a',7,render_enabled=False) as env:
        env.reset(seed=7)
        view=perception_view.wildlands(env)
        assert view['in_frame'], 'the start court faces the first fixtures'
        kinds={e['kind'] for e in view['in_frame']}
        assert kinds<={'post','plaque','movable block','orange-staked ground plate','creature'}
        text=json.dumps(view)
        for secret in ('brake','release','vent','heat','inlet','outlet','install','mark','milestone','A0'):
            assert secret not in text
        # Turning around takes the course out of frame.
        env.agent.yaw+=np.pi; env._frame=None
        ids={e['id'] for e in perception_view.wildlands(env)['in_frame'] if 'id' in e}
        far={e['id'] for e in view['in_frame'] if 'id' in e and e['distance_m']>10}
        assert far and not ids & far
        obs,_=env.reset(seed=7)
        obs['perceived']=view
        assert describe(obs)['perceived']==view
        assert 'PERCEPTION-ORACLE' in WildlandsPerceptionPolicy.guide
