"""Meaningful mechanism interlocks, live goals, and legal reference rollouts."""
from copy import deepcopy
import numpy as np
import pytest
from owb.core import blocks as B
from owb.env.skylands_reference import SkylandsReferenceEnv, LEVELS
from owb.bench.baselines.skylands_reference import neutral
from owb.scripts.skylands_reference import SkylandsWMReferencePolicy as SkylandsReferencePolicy


def test_remote_manipulation_and_incompatible_mount_are_atomic():
    with SkylandsReferenceEnv('a') as e:
        e.reset(); target=next(k for k,o in e.targets.items() if o['role']=='part')
        before=deepcopy(e.parts)
        a=neutral(); a['target']=target
        events=e._dispatch('pick',[],a)
        assert events[0]['type']=='action_failed' and e.held_part is None
        for k in before: np.testing.assert_array_equal(before[k]['pos'],e.parts[k]['pos'])
        r=e.rafts['A02']; e.agent.pos=r['pos'].astype(np.float32)+(0,0,-2)
        e.held_part=next(k for k,p in e.parts.items() if p['owner']=='A02' and p['kind']=='cell')
        held=e.held_part; a['target']=r['target']; a['option']=0
        assert e._dispatch('mount',[],a)[0]['type']=='action_failed'
        assert r['parts']==[None]*4 and e.held_part==held


def test_craft_rejects_overload_missing_power_and_reversed_propulsion():
    with SkylandsReferenceEnv('a') as e:
        e.reset(); r=e.rafts['A02']
        r['parts']=[k for k,p in e.parts.items() if p['owner']=='A02']; r['charge']=100
        r['cargo']='A02_ballast'
        assert e.craft_valid(r)
        e.cargo[r['cargo']]['mass']=281
        assert not e.craft_valid(r)
        e.cargo[r['cargo']]['mass']=90; r['charge']=0
        assert not e.craft_valid(r)
        r['charge']=100; e.parts[r['parts'][2]]['turn']=1
        assert not e.craft_valid(r)


def test_partial_boarding_cannot_move_a_craft():
    with SkylandsReferenceEnv('a') as e:
        e.reset(); r=e.rafts['A02']
        e.agent.pos=r['pos'].astype(np.float32)+(-1.6,0,0)
        assert not e.on_deck(r)
        e.agent.pos=r['pos'].astype(np.float32)+(.5,0,.5)
        assert e.on_deck(r)


def test_lift_can_descend_past_its_old_deck_but_not_an_obstacle():
    with SkylandsReferenceEnv('a') as e:
        e.reset(); mv=e.movers['transit']; start=mv['pos'].copy()
        assert e._move_deck(mv,start+(0,-1,0)) is not None
        np.testing.assert_array_equal(mv['pos'],start+(0,-1,0))
        e.world.set_block(8,int(mv['pos'][1])+2,8,B.BRICK)
        before=mv['pos'].copy()
        assert e._move_deck(mv,before+(0,1,0)) is None
        np.testing.assert_array_equal(mv['pos'],before)


def test_obstructed_reversal_preserves_history_and_cannot_rewind_inventory():
    with SkylandsReferenceEnv('b') as e:
        e.reset(); mv=e.movers['B06_platform']; p=mv['pos'].copy(); q=p+(1,0,0)
        mv['history'].clear(); mv['history'].append(q.tolist()); mv['reverse']=True
        e.satchel['test_item']=1; health=e.agent.health
        e.world.set_block(int(q[0])+1,int(q[1]),int(q[2]),B.BRICK)
        events=e._advance([])
        assert any(x['type']=='motion_blocked' for x in events)
        np.testing.assert_array_equal(mv['pos'],p)
        assert list(mv['history'])==[q.tolist()] and not mv['reverse']
        assert e.satchel['test_item']==1 and e.agent.health==health


@pytest.mark.parametrize('task,count',[('a',15),('b',24)])
def test_reference_finishes_by_legal_actions_and_live_cargo_matters(task,count):
    with SkylandsReferenceEnv(task,seed=23) as e:
        obs,_=e.reset(seed=23); p=SkylandsReferencePolicy(e)
        while not e._ended:
            action=p(obs); assert e.action_space.contains(action)
            obs,_,_,_,_=e.step(action)
        assert e.success and len(e.completed)==count and not e.agent.deaths
        assert e.system_operational()
        cargo=e.cargo['core' if task=='a' else 'relief']
        cargo['pos'][0]+=10
        assert not e.system_operational(), 'Historical milestones must not substitute for live delivery'


def test_scene_export_preserves_state_and_includes_skylands_content():
    from owb.render.scene_export import capture_scene
    with SkylandsReferenceEnv('a') as e:
        e.reset(); before=e.state_digest(); scene=capture_scene(e,width=96,height=54)
        assert e.state_digest()==before
        assert scene.metadata['environment']=='skylands' and scene.metadata['reference_task']=='a'
        assert scene.metadata['point_lights'] and len(scene.meshes)>3
