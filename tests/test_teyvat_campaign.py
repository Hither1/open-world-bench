"""Goal-first campaign contracts. Fixture placement below is only for unit tests;
feasibility traces use the native-action controller without state writes.
"""
from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest

from open_world_bench.campaign_env import TeyvatCampaignEnv
from open_world_bench.campaign_policy import CampaignReferencePolicy
from open_world_bench.campaign_tasks import GOALS, ELEMENTS
from open_world_bench.elements import Element


def make(task):
    return TeyvatCampaignEnv(task,seed=7,render_enabled=False,geography_scale=.5,width=16,height=12)


def stand(env, eid):
    env.char.pos = env.world.get(eid).pos.copy()+np.array((0,0,1.2),np.float32)
    env.char.vel[:] = 0
    env.char.yaw = 0


def use(env, eid):
    stand(env,eid)
    env.campaign.interact(env.char)


def solve(env, site):
    for value in env.campaign.sequence(site):
        element = Element(value)
        stand(env,site.entities[element.name.lower()])
        env.campaign.apply_element(env.char,element)


def shrine_fixture(env, sid):
    c = env.campaign
    site = c.sites[sid]
    use(env,site.entities['ward']);use(env,site.entities['altar']);solve(env,site)
    enemy = env.world.get(site.entities['guard'])
    c.damage_guard(enemy,1000,Element.NONE)
    use(env,site.entities['altar'])
    return site


@pytest.mark.parametrize('task,count',[('summit',9),('cleansing',17),('desert',15)])
def test_goal_only_contract_idle_failure_and_determinism(task,count):
    with make(task) as env:
        obs,info = env.reset(seed=7)
        assert env.observation_space.contains(obs)
        assert obs['instruction']==GOALS[task]
        assert len(env.campaign.milestones)==count
        assert not np.any(obs['task']) and env.objective_target() is None
        assert not {'sites','sequence','facts','position','target'} & info.keys()
        initial = env.state_digest()
        before_rng = deepcopy(env.campaign.rng.bit_generator.state)
        policy = CampaignReferencePolicy(env)
        policy(obs)
        assert env.state_digest()==initial  # Oracle itself may not mutate the simulator.
        assert env.campaign.rng.bit_generator.state==before_rng
        for _ in range(5):
            obs,reward,done,truncated,info=env.step(policy.neutral())
            assert reward==0 and not done and not truncated
            assert obs['instruction']==GOALS[task] and not obs['task'].any()
        assert not env.campaign.facts and not env.campaign.verify()['success']
        env.reset(seed=7)
        assert env.state_digest()==initial
        env.reset(seed=19)
        assert env.state_digest()!=initial


def test_quartz_is_one_strike_consumable_with_expiry_and_native_swing_cooldown():
    with make('summit') as env:
        env.reset(seed=7)
        site=env.campaign.sites['lower_0']
        stand(env,site.entities['ice'])
        env.world.attack(env.char)
        assert site.hits==0
        use(env,site.entities['quartz'])
        stand(env,site.entities['ice'])
        env.world._player_attack_cd=0  # Isolate the charged-hit transition.
        env.world.attack(env.char)
        assert site.hits==1 and env.campaign.charge_until==0
        env.world.attack(env.char)
        assert site.hits==1
        env.world.elapsed+=3
        use(env,site.entities['quartz'])
        env.world.elapsed+=46
        stand(env,site.entities['ice']);env.world._player_attack_cd=0
        env.world.attack(env.char)
        assert site.hits==1
        env.campaign.tick(env.char,.1)
        assert env.campaign.metrics['quartz_expirations']==1


def test_summit_access_cannot_be_bypassed_by_visiting_upper_shards_or_spire():
    with make('summit') as env:
        env.reset(seed=7);c=env.campaign
        use(env,'ascent:gate');use(env,'final:spire')
        site=c.sites['upper_0']
        use(env,site.entities['quartz']);stand(env,site.entities['ice'])
        c.strike_ice(env.char)
        assert site.hits==0 and 'ascent' not in c.facts and 'final' not in c.facts
        assert env.world.architecture.active()
        assert c.metrics['blocked_interactions']>=3


def test_cold_requires_warmth_and_does_not_award_progress():
    with make('summit') as env:
        env.reset(seed=7);c=env.campaign
        env.char.pos=env.world.ground_pos(120,120);c.cold=99.9
        hp=env.char.hp;c.tick(env.char,1)
        assert env.char.hp<hp and c.metrics['cold_damage']>0
        use(env,'camp:fire')
        assert c.cold==0 and not c.facts


def test_guard_and_latch_shards_have_real_preconditions():
    with make('summit') as env:
        env.reset(seed=7);c=env.campaign
        for sid in ('lower_1','lower_2'):
            site=c.sites[sid]
            use(env,site.entities['quartz']);stand(env,site.entities['ice'])
            c.strike_ice(env.char)
            assert site.hits==0
        solve(env,c.sites['lower_2'])
        assert c.sites['lower_2'].solved
        assert 'lower_2' not in c.facts  # Opening a latch is not thawing the shard.


def test_shrine_wards_are_local_unique_consumables_and_branches_are_unordered():
    with make('cleansing') as env:
        env.reset(seed=7);c=env.campaign
        s=c.sites['shrine_4']
        use(env,s.entities['ward'])
        assert not c.facts
        use(env,'lens:tool');use(env,s.entities['ward']);use(env,s.entities['ward'])
        assert c.inventory['shrine_4_ward']==1
        assert 'shrine_4_ward' in c.facts and 'shrine_0_ward' not in c.facts
        use(env,c.sites['shrine_0'].entities['altar'])
        assert not c.sites['shrine_0'].installed
        use(env,s.entities['altar']);use(env,s.entities['altar'])
        assert s.installed and c.inventory['shrine_4_ward']==0
        wrong=next(e for e in ELEMENTS if int(e)!=c.sequence(s)[0])
        stand(env,s.entities[wrong.name.lower()]);c.apply_element(env.char,wrong)
        assert s.progress==0 and s.installed and c.metrics['wrong_applications']==1
        solve(env,s)
        assert s.solved and not s.finished
        guardian=env.world.get(s.entities['guard'])
        c.damage_guard(guardian,1000,Element.NONE);use(env,s.entities['altar'])
        assert 'shrine_4' in c.facts and 'shrine_0' not in c.facts


def test_receivers_cannot_be_activated_remotely_and_shields_cannot_be_bruteforced():
    with make('cleansing') as env:
        env.reset(seed=7);c=env.campaign
        use(env,'lens:tool');s=c.sites['shrine_0']
        use(env,s.entities['ward']);use(env,s.entities['altar'])
        enemy=env.world.get(s.entities['guard']);before=enemy.hp
        for element in ELEMENTS:
            c.damage_guard(enemy,100000,element)
        assert enemy.hp==before and enemy.alive
        env.char.pos=env.world.ground_pos(120,120)
        for value in c.sequence(s): c.apply_element(env.char,Element(value))
        assert s.progress==0 and not s.solved


def test_final_boss_requires_all_branches_and_two_puzzle_combat_phases():
    with make('cleansing') as env:
        env.reset(seed=7);c=env.campaign;heart=c.sites['heart']
        enemy=env.world.get(heart.entities['guard'])
        use(env,'lens:tool')
        for i in (4,2,0,3): shrine_fixture(env,f'shrine_{i}')
        use(env,heart.entities['altar']);c.damage_guard(enemy,1e9,Element.NONE)
        assert not heart.installed and enemy.hp==160
        shrine_fixture(env,'shrine_1')
        assert not env.world.architecture.active()
        use(env,heart.entities['altar']);solve(env,heart)
        c.damage_guard(enemy,1e9,Element.NONE)
        assert enemy.hp==80 and heart.phase==1 and not heart.solved
        c.damage_guard(enemy,1e9,Element.NONE)
        assert enemy.hp==80
        solve(env,heart);c.damage_guard(enemy,1e9,Element.NONE)
        assert not enemy.alive and c.verify()['success']
        assert c.metrics['boss_windows']==2


def test_desert_permission_and_charge_gates_and_recovery():
    with make('desert') as env:
        env.reset(seed=7);c=env.campaign;s=c.sites['temple_2']
        use(env,'slate:tool');use(env,'calibration:console');use(env,s.entities['archive'])
        assert c.inventory['clearance_rank']==1 and 'rank_two' not in c.facts
        assert 'temple_2_record' not in c.facts
        for sid in ('temple_2','temple_0','temple_1'):
            site=c.sites[sid]
            if c.slate_energy<1: use(env,site.entities['recharge'])
            use(env,site.entities['altar']);solve(env,site)
            c.damage_guard(env.world.get(site.entities['guard']),1000,Element.NONE)
            use(env,site.entities['altar'])
        use(env,'calibration:console')
        assert c.inventory['clearance_rank']==2
        c.slate_energy=1;use(env,s.entities['archive'])
        assert 'temple_2_record' not in c.facts and c.slate_energy==1
        use(env,s.entities['recharge']);use(env,s.entities['archive']);use(env,s.entities['archive'])
        assert c.slate_energy==4 and c.inventory['temple_2_record']==1
        assert 'final' not in c.facts


def test_final_door_never_overlaps_its_own_unlocking_receivers():
    with make('desert') as env:
        env.reset(seed=7)
        site=env.campaign.sites['zenith']
        for role in ('altar','mural','ember','tide','gale','guard'):
            point=env.world.get(site.entities[role]).pos+(0,1,0)
            assert not env.world.architecture.obstacles().blocked([point],margin=.6)[0]


def test_shared_adapter_and_gym_registration_have_no_oracle_goal():
    from owb import make_env
    with make_env('teyvat',task='summit',variant='small',image_size=(16,12),max_steps=4) as env:
        obs,info=env.reset(seed=7)
        assert env.observation_space.contains(obs)
        assert GOALS['summit'] in obs['instruction'] and 'Pack:' in obs['instruction']
        assert not obs['goal'].any()
    with gym.make('OpenWorldBench/Teyvat-Desert-v1',render_enabled=False,geography_scale=.5,width=16,height=12) as env:
        obs,info=env.reset(seed=7)
        assert env.observation_space.contains(obs)
        assert not info['success']


def test_cycles_snapshot_has_actual_seals_and_does_not_change_simulation():
    from owb.render.scene_export import capture_scene
    with make('summit') as env:
        env.reset(seed=7);before=env.state_digest()
        scene=capture_scene(env,width=320,height=180)
        assert env.state_digest()==before
        assert scene.metadata['campaign']['task']=='summit'
        assert scene.metadata['campaign']['active_seals']
        for mesh in scene.meshes:
            assert np.isfinite(mesh['triangles']).all()


def test_terminal_reward_is_sparse_and_cannot_be_collected_twice():
    with make('cleansing') as env:
        env.reset(seed=7);c=env.campaign
        use(env,'lens:tool')
        for i in range(5): shrine_fixture(env,f'shrine_{i}')
        _,reward,done,truncated,_=env.step(CampaignReferencePolicy.neutral())
        assert reward==0 and not done and not truncated
        heart=c.sites['heart'];use(env,heart.entities['altar'])
        for _ in range(2):
            solve(env,heart)
            c.damage_guard(env.world.get(heart.entities['guard']),1000,Element.NONE)
        _,reward,done,truncated,info=env.step(CampaignReferencePolicy.neutral())
        assert reward==1 and done and not truncated and info['success']
        assert c.verify()['success']
        with pytest.raises(gym.error.ResetNeeded): env.step(CampaignReferencePolicy.neutral())
        c.inventory['shrine_0_ward']=1
        assert 'ward accounting mismatch: shrine_0' in c.verify()['failures']


def test_step_limit_is_unsuccessful_and_requires_reset():
    with TeyvatCampaignEnv('summit',render_enabled=False,geography_scale=.5,
                          width=16,height=12,max_steps=1) as env:
        env.reset(seed=7)
        _,reward,done,truncated,info=env.step(CampaignReferencePolicy.neutral())
        assert reward==0 and not done and truncated and not info['success']
        with pytest.raises(gym.error.ResetNeeded): env.step(CampaignReferencePolicy.neutral())


def test_state_digest_includes_continuous_cooldowns_and_elemental_status():
    with make('cleansing') as env:
        env.reset(seed=7);before=env.state_digest()
        env.party.members[0].skill_cd=.123
        assert env.state_digest()!=before
        env.party.members[0].skill_cd=0.0
        assert env.state_digest()==before
        enemy=env.world.get(env.campaign.sites['shrine_0'].entities['guard'])
        enemy.elemental.aura.charge+=.125
        assert env.state_digest()!=before


@pytest.mark.parametrize('scale',[.5,2.0])
def test_desert_final_arena_has_walkable_terrain_through_guard_and_archive(scale):
    with TeyvatCampaignEnv('desert',render_enabled=False,geography_scale=scale,
                          width=16,height=12) as env:
        env.reset(seed=7)
        site=env.campaign.sites['zenith']
        # The final puzzle, guardian approach and archive must not lie on the coast.
        for role in ('altar','ember','tide','gale','guard'):
            p=env.world.get(site.entities[role]).pos
            for dz in np.linspace(-3,3,13):
                assert env.world.terrain.slope_at(p[0],p[2]+dz)<np.radians(env.cfg.character.max_walk_slope_deg)
        p=env.world.get('final:atlas').pos
        assert p[1]>env.world.terrain.cfg.sea_level+2
