"""Working-memory probes WM-SA12, WM-SB10 and WM-SB17 (docs/working-memory-probes.md)."""
import json

import numpy as np
import pytest

from owb.bench.baselines.skylands_model import local_view
from owb.bench.baselines import privileged_view
from owb.bench.baselines.skylands_reference import neutral
from owb.core import blocks as B
from owb.env.skylands_reference import SkylandsReferenceEnv, WM_CUE_SECONDS
from owb.env.skylands_tempest import SkylandsTempestEnv
from owb.scripts.skylands_reference import SkylandsWMReferencePolicy


class NoProbes(SkylandsReferenceEnv):
    wm_probes_enabled = False


def code(p, items=None):
    return '-'.join(str(i) for i in (p.items if items is None else items))


def act(e, name, target=0, option=0):
    a = neutral(); a['command'] = e.commands.index(name); a['target'] = int(target); a['option'] = int(option)
    return e.step(a)


def stand_at(e, tid):
    """Place the agent on walkable ground within inspect reach of a fixture."""
    p = np.asarray(e.targets[tid]['pos'], float); y = int(round(p[1] - 1))
    best = None
    for dx in range(-3, 4):
        for dz in range(-3, 4):
            x, z = int(np.floor(p[0])) + dx, int(np.floor(p[2])) + dz
            if not (B.SOLID[e.world.get(x, y - 1, z)] and not B.SOLID[e.world.get(x, y, z)]
                    and not B.SOLID[e.world.get(x, y + 1, z)]):
                continue
            feet = np.array((x + .5, y + .01, z + .5))
            eye = feet + (0, e.cfg.sim_cfg.eye_height, 0)
            d = np.linalg.norm(eye - p)
            if d < 3.5 and e.world.line_of_sight(eye, p) and (best is None or d < best[0]):
                best = (d, feet)
    e.agent.pos = best[1].astype(np.float32); e.agent.vel[:] = 0
    act(e, 'none')
    assert e._reachable(p, 4.5)


@pytest.fixture
def env_b():
    with SkylandsReferenceEnv('b', 7) as e:
        e.reset(seed=7)
        yield e


def test_probes_are_registered_with_contract_fields_and_layout():
    for task, probes in (('a', {'WM-SA12'}), ('b', {'WM-SB10', 'WM-SB17'})):
        with SkylandsReferenceEnv(task, 7) as e:
            e.reset(seed=7)
            summary = e.wm_summary()
            assert set(summary) == probes
            for probe, s in summary.items():
                assert {'milestone', 'kind', 'load', 'solved', 'attempts', 'first_attempt_correct',
                        'cue_reads', 'first_attempt_delay', 'solve_delay'} <= set(s)
                assert s['load'] == 4 and not s['line_of_sight']
                assert np.linalg.norm(np.subtract(s['source_pos'], s['use_pos'])) > 30
                # Visible labels name the fixture, never its content.
                for tid in (s['source_target'], s['use_target']):
                    assert code(e.wm[probe]) not in json.dumps(e.targets[tid], default=str)
            assert e.wm_summary()['WM-SB17' if task == 'b' else 'WM-SA12']['kind'] == (
                'reverse' if task == 'b' else 'hold')


def test_cue_shows_only_after_inspecting_the_source_and_clears(env_b):
    e = env_b; p = e.wm['WM-SB10']; src = e.wm_sites['WM-SB10']['source']
    assert code(p) not in e._observation()['text']
    stand_at(e, src)
    assert code(p) not in e._observation()['text']
    obs, _, _, _, info = act(e, 'inspect', src)
    assert code(p) in obs['text'] and p.cue_reads == 1
    assert [x for x in info['events'] if x['type'] == 'wm_cue_read'] == [dict(type='wm_cue_read', probe='WM-SB10', read_index=1)]
    assert code(p) not in e.last_feedback
    start = e._wm_showing['since']      # simulated time when the inspect decision began
    while e.elapsed - start < WM_CUE_SECONDS - 1e-6:
        assert code(p) in obs['text']
        obs, *_ = act(e, 'none')
    assert code(p) not in obs['text'] and e.last_text == ''
    # Leaving inspect reach clears it at once.
    act(e, 'inspect', src)
    e.agent.pos += np.array((0, 0, 12), np.float32)
    obs, *_ = act(e, 'none')
    assert code(p) not in obs['text']


def test_wrong_complete_entry_resets_and_is_recoverable(env_b):
    e = env_b; p = e.wm['WM-SB10']; keypad = e.wm_sites['WM-SB10']['use']
    lo, hi = e.wm_gate
    stand_at(e, keypad)
    wrong = [(k + 1) % 4 for k in p.expected]
    events = []
    for k in wrong:
        events += act(e, 'use', keypad, k)[4]['events']
    wm = [x for x in events if x['type'].startswith('wm_')]
    assert [x.get('count') for x in wm if x['type'] == 'wm_entry'] == [1, 2, 3, 4]
    assert wm[-1]['type'] == 'wm_attempt' and wm[-1]['correct'] is False
    assert not p.solved and p.buffer == [] and p.attempts == 1
    assert e.world.get(*lo) == B.GATE and e.world.get(*hi) == B.GATE
    control = e.by_id['B10'].state['control']
    assert any(x['type'] == 'action_failed' for x in act(e, 'take', control)[4]['events'])
    assert e.cargo['marsh_coupling']['anchored']
    for k in p.expected:
        info = act(e, 'use', keypad, k)[4]
    assert p.solved and p.attempts == 2 and p.first_attempt_correct is False
    assert any(x['type'] == 'yard_gate_opened' for x in info['events'])
    assert e.world.get(*lo) == B.AIR and e.world.get(*hi) == B.AIR
    act(e, 'take', control)
    assert not e.cargo['marsh_coupling']['anchored']


def test_entries_give_no_per_item_correctness():
    seen = []
    for first in ('right', 'wrong'):
        with SkylandsReferenceEnv('b', 7) as e:
            e.reset(seed=7); p = e.wm['WM-SB17']; panel = e.wm_sites['WM-SB17']['use']
            stand_at(e, panel)
            key = p.expected[0] if first == 'right' else (p.expected[0] + 1) % 4
            obs, reward, _, _, info = act(e, 'use', panel, key)
            seen.append((obs['text'], reward, [x for x in info['events'] if x['type'] != 'land'], e.last_feedback))
    assert seen[0] == seen[1]
    assert seen[0][2] == [dict(type='wm_entry', probe='WM-SB17', count=1)]


def test_reference_keys_from_memory_and_cue_is_absent_at_the_use_site():
    with SkylandsReferenceEnv('a', 7) as e:
        obs, _ = e.reset(seed=7); policy = SkylandsWMReferencePolicy(e); p = e.wm['WM-SA12']
        checked = False
        while not e._ended:
            obs, _, _, _, info = e.step(policy(obs))
            if any(x['type'] == 'wm_entry' for x in info['events']) and not checked:
                checked = True
                assert code(p) not in obs['text'] and code(p) not in e.last_feedback
                views = (local_view(e), privileged_view.skylands(e, info['events']))
                for view in views:
                    assert code(p) not in json.dumps(view, default=str)
                    assert 'items' not in json.dumps(view, default=str)
                src = e.targets[e.wm_sites["WM-SA12"]["source"]]["pos"]
                assert np.linalg.norm(src - e.agent.eye) > 30 and not e.world.line_of_sight(e.agent.eye, src)
            if any(x['type'] == 'freight_held' for x in info['events']):
                assert not e.by_id['A12'].state['done'] and e.cargo['core']['parent'] == 'freight'
        assert checked and e.success
        s = e.wm_summary()['WM-SA12']
        assert s['first_attempt_correct'] and s['cue_reads'] == 1
        assert 10 <= s['first_attempt_delay']['seconds'] <= 60


def test_unpatched_station_does_not_output(env_b):
    e = env_b; s = e.by_id['B17'].state
    s['installed'] = True
    for _ in range(40): e._task_outcome([])
    assert s['stable'] == 0 and 'B17' not in e.online
    e.wm['WM-SB17'].solved = True
    for _ in range(26): e._task_outcome([])
    assert s['stable'] >= 5 and 'B17' in e.online


def test_matched_control_posts_the_cue_at_the_use_site():
    for control in (False, True):
        with SkylandsReferenceEnv('b', 7, wm_control=control) as e:
            e.reset(seed=7); p = e.wm['WM-SB10']; keypad = e.wm_sites['WM-SB10']['use']
            stand_at(e, keypad)
            obs, _, _, _, info = act(e, 'inspect', keypad)
            assert (code(p) in obs['text']) is control
            assert p.cue_reads == int(control) and e.wm_summary()['WM-SB10']['control'] is control
            assert any(x['type'] == 'wm_cue_read' for x in info['events']) is control


def test_content_and_layout_are_deterministic_across_resets():
    with SkylandsReferenceEnv('b', 7) as e:
        e.reset(seed=7); first = (e.wm_state(), e.wm_summary(), e.state_digest())
        e.reset(seed=7); second = (e.wm_state(), e.wm_summary(), e.state_digest())
        assert json.dumps(first, default=str) == json.dumps(second, default=str)
        items = {seed: (e.reset(seed=seed), e.wm['WM-SB10'].items)[1] for seed in range(7, 13)}
        assert len(set(items.values())) > 1


def test_probe_state_is_part_of_the_replay_digest(env_b):
    e = env_b; panel = e.wm_sites['WM-SB17']['use']
    stand_at(e, panel)
    digest = e.state_digest()
    e.wm['WM-SB17'].buffer.append(0)
    assert e.state_digest() != digest


@pytest.mark.parametrize('task', ['a', 'b'])
def test_simulation_rng_and_non_wm_state_are_unchanged(task):
    with SkylandsReferenceEnv(task, 7) as on, NoProbes(task, 7) as off:
        on.reset(seed=7); off.reset(seed=7)
        assert off.wm == {} and 'working_memory' not in off._info([])
        base = len(off.targets)
        assert all(on.targets[k]['role'] == off.targets[k]['role'] for k in range(1, base + 1))
        assert all(on.targets[k].get('wm') for k in range(base + 1, len(on.targets) + 1))
        differs = np.argwhere(on.world.vox != off.world.vox)
        if task == 'a':
            assert not len(differs)
        else:
            lo, hi = on.wm_yard
            assert ((differs >= lo) & (differs <= hi)).all()
        for e in (on, off):
            a = neutral(); a['move'][:] = (1, .3)
            for _ in range(30): e.step(a)
        assert on.np_random.bit_generator.state == off.np_random.bit_generator.state
        np.testing.assert_array_equal(on.agent.pos, off.agent.pos)
        assert on.calibration == off.calibration and on.destination == off.destination
        for cid in off.cargo: np.testing.assert_array_equal(on.cargo[cid]['pos'], off.cargo[cid]['pos'])


def test_published_tempest_courses_keep_probes_off():
    assert SkylandsTempestEnv.wm_probes_enabled is False
