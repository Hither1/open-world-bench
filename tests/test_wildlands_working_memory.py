"""Working-memory probes in the compact Wildlands courses (docs/working-memory-probes.md).

Agent positions are set directly only to unit-test fixtures; recorded reference
traces never do this.
"""
import json

import numpy as np
import pytest

from owb.core.working_memory import WorkingMemoryProbe
from owb.env import wildlands_reference as W
from owb.env.wildlands_reference import WildlandsReferenceEnv, COMMANDS
from owb.bench.baselines.wildlands_reference import neutral
from owb.bench.baselines.wildlands_model import describe


def make(task, **kw):
    e = WildlandsReferenceEnv(task, 7, **kw)
    obs, _ = e.reset(seed=7)
    return e, obs


@pytest.fixture
def env_a():
    e, _ = make('a')
    yield e
    e.close()


@pytest.fixture
def env_b():
    e, _ = make('b')
    yield e
    e.close()


def wm_targets(e, probe, fixture):
    return [o for o in e.targets.values() if o.get('wm') == probe and o['fixture'] == fixture]


def stand(e, pos, offset=(-1.5, -1, 0)):
    e.agent.pos = np.asarray(pos, np.float32) + np.asarray(offset, np.float32)
    e.agent.vel[:] = 0


def act(e, command, target=0):
    a = neutral(); a.update(command=COMMANDS.index(command), target=target)
    return e.step(a)


def prerequisites(e, mid):
    e.completed = [dict(id=k) for k in e.by_id[mid].prereqs]


def key(e, probe, value):
    return next(o['id'] for o in e.targets.values()
                if o.get('wm') == probe and o['fixture'] in ('key', 'dial') and o['value'] == value)


# --- content, determinism and the simulation RNG -----------------------------------------

def test_content_is_deterministic_per_seed_and_drawn_from_the_probe_generator():
    for task in 'ab':
        a, _ = make(task); b, _ = make(task)
        assert a.working_memory_layout() == b.working_memory_layout()
        for k, p in a.wm.items():
            fresh = WorkingMemoryProbe.draw(k, p.milestone, p.kind, 7, W.PITCHES if k == 'WM-WB15' else
                                            (W.KEYPAD if k == 'WM-WA07' else range(1, 7)), p.load)
            assert p.items == fresh.items
        a.close(); b.close()
    c = WildlandsReferenceEnv('b', 8); c.reset(seed=8)
    d, _ = make('b')
    assert c.working_memory_layout() != d.working_memory_layout()
    c.close(); d.close()


def test_remap_is_never_the_board_order(env_b):
    p = env_b.wm['WM-WB15']
    blades = env_b.wm_world['blade_colours']
    assert blades != list(W.BLADE_COLOURS)
    assert p.expected == tuple(p.items[W.BLADE_COLOURS.index(c)] for c in blades)


def test_probes_never_consume_the_simulation_rng(env_b):
    rng = json.dumps(env_b.np_random.bit_generator.state, default=str)
    np.random.seed(3); legacy = np.random.get_state()[1].copy()
    env_b._setup_working_memory()
    assert json.dumps(env_b.np_random.bit_generator.state, default=str) == rng
    assert (np.random.get_state()[1] == legacy).all()
    # The diagnostic control adds fixtures but leaves every simulation stream and
    # all dynamics identical under the same actions.
    plain, _ = make('a'); control, _ = make('a', wm_control=True)
    assert plain.wm['WM-WA07'].items == control.wm['WM-WA07'].items
    rng = np.random.default_rng(0)
    for _ in range(40):
        a = neutral(); a['move'][:] = rng.uniform(-1, 1, 2); a['look'][0] = rng.uniform(-1, 1)
        plain.step(a); control.step(a)
    assert plain.np_random.bit_generator.state == control.np_random.bit_generator.state
    assert plain.agent.pos.tolist() == control.agent.pos.tolist()
    plain.close(); control.close()


# --- cue visibility --------------------------------------------------------------------------

@pytest.mark.parametrize('task,probe', [('a', 'WM-WA07'), ('b', 'WM-WB11'), ('b', 'WM-WB15')])
def test_use_site_is_out_of_sight_of_the_source(task, probe):
    e, _ = make(task)
    source = np.asarray(wm_targets(e, probe, 'source')[0]['pos'], float)
    use = [o for o in e.targets.values() if o.get('wm') == probe and o['fixture'] in ('key', 'bay', 'dial', 'blade')]
    # From the board itself and from every standing spot within reading reach of it.
    eyes = [source]
    for dx in np.arange(-3, 3.01, .5):
        for dz in np.arange(-3, 3.01, .5):
            stand(e, source, (dx, -1, dz))
            if e._reachable(source) and not e.world.is_solid(*np.floor(e.agent.pos + (0, .1, 0)).astype(int)):
                eyes.append(e.agent.eye.astype(float))
    assert len(eyes) > 5
    for eye in eyes:
        for o in use:
            assert not e.world.line_of_sight(eye, np.asarray(o['pos'], float)), (probe, o['symbol'], eye)
    e.close()


def test_cue_shows_only_at_the_source_and_clears_after_five_seconds(env_a):
    e = env_a; cue = e.wm_cue_text('WM-WA07'); code = ' '.join(e.wm['WM-WA07'].items)
    board = wm_targets(e, 'WM-WA07', 'source')[0]
    obs, *_ = act(e, 'inspect', board['id'])
    assert 'out of reach' in obs['text'] and cue not in obs['text']          # read only within reach
    stand(e, board['pos'])
    obs, _, _, _, info = act(e, 'inspect', board['id'])
    assert cue in obs['text'] and 'you read the dispatch board' in obs['text']
    assert [ev for ev in info['events'] if ev['type'] == 'wm_cue_read'] == [
        dict(type='wm_cue_read', probe='WM-WA07', read_index=1)]
    assert code not in json.dumps(info['events'])                             # events carry no content
    shown = 1
    while cue in obs['text']:
        obs, *_ = act(e, 'none'); shown += 1
    assert shown == round(W.WM_CUE_SECONDS / .2)                              # visible for < 5 s
    # Leaving reach clears it at once.
    stand(e, board['pos']); act(e, 'inspect', board['id'])
    stand(e, board['pos'], (-8, -1, 0))
    obs, *_ = act(e, 'none')
    assert cue not in obs['text'] and e.wm_cue is None
    # Visible labels name the board, never its content.
    assert all(code not in o['symbol'] for o in e.targets.values())


@pytest.mark.parametrize('task', 'ab')
def test_cue_never_reaches_policy_observations_at_the_use_site(task):
    """Drive the reference controller; once the source is out of reach the cue
    text is absent from every policy observation, including the model-runner view."""
    from owb.scripts.wildlands_reference import WorkingMemoryReferencePolicy
    e, obs = make(task); policy = WorkingMemoryReferencePolicy(e)
    cues = {k: e.wm_cue_text(k) for k in e.wm}
    seen_at_use = set()
    while not all(p.solved for p in e.wm.values()):
        obs, *_ = e.step(policy(obs))
        text = json.dumps(describe(obs))
        for k, cue in cues.items():
            source = wm_targets(e, k, 'source')[0]
            if not e._reachable(source['pos']):
                assert cue not in obs['text'] and cue not in text
            use = [o for o in e.targets.values() if o.get('wm') == k and o['fixture'] in ('key', 'bay', 'dial')]
            if any(e._reachable(o['pos']) for o in use):
                seen_at_use.add(k)
                assert cue not in obs['text']
    assert seen_at_use == set(e.wm)
    for k, p in e.wm.items():
        s = e.working_memory_summary()[k]
        assert s['solved'] and s['first_attempt_correct'] and s['cue_reads'] == 1
        assert 10 <= s['first_attempt_delay']['seconds'] <= 60, (k, s)
    e.close()


# --- entries, recovery and feedback ----------------------------------------------------------

def test_keypad_wrong_entry_is_recoverable_without_per_item_feedback(env_a):
    e = env_a; p = e.wm['WM-WA07']
    prerequisites(e, 'A07'); e.satchel['salvage_kit'] = 1
    stand(e, W.KEYPAD_STANCE, (0, 0, 0))                       # inside the stores warehouse
    locker = e.by_id['A07'].targets['interact']
    obs, _, _, _, info = act(e, 'interact', locker)
    assert 'the locker is locked' in obs['text'] and not e.by_id['A07'].state['done']
    wrong = [d for d in W.KEYPAD if d not in p.items][:4]
    # The first key is right in one entry and wrong in the other; what the player
    # sees (and every event) is the same apart from the key's own label.
    lines = []
    for first in (p.items[0], wrong[0]):
        obs, _, _, _, info = act(e, 'operate', key(e, 'WM-WA07', first))
        lines.append(obs['text'].split('->')[1])
        assert [ev for ev in info['events'] if ev['type'].startswith('wm_')] == [
            dict(type='wm_entry', probe='WM-WA07', count=len(p.buffer))]
        p.clear()
    assert lines[0] == lines[1]
    for d in wrong:
        obs, _, _, _, info = act(e, 'operate', key(e, 'WM-WA07', d))
    attempt = [ev for ev in info['events'] if ev['type'] == 'wm_attempt'][0]
    assert attempt['correct'] is False and p.buffer == [] and not p.solved
    assert 'buzzes and clears' in obs['text']
    act(e, 'interact', locker)
    assert not e.by_id['A07'].state['done']
    for d in p.items: obs, *_ = act(e, 'operate', key(e, 'WM-WA07', d))
    assert p.solved and 'locker bolt slides back' in obs['text']
    act(e, 'interact', locker)
    assert e.by_id['A07'].state['done'] and e.satchel['regulator'] == 1
    s = e.working_memory_summary()['WM-WA07']
    assert s['attempts'] == 2 and s['first_attempt_correct'] is False and s['solved']


def test_partial_entry_clears_when_walking_away(env_a):
    e = env_a; p = e.wm['WM-WA07']
    prerequisites(e, 'A07'); stand(e, W.KEYPAD_STANCE, (0, 0, 0))
    act(e, 'operate', key(e, 'WM-WA07', p.items[0]))
    assert p.buffer
    stand(e, (110.5, W.STORES_FLOOR, 61.5), (0, 0, 0)); act(e, 'none')   # two aisles away
    assert p.buffer == []


def test_wrong_bay_is_empty_and_recoverable(env_b):
    e = env_b; p = e.wm['WM-WB11']; m = e.by_id['B11']
    prerequisites(e, 'B11'); e.satchel['authorization'] = 1
    pump, engine = p.expected
    bay = lambda k: m.targets[f'bay{k}']
    stand(e, e.targets[bay(pump)]['pos'], (0, -1, 1))
    obs, *_ = act(e, 'interact', bay(pump))
    assert 'has not released it' in obs['text'] and not e.satchel['pump_coupling']
    stand(e, e.targets[m.targets['interact']]['pos'])
    obs, _, _, _, info = act(e, 'interact', m.targets['interact'])
    assert e.wm_world['bays_released'] and not any(ev['type'] == 'milestone_complete' for ev in info['events'])
    wrong = next(k for k in range(1, 7) if k not in p.expected)
    stand(e, e.targets[bay(wrong)]['pos'], (0, -1, 1))
    obs, _, _, _, info = act(e, 'interact', bay(wrong))
    assert f'bay {wrong} swings open: it is empty' in obs['text']
    assert [ev['correct'] for ev in info['events'] if ev['type'] == 'wm_attempt'] == [False]
    assert e.world.get(W.BAY_X[wrong - 1], 29, W.BAY_STORE[3]) == 0          # the door really opened
    obs, _, _, _, info = act(e, 'interact', bay(wrong))                       # looking again is not an attempt
    assert 'stands open and empty' in obs['text'] and p.attempts == 1
    for k in (engine, pump):
        stand(e, e.targets[bay(k)]['pos'], (0, -1, 1))
        obs, _, _, _, info = act(e, 'interact', bay(k))
    assert e.satchel['pump_coupling'] == 1 and e.satchel['engine_coupling'] == 1
    assert p.solved and any(ev['type'] == 'milestone_complete' and ev['milestone'] == 'B11' for ev in info['events'])
    s = e.working_memory_summary()['WM-WB11']
    assert s['first_attempt_correct'] is False and s['attempts'] == 3


def test_dial_remap_wrong_entry_resets_and_station_waits_for_pitch(env_b):
    e = env_b; p = e.wm['WM-WB15']; m = e.by_id['B15']
    prerequisites(e, 'B15'); e.satchel['wind_coupling'] = 1
    stand(e, (90.5, W.Y + 4, 96.5), (0, 0, 0))
    act(e, 'interact', m.targets['interact'])
    assert m.state['installed']
    for _ in range(30): act(e, 'none')
    assert m.state['elapsed'] == 0 and 'B15' not in {r['id'] for r in e.completed}
    obs, _, _, _, info = act(e, 'inspect', m.targets['blade1'])
    assert f"Blade 1 carries a {e.wm_world['blade_colours'][0]} tag" in obs['text']
    for item in p.items:                     # the board's colour order, not the blade order
        obs, _, _, _, info = act(e, 'operate', key(e, 'WM-WB15', item))
    assert not p.solved and p.buffer == [] and 'springs back' in obs['text']
    for item in p.expected:
        obs, *_ = act(e, 'operate', key(e, 'WM-WB15', item))
    assert p.solved and 'rotor steadies' in obs['text']
    for _ in range(26): obs, _, _, _, info = act(e, 'none')
    assert 'B15' in {r['id'] for r in e.completed}


def test_wm_control_posts_the_cue_at_the_use_site():
    for task in 'ab':
        plain, _ = make(task)
        assert not any(o.get('fixture') == 'copy' for o in plain.targets.values())
        plain.close()
    e, _ = make('a', wm_control=True)
    copy = wm_targets(e, 'WM-WA07', 'copy')[0]
    keypad = wm_targets(e, 'WM-WA07', 'key')
    stand(e, W.KEYPAD_STANCE, (0, 0, 0))
    assert e._reachable(copy['pos']) and all(e._reachable(o['pos']) for o in keypad)
    obs, *_ = act(e, 'inspect', copy['id'])
    assert e.wm_cue_text('WM-WA07') in obs['text']
    s = e.working_memory_summary()['WM-WA07']
    assert s['control'] is True and s['cue_reads'] == 1
    e.close()
    e, _ = make('b', wm_control=True)
    for probe, use in (('WM-WB11', 'bay'), ('WM-WB15', 'dial')):
        copy = wm_targets(e, probe, 'copy')[0]
        near = min(np.linalg.norm(np.asarray(o['pos']) - copy['pos']) for o in wm_targets(e, probe, use))
        assert near < 6, (probe, near)
    e.close()


def test_probe_state_is_in_the_digest_and_summary_is_evaluator_only(env_a):
    e = env_a; before = e.state_digest()
    board = wm_targets(e, 'WM-WA07', 'source')[0]
    stand(e, board['pos']); obs, _, _, _, info = act(e, 'inspect', board['id'])
    assert info['working_memory']['WM-WA07']['cue_reads'] == 1
    after = e.state_digest()
    e.wm['WM-WA07'].cue_reads += 1
    assert e.state_digest() != after != before
    assert 'working_memory' not in describe(obs) and 'cue_reads' not in obs['text']


# --- the stores warehouse keeps WM-WA07 above ten seconds for any traversal ------------------

def test_horse_cannot_reach_the_stores_floor(env_a):
    """The doorway sits on a three-block dock: riders cannot climb, a jump cannot
    clear it, and the horse's own route planner finds no way up."""
    from owb.sim.riding import walking_route
    e = env_a; dx, dz = W.STORES_DOOR; F = int(W.STORES_FLOOR)
    assert all(e.world.is_solid(dx + 1 - 1, y, dz) for y in range(29, F))       # dock face under the door
    assert not e.world.is_solid(dx, F, dz) and not e.world.is_solid(dx, F + 1, dz)
    apex = e.cfg.sim_cfg.jump_speed ** 2 / (2 * -e.cfg.sim_cfg.gravity)
    assert apex + e.cfg.sim_cfg.step_height < F - W.Y
    assert walking_route(e.world, np.array((dx + 1.5, W.Y, dz + .5)), np.array((dx + .5, F, dz + .5))) is None


def test_fastest_legal_traversal_stays_above_ten_seconds(env_a):
    """Lower bound: mount and gallop (or sprint) straight from the edge of reading
    reach to the dock, the fastest jump-then-climb, the taut shortest path of the
    agent's body through the aisles at sprint speed, then four key presses."""
    import math
    from owb.sim.riding import GALLOP
    e = env_a; c = e.cfg.sim_cfg; dt = .2; r = c.agent_radius; dx, dz = W.STORES_DOOR
    eye = float(e.agent.eye[1] - e.agent.pos[1])
    board = np.asarray(W.WA07_BOARD, float)
    outdoor = np.linalg.norm(board[[0, 2]] - (dx + 1 + r, dz + .65)) - math.sqrt(3.2 ** 2 - (board[1] - W.Y - eye) ** 2)
    out = min(1 + math.ceil(outdoor / (c.walk_speed * GALLOP * dt)), math.ceil(outdoor / (c.run_speed * dt)))
    g = -c.gravity; v = c.jump_speed
    climb = math.floor(((v - c.climb_speed) / g + (W.STORES_FLOOR - W.Y - (v ** 2 - c.climb_speed ** 2) / (2 * g)) / c.climb_speed) / dt)
    keys = [np.asarray(o['pos'], float) for o in e.targets.values() if o.get('fixture') == 'key']
    reach = max(k[0] for k in keys) - math.sqrt(3.2 ** 2 - min(abs(W.STORES_FLOOR + eye - k[1]) for k in keys) ** 2)
    lo, hi = 105 + r, 120 - r
    pts = [(dx + 1, 57.5), (lo + .3, 58 - r), (lo + .3, 59 + r), (hi - .3, 60 - r), (hi - .3, 61 + r),
           (lo + .3, 62 - r), (lo + .3, 63 + r), (reach, 63 + r)]
    aisles = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    # The racks really force that route: each rack row is open only at its gap.
    for rz, gap in W.STORES_RACKS:
        open_x = [x for x in range(105, 120) if not e.world.is_solid(x, int(W.STORES_FLOOR), rz)]
        assert open_x == [gap]
    bound = (out + climb + math.ceil(aisles / (c.run_speed * dt)) + 4) * dt
    assert bound > 10.5, bound
    assert not any('swift' in p['item'] for p in e.gameplay.plants)     # no speed meal on the course
