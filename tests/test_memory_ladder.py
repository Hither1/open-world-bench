"""Memory-ladder conditions (perfect recall, goal location) and evaluator-only search metrics."""
import json

import numpy as np

from owb.bench import memory_ladder
from owb.env.wildlands_reference import Y, WildlandsReferenceEnv
from owb.scripts import wildlands_model as W
from owb.scripts.checkpoint_report import ladder
from test_wildlands_model import Client, payload


def _prompt(client, k=0):
    return json.loads(client.requests[k]['input'][0]['content'][0]['text'])


def test_ledger_keeps_last_seen_positions_and_separates_the_prefix():
    with WildlandsReferenceEnv('a', 7, render_enabled=False) as env:
        env.reset(seed=7)
        ledger = memory_ladder.Ledger('wildlands')
        before = ledger.look(env, 0)                       # the earlier operator's frame
        assert before
        env.last_text = 'Read during the prefix.'
        ledger.read(env, prefix=True)
        env.agent.yaw += np.pi; env._frame = None
        env.last_text = 'Read by the agent.'
        now = ledger.look(env, 1)
        only_prefix = before - now                         # the work order beside the start stays in frame
        assert only_prefix
        own, full = ledger.view(include_prefix=False), ledger.view(include_prefix=True)
        assert {e['id'] for e in own['objects'] if 'id' in e} == now
        assert {e['id'] for e in full['objects'] if 'id' in e} == before | now
        assert own['texts'] == ['Read by the agent.']
        assert full['texts'] == ['Read during the prefix.', 'Read by the agent.']
        prefix = next(e for e in full['objects'] if e.get('id') in only_prefix)
        assert prefix['first_seen_decision'] == prefix['last_seen_decision'] == 'before takeover'
        # Positions only: nothing relative to a pose the agent no longer has, and no roles.
        text = json.dumps(full)
        for absent in ('distance_m', 'bearing_deg', 'brake', 'release', 'milestone'):
            assert absent not in text


def test_goal_location_condition_supplies_recall_and_site_only(tmp_path, monkeypatch):
    monkeypatch.setattr(WildlandsReferenceEnv, 'render', lambda _: np.zeros((180, 320, 3), np.uint8))
    _, goal = W.ladder(W.WILDLANDS_PERCEPTION)
    client = Client([payload(command='none')])
    W.run('a', tmp_path / 'goal', client, max_calls=1, transport='test', course=goal)
    observation = _prompt(client)['observation']
    with WildlandsReferenceEnv('a', 7, render_enabled=False) as env:
        env.reset(seed=7)
        x, z = env.current.site
    assert observation['objective_location'] == {'position': [x, Y, z]}
    assert observation['recalled']['objects'] and 'perceived' in observation
    guide = client.requests[0]['instructions']
    assert 'PERFECT-RECALL' in guide and 'GOAL-LOCATION' in guide and 'PERCEPTION-ORACLE' in guide
    config = json.loads((tmp_path / 'goal' / 'config.json').read_text())
    assert config['information_condition'] == 'goal_location' and not config['observation_only']
    assert json.loads((tmp_path / 'goal' / 'verification.json').read_text())['replay_verified']
    # Search metrics are written for the evaluator and never reach the model.
    assert (tmp_path / 'goal' / 'search.json').exists()
    assert 'seen_by_agent' not in json.dumps(client.requests)


def test_native_condition_gets_no_ladder_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(WildlandsReferenceEnv, 'render', lambda _: np.zeros((180, 320, 3), np.uint8))
    client = Client([payload(command='none')])
    W.run('a', tmp_path / 'native', client, max_calls=1, transport='test')
    observation = _prompt(client)['observation']
    assert not {'recalled', 'objective_location', 'perceived'} & set(observation)
    assert 'PERFECT-RECALL' not in client.requests[0]['instructions']


def test_search_tracker_splits_search_from_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(WildlandsReferenceEnv, 'render', lambda _: np.zeros((180, 320, 3), np.uint8))
    # A01 is the work order beside the start: inspecting it completes the milestone.
    client = Client([payload(command='inspect', target=1), payload(command='none', move=[1, 0], repeat=10)])
    result = W.run('a', tmp_path / 'run', client, max_calls=2, transport='test')
    segments = json.loads((tmp_path / 'run' / 'search.json').read_text())['segments']
    first, second = segments
    assert first['milestone'] == 'A01' and first['completed'] and first['search_actions'] == 0
    assert first['execution_actions'] == 1 and first['first_targeted'] == dict(command='inspect', on_objective=True, step=1)
    assert first['first_in_frame_decision'] == 1 and first['kind'] == 'unseen'
    assert second['milestone'] == 'A02' and not second['completed'] and second['actions'] == 10
    assert second['path_m'] > 0 and second['start_decision'] == 1
    assert result['search']['segments'] == 2 and result['search']['completed'] == 1


def test_report_ladder_gaps_compare_matched_starts_only():
    def group(condition, rate, starts=(2, 5, 8)):
        return dict(world='wildlands', task='a', model='m', memory='empty', condition=condition, starts=len(starts),
                    next_milestone_rate=rate, reach_rate=1.0, start_points=list(starts))
    text = ladder([group('perception_oracle', .25), group('perfect_recall', .5), group('goal_location', .75, (2, 5))])
    row = text.strip().splitlines()[-1]
    assert row.endswith('| +0.25 | – |')
