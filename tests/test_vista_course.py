"""VISTA adapter: prompt adaptation, play contract, evidence boundary and replay."""
import json

import numpy as np
import pytest

from owb.scripts.vista_model import DEFAULT_VISTA, course_for, import_vista, run

if not (DEFAULT_VISTA / 'src' / 'vista').is_dir():
    pytest.skip('VISTA checkout not present in .cache/vista', allow_module_level=True)
import_vista()
from vista.backends.session import BackendResult  # noqa: E402

from owb.bench.baselines.vista_course import make_profile, play_schema, vista_guide  # noqa: E402
from owb.env.wildlands_reference import WildlandsReferenceEnv  # noqa: E402


def action(**changes):
    out = dict(move=[0, 0], look=[0, 0], vector=[0, 0, 0], buttons=[], command='inspect',
               target=1, slot=0, ingredients=[0] * 5, repeat=1)
    out.update(changes)
    return out


class ScriptedBackend:
    """Plays fixed actions through the real VISTA runtime; no model or container."""

    def __init__(self, runtime, actions):
        self.runtime, self.actions, self.usage_reports = runtime, list(actions), []
        self.results = []

    def run(self, task_input):
        self.runtime.begin_turn(task_input)
        self.inputs = getattr(self, 'inputs', []) + [task_input]
        while self.actions:
            call = f'call-{len(self.results)}'
            result = self.runtime.execute('play', self.actions.pop(0), call)
            self.runtime.acknowledge(call)
            self.results.append(result)
            if result.interrupt_after:
                break
        return BackendResult(0, 'done', 'scripted-session')


@pytest.mark.parametrize('world', ['wildlands', 'skylands'])
def test_guide_keeps_interface_and_drops_single_reply_protocol(world):
    course = course_for(world)
    guide = vista_guide(course.policy.guide)
    assert 'Return only JSON' not in guide and 'replacement memory' not in guide
    assert 'six recent observations' not in guide
    assert 'move=[forward,right]' in guide and 'repeat is 1..30' in guide
    system = make_profile(course, backend='claude').prompts.system
    assert 'GUIDE.md' in system
    assert 'memory' not in play_schema(course.policy(None, memory_chars=1))['properties']


def test_scripted_vista_run_is_replayed_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(WildlandsReferenceEnv, 'render', lambda _: np.zeros((180, 320, 3), np.uint8))
    backends = []

    def factory(runtime, directory):
        backends.append(ScriptedBackend(runtime, [
            action(command='inspect', repeat=2),     # invalid: only none/drag repeat
            action(command='inspect'),
            action(command='none', move=[1, 0], repeat=3),
        ]))
        return backends[-1]

    output = tmp_path / 'episode'
    summary = run('a', output, factory, course=course_for('wildlands'), backend='claude',
                  model='test', effort='low', max_calls=3, max_actions=50)
    backend = backends[0]
    assert [r.success for r in backend.results] == [False, True, True]
    assert backend.results[-1].interrupt_after          # decision budget reached
    assert summary['stop_reason'] == 'model_decision_limit'
    assert summary['model_decisions'] == 3 and summary['rejected_actions'] == 1
    assert summary['primitive_actions'] == 4
    assert json.loads((output / 'verification.json').read_text())['replay_verified']
    assert len(list((output / 'observations').glob('*.png'))) == 3
    # The evidence boundary: sensors allowlist only, no evaluator events or milestones.
    delivered = json.dumps(backend.inputs[0].block_manifest()) + ''.join(
        r.text for r in backend.results)
    assert 'Mission:' in delivered and '"proprio"' in delivered
    assert 'milestone_complete' not in delivered and '"milestones"' not in delivered
