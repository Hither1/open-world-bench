"""Safety interlocks and independent replay for compact reference courses."""
from copy import deepcopy

import numpy as np
import pytest

from owb.env.wildlands_reference import WildlandsReferenceEnv, COMMANDS
from owb.bench.baselines.wildlands_reference import neutral


@pytest.fixture
def env():
    with WildlandsReferenceEnv('b', seed=7) as e:
        e.reset(seed=7)
        yield e


def operate(e, mid, role, command='operate'):
    m=e.by_id[mid]; target=m.targets[role]
    # Unit-test fixture positioning only. The recorded oracle never does this.
    e.agent.pos=np.asarray(e.targets[target]['pos'],np.float32)+(-1.5,-1,0)
    e.completed=[dict(id=k) for k in m.prereqs]
    a=neutral(); a.update(command=COMMANDS.index(command),target=target)
    return e._dispatch(command,[],a)


def test_remote_control_cannot_change_mechanism(env):
    m=env.by_id['B02']; before=deepcopy(m.state)
    a=neutral(); a.update(command=COMMANDS.index('operate'),target=m.targets['brake'])
    events=env._dispatch('operate',[],a)
    assert events[0]['type']=='action_failed'
    assert m.state==before


def test_counterweight_needs_supported_mass_and_brake(env):
    m=env.by_id['B02']; s=m.state
    assert operate(env,m.id,'release')[0]['type']=='brake_trip'
    body=env.bodies.get(s['bodies'][0])
    body.pos=np.asarray(s['slots'][0],np.float32); body.resting=True
    body.held=True
    assert not env._valid_load(m)
    body.held=False
    assert env._valid_load(m)
    assert operate(env,m.id,'release')[0]['type']=='brake_trip'
    operate(env,m.id,'brake')
    assert operate(env,m.id,'release')[0]['type']=='counterweight_control'
    body.held=True
    events=[]; env._task_outcome(events)
    assert not s['released'] and any(e['type']=='brake_trip' for e in events)


def test_flow_trip_and_isolation_recovery(env):
    m=env.by_id['B06']; s=m.state
    assert operate(env,m.id,'inlet')[0]['type']=='pressure_trip'
    assert not s['inlet'] and s['stable']==0
    operate(env,m.id,'outlet'+str(s['expected']))
    operate(env,m.id,'inlet')
    assert s['inlet']
    assert operate(env,m.id,'outlet'+str(s['expected']))[0]['type']=='action_failed'
    operate(env,m.id,'inlet')
    operate(env,m.id,'outlet'+str(s['expected']))
    assert not s['inlet'] and not any(s['outlets'])


def test_missing_parts_are_atomic_and_hot_plume_refuses_ignition(env):
    s=env.by_id['B16'].state; before=env.satchel.copy()
    assert operate(env,'B16','install','install')[0]['type']=='action_failed'
    assert not s['installed'] and env.satchel==before
    s=env.by_id['B18'].state; s['vent']=True; s['safe']=False
    assert operate(env,'B18','heat')[0]['type']=='action_failed'
    assert not s['heat']


def test_final_state_is_not_just_historical_milestones(env):
    env.completed=[dict(id=m.id) for m in env.milestones]
    env.drained=True
    assert not env.system_operational()
    env._task_outcome([])
    assert not env.success


def test_render_resume_reuses_only_verified_frames(tmp_path,monkeypatch):
    """A headless recorder check; synthetic test pixels are never published."""
    from pathlib import Path
    from types import SimpleNamespace
    from owb.scripts import wildlands_reference as recorder
    calls=[]
    class DiagnosticOverlay:
        def __init__(self,*args):
            self.renderer=SimpleNamespace(metadata=dict(backend='diagnostic-test'))
        def submit(self,env,row):
            calls.append(row['step'])
            return row['step']
        def result(self,pending):
            return np.zeros((720,1280,3),np.uint8)
        def close(self): pass
    monkeypatch.setattr(recorder,'Overlay',DiagnosticOverlay)
    # The published demo traces predate the working-memory probes; use the current verified trace.
    trace=Path(__file__).resolve().parents[1]/'assets/evaluation/working-memory/wildlands/task-a-seed-7'
    recorder.render_chunk(trace,tmp_path,1,1,checkpoints=True)
    assert calls==[1]
    recorder.render_chunk(trace,tmp_path,1,1,checkpoints=True)
    assert calls==[1]
    frame=next(tmp_path.glob('*/*.png'))
    frame.write_bytes(b'tampered')
    with pytest.raises(RuntimeError,match='Saved frame'):
        recorder.render_chunk(trace,tmp_path,1,1,checkpoints=True)
