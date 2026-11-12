"""Mechanism/gating checks for the demonstration prototype, not memory scores."""
import numpy as np
import pytest

from open_world_bench.character import MoveState
from open_world_bench.elements import Element
from open_world_bench.reference_tasks import ReferenceTaskEnv
from open_world_bench.reference_policy import ReferencePolicy


@pytest.fixture
def env():
    instance=ReferenceTaskEnv("b",render_enabled=False)
    yield instance
    instance.close()


def select(env,mid,kind):
    """Test fixture placement; reference trajectories never use this helper."""
    env.course.index=next(i for i,m in enumerate(env.course.milestones) if m.id==mid)
    env.course.operation_index=next(i for i,o in enumerate(env.course.current.operations) if o.kind==kind)
    return env.course.op


def stand_at(env,eid):
    env.char.pos=env.world.get(eid).pos.copy()+np.array([0,0,.5],np.float32)


def test_observation_contract_and_idle_cannot_complete(env):
    obs,_=env.reset(seed=7)
    assert env.observation_space.contains(obs)
    for _ in range(12):
        obs,_,done,_,_=env.step(ReferencePolicy.neutral())
        assert not done
    assert env.course.completed==[]
    assert not env.course.verify()["success"]


def test_lesson_is_exposure_not_relay_success(env):
    op=env.course.op
    stand_at(env,op.entities[0])
    env.course.interact(env.char)
    assert op.complete
    assert env.course.op.kind=="relay"
    assert env.course.index==0
    assert env.course.completed==[]


def test_relay_interruption_preserves_progress_and_requires_recovery(env):
    op=select(env,"B02","relay")
    s=op.state
    # Casting at the source while overloaded does not prime it.
    stand_at(env,s["roles"]["source"])
    env.course.apply_element(env.char,Element(int(env.course.relay_elements["source"])))
    assert s["progress"]==0
    for role in ["drain","source"]:
        stand_at(env,s["roles"][role])
        env.course.apply_element(env.char,Element(int(env.course.relay_elements[role])))
    assert s["progress"]==1 and s["overloaded"] and s["interrupted"]
    stand_at(env,s["roles"]["core"])
    env.course.interact(env.char)
    assert not op.complete
    for role in ["drain","vane"]:
        stand_at(env,s["roles"][role])
        env.course.apply_element(env.char,Element(int(env.course.relay_elements[role])))
    assert s["progress"]==2 and not s["overloaded"]
    stand_at(env,s["roles"]["core"])
    env.course.interact(env.char)
    assert op.complete and s["active"]


def test_rule_update_is_scoped_to_double_ring_relays(env):
    for mid,expected in [("B19",0),("B22",1)]:
        op=select(env,mid,"relay")
        stand_at(env,op.state["roles"]["source"])
        env.course.apply_element(env.char,Element(int(env.course.relay_elements["source"])))
        assert op.state["progress"]==expected


def test_remote_cast_cannot_change_relay(env):
    op=select(env,"B11","relay")
    env.char.pos[:]=[250,12,250]
    for element in [Element.EMBER,Element.TIDE,Element.GALE]:
        env.course.apply_element(env.char,element)
    env.course.interact(env.char)
    assert op.state["progress"]==0 and not op.complete


def test_sentinel_cannot_be_defeated_by_attacking_its_shield(env):
    op=select(env,"B03","sentinel")
    enemy=env.world.get(op.entities[0])
    for _ in range(30):
        env.course.sentinel_hit(enemy,100,Element.NONE)
    assert enemy.hp==enemy.max_hp and enemy.alive
    env.course.sentinel_hit(enemy,0,Element(int(env.course.guard_elements[0])))
    assert op.state["phase"]=="exposed"
    env.course.sentinel_hit(enemy,100,Element(int(env.course.guard_elements[1])))
    assert op.state["phase"]=="exposed"  # Interrupt is premature.
    env.course.tick(env.char,1.0)
    env.course.sentinel_hit(enemy,0,Element(int(env.course.guard_elements[1])))
    assert op.state["phase"]=="vulnerable"
    env.course.sentinel_hit(enemy,100,Element.NONE)
    assert not enemy.alive


def test_reforming_guard_requires_another_diagnosis(env):
    op=select(env,"B04","sentinel")
    enemy=env.world.get(op.entities[0])
    for expected in ["shield","vulnerable"]:
        env.course.sentinel_hit(enemy,0,Element(int(env.course.guard_elements[0])))
        env.course.tick(env.char,1.0)
        env.course.sentinel_hit(enemy,0,Element(int(env.course.guard_elements[1])))
        assert op.state["phase"]==expected


def test_item_exchange_is_atomic_and_not_repeatable(env):
    op=select(env,"B10","exchange")
    stand_at(env,op.entities[0])
    env.course.inventory={"kit":1}
    env.course.interact(env.char)
    assert env.course.inventory=={"kit":1} and not op.complete
    env.course.inventory["coupling"]=1
    env.course.interact(env.char)
    assert op.complete
    assert env.course.inventory=={"kit":0,"coupling":0,"causeway_open":1}
    env.course.interact(env.char)
    assert env.course.inventory["causeway_open"]==1


def test_coupling_requires_an_actual_dive(env):
    op=select(env,"B09","dive")
    stand_at(env,op.entities[0])
    env.char.state=MoveState.SWIMMING
    env.char.pos[1]=-.15
    env.course.interact(env.char)
    assert not op.complete
    env.char.pos[1]=-3.5
    env.course.interact(env.char)
    assert op.complete and env.course.inventory["coupling"]==1


def test_vent_boost_requires_charge_jump_and_safe_pulse(env):
    op=select(env,"B05","crossing")
    stand_at(env,op.entities[0])
    env.char.vel[1]=8
    env.char.state=MoveState.AIRBORNE
    env.char.grounded=False
    env.world.elapsed=1.5
    env.course.tick(env.char,.03)
    assert env.char.vel[1]==8 and not op.state["launched"]
    op.state["charged"]=True
    env.world.elapsed=.5
    env.course.tick(env.char,.03)
    assert not op.state["launched"]
    env.world.elapsed=1.5
    env.course.tick(env.char,.03)
    assert env.char.vel[1]==20 and op.state["launched"]
    assert not op.complete and op.state["landings"]==0


def test_reference_actions_replay_deterministically(env):
    obs,_=env.reset(seed=7)
    policy=ReferencePolicy(env)
    actions=[]
    for _ in range(80):
        action=policy(obs)
        assert env.action_space.contains(action)
        actions.append(action)
        obs,_,_,_,_=env.step(action)
    digest=env.state_digest()
    env.reset(seed=7)
    for action in actions:
        env.step(action)
    assert env.state_digest()==digest


def test_reference_recovers_from_landing_too_close_to_sentinel(env):
    op=select(env,"B17","sentinel")
    op.state["phase"]="vulnerable"
    enemy=env.world.get(op.entities[0])
    enemy.elemental.shield=0
    env.char.pos=enemy.pos+np.array([.2,0,.7],np.float32)
    env.char.yaw=np.pi
    policy=ReferencePolicy(env)
    for _ in range(160):
        env.step(policy())
        if not enemy.alive:
            break
    assert not enemy.alive


def test_instruction_names_current_step_without_location(env):
    obs,_=env.reset(seed=7)
    assert obs["instruction"].endswith("| Step 1/2: talk to the instructor (relay lesson)")
    assert obs["task"][4]==0  # still no privileged waypoint


def test_hint_is_scoped_to_current_step(env):
    env.reset(seed=7)
    future=select(env,"B02","relay").state["roles"]["source"]
    env.course.index=env.course.operation_index=0
    stand_at(env,future)
    assert env.world._hint(env.char) is None
    stand_at(env,env.course.op.entities[0])
    assert env.world._hint(env.char)=="Interact: Learn R"


def test_off_step_objects_give_feedback_without_progress(env):
    env.reset(seed=7)
    future=select(env,"B02","relay")
    env.course.index=env.course.operation_index=0
    before=env.course.state()
    stand_at(env,future.state["roles"]["core"])
    env.course.interact(env.char)
    assert "not part of the current step" in env.course.last_feedback
    stand_at(env,future.state["roles"]["source"])
    env.course.last_feedback=""
    env.course.apply_element(env.char,Element(int(env.course.relay_elements["source"])))
    assert "not part of the current step" in env.course.last_feedback
    assert env.course.state()==before
