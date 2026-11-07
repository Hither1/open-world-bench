"""Course runtime of the Wildlands reference courses A/B: milestone record and ledger.

:func:`outcome` is the per-step course update the env runs after the native ticks: every
milestone's mechanism updates, the drained-road survey, then each eligible, incomplete
milestone whose mechanism reports completion is recorded in ``env.completed`` (and emits
``milestone_complete``) together with its course consequences (satchel grants, stations
online, road drainage, the storm). :func:`system_operational` is the final live-world
predicate behind ``env.success``.

``Milestone.state`` is the mechanism state that ``state_digest`` serialises;
``targets`` maps control role -> target id in ``env.targets``; ``prereqs`` is the
assessment DAG (``env.eligible``); ``consumes``/``gives`` are the repair-satchel
transfers. ``requires`` is never set or read (dead field kept for compatibility).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from owb.sims.voxel.core import blocks as B

from .mechanisms import MECHANISMS
from .mechanisms.errands import survey_road


@dataclass
class Milestone:
    id: str
    title: str
    kind: str
    site: tuple
    requires: tuple = ()
    gives: tuple = ()
    consumes: tuple = ()
    state: dict = field(default_factory=dict)
    targets: dict = field(default_factory=dict)
    prereqs: tuple = ()


#: B23 (floodworks online) additionally needs the final outlet network feeding stably and
#: every regional station and the final lock completed.
FLOODWORKS_JOIN = ('B15', 'B16', 'B18', 'B21', 'B22')
#: stations whose completion latches them online; all three online trigger the storm
STATIONS = ('B15', 'B16', 'B18')
#: completing one of these drains the return road
DRAINS_ROAD = ('A11', 'B23')


def outcome(env, events):
    """Advance mechanisms and the ledger by one env step; returns (success, reward)."""
    reward = 0.; kinds = {e['type'] for e in events}; dt = env.cfg.frame_skip*env.cfg.sim_cfg.dt
    for m in env.milestones:
        MECHANISMS[m.kind].update(env, m, events, kinds, dt)
    survey_road(env)
    completed_ids = {r['id'] for r in env.completed}
    for m in env.milestones:
        if m.id in completed_ids or not env.eligible(m): continue
        ok = MECHANISMS[m.kind].complete(env, m)
        if m.id == 'B23':
            flow = env.by_id['B22'].state
            ok = ok and flow['stable'] >= 5 and set(FLOODWORKS_JOIN) <= completed_ids
        if ok:
            if m.kind != 'exchange':
                for item in m.gives: env.satchel[item] += 1
            env.completed.append(dict(id=m.id, step=env.steps, seconds=round(env.elapsed, 6)))
            completed_ids.add(m.id); reward += 1
            events.append(dict(type='milestone_complete', milestone=m.id, title=m.title))
            if m.id in STATIONS: env.online.add(m.id)
            if m.id in DRAINS_ROAD:
                env.world.fill_box((22, 29, 43), (26, 30, 47), B.AIR)
                env.drained = True; events.append(dict(type='road_drained'))
    if env.task == 'b' and env.online == set(STATIONS) and not env.storm:
        env.storm = True; env.world.fill_box((127, 28, 95), (131, 28, 97), B.AIR)
        env.elements.climate.weather = 'storm'
        events.append(dict(type='bridge_closed', version=2))
    if env.task == 'a' and 'A09' in completed_ids: env.elements.climate.weather = 'rain'
    env.success = (len(env.completed) == len(env.milestones) and env.drained
                   and env.agent.alive and system_operational(env))
    return env.success, reward+(10 if env.success else 0)


def system_operational(env):
    """Final success requires working equipment, not just historical flags."""
    if env.task == 'a':
        s = env.by_id['A11'].state
        return s['installed'] and s.get('engaged', False) and s['selected'] == env.calibration and s['elapsed'] >= 5
    wind, pump, engine, lock, flow = (env.by_id[k].state for k in ('B15', 'B16', 'B18', 'B21', 'B22'))
    valid_flow = lambda s: s['inlet'] and s['outlets'] == [k == s['expected'] for k in range(3)] and s['stable'] >= 5
    return (wind['installed'] and wind['elapsed'] >= 5 and pump['installed'] and valid_flow(pump)
            and engine['installed'] and engine['landed'] and engine['elapsed'] >= 5
            and lock['installed'] and lock['released'] and lock['stable'] >= 2 and valid_flow(flow))


def evaluator_events(events):
    """Mechanism events in the form ``MilestoneEvaluator``'s ``event`` predicate matches:
    ``milestone_complete`` carries its milestone id as ``entity``."""
    return [dict(e, entity=e['milestone']) if e.get('type') == 'milestone_complete' else e
            for e in events]


__all__ = ["Milestone", "evaluator_events", "outcome", "system_operational"]
