"""Course commands: inspect, operate, grip, drag, drop, install (and targeted interact).

``drag``/``drop`` move the magnetically held body (bounded tether, collision checked).
Commands with a target first check reach and line of sight; ``inspect`` shows the
mechanism's clue text and records the exposure; everything else needs the milestone's
prerequisites. ``grip`` holds one of a counterweight's weights, ``install`` performs the
atomic parts installation (consuming the satchel parts) once the mechanism's socket is
ready, and ``operate``/``interact`` go to the milestone's mechanism. Any other command is
native (``None`` is returned so the env falls through to survival gameplay).
"""
from __future__ import annotations

import numpy as np

from owb.sims.voxel.physics.collide import box_collides

from . import MECHANISMS
from .base import fail, plain

COURSE_COMMANDS = ('inspect', 'operate', 'grip', 'drag', 'drop', 'install')


def dispatch(env, command, action):
    tid = int(action['target']); obj = env.targets.get(tid)
    if command == 'drag':
        if env.held is None: return fail('no held magnetic body')
        body = env.bodies.get(env.held)
        delta = np.asarray(action['vector'], float)*.4
        candidate = body.pos+delta
        if np.linalg.norm(candidate-env.agent.eye) > 6: return fail('magnetic tether range')
        if box_collides(env.world, candidate, body.half): return fail('body collision')
        if any(np.all(np.abs(candidate-b.pos) < (body.half+b.half)-.02)
               for b in env.bodies.bodies.values() if b.id != body.id):
            return fail('another body blocks movement')
        body.pos = candidate; body.vel[:] = 0
        return [dict(type='magnetic_move', body=body.id, pos=plain(candidate))]
    if command == 'drop':
        if env.held is None: return fail('no held magnetic body')
        bid = env.held; env.bodies.get(bid).held = False; env.held = None
        return [dict(type='magnetic_drop', body=bid)]
    if command in ('inspect', 'operate', 'grip', 'install') or (command == 'interact' and tid):
        if obj is None: return fail('unknown interaction target')
        m = env.by_id[obj['milestone']]; role = obj['role']; s = m.state
        if role.startswith('weight'):
            body = env.bodies.get(s['bodies'][int(role[6:])]); pos = body.pos
        else: pos = obj['pos']
        if not env._reachable(pos, 4 if command == 'grip' else 3.2):
            return fail('target out of reach or occluded', target=tid)
        if command == 'inspect':
            text = env.clue(m)
            env.last_text = text; s['observed'] = True; env.exposed[m.id] = env.elapsed
            return [dict(type='clue_exposed', milestone=m.id, text=text)]
        if not env.eligible(m): return fail('mechanism prerequisites incomplete', target=tid)
        if command == 'grip':
            if not role.startswith('weight') or env.held is not None: return fail('cannot grip target')
            body.held = True; body.vel[:] = 0; env.held = body.id
            return [dict(type='magnetic_grip', body=body.id)]
        if command == 'install':
            if role != 'install': return fail('not an installation socket')
            if s['installed']: return fail('parts already installed')
            blocked = MECHANISMS[m.kind].install_blocked(env, m)
            if blocked: return fail(blocked)
            if any(env.satchel[k] < 1 for k in m.consumes): return fail('required repair parts missing')
            for k in m.consumes: env.satchel[k] -= 1
            s['installed'] = True
            return [dict(type='parts_installed', milestone=m.id, parts=list(m.consumes))]
        return MECHANISMS[m.kind].operate(env, m, role, obj)
    return None
