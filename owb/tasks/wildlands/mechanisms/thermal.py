"""Thermal shaft (skill S2: open the draft, protect the heat, check the plume, launch).

The vent toggles the draft; lighting the sheltered hearth with the draft open (and a
safe plume) creates an updraft of 8 over the launch lane; the exposed hearth never
works. Landing on the service deck after riding the updraft completes the shaft; a
shaft with a parts socket also needs the parts installed and five seconds of service.
The quarry shaft (B18) is safe only while B17's cooling bypass holds the plume at or
below 45 C.
"""
from __future__ import annotations

import numpy as np

from owb.sims.voxel.core import blocks as B

from . import register
from .base import Y, Mechanism, fail, family_roles


def install_updraft(env):
    """Wrap ``env.elements.updraft_at`` with the lanes of working thermal shafts."""
    base = env.elements.updraft_at

    def updraft(pos):
        best = base(pos)
        for m in env.milestones:
            if m.kind != 'thermal': continue
            x, z = m.site; s = m.state
            if (s['vent'] and s['heat'] and s['safe'] and x-1 <= pos[0] <= x+8
                    and abs(pos[2]-(z+.5)) < 3 and Y <= pos[1] < Y+9):
                best = max(best, 8.)
        return best
    env.elements.updraft_at = updraft


@register
class Thermal(Mechanism):
    kinds = ('thermal',)

    def place(self, hooks, m, rng, rules):
        x, z = m.site; y = Y
        m.state.update(vent=False, heat=False, updraft_seen=False, landed=False,
                       hot=m.id == 'B18', safe=m.id != 'B18', launch_seen=False)
        hooks.target(m, 'vent', (x-2, y+1, z+3))
        hooks.target(m, 'heat', (x, y+1, z+3))
        hooks.target(m, 'exposed_heat', (x+2, y+1, z+3))
        hooks.target(m, 'inspect', (x-2, y+1, z-3))
        m.state['landing'] = (x+10.5, Y+4.01, z+.5)
        w = hooks.world
        w.fill_box((x+8, 29, z-3), (x+13, 32, z+3), B.MOSS_STONE)
        w.fill_box((x+8, 32, z-3), (x+13, 32, z+3), B.BRICK)
        # A raised, sheltered source sits beside (not in) the flight path.
        w.fill_box((x+1, 28, z+3), (x+3, 28, z+5), B.BRICK)
        w.set_block(x+2, 29, z+4, B.CAMPFIRE)
        w.fill_box((x-1, 35, z+3), (x+4, 35, z+5), B.PLANK)
        if m.consumes: hooks.target(m, 'install', (x+11, 34, z))

    def install_blocked(self, env, m):
        return None if m.state['landed'] else 'reach service deck first'

    def operate(self, env, m, role, obj):
        s = m.state
        if role == 'vent': s['vent'] = not s['vent']
        elif role == 'heat':
            if not s['vent']: return [dict(type='draft_blocked', milestone=m.id)]
            if not s['safe']: return fail('plume above safe launch temperature')
            s['heat'] = True
            x, z = m.site; env.elements.ignite(env.world, x+2, 29, z+4)
        elif role == 'exposed_heat': return [dict(type='draft_failed', reason='exposed hearth cannot sustain this shaft')]
        else: return fail('not a shaft control')
        return [dict(type='thermal_control', milestone=m.id, control=role)]

    def clue(self, env, m):
        roles = family_roles(env, m)
        hint = f'Family thermal: {roles}. Open draft, protect heat, check temperature, then launch. Exposed hearths fail. ' if m.id in ('A03', 'B04') else ''
        return hint+f'Plume launch safe: {m.state["safe"]}.'

    def update(self, env, m, events, kinds, dt):
        s = m.state; x, z = m.site
        if m.id == 'B18': s['safe'] = env.by_id['B17'].state['temperature'] <= 45
        if 'updraft' in kinds and x-2 < env.agent.pos[0] < x+10 and abs(env.agent.pos[2]-(z+.5)) < 4:
            s['updraft_seen'] = True
        if env.agent.gliding and x-2 < env.agent.pos[0] < x+10 and abs(env.agent.pos[2]-(z+.5)) < 4:
            s['launch_seen'] = True
        target = np.asarray(s['landing'])
        if (s['updraft_seen'] and env.agent.on_ground and env.agent.pos[1] > Y+3.8
                and np.linalg.norm((env.agent.pos-target)[[0, 2]]) < 3):
            if not s['landed']: events.append(dict(type='thermal_landed', milestone=m.id))
            s['landed'] = True
        if s['landed'] and s['installed']: s['elapsed'] += dt

    def complete(self, env, m):
        s = m.state
        return s.get('landed', False) and (not m.consumes or (s['installed'] and s['elapsed'] >= 5))
