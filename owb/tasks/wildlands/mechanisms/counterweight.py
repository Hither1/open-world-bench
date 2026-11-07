"""Counterweighted, latched mechanism (skill S1: brake, support, load, release).

Hold the brake, place magnetic weights on the support cradle until the resting load is
within 1 kg of ``count * 320`` kg, then release. Releasing without a brake or valid load
trips the brake; losing the load while released trips it too. Two seconds of stable,
released motion opens the gate (and, on the A02/B02 training lift, raises the lift deck).
"""
from __future__ import annotations

from owb.sims.voxel.core import blocks as B

from . import register
from .base import Y, Mechanism, fail, family_roles


def valid_load(env, m):
    s = m.state; x, z = m.site
    bodies = [b for b in env.bodies.bodies.values() if not b.held and abs(b.pos[1]-b.half[1]-Y) < .06
              and x+2.4 < b.pos[0] < x+4.7 and z+.5 < b.pos[2] < z+4.7 and b.resting]
    s['load'] = sum(b.spec.mass for b in bodies)
    return s['count']*320-1 <= s['load'] <= s['count']*320+1


@register
class Counterweight(Mechanism):
    kinds = ('balance',)

    def install_blocked(self, env, m):
        return None if m.state['released'] else 'support lock before installation'

    def operate(self, env, m, role, obj):
        s = m.state
        if role == 'brake':
            s['braked'] = True; s['released'] = False; s['stable'] = 0
        elif role == 'release':
            if not s['braked'] or not valid_load(env, m):
                s['released'] = False; s['stable'] = 0
                return [dict(type='brake_trip', milestone=m.id, load=s['load'])]
            s['released'] = True
        else: return fail('not a counterweight control')
        return [dict(type='counterweight_control', milestone=m.id, control=role, load=s['load'])]

    def clue(self, env, m):
        roles = family_roles(env, m)
        hint = f'Family counterweight: {roles}. Hold brake, establish support, then release. ' if m.id in ('A02', 'B02') else ''
        return hint+f'Load safe band: {m.state["count"]*320} kg. Current load: {m.state["load"]} kg.'

    def update(self, env, m, events, kinds, dt):
        s = m.state; x, z = m.site
        valid = valid_load(env, m)
        if s['released'] and not valid:
            s['released'] = False; s['stable'] = 0
            events.append(dict(type='brake_trip', milestone=m.id, reason='support lost'))
        if s['released']:
            s['stable'] += dt
            if s['stable'] >= 2 and not s['done']:
                s['done'] = True
                env.world.fill_box((x+6, 29, z-2), (x+6, 32, z+2), B.AIR)
                if m.id in ('A02', 'B02'):
                    env.world.fill_box((x+7, 29, z-2), (x+10, 29, z+2), B.AIR)
                    env.world.fill_box((x+7, 32, z-2), (x+10, 32, z+2), B.BRICK)
                    s['lift_level'] = 3
                events.append(dict(type='counterweight_open', milestone=m.id))

    def complete(self, env, m):
        s = m.state
        return s['done'] and (not m.consumes or s['installed'])
