"""Isolatable flow network (skill S3: isolate the inlet, route, then feed).

Outlets toggle only while the inlet is isolated; opening the inlet with any outlet
pattern other than "only the expected outlet" trips the pressure interlock. ``flow``
completes after five seconds of valid feed (plus installed parts where a socket
exists); ``cool`` is the marked cooling bypass, which lowers the plume temperature by
8 C/s while valid and completes at 45 C; ``prepare`` completes when the authorized outlet
alone is open with the inlet still isolated.
"""
from __future__ import annotations

from . import register
from .base import Mechanism, fail, family_roles


@register
class FlowNetwork(Mechanism):
    kinds = ('flow',)

    def operate(self, env, m, role, obj):
        s = m.state
        if role.startswith('outlet'):
            if s['inlet']: return fail('isolate inlet before rerouting')
            i = int(role[-1]); s['outlets'][i] = not s['outlets'][i]
        elif role == 'inlet':
            if s['inlet']: s['inlet'] = False
            elif s['outlets'] != [k == s['expected'] for k in range(3)]:
                s['trips'] += 1; s['stable'] = 0
                return [dict(type='pressure_trip', milestone=m.id)]
            else: s['inlet'] = True
        else: return fail('not a flow control')
        return [dict(type='flow_control', milestone=m.id, control=role,
                     value=s['inlet'] if role == 'inlet' else s['outlets'][int(role[-1])])]

    def clue(self, env, m):
        roles = family_roles(env, m)
        hint = f'Family flow: {roles}. Isolate inlet, open authorized outlet, then feed. ' if m.id == 'B06' else ''
        return hint+f'Local requested outlet: {m.state["expected"]+1}. Temperature: {m.state["temperature"]:.1f} C.'

    def update(self, env, m, events, kinds, dt):
        s = m.state
        valid = s['inlet'] and s['outlets'] == [k == s['expected'] for k in range(3)]
        s['stable'] = s['stable']+dt if valid else 0.
        self.while_valid(s, valid, dt)

    def while_valid(self, s, valid, dt):
        pass

    def complete(self, env, m):
        s = m.state
        return s.get('stable', 0) >= 5 and (not m.consumes or s['installed'])


@register
class CoolingBypass(FlowNetwork):
    kinds = ('cool',)

    def while_valid(self, s, valid, dt):
        if valid: s['temperature'] = max(35., s['temperature']-8*dt)

    def complete(self, env, m):
        return m.state.get('temperature', 90) <= 45


@register
class OutletPreparation(FlowNetwork):
    kinds = ('prepare',)

    def clue(self, env, m):
        return 'Three outlet symbols. Consult the previously surveyed basin map and commission authorization.'

    def complete(self, env, m):
        s = m.state
        return s.get('outlets') == [k == s.get('expected') for k in range(3)] and not s.get('inlet', False)
