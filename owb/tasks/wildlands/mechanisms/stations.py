"""Service consoles: the intake calibration (A11) and the regional stations (B15).

``calibrate``: with the repair parts installed, select a mark and engage; engaging on
any mark other than the seeded ``env.calibration`` trips the reversible interlock. Five
seconds engaged on the right mark completes it. ``station``: interacting installs the
required coupling from the satchel and engages the station; five seconds of installed
output completes it.
"""
from __future__ import annotations

from . import register
from .base import Mechanism, fail


@register
class Calibration(Mechanism):
    kinds = ('calibrate',)

    def operate(self, env, m, role, obj):
        s = m.state
        if role.startswith('mark'): s['selected'] = int(role[-1]); return [dict(type='calibration_selected', value=s['selected'])]
        if not s['installed']: return fail('install both repair parts first')
        if s['selected'] != env.calibration: return [dict(type='calibration_trip')]
        s['engaged'] = True; return [dict(type='intake_engaged')]

    def update(self, env, m, events, kinds, dt):
        s = m.state
        if s.get('engaged') and s['selected'] == env.calibration: s['elapsed'] += dt

    def complete(self, env, m):
        return m.state['elapsed'] >= 5


@register
class Station(Mechanism):
    kinds = ('station',)

    def operate(self, env, m, role, obj):
        s = m.state
        if any(env.satchel[k] < 1 for k in m.consumes): return fail('required coupling missing')
        if not s['installed']:
            for k in m.consumes: env.satchel[k] -= 1
            s['installed'] = True
        return [dict(type='station_engaged', milestone=m.id)]

    def update(self, env, m, events, kinds, dt):
        if m.state['installed']: m.state['elapsed'] += dt

    def complete(self, env, m):
        return m.state['elapsed'] >= 5
