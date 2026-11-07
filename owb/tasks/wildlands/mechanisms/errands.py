"""Errand mechanisms: clue boards, supply lockers, the cooking stop and the final report.

``clue``: inspecting the board shows its authored text and completes the milestone (the
exposure time is recorded in ``env.exposed``). ``exchange``: a one-shot locker that
trades the milestone's ``consumes`` items for its ``gives`` (B10/B11 also need B01's
commission). ``cook``: completes once cold- and heat-resist dishes were cooked and the
sheltered campfire is lit. ``report``: walking the drained road (south checkpoint, then
north) and signalling completes the course.
"""
from __future__ import annotations

from owb.sims.voxel.core import blocks as B

from . import register
from .base import Mechanism, fail

#: authored board texts, by milestone id
CLUE_TEXTS = {
    'A01': lambda env: 'Ranger cache: forked tree at the central court. Complete the workshop before collecting repair parts.',
    'A06': lambda env: f'Intake safe calibration mark: {env.calibration+1}. Coastal regulator locker is at the western coastal court.',
    'B01': lambda env: f'Protect Reedbank refuge. Authorized empty overflow basin: {env.authorized+1}.',
    'B08': lambda env: 'Reserve depot: stone arch at the eastern causeway. Normal approach crosses the wooden bridge. Shutter needs all three stations.',
    'B12': lambda env: f'Outlet symbols 1,2,3 lead to basins {[v+1 for v in env.basin_map]}. Furnace launch is safe only at 45 C or below.',
    'B19': lambda env: 'Storm report v2: wooden causeway bridge is closed. Use the north stone detour to the same reserve depot.',
}


@register
class Clue(Mechanism):
    kinds = ('clue',)

    def clue(self, env, m):
        text = CLUE_TEXTS.get(m.id)
        return text(env) if text else m.title

    def complete(self, env, m):
        return m.state['observed']


@register
class Exchange(Mechanism):
    kinds = ('exchange',)

    def operate(self, env, m, role, obj):
        s = m.state
        if s['done']: return fail('locker empty')
        if any(env.satchel[k] < 1 for k in m.consumes): return fail('exchange input missing')
        if m.id in ('B10', 'B11') and not env.satchel['authorization']: return fail('commission required')
        for k in m.consumes: env.satchel[k] -= 1
        for k in m.gives: env.satchel[k] += 1
        s['done'] = True
        return [dict(type='item_exchange', milestone=m.id, inputs=m.consumes, outputs=m.gives)]

    def complete(self, env, m):
        return m.state['done']


@register
class Cook(Mechanism):
    kinds = ('cook',)

    def place(self, hooks, m, rng, rules):
        x, z = m.site
        super().place(hooks, m, rng, rules)
        hooks.world.set_block(x, 29, z+2, B.COOKPOT)
        hooks.world.set_block(x-2, 29, z+2, B.CAMPFIRE)
        hooks.add_plants([dict(pos=(x+.5, 29, z-.5), item='spicy_pepper', count=2),
                          dict(pos=(x+1.5, 29, z-.5), item='chillshroom', count=2)])
        m.state['cooked'] = []

    def update(self, env, m, events, kinds, dt):
        s = m.state; x, z = m.site
        for e in events:
            if e['type'] == 'cook': s['cooked'].extend(k for k in e['effects'] if k in ('cold_resist', 'heat_resist'))
        s['done'] = {'cold_resist', 'heat_resist'} <= set(s['cooked']) and env.world.get(x-2, 29, z+2) == B.CAMPFIRE_LIT

    def complete(self, env, m):
        return m.state['done']


@register
class Report(Mechanism):
    kinds = ('report',)

    def operate(self, env, m, role, obj):
        s = m.state
        if not env.drained or env.road_seen != {'north', 'south'}: return fail('verify the drained road first')
        s['done'] = True; return [dict(type='completion_signal')]

    def complete(self, env, m):
        return m.state['done']


def survey_road(env):
    """Register the drained-road checkpoints the agent has walked (south, then north)."""
    if env.drained:
        x, _, z = env.agent.pos
        if abs(x-24.5) < 2:
            if 48 < z < 53: env.road_seen.add('south')
            if 37 < z < 42 and 'south' in env.road_seen: env.road_seen.add('north')
