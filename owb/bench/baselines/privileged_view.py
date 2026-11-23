"""Privileged-state observations for model upper-bound pilots.

This is a separate, explicitly labelled condition: it reads evaluator-side state
(true roles, world positions, mechanism state, hidden rule constants and the
events/failure reasons of the previous decision) for the current milestone. It
never supplies reference actions, plans, or scripted controller output; the model
still chooses every primitive action. Results under this condition measure
control and execution given full task knowledge, not discovery or memory.
"""
import numpy as np

from ...env.wildlands_reference import Y as WILD_GROUND, plain

NOISE = {'land', 'jump', 'apple_fell', 'npc_denned', 'npc_emerged', 'magnetic_move', 'craft_moved',
         'horse_spooked', 'swing'}


def _r(v, n=2):
    return np.round(np.asarray(v, float), n).tolist()


def feedback(events, limit=15):
    """Events from the previous decision's primitive actions; motion spam is summarised."""
    kept, moves = [], 0
    for e in events:
        if e['type'] in ('magnetic_move', 'craft_moved'):
            moves += 1
        elif e['type'] not in NOISE:
            kept.append(plain(e))
    out = dict(events=kept[-limit:])
    if moves:
        last = next(e for e in reversed(events) if e['type'] in ('magnetic_move', 'craft_moved'))
        out['motion_steps'] = moves; out['last_motion'] = plain(last)
    return out


WILDLANDS_RULES = {
    'clue': 'inspect the milestone inspect control (within 3.2m, visible).',
    'balance': ('operate the brake control (braked=true). Place unheld, resting bodies on the ground inside support_zone '
                'with total mass within count*320 +/- 1 kg (load). Then operate release while braked and load is valid; '
                'release with invalid load trips the brake. Load must stay valid 2 s after release. If the milestone '
                'consumes parts, install at the install socket after release. grip reaches 4m; drag vector moves the '
                'held body up to 0.4m per axis per action; drop it so it rests on the ground (y about ground_y+half height).'),
    'thermal': ('operate vent (toggles open), then operate heat while vent is open and plume safe. Then enter the updraft '
                'region (x in [site_x-2, site_x+10], |z-(site_z+0.5)|<4) and glide/rise, landing on ground higher than '
                'ground_y+3.8 within 3m (horizontal) of state.landing. If parts are consumed, install after landing and '
                'remain 5 s.'),
    'flow': ('with inlet closed, toggle outlets so exactly outlet index state.expected is open, then operate inlet to open '
             'it (a wrong outlet set trips). Keep valid 5 s; install consumed parts if any.'),
    'cool': 'as flow; keep valid until temperature <= 45 C.',
    'prepare': 'set outlets so only state.expected is open and leave the inlet closed.',
    'calibrate': ('install both repair parts at the install socket, operate the mark matching the calibration from the '
                  'A06 clue (mark index = calibration-1), then operate/interact the engage control; stay 5 s.'),
    'station': 'operate/interact with the required coupling in the satchel; it engages and must run 5 s.',
    'exchange': 'interact with the locker while carrying the required input items.',
    'report': 'interact with the report source.',
    'cook': 'cook dishes giving cold_resist and heat_resist with the site campfire (site_x-2, ground, site_z+2) lit.',
}


def wildlands(env, events):
    m = env.current
    if m is None:
        return dict(objective=None, feedback=feedback(events))
    x, z = m.site; s = m.state
    controls = []
    for role, tid in m.targets.items():
        obj = env.targets[tid]; pos = obj['pos']
        if role.startswith('weight'):
            pos = env.bodies.get(s['bodies'][int(role[6:])]).pos
        controls.append(dict(id=tid, role=role, symbol=obj['symbol'], pos=_r(pos),
                             reachable=bool(env._reachable(pos, 4 if role.startswith('weight') else 3.2))))
    out = dict(objective=dict(id=m.id, title=m.title, kind=m.kind, site=[x, WILD_GROUND, z],
                              prerequisites=list(m.prereqs), prerequisites_met=bool(env.eligible(m)),
                              consumes=list(m.consumes), gives=list(m.gives)),
               clue=env.clue(m), state=plain({k: v for k, v in s.items() if k != 'bodies'}),
               controls=controls, rule=WILDLANDS_RULES.get(m.kind, ''), satchel=dict(env.satchel),
               holding_body=env.held is not None, feedback=feedback(events))
    if m.kind == 'balance':
        out['support_zone'] = dict(x=[x+2.4, x+4.7], z=[z+.5, z+4.7], ground_y=WILD_GROUND,
                                   required_load_kg=s['count']*320)
        out['bodies'] = []
        for bid in s['bodies']:
            b = env.bodies.get(bid)
            out['bodies'].append(dict(body=bid, pos=_r(b.pos), mass=float(b.spec.mass), held=bool(b.held),
                                      resting=bool(b.resting), half_extent=_r(b.half)))
    return out


SKYLANDS_RULES = {
    'clue': 'inspect the milestone control.',
    'raft': ('pick each loose part of this craft and mount(target=raft, option=slot): slots 0 and 1 take pontoons, '
             'slot 2 the fan (orientation turn must be 0), slot 3 the cell. Supported when pontoon capacity >= 160 kg + '
             'cargo mass, fan turn 0 and charge > 0. The first workshop cell arrives nearly flat: charge the craft by '
             'attack-striking its mounted fan within 3.4m of the craft; a strike 0.5-1.5 s after the previous strike adds '
             '45, later adds 12 (from rest), sooner jams (-10). Each metre driven costs 0.25 charge. load(target=raft) '
             'with the cargo within 6m. Stand on the deck, drive(target=raft) with build_move sign on ONE horizontal '
             'axis (east or south), then repeat none; the craft moves 1m per action over water only, following '
             'route waypoints. Stop exactly on destination (within 0.1m) and unload(target=raft). Done when cargo is '
             'unloaded at the destination.'),
    'ascend': ('stand at state.launch under the gallery and ascend (empty hands, 20 stamina) to land within 2m of '
               'state.upper, then use the control there.'),
    'stage': ('assemble and load a return craft as for raft, drive it at least 8m and back to its exact origin, then '
              'use(target=raft) to moor it.'),
    'dock': 'use the control with authorization and lamp in the satchel.',
    'extract': 'after the vault is open and the cradle is in the satchel, use the control to free the core.',
    'freight': 'haul the core within 6m of the freight deck, then use(target=freight) to load and raise it.',
    'hoist': 'bring the core onto the hoist deck (use freight target), then use the control on the gallery to raise it.',
    'calibrate': 'with the core at the observatory, use the control with option = calibration from the A07 clue.',
    'return': 'load the core onto the staged return craft, drive to its destination and unload.',
}


def skylands(env, events):
    m = env.current
    if m is None:
        return dict(objective=None, feedback=feedback(events))
    s = m.state; rid = s.get('raft') if isinstance(s.get('raft'), str) else (m.id if m.id in env.rafts else None)
    owners = {m.id, f'{m.id}_ballast'} | ({rid, f'{rid}_ballast'} if rid else set())
    targets = []
    for tid, obj in sorted(env.targets.items()):
        near = np.linalg.norm(obj['pos']-env.agent.pos) < 40
        if obj['owner'] not in owners and not near:
            continue
        entry = dict(id=tid, role=obj['role'], owner=obj['owner'], pos=_r(obj['pos']),
                     reachable_4_5m=bool(env._reachable(obj['pos'], 4.5)), reachable_6m=bool(env._reachable(obj['pos'], 6)))
        if obj['role'] == 'part':
            p = env.parts[obj['part']]
            entry.update(kind=p['kind'], capacity=p['capacity'], mounted=p['mounted'], turn=p['turn'],
                         held=env.held_part == obj['part'])
        if obj['role'] == 'cargo' and obj['owner'] in env.cargo:
            c = env.cargo[obj['owner']]; entry.update(mass=c['mass'], parent=c['parent'], anchored=c['anchored'], pos=_r(c['pos']))
        targets.append(entry)
    out = dict(objective=dict(id=m.id, title=m.title, kind=m.kind, layer=m.layer, pos=_r(m.pos)),
               clue=env.clue(m), state=plain(s), targets=targets, rule=SKYLANDS_RULES.get(m.kind, ''),
               satchel=dict(env.satchel), held_part=(env.parts[env.held_part]['kind'] if env.held_part else None),
               towing=env.hauling, feedback=feedback(events))
    if rid and rid in env.rafts:
        r = env.rafts[rid]
        out['craft'] = dict(id=rid, target=r['target'], pos=_r(r['pos']), origin=_r(r.get('origin_pos', r['pos'])),
                            slots=[env.parts[p]['kind'] if p else None for p in r['parts']], charge=round(r['charge'], 2),
                            cargo=r['cargo'], cargo_id=r['cargo_id'], travel=r['travel'], moored=r['moored'],
                            supported=bool(env.craft_valid(r)), on_deck=bool(env.on_deck(r)),
                            driving=bool(r.get('driving')), destination=_r(r.get('destination', [])),
                            route=[_r(q) for q in r.get('route', [])],
                            seconds_since_fan_strike=(round(env.elapsed-r['spun'], 2) if 'spun' in r else None))
    return out
