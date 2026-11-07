"""Placement of the Wildlands reference courses A/B (``setup: wildlands.reference_course``).

Two phases, both run by ``WildlandsReferenceEnv.reset``:

* :func:`author_world` edits the freshly generated voxel world: one courtyard per site,
  the connected brick lanes (``fill_box`` rows at z = 24/60/96/128 and columns at
  x = 24/52/80/108/136), the flooded return road, the bridge and its detour, dressed
  courtyards, cleared interactables, and the temperate course mask.
* ``reference_course(hooks, rng, {"course": "a"|"b"})`` (registered) draws the seeded
  assignments and places every milestone's fixture through its mechanism
  (:meth:`Mechanism.place`), then the course-specific extras (the B22 inlet on the
  thermal gallery, the yard's bokoblin) and the per-family control symbols.

``rng`` is ``default_rng(seed + 8191)``. Draw order: calibration, authorized basin,
basin map, the three family symbol permutations, then one ``integers(3)`` per
flow/cool milestone in course order (prepare derives its outlet instead).
Positions are course-authoring data, never supplied as policy observations.
"""
from __future__ import annotations

import numpy as np

from owb.sims.voxel.core import blocks as B
from owb.sims.voxel.world.worldgen import Landmark

from ..setups import SetupResult, register_setup
from .course import Milestone
from .mechanisms import MECHANISMS
from .mechanisms.base import Y

A = [
 ('Accept the repair commission','clue',(24,24)),
 ('Restore the workshop lift','balance',(52,24)),
 ('Reach the thermal gallery','thermal',(80,24)),
 ('Open the practice gate','balance',(108,24)),
 ('Recover the salvage kit','thermal',(136,24)),
 ('Survey the valley','clue',(136,60)),
 ('Obtain the replacement regulator','exchange',(108,60)),
 ('Retrieve the cached handle','exchange',(80,60)),
 ('Open the intake housing','balance',(52,96)),
 ('Reach the intake gallery','thermal',(80,96)),
 ('Repair and calibrate the intake','calibrate',(90,96)),
 ('Reopen the pass and report','report',(24,24)),
]
B_TASK = [
 ('Accept the floodworks commission','clue',(24,24)),
 ('Restore the training lift','balance',(52,24)),
 ('Repair the second gate','balance',(80,24)),
 ('Restore the thermal shaft','thermal',(108,24)),
 ('Reach the second gallery','thermal',(136,24)),
 ('Restore the flow circuit','flow',(136,60)),
 ('Commission the second circuit','flow',(108,60)),
 ('Survey the reserve depot','clue',(136,96)),
 ('Prepare expedition supplies','cook',(80,60)),
 ('Recover the wind coupling','exchange',(52,60)),
 ('Recover two couplings','exchange',(24,60)),
 ('Survey channels and safety','clue',(24,96)),
 ('Open the ridge housing','balance',(52,96)),
 ('Reach the wind deck','thermal',(80,96)),
 ('Commission the wind station','station',(90,96)),
 ('Commission the marsh pump','flow',(52,128)),
 ('Make the quarry shaft safe','cool',(108,128)),
 ('Commission the quarry engine','thermal',(136,128)),
 ('Inspect the storm update','clue',(24,96)),
 ('Retrieve the reserve winch','exchange',(136,96)),
 ('Repair the final lock','balance',(80,128)),
 ('Configure final outlets','prepare',(108,96)),
 ('Bring the floodworks online','thermal',(108,96)),
 ('Signal the designated refuge','report',(24,24)),
]
COURSES = {'a': A, 'b': B_TASK}

#: assessment DAG of course B beyond the default chain (each milestone needs the previous)
B_PREREQS = {**{mid: ('B08',) for mid in ('B09', 'B10', 'B11')},
             'B12': ('B09', 'B10', 'B11'),
             **{mid: ('B12',) for mid in ('B13', 'B16', 'B17')},
             'B19': ('B15', 'B16', 'B18')}
#: persistent repair-satchel transfers: milestone -> (consumes, gives)
TRANSFERS = {
    'a': {'A05': ((), ('salvage_kit',)), 'A07': (('salvage_kit',), ('regulator', 'cache_key')),
          'A08': (('cache_key',), ('service_handle',)), 'A11': (('regulator', 'service_handle'), ())},
    'b': {'B01': ((), ('authorization',)), 'B10': ((), ('wind_coupling',)),
          'B11': ((), ('pump_coupling', 'engine_coupling')), 'B15': (('wind_coupling',), ()),
          'B16': (('pump_coupling',), ()), 'B18': (('engine_coupling',), ()),
          'B20': ((), ('winch',)), 'B21': (('winch',), ())},
}
#: the elevated service console shares the intake/wind-deck gallery; it gets no new pad
ELEVATED_CONSOLE = (90, 96)


def author_world(result, course):
    """Authored access lanes and courtyards within the intact procedural landscape.

    Rendering exports this exact collision world.
    """
    w = result.world
    coords = {p for _, _, p in COURSES[course]}
    for x, z in coords:
        if (x, z) == ELEVATED_CONSOLE: continue  # elevated service console, not a new pad
        w.fill_box((x-10, 0, z-9), (x+13, 28, z+9), B.DIRT)
        w.fill_box((x-10, 29, z-9), (x+13, 63, z+9), B.AIR)
        w.fill_box((x-10, 28, z-9), (x+13, 28, z+9), B.GRASS)
        w.fill_box((x-4, 28, z-3), (x+5, 28, z+3), B.BRICK)
    # Routes form a connected network; no reset/teleport between milestones.
    for z in (24, 60, 96, 128):
        w.fill_box((20, 0, z-1), (149, 28, z+1), B.DIRT)
        w.fill_box((20, 29, z-1), (149, 63, z+1), B.AIR)
        w.fill_box((20, 28, z-1), (149, 28, z+1), B.BRICK)
    for x in (24, 52, 80, 108, 136):
        w.fill_box((x-1, 0, 20), (x+1, 28, 132), B.DIRT)
        w.fill_box((x-1, 29, 20), (x+1, 63, 132), B.AIR)
        w.fill_box((x-1, 28, 20), (x+1, 28, 132), B.BRICK)
    # The return road starts flooded and changes collision on drainage.
    w.fill_box((22, 29, 43), (26, 30, 47), B.WATER)
    # A bridge across a real cut, plus an always-passable side detour.
    w.fill_box((127, 23, 94), (131, 28, 98), B.AIR)
    w.fill_box((127, 23, 94), (131, 25, 98), B.WATER)
    w.fill_box((127, 28, 95), (131, 28, 97), B.PLANK)
    for lo, hi in [((123, 0, 89), (140, 28, 91)), ((123, 0, 89), (125, 28, 97)), ((138, 0, 89), (140, 28, 97))]:
        w.fill_box(lo, hi, B.DIRT)
        w.fill_box((lo[0], 29, lo[2]), (hi[0], 63, hi[2]), B.AIR)
        w.fill_box((lo[0], 28, lo[2]), hi, B.BRICK)
    # Dressed stone courtyards: columns, roofs, braziers and readable props.
    for x, z in coords:
        if (x, z) == ELEVATED_CONSOLE: continue
        for dx in (-5, 5):
            w.fill_box((x+dx, 29, z+6), (x+dx+1, 33, z+7), B.MOSS_STONE)
        w.fill_box((x-5, 34, z+6), (x+6, 34, z+7), B.BRICK)
        w.set_block(x-7, 29, z+6, B.FLOWER)
        w.set_block(x+7, 29, z-6, B.TALL_GRASS)
    # Preserve scenery outside authored footprints; clear random interactables
    # on the course so a gather command has an unambiguous nearby target.
    result.npc_spawns = []; result.chests = []; result.bodies = []; result.plants = []
    result.landmarks = [Landmark('course', 'Reedbank reference course', (24.5, Y, 24.5), 4)]
    result.spawn = (22.5, Y+.01, 24.5)
    w.temperature[:] = .15
    # The authored workshop/service routes are temperate training areas.
    # Preserve native exposure rules and procedural climate elsewhere.
    course_mask = np.zeros(w.biome.shape, bool)
    for x, z in coords:
        course_mask[max(0, x-10):min(160, x+14), max(0, z-9):min(160, z+10)] = True
    for z in (24, 60, 96, 128): course_mask[20:150, z-2:z+3] = True
    for x in (24, 52, 80, 108, 136): course_mask[x-2:x+3, 20:133] = True
    course_mask[122:141, 88:99] = True
    w.biome[course_mask] = 0
    w.temperature[course_mask] = .7
    w.recompute_heightmaps()
    return result


class WorldHooks:
    """The placement surface a Wildlands setup may use (over a reset env)."""

    def __init__(self, env):
        self.env = env
        self.world = env.world

    def target(self, m, role, pos):
        targets = self.env.targets
        tid = len(targets)+1
        obj = dict(id=tid, milestone=m.id, role=role, pos=np.array(pos, float), symbol='')
        targets[tid] = obj; m.targets[role] = tid
        return tid

    def get_target(self, tid):
        return self.env.targets[tid]

    def add_body(self, kind, pos):
        return self.env.bodies.add(kind, pos)

    def body(self, bid):
        return self.env.bodies.get(bid)

    def add_plants(self, plants):
        self.env.gameplay.plants += plants

    def spawn_npc(self, kind, pos, **kw):
        return self.env.npcs.spawn(kind, pos, **kw)


def draw_rules(rng):
    calibration = int(rng.integers(3)); authorized = int(rng.integers(3))
    basin_map = rng.permutation(3).tolist()
    family_symbols = {kind: rng.permutation(['sun', 'wave', 'leaf', 'star', 'ring']).tolist()
                      for kind in ('balance', 'thermal', 'flow')}
    return dict(calibration=calibration, authorized=authorized, basin_map=basin_map,
                family_symbols=family_symbols)


@register_setup('wildlands.reference_course')
def reference_course(hooks, rng, params) -> SetupResult:
    course = params['course']
    if course not in COURSES: raise ValueError('task must be a or b')
    rules = draw_rules(rng)
    prefix = course.upper()
    milestones = []
    for i, (title, kind, site) in enumerate(COURSES[course], 1):
        mid = f'{prefix}{i:02}'
        milestones.append(Milestone(mid, title, kind, site, prereqs=(() if i == 1 else (f'{prefix}{i-1:02}',))))
    by_id = {m.id: m for m in milestones}
    if course == 'b':
        for mid, prereqs in B_PREREQS.items(): by_id[mid].prereqs = prereqs
    for mid, (inputs, outputs) in TRANSFERS[course].items():
        by_id[mid].consumes = inputs; by_id[mid].gives = outputs
    for m in milestones:
        m.state.update(done=False, installed=False, elapsed=0., observed=False)
        MECHANISMS[m.kind].place(hooks, m, rng, rules)
    if course == 'b':
        # The final inlet is physically on the elevated thermal gallery.
        inlet = hooks.get_target(by_id['B22'].targets['inlet'])
        inlet['pos'] = np.array((119, 34, 96.5), float)
        # Ordinary enemies guard the yard; the safe reference uses the back.
        hooks.spawn_npc('bokoblin', (62.5, Y, 67.5), home=(62.5, 67.5))
    for m in milestones:
        family = m.kind if m.kind in ('balance', 'thermal') else 'flow'
        for k, tid in enumerate(m.targets.values()):
            hooks.get_target(tid)['symbol'] = rules['family_symbols'][family][k % 5]
    hooks.world.recompute_heightmaps()
    record = dict(course=course, calibration=rules['calibration'], authorized=rules['authorized'],
                  basin_map=list(rules['basin_map']), family_symbols=rules['family_symbols'],
                  expected_outlets={m.id: m.state['expected'] for m in milestones if 'expected' in m.state},
                  milestones=[dict(id=m.id, kind=m.kind, site=list(m.site), prereqs=list(m.prereqs))
                              for m in milestones])
    return SetupResult(milestones, rules, record)


__all__ = ['A', 'B_TASK', 'COURSES', 'WorldHooks', 'author_world', 'draw_rules', 'reference_course']
