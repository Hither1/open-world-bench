"""Executable Wildlands reference courses, using native movement and physics.

This is a compact authored implementation of the 12/24 milestone designs.
The privileged controller is separate. No action can set progress or teleport.
Mechanisms are local, recoverable state machines with physical loads/landings.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from itertools import permutations
import json
import numpy as np
from gymnasium import spaces
from .wildlands import WildlandsEnv, COMMANDS as BASE_COMMANDS
from ..core.config import EnvConfig, WorldConfig, RenderConfig, SimConfig
from ..core import blocks as B
from ..core.working_memory import WorkingMemoryProbe, summaries as wm_summaries
from ..physics.collide import box_collides
from ..world.worldgen import generate, Landmark
from ..world.fauna import populate
from ..sim.riding import COMMANDS as RIDING_COMMANDS
from ..world.organic import (Road, blob, rounded_box, shape_course, top_solid, unit, carve_river,
                             settle_water, assert_water_contained, water_top)
from scipy import ndimage

# Riding commands are appended last: traces store integer command indices.
# Afternoon starts chosen so each course spends its first half in daylight
# and crosses dusk (monsters leave their dens) before it ends.
START_TIME = {'a': .66, 'b': .55}
COMMANDS = BASE_COMMANDS + ('inspect', 'operate', 'grip', 'drag', 'drop', 'install') + RIDING_COMMANDS
Y = 29.0


def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k): plain(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    if isinstance(x, set): return sorted(x)
    return x


# Player-facing interaction feedback: what a player would see or hear when a
# lever moves, a socket refuses or a magnet slips. It replaces evaluator reason
# codes with diegetic text and reveals no hidden constants; observation text
# never enters the state digest, so replay and digests are unaffected.
INTERACTIONS = ('inspect', 'operate', 'grip', 'drag', 'drop', 'install')
REFUSALS = {
    'target out of reach or occluded': 'out of reach or blocked from here',
    'unknown interaction target': 'there is no such object here',
    'mechanism prerequisites incomplete': "it won't move yet",
    'cannot grip target': "the magnet can't hold that",
    'not an installation socket': 'there is nothing to install into',
    'parts already installed': 'already installed',
    'support lock before installation': 'the socket is locked',
    'reach service deck first': "it can't be installed from here",
    'required repair parts missing': "you don't have the parts it needs",
    'required coupling missing': "you don't have the coupling it needs",
    'exchange input missing': "you don't have what the locker asks for",
    'plume above safe launch temperature': 'too hot to light safely',
    'isolate inlet before rerouting': "the valve won't turn while the inlet is open",
    'install both repair parts first': 'it needs its repair parts installed first',
    'locker empty': 'the locker is empty',
    'commission required': 'locked: it needs a commission',
    'verify the drained road first': 'not yet',
    'inspect this source instead': 'nothing to operate; it can be read',
    'no held magnetic body': 'the magnet holds nothing',
    'magnetic tether range': "the magnet's tether won't reach further",
    'body collision': 'the slab is blocked',
    'another body blocks movement': 'the slab is blocked',
    'dismount first': 'dismount first',
    'locker keypad locked': 'the locker is locked',
    'bays not released': 'the bay is locked; the yard office has not released it',
}
NO_EFFECT = ('not a counterweight control', 'not a shaft control', 'not a flow control')
# Control posts carry their symbol as a painted colour (policy view only).
SYMBOL_COLOURS = {'sun': (226, 182, 58), 'wave': (62, 120, 206), 'leaf': (78, 166, 74),
                  'star': (232, 230, 222), 'ring': (190, 104, 58)}
STAKE_COLOUR = (236, 128, 36)

# Working-memory probes (docs/working-memory-probes.md). Each adds a physical
# requirement to an existing milestone: a cue read at a source fixture must be
# used at a fixture out of its sight. Content comes from the probe's own
# generator (owb/core/working_memory.py), never the simulation RNG. Fixture
# positions are course-authoring data, like milestone sites.
KEYPAD = tuple('0123456789')
PITCHES = ('10', '20', '30', '40', '50')          # blade pitch settings, degrees
BLADE_COLOURS = ('red', 'yellow', 'blue')         # weather-board listing order
BAY_LABELS = ('pump fittings', 'engine fittings', 'couplings')
WA07_BOARD = (130.5, Y + 1, 58.5)                 # depot entrance, on the access road
# The stores room is a closed warehouse on a three-block loading dock. Its one
# doorway is reached by climbing the dock face (a ridden horse cannot climb), and
# floor-to-ceiling racks lay out a four-lane aisle to the locker at the far end.
STORES = (104, 56, 120, 64)                       # footprint (x0, z0, x1, z1)
STORES_FLOOR = Y + 3                              # standing height inside
STORES_DOOR = (120, 57)                           # doorway in the east wall, above the dock face
STORES_RACKS = ((58, 105), (60, 119), (62, 105))  # rack rows (z, gap x)
LOCKER = (119.5, Y + 4, 63.5)                     # supply locker at the end of the last aisle
KEYPAD_STANCE = (117.5, Y + 3, 63.5)              # a spot within reach of the keypad
WA07_COPY = (117.0, Y + 4, 63.2)                  # wm_control copy, beside the keypad
WB11_MANIFEST = (35.5, Y + 1, 62.5)               # yard gate, east of the yard office
BAY_STORE = (27, 104, 34, 106)                    # coupling store south of the archive court
BAY_X = tuple(range(28, 34))                      # bay 1..6 cells; doors face south (z = 106)
WB11_COPY = (25.5, Y + 1, 107.5)
WB15_BOARD = (95.5, Y + 1, 96.0)                  # weather board against the station base's east face
DIAL = tuple((92.6, Y + 5.2, 95.4 + .3 * k) for k in range(len(PITCHES)))
ROTOR_HUB = (92.6, Y + 6.0, 97.8)
DECK_PARAPET = ((93, 33, 93), (93, 34, 99))       # two blocks high on the deck's east edge
WB15_COPY = (91.5, Y + 5, 94.2)
WM_TEXT = {
    'WM-WA07': 'Depot dispatch board: the stores locker code is {code}. Key it in on the locker keypad inside the stores room.',
    'WM-WB11': ('Yard manifest: pump coupling in bay {pump}; engine coupling in bay {engine}. '
                'The coupling bays stand behind the archive court; the yard office releases them.'),
    'WM-WB15': ('Weather board: set blade pitch by tag colour - red {red}, yellow {yellow}, blue {blue} degrees. '
                "The deck dial sets blades by number (1, then 2, then 3); match each numbered blade's colour tag."),
}
WM_NAMES = {'WM-WA07': 'dispatch board', 'WM-WB11': 'yard manifest', 'WM-WB15': 'weather board'}
WM_CUE_SECONDS = 5.0
# Policy-view impostor colours (presentation only).
WM_COLOURS = {'dispatch board': (222, 208, 170), 'yard manifest': (84, 120, 88), 'weather board': (120, 150, 176),
              'keypad': (70, 74, 78), 'dial': (54, 58, 62), 'plate': (214, 214, 206),
              'red': (206, 52, 40), 'yellow': (232, 196, 46), 'blue': (52, 92, 206)}


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


# Positions are course-authoring data, never supplied as policy observations.
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


GRID_X = (24, 52, 80, 108, 136)
STABLE = (19.5, Y, 29.5)                   # horse tethered at the start court
FLOOD = ((22, 28, 43), (26, 28, 47))       # water flush with the road surface
FLOOD_BED = ((22, 27, 43), (26, 27, 47))
GRID_Z = (24, 60, 96, 128)
GULLY = (129.5, 96.5, 2.5, 2.6)
# A river from the northern hills runs south between the western pad columns,
# under three roads, then west under the return road (north of the flood) to
# the sea. Water level only descends and stays below the banks.
RIVER = Road.smooth([(43, 163), (43, 146), (43.5, 128.5), (42.5, 112), (43, 96.5),
                     (43.5, 78), (43, 60.5), (42, 49), (36, 40.5), (24.5, 38.5),
                     (14, 39), (4, 40.5), (-4, 41)], 1.8)   # centre and radii; spans the plank cells x127-131, z95-97


def storm_gully(shape):
    gx, gz = np.meshgrid(np.arange(shape[0]) + .5, np.arange(shape[1]) + .5, indexing='ij')
    cx, cz, rx, rz = GULLY
    return ((gx - cx) / rx) ** 2 + ((gz - cz) / rz) ** 2 <= 1


def course_ground(shape, coords, seed):
    """Walkable course cells, paved cells and road centrelines.

    Roads link neighbouring grid intersections along meandering, varying-width
    curves. Two stretches stay straight because mechanisms are authored on
    exact cells: the flooded return road (x 22-26, z 43-47, walked along
    x = 24.5) and the storm bridge (plank cells x 127-131, z 95-97).
    """
    centre = lambda x, z: (x + .5, z + .5)
    roads = []
    for z in GRID_Z:
        for x0, x1 in zip(GRID_X, GRID_X[1:]):
            identity = ('wildlands-road', seed, 'row', z, x0)
            pins = ((13 / 28, 27 / 28, 1.5),) if (z, x0) == (96, 108) else ()
            roads.append(Road.meander(centre(x0, z), centre(x1, z), identity,
                                      amplitude=2.5 + 3.5 * unit(*identity), pins=pins))
    for x in GRID_X:
        for z0, z1 in zip(GRID_Z, GRID_Z[1:]):
            identity = ('wildlands-road', seed, 'column', x, z0)
            if (x, z0) == (24, 24):
                roads.append(Road.meander(centre(x, z0), centre(x, z1), identity, amplitude=0,
                                          pins=((17 / 36, 25 / 36, 2.5),)))
            else:
                roads.append(Road.meander(centre(x, z0), centre(x, z1), identity,
                                          amplitude=2.5 + 4.0 * unit(*identity)))
    # The stone detour bends north around the gully.
    roads.append(Road.smooth([(120.5, 96.5), (122.5, 92.5), (126, 89.8), (130, 89.3),
                              (134.5, 90), (138.5, 92.5), (141.5, 96.5)], 1.7))
    corridor = ndimage.binary_dilation(RIVER.mask(shape), iterations=1)
    surface = np.zeros(shape, bool)
    for road in roads:
        surface |= road.mask(shape)
    paving = surface.copy()
    gx, gz = np.meshgrid(np.arange(shape[0]) + .5, np.arange(shape[1]) + .5, indexing='ij')
    for x in GRID_X:
        for z in GRID_Z:
            surface |= np.hypot(gx - x - .5, gz - z - .5) <= 3.2
    for x, z in coords:
        if (x, z) == (90, 96): continue  # elevated service console, not a new pad
        identity = ('wildlands-pad', seed, x, z)
        # Fixtures occupy x-5..x+14, z-5..z+9; the outline is irregular beyond them.
        fixtures = rounded_box(shape, (x - 5, z - 5), (x + 15, z + 10), 2.5)
        assert not (fixtures & corridor).any(), (x, z)
        surface |= fixtures | (blob(shape, (x + 5, z + 2.5), (12.5, 10.5), identity) & ~corridor)
        paving |= blob(shape, (x + 1, z + .5), (5.5, 4.2), identity + ('court',), amplitude=.16)
    # Level rim around the gully, so its water is held by ground on every side.
    surface |= ndimage.binary_dilation(storm_gully(shape), iterations=2)
    return surface, paving & surface, roads


class WildlandsReferenceEnv(WildlandsEnv):
    commands = COMMANDS
    presentation_kind = 'wildlands_reference'
    world_features = frozenset({'riding', 'orchard'})

    def __init__(self, task='a', seed=7, *, render_enabled=False, wm_control=False):
        if task not in ('a','b'): raise ValueError('task must be a or b')
        self.task = task
        # Matched diagnostic: also post each WM cue at its use site. Never a default.
        self.wm_control = bool(wm_control)
        cfg = EnvConfig(seed=seed, starter_camp=False, max_steps=14000,
            obs_rgb=render_enabled, obs_text=True, reward_mode='sparse',
            world_cfg=WorldConfig(size_x=160,size_y=64,size_z=160,sea_level=12,
                relief=12,mountain_relief=10,landmarks=False),
            sim_cfg=SimConfig(weather_enabled=False,wind_enabled=False,start_time=START_TIME[task]),
            render_cfg=RenderConfig(width=320,height=180,third_person=True,
                cam_distance=7,cam_pitch_offset=-.16,max_dist=150))
        super().__init__(cfg)
        self.action_space['target'] = spaces.Discrete(256)
        self.action_space['vector'] = spaces.Box(-1,1,(3,),np.float32)

    def _author_relief(self, w, surface, coords):
        """Ridges, mesas, a lake and caves between and around the course roads,
        rising away from every road, court, the river and the mechanism cells
        (see owb/world/relief.py). Routes keep their walking level."""
        from ..world import relief as R
        identity = ('wildlands-reference', self._seed, self.task)
        shape = (w.size_x, w.size_z)
        keep = ndimage.binary_dilation(surface, iterations=2)
        keep |= ndimage.binary_dilation(RIVER.stamp(shape), iterations=3)
        keep |= ndimage.binary_dilation((w.vox == B.WATER).any(axis=1), iterations=2)
        xx, zz = np.ogrid[:shape[0], :shape[1]]
        for x, z in coords:
            keep |= (xx - x) ** 2 + (zz - z) ** 2 <= 10 ** 2
        for (x0, _, z0), (x1, _, z1) in (FLOOD, ((125, 0, 92), (133, 0, 100))):
            keep[x0 - 3:x1 + 4, z0 - 3:z1 + 4] = True
        keep |= (xx - STABLE[0]) ** 2 + (zz - STABLE[2]) ** 2 <= 6 ** 2
        lakes, made = R.basin_lakes(w, keep, identity, count=1, level=int(Y) - 3, ylo=13, yhi=60,
                                    distance=8.0, radii=(6, 9))
        keep |= ndimage.binary_dilation(lakes, iterations=2)
        T, G, mesa = R.raise_ground(w, keep, identity, ylo=13, yhi=60, cap=60, wavelength=.4,
                                    start=5.0, width=(6.0, 14.0))
        mouths = R.caves(w, T, G, keep, identity, count=2, ylo=14, length=10)
        raised = T > G
        self.relief_summary = dict(raised_fraction=float(raised.mean()), max_height=int(T.max()),
                                   relief_above_course=int(T.max()) - int(Y), lakes=[list(map(int, c)) for c in made],
                                   caves=mouths, mesa_fraction=float((mesa > .5).mean()))
        w.meta['relief'] = self.relief_summary

    def _generate_world(self):
        cfg = deepcopy(self.cfg.world_cfg); cfg.seed = self._seed
        result = generate(cfg)
        w = result.world
        # Organic access roads and courtyards within the intact procedural
        # landscape: curved roads, irregular pads and graded earth banks.
        # Rendering exports this exact collision world.
        coords = {p for _,_,p in (A if self.task=='a' else B_TASK)}
        surface, paving, roads = course_ground(w.vox.shape[::2], coords, self._seed)
        shape_course(w, surface, paving, 28, ('wildlands-reference', self._seed, self.task))
        w.meta['course_roads'] = [dict(points=r.points.round(3).tolist(), halfwidth=r.halfwidth.round(3).tolist())
                                  for r in roads]
        levels, river = carve_river(w, RIVER, surface, ('wildlands-reference', self._seed),
                                    sea_level=w.sea_level)
        w.meta['river'] = dict(points=RIVER.points.round(3).tolist(), halfwidth=RIVER.halfwidth.round(3).tolist(),
                               water_top=[int(v) for v in levels])
        # The return road starts flooded: a shallow dip whose water lies flush
        # with the road, held by low earth berms. Draining leaves silt (mud)
        # at road level.
        for x in (21, 27):
            for z in range(42, 49):
                h = int(top_solid(w)[x, z])
                w.vox[x, h+1:29, z] = B.DIRT
                w.vox[x, 28, z] = B.GRASS
        w.fill_box(*FLOOD_BED, B.DIRT)
        w.fill_box(*FLOOD, B.WATER)
        # A plank bridge across a rounded gully, plus an always-passable detour.
        for x, z in np.argwhere(storm_gully(w.vox.shape[::2])):
            w.vox[x, 23:29, z] = B.AIR
            w.vox[x, 23:26, z] = B.WATER
        w.fill_box((127,28,95),(131,28,97),B.PLANK)
        self._author_relief(w, surface, coords)
        settle_water(w)
        tops = water_top(w)
        w.meta['river']['water_top'] = [int(tops[int(np.clip(px, 0, 159)), int(np.clip(pz, 0, 159))])
                                        for px, pz in RIVER.points]
        assert (w.vox[22:27, 28, 43:48] == B.WATER).all(), 'flooded road drained during settling'
        # Dressed stone courtyards: columns, roofs, braziers and readable props.
        for x,z in coords:
            if (x,z)==(90,96): continue
            for dx in (-5,5):
                w.fill_box((x+dx,29,z+6),(x+dx+1,33,z+7),B.MOSS_STONE)
            w.fill_box((x-5,34,z+6),(x+6,34,z+7),B.BRICK)
            w.set_block(x-7,29,z+6,B.FLOWER)
            w.set_block(x+7,29,z-6,B.TALL_GRASS)
        # Preserve scenery outside authored footprints; clear random interactables
        # on the course so a gather command has an unambiguous nearby target.
        result.chests=[]; result.bodies=[]; result.plants=[]
        # Native creatures in the lowlands beside the course, and a saddled
        # horse tethered at the start court.
        result.npc_spawns=populate(w, surface, ('wildlands-reference', self._seed, self.task),
                                   sea_level=w.sea_level, wildlife_band=(5.0, 25.0), wildlife_scale=3)
        result.npc_spawns.insert(0, dict(kind='horse', pos=STABLE, home=(STABLE[0], STABLE[2]), roam=1.5, leash=3.0))
        result.landmarks=[Landmark('course','Reedbank reference course',(24.5,Y,24.5),4)]
        result.spawn=(22.5,Y+.01,24.5)
        w.temperature[:] = .15
        # The authored workshop/service routes are temperate training areas.
        # Preserve native exposure rules and procedural climate elsewhere.
        course_mask=ndimage.binary_dilation(surface,iterations=2)
        w.biome[course_mask]=0
        w.temperature[course_mask]=.7
        assert_water_contained(w)
        w.recompute_heightmaps()
        return result

    def _target(self,m,role,pos):
        tid=len(self.targets)+1
        obj=dict(id=tid,milestone=m.id,role=role,pos=np.array(pos,float),symbol='')
        self.targets[tid]=obj; m.targets[role]=tid
        return tid

    def _reset_mechanics(self):
        self.targets={}; self.completed=[]; self.satchel=Counter(); self.exposed={}
        self.success=False; self.storm=False; self.drained=False; self.road_seen=set()
        self.last_text=''; self.last_feedback=''; self.held=None; self.online=set(); self._pending=[]
        rng=np.random.default_rng(self._seed+8191)
        self.calibration=int(rng.integers(3)); self.authorized=int(rng.integers(3))
        self.basin_map=rng.permutation(3).tolist()
        self.family_symbols={kind:rng.permutation(['sun','wave','leaf','star','ring']).tolist()
                             for kind in ('balance','thermal','flow')}
        self.milestones=[]
        for i,(title,kind,site) in enumerate(A if self.task=='a' else B_TASK,1):
            mid=f'{self.task.upper()}{i:02}'
            m=Milestone(mid,title,kind,site,prereqs=(() if i==1 else (f'{self.task.upper()}{i-1:02}',)))
            self.milestones.append(m)
        self.by_id={m.id:m for m in self.milestones}
        if self.task=='b':
            for mid in ('B09','B10','B11'): self.by_id[mid].prereqs=('B08',)
            self.by_id['B12'].prereqs=('B09','B10','B11')
            for mid in ('B13','B16','B17'): self.by_id[mid].prereqs=('B12',)
            self.by_id['B19'].prereqs=('B15','B16','B18')
        # Persistent repair satchel, distinct from edible/equippable inventory.
        transfers=({'A05':((),('salvage_kit',)), 'A07':(('salvage_kit',),('regulator','cache_key')),
                    'A08':(('cache_key',),('service_handle',)), 'A11':(('regulator','service_handle'),())}
            if self.task=='a' else {'B01':((),('authorization',)), 'B10':((),('wind_coupling',)),
                'B11':((),('pump_coupling','engine_coupling')), 'B15':(('wind_coupling',),()),
                'B16':(('pump_coupling',),()), 'B18':(('engine_coupling',),()),
                'B20':((),('winch',)), 'B21':(('winch',),())})
        for mid,(inputs,outputs) in transfers.items():
            self.by_id[mid].consumes=inputs; self.by_id[mid].gives=outputs
        for m in self.milestones:
            x,z=m.site; y=Y
            m.state.update(done=False,installed=False,elapsed=0.,observed=False)
            if m.kind=='balance':
                count=1 if m.id in ('A02','B02') else 2
                m.state.update(braked=False,released=False,stable=0.,count=count,load=0.,bodies=[],lift_level=0)
                # Controls move between fixtures; symbols retain family meaning.
                self._target(m,'brake',(x-1,y+1,z-2))
                self._target(m,'release',(x+1,y+1,z-2))
                self._target(m,'install',(x,y+1,z-2))
                self._target(m,'inspect',(x,y+1,z-3))
                m.state['slots']=[(x+3.,y+.501,z+2.),(x+3.,y+.501,z+3.5)]
                for k in range(count):
                    bid=self.bodies.add('metal_slab',(x-2.+k*1.4,y+.501,z+2.))
                    m.state['bodies'].append(bid)
                    self._target(m,f'weight{k}',self.bodies.get(bid).pos)
                self.world.fill_box((x+3,28,z+1),(x+4,28,z+4),B.MOSS_STONE)
                self.world.fill_box((x+6,29,z-2),(x+6,32,z+2),B.PLANK)
                if m.id in ('A02','B02'):
                    self.world.fill_box((x+7,29,z-2),(x+10,29,z+2),B.BRICK)
            elif m.kind=='thermal':
                m.state.update(vent=False,heat=False,updraft_seen=False,landed=False,
                               hot=m.id=='B18',safe=m.id!='B18',launch_seen=False)
                self._target(m,'vent',(x-2,y+1,z+3))
                self._target(m,'heat',(x,y+1,z+3))
                self._target(m,'exposed_heat',(x+2,y+1,z+3))
                self._target(m,'inspect',(x-2,y+1,z-3))
                m.state['landing']=(x+10.5,Y+4.01,z+.5)
                self.world.fill_box((x+8,29,z-3),(x+13,32,z+3),B.MOSS_STONE)
                self.world.fill_box((x+8,32,z-3),(x+13,32,z+3),B.BRICK)
                # A raised, sheltered source sits beside (not in) the flight path.
                self.world.fill_box((x+1,28,z+3),(x+3,28,z+5),B.BRICK)
                self.world.set_block(x+2,29,z+4,B.CAMPFIRE)
                self.world.fill_box((x-1,35,z+3),(x+4,35,z+5),B.PLANK)
                if m.consumes: self._target(m,'install',(x+11,34,z))
            elif m.kind in ('flow','cool','prepare'):
                m.state.update(outlets=[False]*3,inlet=False,stable=0.,temperature=90.,trips=0)
                expected=(self.basin_map.index(self.authorized) if m.kind=='prepare' else int(rng.integers(3)))
                m.state['expected']=expected
                self._target(m,'inlet',(x-3,y+1,z-3))
                for k in range(3): self._target(m,f'outlet{k}',(x-2+k*2,y+1,z-3))
                self._target(m,'inspect',(x,y+1,z-4))
                self._target(m,'install',(x+3,y+1,z-3))
            else:
                elevated=m.id in ('A11','B15')
                y=Y+4 if elevated else Y
                self._target(m,'inspect' if m.kind=='clue' else 'interact',(x+.5,y+1,z+.5))
                if m.kind=='calibrate':
                    self._target(m,'install',(x+.5,y+1,z+.5))
                    for k in range(3): self._target(m,f'mark{k}',(x-2+k*2,y+1,z+2))
                    m.state['selected']=None
                if m.kind=='cook':
                    self.world.set_block(x,29,z+2,B.COOKPOT)
                    self.world.set_block(x-2,29,z+2,B.CAMPFIRE)
                    self.gameplay.plants += [dict(pos=(x+.5,29,z-.5),item='spicy_pepper',count=2),
                                             dict(pos=(x+1.5,29,z-.5),item='chillshroom',count=2)]
                    m.state['cooked']=[]
        if self.task=='b':
            # The final inlet is physically on the elevated thermal gallery.
            inlet=self.targets[self.by_id['B22'].targets['inlet']]
            inlet['pos']=np.array((119,34,96.5),float)
            # Ordinary enemies guard the yard; the safe reference uses the back.
            self.npcs.spawn('bokoblin',(62.5,Y,67.5),home=(62.5,67.5))
        for m in self.milestones:
            family=m.kind if m.kind in ('balance','thermal') else 'flow'
            for k,tid in enumerate(m.targets.values()):
                self.targets[tid]['symbol']=self.family_symbols[family][k%5]
        # Added last, so every earlier target keeps its ID and symbol.
        self._setup_working_memory()
        self.world.recompute_heightmaps()
        self.agent.pitch=-.08
        self.agent.inventory.add('rune_magnet',1)
        # Nights are cold; a packed doublet must be put on (equip) after dusk.
        self.agent.inventory.add('warm_doublet',1)
        base=self.elements.updraft_at
        def updraft(pos):
            best=base(pos)
            for m in self.milestones:
                if m.kind!='thermal': continue
                x,z=m.site; s=m.state
                if (s['vent'] and s['heat'] and s['safe'] and x-1<=pos[0]<=x+8
                        and abs(pos[2]-(z+.5))<3 and Y<=pos[1]<Y+9):
                    best=max(best,8.)
            return best
        self.elements.updraft_at=updraft

    @property
    def current(self):
        done={r['id'] for r in self.completed}
        return next((m for m in self.milestones if m.id not in done),None)

    def eligible(self,m):
        return set(m.prereqs)<={r['id'] for r in self.completed}

    def _fail(self,reason,**kw): return [dict(type='action_failed',reason=reason,**kw)]

    def _reachable(self,pos,radius=3.2):
        delta=np.asarray(pos)-self.agent.eye; distance=float(np.linalg.norm(delta))
        if distance>radius: return False
        hit=self.world.raycast(self.agent.eye,delta,max(0,distance-.35))
        return distance<=radius and not hit.hit

    def _dispatch(self,command,ingredients,action):
        self.last_text = ''
        tid=int(action['target']); obj=self.targets.get(tid)
        if command=='drag':
            if self.held is None: return self._fail('no held magnetic body')
            body=self.bodies.get(self.held)
            delta=np.asarray(action['vector'],float)*.4
            candidate=body.pos+delta
            if np.linalg.norm(candidate-self.agent.eye)>6: return self._fail('magnetic tether range')
            if box_collides(self.world,candidate,body.half): return self._fail('body collision')
            if any(np.all(np.abs(candidate-b.pos)<(body.half+b.half)-.02)
                   for b in self.bodies.bodies.values() if b.id!=body.id):
                return self._fail('another body blocks movement')
            body.pos=candidate; body.vel[:]=0
            return [dict(type='magnetic_move',body=body.id,pos=plain(candidate))]
        if command=='drop':
            if self.held is None: return self._fail('no held magnetic body')
            bid=self.held; self.bodies.get(bid).held=False; self.held=None
            return [dict(type='magnetic_drop',body=bid)]
        if command in ('inspect','operate','grip','install') or (command=='interact' and tid):
            if obj is None: return self._fail('unknown interaction target')
            m=self.by_id[obj['milestone']]; role=obj['role']; s=m.state
            if role.startswith('weight'):
                body=self.bodies.get(s['bodies'][int(role[6:])]); pos=body.pos
            else: pos=obj['pos']
            if not self._reachable(pos,4 if command=='grip' else 3.2):
                return self._fail('target out of reach or occluded',target=tid)
            if obj.get('wm'): return self._wm_interaction(command,obj,m)
            if command=='inspect':
                text=self.clue(m)
                self.last_text=text; s['observed']=True; self.exposed[m.id]=self.elapsed
                return [dict(type='clue_exposed',milestone=m.id,text=text)]
            if not self.eligible(m): return self._fail('mechanism prerequisites incomplete',target=tid)
            if command=='grip':
                if not role.startswith('weight') or self.held is not None: return self._fail('cannot grip target')
                body.held=True; body.vel[:]=0; self.held=body.id
                return [dict(type='magnetic_grip',body=body.id)]
            if command=='install':
                if role!='install': return self._fail('not an installation socket')
                if s['installed']: return self._fail('parts already installed')
                if m.kind=='balance' and not s['released']: return self._fail('support lock before installation')
                if m.kind=='thermal' and not s['landed']: return self._fail('reach service deck first')
                if any(self.satchel[k]<1 for k in m.consumes): return self._fail('required repair parts missing')
                for k in m.consumes: self.satchel[k]-=1
                s['installed']=True
                return [dict(type='parts_installed',milestone=m.id,parts=list(m.consumes))]
            if m.kind=='balance':
                if role=='brake':
                    s['braked']=True; s['released']=False; s['stable']=0
                elif role=='release':
                    if not s['braked'] or not self._valid_load(m):
                        s['released']=False; s['stable']=0
                        return [dict(type='brake_trip',milestone=m.id,load=s['load'])]
                    s['released']=True
                else: return self._fail('not a counterweight control')
                return [dict(type='counterweight_control',milestone=m.id,control=role,load=s['load'])]
            if m.kind=='thermal':
                if role=='vent': s['vent']=not s['vent']
                elif role=='heat':
                    if not s['vent']: return [dict(type='draft_blocked',milestone=m.id)]
                    if not s['safe']: return self._fail('plume above safe launch temperature')
                    s['heat']=True
                    x,z=m.site; self.elements.ignite(self.world,x+2,29,z+4)
                elif role=='exposed_heat': return [dict(type='draft_failed',reason='exposed hearth cannot sustain this shaft')]
                else: return self._fail('not a shaft control')
                return [dict(type='thermal_control',milestone=m.id,control=role)]
            if m.kind in ('flow','cool','prepare'):
                if role.startswith('outlet'):
                    if s['inlet']: return self._fail('isolate inlet before rerouting')
                    i=int(role[-1]); s['outlets'][i]=not s['outlets'][i]
                elif role=='inlet':
                    if s['inlet']: s['inlet']=False
                    elif s['outlets']!=[k==s['expected'] for k in range(3)]:
                        s['trips']+=1; s['stable']=0
                        return [dict(type='pressure_trip',milestone=m.id)]
                    else: s['inlet']=True
                else: return self._fail('not a flow control')
                return [dict(type='flow_control',milestone=m.id,control=role,value=s['inlet'] if role=='inlet' else s['outlets'][int(role[-1])])]
            if m.kind=='calibrate':
                if role.startswith('mark'): s['selected']=int(role[-1]); return [dict(type='calibration_selected',value=s['selected'])]
                if not s['installed']: return self._fail('install both repair parts first')
                if s['selected']!=self.calibration: return [dict(type='calibration_trip')]
                s['engaged']=True; return [dict(type='intake_engaged')]
            if m.kind=='station':
                if any(self.satchel[k]<1 for k in m.consumes): return self._fail('required coupling missing')
                if not s['installed']:
                    for k in m.consumes: self.satchel[k]-=1
                    s['installed']=True
                return [dict(type='station_engaged',milestone=m.id)]
            if m.kind=='exchange':
                if s['done']: return self._fail('locker empty')
                if m.id=='A07' and not self.wm['WM-WA07'].solved: return self._fail('locker keypad locked')
                if any(self.satchel[k]<1 for k in m.consumes): return self._fail('exchange input missing')
                if m.id in ('B10','B11') and not self.satchel['authorization']: return self._fail('commission required')
                for k in m.consumes: self.satchel[k]-=1
                for k in m.gives: self.satchel[k]+=1
                s['done']=True
                if m.id=='B11':
                    # The couplings themselves wait in the numbered bays (WM-WB11).
                    self.wm_world['bays_released']=True
                    return [dict(type='bays_released',milestone=m.id)]
                return [dict(type='item_exchange',milestone=m.id,inputs=m.consumes,outputs=m.gives)]
            if m.kind=='report':
                if not self.drained or self.road_seen!={'north','south'}: return self._fail('verify the drained road first')
                s['done']=True; return [dict(type='completion_signal')]
            return self._fail('inspect this source instead')
        return super()._dispatch(command,ingredients,action)

    def _command(self,command,ingredients,action):
        events=super()._command(command,ingredients,action)
        tid=int(action['target'])
        if command in INTERACTIONS or (command=='interact' and tid):
            self.last_feedback=self._feedback(command,tid,events)
        return events

    def _feedback(self,command,tid,events):
        """One player-facing line for the latest interaction (see REFUSALS)."""
        obj=self.targets.get(tid); said=[]
        for e in events:
            t=e['type']; m=self.by_id.get(e.get('milestone'))
            if t=='action_failed':
                r=e.get('reason','')
                said.append('nothing happens' if r in NO_EFFECT else REFUSALS.get(r,'nothing happens'))
            elif t=='counterweight_control':
                said.append('the brake locks' if e['control']=='brake' else 'the brake releases; the lift takes the load')
            elif t=='brake_trip':
                said.append(f"the brake trips; the support plate reads {e.get('load',m.state['load'] if m else 0):g} kg")
            elif t=='magnetic_grip': said.append('the magnet grips the slab')
            elif t=='magnetic_drop': said.append('the slab is released')
            elif t=='magnetic_move': said.append('the slab moves')
            elif t=='parts_installed': said.append('installed '+', '.join(e['parts']))
            elif t=='thermal_control':
                said.append(('the vent opens' if m.state['vent'] else 'the vent closes') if e['control']=='vent'
                            else 'the hearth lights')
            elif t=='draft_blocked': said.append('the heat has no draft to rise through')
            elif t=='draft_failed': said.append('the exposed hearth flares but cannot hold a draft')
            elif t=='flow_control': said.append('the valve opens' if e['value'] else 'the valve closes')
            elif t=='pressure_trip': said.append('pressure trips the inlet shut')
            elif t=='calibration_selected': said.append(f"mark {e['value']+1} selected")
            elif t=='calibration_trip': said.append('the calibration trips')
            elif t=='intake_engaged': said.append('the intake engages')
            elif t=='station_engaged': said.append('the station engages')
            elif t=='item_exchange': said.append(f"exchanged {list(e['inputs']) or 'nothing'} for {list(e['outputs'])}")
            elif t=='completion_signal': said.append('report delivered')
            elif t=='clue_exposed': said.append('you read it')
            elif t=='wm_cue_read': said.append(f"you read the {WM_NAMES[e['probe']]}")
            elif t=='fixture_read': said.append('you look it over')
            elif t=='wm_entry':
                said.append('the key clicks' if e['probe']=='WM-WA07' else 'the dial clicks')
            elif t=='wm_attempt' and e['probe']=='WM-WA07':
                said.append('the keypad chimes and the locker bolt slides back' if e['correct']
                            else 'the keypad buzzes and clears')
            elif t=='wm_attempt' and e['probe']=='WM-WB15':
                said.append('the blades bite the wind and the rotor steadies' if e['correct']
                            else 'the blades feather and the dial springs back to blade 1')
            elif t=='bays_released': said.append('the yard office stamps your authorization and releases the coupling bays')
            elif t=='bay_opened':
                part=e['contents']
                said.append(f"bay {e['bay']} swings open: it holds a {part.replace('_',' ')}; you take it" if part
                            else f"bay {e['bay']} swings open: it is empty")
            elif t=='bay_looked': said.append(f"bay {e['bay']} stands open and empty")
            elif t=='wm_noop': said.append('nothing more happens')
        name=(f"{command} {tid}"+(f":{obj['symbol']}" if obj else '')) if tid else command
        return f"Last interaction: {name} -> {'; '.join(said) or 'nothing happens'}."

    def _extra_entities(self):
        """Policy-view impostors for fixtures that have no voxels: symbol-painted
        control posts (plaques are short and broad) and stakes at the corners of
        each counterweight support plate. Presentation only; no collision."""
        out=super()._extra_entities()
        out+=self._wm_entities()
        for obj in self.targets.values():
            if obj['role'].startswith('weight') or obj.get('wm'): continue
            x,y,z=(float(v) for v in obj['pos']); plaque=obj['role']=='inspect'
            h,r=(.8,.2) if plaque else (1.05,.13)
            out.append(dict(pos=(x,y-1+h/2,z),radius=r,height=h,shape='box',
                            color=SYMBOL_COLOURS.get(obj['symbol'],(150,150,150)),semantic=B.SEM_INTERACTIVE))
        for m in self.milestones:
            if m.kind!='balance': continue
            x,z=m.site
            for cx in (x+2.4,x+4.7):
                for cz in (z+.5,z+4.7):
                    out.append(dict(pos=(cx,Y+.3,cz),radius=.08,height=.6,shape='box',
                                    color=STAKE_COLOUR,semantic=B.SEM_INTERACTIVE))
        return out

    # ---------------- working-memory probes (docs/working-memory-probes.md) ----------------
    def _wm_target(self,m,role,pos,label,probe,fixture,value=None):
        tid=self._target(m,role,pos)
        self.targets[tid].update(symbol=label,wm=probe,fixture=fixture,value=value)
        return tid

    def _setup_working_memory(self):
        """Draw each probe's content from its own generator and build its fixtures.

        Visible labels name a fixture but never carry the cue. The stores room and
        the coupling-bay store are collision geometry; boards, keypad, dial and
        blade tags are non-colliding fixtures like the other control posts."""
        seed,control=self._seed,self.wm_control
        self.wm={}; self.wm_world={}; self.wm_cue=None
        w=self.world
        if self.task=='a':
            p=WorkingMemoryProbe.draw('WM-WA07','A07','hold',seed,KEYPAD,4,control=control)
            self.wm[p.probe]=p; m=self.by_id['A07']
            x0,z0,x1,z1=STORES; F=int(STORES_FLOOR)
            w.fill_box((x0,29,z0),(x1,F-1,z1),B.STONE)             # loading dock
            w.fill_box((x0,F,z0),(x1,F+1,z1),B.BRICK)
            w.fill_box((x0+1,F,z0+1),(x1-1,F+1,z1-1),B.AIR)
            w.fill_box((x0,F+2,z0),(x1,F+2,z1),B.PLANK)            # roof
            for rz,gap in STORES_RACKS:
                w.fill_box((x0+1,F,rz),(x1-1,F+1,rz),B.PLANK)
                w.fill_box((gap,F,rz),(gap,F+1,rz),B.AIR)
            dx,dz=STORES_DOOR
            w.fill_box((dx,F,dz),(dx,F+1,dz),B.AIR)
            self.targets[m.targets['interact']]['pos']=np.array(LOCKER,float)
            self._wm_target(m,'inspect',WA07_BOARD,'dispatch board',p.probe,'source')
            for i,d in enumerate(KEYPAD):
                self._wm_target(m,f'key{d}',(119.05,F+1.3-.25*(i//5),63.1+.2*(i%5)),f'key {d}',p.probe,'key',d)
            if control: self._wm_target(m,'inspect_copy',WA07_COPY,'dispatch board copy',p.probe,'copy')
            self.wm_world['panel']=(119.05,F+1.2,63.5)
        else:
            p=WorkingMemoryProbe.draw('WM-WB11','B11','select',seed,range(1,len(BAY_X)+1),2,control=control)
            self.wm[p.probe]=p; m=self.by_id['B11']
            # The couplings wait in the bays; the yard office only releases them.
            m.gives=()
            rng=p.rng(seed)
            labels=[BAY_LABELS[int(i)] for i in rng.permutation(np.repeat(np.arange(len(BAY_LABELS)),2))]
            x0,z0,x1,z1=BAY_STORE
            w.fill_box((x0,29,z0),(x1,30,z1),B.BRICK)
            w.fill_box((x0+1,29,z0+1),(x1-1,30,z1-1),B.AIR)
            w.fill_box((x0+1,29,z1),(x1-1,30,z1),B.PLANK)       # one plank door per bay
            w.fill_box((x0,31,z0),(x1,31,z1),B.PLANK)
            for k,x in enumerate(BAY_X,1):
                self._wm_target(m,f'bay{k}',(x+.5,Y+1,z1+1.05),f'bay {k} {labels[k-1]}',p.probe,'bay',k)
            self._wm_target(m,'inspect',WB11_MANIFEST,'yard manifest',p.probe,'source')
            if control: self._wm_target(m,'inspect_copy',WB11_COPY,'yard manifest copy',p.probe,'copy')
            self.wm_world.update(bays_released=False,bays_open=[],bay_labels=labels)

            q=WorkingMemoryProbe.draw('WM-WB15','B15','remap',seed,PITCHES,3,control=control)
            self.wm[q.probe]=q; m=self.by_id['B15']
            # Blade numbers carry a colour-tag assignment drawn from the probe's layout
            # stream; never the board's listing order, so the remap is always needed.
            orders=[o for o in permutations(range(len(BLADE_COLOURS))) if o!=tuple(range(len(BLADE_COLOURS)))]
            order=orders[int(q.rng(seed).integers(len(orders)))]
            blades=[BLADE_COLOURS[i] for i in order]
            q.set_expected(q.items[BLADE_COLOURS.index(c)] for c in blades)
            # A parapet along the deck's east edge keeps the dial and rotor out of
            # sight from the weather board's reading spots below.
            w.fill_box(DECK_PARAPET[0],DECK_PARAPET[1],B.BRICK)
            self._wm_target(m,'inspect',WB15_BOARD,'weather board',q.probe,'source')
            for k,pos in enumerate(DIAL): self._wm_target(m,f'pitch{PITCHES[k]}',pos,f'pitch {PITCHES[k]}',q.probe,'dial',PITCHES[k])
            for j,c in enumerate(blades):
                self._wm_target(m,f'blade{j+1}',self._blade_tag(j),f'blade {j+1} {c} tag',q.probe,'blade',j+1)
            if control: self._wm_target(m,'inspect_copy',WB15_COPY,'weather board copy',q.probe,'copy')
            self.wm_world.update(blade_colours=blades,panel=DIAL[len(DIAL)//2])

    @staticmethod
    def _blade_tag(j,radius=.55):
        angle=np.pi/2+j*2*np.pi/3
        return (ROTOR_HUB[0],ROTOR_HUB[1]+radius*np.sin(angle),ROTOR_HUB[2]+radius*np.cos(angle))

    def wm_cue_text(self,probe):
        p=self.wm[probe]
        if probe=='WM-WA07': return WM_TEXT[probe].format(code=' '.join(p.items))
        if probe=='WM-WB11': return WM_TEXT[probe].format(pump=p.items[0],engine=p.items[1])
        return WM_TEXT[probe].format(**dict(zip(BLADE_COLOURS,p.items)))

    def _wm_ready(self,m):
        p=next((p for p in self.wm.values() if p.milestone==m.id),None)
        return p is None or p.solved

    def _wm_interaction(self,command,obj,m):
        p=self.wm[obj['wm']]; fixture=obj['fixture']; tid=obj['id']
        if command=='inspect':
            if fixture in ('source','copy'):
                self.wm_cue=dict(probe=p.probe,target=tid,since=round(self.elapsed,6),text=self.wm_cue_text(p.probe))
                return [p.read_cue(self.steps,self.elapsed)]
            self.last_text=self._wm_describe(obj)
            return [dict(type='fixture_read',target=tid)]
        if fixture in ('source','copy'): return self._fail('inspect this source instead')
        if fixture=='blade': return [dict(type='wm_noop',target=tid)]
        if command not in ('operate','interact'): return [dict(type='wm_noop',target=tid)]
        if not self.eligible(m): return self._fail('mechanism prerequisites incomplete',target=tid)
        if fixture in ('key','dial'):
            events=p.enter(obj['value'],self.steps,self.elapsed)
            return events or [dict(type='wm_noop',target=tid)]
        # A numbered bay: opening it is the selection; a wrong bay is simply empty.
        k=obj['value']; world=self.wm_world
        if not world['bays_released']: return self._fail('bays not released',target=tid)
        if k in world['bays_open']: return [dict(type='bay_looked',bay=k)]
        world['bays_open'].append(k)
        x=BAY_X[k-1]; self.world.fill_box((x,29,BAY_STORE[3]),(x,30,BAY_STORE[3]),B.AIR)
        parts=('pump_coupling','engine_coupling')
        slot=next((i for i,b in enumerate(p.expected) if b==k),None)
        contents=parts[slot] if slot is not None else None
        if slot is None: slot=next((i for i,done in enumerate(p.slot_done) if not done),None)
        events=[dict(type='bay_opened',bay=k,contents=contents)]
        if contents: self.satchel[contents]+=1
        if slot is not None: events+=p.choose(slot,k,self.steps,self.elapsed)
        return events

    def _wm_describe(self,obj):
        f,v=obj['fixture'],obj['value']
        if f=='key': return f'A keypad key marked {v}.'
        if f=='dial': return f'A pitch dial stop marked {v} degrees. The dial sets blade 1, then 2, then 3.'
        if f=='blade': return f"Blade {v} carries a {self.wm_world['blade_colours'][v-1]} tag."
        if f=='bay': return f"Bay {v}, stencilled '{self.wm_world['bay_labels'][v-1]}'."
        return ''

    def _wm_tick(self):
        """The cue clears on leaving reach or after five simulated seconds; a partial
        keypad/dial entry clears when the agent walks away from the panel."""
        cue=self.wm_cue
        if cue is not None:
            if (self.elapsed-cue['since']>=WM_CUE_SECONDS-1e-9
                    or not self._reachable(self.targets[cue['target']]['pos'])):
                self.wm_cue=None
        for p in self.wm.values():
            if p.buffer and np.linalg.norm(self.agent.pos-np.asarray(self.wm_world['panel']))>4.5:
                p.clear()

    def working_memory_summary(self):
        """Evaluator-only: per-probe summary (contract section 7)."""
        return wm_summaries(self.wm[k] for k in sorted(self.wm))

    def working_memory_layout(self):
        """Evaluator-only: source and use-site positions, and the drawn content."""
        out={}
        for k,p in sorted(self.wm.items()):
            fixtures=[o for o in self.targets.values() if o.get('wm')==k]
            source=next(o for o in fixtures if o['fixture']=='source')
            use=[o for o in fixtures if o['fixture'] in ('key','bay','dial')]
            centre=np.mean([o['pos'] for o in use],axis=0)
            out[k]=dict(milestone=p.milestone,kind=p.kind,load=p.load,source=plain(source['pos']),
                        use_site=plain(np.round(centre,3)),
                        straight_line_m=round(float(np.linalg.norm(centre-source['pos'])),2),
                        items=list(p.items),expected=list(p.expected),cue=self.wm_cue_text(k),
                        control=p.control)
            if k=='WM-WB11': out[k]['bay_labels']=list(self.wm_world['bay_labels'])
            if k=='WM-WB15': out[k]['blade_colours']=list(self.wm_world['blade_colours'])
        return out

    def _wm_entities(self):
        out=[]
        for obj in self.targets.values():
            if not obj.get('wm'): continue
            x,y,z=(float(v) for v in obj['pos']); f=obj['fixture']
            if f in ('source','copy'):
                name=obj['symbol'].replace(' copy','')
                out.append(dict(pos=(x,y-.15,z),radius=.5,height=.95,shape='box',color=WM_COLOURS[name],semantic=B.SEM_INTERACTIVE))
            elif f=='bay':
                out.append(dict(pos=(x,y+.55,z-.02),radius=.22,height=.16,shape='box',color=WM_COLOURS['plate'],semantic=B.SEM_INTERACTIVE))
            elif f=='blade':
                colour=self.wm_world['blade_colours'][obj['value']-1]
                out.append(dict(pos=(x,y,z),radius=.12,height=.22,shape='box',color=WM_COLOURS[colour],semantic=B.SEM_INTERACTIVE))
        if self.task=='a':   # the ladder up the stores dock
            dx,dz=STORES_DOOR
            out.append(dict(pos=(dx+1.08,Y+1.5,dz+.5),radius=.32,height=3.0,shape='box',color=(120,88,52),semantic=B.SEM_INTERACTIVE))
        panel=self.wm_world.get('panel')
        if panel is not None:
            name='keypad' if self.task=='a' else 'dial'
            out.append(dict(pos=tuple(panel),radius=.2,height=.5,shape='box',
                            color=WM_COLOURS[name],semantic=B.SEM_INTERACTIVE))
        return out

    def _wm_meshes(self,batch):
        """Cycles presentation of the WM fixtures. Appearance variation is keyed to
        stable fixture identity, never to the simulation RNG; state (open bays,
        entered digits, a solved panel) is read, not changed."""
        from open_world_bench.render.assets import appearance_seed
        wood,dark,metal,paper=(.42,.30,.18),(.22,.16,.10),(.36,.37,.38),(.90,.87,.78)
        tags={'red':(.78,.14,.10),'yellow':(.90,.72,.12),'blue':(.14,.30,.78)}
        def rng(*identity): return np.random.default_rng(appearance_seed(('wildlands-wm',self._seed,self.task)+identity))
        def tint(r,c,a=.06): return tuple(np.clip(np.asarray(c)*(1+r.uniform(-a,a,3)),0,1))
        def sign(pos,name,face):
            # Two posts, a framed board and a shingled roof; face = +-1 along x or z.
            x,y,z=pos; r=rng('board',name,round(x,1),round(z,1)); along=abs(face)==2
            w=(0,0,.75) if not along else (.75,0,0)
            for s in (-1,1):
                foot=np.array((x,Y,z))+s*np.array(w)
                batch.current_material=2
                batch.tube(foot,foot+(0,2.25,0),.07,tint(r,dark))
            batch.current_material=2
            size=(.08,1.0,1.5) if not along else (1.5,1.0,.08)
            batch.box((x,y+.25,z),size,tint(r,wood))
            batch.box((x,y+.82,z),(size[0]+.06,.08,size[2]+.1),tint(r,dark))
            batch.box((x,y+1.27,z),(size[0]+.5,.08,size[2]+.4),tint(r,(.30,.22,.16)))
            batch.current_material=5
            n=int(r.integers(3,6)); off=np.sign(face)*.05
            for i in range(n):
                u=-.55+1.1*(i+.5)/n+r.uniform(-.05,.05); v=r.uniform(-.2,.35)
                c=tint(r,paper,.08); dims=(.012,r.uniform(.28,.42),r.uniform(.18,.26))
                if along: batch.box((x+u,y+.2+v,z+off),(dims[2],dims[1],.012),c)
                else: batch.box((x+off,y+.2+v,z+u),dims,c)
            if name=='weather board':
                for i,c in enumerate(BLADE_COLOURS):
                    u=-.45+.45*i
                    if along: batch.box((x+u,y-.12,z+off*1.3),(.2,.14,.015),tags[c])
                    else: batch.box((x+off*1.3,y-.12,z+u),(.015,.14,.2),tags[c])
                batch.current_material=4
                batch.tube((x,y+1.08,z),(x,y+1.6,z),.025,metal)
                batch.box((x,y+1.55,z),(.5,.05,.05),metal); batch.cone((x+.27,y+1.5,z),.06,.12,metal,6)
            if name=='dispatch board':
                batch.box((x+off*1.2,y+.66,z) if not along else (x,y+.66,z+off*1.2),
                          (.014,.1,1.3) if not along else (1.3,.1,.014),(.62,.12,.10))
            batch.current_material=0
        p=self.wm
        if self.task=='a':
            r=rng('stores'); F=STORES_FLOOR; dx,dz=STORES_DOOR
            sign(WA07_BOARD,'dispatch board',2)
            if self.wm_control: sign(WA07_COPY,'dispatch board',1)
            # A wooden ladder on the dock face below the doorway, and crates on the dock.
            batch.current_material=2
            for side in (-.32,.32):
                batch.tube((dx+1.06,Y,dz+.5+side),(dx+1.06,F+.9,dz+.5+side),.04,tint(r,dark))
            for k in range(9):
                batch.tube((dx+1.06,Y+.3+.32*k,dz+.18),(dx+1.06,Y+.3+.32*k,dz+.82),.025,tint(r,wood))
            for k,(cx,cz) in enumerate(((dx+1.6,dz-1.4),(dx+1.7,dz+2.2),(dx+2.4,dz-1.1))):
                batch.box((cx,Y+.3,cz),(.6,.6,.6),tint(r,(.55,.40,.22),.12),yaw=r.uniform(-.3,.3))
            # Rack shelves: boards and stock along each rack row.
            x0,z0,x1,z1=STORES
            for rz,gap in STORES_RACKS:
                for x in range(x0+1,x1):
                    if x==gap or r.random()<.35: continue
                    for face in (-.06,1.06):
                        batch.box((x+.5,F+.55+r.choice((0,.7)),rz+face),(.32,.3,.1),tint(r,(.55,.40,.22),.14))
            # Steel locker cabinet against the east wall; the keypad is on its west face.
            batch.current_material=4
            batch.box((119.55,F+.95,63.5),(.8,1.9,.9),tint(r,(.34,.40,.42)))
            batch.box((119.14,F+.95,63.5),(.02,1.7,.03),(.2,.22,.23))
            batch.box((119.13,F+.7,63.85),(.04,.3,.05),(.65,.65,.6))
            batch.box((119.12,F+1.17,63.5),(.04,.62,.95),(.12,.13,.14))
            pr=p['WM-WA07']
            for o in self.targets.values():
                if o.get('fixture')!='key': continue
                kx,ky,kz=o['pos']; pressed=len(pr.buffer)>0 and o['value']==pr.buffer[-1]
                batch.box((kx+(.09 if pressed else .07),ky,kz),(.06,.16,.16),(.80,.78,.70))
            lamp=(.25,.85,.35) if pr.solved else (.85,.35,.15)
            batch.current_material=0
            batch.box((119.09,F+1.45,63.5),(.03,.06,.06),lamp,emissive=1.5)
            return
        # Task B. Yard manifest lectern at the yard gate.
        x,y,z=WB11_MANIFEST; r=rng('manifest')
        batch.current_material=2
        batch.tube((x,Y,z),(x,Y+1.05,z),.06,tint(r,dark))
        batch.box((x,Y+.02,z),(.5,.04,.5),tint(r,dark))
        batch.box((x-.05,Y+1.12,z),(.55,.06,.75),tint(r,wood))
        batch.current_material=5
        batch.box((x-.05,Y+1.16,z),(.42,.02,.6),tint(r,paper)); batch.box((x-.05,Y+1.18,z-.2),(.36,.01,.04),(.25,.25,.28))
        batch.current_material=0
        if self.wm_control: sign(WB11_COPY,'yard manifest',2)
        # Bay store: number plates (k studs for bay k), stencil strips and doors.
        x0,z0,x1,z1=BAY_STORE; world=self.wm_world
        stripe={'pump fittings':(.32,.40,.48),'engine fittings':(.52,.30,.18),'couplings':(.30,.44,.30)}
        for k,bx in enumerate(BAY_X,1):
            r=rng('bay',k); front=z1+1.0
            batch.current_material=4
            batch.box((bx+.5,Y+2.2,front+.02),(.7,.26,.04),tint(r,(.78,.78,.72)))
            for s in range(k):
                batch.box((bx+.5-.25+.5*(s+.5)/k,Y+2.2,front+.05),(.06,.1,.03),(.15,.15,.16))
            batch.current_material=5
            batch.box((bx+.5,Y+1.95,front+.02),(.8,.1,.03),stripe[world['bay_labels'][k-1]])
            batch.current_material=2
            if k in world['bays_open']:
                batch.box((bx+.05,Y+1.0,front+.45),(.06,1.9,.9),tint(r,wood))      # door swung outwards
                batch.box((bx+.5,Y+.15,z1-.6),(.6,.3,.5),tint(r,(.45,.33,.2),.1))   # an empty crate inside
            batch.current_material=0
        # Wind deck: weather board at the base, pitch dial and tagged rotor above.
        sign(WB15_BOARD,'weather board',1)
        if self.wm_control: sign(WB15_COPY,'weather board',2)
        q=p['WM-WB15']; r=rng('rotor')
        batch.current_material=4
        batch.tube((92.6,Y+4,95.9),(92.6,Y+4.9,95.9),.08,metal)
        batch.box((92.75,Y+5.2,96.0),(.08,.36,1.6),(.18,.19,.2))
        for o in self.targets.values():
            if o.get('fixture')!='dial': continue
            dx,dy,dz=o['pos']; batch.box((dx+.08,dy,dz),(.06,.14,.14),(.82,.80,.72))
        k=len(q.buffer); batch.tube((92.68,Y+5.2,95.4),(92.68,Y+5.2+.18,95.4+.3*k),.02,(.75,.2,.15))
        batch.current_material=0
        batch.box((92.66,Y+5.42,96.7),(.03,.06,.06),(.25,.85,.35) if q.solved else (.85,.35,.15),emissive=1.5)
        hx,hy,hz=ROTOR_HUB
        station=self.by_id['B15'].state
        spin=(self.elapsed*2.4) if (q.solved and station['installed']) else 0.0
        batch.current_material=4
        batch.tube((hx,Y+4,hz),(hx,hy,hz),.09,metal)
        batch.ellipsoid((hx+.15,hy,hz),(.35,.2,.2),tint(r,(.62,.64,.62)))
        for j,colour in enumerate(world['blade_colours']):
            angle=np.pi/2+j*2*np.pi/3+spin; d=np.array((0,np.sin(angle),np.cos(angle)))
            root=np.array((hx-.05,hy,hz))
            batch.tube(root+.15*d,root+1.7*d,.07,tint(r,(.86,.86,.82)),end_radius=.03)
            for s in range(j+1):     # blade number: j+1 dark bands at the root
                c=root+(.28+.07*s)*d; batch.tube(c-.02*d,c+.02*d,.08,(.12,.12,.13))
            t=root+.55*d; batch.current_material=5
            batch.tube(t-.09*d,t+.09*d,.085,tags[colour]); batch.current_material=4
        batch.current_material=0

    def clue(self,m):
        if m.id=='A01': return 'Ranger cache: forked tree at the central court. Complete the workshop before collecting repair parts.'
        if m.id=='A06': return f'Intake safe calibration mark: {self.calibration+1}. Coastal regulator locker is at the western coastal court.'
        if m.id=='B01': return f'Protect Reedbank refuge. Authorized empty overflow basin: {self.authorized+1}.'
        if m.id=='B08': return 'Reserve depot: stone arch at the eastern causeway. Normal approach crosses the wooden bridge. Shutter needs all three stations.'
        if m.id=='B12': return f'Outlet symbols 1,2,3 lead to basins {[v+1 for v in self.basin_map]}. Furnace launch is safe only at 45 C or below.'
        if m.id=='B19': return 'Storm report v2: wooden causeway bridge is closed. Use the north stone detour to the same reserve depot.'
        roles={self.targets[v]['symbol']:k for k,v in m.targets.items() if k in ('brake','release','vent','heat','inlet')}
        if m.kind=='balance':
            hint=f'Family counterweight: {roles}. Hold brake, establish support, then release. ' if m.id in ('A02','B02') else ''
            return hint+f'Load safe band: {m.state["count"]*320} kg. Current load: {m.state["load"]} kg.'
        if m.kind=='thermal':
            hint=f'Family thermal: {roles}. Open draft, protect heat, check temperature, then launch. Exposed hearths fail. ' if m.id in ('A03','B04') else ''
            return hint+f'Plume launch safe: {m.state["safe"]}.'
        if m.kind in ('flow','cool','prepare'):
            if m.kind=='prepare': return 'Three outlet symbols. Consult the previously surveyed basin map and commission authorization.'
            hint=f'Family flow: {roles}. Isolate inlet, open authorized outlet, then feed. ' if m.id=='B06' else ''
            return hint+f'Local requested outlet: {m.state["expected"]+1}. Temperature: {m.state["temperature"]:.1f} C.'
        return m.title

    def _valid_load(self,m):
        s=m.state; x,z=m.site
        bodies=[b for b in self.bodies.bodies.values() if not b.held and abs(b.pos[1]-b.half[1]-Y)<.06
                and x+2.4<b.pos[0]<x+4.7 and z+.5<b.pos[2]<z+4.7 and b.resting]
        s['load']=sum(b.spec.mass for b in bodies)
        return s['count']*320-1<=s['load']<=s['count']*320+1

    def _task_outcome(self,events):
        reward=0.; kinds={e['type'] for e in events}; dt=self.cfg.frame_skip*self.cfg.sim_cfg.dt
        for m in self.milestones:
            s=m.state; x,z=m.site
            if m.kind=='balance':
                valid=self._valid_load(m)
                if s['released'] and not valid:
                    s['released']=False; s['stable']=0
                    events.append(dict(type='brake_trip',milestone=m.id,reason='support lost'))
                    self.last_feedback+=f" Lift {m.id}: the brake tripped; the support plate now reads {s['load']:g} kg."
                if s['released']:
                    s['stable']+=dt
                    if s['stable']>=2 and not s['done']:
                        s['done']=True
                        self.world.fill_box((x+6,29,z-2),(x+6,32,z+2),B.AIR)
                        if m.id in ('A02','B02'):
                            self.world.fill_box((x+7,29,z-2),(x+10,29,z+2),B.AIR)
                            self.world.fill_box((x+7,32,z-2),(x+10,32,z+2),B.BRICK)
                            s['lift_level']=3
                        events.append(dict(type='counterweight_open',milestone=m.id))
                        self.last_feedback+=f' Lift {m.id}: the counterweight holds and the gate opens.'
            if m.kind=='thermal':
                if m.id=='B18': s['safe']=self.by_id['B17'].state['temperature']<=45
                if 'updraft' in kinds and x-2<self.agent.pos[0]<x+10 and abs(self.agent.pos[2]-(z+.5))<4:
                    s['updraft_seen']=True
                if self.agent.gliding and x-2<self.agent.pos[0]<x+10 and abs(self.agent.pos[2]-(z+.5))<4:
                    s['launch_seen']=True
                target=np.asarray(s['landing'])
                if (s['updraft_seen'] and self.agent.on_ground and self.agent.pos[1]>Y+3.8
                    and np.linalg.norm((self.agent.pos-target)[[0,2]])<3):
                    if not s['landed']: events.append(dict(type='thermal_landed',milestone=m.id))
                    s['landed']=True
                if s['landed'] and s['installed']: s['elapsed']+=dt
            if m.kind in ('flow','cool','prepare'):
                valid=s['inlet'] and s['outlets']==[k==s['expected'] for k in range(3)]
                s['stable']=s['stable']+dt if valid else 0.
                if m.kind=='cool' and valid: s['temperature']=max(35.,s['temperature']-8*dt)
            if m.kind=='cook':
                for e in events:
                    if e['type']=='cook': s['cooked'].extend(k for k in e['effects'] if k in ('cold_resist','heat_resist'))
                s['done']={'cold_resist','heat_resist'}<=set(s['cooked']) and self.world.get(x-2,29,z+2)==B.CAMPFIRE_LIT
            if m.kind=='calibrate' and s.get('engaged') and s['selected']==self.calibration: s['elapsed']+=dt
            if m.kind=='station' and s['installed'] and self._wm_ready(m): s['elapsed']+=dt
        if self.drained:
            x,_,z=self.agent.pos
            if abs(x-24.5)<2:
                if 48<z<53: self.road_seen.add('south')
                if 37<z<42 and 'south' in self.road_seen: self.road_seen.add('north')
        completed_ids={r['id'] for r in self.completed}
        for m in self.milestones:
            if m.id in completed_ids or not self.eligible(m): continue
            s=m.state
            ok={'clue':s['observed'],'balance':s['done'] and (not m.consumes or s['installed']),
                'thermal':s.get('landed',False) and (not m.consumes or (s['installed'] and s['elapsed']>=5)),
                'flow':s.get('stable',0)>=5 and (not m.consumes or s['installed']),
                'cool':s.get('temperature',90)<=45, 'prepare':s.get('outlets')==[k==s.get('expected') for k in range(3)] and not s.get('inlet',False),
                'calibrate':s['elapsed']>=5,'station':s['elapsed']>=5,
                'cook':s['done'],'exchange':s['done'],'report':s['done']}[m.kind]
            ok=ok and self._wm_ready(m)
            if m.id=='B23':
                flow=self.by_id['B22'].state
                ok=ok and flow['stable']>=5 and {'B15','B16','B18','B21','B22'}<=completed_ids
            if ok:
                if m.kind!='exchange':
                    for item in m.gives: self.satchel[item]+=1
                self.completed.append(dict(id=m.id,step=self.steps,seconds=round(self.elapsed,6)))
                completed_ids.add(m.id); reward+=1
                events.append(dict(type='milestone_complete',milestone=m.id,title=m.title))
                if m.id in ('B15','B16','B18'): self.online.add(m.id)
                if m.id in ('A11','B23'):
                    self.world.fill_box(*FLOOD,B.MUD)
                    self.drained=True; events.append(dict(type='road_drained'))
        if self.task=='b' and self.online=={'B15','B16','B18'} and not self.storm:
            self.storm=True; self.world.fill_box((127,28,95),(131,28,97),B.AIR)
            self.elements.climate.weather='storm'
            events.append(dict(type='bridge_closed',version=2))
        if self.task=='a' and 'A09' in completed_ids: self.elements.climate.weather='rain'
        self._wm_tick()
        self.success=(len(self.completed)==len(self.milestones) and self.drained
                      and self.agent.alive and self.system_operational())
        return self.success,reward+(10 if self.success else 0)

    def system_operational(self):
        """Final success requires working equipment, not just historical flags."""
        if self.task=='a':
            s=self.by_id['A11'].state
            return s['installed'] and s.get('engaged',False) and s['selected']==self.calibration and s['elapsed']>=5
        wind,pump,engine,lock,flow=(self.by_id[k].state for k in ('B15','B16','B18','B21','B22'))
        valid_flow=lambda s: s['inlet'] and s['outlets']==[k==s['expected'] for k in range(3)] and s['stable']>=5
        return (wind['installed'] and wind['elapsed']>=5 and pump['installed'] and valid_flow(pump)
                and engine['installed'] and engine['landed'] and engine['elapsed']>=5
                and lock['installed'] and lock['released'] and lock['stable']>=2 and valid_flow(flow))

    def _observation(self):
        obs=super()._observation()
        if self.current:
            visible=[f'{tid}:{obj["symbol"]}' for tid,obj in self.targets.items() if self._reachable(obj['pos'])]
            parts=(self.current.title,'Nearby controls '+', '.join(visible),self.last_feedback.strip(),
                   self.last_text,self.wm_cue['text'] if self.wm_cue else '','Repair satchel: '+str(dict(self.satchel)))
            obs['text']=' '.join(p.strip().rstrip('.')+'.' for p in parts if p.strip())[:1024]
        return obs

    def _info(self,events):
        out=super()._info(events)
        out.update(success=self.success,milestones=deepcopy(self.completed),task_progress=len(self.completed)/len(self.milestones),
                   working_memory=self.working_memory_summary())   # evaluator-only
        return out

    def _release_for_recovery(self):
        events=super()._release_for_recovery()
        if self.held is not None:
            bid=self.held; self.bodies.get(bid).held=False; self.held=None
            events.append(dict(type='magnetic_drop',body=bid,reason='recovery'))
        return events

    def state_digest(self):
        data=dict(task=self.task,seed=self._seed,steps=self.steps,elapsed=self.elapsed,
            agent=vars(self.agent).copy(),milestones=[m.state for m in self.milestones],
            completed=self.completed,satchel=dict(self.satchel),exposed=self.exposed,
            bodies={k:vars(b) for k,b in self.bodies.bodies.items()},held=self.held,
            terrain=hashlib.sha256(self.world.vox.tobytes()).hexdigest(),
            rng=self.np_random.bit_generator.state,weather=vars(self.elements.climate),
            npcs=[vars(n) for n in self.npcs.npcs if n.alive],online=self.online,drained=self.drained,road=self.road_seen,
            riding=self.riding.state(),orchard=self.orchard.state(),plants=self.gameplay.plants,
            wm={k:p.state() for k,p in sorted(self.wm.items())},wm_world=self.wm_world,wm_cue=self.wm_cue,
            wm_control=self.wm_control)
        data['agent']['inventory']=self.agent.inventory.to_dict()
        data['agent']['effects']=[vars(e) for e in self.agent.effects]
        recovery=self.recovery_state()
        if recovery is not None: data['recovery']=recovery   # absent unless used: old digests unchanged
        return hashlib.sha256(json.dumps(plain(data),sort_keys=True,default=str).encode()).hexdigest()

    def presentation_meshes(self,batch):
        """Actual control, cable, gauge and flow state, exported into Cycles."""
        for m in self.milestones:
            x,z=m.site; s=m.state
            if m.kind=='balance':
                for a,b in [((x+3,Y+5,z+2),(x+3,Y+.6,z+2)),((x+3,Y+5,z+2),(x+7,Y+5,z)),((x+7,Y+5,z),(x+7,Y+1,z))]:
                    batch.tube(a,b,.07,(.37,.29,.14))
                batch.box((x+3.5,Y+.05,z+2.6),(2,.1,4),(.44,.39,.25))
            if m.kind in ('flow','cool','prepare'):
                for k in range(3):
                    color=(.12,.65,.78) if s['outlets'][k] else (.30,.34,.36)
                    batch.tube((x-3,Y+.6,z-3),(x-2+2*k,Y+.6,z-1),.18,color)
                    batch.tube((x-2+2*k,Y+.6,z-1),(x-2+2*k,Y+.6,z+2),.18,color)
            if m.kind=='thermal' and s['heat']:
                color=(.15,.75,.72) if s['safe'] else (.95,.25,.1)
                for k in range(5):
                    yy=Y+.7+(k+(self.elapsed%1))*1.1
                    batch.ellipsoid((x+3,yy,z+.5),(.13,.22,.13),color)
        self._wm_meshes(batch)
        for obj in self.targets.values():
            if obj['role'].startswith('weight') or obj.get('wm'): continue
            m=self.by_id[obj['milestone']]; pos=obj['pos']; role=obj['role']
            color=(.19,.49,.53)
            if m.kind in ('flow','cool','prepare') and (role=='inlet' or role.startswith('outlet')):
                on=m.state['inlet'] if role=='inlet' else m.state['outlets'][int(role[-1])]
                color=(.2,.67,.38) if on else (.65,.27,.16)
            batch.box(pos-(0,.35,0),(.38,.7,.38),(.39,.31,.21))
            batch.ellipsoid(pos,(.26,.26,.12),color)

    def lantern_positions(self):
        """Two lanterns hang under each courtyard lintel (presentation dressing)."""
        coords=sorted({m.site for m in self.milestones if m.site!=(90,96)})
        return [(x+dx,Y+4.1,z+5.75) for x,z in coords for dx in (-2.5,3.5)]

    def presentation_lights(self):
        t=self.elements.climate.time; dusk=min(1.,max(0.,(t-.72)/.06)) if t>.5 else min(1.,max(0.,(.28-t)/.06))
        lights=[dict(pos=list(p),color=[1.,.7,.38],power=650.*dusk,radius=.25)
                for p in self.lantern_positions()] if dusk>0 else []
        for m in self.milestones:
            if m.kind=='thermal' and m.state.get('heat'):
                x,z=m.site
                lights.append(dict(pos=[x+2.5,Y+1.2,z+4.5],color=[1.,.55,.22],power=500.,radius=.3))
        return lights

    def presentation_camera(self):
        m=self.by_id.get(getattr(self,'reference_view_id','')) or self.current
        focus=self.agent.pos.astype(float)+(0,1,0)
        if m and np.linalg.norm(focus[[0,2]]-np.asarray(m.site))<16:
            focus=.45*focus+.55*np.array((m.site[0]+2,Y+1,m.site[1]))
        return focus+(-11,9,-13),focus
