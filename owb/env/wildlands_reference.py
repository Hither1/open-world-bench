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
import json
import numpy as np
from gymnasium import spaces
from .wildlands import WildlandsEnv, COMMANDS as BASE_COMMANDS
from ..core.config import EnvConfig, WorldConfig, RenderConfig, SimConfig
from ..core import blocks as B
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

    def __init__(self, task='a', seed=7, *, render_enabled=False):
        if task not in ('a','b'): raise ValueError('task must be a or b')
        self.task = task
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
                                   sea_level=w.sea_level)
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
        self.last_text=''; self.held=None; self.online=set(); self._pending=[]
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
                if any(self.satchel[k]<1 for k in m.consumes): return self._fail('exchange input missing')
                if m.id in ('B10','B11') and not self.satchel['authorization']: return self._fail('commission required')
                for k in m.consumes: self.satchel[k]-=1
                for k in m.gives: self.satchel[k]+=1
                s['done']=True
                return [dict(type='item_exchange',milestone=m.id,inputs=m.consumes,outputs=m.gives)]
            if m.kind=='report':
                if not self.drained or self.road_seen!={'north','south'}: return self._fail('verify the drained road first')
                s['done']=True; return [dict(type='completion_signal')]
            return self._fail('inspect this source instead')
        return super()._dispatch(command,ingredients,action)

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
            if m.kind=='station' and s['installed']: s['elapsed']+=dt
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
            obs['text']=(self.current.title+'. Nearby controls '+', '.join(visible)+'. '+self.last_text+
                         '. Repair satchel: '+str(dict(self.satchel)))[:1024]
        return obs

    def _info(self,events):
        out=super()._info(events)
        out.update(success=self.success,milestones=deepcopy(self.completed),task_progress=len(self.completed)/len(self.milestones))
        return out

    def state_digest(self):
        data=dict(task=self.task,seed=self._seed,steps=self.steps,elapsed=self.elapsed,
            agent=vars(self.agent).copy(),milestones=[m.state for m in self.milestones],
            completed=self.completed,satchel=dict(self.satchel),exposed=self.exposed,
            bodies={k:vars(b) for k,b in self.bodies.bodies.items()},held=self.held,
            terrain=hashlib.sha256(self.world.vox.tobytes()).hexdigest(),
            rng=self.np_random.bit_generator.state,weather=vars(self.elements.climate),
            npcs=[vars(n) for n in self.npcs.npcs if n.alive],online=self.online,drained=self.drained,road=self.road_seen,
            riding=self.riding.state(),orchard=self.orchard.state(),plants=self.gameplay.plants)
        data['agent']['inventory']=self.agent.inventory.to_dict()
        data['agent']['effects']=[vars(e) for e in self.agent.effects]
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
        for obj in self.targets.values():
            if obj['role'].startswith('weight'): continue
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
