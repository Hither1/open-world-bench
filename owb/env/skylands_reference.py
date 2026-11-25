"""Compact, executable Skylands reference courses (not the full benchmark).

Native character movement/ascent plus discrete collision decks for authored
cargo craft, freight lifts, and reversible platforms. The oracle only acts
through step(); it cannot assign progress or teleport between sites.
"""
from __future__ import annotations
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import numpy as np
from gymnasium import spaces
from .skylands import SkylandsEnv, COMMANDS as BASE_COMMANDS
from .wildlands import WildlandsEnv
from .wildlands_reference import plain
from ..core import blocks as B
from ..core.working_memory import WorkingMemoryProbe, summaries as wm_summaries
from ..core.config import EnvConfig, WorldConfig, RenderConfig, SimConfig
from ..world.skylands import generate, _island, _tree
from ..world.worldgen import Landmark

COMMANDS = BASE_COMMANDS + ('inspect', 'pick', 'mount', 'turn_part', 'load', 'unload',
    'drive', 'use', 'take', 'cache', 'retrieve', 'reverse', 'cancel', 'haul', 'release')
LEVELS = (4, 38, 103)

# Player-facing interaction feedback (see wildlands_reference.REFUSALS). Many
# evaluator reasons here name the missing procedure step, so refusals collapse
# to coarse, perceivable categories; success events become plain phrases.
SKY_INTERACTIONS = COMMANDS[len(BASE_COMMANDS):]
OUT_OF_REACH = {'target not visible and within reach', 'cache not reachable', 'loaded craft not in reach',
                'moor within reach', 'requested loose cargo is not within tow reach',
                'no loose cargo within tow reach', 'cargo missing, secured elsewhere, or too far',
                'inspect within local visible reach'}
LACKING = {'survey instrument required', 'reserve item not carried', 'reserve kit missing', 'spindle missing',
           'service token and lamp required', 'sky shuttle requires service cell', 'retrieve reserved coupling',
           'retrieve lamp kit from chosen cache'}
SKY_REFUSALS = {'no held part': "you aren't holding a part", 'part unavailable': "that part can't be taken",
                'incompatible or occupied mount': "it doesn't fit there", 'cache empty': 'the cache is empty',
                'requested reserve is not in this cache': "that isn't in this cache",
                'one steering axis per decision': 'steer along one axis at a time',
                'release load before ascent': "you can't ascend while carrying something",
                'unsupported local operation': 'nothing happens',
                'inspect source or operate physical mechanism': 'nothing happens',
                'yard gate locked': 'the yard gate is locked'}
FAN_SOUNDS = {'clean': 'the fan spins up cleanly', 'from rest': 'the fan turns over from rest',
              'jammed': 'the fan jams'}
SILENT = {'land', 'jump', 'swing', 'milestone_complete', 'npc_denned', 'npc_emerged', 'apple_fell',
          'horse_spooked'}


# Working-memory probes (docs/working-memory-probes.md). Each adds a physical
# requirement to an existing milestone. Entries are `use` on the use-site
# fixture with option = key 0-3; only a complete entry is judged.
#   probe, milestone, kind, load, distinct items, source role, use-site role
WM_PROBES = (
    ('WM-SA12', 'A12', 'hold', 4, False, 'dispatch_placard', 'lift_panel'),
    ('WM-SB10', 'B10', 'hold', 4, False, 'gate_code_board', 'gate_keypad'),
    ('WM-SB17', 'B17', 'reverse', 4, True, 'wiring_schematic', 'patch_panel'),
)
WM_CUE_SECONDS = 5.0     # a read cue stays in the text channel at most this long
WM_READ_REACH = 4.5      # ... and only while the source is within inspect reach
WM_KEYS = 4              # keys/jacks 0-3, selected with the existing `option`
WM_PANEL_TEXT = {
    'WM-SA12': 'Receiving-station control panel: four keys, 0 to 3. Key the dispatch release sequence to discharge the freight lift.',
    'WM-SB10': 'Inner yard gate keypad: four keys, 0 to 3. Key the gate code to open the yard.',
    'WM-SB17': 'Station patch panel: four jacks, 0 to 3. Wire them as the station wiring schematic directs.',
}
WM_SOLVED_EVENT = {'WM-SA12': 'freight_release_keyed', 'WM-SB10': 'yard_gate_opened', 'WM-SB17': 'station_patched'}


def wm_cue_text(probe, items):
    code = '-'.join(str(int(i)) for i in items)
    if probe == 'WM-SA12':
        return (f'Dispatch placard, freight lift: release sequence {code}. Key it on the receiving-station '
                'control panel at the top of the lift (keys 0 to 3). The lift holds its load until then.')
    if probe == 'WM-SB10':
        return (f'Marsh yard outer fence: inner gate code {code}. Key it on the inner yard gate keypad '
                '(keys 0 to 3).')
    return (f'Surface station wiring schematic: jack order {code}. Wire the patch panel inside in the '
            'REVERSE of this order (jacks 0 to 3).')


def sky_refusal(reason):
    if reason in OUT_OF_REACH: return 'out of reach or blocked from here'
    if reason in LACKING: return "you don't have what it needs"
    return SKY_REFUSALS.get(reason, "it won't work yet")

@dataclass
class Site:
    id: str
    title: str
    kind: str
    layer: int
    x: int
    z: int
    state: dict = field(default_factory=dict)
    elevation: int | None = None
    @property
    def y(self): return LEVELS[self.layer] if self.elevation is None else self.elevation
    @property
    def pos(self): return np.array((self.x,self.y,self.z),float)

A = [
 ('Take the observatory commission','clue',1,18,18),
 ('Restore the workshop cargo raft','raft',1,42,18),
 ('Deliver the practice load','raft',1,70,18),
 ('Open the ascent workshop','ascend',1,18,46),
 ('Recover the survey instrument','ascend',1,42,46),
 ('Stage the surface return raft','stage',1,70,46),
 ('Survey the buried works','clue',1,18,74),
 ('Establish the depths staging dock','dock',0,18,18),
 ('Open the vault service entrance','ascend',0,42,18),
 ('Extract and secure the navigation core','extract',0,70,18),
 ('Ferry the core through the flooded gallery','raft',0,42,46),
 ('Raise the core to the surface','freight',0,70,46),
 ('Recover the staged raft and ferry the core','return',1,70,46),
 ('Raise the core to the observatory','hoist',1,70,74),
 ('Restore navigation service','calibrate',2,70,74),
]
B_TASK = [
 ('Accept the lifeline commission','clue',1,18,18),
 ('Commission the training barge','raft',1,42,18),
 ('Deliver the independent practice load','raft',1,70,18),
 ('Open the ascent workshop','ascend',1,18,46),
 ('Recover the upper depot service kit','ascend',1,42,46),
 ('Restore the moving-platform training route','reverse',1,18,74),
 ('Retrieve the shuttle survey pack','reverse',1,42,74),
 ('Survey the three freight regions','clue',2,18,18),
 ('Distribute the expedition reserves','cache',1,70,74),
 ('Recover the marsh coupling','take',1,18,100),
 ('Charge the sky shuttle power cell','take',1,42,100),
 ('Establish the depths freight outpost','outpost',0,18,18),
 ('Transport the depths power module','raft',0,42,18),
 ('Reach the depths regulator gallery','ascend',0,70,18),
 ('Restore the depths station','station',0,70,46),
 ('Deliver the marsh coupling','raft',1,18,46),
 ('Restore the surface station','station',1,18,100),
 ('Recover the sky station spindle','reverse',2,42,18),
 ('Restore the sky station','ascend_station',2,70,18),
 ('Survey the replacement lock approach','update',1,42,100),
 ('Recover and fit the reserved coupling','retrieve',1,70,74),
 ('Reach the lock service terrace','reverse',1,42,46),
 ('Commission the final cargo lock and barge','compose',1,70,100),
 ('Deliver the relief shipment','delivery',1,70,100),
]

class SkylandsReferenceEnv(SkylandsEnv):
    commands = COMMANDS
    presentation_kind = 'skylands_reference'
    levels = LEVELS

    def site_specs(self):
        return A if self.task=='a' else B_TASK

    def boat_length(self,m): return 14

    def shuttle_span(self,m): return 11

    #: Working-memory probes WM-SA12/SB10/SB17. Subclasses with traces
    #: published before the probes (tempest, restoration) switch them off.
    wm_probes_enabled = True
    #: Walking-path span (metres) between a ground-level source and its use site.
    wm_source_span = (40,55)

    def __init__(self,task='a',seed=7,*,render_enabled=False,wm_control=False):
        if task not in ('a','b'): raise ValueError('task must be a or b')
        self.task=task
        # Diagnostic matched control: the use site also posts the cue.
        self.wm_control=bool(wm_control)
        cfg=EnvConfig(seed=seed,starter_camp=False,max_steps=30000,obs_rgb=render_enabled,
            obs_text=True,reward_mode='sparse',
            world_cfg=WorldConfig(size_x=96,size_y=112,size_z=112),
            sim_cfg=SimConfig(weather_enabled=False,wind_enabled=False,fire_enabled=False,start_time=.38),
            render_cfg=RenderConfig(width=320,height=180,third_person=True,cam_distance=8,max_dist=160))
        super().__init__(cfg)
        self.action_space['target']=spaces.Discrete(512)
        self.action_space['option']=spaces.Discrete(4)

    def _generate_world(self):
        cfg=deepcopy(self.cfg.world_cfg); cfg.seed=self._seed
        result=generate(cfg); w=result.world
        # Exact task collision geometry. Rendering never replaces this world.
        specs=self.site_specs()
        sites={(layer,x,z) for _,_,layer,x,z in specs}
        for layer,x,z in sites:
            y=self.levels[layer]
            if layer==2: _island(w,x+3,z+3,15,14,y-1,12,(x+z)*.17)
            w.fill_box((x-7,y,z-7),(min(x+21,95),min(y+8,111),min(z+10,111)),B.AIR)
            w.fill_box((x-7,y-1,z-7),(min(x+21,95),y-1,min(z+10,111)),B.MOSS_STONE if layer==0 else B.GRASS)
            for dx,dz in ((-5,7),(18,7)):
                if x+dx<95 and z+dz<110:
                    w.fill_box((x+dx,y,z+dz),(x+dx+1,y+3,z+dz+1),B.MOSS_STONE)
            w.fill_box((x-3,y-1,z-2),(x+2,y-1,z+2),B.BRICK)
        # One genuine maintenance lift joins the layers. It carries its rider
        # through this open shaft, taking time; no reset or hidden teleport.
        w.fill_box((5,4,5),(11,110,11),B.AIR)
        for layer,y in enumerate(self.levels):
            for z in (8,18,46,74,100):
                w.fill_box((12,y-1,z-1),(92,y-1,z+1),B.BRICK)
                w.fill_box((12,y,z-1),(92,min(y+2,111),z+1),B.AIR)
            for x in (12,18,42,70,90):
                w.fill_box((x-1,y-1,8),(x+1,y-1,105),B.BRICK)
                w.fill_box((x-1,y,8),(x+1,min(y+2,111),105),B.AIR)
            w.fill_box((10,y-1,7),(14,y-1,9),B.BRICK)
        # Stable botanical clusters and irregular outcrops outside access lanes.
        # These are world geometry, so every rollout validates the same collisions.
        for layer in (1,2):
            y=self.levels[layer]
            for i,(x,z) in enumerate(((26,32),(55,32),(85,32),(26,61),(56,61),(85,61),(27,87),(56,87),(86,87))):
                if B.SOLID[w.get(x,y-1,z)]:
                    _tree(w,x,y,z,5+i%3)
                    if i%2==0: _tree(w,x+3,y,z+2,4)
            for x,z in ((31,28),(61,59),(31,84),(80,89)):
                if B.SOLID[w.get(x,y-1,z)]:
                    for dx in range(-2,3):
                        for dz in range(-2,3):
                            height=max(0,3-abs(dx)-abs(dz)+(1 if (dx+dz)%3==0 else 0))
                            for dy in range(height): w.set_block(x+dx,y+dy,z+dz,B.MOSS_STONE)
        w.temperature[:]=.8; w.biome[:]=0
        result.spawn=(16.5,self.levels[1]+.01,18.5)
        result.landmarks=[Landmark('reference','Aster Reach freight course',result.spawn,3)]
        w.recompute_heightmaps()
        return result

    def _reset_mechanics(self):
        self._reset_course()
        self._setup_working_memory()

    def _reset_course(self):
        super()._reset_mechanics()
        self.agent.inventory.add('warm_doublet',1)
        self.agent.inventory.equip(self.agent.inventory.find('warm_doublet'))
        self.milestones=[Site(f'{self.task.upper()}{i:02}',*row,elevation=self.levels[row[2]]) for i,row in enumerate(self.site_specs(),1)]
        self.by_id={m.id:m for m in self.milestones}; self.completed=[]; self.targets={}
        self.satchel=Counter(); self.exposed={}; self.held_part=None; self.hauling=None
        self.cargo={}; self.rafts={}; self.movers={}; self.parts={}; self.online=set()
        self.last_text=''; self.last_feedback=''; self.closed=False; self.selected_calibration=None; self.stable=0.; self._pending=[]
        self.cache_contents={0:[],1:[],2:[]}; self.cache_choice=None; self.light_on=False
        rng=np.random.default_rng(self._seed+21811)
        self.calibration=int(rng.integers(3)); self.destination=int(rng.integers(2))
        self._add_mover('transit',(8,self.levels[1],8),'lift',top=self.levels[1])
        self.transit_target=self._target('transit','lift',(8,self.levels[1]+1,8))
        self.lift_calls=[self._target('transit','call',(12,y+1,8),level=y) for y in self.levels]
        for m in self.milestones:
            m.state=dict(done=False,observed=False,landed=False,installed=False,stable=0.)
            m.state['control']=self._target(m.id,'control',m.pos+(0,1,0))
            if m.kind in ('ascend','ascend_station','compose','hoist'):
                # Offset 5 x 5 gallery, six metres above its launch point.
                x,y,z=m.x,m.y,m.z
                self.world.fill_box((x-2,y+5,z-2),(x+2,y+5,z+2),B.BRICK)
                m.state['launch']=(x+.5,y+.01,z+.5)
                m.state['upper']=(x+.5,y+6.01,z+.5)
                self.targets[m.state['control']]['pos']=m.pos+(0,7,0)
                # The separate lower receiver/drive stays reachable with cargo.
                m.state['lower']=self._target(m.id,'lower',m.pos+(-3,1,0))
            if m.kind in ('raft','stage','compose'):
                rid=m.id; self._add_raft(m,rid)
                m.state['raft']=rid
                if m.id in ('A02','A03','A06','B02','B03'):
                    self._add_cargo(rid+'_ballast',m.pos+(-2,0,4),90 if m.id.endswith('02') else 180)
                    self.rafts[rid]['cargo_id']=rid+'_ballast'
                elif m.id=='A11': self.rafts[rid]['cargo_id']='core'
                elif m.id=='B13': self.rafts[rid]['cargo_id']='power_module'
                elif m.id=='B16': self.rafts[rid]['cargo_id']='marsh_coupling'
                elif m.id=='B23':
                    self._add_cargo('relief',m.pos+(-2,0,4),220)
                    self.rafts[rid]['cargo_id']='relief'
            if m.kind=='return': m.state['raft']='A06'
            if m.kind in ('freight','hoist'):
                mid='freight' if m.kind=='freight' else 'hoist'
                base=m.pos+(5,0,0)
                top=self.levels[1] if mid=='freight' else self.levels[2]
                self.world.fill_box((m.x+3,m.y,m.z-2),(m.x+7,top+3,m.z+2),B.AIR)
                self.world.fill_box((m.x+7,top-1,m.z-2),(m.x+12,top-1,m.z+2),B.BRICK)
                self._add_mover(mid,base,'freight',top=top)
                m.state['mover']=mid
                m.state['load_target']=self._target(m.id,'freight',base+(0,1,0))
            if m.kind=='reverse': self._add_reversal(m)
        self._add_cargo('core',self.by_id['A10'].pos+(0,0,3),240,anchored=True) if self.task=='a' else None
        if self.task=='b':
            self._add_cargo('power_module',self.by_id['B12'].pos+(0,0,4),260,anchored=True)
            self._add_cargo('marsh_coupling',self.by_id['B10'].pos+(0,0,3),150,anchored=True)
        # Two observable reserve locations. The recorded policy chooses one.
        for k in range(3):
            self._target('reserves','cache',np.array((66+k*3,self.levels[1]+1,78)),option=k)
        self.world.recompute_heightmaps()

    # ------------------------------------------------------------------
    # Working-memory probes (docs/working-memory-probes.md)
    def _setup_working_memory(self):
        """Draw probe content from dedicated generators and author the fixtures.
        Runs after all course geometry exists; never touches the simulation RNG."""
        self.wm={}; self.wm_sites={}; self._wm_showing=None; self.wm_gate=None
        if not self.wm_probes_enabled: return
        for probe,mid,kind,load,distinct,source_role,use_role in WM_PROBES:
            if mid not in self.by_id: continue
            self.wm[probe]=WorkingMemoryProbe.draw(probe,mid,kind,self._seed,range(WM_KEYS),load,
                                                   distinct=distinct,control=self.wm_control)
        if 'WM-SB10' in self.wm: self._build_inner_yard(self.by_id['B10'])
        placed=[]
        for probe,mid,kind,load,distinct,source_role,use_role in WM_PROBES:
            if probe not in self.wm: continue
            use,origin,toward,span=self.wm_use_site(probe)
            source,path,sight=self._wm_place_source(use,origin,toward,span,placed)
            placed += [source,use]
            self.wm_sites[probe]=dict(
                source=self._target(probe,source_role,source,probe=probe,wm='source'),
                use=self._target(probe,use_role,use,probe=probe,wm='use'),
                path_metres=path,line_of_sight=sight)
        self.world.recompute_heightmaps()

    def wm_use_site(self,probe):
        """(use-site fixture position, floor point the source's walking distance
        is measured from, approach anchor for the source, allowed path span in
        metres)."""
        if probe=='WM-SA12':
            m=self.by_id['A12']; mv=self.movers['freight']; top=mv['top']
            r=self.rafts['A11']
            toward=r.get('receiver',np.array((r['end']+3,m.y,r['pos'][2])))
            # Panel on the receiving station at the top of the lift. The source
            # is on the loading berth below, a short walk from the lift deck;
            # the lift ride itself is most of the interval.
            use=self._wm_spot(np.array((mv['pos'][0]+7,top+1,mv['pos'][2]-4)))
            return use,mv['pos'].copy(),toward,(6,12)
        if probe=='WM-SB10':
            m=self.by_id['B10']; use=self._wm_spot(m.pos+(3,1,0))
            return use,use-(0,1,0),self.by_id['B09'].pos,self.wm_source_span
        m=self.by_id['B17']; r=self.rafts['B16']; use=self._wm_spot(m.pos+(-5,1,-1))
        toward=r.get('receiver',np.array((r['end']+3,m.y,r['pos'][2])))
        return use,use-(0,1,0),toward,self.wm_source_span

    def _build_inner_yard(self,m):
        """Collision-backed yard around the marsh coupling with a 3-wide gate
        facing the take control; the gate opens only for the correct code."""
        c=np.rint(self.cargo['marsh_coupling']['pos']).astype(int); x,y,z=c
        lo,hi=(x-4,y,z-2),(x+4,y+2,z+4)
        w=self.world
        for xx in range(lo[0],hi[0]+1):
            for zz in range(lo[2],hi[2]+1):
                ring=xx in (lo[0],hi[0]) or zz in (lo[2],hi[2])
                if not ring: continue
                corner=xx in (lo[0],hi[0]) and zz in (lo[2],hi[2])
                for yy in range(y,y+3): w.set_block(xx,yy,zz,B.LOG if corner else B.PLANK,update_height=False)
        self.wm_gate=((x-1,y,z-2),(x+1,y+2,z-2))
        w.fill_box(*self.wm_gate,B.GATE)
        self.wm_yard=(lo,hi)

    def _wm_spot(self,nominal):
        """The nominal fixture point if it stands over walkable ground in open
        air, else the nearest such cell centre (stable world geometry only)."""
        nominal=np.asarray(nominal,float); y=int(round(nominal[1]))-1
        walk,x0,z0=self._walk_window(y,nominal,6)
        i,j=int(np.floor(nominal[0]))-x0,int(np.floor(nominal[2]))-z0
        if 0<=i<walk.shape[0] and 0<=j<walk.shape[1] and walk[i,j]: return nominal
        cells=np.argwhere(walk)
        if not len(cells): raise RuntimeError(f'No walkable ground for a fixture near {nominal}')
        d=np.hypot(cells[:,0]+x0+.5-nominal[0],cells[:,1]+z0+.5-nominal[2])
        i,j=cells[int(np.argmin(d))]
        return np.array((x0+i+.5,nominal[1],z0+j+.5))

    def _wm_trail_mask(self,y):
        """Cells where a roadside source is preferred (None: no preference)."""
        return None

    def _walk_window(self,y,center,radius):
        w=self.world; cx,cz=int(center[0]),int(center[2])
        x0,x1=max(1,cx-radius),min(w.size_x-1,cx+radius+1); z0,z1=max(1,cz-radius),min(w.size_z-1,cz+radius+1)
        v=w.vox[x0:x1,y-1:y+2,z0:z1]
        walk=B.SOLID[v[:,0,:]]&~B.SOLID[v[:,1,:]]&~B.SOLID[v[:,2,:]]
        return walk,x0,z0

    def _wm_place_source(self,use,origin,toward,span,avoid):
        """Deterministic source placement from final course geometry: an edge
        cell of walkable ground whose walking-path distance from the use site
        lies within `span` metres and which has no line of sight to it,
        nearest to where the agent approaches from."""
        from collections import deque as _dq
        y=int(round(origin[1]))
        walk,x0,z0=self._walk_window(y,origin,int(span[1])+8)
        dist=np.full(walk.shape,-1,np.int32); queue=_dq()
        ux,uz=origin[0]-x0,origin[2]-z0
        for i,j in np.argwhere(walk):
            if np.hypot(i+.5-ux,j+.5-uz)<=3: dist[i,j]=0; queue.append((i,j))
        while queue:
            i,j=queue.popleft()
            for a,b in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
                if 0<=a<walk.shape[0] and 0<=b<walk.shape[1] and walk[a,b] and dist[a,b]<0:
                    dist[a,b]=dist[i,j]+1; queue.append((a,b))
        pad=np.pad(walk,1)
        edge=~(pad[2:,1:-1]&pad[:-2,1:-1]&pad[1:-1,2:]&pad[1:-1,:-2])
        targets=[np.asarray(o['pos'],float) for o in self.targets.values()]+[np.asarray(q,float) for q in avoid]
        trail=self._wm_trail_mask(y)
        best=None
        for i,j in np.argwhere((dist>=span[0])&(dist<=span[1])):
            pos=np.array((x0+i+.5,y+1.2,z0+j+.5))
            if any(np.linalg.norm(pos-q)<4 for q in targets): continue
            sight=self.world.line_of_sight(pos,use) or self.world.line_of_sight(pos+(0,.6,0),np.asarray(use)+(0,.6,0))
            off=trail is not None and not trail[x0+i,z0+j]
            key=(bool(sight),off,not edge[i,j],round(float(np.hypot(pos[0]-toward[0],pos[2]-toward[2])),3),i,j)
            if best is None or key<best[0]: best=(key,pos,int(dist[i,j]),bool(sight))
        if best is None: raise RuntimeError(f'No working-memory source location near {use}')
        return best[1],best[2],best[3]

    def wm_solved(self,probe):
        p=getattr(self,'wm',{}).get(probe)
        return p is None or p.solved

    def _wm_interact(self,command,tid,obj,option):
        p=self.wm[obj['probe']]; site=obj['wm']
        if command=='inspect':
            if site=='source' or p.control:
                text=wm_cue_text(p.probe,p.items)
                if site=='use': text=WM_PANEL_TEXT[p.probe]+' Posted copy: '+text
                self.last_text=text; self._wm_showing=dict(probe=p.probe,target=tid,since=self.elapsed,text=text)
                return [p.read_cue(self.steps,self.elapsed),dict(type='clue_exposed',site=p.probe,text=text)]
            self.last_text=WM_PANEL_TEXT[p.probe]
            return [dict(type='clue_exposed',site=p.probe,text=self.last_text)]
        if command=='use' and site=='use':
            if p.solved: return self._fail('unsupported local operation')
            events=p.enter(int(option),self.steps,self.elapsed)
            if any(e['type']=='wm_attempt' and e['correct'] for e in events):
                events+=self._wm_effect(p.probe)
            return events
        return self._fail('inspect source or operate physical mechanism')

    def _wm_effect(self,probe):
        events=[dict(type=WM_SOLVED_EVENT[probe],probe=probe)]
        if probe=='WM-SA12':
            mv=self.movers['freight']
            if mv.get('cargo') and not mv['moving'] and mv['pos'][1]==mv['top']: events+=self._discharge('freight')
        elif probe=='WM-SB10':
            self.world.fill_box(*self.wm_gate,B.AIR)
        return events

    def _wm_tick(self):
        """Clear a shown cue after WM_CUE_SECONDS or once its source leaves
        inspect reach; abandon a partial entry once the agent walks away."""
        show=self._wm_showing if self.wm else None
        if show:
            pos=self.targets[show['target']]['pos']
            if self.elapsed-show['since']>=WM_CUE_SECONDS-1e-6 or not self._reachable(pos,WM_READ_REACH):
                if self.last_text==show['text']: self.last_text=''
                self._wm_showing=None
        for probe,site in self.wm_sites.items():
            p=self.wm[probe]
            if p.buffer and not self._reachable(self.targets[site['use']]['pos'],6): p.clear()

    def wm_summary(self):
        out=wm_summaries(self.wm.values())
        for probe,site in self.wm_sites.items():
            out[probe].update(source_target=site['source'],use_target=site['use'],
                source_pos=plain(self.targets[site['source']]['pos']),use_pos=plain(self.targets[site['use']]['pos']),
                path_metres=site['path_metres'],line_of_sight=site['line_of_sight'])
        return out

    def wm_state(self):
        return dict(probes={k:p.state() for k,p in self.wm.items()},showing=self._wm_showing,
                    last_text=self.last_text,gate=self.wm_gate)

    def _target(self,owner,role,pos,**extra):
        tid=max(self.targets,default=0)+1
        self.targets[tid]=dict(owner=owner,role=role,pos=np.array(pos,float),**extra)
        return tid

    def _add_cargo(self,cid,pos,mass,anchored=False):
        self.cargo[cid]=dict(pos=np.array(pos,float),mass=mass,parent=None,anchored=anchored,intact=True)
        self.cargo[cid]['target']=self._target(cid,'cargo',np.array(pos,float)+(0,1,0))

    def _add_mover(self,name,pos,kind,top=None):
        mover=dict(pos=np.array(pos,float),kind=kind,top=top,moving=False,goal=None,
                   under={},history=deque(maxlen=100),reverse=False,reversed_distance=0.,cargo=None)
        self.movers[name]=mover; self._write_deck(mover)

    def _add_raft(self,m,rid):
        x,y,z=m.x,m.y,m.z; length=self.boat_length(m)
        self.world.fill_box((x,y-2,z+2),(x+length+1,y-1,z+6),B.WATER)
        self.world.fill_box((x,y,z+2),(x+length+1,y+3,z+6),B.AIR)
        r=dict(pos=np.array((x+1.,y,z+4.)),origin=x+1.,end=float(x+length),parts=[None]*4,
               charge=0.,under={},cargo=None,cargo_id=None,throttle=0,tested=False,moored=False,travel=0.)
        self.rafts[rid]=r
        r['target']=self._target(rid,'raft',r['pos']+(0,1,0))
        for k,kind in enumerate(('pontoon','pontoon','fan','cell')):
            pid=len(self.parts)+1
            self.parts[pid]=dict(kind=kind,pos=m.pos+(-3,0,2+k),owner=rid,mounted=False,turn=0,
                                 capacity=220 if k<2 else 0)
            self._target(rid,'part',self.parts[pid]['pos']+(0,.7,0),part=pid)

    def _add_reversal(self,m):
        x,y,z=m.x,m.y,m.z; span=self.shuttle_span(m)
        self.world.fill_box((x+1,y-1,z-1),(x+span+1,y-1,z+1),B.AIR)
        # An exposed supported rail with a real gap and a far landing.
        self.world.fill_box((x+span+1,y-1,z-2),(x+span+5,y-1,z+2),B.BRICK)
        name=m.id+'_platform'
        self._add_mover(name,(x+2,y,z),'shuttle',top=None)
        mv=self.movers[name]; mv['start']=x+2; mv['end']=x+span; mv['direction']=-1
        mv['phase']='waiting'; mv['wait']=0
        m.state['mover']=name
        m.state['platform_target']=self._target(m.id,'platform',mv['pos']+(0,1,0),mover=name)
        self.targets[m.state['control']]['pos']=np.array((x+span+3,y+1,z),float)

    def _clear_deck(self,obj):
        for cell,value in obj['under'].items(): self.world.set_block(*cell,value)
        obj['under']={}

    def _write_deck(self,obj):
        x,y,z=np.rint(obj['pos']).astype(int)
        for xx in range(x-1,x+2):
            for zz in range(z-1,z+2):
                cell=(xx,y-1,zz); obj['under'][cell]=self.world.get(*cell)
                self.world.set_block(*cell,B.PLANK)

    def on_deck(self,obj):
        p=self.agent.pos; q=obj['pos']
        return abs(p[1]-q[1])<.15 and -.4<p[0]-q[0]<1.4 and -.4<p[2]-q[2]<1.4

    def _move_deck(self,obj,newpos):
        newpos=np.array(newpos,float); delta=newpos-obj['pos']; rider=self.on_deck(obj)
        x,y,z=np.rint(newpos).astype(int)
        if any((xx,yy,zz) not in obj['under'] and B.SOLID[self.world.get(xx,yy,zz)] for xx in range(x-1,x+2)
               for yy in range(y,y+3) for zz in range(z-1,z+2)):
            return None
        self._clear_deck(obj); obj['pos']=newpos; self._write_deck(obj)
        if rider:
            self.agent.pos+=delta.astype(np.float32); self.agent.vel[:]=0; self.agent.on_ground=True
        if obj.get('cargo'):
            self.cargo[obj['cargo']]['pos']=newpos+(0,0,.5)
        return rider

    @property
    def current(self):
        done={r['id'] for r in self.completed}
        return next((m for m in self.milestones if m.id not in done),None)

    def _reachable(self,p,reach=4.5):
        return np.linalg.norm(np.asarray(p)-self.agent.eye)<=reach and self.world.line_of_sight(self.agent.eye,np.asarray(p))

    def _fail(self,why): return [dict(type='action_failed',reason=why)]

    def _command(self,command,ingredients,action):
        events=super()._command(command,ingredients,action)
        if command in SKY_INTERACTIONS or any(e['type']=='fan_struck' for e in events):
            self.last_feedback=self._feedback(command,int(action['target']),events)
        return events

    def _feedback(self,command,tid,events):
        """One player-facing line for the latest interaction; observation text
        only, never part of the digest."""
        said=[]
        for e in events:
            t=e['type']
            if t in SILENT: continue
            if t=='action_failed': line=sky_refusal(e.get('reason',''))
            elif t=='fan_struck': line=FAN_SOUNDS.get(e.get('quality'),'the fan spins')
            elif t=='clue_exposed': line='you read it'
            elif t=='wm_cue_read': continue
            elif t=='wm_entry': line='click'
            elif t=='wm_attempt': line='the entry is accepted' if e['correct'] else 'the entry is rejected and the panel clears'
            elif t=='freight_held': line='the lift stops at the receiving station and holds its load'
            elif t=='freight_discharged': line='the lift releases its load'
            elif t=='freight_release_keyed': line='the release is keyed'
            elif t=='craft_moved': line='the craft moves'
            elif t=='motion_blocked': line='movement is blocked'
            else: line=t.replace('_',' ')
            if line not in said: said.append(line)
        name=f'{command} {tid}' if tid else command
        return f"Last interaction: {name} -> {'; '.join(said[:4]) or 'nothing happens'}."

    def craft_valid(self,r):
        if any(p is None for p in r['parts']): return False
        left,right,fan,cell=[self.parts[p] for p in r['parts']]
        mass=160+(self.cargo[r['cargo']]['mass'] if r['cargo'] else 0)
        return (left['capacity']+right['capacity']>=mass and fan['turn']==0 and r['charge']>0
                and left['kind']==right['kind']=='pontoon' and fan['kind']=='fan' and cell['kind']=='cell')

    def _dispatch(self,command,ingredients,action):
        events=[]; tid=int(action['target']); option=int(action['option'])
        # Targets follow their physical objects, including the maintenance lift.
        if 'transit' in self.movers: self.targets[self.transit_target]['pos']=self.movers['transit']['pos']+(0,1,0)
        for r in self.rafts.values(): self.targets[r['target']]['pos']=r['pos']+(0,1,0)
        for c in self.cargo.values(): self.targets[c['target']]['pos']=c['pos']+(0,1,0)
        for obj in self.targets.values():
            if 'mover' in obj: obj['pos']=self.movers[obj['mover']]['pos']+(0,1,0)
        if command=='none': return self._advance(events)
        if command=='ascend':
            if self.held_part or self.hauling: return self._advance(self._fail('release load before ascent'))
            events=super()._dispatch(command,ingredients,action)
            for m in self.milestones:
                if 'upper' in m.state and np.linalg.norm(self.agent.pos-np.array(m.state['upper']))<2:
                    m.state['landed']=True
            return self._advance(events)
        if command in BASE_COMMANDS: return self._advance(super()._dispatch(command,ingredients,action))
        obj=self.targets.get(tid)
        if not obj or not self._reachable(obj['pos'],6 if command in ('pick','mount','load','unload','drive','reverse','cancel') else 4.5):
            return self._advance(self._fail('target not visible and within reach'))
        if 'wm' in obj: return self._advance(self._wm_interact(command,tid,obj,option))
        owner=obj['owner']; m=self.by_id.get(owner); s=m.state if m else {}; role=obj['role']
        if command=='inspect':
            if m:
                if m.id in ('A07','B08') and not self.satchel['survey']: return self._advance(self._fail('survey instrument required'))
                s['observed']=True; self.exposed[m.id]=self.steps
                self.last_text=self.clue(m)
                if m.kind in ('clue','update'): s['done']=m.kind!='update' or self.closed
                if m.id in ('A01','B01'):
                    self.satchel['authorization']=1; self.satchel['lamp']=1
            events.append(dict(type='clue_exposed',site=owner,text=self.last_text))
        elif command=='pick' and role=='part':
            p=self.parts[obj['part']]
            if self.held_part or p['mounted']: events+=self._fail('part unavailable')
            else:
                self.held_part=obj['part']; events.append(dict(type='part_picked',part=self.held_part,kind=p['kind']))
        elif command=='turn_part':
            if not self.held_part: events+=self._fail('no held part')
            else: self.parts[self.held_part]['turn']^=1; events.append(dict(type='part_rotated',part=self.held_part))
        elif command=='mount' and owner in self.rafts:
            r=self.rafts[owner]; pid=self.held_part
            expected=('pontoon','pontoon','fan','cell')[option]
            if pid is None or r['parts'][option] is not None or self.parts[pid]['kind']!=expected:
                events+=self._fail('incompatible or occupied mount')
            else:
                r['parts'][option]=pid; self.parts[pid]['mounted']=True; self.held_part=None
                if option==3: r['charge']=100.
                if all(p is not None for p in r['parts'][:2]) and not r['under']: self._write_deck(r)
                events.append(dict(type='part_mounted',raft=owner,part=pid,slot=option))
        elif command in ('load','unload') and owner in self.rafts:
            r=self.rafts[owner]
            if command=='load':
                cid='core' if self.task=='a' and owner in getattr(self,'return_rafts',{'A06'}) and self.cargo['core']['anchored']==False else r['cargo_id']
                c=self.cargo.get(cid)
                if r['cargo'] or not c or c['anchored'] or c['parent'] or np.linalg.norm(c['pos']-r['pos'])>6:
                    events+=self._fail('cargo missing, secured elsewhere, or too far')
                else:
                    if self.hauling==cid: self.hauling=None
                    r['cargo']=cid; c['parent']=owner; c['pos']=r['pos']+(0,0,.5)
                    events.append(dict(type='cargo_loaded',cargo=cid,raft=owner,mass=c['mass']))
            elif not r['cargo'] or (r['pos'][0]<r['end']-.1 and not r['moored']): events+=self._fail('dock before unloading')
            else:
                cid=r['cargo']; c=self.cargo[cid]; c['parent']=None; c['pos']=np.array((r['end']+3 if r['pos'][0]>=r['end']-.1 else r['origin']-3,r['pos'][1],r['pos'][2]))
                r['cargo']=None; events.append(dict(type='cargo_delivered',cargo=cid,raft=owner))
                if m and m.kind=='raft': s['done']=True
                if owner in getattr(self,'return_rafts',{'A06'}) and cid=='core': self.by_id['A13'].state['done']=True
                if owner=='B23' and cid=='relief':
                    c['pos'][2]+=(-3 if option==0 else 3)
                    c['delivered_to']=option
                    if option==self.destination:
                        c['anchored']=True; c['parent']='refuge'; self.by_id['B24'].state['done']=True
                    else: events+=self._fail('wrong refuge; cargo remains at this dock')
        elif command=='drive' and owner in self.rafts:
            r=self.rafts[owner]
            if not self.on_deck(r) or not self.craft_valid(r): events+=self._fail('board a charged, supported, correctly driven craft')
            else: r['throttle']=1 if option==0 else -1 if option==1 else 0; r['moored']=False
        elif command=='use' and role=='call':
            mv=self.movers['transit']; mv['goal']=obj['level']; mv['moving']=mv['pos'][1]!=mv['goal']
            events.append(dict(type='maintenance_lift_called',level=obj['level']))
        elif command=='use' and role=='lift':
            mv=self.movers['transit']
            if not self.on_deck(mv): events+=self._fail('board maintenance lift first')
            else: mv['goal']=self.levels[min(option,2)]; mv['moving']=True
        elif command in ('reverse','cancel') and role=='platform':
            mv=self.movers[obj['mover']]
            if command=='cancel': mv['reverse']=False; mv['paused']=True; events.append(dict(type='reversal_cancelled',platform=obj['mover']))
            elif not self.on_deck(mv) or len(mv['history'])<2: events+=self._fail('board platform with recorded history')
            else: mv['reverse']=True; events.append(dict(type='reversal_started',platform=obj['mover'],history=len(mv['history'])))
        elif command=='haul':
            candidates=[(np.linalg.norm(c['pos']-self.agent.pos),cid) for cid,c in self.cargo.items()
                        if not c['anchored'] and not c['parent']]
            if not candidates or min(candidates)[0]>5: events+=self._fail('no loose cargo within tow reach')
            else:
                _,self.hauling=min(candidates); self.tow_offset=self.cargo[self.hauling]['pos']-self.agent.pos
                events.append(dict(type='cargo_towed',cargo=self.hauling))
        elif command=='release': self.hauling=None
        elif command in ('cache','retrieve') and role=='cache':
            k=obj['option']
            if command=='cache':
                if not self.satchel['lock_coupling'] or not self.satchel['lamp_kit']: events+=self._fail('reserve kit missing')
                else:
                    self.cache_contents[k]=['lock_coupling','lamp_kit']; self.satchel['lock_coupling']-=1; self.satchel['lamp_kit']-=1
                    self.cache_choice=k; self.by_id['B09'].state['done']=True
                    events.append(dict(type='reserves_cached',cache=k,items=self.cache_contents[k].copy()))
            elif not self.cache_contents[k]: events+=self._fail('cache empty')
            else:
                item='lamp_kit' if option==0 else 'lock_coupling'
                if item not in self.cache_contents[k]: return self._advance(self._fail('requested reserve is not in this cache'))
                items=[item]; self.satchel[item]+=1; self.cache_contents[k].remove(item)
                events.append(dict(type='cache_retrieved',cache=k,items=items))
        elif command in ('use','take') and m:
            events+=self._operate(m,role,option)
        else: events+=self._fail('unsupported local operation')
        return self._advance(events)

    def _operate(self,m,role,option):
        s=m.state; mid=m.id
        if m.kind in ('ascend','ascend_station','compose','hoist') and role=='control' and not s['landed']:
            return self._fail('reach gallery through valid ascent')
        if m.kind=='ascend':
            s['done']=True
            if mid in ('A05','B07'): self.satchel['survey']=1
            if mid=='B05': self.satchel.update(lock_coupling=1,lamp_kit=1)
            return [dict(type='gallery_serviced',site=mid)]
        if m.kind=='dock':
            if not self.satchel['authorization'] or not self.satchel['lamp']: return self._fail('service token and lamp required')
            self.light_on=True; s['done']=True
            return [dict(type='dock_lit',site=mid)]
        if m.kind=='extract':
            if not self.by_id['A09'].state['done'] or not self.satchel['cradle']: return self._fail('open vault and obtain cradle first')
            self.cargo['core']['anchored']=False; s['done']=True
            return [dict(type='core_extracted',cargo='core')]
        if m.kind=='stage':
            r=self.rafts[mid]
            if self.craft_valid(r) and r['travel']>=6 and r['pos'][0]<=r['origin']+.1:
                r['throttle']=0; r['moored']=True; s['done']=True
                return [dict(type='raft_staged',raft=mid,position=r['pos'].tolist())]
            return self._fail('test loaded raft and return it to its mooring')
        if m.kind in ('freight','hoist'):
            mv=self.movers[s['mover']]; c=self.cargo['core']
            if role=='freight':
                if c['parent'] or np.linalg.norm(c['pos']-mv['pos'])>6: return self._fail('bring core to freight loading deck')
                self.hauling=None; c['parent']=s['mover']; mv['cargo']='core'; c['pos']=mv['pos']+(0,0,.5)
                if m.kind=='freight': mv['moving']=True; mv['goal']=mv['top']
                return [dict(type='freight_loaded',cargo='core',lift=s['mover'])]
            if role=='control' and mv['cargo']=='core':
                mv['moving']=True; mv['goal']=mv['top']; return [dict(type='hoist_started',lift=s['mover'])]
            return self._fail('load freight before engaging hoist')
        if m.kind=='calibrate':
            c=self.cargo['core']
            if np.linalg.norm(c['pos']-m.pos)>10 or c['pos'][1]<self.levels[2]-.1: return self._fail('core has not reached observatory')
            if option!=self.calibration: return [dict(type='calibration_trip',selected=option)]
            self.selected_calibration=option; s['installed']=True; c['anchored']=True; c['parent']='observatory'
            return [dict(type='core_installed',calibration=option)]
        if m.kind=='take':
            if mid=='B10' and not self.wm_solved('WM-SB10'): return self._fail('yard gate locked')
            if mid=='B10': self.cargo['marsh_coupling']['anchored']=False
            else: self.satchel['sky_cell']=1
            s['done']=True; return [dict(type='service_part_recovered',site=mid)]
        if m.kind=='outpost':
            if not self.satchel['lamp_kit']: return self._fail('retrieve lamp kit from chosen cache')
            self.satchel['lamp_kit']-=1; self.light_on=True; self.cargo['power_module']['anchored']=False; s['done']=True
            return [dict(type='outpost_opened',site=mid)]
        if m.kind=='reverse':
            mv=self.movers[s['mover']]
            if mv['reversed_distance']<6: return self._fail('service landing requires reversed platform route')
            if mid=='B18' and not self.satchel['sky_cell']: return self._fail('sky shuttle requires service cell')
            if mid=='B22' and len(self.online)<3: return self._fail('three station outputs required')
            s['done']=True
            if mid=='B07': self.satchel['survey']=1
            if mid=='B18': self.satchel['spindle']=1
            return [dict(type='platform_landing_reached',site=mid)]
        if m.kind in ('station','ascend_station'):
            if mid=='B19':
                if not self.satchel['spindle']: return self._fail('spindle missing')
            else:
                cid='power_module' if mid=='B15' else 'marsh_coupling'; c=self.cargo[cid]
                if c['parent'] or np.linalg.norm(c['pos']-m.pos)>6: return self._fail('bring the unique module to its station')
                if mid=='B15' and not self.by_id['B14'].state['done']: return self._fail('open regulator gallery')
                self.hauling=None; c['anchored']=True; c['parent']=mid
            if mid=='B19' and not s['installed']:
                self.satchel['spindle']-=1; s['station_part']='spindle'
            s['installed']=True; return [dict(type='station_engaged',site=mid)]
        if m.kind=='retrieve':
            if not self.satchel['lock_coupling']: return self._fail('retrieve reserved coupling')
            self.satchel['lock_coupling']-=1; s['done']=True; s['installed']=True
            return [dict(type='lock_coupling_installed')]
        if m.kind=='compose':
            if len(self.online)<3 or not self.by_id['B21'].state['installed']: return self._fail('network and repaired drive required')
            s['installed']=True; return [dict(type='cargo_berth_opened')]
        return self._fail('inspect source or operate physical mechanism')

    def _advance(self,events):
        dt=.2
        if self.hauling:
            c=self.cargo[self.hauling]; goal=self.agent.pos.astype(float)+self.tow_offset
            x,y,z=np.rint(goal).astype(int)
            # A wheeled cradle can follow solid ground, not traverse open water.
            if self.world.in_bounds(x,y,z) and B.SOLID[self.world.get(x,y-1,z)] and not B.SOLID[self.world.get(x,y,z)]:
                c['pos']=goal
            elif np.linalg.norm(c['pos']-self.agent.pos)>7:
                self.hauling=None; events.append(dict(type='tow_released',reason='unsupported cargo route'))
        for rid,r in self.rafts.items():
            if r['throttle'] and self.craft_valid(r) and self.on_deck(r):
                q=r['pos'].copy(); q[0]=np.clip(q[0]+r['throttle'],r['origin'],r['end'])
                distance=abs(q[0]-r['pos'][0])
                if distance:
                    if self._move_deck(r,q) is None:
                        r['throttle']=0; events.append(dict(type='motion_blocked',object=rid)); continue
                    r['charge']-=distance*.75; r['travel']+=distance
                    r['tested']=r['travel']>=3
                    events.append(dict(type='craft_moved',raft=rid,position=q.tolist(),charge=r['charge']))
                else: r['throttle']=0
        for name,mv in self.movers.items():
            if mv['kind']=='shuttle':
                if name=='B18_platform' and not self.satchel['sky_cell']: continue
                if name=='B22_platform' and len(self.online)<3: continue
                if mv.get('paused'): continue
                if mv['reverse']:
                    if mv['history']:
                        q=np.array(mv['history'].pop()); dist=float(np.linalg.norm(q-mv['pos']))
                        if self._move_deck(mv,q) is None:
                            mv['history'].append(q.tolist()); mv['reverse']=False; mv['paused']=True
                            events.append(dict(type='motion_blocked',object=name)); continue
                        mv['reversed_distance']+=dist
                        if dist: events.append(dict(type='platform_reversed',platform=name,position=q.tolist()))
                    else: mv['reverse']=False; mv['wait']=0
                else:
                    # Fresh local motion, not old episode history: travel from
                    # far landing toward agent, then dwell for safe boarding.
                    mv['history'].append(mv['pos'].tolist())
                    if mv['phase']=='waiting':
                        mv['wait']+=1
                        if mv['wait']>=12:
                            mv['phase']='outbound'; mv['wait']=0
                    elif mv['phase']=='outbound':
                        q=mv['pos'].copy(); q[0]=min(mv['end'],q[0]+1); self._move_deck(mv,q)
                        if q[0]==mv['end']: mv['phase']='inbound'
                    else:
                        q=mv['pos'].copy(); q[0]=max(mv['start'],q[0]-1); self._move_deck(mv,q)
                        if q[0]==mv['start']: mv['phase']='waiting'
            elif mv['moving']:
                q=mv['pos'].copy(); q[1]+=np.sign(mv['goal']-q[1]); rider=self._move_deck(mv,q)
                if rider is None:
                    mv['moving']=False; events.append(dict(type='motion_blocked',object=name)); continue
                if q[1]==mv['goal']:
                    mv['moving']=False; events.append(dict(type='lift_arrived',lift=name,level=q[1],rider=rider))
                    if mv.get('cargo'):
                        if name=='freight' and not self.wm_solved('WM-SA12'):
                            # The receiving station holds the deck until the
                            # dispatch release is keyed on its control panel.
                            events.append(dict(type='freight_held',lift=name))
                        else: events+=self._discharge(name)
        return events

    def _discharge(self,name):
        mv=self.movers[name]; cid=mv['cargo']; c=self.cargo[cid]
        c['parent']=None; c['pos']=mv['pos']+(4,0,0); mv['cargo']=None
        self.by_id['A12' if name=='freight' else 'A14'].state['done']=True
        return [dict(type='freight_discharged',lift=name,cargo=cid)] if name=='freight' and 'WM-SA12' in self.wm else []

    def _task_outcome(self,events):
        reward=0.
        self._wm_tick()
        for m in self.milestones:
            s=m.state
            if m.id=='B17' and not self.wm_solved('WM-SB17'): pass   # output not patched through
            elif s['installed'] and m.kind in ('station','ascend_station','calibrate'):
                s['stable']+=.2
                if s['stable']>=5:
                    s['done']=True
                    if m.kind!='calibrate': self.online.add(m.id)
            if m.kind=='compose':
                r=self.rafts[m.id]
                s['done']=s['installed'] and r['cargo']=='relief' and self.craft_valid(r) and r['travel']>=3
            if m.id in ('A02','B02') and s['done']: self.satchel['cradle']=1
        if self.task=='b' and len(self.online)==3 and not self.closed:
            self.closed=True
            # Closure is real collision at the normal approach, not just text.
            self.world.fill_box(*self.closure_bounds(),B.BRICK)
            events.append(dict(type='maintenance_bridge_closed',version=2))
        old={r['id'] for r in self.completed}
        for m in self.milestones:
            if m.state['done'] and m.id not in old:
                self.completed.append(dict(id=m.id,step=self.steps,seconds=round(self.elapsed,6)))
                events.append(dict(type='milestone_complete',milestone=m.id,title=m.title)); reward+=1
        self.success=self.system_operational() and len(self.completed)==len(self.milestones) and self.agent.alive
        return self.success,reward+(10 if self.success else 0)

    def closure_bounds(self):
        return (36,self.levels[1],98),(38,self.levels[1]+2,100)

    def cache_position(self,k):
        return np.array((66+k*3,self.levels[1],78),float)

    def refuge_position(self,k):
        r=self.rafts['B23']
        return np.array((r['end']+3,r['pos'][1],r['pos'][2]+(-3 if k==0 else 3)))

    def system_operational(self):
        if self.task=='a':
            s=self.by_id['A15'].state; c=self.cargo['core']
            receiver=np.array((self.by_id['A14'].x+9,self.levels[2],self.by_id['A14'].z),float)
            return (s['installed'] and s['stable']>=5 and self.selected_calibration==self.calibration
                    and c['intact'] and c['anchored'] and c['parent']=='observatory' and np.linalg.norm(c['pos']-receiver)<.1)
        relief=self.cargo['relief']; r=self.rafts['B23']
        destination=np.array((r['end']+3,r['pos'][1],r['pos'][2]+(-3 if self.destination==0 else 3)))
        inputs=all(self.cargo[cid]['anchored'] and self.cargo[cid]['parent']==mid
                   and np.linalg.norm(self.cargo[cid]['pos']-self.by_id[mid].pos)<6
                   for cid,mid in (('power_module','B15'),('marsh_coupling','B17')))
        return (inputs and self.wm_solved('WM-SB17') and self.by_id['B19'].state.get('station_part')=='spindle'
                and relief.get('delivered_to')==self.destination and relief['parent']=='refuge'
                and np.linalg.norm(relief['pos']-destination)<.1
                and all(self.by_id[mid].state['installed'] and self.by_id[mid].state['stable']>=5 for mid in ('B15','B17','B19'))
                and self.by_id['B21'].state['installed'] and self.by_id['B23'].state['installed']
                and self.by_id['B24'].state['done'] and self.cargo['relief']['intact'])

    def clue(self,m):
        if m.id=='A07': return f'Vault: north crystal arch. Receiver calibration {self.calibration}. Freight route through the lit dock.'
        if m.id=='B01': return f'Relief shipment destination: refuge {self.destination}. Restore depths, surface and sky station outputs.'
        if m.id=='B08': return 'Depths outpost: north crystal arch. Sky spindle: eastern shuttle. Normal lock approach: west maintenance bridge.'
        if m.kind=='update': return 'Maintenance bridge closed, report v2. Use the moving platform east of the approach and ascend into the loft.'
        if m.kind in ('raft','stage'): return 'Inspect buoyancy and payload. Join two hulls, forward fan and cell; secure cargo; test the loaded craft before departure.'
        if m.kind=='reverse': return 'Inspect a fresh path; board after the inbound trip; reverse toward the far landing and dismount safely.'
        return m.title

    def _observation(self):
        obs=super()._observation()
        parts=((self.current.title if self.current else 'Course complete'),self.last_feedback,self.last_text,
               f'Satchel {dict(self.satchel)}')
        obs['text']=' '.join(p.strip().rstrip('.')+'.' for p in parts if p.strip())[:1024]
        return obs

    def _info(self,events):
        out=WildlandsEnv._info(self,events)
        out.update(success=self.success,milestones=deepcopy(self.completed),task_progress=len(self.completed)/len(self.milestones))
        if self.wm: out['working_memory']=self.wm_summary()
        return out

    def _safe_ground_ok(self):
        # A raft or lift deck moves: never a place to return to.
        return super()._safe_ground_ok() and not any(self.on_deck(o) for o in (*self.rafts.values(),*self.movers.values()))

    def _release_for_recovery(self):
        events=super()._release_for_recovery()
        if self.held_part is not None:
            events.append(dict(type='part_dropped',part=self.held_part,reason='recovery')); self.held_part=None
        if self.hauling is not None:
            events.append(dict(type='tow_released',reason='recovery')); self.hauling=None
        return events

    def state_digest(self):
        def deck(d): return {k:(list(v) if isinstance(v,deque) else v) for k,v in d.items() }
        data=dict(task=self.task,seed=self._seed,steps=self.steps,elapsed=self.elapsed,
            agent=vars(self.agent).copy(),completed=self.completed,states=[m.state for m in self.milestones],
            rafts={k:deck(v) for k,v in self.rafts.items()},movers={k:deck(v) for k,v in self.movers.items()},
            cargo=self.cargo,parts=self.parts,satchel=self.satchel,exposed=self.exposed,held=self.held_part,hauling=self.hauling,
            caches=self.cache_contents,online=self.online,closed=self.closed,calibration=self.selected_calibration,
            terrain=hashlib.sha256(self.world.vox.tobytes()).hexdigest(),rng=self.np_random.bit_generator.state,
            climate=vars(self.elements.climate),maps=self.layer_maps)
        data['agent']['inventory']=self.agent.inventory.to_dict(); data['agent']['effects']=[vars(e) for e in self.agent.effects]
        recovery=self.recovery_state()
        if recovery is not None: data['recovery']=recovery   # absent unless used: old digests unchanged
        if self.wm: data['working_memory']=self.wm_state()   # absent when probes are off
        return hashlib.sha256(json.dumps(plain(data),sort_keys=True,default=str).encode()).hexdigest()

    def presentation_camera(self):
        m=self.by_id.get(getattr(self,'reference_view_id','')) or self.current or self.milestones[-1]
        focus=self.agent.pos.astype(float)+(0,1,0)
        if np.linalg.norm(focus[[0,2]]-m.pos[[0,2]])<22:
            relevant=m.pos+(4,1,2)
            if m.state.get('raft'):
                relevant=self.rafts[m.state['raft']]['pos']+(0,1,0)
            elif m.state.get('mover'):
                relevant=self.movers[m.state['mover']]['pos']+(0,1,0)
            focus=.55*focus+.45*relevant
        rise=7 if focus[1]<30 else 10
        return focus+(-12,rise,-15),focus

    def presentation_meshes(self,batch):
        for rid,r in self.rafts.items():
            q=r['pos']
            for k,pid in enumerate(r['parts']):
                if pid is None: continue
                color=(.53,.34,.14)
                if k<2:
                    dz=(-1 if k==0 else 1)
                    for v in (-.22,.22): batch.tube(q+(-1.5,-.5,dz+v),q+(1.5,-.5,dz+v),.25,color)
                    for dx in (-1.,1.): batch.tube(q+(dx,-.15,-1.4),q+(dx,-.15,1.4),.10,(.3,.22,.12))
                elif k==2:
                    c=q+(-1.1,1.1,0)
                    batch.tube(q+(-1.1,0,0),c,.12,(.22,.29,.3))
                    batch.tube(c+(-.2,0,0),c+(.2,0,0),.25,(.12,.5,.48))
                    phase=self.elapsed*12 if r['throttle'] or r.get('driving') else 0
                    for angle in np.arange(4)*np.pi/2+phase:
                        end=c+(0,np.cos(angle)*.8,np.sin(angle)*.8)
                        batch.tube(c,end,.12,(.55,.66,.58),end_radius=.24)
                else:
                    batch.box(q+(0,.25,-.7),(.7,.5,.5),(.12,.3,.32))
                    batch.box(q+(0,.5,-.7),(.5,.06,.3),(.25,.9,.7),emissive=1)
        for pid,p in self.parts.items():
            if p['mounted']: continue
            q=self.agent.pos+(0,1.5,1) if self.held_part==pid else p['pos']
            if p['kind']=='pontoon': batch.tube(q+(-.8,.3,0),q+(.8,.3,0),.3,(.53,.34,.14))
            elif p['kind']=='fan':
                batch.box(q+(0,.45,0),(.3,.9,1),(.23,.48,.46))
                batch.tube(q+(0,.2,-.5),q+(0,.8,.5),.09,(.67,.7,.55))
            else: batch.box(q+(0,.3,0),(.6,.6,.6),(.12,.65,.58),emissive=.4)
        for cid,c in self.cargo.items():
            q=c['pos']; core=cid=='core'
            batch.box(q+(0,.15,0),(1.5,.3,1.5),(.25,.22,.18))
            if core:
                # Faceted original navigation core with bronze ribs and crystal.
                for k in range(6):
                    angle=k*np.pi/3; off=np.array((np.cos(angle)*.55,0,np.sin(angle)*.55))
                    batch.tube(q+off+(0,.3,0),q+off*.7+(0,1.7,0),.09,(.57,.41,.18))
                batch.box(q+(0,.95,0),(.7,1.2,.7),(.12,.7,.73),yaw=.4,emissive=.5)
            else:
                batch.box(q+(0,.7,0),(1.25,1.1,1.25),(.42,.27,.13))
                for dx in (-.45,.45): batch.box(q+(dx,.72,0),(.1,1.15,1.28),(.27,.34,.34))
                batch.box(q+(0,1.28,0),(.5,.06,.5),(.18,.7,.61),emissive=.3)
        for name,mv in self.movers.items():
            q=mv['pos']
            batch.box(q+(0,-.15,0),(3,.3,3),(.38,.35,.27))
            for dz in (-1.3,1.3):
                batch.tube(q+(-1.3,.1,dz),q+(-1.3,1.2,dz),.08,(.31,.41,.39))
            if mv['kind']!='shuttle':
                for dx,dz in ((-1.4,-1.4),(1.4,1.4)):
                    batch.tube(q+(dx,.1,dz),(q[0]+dx,108,q[2]+dz),.045,(.25,.3,.27))
            else:
                color=(.18,.9,.72) if mv['reverse'] else (.68,.5,.2)
                batch.box(q+(0,-.02,0),(1.3,.05,1.3),color,emissive=.5)
        for m in self.milestones:
            q=self.targets[m.state['control']]['pos']
            batch.box(q+(0,-.6,0),(.7,1.2,.6),(.39,.4,.32))
            batch.box(q+(0,.05,0),(.8,.18,.65),(.17,.53,.48) if m.state['done'] else (.58,.41,.16))
            batch.tube(q+(0,.1,0),q+(.3,.5,0),.07,(.42,.5,.48))
            if m.layer==0:
                lamp=m.pos+(-3,4,-3)
                batch.tube(m.pos+(-3,0,-3),lamp,.10,(.28,.29,.25))
                batch.box(lamp,(.6,.8,.6),(.2,.8,.7),emissive=4)
        if self.task=='b':
            r=self.rafts['B23']
            for k in range(2):
                q=self.refuge_position(k)
                batch.box(q+(0,-.05,0),(2.4,.1,2.4),(.24,.42,.37))
                batch.tube(q+(1,0,0),q+(1,2,0),.09,(.43,.34,.19))
                for j in range(k+1): batch.box(q+(1,1.7-j*.35,0),(.12,.18,.7),(.2,.8,.65),emissive=.3)
        for k,items in self.cache_contents.items():
            q=self.cache_position(k)
            batch.box(q+(0,.6,0),(1.5,1.2,1.5),(.3,.24,.16))
            batch.box(q+(0,1.25,0),(1.6,.15,1.6),(.18,.45,.4))
            if items: batch.box(q+(0,1.4,0),(.5,.15,.5),(.3,.8,.6),emissive=.5)
        if getattr(self,'wm_sites',None): self._wm_fixture_meshes(batch)

    def _wm_fixture_meshes(self,batch):
        """Distinct, recognisable fixtures for the working-memory probes. Shape
        variation is keyed to the probe id (stable entity identity), never to
        the simulation RNG; nothing shown encodes the probe content."""
        import zlib
        for probe,site in self.wm_sites.items():
            p=self.wm[probe]; h=zlib.crc32(probe.encode())
            src=np.asarray(self.targets[site['source']]['pos'],float)-(0,1.2,0)
            use=np.asarray(self.targets[site['use']]['pos'],float)-(0,1,0)
            yaw=(h%8)*np.pi/4
            lit=(.25,.85,.45) if p.solved else (.95,.62,.2)
            if probe=='WM-SA12':
                # Dispatch placard: two timber posts, slate board, brass frame, pinned slips.
                for dz in (-.55,.55): batch.tube(src+(0,0,dz),src+(0,1.9,dz),.06,(.36,.25,.14))
                batch.box(src+(0,1.35,0),(.08,.8,1.25),(.18,.2,.22),yaw=yaw)
                batch.box(src+(.03,1.78,0),(.1,.06,1.3),(.62,.48,.2),yaw=yaw)
                for k in range(2+h%3):
                    batch.box(src+(.06,1.2+.14*(k%2),-.35+.28*k),(.02,.22,.18),(.9,.86,.74),yaw=yaw)
                # Receiving-station control panel: steel cabinet, angled face, four keys, status lamp.
                batch.box(use+(0,.5,0),(.6,1.0,.7),(.34,.37,.38))
                batch.box(use+(0,1.06,0),(.66,.12,.76),(.24,.27,.28),yaw=.0)
                for k in range(WM_KEYS):
                    batch.tube(use+(.3,.85,-.27+.18*k),use+(.38,.85,-.27+.18*k),.05,(.82,.78,.66))
                batch.box(use+(0,1.25,.25),(.14,.14,.14),lit,emissive=1.5)
                batch.tube(use+(-.3,.1,0),use+(-2.6,.1,0),.07,(.3,.3,.28))
            elif probe=='WM-SB10':
                # Outer fence section with a hanging enamel gate-code board.
                for dx in (-1.4,0,1.4): batch.tube(src+(dx,0,0),src+(dx,1.5,0),.07,(.42,.3,.17))
                for y in (.5,1.15): batch.tube(src+(-1.5,y,0),src+(1.5,y,0),.04,(.47,.35,.2))
                batch.box(src+(.7,1.0,.06),(.75,.5,.04),(.86,.88,.84))
                batch.box(src+(.7,1.0,.09),(.6,.08,.02),(.16,.3,.55))
                batch.box(src+(.7,.68,.06),(.12,.14,.06),(.55,.5,.3))
                # Inner gate keypad: stone pedestal, housing, 2 x 2 keys, lamp.
                batch.box(use+(0,.45,0),(.38,.9,.38),(.46,.45,.41))
                batch.box(use+(0,1.0,0),(.32,.4,.12),(.2,.22,.24))
                for k in range(WM_KEYS):
                    batch.box(use+(-.07+.14*(k%2),.94+.12*(k//2),-.07),(.09,.09,.03),(.85,.82,.7))
                batch.box(use+(0,1.24,-.05),(.06,.06,.04),lit,emissive=1.5)
            else:
                # Wiring schematic: easel frame, pale board, four printed traces.
                batch.tube(src+(-.5,0,-.2),src+(0,1.9,0),.05,(.4,.29,.17))
                batch.tube(src+(.5,0,-.2),src+(0,1.9,0),.05,(.4,.29,.17))
                batch.tube(src+(0,0,.5),src+(0,1.9,0),.05,(.4,.29,.17))
                batch.box(src+(0,1.3,-.06),(1.1,.8,.04),(.88,.9,.86))
                for k,color in enumerate(((.75,.2,.15),(.15,.35,.7),(.2,.55,.25),(.8,.65,.1))):
                    batch.box(src+(0,1.05+.15*k,-.09),(.85-.1*((h>>k)&1),.03,.01),color)
                # Patch panel: tall cabinet, four jacks, cables for each entry made.
                batch.box(use+(0,.75,0),(.5,1.5,.9),(.3,.33,.3))
                plugged=WM_KEYS if p.solved else len(p.buffer)
                for k in range(WM_KEYS):
                    jack=use+(.27,1.1,-.3+.2*k)
                    batch.tube(jack,jack+(.06,0,0),.045,(.1,.1,.1))
                    if k<plugged: batch.tube(jack+(.06,0,0),use+(.45,.35,-.3+.2*((k+1)%WM_KEYS)),.03,(.12,.12,.1))
                batch.box(use+(.27,1.4,0),(.04,.06,.06),lit,emissive=1.5)

    def presentation_lights(self):
        return [dict(pos=(m.pos+(-3,4,-3)).tolist(),color=[.35,.8,.75],power=1800.,radius=1.)
                for m in self.milestones if m.layer==0]
