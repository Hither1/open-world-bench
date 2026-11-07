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
from ..core.config import EnvConfig, WorldConfig, RenderConfig, SimConfig
from ..world.skylands import generate, _island, _tree
from ..world.worldgen import Landmark

COMMANDS = BASE_COMMANDS + ('inspect', 'pick', 'mount', 'turn_part', 'load', 'unload',
    'drive', 'use', 'take', 'cache', 'retrieve', 'reverse', 'cancel', 'haul', 'release')
LEVELS = (4, 38, 103)

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

    def __init__(self,task='a',seed=7,*,render_enabled=False):
        if task not in ('a','b'): raise ValueError('task must be a or b')
        self.task=task
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
        super()._reset_mechanics()
        self.agent.inventory.add('warm_doublet',1)
        self.agent.inventory.equip(self.agent.inventory.find('warm_doublet'))
        self.milestones=[Site(f'{self.task.upper()}{i:02}',*row,elevation=self.levels[row[2]]) for i,row in enumerate(self.site_specs(),1)]
        self.by_id={m.id:m for m in self.milestones}; self.completed=[]; self.targets={}
        self.satchel=Counter(); self.exposed={}; self.held_part=None; self.hauling=None
        self.cargo={}; self.rafts={}; self.movers={}; self.parts={}; self.online=set()
        self.last_text=''; self.closed=False; self.selected_calibration=None; self.stable=0.; self._pending=[]
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
                        cid=mv['cargo']; c=self.cargo[cid]
                        c['parent']=None; c['pos']=q+(4,0,0); mv['cargo']=None
                        self.by_id['A12' if name=='freight' else 'A14'].state['done']=True
        return events

    def _task_outcome(self,events):
        reward=0.
        for m in self.milestones:
            s=m.state
            if s['installed'] and m.kind in ('station','ascend_station','calibrate'):
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
        return (inputs and self.by_id['B19'].state.get('station_part')=='spindle'
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
        obs['text']=((self.current.title if self.current else 'Course complete')+'. '+self.last_text+f'. Satchel {dict(self.satchel)}')[:1024]
        return obs

    def _info(self,events):
        out=WildlandsEnv._info(self,events)
        out.update(success=self.success,milestones=deepcopy(self.completed),task_progress=len(self.completed)/len(self.milestones))
        return out

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

    def presentation_lights(self):
        return [dict(pos=(m.pos+(-3,4,-3)).tolist(),color=[.35,.8,.75],power=1800.,radius=1.)
                for m in self.milestones if m.layer==0]
