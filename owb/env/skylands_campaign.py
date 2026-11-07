"""Full-scale Skylands campaigns with regional logistics and complete travel.

Shares tested local mechanics with the compact testbed, but authors a separate
world: regional roads/caverns/islands, real ascent/descent connections, several
return moorings, separate regional reserves, and navigable cargo waterways.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import heapq
import hashlib
import json
import numpy as np
from .skylands_reference import SkylandsReferenceEnv, Site, A, B_TASK, plain
from ..core import blocks as B
from ..world.voxel import VoxelWorld
from ..world.worldgen import WorldGenResult, Landmark
from ..world.skylands import _island, _tree

A_PLACES=[(90,90),(170,100),(300,130),(400,170),(480,240),(80,500),(500,400),
          (430,400),(80,410),(120,430),(150,430),(430,400),(80,500),(460,520),(460,520)]
B_PLACES=[(90,90),(170,100),(300,130),(400,170),(480,240),(300,300),(150,330),
          (400,100),(90,120),(480,440),(70,430),(100,500),(150,500),(430,470),
          (470,470),(300,360),(480,360),(450,360),(450,450),(200,540),
          (250,540),(290,540),(350,540),(350,540)]


class SkylandsCampaignEnv(SkylandsReferenceEnv):
    levels=(8,48,112)
    presentation_kind='skylands_campaign'

    def __init__(self,task='a',seed=7,*,render_enabled=False):
        self.geography_scale=1.5 if task=='a' else 2.0
        super().__init__(task,seed,render_enabled=render_enabled)
        self.cfg.world_cfg=replace(self.cfg.world_cfg,size_x=int(640*self.geography_scale),
                                  size_y=144,size_z=int(704*self.geography_scale))
        self.cfg.max_steps=80000
        self.cfg.render_cfg.max_dist=300
        self.vertical_xy=self.point(47,140)
        self.descent_xy=self.point(40,120)

    def point(self,x,z): return (int(round(x*self.geography_scale)),int(round(z*self.geography_scale)))

    def site_specs(self):
        rows=A if self.task=='a' else B_TASK; places=A_PLACES if self.task=='a' else B_PLACES
        return [(title,kind,layer,*self.point(x,z)) for (title,kind,layer,_,_),(x,z) in zip(rows,places)]

    def boat_length(self,m):
        base={'A02':70,'A03':80,'A06':10,'A11':260,'B02':70,'B03':80,'B13':220,'B16':150,'B23':160}
        return int(base.get(m.id,80)*self.geography_scale)

    def shuttle_span(self,m): return 26 if m.id in ('B06','B07') else 34

    def _generate_world(self):
        sx,sy,sz=self.cfg.world_cfg.size_x,self.cfg.world_cfg.size_y,self.cfg.world_cfg.size_z
        w=VoxelWorld(sx,sy,sz,sea_level=0); self._road_mask=np.zeros((sx,sz),bool)
        xx=np.arange(sx)[:,None]; zz=np.arange(sz)[None,:]
        h=(47+5*np.sin(xx/77)*np.cos(zz/61)+3*np.sin((xx+zz)/121)).astype(np.int16)
        w.fill_box((0,0,0),(sx-1,7,sz-1),B.BASALT)
        # The surface is an irregular stone roof over connected deep caverns.
        for x in range(sx):
            for offset in range(4): w.vox[x,h[x]-offset,np.arange(sz)]=B.STONE
            w.vox[x,h[x],np.arange(sz)]=B.GRASS
        specs=self.site_specs()
        hubs={0:self.point(60,180),1:self.point(60,180),2:self.point(60,180)}
        for _,_,layer,x,z in specs:
            y=self.levels[layer]
            if layer==2: _island(w,x,z,48,42,y-1,27,(x+z)*.013)
            self._pad(w,(x,y,z),radius=12)
        # A connected regional trail, rather than the compact course's tile grid.
        # Surface tracks cut through hills; sky tracks are physical stone bridges.
        for layer in range(3):
            positions=[(x,z) for _,_,l,x,z in specs if l==layer]
            positions += [self.vertical_xy,self.descent_xy]
            if layer==0: positions += [self.point(80,100),self.point(120,140),self.point(160,100)]
            if layer==1: positions += [self.point(70,110),self.point(90,130),self.point(110,110)]
            for x,z in positions:
                hx,hz=hubs[layer]
                bend=(hx,int((hz+z)/2))
                self._road(w,layer,[(x,z),(x,hz),bend,(hx,hz)])
        # Dense, stable botanical stands and irregular rock/column silhouettes.
        rng=np.random.default_rng(self._seed+71777)
        for _ in range(int(170*self.geography_scale)):
            x=int(rng.integers(20,sx-20)); z=int(rng.integers(20,sz-20))
            if self._road_mask[max(0,x-5):x+6,max(0,z-5):z+6].any(): continue
            y=int(h[x,z])+1
            if any(l==1 and abs(x-px)<25 and abs(z-pz)<25 for _,_,l,px,pz in specs): continue
            _tree(w,x,y,z,int(rng.integers(5,10)))
        for i in range(int(65*self.geography_scale)):
            x=int(rng.integers(20,sx-20)); z=int(rng.integers(20,sz-20))
            if self._road_mask[x-7:x+8,z-7:z+8].any(): continue
            if any(abs(x-px)<24 and abs(z-pz)<24 for _,_,_,px,pz in specs): continue
            width=int(rng.integers(2,5)); height=int(rng.integers(14,31))
            for dx in range(-width,width+1):
                for dz in range(-width,width+1):
                    if dx*dx+dz*dz<(width+.3)**2:
                        top=8+height-abs(dx)-abs(dz)
                        w.fill_box((x+dx,8,z+dz),(x+dx,top,z+dz),B.BASALT)
            w.set_block(x-width-1,8,z,B.CRYSTAL)
        for _,_,layer,x,z in specs:
            if layer==2:
                for dx,dz in ((-20,-18),(22,15),(-24,19)):
                    if B.SOLID[w.get(x+dx,111,z+dz)]: _tree(w,x+dx,112,z+dz,5+(x+z)%3)
        self._author_vertical_connections(w)
        w.temperature[:]=.8; w.moisture[:]=.5; w.biome[:]=0
        w.meta.update(theme='skylands',layered=True,layer_heights=(7,47,111),
            bridge=(10,111,10,6),beacon=(12,112,12),parts=[],scenery=[],
            crystal_lights=[tuple(p+.5) for p in np.argwhere(w.vox==B.CRYSTAL)])
        w.set_region_names({0:'Aster Reach campaign'})
        w.recompute_heightmaps()
        x,z=self.point(90,90); spawn=(x-2.5,48.01,z+.5)
        return WorldGenResult(w,spawn,landmarks=[Landmark('camp','Aster Reach landing',spawn,5)])

    def _pad(self,w,pos,radius=10):
        x,y,z=map(int,pos)
        w.fill_box((x-radius,y-1,z-radius),(x+radius,y-1,z+radius),B.MOSS_STONE if y<30 else B.GRASS)
        w.fill_box((x-radius,y,z-radius),(x+radius,y+9,z+radius),B.AIR)

    def _road(self,w,layer,points):
        y=self.levels[layer]
        for (ax,az),(bx,bz) in zip(points,points[1:]):
            # Manhattan segments retain explicit traversable geometry.
            for (x0,z0),(x1,z1) in [((ax,az),(bx,az)),((bx,az),(bx,bz))]:
                lo=(min(x0,x1)-2,y-1,min(z0,z1)-2); hi=(max(x0,x1)+2,y-1,max(z0,z1)+2)
                w.fill_box(lo,hi,B.BRICK)
                w.fill_box((lo[0],y,lo[2]),(hi[0],y+2,hi[2]),B.AIR)
                self._road_mask[lo[0]:hi[0]+1,lo[2]:hi[2]+1]=True

    def _author_vertical_connections(self,w):
        x,z=self.vertical_xy
        w.fill_box((x-2,8,z-3),(x+4,110,z+3),B.AIR)
        # Three genuine Ascend steps. Offset holes provide safe reverse travel.
        for floor,left,right,hole in ((27,-13,4,-11),(71,-13,4,-11),(95,-9,4,-7),(111,-5,10,-3)):
            w.fill_box((x+left,floor,z-4),(x+right,floor,z+4),B.BRICK)
            w.fill_box((x+hole,floor,z-1),(x+hole+2,floor,z+1),B.AIR)
        for y in self.levels: self._road(w,self.levels.index(y),[(x,z),self.point(60,180)])
        # Restore the intended holes after road construction.
        for floor,hole in ((27,-11),(71,-11),(95,-7),(111,-3)):
            w.fill_box((x+hole,floor,z-1),(x+hole+2,floor,z+1),B.AIR)
        # Deep ascent needs an intermediate ceiling within the native 32 m reach.
        w.fill_box((x-2,27,z-3),(x+4,27,z+3),B.BRICK)
        dx,dz=self.descent_xy
        w.fill_box((dx-3,8,dz-3),(dx+3,57,dz+3),B.AIR)
        w.fill_box((dx-4,7,dz-4),(dx+4,7,dz+4),B.BASALT)
        # Separate sky descent aperture lands on the surface, above solid ground.
        w.fill_box((dx+15,49,dz-4),(dx+23,114,dz+4),B.AIR)
        w.fill_box((dx+14,47,dz-5),(dx+24,47,dz+5),B.BRICK)
        w.fill_box((dx+14,48,dz-5),(dx+24,50,dz+5),B.AIR)
        self._road(w,2,[(dx+12,dz),(dx+12,dz+10),self.point(60,180)])
        # The sky hoist has its own service ascent chain, not a return to a hub.
        if self.task=='a':
            hx,hz=self.point(460,520)
            w.fill_box((hx-2,54,hz-2),(hx+2,110,hz+2),B.AIR)
            for floor in (77,101,111):
                w.fill_box((hx-2,floor,hz-2),(hx+2,floor,hz+2),B.BRICK)
            self._road(w,2,[(hx,hz),self.point(60,180)])

    def _reset_mechanics(self):
        super()._reset_mechanics()
        # Remove the testbed's universal maintenance lift. Regional physical
        # ascent/descent routes and the task's actual freight lifts do the work.
        self._clear_deck(self.movers['transit']); self.movers.pop('transit')
        for tid in [self.transit_target,*self.lift_calls]: self.targets.pop(tid,None)
        self.cache_bindings={}; self.return_choice=None; self.return_rafts=set()
        self._drive_events=[]
        if self.task=='a':
            m=self.by_id['A06']; self.return_rafts.add('A06')
            self.cache_contents={}
            for tid in [k for k,o in self.targets.items() if o['role']=='cache']: self.targets.pop(tid)
            for k in (1,2):
                x,z=self.point(80,500+20*k)
                alt=Site(f'A06_dock{k}',m.title,'stage',1,x,z,elevation=48)
                self._pad(self.world,alt.pos,12); self._road(self.world,1,[(x,z),self.point(60,180)])
                self._add_raft(alt,alt.id); self.return_rafts.add(alt.id)
                cid=alt.id+'_ballast'; self._add_cargo(cid,alt.pos+(-2,0,4),240)
                self.rafts[alt.id]['cargo_id']=cid
            dest=self.by_id['A14'].pos+(-16,0,4)
            for rid in sorted(self.return_rafts):
                r=self.rafts[rid]; r['destination']=dest.copy(); r['receiver']=dest+(5,0,0)
                r['route']=[np.array((r['pos'][0],48,dest[2])),dest.copy()]
                self._water_route([r['pos'],*r['route']],width=3)
            # Solid receiver apron connects all return choices to the hoist.
            self._pad(self.world,self.by_id['A14'].pos+(8,0,4),5)
            self._author_vault()
        else:
            for tid in [k for k,o in self.targets.items() if o['role']=='cache']: self.targets.pop(tid)
            self.cache_contents={k:[] for k in range(6)}
            for k,(layer,item,x,z) in enumerate([(0,'lamp_kit',80,100),(0,'lamp_kit',120,140),(0,'lamp_kit',160,100),
                                                (1,'lock_coupling',70,110),(1,'lock_coupling',90,130),(1,'lock_coupling',110,110)]):
                px,pz=self.point(x,z); pos=np.array((px,self.levels[layer],pz),float)
                self._pad(self.world,pos,7)
                self._target('regional_reserves','cache',pos+(0,1,0),option=k,item=item,layer=layer)
            r=self.rafts['B23']
            self.refuge_docks=[np.array((*self.point(510,580),),float),np.array((*self.point(450,630),),float)]
            self.refuge_docks=[np.array((p[0],48,p[1]+4)) for p in self.refuge_docks]
            self.refuge_receivers=[p+(0,0,5) for p in self.refuge_docks]
            for dest in self.refuge_docks:
                self._water_route([r['pos'],np.array((dest[0],48,r['pos'][2])),dest],width=6)
                self._pad(self.world,dest+(0,0,6),4)
            r['destination']=self.refuge_docks[self.destination].copy()
            r['receiver']=self.refuge_receivers[self.destination].copy()
            r['route']=[np.array((r['destination'][0],48,r['pos'][2])),r['destination'].copy()]
        self._connect_logistics()
        for mv in self.movers.values():
            self._clear_deck(mv); self._write_deck(mv)
        self.world.recompute_heightmaps()

    def _add_raft(self,m,rid):
        if rid=='B23': m=replace(m,z=m.z+8)
        super()._add_raft(m,rid); r=self.rafts[rid]
        r.update(destination=np.array((r['end'],m.y,m.z+4)),receiver=np.array((r['end']+5,m.y,m.z+4)),
                 drive_vector=np.zeros(3),driving=False,origin_pos=r['pos'].copy())
        r['route']=[r['destination'].copy()]
        self._water_route([r['pos'],r['destination']],width=6)
        # Real gathering work across the supply yard, with inspectable variants.
        offsets=((-18,-16),(-28,14),(-14,-27),(-30,-22))
        for (pid,p),(dx,dz) in zip([(pid,p) for pid,p in self.parts.items() if p['owner']==rid],offsets):
            pos=m.pos+(dx,0,dz); p['pos']=pos
            self._pad(self.world,pos,3)
            self._road(self.world,m.layer,[(int(pos[0]),int(pos[2])),(m.x-3,m.z)])
            for target in self.targets.values():
                if target.get('part')==pid: target['pos']=pos+(0,.7,0)
        # Spare compatible hull: several supported assemblies are possible.
        pid=max(self.parts,default=0)+1
        self.parts[pid]=dict(kind='pontoon',pos=m.pos+(-23,0,21),owner=rid,mounted=False,turn=0,capacity=260)
        self._pad(self.world,self.parts[pid]['pos'],3)
        self._road(self.world,m.layer,[(m.x-23,m.z+21),(m.x-3,m.z)])
        self._target(rid,'part',self.parts[pid]['pos']+(0,.7,0),part=pid)

    def _add_reversal(self,m):
        super()._add_reversal(m)
        self.world.fill_box((m.x+1,m.y,m.z-2),(m.x+self.shuttle_span(m)+5,m.y+3,m.z+2),B.AIR)
        end=m.x+self.shuttle_span(m)+3
        # A supported service egress from the landing prevents a successful
        # platform crossing from stranding the traveler above uneven terrain.
        self._road(self.world,m.layer,[(end,m.z),(end,m.z+8),(m.x-4,m.z+8),(m.x-4,m.z)])

    def _water_route(self,points,width):
        y=int(points[0][1])
        for a,b in zip(points,points[1:]):
            ax,az=int(a[0]),int(a[2]); bx,bz=int(b[0]),int(b[2])
            if ax!=bx and az!=bz: raise ValueError('Author orthogonal navigable waterway legs')
            if az==bz:
                lo=(min(ax,bx)-1,y-3,az-width); hi=(max(ax,bx)+1,y-1,az+width)
            else:
                lo=(ax-width,y-3,min(az,bz)-1); hi=(ax+width,y-1,max(az,bz)+1)
            self.world.fill_box(lo,hi,B.WATER)
            self.world.fill_box((lo[0],y,lo[2]),(hi[0],y+4,hi[2]),B.AIR)

    def _author_vault(self):
        m=self.by_id['A10']; x,y,z=m.x,m.y,m.z
        for lo,hi in [((x-10,y,z-10),(x-10,y+6,z+10)),((x+10,y,z-10),(x+10,y+6,z+10)),
                      ((x-10,y,z-10),(x+10,y+6,z-10)),((x-10,y,z+10),(x+10,y+6,z+10))]:
            self.world.fill_box(lo,hi,B.BRICK)
        self.vault_gate=((x-10,y,z-2),(x-10,y+3,z+2))

    def _dispatch(self,command,ingredients,action):
        tid=int(action['target']); obj=self.targets.get(tid); option=int(action['option'])
        for m in self.milestones:
            if 'load_target' in m.state:
                self.targets[m.state['load_target']]['pos']=self.movers[m.state['mover']]['pos']+(0,1,0)
        if obj and obj['role']=='raft' and obj['owner'] in self.rafts: obj['pos']=self.rafts[obj['owner']]['pos']+(0,1,0)
        if obj and command=='haul' and obj['role']=='cargo':
            cid=obj['owner']; c=self.cargo[cid]
            if c['anchored'] or c['parent'] or np.linalg.norm(c['pos']-self.agent.pos)>5:
                return self._advance(self._fail('requested loose cargo is not within tow reach'))
            self.hauling=cid; self.tow_offset=c['pos']-self.agent.pos
            return self._advance([dict(type='cargo_towed',cargo=cid)])
        if obj and command=='inspect' and obj['role'] in ('part','raft'):
            if not self._reachable(obj['pos'],6): return self._advance(self._fail('inspect within local visible reach'))
            if obj['role']=='part':
                p=self.parts[obj['part']]; text=f"{p['kind']}; flotation rating {p['capacity']} kg; orientation {p['turn']}."
            else:
                r=self.rafts[obj['owner']]; mass=160+(self.cargo[r['cargo']]['mass'] if r['cargo'] else 0)
                text=f"Total load {mass} kg; charge {r['charge']}; supported {self.craft_valid(r)}."
            self.last_text=text; return self._advance([dict(type='clue_exposed',site=obj['owner'],text=text)])
        if obj and command=='cache' and obj['role']=='cache':
            if not self._reachable(obj['pos']): return self._advance(self._fail('cache not reachable'))
            item=obj['item']; key=obj['option']
            if not self.satchel[item]: return self._advance(self._fail('reserve item not carried'))
            self.satchel[item]-=1; self.cache_contents[key].append(item); self.cache_bindings[item]=key
            self.by_id['B09'].state['done']={'lamp_kit','lock_coupling'}<=self.cache_bindings.keys()
            return self._advance([dict(type='reserves_cached',cache=key,item=item,layer=obj['layer'])])
        if obj and obj['owner'] in self.rafts:
            rid=obj['owner']; r=self.rafts[rid]
            if command=='drive':
                if not self._reachable(obj['pos'],6) or not self.on_deck(r) or not self.craft_valid(r):
                    return self._advance(self._fail('board a charged, supported craft before driving'))
                v=np.sign(action['build_move']).astype(float); v[1]=0
                if np.count_nonzero(v)>1: return self._advance(self._fail('one steering axis per decision'))
                r['drive_vector']=v if option!=2 else np.zeros(3); r['driving']=bool(np.any(r['drive_vector']))
                r['moored']=False
                return self._advance([])
            if command=='use' and rid in self.return_rafts:
                if not self._reachable(obj['pos']): return self._advance(self._fail('moor within reach'))
                if not self.craft_valid(r) or r['travel']<8 or np.linalg.norm(r['pos']-r['origin_pos'])>.1:
                    return self._advance(self._fail('complete the loaded trial and return to the chosen dock'))
                r['drive_vector'][:]=0; r['driving']=False; r['moored']=True; self.return_choice=rid
                self.by_id['A06'].state.update(done=True,raft=rid); self.by_id['A13'].state['raft']=rid
                return self._advance([dict(type='raft_staged',raft=rid,position=r['pos'].tolist())])
            if command=='unload':
                if not self._reachable(obj['pos'],6) or not r['cargo']: return self._advance(self._fail('loaded craft not in reach'))
                cid=r['cargo']; c=self.cargo[cid]
                staging=r['moored'] and np.linalg.norm(r['pos']-r['origin_pos'])<.1
                destination=self.refuge_docks[option] if rid=='B23' and option<2 else r['destination']
                if not staging and np.linalg.norm(r['pos']-destination)>.1: return self._advance(self._fail('reach the actual receiving dock before unloading'))
                receiver=r['origin_pos']+(-3,0,0) if staging else (self.refuge_receivers[option] if rid=='B23' else r['receiver'])
                c.update(parent=None,pos=receiver.copy()); r['cargo']=None; r['drive_vector'][:]=0; r['driving']=False
                if rid in self.by_id and self.by_id[rid].kind=='raft': self.by_id[rid].state['done']=True
                if rid in self.return_rafts and cid=='core': self.by_id['A13'].state['done']=True
                if rid=='B23':
                    c['delivered_to']=option
                    if option==self.destination:
                        c['anchored']=True; c['parent']='refuge'; self.by_id['B24'].state['done']=True
                return self._advance([dict(type='cargo_delivered',cargo=cid,raft=rid,receiver=receiver.tolist())])
        events=super()._dispatch(command,ingredients,action)
        if command=='mount' and obj and obj['owner'] in self.rafts and option==3 and any(e['type']=='part_mounted' for e in events):
            self.rafts[obj['owner']]['charge']=300.
        return events

    def _advance(self,events):
        for rid,r in self.rafts.items():
            v=r.get('drive_vector',np.zeros(3))
            if not np.any(v) or not self.on_deck(r) or not self.craft_valid(r): continue
            q=r['pos']+v; x,y,z=np.rint(q).astype(int)
            cell=(x,y-1,z); material=r['under'].get(cell,self.world.get(*cell))
            if material!=B.WATER or self._move_deck(r,q) is None:
                r['drive_vector'][:]=0; r['driving']=False
                events.append(dict(type='motion_blocked',object=rid)); continue
            r['charge']-=.25; r['travel']+=1; r['tested']=r['travel']>=3
            events.append(dict(type='craft_moved',raft=rid,position=q.tolist(),charge=r['charge']))
        return super()._advance(events)

    def _operate(self,m,role,option):
        events=super()._operate(m,role,option)
        if m.id=='A09' and m.state['done']:
            self.world.fill_box(*self.vault_gate,B.AIR)
            events.append(dict(type='vault_door_opened'))
        return events

    def system_operational(self):
        if self.task=='a': return super().system_operational()
        c=self.cargo['relief']
        return (all(self.by_id[mid].state['installed'] and self.by_id[mid].state['stable']>=5 for mid in ('B15','B17','B19'))
            and all(self.cargo[cid]['anchored'] and self.cargo[cid]['parent']==mid and np.linalg.norm(self.cargo[cid]['pos']-self.by_id[mid].pos)<6
                    for cid,mid in (('power_module','B15'),('marsh_coupling','B17')))
            and self.by_id['B19'].state.get('station_part')=='spindle' and self.by_id['B21'].state['installed']
            and self.by_id['B23'].state['installed'] and self.by_id['B24'].state['done']
            and c['intact'] and c['parent']=='refuge' and c.get('delivered_to')==self.destination
            and np.linalg.norm(c['pos']-self.refuge_receivers[self.destination])<.1)

    def state_digest(self):
        base=super().state_digest()
        return hashlib.sha256((base+json.dumps(plain(dict(bindings=self.cache_bindings,return_choice=self.return_choice)),sort_keys=True)).encode()).hexdigest()


    def cache_position(self,k):
        return next(o['pos']-(0,1,0) for o in self.targets.values() if o['role']=='cache' and o['option']==k)

    def refuge_position(self,k): return self.refuge_receivers[k]

    def closure_bounds(self):
        m=self.by_id['B20']
        return (m.x+7,m.y,m.z-2),(m.x+9,m.y+3,m.z+2)

    def _connect_logistics(self):
        """Connect real loading aprons and trails without paving over waterways."""
        from scipy.ndimage import binary_dilation
        endpoints={k:[] for k in range(3)}
        for m in self.milestones:
            endpoints[m.layer].append((m.x-(14 if m.id=='A10' else 4),m.z))
        for r in self.rafts.values():
            layer=self.levels.index(int(r['pos'][1]))
            for pos in (r['origin_pos']+(-4,0,0),r['receiver']):
                self._pad(self.world,pos,3)
                endpoints[layer].append((int(pos[0]),int(pos[2])))
        for mv in self.movers.values():
            if mv['kind']=='freight':
                pos=mv['pos']+(4,mv['top']-mv['pos'][1],0)
                self._pad(self.world,pos,2)
                endpoints[self.levels.index(int(pos[1]))].append((int(pos[0]),int(pos[2])))
        if self.task=='b':
            for p in self.refuge_receivers:
                self._pad(self.world,p,3); endpoints[1].append((int(p[0]),int(p[2])))
            for o in self.targets.values():
                if o['role']=='cache': endpoints[o['layer']].append(tuple(o['pos'][[0,2]].astype(int)))
        # Route each trail around water; 2 m shoulders keep the cradle supported.
        for layer,points in endpoints.items():
            y=self.levels[layer]; water=self.world.vox[:,y-1,:]==B.WATER
            blocked=binary_dilation(water,iterations=3)
            # Avoid authored service walls and the known descent apertures.
            solid=B.SOLID[self.world.vox[:,y,:]] & (self.world.vox[:,y,:]==B.BRICK)
            blocked |= binary_dilation(solid,iterations=3)
            for mv in self.movers.values():
                if mv['kind']=='freight' and mv['pos'][1]<y<=mv['top']:
                    mx,mz=map(int,mv['pos'][[0,2]])
                    blocked[mx-3:mx+4,mz-3:mz+4]=True
            for x,z in [self.descent_xy,(self.descent_xy[0]+19,self.descent_xy[1])]:
                blocked[x-6:x+7,z-6:z+7]=True
            hx,hz=self.point(60,180); goal=(hx,hz)
            for point in points:
                start=tuple(point)
                # The loading apron itself is traversable, even close to water.
                blocked_here=blocked.copy(); px,pz=start
                blocked_here[px-1:px+2,pz-1:pz+2]=False
                queue=[(0,0,start)]; previous={start:None}; cost={start:0}; found=None
                while queue:
                    _,g,q=heapq.heappop(queue)
                    if g!=cost[q]: continue
                    if q==goal: found=q; break
                    for n in ((q[0]-1,q[1]),(q[0]+1,q[1]),(q[0],q[1]-1),(q[0],q[1]+1)):
                        nx,nz=n
                        if not (6<nx<self.world.size_x-6 and 6<nz<self.world.size_z-6) or blocked_here[n]: continue
                        # Existing trails receive preference, preserving regional branches.
                        c=g+(1 if self._road_mask[n] else 2)
                        if c<cost.get(n,1e12):
                            cost[n]=c; previous[n]=q
                            heapq.heappush(queue,(c+abs(nx-hx)+abs(nz-hz),c,n))
                if found is None: raise RuntimeError(f'No logistics trail from {layer}:{start}')
                path=[]
                while found is not None: path.append(found); found=previous[found]
                # Group straight stretches to avoid voxel-by-voxel height-map updates.
                path.reverse(); corners=[path[0]]
                for i in range(1,len(path)-1):
                    if (path[i][0]-path[i-1][0],path[i][1]-path[i-1][1]) != (path[i+1][0]-path[i][0],path[i+1][1]-path[i][1]): corners.append(path[i])
                corners.append(path[-1]); self._road(self.world,layer,corners)
        # Preserve freight shafts after the connected road network is authored.
        for mv in self.movers.values():
            if mv['kind']=='freight':
                x,y,z=map(int,mv['pos'])
                self.world.fill_box((x-1,y,z-1),(x+1,mv['top']+3,z+1),B.AIR)
        # Reopen intended holes after nearby trail work.
        dx,dz=self.descent_xy
        self._road(self.world,1,[(dx-6,dz),(dx-6,dz+10),self.point(60,180)])
        self._road(self.world,1,[(dx+19,dz),(dx+19,dz+10),self.point(60,180)])
        self.world.fill_box((dx-3,8,dz-3),(dx+3,57,dz+3),B.AIR)
        self.world.fill_box((dx+15,49,dz-4),(dx+23,114,dz+4),B.AIR)


    def presentation_lights(self):
        lights=super().presentation_lights()
        if self.satchel['lamp']:
            lights.append(dict(pos=(self.agent.pos+(0,2.4,0)).tolist(),color=[1.,.78,.46],power=1100.,radius=.45))
        return lights
