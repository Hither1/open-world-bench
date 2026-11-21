"""Action-only full-campaign oracle; routing uses privileged geometry reads."""
import math
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label
from scipy.spatial import cKDTree
from .skylands_reference import SkylandsReferencePolicy, neutral
from ...core import blocks as B
from ...env.skylands_reference import COMMANDS
from ...sim.riding import MOUNTED_OK, WHISTLE_RANGE, gait

APPLE_RESERVE=3       # apples kept for healing and for calming horses
FORAGE_RADIUS,FORAGE_TICKS=18.0,300
GLIDE_REACH=120.0   # metres of open-air glide from a full stamina bar, with margin


class SkylandsCampaignPolicy(SkylandsReferencePolicy):
    def __init__(self,env):
        super().__init__(env)
        self.strict_paths=True
        self.descending=None; self.chosen_return='A06'+('_dock'+str(env._seed%3) if env._seed%3 else '')
        self.bindings={'lamp_kit':env._seed%3,'lock_coupling':3+(env._seed+1)%3}
        self.walking=False; self.glide_plan=None; self._edges=None

    # -- riding, open-air glides and fan striking (organic campaign) --------
    def walk(self,pos):
        self.walking=True
        e=self.env
        if getattr(e,'riding',None) is not None and e.riding.mounted and self.path:
            y=int(round(pos[1]))
            if self.stuck>4:
                self.path=[]; self.key=None        # replan from the rider's own cell
            else:
                near=[float(np.linalg.norm(e.agent.pos[[0,2]]-(np.asarray(q)+.5))) for q in self.path[:10]]
                k=int(np.argmin(near))
                if k and self.clear_line(e.agent.pos,self.path[k],y): del self.path[:k]
                while (len(self.path)>3 and np.linalg.norm(e.agent.pos[[0,2]]-(np.asarray(self.path[0])+.5))<1.2
                       and self.clear_line(e.agent.pos,self.path[1],y)):
                    self.path.pop(0)
        a=super().walk(pos)
        mounted=getattr(e,'riding',None) is not None and e.riding.mounted
        if not mounted and e.agent.on_ground and .5<float(pos[1])-float(e.agent.pos[1])<1.6:
            a['buttons'][0]=1; self.explanation='Hop up out of the dip onto the trail'
        if mounted and len(self.path)>3:
            ahead=self.path[:6]; y=int(round(pos[1]))
            # Look ahead only to a node the rider's body can reach in a straight line.
            target=self.path[0] if self.stuck>4 else next(
                (q for q in reversed(ahead[:3]) if self.clear_line(e.agent.pos,q,y)),self.path[0])
            a=self.steer((target[0]+.5,pos[1],target[1]+.5))
            straight=len(ahead)==6 and (len({q[0] for q in ahead})==1 or len({q[1] for q in ahead})==1)
            gallop=straight and e.riding.stamina>30 and gait(e.riding.distance)[0]=='canter'
            a['buttons'][1]=int(gallop)
            g=e.riding.state()
            self.explanation=f"Ride the trail ({'gallop' if gallop else g['gait']}, {g['distance']:.0f} m ridden)"
        return a

    def clear_line(self,start,cell,y,half=.4):
        """True when a body of half-width ``half`` can move straight from
        ``start`` to the centre of ``cell`` over walkable cells at ``y``."""
        a=np.asarray(start,float)[[0,2]]; b=np.asarray(cell,float)+.5
        d=b-a; n=float(np.linalg.norm(d))
        if n<1e-6: return True
        side=np.array((-d[1],d[0]))/n*half
        for t in np.linspace(0,1,max(2,int(n/.25)+1)):
            for o in (side,-side):
                q=np.floor(a+d*t+o).astype(int)
                if not self.walkable(int(q[0]),int(q[1]),y): return False
        return True

    def cmd(self,name,**kw):
        a=neutral(); a['command']=self.env.commands.index(name); a.update(kw); return a

    def with_horse(self,action):
        """Ride long surface legs; dismount for any command, jump, glide or layer change."""
        e=self.env; r=getattr(e,'riding',None)
        if r is None: return action
        command=e.commands[int(action['command'])]
        surface=abs(e.agent.pos[1]-e.levels[1])<1.5
        if r.mounted:
            leaving=(command not in MOUNTED_OK or action['buttons'][0] or action['buttons'][3] or self.descending
                     or getattr(self,'glide_plan',None) is not None or not surface
                     or any(e.on_deck(x) for x in e.rafts.values()))
            if leaving or (self.walking and len(self.path)<4 and command=='none'):
                if not e.agent.on_ground: return action
                self.path=[]; self.key=None
                a=self.cmd('dismount'); self.explanation='Dismount and leave the horse on the trail'; return a
            return action
        if self.__dict__.get('forage_plan'): return action      # finish gathering apples first
        horse=next((n for n in e.npcs.npcs if n.kind=='horse' and n.alive),None)
        if (horse is None or not self.walking or command!='none' or not surface or e.hauling or e.held_part
                or not e.agent.on_ground or len(self.path)<20 or abs(horse.pos[1]-e.agent.pos[1])>.6):
            return action
        gap=float(np.linalg.norm((horse.pos-e.agent.pos)[[0,2]]))
        if horse.data.get('route') and getattr(self,'whistled_at',None) is not None:
            a=neutral(); self.explanation=f'Wait for the horse to answer the whistle ({gap:.0f} m away)'; return a
        if gap>20:
            # Whistle a bonded horse over before a long ride, at most once per 40 m walked.
            last=getattr(self,'whistled_at',None)
            if (horse.data.get('bonded') and gap<=WHISTLE_RANGE-5 and len(self.path)>=max(60,1.5*gap)
                    and (last is None or np.linalg.norm(e.agent.pos-last)>40)):
                self.whistled_at=e.agent.pos.copy()
                a=self.cmd('whistle'); self.explanation=f'Whistle for the horse ({gap:.0f} m away)'; return a
            return action
        if gap>1.8:
            hx,hz=np.floor(horse.pos[[0,2]]).astype(int)
            if not self.walkable(int(hx),int(hz),int(e.levels[1])): return action
            saved=(self.path,self.key); self.path=[]; self.key=None
            try: a=SkylandsReferencePolicy.walk(self,horse.pos)
            except RuntimeError: a=None
            self.path,self.key=saved
            if a is None: return action
            self.explanation='Walk to the horse'; return a
        delta=(horse.pos-e.agent.pos)[[0,2]]
        error=(math.atan2(delta[1],delta[0])-e.agent.yaw+math.pi)%(2*math.pi)-math.pi
        if abs(error)>.5:
            a=neutral(); a['look'][0]=float(np.clip(error/.36,-1,1)); self.explanation='Face the horse calmly'; return a
        self.path=[]; self.key=None
        if not (horse.data.get('calm') or horse.data.get('bonded')):
            inv=e.agent.inventory
            slot=next((i for i,st in enumerate(inv.stacks) if 'fruit' in st.spec.tags),None)
            if slot is None: return action
            a=self.cmd('feed',slot=slot); self.explanation='Offer the horse an apple to calm it'; return a
        a=self.cmd('ride'); self.explanation='Mount the calmed horse'; return a

    def sky_edges(self):
        """Walkable sky-island cells beside open air down to the surface."""
        if getattr(self,'_edges',None) is None:
            e=self.env; w=e.world; y=e.levels[2]
            stand=B.SOLID[w.vox[:,y-1,:]]&~B.SOLID[w.vox[:,y,:]]&~B.SOLID[w.vox[:,y+1,:]]
            open_air=~B.SOLID[w.vox[:,60:y-1,:]].any(axis=1)
            edge=stand&binary_dilation(open_air,iterations=1)&~open_air
            self._edges=np.argwhere(edge)
        return self._edges

    def landing_site(self,p,edges):
        """A surface landing cell in the walkable region of target ``p``, with
        open sky above, as close to ``p`` as an open-air glide from one of
        ``edges`` can reach. Returns ``(cell, reachable edge mask)`` or None."""
        e=self.env; w=e.world; y=e.levels[1]
        key=(tuple(np.floor(p[[0,2]]).astype(int)),len(edges),tuple(edges[0]) if len(edges) else None)
        cache=self.__dict__.setdefault('_landing_cache',{})
        if key in cache: return cache[key]
        result=None
        stand=B.SOLID[w.vox[:,y-1,:]]&~B.SOLID[w.vox[:,y,:]]&~B.SOLID[w.vox[:,y+1,:]]
        stand[[0,-1],:]=False; stand[:,[0,-1]]=False
        labels,_=label(stand)
        own=labels[int(p[0]),int(p[2])]
        if own and len(edges):
            clear=~B.SOLID[w.vox[:,y:e.levels[2]-1,:]].any(axis=1)
            cells=np.argwhere(binary_erosion((labels==own)&clear,iterations=2))
            if len(cells):
                reach,_=cKDTree(edges).query(cells,distance_upper_bound=GLIDE_REACH)
                ok=np.isfinite(reach)
                if ok.any():
                    cells=cells[ok]
                    c=cells[int(np.argmin(np.linalg.norm(cells+.5-p[[0,2]],axis=1)))]
                    result=(c,np.linalg.norm(edges-c,axis=1)<=GLIDE_REACH)
        cache[key]=result
        return result

    def island_edges(self):
        """Sky edges on the island the agent is standing on."""
        e=self.env; w=e.world; y=e.levels[2]
        labels=self.__dict__.get('_sky_labels')
        if labels is None:
            stand=B.SOLID[w.vox[:,y-1,:]]&~B.SOLID[w.vox[:,y,:]]&~B.SOLID[w.vox[:,y+1,:]]
            labels=self._sky_labels=label(stand)[0]
        x,z=np.floor(e.agent.pos[[0,2]]).astype(int)
        own=labels[x,z]
        edges=self.sky_edges()
        if not own or not len(edges): return edges[:0]
        return edges[labels[edges[:,0],edges[:,1]]==own]

    def glide_descent(self,p):
        """Step off a sky island and glide in open air to a surface landing
        cell connected to ``p``, trading glide against free fall so the
        glider runs out of height over the landing."""
        e=self.env; a_=e.agent; y=float(a_.pos[1])
        plan=getattr(self,'glide_plan',None)
        if plan is None:
            edges=self.island_edges()
            site=self.landing_site(p,edges)
            if site is None: return None
            cell,reach=site
            mine=np.linalg.norm(edges-a_.pos[[0,2]],axis=1)
            ex,ez=edges[int(np.argmin(np.where(reach,mine,np.inf)))]
            launch=np.array((ex+.5,e.levels[2]+.01,ez+.5))
            if not self.near(launch,1.0): return self.walk(launch)
            if a_.stamina<95: self.explanation='Recover stamina before the open-air glide'; return neutral()
            self.glide_plan=dict(target=p.copy(),land=np.array((cell[0]+.5,e.levels[1]+.01,cell[1]+.5)))
            a=self.steer(self.glide_plan['land']); a['buttons'][0]=1; a['move'][:]=(1.,0.)
            self.explanation='Step off the island edge into open air'; return a
        if a_.on_ground and y<e.levels[1]+1.5:
            self.glide_plan=None; self.path=[]; self.key=None
            return None
        land=plan['land']
        a=self.steer(land); a['move'][:]=(1.,0.)
        height=y-land[1]; gap=float(np.linalg.norm((land-a_.pos)[[0,2]]))
        # Glide when the remaining height is only just enough to cover the
        # distance (or the ground is near); otherwise free fall over the site.
        soon=height+min(0.,float(a_.vel[1]))*.5
        if gap/9.0>=height/2.2-.4 or soon<5:
            a['buttons'][3]=1; self.explanation='Glide down over open air toward the surface landing'
        else:
            self.explanation='Free fall over open air; the glider opens for the landing'
        return a

    def charge_fan(self,r):
        """Strike the mounted fan in rhythm: each strike as the last spin fades."""
        e=self.env
        since=e.elapsed-r.get('spun',-99.)
        if np.linalg.norm(r['pos']+(0,1,0)-e.agent.eye)<=3.2 and since<.8:
            self.explanation='Let the fan spin down before the next strike'; return neutral()
        a=self.approach(r['target'],'attack',reach=3.0)
        if e.commands[int(a['command'])]=='attack':
            self.explanation=f"Strike the fan to charge the cell ({r['charge']:.0f} charge)"
        return a

    def walkable(self,x,z,y):
        if not super().walkable(x,z,y): return False
        e=self.env
        if e.hauling:
            q=np.rint(np.array((x+.5,y+.001,z+.5))+e.tow_offset).astype(int)
            return super().walkable(q[0],q[2],q[1])
        return True

    def command(self,name,target,option=0):
        if name in ('haul','release'): self.path=[]; self.key=None
        return super().command(name,target,option)

    def tow(self,cid,goal,control):
        e=self.env;c=e.cargo[cid]
        if e.hauling!=cid:
            return self.approach(c['target'],'haul',reach=1.55)
        if np.linalg.norm(c['pos']-goal)<2.8:return self.command('release',c['target'])
        return self.go(np.asarray(goal)-e.tow_offset)

    def go(self,pos):
        e=self.env; p=np.asarray(pos,float); y=float(e.agent.pos[1]); desired=min(e.levels,key=lambda h:abs(h-p[1]))
        if getattr(self,'glide_plan',None) is not None:
            a=self.glide_descent(p)
            if a is not None: return a
        if self.descending:
            dest,x,z=self.descending
            if y<dest+.2 and e.agent.on_ground:
                self.descending=None; self.path=[]; self.key=None
            else:
                # Free fall in the open shaft, then deploy the native paraglider
                # for the final 17 m. This conserves stamina without fall damage.
                aim=np.array((x+.5,dest,z+.5))
                a=self.steer(aim)
                if y<dest+17:
                    a['buttons'][3]=1
                    if np.linalg.norm(a['move'])<.12: a['move'][1]=.12 if e.steps%12<6 else -.12
                self.explanation=f'Descend the open shaft; deploy glider before the {dest:g} m landing'
                return a
        current=min(e.levels,key=lambda h:abs(h-y))
        # Continue an already-entered ascent tower from its intermediate decks.
        x,z=e.vertical_xy
        at_tower=np.linalg.norm(e.agent.pos[[0,2]]-(x+.5,z+.5))<1.1
        local=e.task=='a' and e.by_id['A14'].state['landed'] and np.linalg.norm(e.agent.pos[[0,2]]-(e.by_id['A14'].pos[[0,2]]+.5))<1.1
        if desired>y+8 and (at_tower or local):
            if e.agent.stamina<25: self.explanation='Recover stamina on the service landing'; return neutral()
            a=neutral(); a['command']=COMMANDS.index('ascend'); self.explanation='Ascend through the next physical tower ceiling'; return a
        if abs(y-desired)>8 and not (current==desired and y<desired+8):
            if desired>y:
                if e.task=='a' and desired==112 and y>50 and e.by_id['A14'].state['landed'] and np.linalg.norm(e.agent.pos[[0,2]]-e.by_id['A14'].pos[[0,2]])<15:
                    target=e.by_id['A14'].pos+(.5,6.01,.5)
                else: target=np.array((x+.5,current+.01,z+.5))
                return self.walk(target)
            dest=48 if y>90 else 8
            if dest==48 and getattr(e,'organic_world',False):
                a=self.glide_descent(p)
                if a is not None: return a
            dx,dz=e.descent_xy; sx=dx+19 if dest==48 else dx
            edge=np.array((sx-5.5,current+.01,dz+.5))
            if not self.near(edge,1): return self.walk(edge)
            if e.agent.stamina<95: self.explanation='Recover stamina before the glider descent'; return neutral()
            self.descending=(dest,sx,dz)
            return self.steer((sx+.5,dest,dz+.5))
        return self.walk(p)

    def approach(self,tid,command='use',option=0,reach=2.8):
        e=self.env; p=e.targets[tid]['pos']
        if e._reachable(p,reach): return self.command(command,tid,option)
        y=int(round(p[1]-1)); candidates=[]
        for dx in range(-4,5):
            for dz in range(-4,5):
                x,z=int(p[0])+dx,int(p[2])+dz
                q=np.array((x+.5,y+.01,z+.5))
                if self.walkable(x,z,y) and np.linalg.norm(q+(0,e.cfg.sim_cfg.eye_height,0)-p)<reach and e.world.line_of_sight(q+(0,e.cfg.sim_cfg.eye_height,0),p):
                    candidates.append((np.linalg.norm(q-e.agent.pos),q))
        if not candidates: raise RuntimeError(f'No physical interaction approach for {tid}:{e.targets[tid]}')
        return self.go(min(candidates,key=lambda pair:pair[0])[1])

    def drive(self,r,vec):
        a=self.command('drive',r['target']); a['build_move']=np.asarray(vec,np.float32)
        self.explanation='Steer the loaded craft through the physical waterway'
        return a

    def raft(self,m,rid,*,stage=False,test_only=False):
        e=self.env
        if stage: rid=self.chosen_return
        if m.kind=='return': rid=e.return_choice
        r=e.rafts[rid]; target=r['target']
        for k,pid in enumerate(r['parts']):
            if pid is None:
                if e.held_part: return self.approach(target,'mount',k,4.8)
                item=next(tid for tid,o in e.targets.items() if o['owner']==rid and o['role']=='part'
                          and not e.parts[o['part']]['mounted'] and e.parts[o['part']]['kind']==('pontoon','pontoon','fan','cell')[k])
                return self.approach(item,'pick',reach=4.8)
        if getattr(e,'organic_world',False) and rid in ('A02','B02') and r['charge']<100 and not e.on_deck(r):
            return self.charge_fan(r)
        cid='core' if m.kind=='return' else r['cargo_id']
        if m.kind=='return' and r['cargo'] and r['cargo']!='core': return self.approach(target,'unload',reach=4.8)
        if r['cargo'] is None:
            c=e.cargo[cid]
            if np.linalg.norm(c['pos']-r['pos'])>5:
                a=self.transport_cargo(cid,r['origin_pos']+(-2.5,0,0),target)
                if a is not None: return a
            return self.approach(target,'load',reach=4.8)
        if not e.on_deck(r): return self.go(r['pos']+(.5,.01,.5))
        if stage:
            if r['travel']>=8 and np.linalg.norm(r['pos']-r['origin_pos'])<.1: return self.command('use',target)
            return self.drive(r,(-1,0,0) if r['pos'][0]>=r['origin']+4 or r['drive_vector'][0]<0 else (1,0,0))
        if test_only and r['travel']>=3: return self.command('drive',target,2)
        if np.linalg.norm(r['pos']-r['destination'])<.1:
            return self.command('unload',target,e.destination if m.kind=='delivery' else 0)
        # The route is a chain of axis-aligned legs from the dock (meandering
        # rivers jog side to side). The current leg is the first one the craft
        # lies on and has not finished, found geometrically, with no writes to
        # simulator state.
        poly=[r['origin_pos'],*r['route']]
        for a,b in zip(poly,poly[1:]):
            if np.linalg.norm(b-r['pos'])<.1: continue
            axis=0 if abs(b[0]-a[0])>.1 else 2; other=2-axis
            lo,hi=sorted((a[axis],b[axis]))
            if abs(r['pos'][other]-a[other])<.1 and lo-.1<=r['pos'][axis]<=hi+.1:
                v=np.zeros(3); v[axis]=np.sign(b[axis]-r['pos'][axis]); return self.drive(r,v)
        raise RuntimeError('Craft is off its route')

    def __call__(self,obs):
        self.walking=False
        action=self.decide(obs)
        if self.env.commands[int(action['command'])]=='none' and (self.walking or self.__dict__.get('forage_plan')):
            food=self.forage()
            if food is not None: action=food
        return self.with_horse(action)

    # -- apples: eat when hurt, keep a few by shaking apple trees ---------
    def forage(self):
        e=self.env; a_=e.agent; orchard=getattr(e,'orchard',None)
        if orchard is None or not getattr(e,'organic_world',False): return None
        inv=a_.inventory
        apples=sum(s.count for s in inv.stacks if s.spec.id=='apple')
        if apples and a_.health<=a_.max_health-2:
            slot=next(i for i,s in enumerate(inv.stacks) if s.spec.id=='apple')
            self.explanation=f'Eat an apple to restore half a heart ({a_.health:.1f}/{a_.max_health:.0f} health)'
            return self.cmd('eat',slot=slot)
        plan=self.__dict__.get('forage_plan')
        level=e.levels[1]
        if (plan is None and apples>=APPLE_RESERVE) or e.riding.mounted or e.hauling or e.held_part \
                or abs(a_.pos[1]-level)>1.5 or self.descending or self.glide_plan is not None \
                or any(e.on_deck(x) for x in e.rafts.values()):
            self.forage_plan=None; return None
        skip=self.__dict__.setdefault('forage_skip',set())
        if plan is None:
            if not a_.on_ground: return None
            best=None
            for t in orchard.trees:
                if not t['apples'] or t['id'] in skip or abs(t['trunk'][1]-level)>2: continue
                d=float(np.hypot(t['trunk'][0]-a_.pos[0],t['trunk'][2]-a_.pos[2]))
                if d<=FORAGE_RADIUS and (best is None or d<best[0]): best=(d,t)
            if best is None: return None
            t=best[1]; tx,tz=int(t['trunk'][0]),int(t['trunk'][2])
            # Any trail cell within striking reach of the trunk.
            stands=[(tx+dx,tz+dz) for dx in range(-2,3) for dz in range(-2,3)
                    if 0<dx*dx+dz*dz<=4 and self.walkable(tx+dx,tz+dz,int(level))]
            if not stands: skip.add(t['id']); return None
            stand=min(stands,key=lambda c:np.hypot(c[0]+.5-a_.pos[0],c[1]+.5-a_.pos[2]))
            plan=self.forage_plan=dict(tree=t['id'],stand=stand,ticks=0,struck=False,path=self.path,key=self.key)
            self.path=[]; self.key=None
        plan['ticks']+=1
        tree=orchard.trees[plan['tree']]
        def finish(give_up=False):
            if give_up: skip.add(tree['id'])
            self.forage_plan=None; self.path=plan['path']; self.key=plan['key']
            return None
        if plan['ticks']>FORAGE_TICKS: return finish(True)
        try:
            if plan['struck']:
                fallen=[q for q in e.gameplay.plants if q.get('tree')==tree['id'] and not q.get('used')
                        and abs(q['pos'][1]-level)<.6 and self.walkable(int(q['pos'][0]),int(q['pos'][2]),int(level))]
                if not fallen or apples>=APPLE_RESERVE+2: return finish()
                q=min(fallen,key=lambda q:np.hypot(q['pos'][0]-a_.pos[0],q['pos'][2]-a_.pos[2]))
                if e.gameplay.reachable(q['pos']):
                    self.explanation='Pick up the fallen apple'; return self.cmd('interact')
                a=SkylandsReferencePolicy.walk(self,(q['pos'][0],level+.01,q['pos'][2]))
                self.explanation='Walk to the fallen apple'; return a
            sx,sz=plan['stand']
            if np.hypot(sx+.5-a_.pos[0],sz+.5-a_.pos[2])>.45:
                a=SkylandsReferencePolicy.walk(self,(sx+.5,level+.01,sz+.5))
                self.explanation='Walk to an apple tree'; return a
            delta=np.array(tree['trunk'])[[0,2]]-a_.pos[[0,2]]
            error=(math.atan2(delta[1],delta[0])-a_.yaw+math.pi)%(2*math.pi)-math.pi
            if abs(error)>.25:
                a=neutral(); a['look'][0]=float(np.clip(error/.36,-1,1)); self.explanation='Face the apple tree'; return a
            if a_.cooldown>0: self.explanation='Ready a swing at the trunk'; return neutral()
            plan['struck']=True
            self.explanation='Strike the apple tree to shake apples loose'; return self.cmd('attack')
        except RuntimeError:
            return finish(True)

    def decide(self,obs):
        e=self.env; m=e.current
        if not m: return neutral()
        if m.id!=self.last: self.last=m.id; self.count=0; self.path=[]; self.key=None
        self.count+=1
        self.stuck=getattr(self,'stuck',0)+1 if np.linalg.norm(e.agent.pos-getattr(self,'last_position',e.agent.pos))<1e-3 else 0
        self.last_position=e.agent.pos.copy()
        if self.stuck>500: raise RuntimeError(f'No physical progress at {m.id}: {e.agent.pos}, {self.explanation}')
        if self.count>24000: raise RuntimeError(f'Stalled at {m.id}: {e.agent.pos}, {m.state}')
        s=m.state; tid=s['control']
        if m.kind=='cache':
            for item in ('lamp_kit','lock_coupling'):
                if item not in e.cache_bindings:
                    target=next(k for k,o in e.targets.items() if o['role']=='cache' and o['option']==self.bindings[item])
                    return self.approach(target,'cache')
        if m.kind in ('outpost','retrieve'):
            item='lamp_kit' if m.kind=='outpost' else 'lock_coupling'
            if not e.satchel[item]:
                target=next(k for k,o in e.targets.items() if o['role']=='cache' and o['option']==self.bindings[item])
                return self.approach(target,'retrieve',0 if item=='lamp_kit' else 1)
            return self.approach(tid)
        if m.kind=='freight':
            mv=e.movers[s['mover']]
            if mv['moving']: self.explanation='Ride the freight lift with the original core'; return neutral()
            if mv['cargo']!='core':
                a=self.transport_cargo('core',mv['pos']+(-3,0,0),s['load_target'])
                if a is not None: return a
                if not e.on_deck(mv): return self.go(mv['pos']+(.5,.01,.5))
                return self.command('use',s['load_target'])
        # Base behavior uses the overriden movement and craft methods. Preserve
        # the larger per-milestone route budget instead of its compact watchdog.
        count=self.count; self.count=0
        action=super().__call__(obs); self.count=count
        return action
