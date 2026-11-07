"""Action-only full-campaign oracle; routing uses privileged geometry reads."""
import numpy as np
from .skylands_reference import SkylandsReferencePolicy, neutral
from ...sims.voxel.env.skylands_reference import COMMANDS


class SkylandsCampaignPolicy(SkylandsReferencePolicy):
    def __init__(self,env):
        super().__init__(env)
        self.strict_paths=True
        self.descending=None; self.chosen_return='A06'+('_dock'+str(env._seed%3) if env._seed%3 else '')
        self.bindings={'lamp_kit':env._seed%3,'lock_coupling':3+(env._seed+1)%3}

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
        for q in r['route']:
            # Each route is x-first then z, or z-first then x. Completed legs
            # are identified geometrically, with no writes to simulator state.
            if np.linalg.norm(q-r['pos'])<.1: continue
            if q is not r['route'][-1]:
                nextq=r['route'][-1]
                if q[0]==nextq[0] and r['pos'][0]==q[0]: continue
                if q[2]==nextq[2] and r['pos'][2]==q[2]: continue
            delta=q-r['pos']; axis=0 if abs(delta[0])>.1 else 2
            v=np.zeros(3); v[axis]=np.sign(delta[axis]); return self.drive(r,v)
        raise RuntimeError('No unfinished route leg')

    def __call__(self,obs):
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
