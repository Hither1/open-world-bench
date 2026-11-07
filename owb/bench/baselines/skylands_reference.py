"""Privileged reference controller. All changes occur via environment actions."""
from __future__ import annotations
import heapq
import math
import numpy as np
from ...core import blocks as B
from ...env.skylands_reference import COMMANDS, LEVELS


def neutral():
    return dict(move=np.zeros(2,np.float32),look=np.zeros(2,np.float32),buttons=np.zeros(5,np.int8),
        command=0,slot=0,ingredients=np.zeros(5,np.int64),part=0,build_move=np.zeros(3,np.float32),target=0,option=0)


class SkylandsReferencePolicy:
    def __init__(self,env):
        self.env=env; self.explanation='Begin executed reference course'; self.last=None; self.count=0
        self.path=[]; self.key=None; self._tow_phase={}

    def command(self,name,target,option=0):
        a=neutral(); a.update(command=COMMANDS.index(name),target=int(target),option=int(option))
        self.explanation=f'{name}: '+self.env.targets[target]['role']
        return a

    def steer(self,pos):
        e=self.env; p=np.asarray(pos); delta=p[[0,2]]-e.agent.pos[[0,2]]; dist=float(np.linalg.norm(delta))
        yaw=math.atan2(delta[1],delta[0]) if dist>.05 else e.agent.yaw
        error=(yaw-e.agent.yaw+math.pi)%(2*math.pi)-math.pi; turn=float(np.clip(error/.36,-1,1))
        heading=e.agent.yaw+turn*.18; wish=delta/max(dist,1e-8)*min(1.,dist/.55)
        a=neutral(); a['look'][0]=turn
        a['move'][:]=(np.dot(wish,(math.cos(heading),math.sin(heading))),np.dot(wish,(-math.sin(heading),math.cos(heading))))
        self.explanation='Walk the physical route'
        return a

    def near(self,p,r=1.1): return np.linalg.norm(self.env.agent.pos-np.asarray(p))<r

    def walkable(self,x,z,y):
        w=self.env.world
        return (1<=x<w.size_x-1 and 1<=z<w.size_z-1 and B.SOLID[w.get(x,y-1,z)]
                and not B.SOLID[w.get(x,y,z)] and not B.SOLID[w.get(x,y+1,z)])

    def walk(self,pos):
        e=self.env; p=np.asarray(pos,float); y=int(round(p[1]))
        # Step off a six-metre service gallery; ordinary gravity does the descent.
        if e.agent.pos[1]>y+2:
            candidates=[m for m in e.milestones if 'upper' in m.state and np.linalg.norm(e.agent.pos[[0,2]]-m.pos[[0,2]])<5]
            if candidates:
                m=candidates[0]; return self.steer((m.x-4.5,m.y,m.z+.5))
        start=tuple(np.floor(e.agent.pos[[0,2]]).astype(int)); goal=tuple(np.floor(p[[0,2]]).astype(int))
        if start==goal or np.linalg.norm(e.agent.pos[[0,2]]-p[[0,2]])<1: return self.steer(p)
        key=(goal,y,e.closed)
        if self.key!=key or not self.path or self.path[0] not in (start,) and np.linalg.norm(np.asarray(self.path[0])-np.asarray(start))>2:
            queue=[(0,start)]; previous={start:None}; cost={start:0}; found=None
            while queue:
                _,q=heapq.heappop(queue)
                if q==goal: found=q; break
                for n in ((q[0]-1,q[1]),(q[0]+1,q[1]),(q[0],q[1]-1),(q[0],q[1]+1)):
                    if not self.walkable(*n,y): continue
                    c=cost[q]+1
                    if c<cost.get(n,1e9):
                        cost[n]=c; previous[n]=q
                        heapq.heappush(queue,(c+abs(n[0]-goal[0])+abs(n[1]-goal[1]),n))
            if found is None:
                if getattr(self,'strict_paths',False): raise RuntimeError(f'No walkable path {start} -> {goal} at {y}, agent={e.agent.pos}')
                self.explanation=f'Approach local landing {goal}'
                return self.steer(p)
            path=[]
            while found!=start: path.append(found); found=previous[found]
            self.path=list(reversed(path)); self.key=key
        while len(self.path)>1 and np.linalg.norm(e.agent.pos[[0,2]]-(np.asarray(self.path[0])+.5))<.65:
            self.path.pop(0)
        q=self.path[0]
        return self.steer((q[0]+.5,p[1],q[1]+.5))

    def go(self,pos):
        e=self.env; p=np.asarray(pos,float)
        layer=min(range(3),key=lambda i:abs(p[1]-e.levels[i])); current=min(range(3),key=lambda i:abs(e.agent.pos[1]-e.levels[i]))
        mv=e.movers['transit']
        if mv['moving'] and e.on_deck(mv): self.explanation='Ride the maintenance lift between layers'; return neutral()
        if current!=layer:
            y=e.levels[current]
            if abs(e.agent.pos[1]-y)>2: return self.walk((12,y+.01,8.5))
            call=np.array((12.5,y+.01,8.5))
            if mv['pos'][1]!=y:
                if not self.near(call,1): return self.walk(call)
                if not mv['moving']: return self.command('use',e.lift_calls[current])
                self.explanation='Wait for the summoned maintenance lift'; return neutral()
            if not e.on_deck(mv): return self.walk(mv['pos']+(.5,.01,.5))
            return self.command('use',e.transit_target,layer)
        return self.walk(p)

    def approach(self,tid,command='use',option=0,reach=2.8):
        e=self.env; p=e.targets[tid]['pos']
        if e._reachable(p,reach): return self.command(command,tid,option)
        return self.go(p+(0,-1,.5))

    def tow(self,cid,goal,control):
        e=self.env; c=e.cargo[cid]; goal=np.asarray(goal,float)
        if e.hauling!=cid:
            if np.linalg.norm(e.agent.pos-c['pos'])>2.4: return self.go(c['pos']+(-1.5,.01,0))
            return self.command('haul',c['target'])
        if np.linalg.norm(c['pos']-goal)<2.8:
            return self.command('release',c['target'])
        return self.go(goal-e.tow_offset)

    def transport_cargo(self,cid,goal,control):
        e=self.env; c=e.cargo[cid]
        if np.linalg.norm(c['pos']-goal)<3:
            if e.hauling: return self.command('release',c['target'])
            return None
        return self.tow(cid,goal,control)

    def raft(self,m,rid,*,stage=False,test_only=False):
        e=self.env; r=e.rafts[rid]; target=r['target']
        # Build from individual local parts; no blueprint or completion action.
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
                goal=np.array((r['origin']-2,r['pos'][1],r['pos'][2]))
                action=self.transport_cargo(cid,goal,target)
                if action is not None: return action
            return self.approach(target,'load',reach=4.8)
        if not e.on_deck(r): return self.go(r['pos']+(.5,.01,.5))
        if stage:
            if r['travel']>=6 and r['pos'][0]<=r['origin']+.1:
                return self.command('use',target)
            if r['pos'][0]>=r['origin']+4 or r['throttle']==-1: return self.command('drive',target,1)
        if test_only and r['travel']>=3: return self.command('drive',target,2)
        if r['pos'][0]>=r['end']-.1: return self.command('unload',target,e.destination if m.kind=='delivery' else 0)
        return self.command('drive',target,0)

    def ascent(self,m):
        e=self.env; s=m.state
        if not s['landed']:
            p=np.array(s['launch'])
            if not self.near(p,.6): return self.go(p)
            a=neutral(); a['command']=COMMANDS.index('ascend'); self.explanation='Ascend through the inspected ceiling into the service gallery'; return a
        return self.approach(s['control'])

    def reverse(self,m):
        e=self.env; mv=e.movers[m.state['mover']]; target=m.state['platform_target']
        if mv.get('paused'):
            return self.approach(m.state['control'])
        if mv['reversed_distance']>=8 and mv['pos'][0]>=mv['end']-.1:
            return self.command('cancel',target)
        if mv['reverse']: self.explanation='Ride the platform along its recorded reverse trajectory'; return neutral()
        if e.on_deck(mv):
            history=[p[0] for p in mv['history']]
            if history and max(history)-mv['pos'][0]>=8: return self.command('reverse',target)
        start=np.array((mv['start']-2,m.y+.01,m.z+.5))
        if not self.near(start,2.8) and not e.on_deck(mv): return self.go(start)
        history=[p[0] for p in mv['history']]
        if mv['pos'][0]==mv['start'] and mv['phase']=='waiting' and history and max(history)>=mv['end']:
            return self.steer(mv['pos']+(.5,.01,.5))
        self.explanation='Observe a fresh inbound platform cycle before boarding'; return neutral()

    def __call__(self,obs):
        e=self.env; m=e.current
        if not m: return neutral()
        if m.id!=self.last: self.last=m.id; self.count=0; self.path=[]; self.key=None
        self.count+=1
        if self.count>2500: raise RuntimeError(f'Stalled at {m.id}: pos={e.agent.pos}, state={m.state}, hauling={e.hauling}')
        s=m.state; tid=s['control']
        if not s['observed'] and m.kind in ('clue','update'):
            return self.approach(tid,'inspect')
        if m.kind in ('raft','stage','return'): return self.raft(m,s['raft'],stage=m.kind=='stage')
        if m.kind in ('ascend','ascend_station'):
            if s['installed']: self.explanation='Verify five seconds of stable station output'; return neutral()
            return self.ascent(m)
        if m.kind in ('dock','extract','take'): return self.approach(tid,'take' if m.kind=='take' else 'use')
        if m.kind in ('freight','hoist'):
            mv=e.movers[s['mover']]; c=e.cargo['core']
            if mv['moving']: self.explanation='Freight lift carries the original core upward'; return neutral()
            if mv['cargo']!='core':
                goal=mv['pos']+(-3,0,0)
                a=self.transport_cargo('core',goal,s['load_target'])
                if a is not None: return a
                return self.approach(s['load_target'])
            if m.kind=='hoist': return self.ascent(m)
            return neutral()
        if m.kind=='calibrate':
            if s['installed']: self.explanation='Verify correct navigation output for five seconds'; return neutral()
            return self.approach(tid,'use',e.calibration)
        if m.kind=='cache':
            target=next(k for k,o in e.targets.items() if o['role']=='cache' and o['option']==e._seed%3)
            return self.approach(target,'cache')
        if m.kind in ('outpost','retrieve'):
            item='lamp_kit' if m.kind=='outpost' else 'lock_coupling'
            if not e.satchel[item]:
                target=next(k for k,o in e.targets.items() if o['role']=='cache' and o['option']==e.cache_choice)
                return self.approach(target,'retrieve',0 if item=='lamp_kit' else 1)
            return self.approach(tid)
        if m.kind=='station':
            if s['installed']: self.explanation='Verify five seconds of stable station output'; return neutral()
            cid='power_module' if m.id=='B15' else 'marsh_coupling'
            a=self.transport_cargo(cid,m.pos,tid)
            if a is not None: return a
            return self.approach(tid)
        if m.kind=='reverse': return self.reverse(m)
        if m.kind=='compose':
            if not s['installed']: return self.ascent(m)
            return self.raft(m,m.id,test_only=True)
        if m.kind=='delivery': return self.raft(m,'B23')
        return self.approach(tid,'inspect')
