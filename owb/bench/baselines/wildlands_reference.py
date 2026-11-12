"""Privileged action-only reference controller; not a learned-memory agent."""
from __future__ import annotations
import heapq
import math
import numpy as np
from ...core import blocks as B
from ...env.wildlands_reference import COMMANDS, Y


def neutral():
    return dict(move=np.zeros(2,np.float32),look=np.zeros(2,np.float32),buttons=np.zeros(5,np.int8),
                command=0,slot=0,ingredients=np.zeros(5,np.int64),target=0,vector=np.zeros(3,np.float32),
                pointer=np.zeros(2,np.float32),click=0)


class WildlandsReferencePolicy:
    def __init__(self,env):
        self.env=env; self.explanation='Begin reference course'
        self.path=[]; self.path_key=None; self.demonstrated=set(); self.carry_stage=0
        self.last_mid=None; self.chapter_steps=0

    def command(self,name,target=0,**kw):
        a=neutral(); a['command']=COMMANDS.index(name); a['target']=target; a.update(kw)
        self.explanation=f'{name}: '+(self.env.targets[target]['role'] if target else 'native survival action')
        return a

    def steer(self,pos,*,glide=False,jump=False):
        e=self.env; delta=np.asarray(pos)[[0,2]]-e.agent.pos[[0,2]]
        distance=float(np.linalg.norm(delta)); yaw=math.atan2(delta[1],delta[0]) if distance>.1 else e.agent.yaw
        error=(yaw-e.agent.yaw+math.pi)%(2*math.pi)-math.pi
        turn=float(np.clip(error/.36,-1,1)); heading=e.agent.yaw+turn*.18
        wish=delta/max(distance,1e-6)*min(1.,distance/.65)
        a=neutral(); a['look'][0]=turn
        a['move'][:]=(np.dot(wish,(math.cos(heading),math.sin(heading))),np.dot(wish,(-math.sin(heading),math.cos(heading))))
        a['buttons'][0]=jump; a['buttons'][3]=glide
        self.explanation='Glide on the established updraft' if glide else 'Walk the physical access route'
        return a

    def walkable(self,x,z):
        w=self.env.world
        return (1<=x<w.size_x-1 and 1<=z<w.size_z-1 and B.SOLID[w.get(x,28,z)]
            and not B.SOLID[w.get(x,29,z)] and not B.SOLID[w.get(x,30,z)] and w.get(x,29,z)!=B.WATER)

    def navigate(self,pos):
        e=self.env; p=np.asarray(pos,float)
        if e.agent.pos[1]>Y+.5 and p[1]<Y+2:
            # Walk off the service deck; gravity performs the descent.
            candidates=[m for m in e.milestones if m.kind=='thermal' and abs(e.agent.pos[0]-(m.site[0]+10))<8 and abs(e.agent.pos[2]-m.site[1])<5]
            if candidates:
                x,z=candidates[0].site
                return self.steer((x+6,Y,z+.5))
        if p[1]>Y+2:
            return self.steer(p)
        goal=(int(p[0]),int(p[2])); start=tuple(e.agent.pos[[0,2]].astype(int))
        key=(goal,e.storm,e.drained)
        if key!=self.path_key or not self.path:
            self.path_key=key
            if not self.walkable(*goal):
                near=[(goal[0]+dx,goal[1]+dz) for dx in range(-2,3) for dz in range(-2,3) if self.walkable(goal[0]+dx,goal[1]+dz)]
                if not near: raise RuntimeError(f'No accessible stance around {goal}')
                goal=min(near,key=lambda q:(q[0]+.5-p[0])**2+(q[1]+.5-p[2])**2)
            heap=[(0,0,start)]; prev={}; cost={start:0}
            while heap:
                _,g,n=heapq.heappop(heap)
                if n==goal: break
                if g!=cost[n]: continue
                for dx,dz in ((1,0),(-1,0),(0,1),(0,-1)):
                    q=(n[0]+dx,n[1]+dz)
                    if self.walkable(*q) and g+1<cost.get(q,1e9):
                        cost[q]=g+1; prev[q]=n
                        heapq.heappush(heap,(g+1+abs(q[0]-goal[0])+abs(q[1]-goal[1]),g+1,q))
            if goal not in cost: raise RuntimeError(f'No ground route {start} -> {goal}, {e.current.id}')
            nodes=[goal]
            while nodes[-1]!=start: nodes.append(prev[nodes[-1]])
            self.path=list(reversed(nodes))[1:]
        while self.path and np.linalg.norm(e.agent.pos[[0,2]]-(np.array(self.path[0])+.5))<.3:
            self.path.pop(0)
        target=(*self.path[0],) if self.path else goal
        return self.steer((target[0]+.5,Y,target[1]+.5))

    def at_target(self,tid,name='operate'):
        obj=self.env.targets[tid]; p=obj['pos']
        if self.env._reachable(p):
            self.path=[]
            return self.command(name,tid)
        return self.navigate(p-(1.5,1,0))

    def __call__(self,obs):
        e=self.env; m=e.current
        if m is None: return neutral()
        if m.id!=self.last_mid:
            self.last_mid=m.id; self.chapter_steps=0; self.path=[]; self.path_key=None
        self.chapter_steps+=1
        if self.chapter_steps>1500: raise RuntimeError(f'Reference stuck at {m.id}: pos={e.agent.pos}, state={m.state}')
        s=m.state; x,z=m.site
        if m.kind=='clue': return self.at_target(m.targets['inspect'],'inspect')
        if m.kind=='exchange': return self.at_target(m.targets['interact'],'interact')
        if m.kind=='balance':
            if m.id in ('A02','B02') and not s['observed']:
                return self.at_target(m.targets['inspect'],'inspect')
            if m.id in ('A02','B02') and m.id not in self.demonstrated:
                if e._reachable(e.targets[m.targets['release']]['pos']):
                    self.demonstrated.add(m.id)
                return self.at_target(m.targets['release'])
            if not s['braked']: return self.at_target(m.targets['brake'])
            stance=np.array((x+.5,Y,z+.5))
            if e.held is None:
                if not e._valid_load(m):
                    for k,bid in enumerate(s['bodies']):
                        body=e.bodies.get(bid); target=np.asarray(s['slots'][k])
                        if np.linalg.norm(body.pos-target)>.12:
                            if np.linalg.norm(e.agent.pos[[0,2]]-stance[[0,2]])>.35: return self.navigate(stance)
                            self.carry_stage=0
                            return self.command('grip',m.targets[f'weight{k}'])
                    self.explanation='Wait for dropped weights to settle on their supports'; return neutral()
                if not s['released']: return self.at_target(m.targets['release'])
                if m.consumes and not s['installed']: return self.at_target(m.targets['install'],'install')
                self.explanation='Verify supported load and stable mechanism movement'; return neutral()
            body=e.bodies.get(e.held); k=s['bodies'].index(e.held); target=np.asarray(s['slots'][k]).copy()
            if self.carry_stage==0:
                target=body.pos.copy(); target[1]=Y+2
                if abs(body.pos[1]-target[1])<.04: self.carry_stage=1
            if self.carry_stage==1:
                target=np.asarray(s['slots'][k]).copy()
                target[1]=Y+2
                if np.linalg.norm(body.pos-target)<.04: self.carry_stage=2
            if self.carry_stage==2: target=np.asarray(s['slots'][k])
            if self.carry_stage==2 and np.linalg.norm(body.pos-target)<.02: return self.command('drop')
            return self.command('drag',vector=np.clip((target-body.pos)/.4,-1,1).astype(np.float32))
        if m.kind=='thermal':
            if m.id in ('A03','B04') and not s['observed']: return self.at_target(m.targets['inspect'],'inspect')
            if not s['vent']: return self.at_target(m.targets['vent'])
            if not s['heat']: return self.at_target(m.targets['heat'])
            if not s['landed']:
                if e.agent.on_ground and e.agent.pos[1]<Y+1:
                    launch=np.array((x+.5,Y,z+.5))
                    if np.linalg.norm((e.agent.pos-launch)[[0,2]])>.45: return self.navigate(launch)
                    return self.steer(s['landing'],jump=True,glide=True)
                landing=np.asarray(s['landing'])
                if e.agent.pos[0]>x+8.6 and e.agent.pos[1]>Y+3.6:
                    return self.steer(landing,glide=False)
                return self.steer(landing,glide=True)
            if m.consumes and not s['installed']: return self.at_target(m.targets['install'],'install')
            if m.id=='B23':
                flow=e.by_id['B22']
                if not flow.state['inlet']: return self.at_target(flow.targets['inlet'])
            self.explanation='Verify service landing and stable station output'; return neutral()
        if m.kind in ('flow','cool','prepare'):
            if m.id in ('B06','B07','B16','B17') and not s['observed']: return self.at_target(m.targets['inspect'],'inspect')
            if m.consumes and not s['installed']: return self.at_target(m.targets['install'],'install')
            expected=s['expected']
            if s['outlets']!=[k==expected for k in range(3)]:
                if s['inlet']: return self.at_target(m.targets['inlet'])
                for k in range(3):
                    if s['outlets'][k]!=(k==expected): return self.at_target(m.targets[f'outlet{k}'])
            if m.kind!='prepare' and not s['inlet']: return self.at_target(m.targets['inlet'])
            self.explanation='Check safe plume temperature' if m.kind=='cool' else 'Verify stable authorized flow for five seconds'
            return neutral()
        if m.kind=='station':
            if not s['installed']: return self.at_target(m.targets['interact'],'interact')
            self.explanation='Verify stable wind-station output for five seconds'; return neutral()
        if m.kind=='calibrate':
            if not s['installed']: return self.at_target(m.targets['install'],'install')
            if s['selected']!=e.calibration: return self.at_target(m.targets[f'mark{e.calibration}'])
            if not s.get('engaged'): return self.at_target(m.targets['interact'],'interact')
            self.explanation='Verify calibrated flow and drain the road'; return neutral()
        if m.kind=='cook':
            stance=(x+.5,Y,z+.5)
            if np.linalg.norm(e.agent.pos[[0,2]]-np.array(stance)[[0,2]])>.35: return self.navigate(stance)
            inv=e.agent.inventory
            for item,effect in (('spicy_pepper','cold_resist'),('chillshroom','heat_resist')):
                if effect in s['cooked']: continue
                slots=[i for i,st in enumerate(inv.stacks) if st.item==item and not st.data.get('dish')]
                if not slots: return self.command('interact')
                ingredients=np.zeros(5,np.int64); ingredients[0]=slots[0]+1
                return self.command('cook',ingredients=ingredients)
            return self.command('interact')
        if m.kind=='report':
            if 'south' not in e.road_seen: return self.navigate((24.5,Y,50.5))
            if 'north' not in e.road_seen: return self.navigate((24.5,Y,39.5))
            return self.at_target(m.targets['interact'],'interact')
        raise RuntimeError(m.kind)
