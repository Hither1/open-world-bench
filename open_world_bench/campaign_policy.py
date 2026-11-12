"""Privileged feasibility controller: reads evaluator state, emits native actions.

Never use this as an observation-only model baseline or evidence of memory skill.
It does not write character state, terrain, inventory, mechanism state or facts.
"""
from __future__ import annotations
import numpy as np
from .elements import Element


class CampaignReferencePolicy:
    def __init__(self, env, *, reverse=False):
        self.env, self.reverse = env, reverse
        self.explanation = ''
        self.seen_murals = set()
        self.seen_archives = set()
        self._route_key = None
        self._route = []
        self._retreat = None

    @staticmethod
    def neutral():
        return dict(move=np.zeros(2,np.float32),look=np.zeros(2,np.float32),
                    buttons=np.zeros(7,np.int8),swap=0,pointer=np.zeros(2,np.float32),click=0)

    def order(self, count):
        return list(reversed(range(count))) if self.reverse else list(range(count))

    def _clear(self, start, end):
        env = self.env
        a,b = np.asarray(start)[[0,2]],np.asarray(end)[[0,2]]
        count = max(2,int(np.linalg.norm(b-a)/.8)+1)
        xz = a+np.linspace(0,1,count)[:,None]*(b-a)
        points = np.column_stack((xz[:,0],env.world.terrain.height_at(xz[:,0],xz[:,1])+.85,xz[:,1]))
        return not env.world.architecture.blocked(points,margin=.52).any()

    def _plan(self, target):
        start = self.env.char.pos
        if self._clear(start,target): return [np.asarray(target)]
        delta = np.asarray(target)-start
        flat = delta[[0,2]]
        perpendicular = np.array((-flat[1],flat[0]))/max(np.linalg.norm(flat),1e-6)
        # Conservative exterior detours around a pavilion post or sealed doorway.
        for width in (7,14,24,40):
            for sign in (-1,1):
                for fraction in (.2,.5,.8):
                    xz = start[[0,2]]+flat*fraction+perpendicular*width*sign
                    mid = self.env.world.ground_pos(*xz)
                    if self._clear(start,mid) and self._clear(mid,target): return [mid,np.asarray(target)]
        return [np.asarray(target)]

    def walk(self, action, target, stop=1.0):
        env = self.env
        key = tuple(np.round(target,2))
        if key != self._route_key:
            self._route_key = key; self._route = self._plan(target)
        while len(self._route)>1 and np.linalg.norm((self._route[0]-env.char.pos)[[0,2]])<1.2:
            self._route.pop(0)
        point = self._route[0] if self._route else np.asarray(target)
        delta = point-env.char.pos
        dist = float(np.linalg.norm(delta[[0,2]]))
        if len(self._route)<=1 and dist<stop: return True
        desired = float(np.arctan2(delta[0],-delta[2]))
        error = (desired-env.cam.yaw+np.pi)%(2*np.pi)-np.pi
        action['look'][0] = np.clip(error/(env.cfg.camera.yaw_rate*.2),-1,1)
        action['look'][1] = np.clip((.3-env.cam.pitch)/.34,-1,1)
        direction = delta[[0,2]]/max(dist,1e-6)
        cy,sy = np.cos(env.cam.yaw),np.sin(env.cam.yaw)
        magnitude = min(1.0,max(.12,(dist-stop)/1.5))
        action['move'][:] = (direction[0]*cy+direction[1]*sy,direction[0]*sy-direction[1]*cy)
        action['move'] *= magnitude
        np.clip(action['move'],-1,1,out=action['move'])
        action['buttons'][4] = 1
        self.explanation = 'Travel using native movement'
        return False

    def interact(self, action, eid):
        e = self.env.world.get(eid)
        if self.walk(action,e.pos) and abs(float(e.pos[1]-self.env.char.pos[1]))<2.4:
            action['buttons'][2] = 1
            self.explanation = 'Interact with '+e.label
            return True
        return False

    def face(self, action, point):
        env = self.env; delta = point-env.char.pos
        distance = np.linalg.norm(delta[[0,2]])
        desired = np.arctan2(delta[0],-delta[2])
        error = (desired-env.char.yaw+np.pi)%(2*np.pi)-np.pi
        if distance>.25 and abs(error)>.22:
            dx,dz = delta[[0,2]]/distance
            cy,sy = np.cos(env.cam.yaw),np.sin(env.cam.yaw)
            action['move'][:] = np.array((dx*cy+dz*sy,dx*sy-dz*cy))*.4
            return False
        return True

    def attack(self, action, entity):
        distance = np.linalg.norm((entity.pos-self.env.char.pos)[[0,2]])
        if distance<.6 and self._retreat is None:
            self._retreat = entity.pos+np.array((0,0,3.2))
        if self._retreat is not None:
            if self.walk(action,self._retreat,stop=.5): self._retreat=None
            return
        if self.walk(action,entity.pos,stop=2.25) and abs(float(entity.pos[1]-self.env.char.pos[1]))<2.0:
            if self.face(action,entity.pos): action['buttons'][3]=1

    def cast(self, action, element):
        party = self.env.party
        index = next(i for i,m in enumerate(party.members) if m.spec.element==element)
        if party.active_index != index:
            if party.swap_timer>0: return
            action['swap'] = index+1
        if party.members[index].skill_cd<=1e-5:
            action['buttons'][5] = 1
            self.explanation = 'Apply '+element.name.lower()+' through the native skill'

    def ritual(self, action, site):
        env,c = self.env,self.env.campaign
        if c.task=='cleansing' and site.kind!='boss' and site.id+'_ward' not in c.facts:
            self.interact(action,site.entities['ward']); return
        if c.task=='desert' and not site.installed and c.slate_energy<2:
            self.recharge(action); return
        if not site.installed:
            self.interact(action,site.entities['altar']); return
        key = (site.id,site.phase)
        if key not in self.seen_murals:
            if c.task=='desert' and c.slate_energy<1:
                self.recharge(action); return
            if self.interact(action,site.entities['mural']): self.seen_murals.add(key)
            return
        if not site.solved:
            element = Element(c.sequence(site)[site.progress])
            target = env.world.get(site.entities[element.name.lower()])
            if self.walk(action,target.pos) and abs(float(target.pos[1]-env.char.pos[1]))<2.4:
                self.cast(action,element)
            return
        enemy = env.world.get(site.entities['guard'])
        if enemy.alive:
            self.attack(action,enemy)
            self.explanation = 'Fight the exposed guardian with native attacks'
            return
        if site.kind!='boss': self.interact(action,site.entities['altar'])

    def recharge(self, action):
        env = self.env
        choices = [e for e in env.world.entities.values() if env.campaign.roles[e.id][1]=='recharge']
        target = min(choices,key=lambda e:np.linalg.norm(e.pos-env.char.pos))
        self.interact(action,target.id)

    def __call__(self, observation=None):
        env,c = self.env,self.env.campaign
        action = self.neutral()
        if 'final' in c.facts: return action
        if c.task=='summit':
            if c.cold>65:
                fires = [env.world.get(eid) for eid,(_,role) in c.roles.items() if role=='fire']
                target = min(fires,key=lambda e:np.linalg.norm(e.pos-env.char.pos))
                self.interact(action,target.id); return action
            targets = [f'lower_{i}' for i in self.order(3)]
            if all(key in c.facts for key in targets):
                if 'ascent' not in c.facts:
                    self.interact(action,'ascent:gate'); return action
                targets = [f'upper_{i}' for i in self.order(3)]
            pending = [key for key in targets if key not in c.facts]
            if not pending:
                self.interact(action,'final:spire'); return action
            site = c.sites[pending[0]]
            if 'guard' in site.entities and env.world.get(site.entities['guard']).alive:
                enemy = env.world.get(site.entities['guard'])
                self.attack(action,enemy)
                return action
            if 'ember' in site.entities and not site.solved:
                key = (site.id,site.phase)
                if key not in self.seen_murals:
                    if self.interact(action,site.entities['mural']): self.seen_murals.add(key)
                else:
                    element = Element(c.sequence(site)[site.progress])
                    target = env.world.get(site.entities[element.name.lower()])
                    if self.walk(action,target.pos) and abs(float(target.pos[1]-env.char.pos[1]))<2.4:
                        self.cast(action,element)
                return action
            if c.charge_until<=env.world.elapsed:
                self.interact(action,site.entities['quartz'])
            else:
                target = env.world.get(site.entities['ice'])
                self.attack(action,target)
            return action
        if c.task=='cleansing':
            if 'lens' not in c.facts:
                self.interact(action,'lens:tool'); return action
            pending = [f'shrine_{i}' for i in self.order(5) if f'shrine_{i}' not in c.facts]
            self.ritual(action,c.sites[pending[0] if pending else 'heart'])
            return action
        if 'slate' not in c.facts:
            self.interact(action,'slate:tool'); return action
        pending = [f'temple_{i}' for i in self.order(3) if f'temple_{i}' not in c.facts]
        if pending:
            site = c.sites[pending[0]]
            if site.id not in self.seen_archives:
                if self.interact(action,site.entities['archive']): self.seen_archives.add(site.id)
            else:
                self.ritual(action,site)
            return action
        if 'rank_two' not in c.facts:
            self.interact(action,'calibration:console'); return action
        pending = [f'temple_{i}' for i in self.order(3) if f'temple_{i}_record' not in c.facts]
        if pending:
            if c.slate_energy<2: self.recharge(action)
            else: self.interact(action,c.sites[pending[0]].entities['archive'])
            return action
        if 'atlas_key' not in c.facts:
            self.interact(action,'cipher:console'); return action
        if 'zenith' not in c.facts:
            self.ritual(action,c.sites['zenith']); return action
        self.interact(action,'final:atlas')
        return action
