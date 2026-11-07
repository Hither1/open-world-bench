"""Privileged reference controller. Emits native actions; never writes world state."""
from __future__ import annotations

import numpy as np

from .character import MoveState
from .elements import Element


class ReferencePolicy:
    def __init__(self, env):
        self.env=env
        self.explanation=""
        self._guard_id=None
        self._guard_hp=None
        self._guard_stall=0
        self._guard_route=[]
        self._guard_anchored=False

    @staticmethod
    def neutral():
        return {"move":np.zeros(2,np.float32),"look":np.zeros(2,np.float32),
                "buttons":np.zeros(7,np.int8),"swap":0}

    def move_to(self, action, target, stop=1.1):
        env=self.env
        d=np.asarray(target)-env.char.pos
        dist=float(np.linalg.norm(d[[0,2]]))
        if dist<stop:
            return True
        desired=float(np.arctan2(d[0],-d[2]))
        delta=(desired-env.cam.yaw+np.pi)%(2*np.pi)-np.pi
        action["look"][0]=np.clip(delta/(env.cfg.camera.yaw_rate*.2),-1,1)
        action["look"][1]=np.clip((.34-env.cam.pitch)/.34,-1,1)
        direction=d[[0,2]]/max(dist,1e-6)
        cy,sy=np.cos(env.cam.yaw),np.sin(env.cam.yaw)
        magnitude=min(1.0,max(.4,(dist-stop)/1.5))
        action["move"][:]=(direction[0]*cy+direction[1]*sy,
                           direction[0]*sy-direction[1]*cy)
        action["move"]*=magnitude
        np.clip(action["move"],-1.0,1.0,out=action["move"])
        if env.char.state!=MoveState.GLIDING:
            action["buttons"][4]=1
        self.explanation="Walk to "+(env.course.target().label if env.course.target() else "destination")
        return False

    def cast(self, action, element):
        party=self.env.party
        idx=next(i for i,m in enumerate(party.members) if m.spec.element==element)
        member=party.members[idx]
        if idx!=party.active_index:
            if party.swap_timer>0:
                self.explanation="Wait for party-swap cooldown"
                return
            action["swap"]=idx+1
        if member.skill_cd>1e-5:
            self.explanation=f"Wait for {member.spec.name}'s skill cooldown"
            return
        action["buttons"][5]=1
        self.explanation=f"Cast {member.spec.name}'s {element.name.lower()} skill"

    def face(self, action, target):
        """Native attacks follow the character's facing, not camera orientation."""
        env=self.env
        d=np.asarray(target)-env.char.pos
        dist=float(np.linalg.norm(d[[0,2]]))
        desired=float(np.arctan2(d[0],-d[2]))
        error=(desired-env.char.yaw+np.pi)%(2*np.pi)-np.pi
        if dist>.4 and abs(error)>.25:
            cy,sy=np.cos(env.cam.yaw),np.sin(env.cam.yaw)
            dx,dz=d[[0,2]]/dist
            action["move"][:]=np.array([dx*cy+dz*sy,dx*sy-dz*cy])*.4
            self.explanation="Turn toward the enemy before attacking"
            return False
        return True

    def __call__(self, observation=None):
        env=self.env
        op=env.course.op
        a=self.neutral()
        if op is None:
            self.explanation="All milestones verified"
            return a
        s=op.state
        target=env.course.target()
        if op.kind=="sentinel":
            if self._guard_id!=op.id:
                self._guard_id=op.id
                self._guard_hp=target.hp
                self._guard_stall=0
                self._guard_route=[]
                self._guard_anchored=False
            if s["phase"]=="vulnerable":
                self._guard_stall=self._guard_stall+1 if target.hp==self._guard_hp else 0
                self._guard_hp=target.hp
                if self._guard_stall>=16 and not self._guard_route:
                    # Recover from landing inside the enemy's attack cone origin:
                    # walk out, then approach a stable stance facing its core.
                    self._guard_route=[target.pos+np.array([0,0,4.0]),target.pos+np.array([0,0,2.0])]
                    self._guard_stall=0
                    self._guard_anchored=False
                if self._guard_route:
                    if self.move_to(a,self._guard_route[0],stop=.18):
                        self._guard_route.pop(0)
                        self._guard_stall=0
                        self._guard_anchored=not self._guard_route
                    self.explanation="Reposition, then approach a stable facing stance"
                    return a
                if self._guard_anchored:
                    if not self.face(a,target.pos):
                        return a
                    a["buttons"][3]=1
                    self.explanation="Physical attacks from a stable stance facing the exposed core"
                    return a
        if op.kind=="crossing":
            if s["launched"]:
                self.move_to(a,target.pos,stop=.6)
                a["buttons"][4]=0
                a["buttons"][0]=1
                self.explanation="Hold jump to glide; steer toward the landing shelf"
                return a
            if not self.move_to(a,target.pos,stop=.6):
                if env.char.state==MoveState.SWIMMING:
                    a["buttons"][0]=0
                return a
            if not env.char.grounded:
                self.explanation="Settle onto the vent platform"
                return a
            if env.char.stamina<82:
                self.explanation="Rest on the shelf to recover stamina"
                return a
            if not s["charged"]:
                self.cast(a,env.course.vent_element)
                return a
            if 1.15<=env.world.elapsed%4<=3.15:
                a["buttons"][0]=int(not env.char._jump_was_down)
                self.explanation="Bright pulse: jump into the charged updraft"
            else:
                self.explanation="Wait for the safe bright pulse"
            return a
        if op.kind=="lift" and s["riding"]:
            self.explanation="Ride the moving service platform"
            return a
        stop=2.35 if op.kind=="fight" else 1.5 if op.kind=="sentinel" else 1.0
        if not self.move_to(a,target.pos,stop):
            if op.kind=="dive" and env.char.state==MoveState.SWIMMING:
                a["buttons"][0]=1
            return a
        if op.kind=="dive":
            a["buttons"][0]=1
            if env.char.pos[1]<=-2.9:
                a["buttons"][2]=1
                self.explanation="Recover the coupling from the submerged chest"
            else:
                self.explanation="Hold jump to dive to the chest"
            return a
        if abs(float(env.char.pos[1]-target.pos[1]))>1.8:
            self.explanation="Wait to land within interaction range"
            return a
        if op.kind=="relay":
            role=next(k for k,v in s["roles"].items() if v==target.id)
            if role=="core":
                a["buttons"][2]=1
                self.explanation="Confirm the routed, fault-free relay at its core"
            else:
                self.cast(a,Element(int(env.course.relay_elements[role])))
                self.explanation+=f" at the {role}"
        elif op.kind=="sentinel":
            if s["phase"]=="shield":
                self.cast(a,Element(int(env.course.guard_elements[0])))
            elif s["phase"]=="windup":
                self.cast(a,Element(int(env.course.guard_elements[1])))
            elif s["phase"]=="vulnerable":
                a["buttons"][3]=1
                self.explanation="Physical attacks while the core is vulnerable"
            else:
                self.explanation="Wait for the sentinel's visible wind-up"
        elif op.kind=="fight":
            # Field skills provide a normal, robust solution for pursuing enemies.
            available=[m.spec.element for m in env.party.members if m.skill_cd<=1e-5]
            if Element.TIDE in available:
                self.cast(a,Element.TIDE)
            elif Element.GALE in available:
                self.cast(a,Element.GALE)
            else:
                self.face(a,target.pos)
                a["buttons"][3]=1
                self.explanation="Attack the camp sentry while elemental skills recharge"
        else:
            a["buttons"][2]=1
            self.explanation="Interact: "+op.title
        return a
