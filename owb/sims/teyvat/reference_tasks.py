"""Compact, executable reference courses for the two Teyvat task designs.

This is a demonstration prototype, not the long-horizon evaluation suite.
It reuses native movement, party cooldowns, combat, terrain, and rendering.
Mission travel is shortened; a privileged reference controller is kept separate
(owb/agents/oracle/teyvat_reference.py).

The course itself lives in the task layer: ``Course`` (runtime and completion ledger) in
owb/tasks/teyvat/course.py, its state machines in owb/tasks/teyvat/mechanisms/. This
module keeps the simulator-side hooks (``CourseWorld``, ``CourseCombat``) and the env.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np

from .camera import CameraConfig, OrbitCamera
from .character import CharacterController
from .elements import Element
from .env import EnvConfig, OpenWorldEnv
from .party import Party
from .quests import ObjKind, Objective, Quest, QuestEngine, QuestGraph, Stage
from .render import RenderSettings
from .scatter import PropField
from .terrain import Terrain, TerrainConfig
from .world import CombatSystem, World, WorldConfig

from owb.tasks.teyvat.course import Course, Milestone, Operation  # noqa: F401  (re-exported)
from owb.tasks.teyvat.setup import author_plateau, settle_entities


class CourseCombat(CombatSystem):
    def cast(self,char,controller,*,burst=False):
        events=super().cast(char,controller,burst=burst)
        for ev in events:
            if ev.type in ("skill","burst"):
                self.world.course.apply_element(char,Element[ev.get("element").upper()])
        return events

    def _hit(self,enemy,char,damage,element,charge=1.0):
        result=self.world.course.sentinel_hit(enemy,damage,element)
        if result is not None:
            return result
        return super()._hit(enemy,char,damage,element,charge)


class CourseWorld(World):
    def interact(self,char):
        self.course.interact(char)
        return []

    def _update_enemy(self,e,char,dt):
        op=self.course.op
        if not op or e.id not in op.entities:
            return []
        # Sentinels hold an authored arena position; ordinary camp enemies pursue.
        if self.course.mechanism(op).holds_position:
            return []
        return super()._update_enemy(e,char,dt)

    def update(self,char,dt,char_cfg):
        events=super().update(char,dt,char_cfg)
        self.course.tick(char,dt)
        events.extend(self.course.events)
        self.course.events.clear()
        return events


class ReferenceTaskEnv(OpenWorldEnv):
    """Native-action demonstration env with compact authored course geometry."""

    def __init__(self,task="a",seed=7,*,render_enabled=True,width=384,height=216):
        self.task=task
        self.render_enabled=render_enabled
        cfg=EnvConfig(seed=seed,text_obs=True,backend="soft",action_repeat=6,max_episode_steps=30000,
                      world=WorldConfig(terrain=TerrainConfig(size=640,resolution=321,max_height=65),
                                        prop_density=0,day_length=100000,start_time_of_day=.43),
                      camera=CameraConfig(distance=9,far=160),
                      render=RenderSettings(width=width,height=height,march_steps=48,refine_steps=4,
                                            shadow_steps=0,prop_draw_distance=65,max_view_distance=160,
                                            ground_cover=False,clouds=False))
        cfg.reward.shaping=0
        cfg.reward.discovery=0
        super().__init__(cfg)
        from gymnasium import spaces
        import string
        self.observation_space["local_text"]=spaces.Text(max_length=1024,charset=string.printable)

    def _build(self,seed):
        self.rng=np.random.default_rng(seed)
        terrain=Terrain(replace(self.cfg.world.terrain,seed=seed))
        author_plateau(terrain)
        self.party=Party(self.cfg.party)
        empty=PropField(np.empty((0,3),np.float32),np.empty(0,np.float32),np.empty(0,np.float32),np.empty(0,np.int8))
        self.world=CourseWorld(replace(self.cfg.world,seed=seed),terrain=terrain,props=empty,party=self.party)
        self.course=Course(self.world,self.task,seed)
        self.world.course=self.course
        settle_entities(self.world,terrain)
        self.world.combat=CourseCombat(self.world,self.party)
        stages=[Stage([Objective(m.id,ObjKind.ACTIVATE,m.title,targets=(m.id,))]) for m in self.course.milestones]
        self.graph=QuestGraph([Quest(f"reference_{self.task}",f"Task {self.task.upper()}","Reference course",stages,reward=0)])
        self.quests=QuestEngine(self.graph)
        self.controller=CharacterController(terrain,empty,replace(self.cfg.character))
        self.camera_rig=OrbitCamera(terrain,empty,self.cfg.camera)
        target=self.course.target()
        self.char=self.controller.spawn(float(target.pos[0]),float(target.pos[2])+5)
        from .architecture import populate_architecture
        architecture=populate_architecture(self.world,self.char.pos)
        self.controller.architecture=self.camera_rig.architecture=architecture
        self.cam=self.camera_rig.reset(self.char)
        self.cam.pitch=.34
        self.steps=0
        self._prev_dist=None
        self._used_entities=set()
        self._episode_reward=0.0
        self._last_events=[]
        self._last_frame=None

    def _observe(self):
        if self.render_enabled:
            obs=super()._observe()
        else:
            obs={"rgb":np.zeros((self.cfg.render.height,self.cfg.render.width,3),np.uint8),
                 "proprio":self._proprio(),"task":self._task_vector(),"party":self.party.as_vector(),
                 "instruction":self.quests.instruction()}
        # Local dialogue/feedback is observable, but rule tables and target coordinates are private.
        obs["local_text"]=self.course.last_feedback
        return obs

    def _info(self,events=None):
        result=super()._info(events)
        result["inventory"]=dict(self.course.inventory)
        result["reference_course"]={"milestone_index":self.course.index,"operation_index":self.course.operation_index}
        return result

    def state_digest(self):
        state={"course":self.course.state(),"position":self.char.pos.tolist(),
               "party":self.party.summary(),"time":self.world.elapsed,"quest":self.quests.snapshot(),
               "entities":{e.id:{"position":e.pos.tolist(),"hp":e.hp,"alive":e.alive,"used":e.used}
                           for e in self.world.entities.values()}}
        return hashlib.sha256(json.dumps(state,sort_keys=True,default=lambda v:int(v),allow_nan=False).encode()).hexdigest()
