"""Executable Wildlands reference courses, using native movement and physics.

This is a compact authored implementation of the 12/24 milestone designs.
The privileged controller is separate. No action can set progress or teleport.
Mechanisms are local, recoverable state machines with physical loads/landings.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import json
import numpy as np
from gymnasium import spaces
from .wildlands import WildlandsEnv, COMMANDS as BASE_COMMANDS
from ..core.config import EnvConfig, WorldConfig, RenderConfig, SimConfig
from ..world.worldgen import generate
from owb.tasks.setups import run_setup
from owb.tasks.wildlands.course import Milestone, outcome as course_outcome, system_operational  # noqa: F401  (Milestone re-exported)
from owb.tasks.wildlands.setup import A, B_TASK, WorldHooks, author_world  # noqa: F401  (A, B_TASK re-exported)
from owb.tasks.wildlands.mechanisms import MECHANISMS
from owb.tasks.wildlands.mechanisms.base import Y, plain  # noqa: F401  (re-exported)
from owb.tasks.wildlands.mechanisms.commands import COURSE_COMMANDS, dispatch
from owb.tasks.wildlands.mechanisms.counterweight import valid_load
from owb.tasks.wildlands.mechanisms.thermal import install_updraft

COMMANDS = BASE_COMMANDS + COURSE_COMMANDS


class WildlandsReferenceEnv(WildlandsEnv):
    commands = COMMANDS
    presentation_kind = 'wildlands_reference'

    def __init__(self, task='a', seed=7, *, render_enabled=False):
        if task not in ('a','b'): raise ValueError('task must be a or b')
        self.task = task
        cfg = EnvConfig(seed=seed, starter_camp=False, max_steps=14000,
            obs_rgb=render_enabled, obs_text=True, reward_mode='sparse',
            world_cfg=WorldConfig(size_x=160,size_y=64,size_z=160,sea_level=12,
                relief=12,mountain_relief=10,landmarks=False),
            sim_cfg=SimConfig(weather_enabled=False,wind_enabled=False,start_time=.37),
            render_cfg=RenderConfig(width=320,height=180,third_person=True,
                cam_distance=7,cam_pitch_offset=-.16,max_dist=150))
        super().__init__(cfg)
        self.action_space['target'] = spaces.Discrete(256)
        self.action_space['vector'] = spaces.Box(-1,1,(3,),np.float32)

    def _generate_world(self):
        cfg = deepcopy(self.cfg.world_cfg); cfg.seed = self._seed
        result = generate(cfg)
        return author_world(result,self.task)

    def _target(self,m,role,pos):
        return self._hooks.target(m,role,pos)

    def _reset_mechanics(self):
        self.targets={}; self.completed=[]; self.satchel=Counter(); self.exposed={}
        self.success=False; self.storm=False; self.drained=False; self.road_seen=set()
        self.last_text=''; self.held=None; self.online=set(); self._pending=[]
        rng=np.random.default_rng(self._seed+8191)
        self._hooks=WorldHooks(self)
        self.setup=run_setup('wildlands.reference_course',self._hooks,rng,{'course':self.task})
        rules=self.setup.rules
        self.calibration=rules['calibration']; self.authorized=rules['authorized']
        self.basin_map=rules['basin_map']; self.family_symbols=rules['family_symbols']
        self.milestones=self.setup.milestones
        self.by_id={m.id:m for m in self.milestones}
        self.agent.pitch=-.08
        self.agent.inventory.add('rune_magnet',1)
        install_updraft(self)

    @property
    def current(self):
        done={r['id'] for r in self.completed}
        return next((m for m in self.milestones if m.id not in done),None)

    def eligible(self,m):
        return set(m.prereqs)<={r['id'] for r in self.completed}

    def _fail(self,reason,**kw): return [dict(type='action_failed',reason=reason,**kw)]

    def _reachable(self,pos,radius=3.2):
        delta=np.asarray(pos)-self.agent.eye; distance=float(np.linalg.norm(delta))
        if distance>radius: return False
        hit=self.world.raycast(self.agent.eye,delta,max(0,distance-.35))
        return distance<=radius and not hit.hit

    def _dispatch(self,command,ingredients,action):
        self.last_text = ''
        events=dispatch(self,command,action)
        if events is not None: return events
        return super()._dispatch(command,ingredients,action)

    def clue(self,m):
        return MECHANISMS[m.kind].clue(self,m)

    def _valid_load(self,m):
        return valid_load(self,m)

    def _task_outcome(self,events):
        return course_outcome(self,events)

    def system_operational(self):
        """Final success requires working equipment, not just historical flags."""
        return system_operational(self)

    def _observation(self):
        obs=super()._observation()
        if self.current:
            visible=[f'{tid}:{obj["symbol"]}' for tid,obj in self.targets.items() if self._reachable(obj['pos'])]
            obs['text']=(self.current.title+'. Nearby controls '+', '.join(visible)+'. '+self.last_text+
                         '. Repair satchel: '+str(dict(self.satchel)))[:1024]
        return obs

    def _info(self,events):
        out=super()._info(events)
        out.update(success=self.success,milestones=deepcopy(self.completed),task_progress=len(self.completed)/len(self.milestones))
        return out

    def state_digest(self):
        data=dict(task=self.task,seed=self._seed,steps=self.steps,elapsed=self.elapsed,
            agent=vars(self.agent).copy(),milestones=[m.state for m in self.milestones],
            completed=self.completed,satchel=dict(self.satchel),exposed=self.exposed,
            bodies={k:vars(b) for k,b in self.bodies.bodies.items()},held=self.held,
            terrain=hashlib.sha256(self.world.vox.tobytes()).hexdigest(),
            rng=self.np_random.bit_generator.state,weather=vars(self.elements.climate),
            npcs=[vars(n) for n in self.npcs.alive],online=self.online,drained=self.drained,road=self.road_seen)
        data['agent']['inventory']=self.agent.inventory.to_dict()
        data['agent']['effects']=[vars(e) for e in self.agent.effects]
        return hashlib.sha256(json.dumps(plain(data),sort_keys=True,default=str).encode()).hexdigest()

    def presentation_meshes(self,batch):
        """Actual control, cable, gauge and flow state, exported into Cycles."""
        for m in self.milestones:
            x,z=m.site; s=m.state
            if m.kind=='balance':
                for a,b in [((x+3,Y+5,z+2),(x+3,Y+.6,z+2)),((x+3,Y+5,z+2),(x+7,Y+5,z)),((x+7,Y+5,z),(x+7,Y+1,z))]:
                    batch.tube(a,b,.07,(.37,.29,.14))
                batch.box((x+3.5,Y+.05,z+2.6),(2,.1,4),(.44,.39,.25))
            if m.kind in ('flow','cool','prepare'):
                for k in range(3):
                    color=(.12,.65,.78) if s['outlets'][k] else (.30,.34,.36)
                    batch.tube((x-3,Y+.6,z-3),(x-2+2*k,Y+.6,z-1),.18,color)
                    batch.tube((x-2+2*k,Y+.6,z-1),(x-2+2*k,Y+.6,z+2),.18,color)
            if m.kind=='thermal' and s['heat']:
                color=(.15,.75,.72) if s['safe'] else (.95,.25,.1)
                for k in range(5):
                    yy=Y+.7+(k+(self.elapsed%1))*1.1
                    batch.ellipsoid((x+3,yy,z+.5),(.13,.22,.13),color)
        for obj in self.targets.values():
            if obj['role'].startswith('weight'): continue
            m=self.by_id[obj['milestone']]; pos=obj['pos']; role=obj['role']
            color=(.19,.49,.53)
            if m.kind in ('flow','cool','prepare') and (role=='inlet' or role.startswith('outlet')):
                on=m.state['inlet'] if role=='inlet' else m.state['outlets'][int(role[-1])]
                color=(.2,.67,.38) if on else (.65,.27,.16)
            batch.box(pos-(0,.35,0),(.38,.7,.38),(.39,.31,.21))
            batch.ellipsoid(pos,(.26,.26,.12),color)

    def presentation_camera(self):
        m=self.by_id.get(getattr(self,'reference_view_id','')) or self.current
        focus=self.agent.pos.astype(float)+(0,1,0)
        if m and np.linalg.norm(focus[[0,2]]-np.asarray(m.site))<16:
            focus=.45*focus+.55*np.array((m.site[0]+2,Y+1,m.site[1]))
        return focus+(-11,9,-13),focus
