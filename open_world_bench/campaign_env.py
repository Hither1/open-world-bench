"""Native-action, goal-first campaigns with regional terrain and architecture."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
import string

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .architecture import Architecture, Building, RADII, _distance_to_routes
from .camera import CameraConfig, OrbitCamera
from .campaign_tasks import Campaign, GOALS, SUITE_VERSION
from .character import CharacterController
from .elements import Element
from .env import EnvConfig, OpenWorldEnv, RewardConfig
from .events import Event
from .party import Party
from .quests import Quest, QuestGraph, QuestEngine, Stage, Objective, ObjKind
from .render import RenderSettings
from .scatter import PropField, scatter_props, PINE, ROCK, BOULDER, CRYSTAL
from .terrain import Terrain, TerrainConfig
from .world import World, WorldConfig, CombatSystem


def plain(value):
    if is_dataclass(value): return plain(asdict(value))
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    return value


class GateCollisions:
    """Visible seal boxes share exactly the same dimensions with collision."""
    def __init__(self, base, campaign):
        self.base, self.campaign = base, campaign
        self.buildings = base.buildings
        self._key = None
        self._closed = Architecture()

    def active(self):
        result = []
        for gate in self.campaign.gates:
            requires = gate['opens']
            requires = (requires,) if isinstance(requires, str) else requires
            if not set(requires) <= self.campaign.facts:
                result.append(gate)
        return result

    def obstacles(self):
        active = self.active()
        key = tuple(g['id'] for g in active)
        if key != self._key:
            boxes = Architecture()
            boxes.centers = np.array([np.asarray(g['position'])+(0,g['size'][1]/2,0) for g in active]).reshape(-1,3)
            boxes.halves = np.array([np.asarray(g['size'])/2 for g in active]).reshape(-1,3)
            boxes.rotations = np.repeat(np.eye(3)[None], len(active), axis=0)
            boxes.radii = np.linalg.norm(boxes.halves[:,[0,2]], axis=1)
            self._closed, self._key = boxes, key
        return self._closed

    def resolve(self, state, radius, height):
        self.base.resolve(state, radius, height)
        self.obstacles().resolve(state, radius, height)

    def blocked(self, points, margin=.35):
        return self.base.blocked(points, margin) | self.obstacles().blocked(points, margin)


class CampaignCombat(CombatSystem):
    def attack(self, char):
        events = super().attack(char)
        if any(e.type=='swing' for e in events):
            self.world.campaign.strike_ice(char)
        return events+self.world.drain_campaign_events()

    def cast(self, char, controller, *, burst=False):
        events = super().cast(char, controller, burst=burst)
        for event in events:
            if event.type in ('skill','burst'):
                self.world.campaign.apply_element(char, Element[event.get('element').upper()])
        return events+self.world.drain_campaign_events()

    def _damage(self, enemy, damage, element):
        return self.world.campaign.damage_guard(enemy, damage, element)


class CampaignWorld(World):
    def drain_campaign_events(self):
        events = self.campaign.events[:]
        self.campaign.events.clear()
        return events

    def interact(self, char):
        self.campaign.interact(char)
        return self.drain_campaign_events()

    def update(self, char, dt, char_cfg):
        events = super().update(char, dt, char_cfg)
        self.campaign.tick(char, dt)
        return events+self.drain_campaign_events()

    def _update_enemy(self, entity, char, dt):
        site = self.campaign.sites[self.campaign.roles[entity.id][0]]
        # Anchored guardians defend their arena. Native attacks, cooldowns, range,
        # elements and damage still apply; no target selection action is added.
        if self.campaign.available(site) and site.solved and np.linalg.norm(entity.pos-char.pos)<2.5:
            return super()._update_enemy(entity, char, dt)
        entity.attack_cd = max(0.0, entity.attack_cd-dt)
        return []


def _author_terrain(terrain, task, scale):
    axis = np.linspace(-terrain._half, terrain._half, terrain.cfg.resolution)
    x,z = np.meshgrid(axis/scale, axis/scale, indexing='ij')
    if task == 'summit':
        height = 12+48/(1+np.exp((z+45)/40))+2.2*np.sin(x/40)*np.cos(z/51)
        terrain.moisture[:] = .38
    elif task == 'cleansing':
        height = 18+4*np.sin(x/57)*np.cos(z/71)+3*np.cos((x+z)/85)
        terrain.moisture[:] = .7
    else:
        height = 18+3*np.sin(x/43)+2*np.cos(z/38)+4*np.exp(-((z+160)/55)**2)
        terrain.moisture[:] = .12
    # Keep the entire compact Zenith complex inland, including its final atlas.
    # A coastal drop under overlapping site pads otherwise creates an impassable lip.
    coast_start = 300 if task=='desert' else 265
    coast = np.clip((np.maximum(np.abs(x),np.abs(z))-coast_start)/(320-coast_start),0,1)
    terrain.height[:] = height*(1-coast)-8*coast
    terrain._compute_normals(); terrain._compute_biomes()


def _site_pads(world, campaign):
    t = world.terrain
    axis = np.linspace(-t._half,t._half,t.cfg.resolution)
    x,z = np.meshgrid(axis,axis,indexing='ij')
    centers = [s.center for s in campaign.sites.values()]+[campaign.spawn_xz]
    centers += [world.get(eid).pos[[0,2]] for eid,(sid,_) in campaign.roles.items() if sid not in campaign.sites]
    for cx,cz in centers:
        h = float(t.height_at(cx,cz))
        weight = np.clip((28-np.hypot(x-cx,z-cz))/10,0,1)
        t.height[:] = t.height*(1-weight)+h*weight
    t._compute_normals(); t._compute_biomes()
    for e in world.entities.values():
        e.pos = world.ground_pos(e.pos[0], e.pos[2]); e.home = e.pos.copy()
    for g in campaign.gates:
        g['position'] = world.ground_pos(g['position'][0],g['position'][2]).tolist()


def _scenery(world, campaign):
    """Regional scenery, retaining clear authored approaches to all fixtures."""
    props = scatter_props(world.terrain, campaign.seed, .65)
    points = np.array([e.pos[[0,2]] for e in world.entities.values()])
    segments = np.array([(a,b) for i,a in enumerate(points) for b in points[i+1:]])
    if campaign.task == 'desert':
        keep = np.isin(props.kind, (ROCK, BOULDER, CRYSTAL))
    else:
        keep = np.ones(len(props), bool)
        if campaign.task == 'summit':
            props.kind[np.isin(props.kind,(0,1,2,3))] = PINE
    # Conservative route corridors are world construction, not render cutaways.
    for i,p in enumerate(props.pos):
        if not keep[i]: continue
        if np.min(np.linalg.norm(points-p[[0,2]],axis=1))<16 or _distance_to_routes(p[[0,2]],segments)<4:
            keep[i] = False
    props = PropField(props.pos[keep],props.yaw[keep],props.scale[keep],props.kind[keep])
    buildings = []
    for i,site in enumerate(campaign.sites.values()):
        if campaign.task=='cleansing':
            # Open shrine pavilions frame the actual ritual sockets.
            center = world.ground_pos(*site.center)
            buildings.append(Building('shelter:'+site.id,'pavilion',tuple(center),0,campaign.seed*100+i))
            continue
        kind = ('timber_house','windmill','watchtower')[i%3] if campaign.task=='summit' else 'watchtower'
        for angle in np.arange(12)*np.pi/6:
            xz = np.asarray(site.center)+np.array((np.cos(angle),np.sin(angle)))*32
            r = RADII[kind]
            if _distance_to_routes(xz,segments)<r+2: continue
            if any(np.linalg.norm(xz-np.asarray(b.pos)[[0,2]])<r+RADII[b.kind]+2 for b in buildings): continue
            samples = xz+np.array([(a,b) for a in (-r,0,r) for b in (-r,0,r)])
            hs = world.terrain.height_at(samples[:,0],samples[:,1])
            if np.ptp(hs)>3.2: continue
            buildings.append(Building('shelter:'+site.id,kind,(float(xz[0]),float(hs.max()),float(xz[1])),
                                      float(np.arctan2(xz[0]-site.center[0],xz[1]-site.center[1])),
                                      campaign.seed*100+i,float(np.ceil(np.ptp(hs)*10)/10)))
            break
    for i,g in enumerate(campaign.gates):
        buildings.append(Building('frame:'+g['id'],'ruin_gate',tuple(g['position']),0,campaign.seed*101+i))
    # Clear vegetation from complete building footprints, including eaves.
    keep = np.ones(len(props),bool)
    for b in buildings:
        keep &= np.linalg.norm(props.pos[:,[0,2]]-np.asarray(b.pos)[[0,2]],axis=1)>RADII[b.kind]+2
    world.props = PropField(props.pos[keep],props.yaw[keep],props.scale[keep],props.kind[keep])
    world.architecture = GateCollisions(Architecture(buildings),campaign)


class TeyvatCampaignEnv(OpenWorldEnv):
    """Three original campaign tiers; the policy gets one final goal and local cues."""
    def __init__(self, task='summit', seed=7, *, render_enabled=True, width=384, height=216,
                 geography_scale=2.0, max_steps=40000):
        if task not in GOALS: raise ValueError(f'task must be one of {tuple(GOALS)}')
        if not np.isfinite(geography_scale) or geography_scale<.5: raise ValueError('geography_scale must be at least 0.5')
        self.task, self.render_enabled, self.geography_scale = task, render_enabled, float(geography_scale)
        scale = self.geography_scale
        cfg = EnvConfig(seed=seed, text_obs=True, backend='soft', action_repeat=6, max_episode_steps=max_steps,
            world=WorldConfig(terrain=TerrainConfig(size=640*scale,resolution=257,max_height=70,
                              snow_line=32 if task=='summit' else 178,tree_line=50 if task=='summit' else 126),
                              prop_density=.65,respawn_on_death=False,day_length=100000,start_time_of_day=.4),
            camera=CameraConfig(distance=7,far=400),
            render=RenderSettings(width=width,height=height,march_steps=64,refine_steps=4,shadow_steps=0,
                                  prop_draw_distance=180,max_view_distance=400),
            reward=RewardConfig(objective=0,quest=0,shaping=0,time_penalty=0,damage_penalty=0,
                                death_penalty=0,discovery=0,finish_bonus=1))
        super().__init__(cfg)
        self.observation_space['local_text'] = spaces.Text(max_length=900,charset=string.printable)

    def _build(self, seed):
        self.rng = np.random.default_rng(seed)
        terrain = Terrain(replace(self.cfg.world.terrain,seed=seed))
        _author_terrain(terrain,self.task,self.geography_scale)
        self.party = Party(self.cfg.party)
        empty = PropField(np.empty((0,3),np.float32),np.empty(0,np.float32),np.empty(0,np.float32),np.empty(0,np.int8))
        self.world = CampaignWorld(replace(self.cfg.world,seed=seed),terrain=terrain,props=empty,party=self.party)
        self.campaign = Campaign(self.world,self.task,seed,self.geography_scale)
        self.world.campaign = self.campaign
        _site_pads(self.world,self.campaign)
        _scenery(self.world,self.campaign)
        self.world.combat = CampaignCombat(self.world,self.party)
        # Only the terminal receipt reaches the old quest engine. There is no
        # active intermediate objective, stage index, or global target vector.
        objective = Objective('campaign_goal',ObjKind.ACTIVATE,GOALS[self.task],targets=('campaign_final',),reward=0)
        self.graph = QuestGraph([Quest('campaign',GOALS[self.task],GOALS[self.task],[Stage([objective])],reward=0)])
        self.quests = QuestEngine(self.graph)
        self.controller = CharacterController(terrain,self.world.props,replace(self.cfg.character))
        self.camera_rig = OrbitCamera(terrain,self.world.props,self.cfg.camera)
        self.controller.architecture = self.camera_rig.architecture = self.world.architecture
        self.char = self.controller.spawn(*self.campaign.spawn_xz)
        self.cam = self.camera_rig.reset(self.char)
        self.steps = 0; self._prev_dist = None; self._used_entities = set()
        self._episode_reward = 0.; self._last_events = []; self._last_frame = None
        self._campaign_done = False

    @property
    def instruction(self):
        return GOALS[self.task]

    def _task_vector(self):
        return np.zeros(15,np.float32)

    def local_text(self):
        c = self.campaign
        inventory = ', '.join(f'{k}={v}' for k,v in sorted(c.inventory.items()) if v) or 'empty'
        parts = [f'Pack: {inventory}.']
        if self.task=='summit':
            parts.append(f'Cold: {c.cold:.0f}/100. Quartz coating: {max(0,c.charge_until-self.world.elapsed):.0f}s.')
        if self.task=='desert': parts.append(f'Slate energy: {c.slate_energy}/6.')
        if self.world.elapsed-c.feedback_time<5:
            parts.append(c.feedback)
        forward = self.cam.pivot-self.cam.eye
        forward /= max(np.linalg.norm(forward),1e-8)
        visible = []
        for e in self.world.entities.values():
            if not e.alive: continue
            delta = e.pos+(0,1,0)-self.cam.eye
            distance = np.linalg.norm(e.pos-self.char.pos)
            if distance>24 or delta@forward/max(np.linalg.norm(delta),1e-8)<.70: continue
            samples = self.cam.eye+np.linspace(.08,.92,12)[:,None]*delta
            if np.any(samples[:,1]<self.world.terrain.height_at(samples[:,0],samples[:,2])): continue
            if self.world.architecture.blocked(samples,margin=0).any(): continue
            visible.append((distance,e.label+(' [active]' if e.used else '')))
        if visible:
            parts.append('Visible: '+'; '.join(f'{label} ({d:.0f}m)' for d,label in sorted(visible)[:5])+'.')
        return ' '.join(parts)[:900]

    def _observe(self):
        if self.render_enabled:
            out = super()._observe()
        else:
            # Without 3-D rendering the screen is blank, but the map UI still works.
            blank = np.zeros((self.cfg.render.height,self.cfg.render.width,3),np.uint8)
            out = dict(rgb=self._screen(blank),
                       proprio=self._proprio(),party=self.party.as_vector(),task=self._task_vector())
        out.update(instruction=self.instruction,local_text=self.local_text())
        return out

    def _info(self, events=None):
        c = self.campaign
        return dict(instruction=self.instruction,success='final' in c.facts,
                    quest_completion=len(c.facts)/len(c.milestones),quests_done=int('final' in c.facts),
                    objectives_done=len(c.facts),inventory=dict(c.inventory),episode_reward=self._episode_reward,
                    events=[e.type for e in (events or [])],
                    event_details=[dict(type=e.type,**plain(e.payload)) for e in (events or [])],
                    seed=self._episode_seed,suite_version=SUITE_VERSION,**self._map_info())

    def step(self, action):
        if self._campaign_done: raise gym.error.ResetNeeded('reset after campaign termination or truncation')
        result = super().step(action)
        self._campaign_done = result[2] or result[3]
        return result

    def state_digest(self):
        state = dict(campaign=self.campaign.state(),character=vars(self.char),camera=vars(self.cam),
                     world_rng=self.world.rng.bit_generator.state,episode_rng=self.rng.bit_generator.state,
                     campaign_rng=self.campaign.rng.bit_generator.state,
                     attack_cooldown=self.world._player_attack_cd,
                     party=vars(self.party),time=self.world.elapsed,time_of_day=self.world.time_of_day,
                     steps=self.steps,quest=self.quests.snapshot(),episode_reward=self._episode_reward,
                     terminal=self._campaign_done,entities=self.world.entities)
        return hashlib.sha256(json.dumps(plain(state),sort_keys=True,allow_nan=False).encode()).hexdigest()
