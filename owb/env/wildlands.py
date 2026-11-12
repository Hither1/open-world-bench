"""Gymnasium interface to the Wildlands exploration and survival sandbox."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import string

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..core import blocks as B, items as I
from ..core.config import EnvConfig, RenderConfig, WorldConfig
from ..entities.agent import AgentState, PROPRIO_DIM
from ..entities.npc import NpcSystem
from ..physics.bodies import BodySystem
from ..physics.character import CharacterController, MoveIntent
from ..physics.elements import ElementSystem, WEATHERS
from ..render.camera import resolve_third_person
from ..render import make_renderer
from ..sim.inventory import EQUIP_SLOTS
from ..sim import world_map as WM
from ..sim.survival import SurvivalInteractions
from ..world.worldgen import Landmark, generate

BUTTONS = ("jump", "sprint", "climb", "glide", "crouch")
COMMANDS = ("none", "interact", "eat", "cook", "equip", "attack", "stow")
DISCRETE_ACTIONS = (
    "wait", "forward", "back", "left", "right", "turn_left", "turn_right",
    "look_up", "look_down", "jump", "sprint", "climb", "glide", "dive",
    "interact", "eat", "cook", "equip", "attack", "stow", "next_slot", "prev_slot",
)
#: Appended to every discrete action list when the world map is enabled:
#: clicks the on-screen map logo (opens or closes the map).
MAP_ACTION = "map"
ITEM_IDS = tuple(I.ITEMS)
ITEM_INDEX = {item: i + 1 for i, item in enumerate(ITEM_IDS)}
INVENTORY_COLUMNS = ("item", "count", "durability", "equipped_slot", "dish", "duration") + I.ALL_EFFECTS
SURVIVAL_NAMES = ("time_of_day", "day", "weather", "wind_x", "wind_z",
                  "ambient_temperature", "elapsed_seconds", "selected_slot")


class WildlandsEnv(gym.Env):
    """Death terminates; max_steps truncates. This sandbox has no quest terminal.

    The minimap records visited cells only. Milestone/dense rewards pay once per
    landmark; dense additionally pays exploration_bonus per new map cell.

    With ``world_map`` enabled, the RGB view shows a map logo; ``click`` with
    ``pointer`` on it opens a fog-of-war map of the whole world, pausing the
    simulation until it is closed (see :mod:`owb.sim.world_map`).
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}
    commands = COMMANDS
    discrete_actions = DISCRETE_ACTIONS

    def __init__(self, cfg: EnvConfig | None = None, render_mode="rgb_array", **overrides):
        super().__init__()
        self.cfg = deepcopy(cfg) if cfg is not None else EnvConfig()
        for name, value in overrides.items():
            if not hasattr(self.cfg, name):
                raise TypeError(f"unknown EnvConfig field {name!r}")
            setattr(self.cfg, name, deepcopy(value))
        c, s, r = self.cfg, self.cfg.sim_cfg, self.cfg.render_cfg
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        if c.action_space not in ("multi", "discrete"):
            raise ValueError("action_space must be 'multi' or 'discrete'")
        if c.reward_mode not in ("milestone", "sparse", "dense"):
            raise ValueError("unknown reward_mode")
        if s.hunger_enabled or s.respawn_on_death:
            raise ValueError("hunger and automatic respawning are not implemented")
        if min(c.frame_skip, c.max_steps, c.minimap_size, c.minimap_scale,
               r.width, r.height, r.max_dist, s.dt, s.max_health, s.max_stamina) <= 0:
            raise ValueError("step, map, render and vital limits must be positive")
        if c.world_map and (c.map_cell <= 0 or (c.map_reveal_radius is not None and c.map_reveal_radius <= 0)):
            raise ValueError("map_cell and map_reveal_radius must be positive")
        if s.dt > 0.1:
            raise ValueError("sim_cfg.dt must be at most 0.1 seconds for stable physics")
        if min(c.world_cfg.size_x, c.world_cfg.size_z, c.world_cfg.size_y) < 16:
            raise ValueError("world dimensions must be at least 16")
        self.render_mode = render_mode
        self.metadata = dict(self.metadata, render_fps=max(1, round(1 / (s.dt * c.frame_skip))))
        self.renderer, self.controller = make_renderer(r), CharacterController(s)
        self.world, self._frame = None, None
        self._ended, self._seed = False, c.seed
        if c.world_map and MAP_ACTION not in self.discrete_actions:
            self.discrete_actions = tuple(self.discrete_actions) + (MAP_ACTION,)
        if c.action_space == "discrete":
            self.action_space = spaces.Discrete(len(self.discrete_actions))
        else:
            self.action_space = spaces.Dict({
                "move": spaces.Box(-1, 1, (2,), np.float32),  # forward, strafe
                "look": spaces.Box(-1, 1, (2,), np.float32),  # yaw, pitch rate
                "buttons": spaces.MultiBinary(len(BUTTONS)),
                "command": spaces.Discrete(len(self.commands)),
                "slot": spaces.Discrete(32),
                "ingredients": spaces.MultiDiscrete([33] * 5),  # 0 empty, 1..32 slots
            })
            if c.world_map:
                # Normalised screen position (u right, v down) and a click button.
                self.action_space["pointer"] = spaces.Box(0, 1, (2,), np.float32)
                self.action_space["click"] = spaces.Discrete(2)
        obs = {"survival": spaces.Box(-np.inf, np.inf, (len(SURVIVAL_NAMES),), np.float32)}
        if c.obs_rgb:
            obs["rgb"] = spaces.Box(0, 255, (r.height, r.width, 3), np.uint8)
        if c.obs_depth:
            obs["depth"] = spaces.Box(0, r.max_dist, (r.height, r.width), np.float32)
        if c.obs_semantic:
            obs["semantic"] = spaces.Box(0, 255, (r.height, r.width), np.uint8)
        if c.obs_proprio:
            obs["proprio"] = spaces.Box(-np.inf, np.inf, (PROPRIO_DIM,), np.float32)
        if c.obs_inventory:
            obs["inventory"] = spaces.Box(0, np.inf, (32, len(INVENTORY_COLUMNS)), np.float32)
        if c.obs_minimap:
            obs["minimap"] = spaces.Box(0, 255, (c.minimap_size, c.minimap_size), np.uint8)
        if c.obs_text:
            obs["text"] = spaces.Text(max_length=1024, charset=string.printable)
        self.observation_space = spaces.Dict(obs)

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("reset options are not supported")
        self._seed = self._seed if seed is None else int(seed)
        super().reset(seed=self._seed)
        result = self._generate_world()
        self.world = result.world
        if self.cfg.starter_camp:
            self._starter_camp(result)
        self.agent = AgentState(pos=np.array(result.spawn, dtype=np.float32))
        self.agent.reset_vitals(self.cfg.sim_cfg)
        for item, count in (("glider", 1), ("sword_wood", 1), ("apple", 3)):
            self.agent.inventory.add(item, count)
        self.agent.inventory.equip(self.agent.inventory.find("sword_wood"))
        self.elements = ElementSystem(self.cfg.sim_cfg, self.np_random)
        if not self.cfg.sim_cfg.wind_enabled:
            self.elements.climate.wind[:] = 0
            self.elements.climate.wind_target[:] = 0
        self.npcs = NpcSystem(self.cfg.sim_cfg, self.np_random)
        for spawn in result.npc_spawns[:self.cfg.sim_cfg.max_npcs]:
            self.npcs.spawn(**spawn)
        self.bodies = BodySystem(self.cfg.sim_cfg, self.np_random)
        for spawn in result.bodies[:self.cfg.sim_cfg.max_bodies]:
            self.bodies.add(**spawn)
        self.gameplay = SurvivalInteractions(self.world, self.agent, self.elements,
                                             self.npcs, result.plants, result.chests)
        self.landmarks, self.discovered = result.landmarks, set()
        scale = self.cfg.minimap_scale
        self.map_memory = np.zeros(((self.world.size_x + scale - 1) // scale,
                                    (self.world.size_z + scale - 1) // scale), np.uint8)
        self.steps, self.ticks, self.elapsed, self.selected_slot = 0, 0, 0.0, 0
        self._ended, self._frame = False, None
        self._reset_mechanics()
        self._explore()
        self._init_world_map()
        return self._observation(), self._info([])

    def _generate_world(self):
        return generate(replace(self.cfg.world_cfg, seed=self._seed))

    def _reset_mechanics(self):
        """Extension point for worlds with additional mechanics."""

    def _dispatch(self, command, ingredients, action):
        return self.gameplay.dispatch(command, self.selected_slot, ingredients,
                                      self.cfg.sim_cfg.durability_enabled)

    def _task_outcome(self, events):
        return False, 0.0

    def _extra_entities(self):
        return []

    def _extra_lights(self):
        return []

    def _starter_camp(self, result):
        """Give every seed a safe clearing and an immediately usable cooking loop."""
        w = result.world
        x = int(np.clip(result.spawn[0], 5, w.size_x - 6))
        z = int(np.clip(result.spawn[2], 5, w.size_z - 6))
        # Use terrain height, not tree canopies or structure roofs at the spawn.
        ground = result.heightmap[x, z] if result.heightmap is not None else result.spawn[1] - 1
        y = int(np.clip(ground, min(w.sea_level + 1, w.size_y - 5), w.size_y - 5))
        w.fill_box((x - 4, 0, z - 4), (x + 4, y, z + 4), B.DIRT)
        w.fill_box((x - 4, y + 1, z - 4), (x + 4, w.size_y - 1, z + 4), B.AIR)
        w.fill_box((x - 4, y, z - 4), (x + 4, y, z + 4), B.GRASS)
        for collection in (result.plants, result.npc_spawns, result.chests, result.bodies,
                           result.landmarks):
            collection[:] = [obj for obj in collection if not (
                abs((obj.pos if isinstance(obj, Landmark) else obj["pos"])[0] - (x + 0.5)) < 6
                and abs((obj.pos if isinstance(obj, Landmark) else obj["pos"])[2] - (z + 0.5)) < 6)]
        w.set_block(x, y + 1, z + 2, B.COOKPOT)
        w.set_block(x - 2, y + 1, z, B.CAMPFIRE)
        result.spawn = (x + 0.5, y + 1.01, z + 0.5)
        for dx, dz, item in ((1, 0, "spicy_pepper"), (1, 1, "mushroom"), (2, 1, "apple")):
            result.plants.append(dict(pos=(x + dx + 0.5, y + 1, z + dz + 0.5), item=item))
        result.landmarks.append(Landmark("camp", "Trailhead camp", result.spawn, radius=4))

    def _decode(self, action):
        self._click = (False, 0.0, 0.0)
        if self.cfg.world_map and self.cfg.action_space == "multi" and isinstance(action, dict):
            # Actions recorded before the map existed carry no pointer fields.
            action = dict(action)
            action.setdefault("pointer", np.zeros(2, np.float32))
            action.setdefault("click", 0)
        if not self.action_space.contains(action):
            raise ValueError("action is outside action_space")
        intent, look, command, ingredients = MoveIntent(), np.zeros(2), "none", []
        if self.cfg.action_space == "multi":
            intent.forward, intent.strafe = map(float, action["move"])
            for name, pressed in zip(BUTTONS, action["buttons"]):
                setattr(intent, name, bool(pressed))
            look = np.asarray(action["look"])
            command = self.commands[int(action["command"])]
            self.selected_slot = int(action["slot"])
            ingredients = [int(i) - 1 for i in action["ingredients"] if i]
            if self.cfg.world_map:
                self._click = WM.parse_click(action["click"], action["pointer"])
        else:
            name = self.discrete_actions[int(action)]
            if name in ("forward", "back", "jump", "sprint", "climb", "glide"):
                intent.forward = -1.0 if name == "back" else 1.0
            if name in ("left", "right"):
                intent.strafe = -1.0 if name == "left" else 1.0
            if name in ("jump", "sprint", "climb", "glide"):
                setattr(intent, name, True)
            intent.crouch = name == "dive"
            if name in ("turn_left", "turn_right"):
                look[0] = -1 if name == "turn_left" else 1
            if name in ("look_up", "look_down"):
                look[1] = 1 if name == "look_up" else -1
            if name in self.commands:
                command = name
            if name in ("next_slot", "prev_slot"):
                self.selected_slot = (self.selected_slot + (1 if name == "next_slot" else -1)) % 32
            ingredients = [self.selected_slot]
            if name == MAP_ACTION:
                r = self.cfg.render_cfg
                self._click = (True, *WM.logo_center(r.width, r.height))
                command = "none"
        return intent, look, command, ingredients

    def step(self, action):
        if self.world is None or self._ended:
            raise gym.error.ResetNeeded("call reset() before starting or after an episode ends")
        intent, look, command, ingredients = self._decode(action)
        c, a, elements = self.cfg, self.agent, self.elements
        ui_events = self._map_click()
        if self.map_open:
            return self._paused_step(ui_events)
        events = ui_events + self._dispatch(command, ingredients, action)
        for tick in range(c.frame_skip):
            dt = c.sim_cfg.dt
            a.tick_effects(dt)
            a.cooldown = max(0, a.cooldown - dt)
            a.yaw = float((a.yaw + look[0] * 1.8 * dt) % (2 * np.pi))
            a.pitch = float(np.clip(a.pitch + look[1] * 1.4 * dt, -1.45, 1.45))
            events.extend(elements.step_climate(dt, elements.ambient_temp(self.world, a.pos)))
            cl = elements.climate
            context = dict(raining=cl.raining, wind=cl.wind, rng=self.np_random,
                           updraft=elements.updraft_at(a.pos))
            events.extend(self.controller.step(self.world, a, intent, dt, context))
            if tick == 0:
                intent.jump = False  # press once; other movement buttons stay held
            events.extend(self.bodies.step(self.world, dt, context))
            events.extend(elements.step_fire(self.world, dt))
            if a.alive:
                events.extend(self.npcs.step(self.world, a, dt, dict(is_night=cl.is_night)))
            if a.alive:
                events.extend(elements.step_lightning(self.world, a, self.bodies, self.npcs.alive, dt))
            if a.alive:
                events.extend(elements.step_temperature(self.world, a, dt))
            self.ticks += 1
            self.elapsed += dt
            if not a.alive:
                break
        for event in events:
            if event["type"] == "npc_died":
                for item, count in event["drops"]:
                    self.gameplay.plants.append(dict(pos=event["pos"], item=item, count=count))
            elif event["type"] == "ignite_agent":
                a.hurt(c.sim_cfg.fire_damage * c.sim_cfg.dt, "fire")
        self.steps += 1
        new_cells, discoveries = self._explore()
        events.extend(discoveries)
        self._reveal_map()
        success, task_reward = self._task_outcome(events)
        terminated = not a.alive or success
        truncated = self.steps >= c.max_steps and not terminated
        if not a.alive:
            a.deaths += 1
            if not any(e["type"] == "death" for e in events):
                events.append(dict(type="death"))
        reward = -c.time_penalty + task_reward
        if c.reward_mode != "sparse":
            reward += len(discoveries)
        if c.reward_mode == "dense":
            reward += new_cells * c.exploration_bonus
        if not a.alive:
            reward -= c.death_penalty
        self._ended, self._frame = terminated or truncated, None
        return self._observation(), float(reward), bool(terminated), bool(truncated), self._info(events)

    def _explore(self):
        a, w = self.agent, self.world
        x, z = np.clip(a.pos[[0, 2]].astype(int), (0, 0), (w.size_x - 1, w.size_z - 1))
        mx, mz = x // self.cfg.minimap_scale, z // self.cfg.minimap_scale
        fresh = int(self.map_memory[mx, mz] == 0)
        self.map_memory[mx, mz] = int(w.biome[x, z]) + 1
        events = []
        for index, landmark in enumerate(self.landmarks):
            if index in self.discovered:
                continue
            if np.linalg.norm(np.asarray(landmark.pos) - a.pos) <= landmark.radius:
                if w.line_of_sight(a.eye, np.asarray(landmark.pos) + (0, 1, 0)):
                    self.discovered.add(index)
                    events.append(dict(type="discover", name=landmark.name, kind=landmark.kind))
        return fresh, events

    def _minimap(self):
        n, scale = self.cfg.minimap_size, self.cfg.minimap_scale
        # North-up, agent-centred: rows are z, columns are x. 255 is the agent.
        cx, cz = (self.agent.pos[[0, 2]] / scale).astype(int)
        xs, zs = np.arange(n) + cx - n // 2, np.arange(n) + cz - n // 2
        valid_x = (xs >= 0) & (xs < self.map_memory.shape[0])
        valid_z = (zs >= 0) & (zs < self.map_memory.shape[1])
        out = np.zeros((n, n), np.uint8)
        out[np.ix_(valid_z, valid_x)] = self.map_memory[np.ix_(xs[valid_x], zs[valid_z])].T
        out[n // 2, n // 2] = 255
        return out

    def _inventory(self):
        out = np.zeros((32, len(INVENTORY_COLUMNS)), np.float32)
        inv = self.agent.inventory
        for index, st in enumerate(inv.stacks):
            equipped = next((i + 1 for i, slot in enumerate(EQUIP_SLOTS)
                             if inv.equipped[slot] == st.uid), 0)
            out[index, :6] = (ITEM_INDEX[st.item], st.count, st.durability or 0,
                             equipped, bool(st.data.get("dish")), st.duration())
            out[index, 6:] = [st.effects().get(effect, 0) for effect in I.ALL_EFFECTS]
        return out

    def _observation(self):
        a, c, cl = self.agent, self.cfg, self.elements.climate
        obs = {"survival": np.array([
            cl.time, cl.day, WEATHERS.index(cl.weather), cl.wind[0], cl.wind[2],
            self.elements.ambient_temp(self.world, a.pos), self.elapsed, self.selected_slot,
        ], np.float32)}
        if c.obs_rgb or c.obs_depth or c.obs_semantic:
            frame = self._render_buffers()
            for key in ("rgb", "depth", "semantic"):
                if getattr(c, "obs_" + key):
                    obs[key] = frame[key].copy()
            if c.obs_rgb:
                obs["rgb"] = self._screen(frame["rgb"])
        if c.obs_proprio:
            obs["proprio"] = a.proprio()
        if c.obs_inventory:
            obs["inventory"] = self._inventory()
        if c.obs_minimap:
            obs["minimap"] = self._minimap()
        if c.obs_text:
            obs["text"] = (f"Explore and survive. {cl.weather}; {self.world.region_name(a.pos[0], a.pos[2])}. "
                           f"Health {a.health:.1f}; stamina {a.stamina:.1f}. "
                           f"Inventory: {a.inventory.describe()}")[:1024]
        return obs

    def _info(self, events):
        return dict(seed=self._seed, steps=self.steps, elapsed=self.elapsed,
                    health=float(self.agent.health), stamina=float(self.agent.stamina),
                    weather=self.elements.climate.weather, events=deepcopy(events),
                    inventory=deepcopy(self.agent.inventory.to_dict()), selected_slot=self.selected_slot,
                    effects=[dict(kind=e.kind, magnitude=e.magnitude, ttl=e.ttl) for e in self.agent.effects],
                    visited_cells=int(np.count_nonzero(self.map_memory)),
                    discovered_landmarks=[self.landmarks[i].name for i in sorted(self.discovered)],
                    map_open=self.map_open, **self._map_info())

    def _render_buffers(self):
        if self._frame is None:
            a, cl, c = self.agent, self.elements.climate, self.cfg.render_cfg
            eye = a.pos + np.array((0, self.cfg.sim_cfg.eye_height, 0), np.float32)
            yaw, pitch = a.yaw, a.pitch
            if c.third_person:
                cam = resolve_third_person(self.world, eye, yaw, pitch, c.cam_distance, c.cam_pitch_offset)
                eye, yaw, pitch = cam.pos, cam.yaw, cam.pitch
            entities = self.npcs.render_list(a.pos) + self.bodies.render_list() + self.gameplay.render_list()
            entities += self._extra_entities()
            if c.third_person:
                entities.append(dict(pos=tuple(a.pos + (0, 0.9, 0)), radius=0.35, height=1.75,
                                     color=(59, 128, 147), semantic=B.SEM_SELF,
                                     yaw=a.yaw, gliding=a.gliding,
                                     phase=a.distance_travelled * 2.5))
            self._frame = self.renderer.render(
                self.world, eye, yaw, pitch, time_of_day=cl.time, weather=cl.weather,
                lights=self.elements.light_sources(self.world, a.pos) + self._extra_lights(), entities=entities,
                flash=cl.flash, rain_intensity=float(cl.raining), frame=self.ticks)
            if c.hud:
                from ..render.hud import draw_hud
                draw_hud(self._frame["rgb"], a)
        return self._frame

    # ------------------------------------------------------------ world map
    @property
    def map_open(self):
        return bool(self.cfg.world_map and self.map_ui.open)

    def _map_layers(self):
        """(name, standing-height low, high, tab colour) per map layer."""
        return (("world", 0, self.world.size_y, (120, 168, 96)),)

    def _map_layer_index(self):
        return 0

    def _map_reveal_radius(self):
        a, w = self.agent, self.world
        x, z = np.clip(a.pos[[0, 2]].astype(int), (0, 0), (w.size_x - 1, w.size_z - 1))
        return WM.reveal_radius(self.map_fog[0].radius, a.pos[1] - (w.height[x, z] + 1))

    def _init_world_map(self):
        self._map_cache = {}
        if not self.cfg.world_map:
            return
        layers = self._map_layers()
        self.map_ui = WM.WorldMapUI([name for name, *_ in layers], [color for *_, color in layers])
        bounds = (0.0, 0.0, float(self.world.size_x), float(self.world.size_z))
        radius = self.cfg.map_reveal_radius
        if radius is None:
            radius = float(np.clip(0.08 * min(self.world.size_x, self.world.size_z), 8.0, 24.0))
        self.map_fog = [WM.FogOfWar(bounds, self.cfg.map_cell, radius) for _ in layers]
        self._map_layer = None
        self._reveal_map()

    def _reveal_map(self):
        if not self.cfg.world_map:
            return
        layer = self._map_layer_index()
        if layer != self._map_layer:
            self.map_fog[layer].reset_trail()  # no trail across a layer change
            self._map_layer = layer
        self.map_fog[layer].reveal(float(self.agent.pos[0]), float(self.agent.pos[2]), self._map_reveal_radius())

    def _map_click(self):
        if not self.cfg.world_map or not self._click[0]:
            return []
        r = self.cfg.render_cfg
        return self.map_ui.click(self._click[1], self._click[2], r.width, r.height, self._map_layer_index())

    def _paused_step(self, events):
        """The map is open: no simulated time passes, but the decision counts."""
        c = self.cfg
        self.steps += 1
        truncated = self.steps >= c.max_steps
        self._ended = truncated
        return self._observation(), float(-c.time_penalty), False, bool(truncated), self._info(events)

    def _map_markers(self, layer):
        _, lo, hi, _ = self._map_layers()[layer]
        return [(lm.pos[0], lm.pos[2]) for i, lm in enumerate(self.landmarks)
                if i in self.discovered and (len(self.map_fog) == 1 or lo <= lm.pos[1] + 1 < hi)]

    def _map_base(self, layer):
        key = (layer, self.world.rev)
        if key not in self._map_cache:
            _, lo, hi, _ = self._map_layers()[layer]
            self._map_cache = {k: v for k, v in self._map_cache.items() if k[0] != layer}
            self._map_cache[key] = WM.voxel_surface(self.world, lo, hi)
        return self._map_cache[key]

    def _render_open_map(self, width, height):
        layer = self.map_ui.view_layer
        here = layer == self._map_layer_index()
        heading = (np.cos(self.agent.yaw), np.sin(self.agent.yaw))
        return WM.render_map(width, height, self._map_base(layer),
                             (0.0, 0.0, float(self.world.size_x), float(self.world.size_z)),
                             self.map_fog[layer], agent=self.agent.pos[[0, 2]] if here else None,
                             heading=heading, markers=self._map_markers(layer), tabs=self.map_ui.tabs())

    def _screen(self, rgb):
        """The policy's view: 3-D frame with the map logo, or the open map."""
        if not self.cfg.world_map:
            return rgb.copy()
        return self.map_ui.compose(rgb, self._render_open_map)

    def _map_info(self):
        if not self.cfg.world_map:
            return {}
        return dict(explored_fraction=self.map_fog[self._map_layer_index()].fraction)

    def render(self):
        if self.world is None:
            raise gym.error.ResetNeeded("call reset() before render()")
        return self._screen(self._render_buffers()["rgb"])

    def close(self):
        self._frame = None


def make_small(**kwargs):
    """A compact landscape for CPU rollouts; keeps the same gameplay contract."""
    cfg = kwargs.pop("cfg", None)
    if cfg is None:
        cfg = EnvConfig(world_cfg=WorldConfig(size_x=96, size_z=96, size_y=64,
                                             relief=16, mountain_relief=18),
                        render_cfg=RenderConfig(width=96, height=72, max_dist=72))
    return WildlandsEnv(cfg=cfg, **kwargs)
