"""Skylands: layered traversal and a small, solvable construction trial."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import numpy as np
from gymnasium import spaces

from ..core import blocks as B
from ..core.config import EnvConfig, WorldConfig, RenderConfig, SimConfig
from ..physics import collide
from ..sim.construction import (Construction, MAX_PARTS, PART_COLUMNS, BUILD_STATE_NAMES,
                                BUILD_COMMANDS)
from ..world.skylands import generate
from .wildlands import WildlandsEnv, COMMANDS as SURVIVAL_COMMANDS

COMMANDS = SURVIVAL_COMMANDS + BUILD_COMMANDS
LAYERS = ("depths", "surface", "sky")


def default_config(small=False):
    return EnvConfig(
        starter_camp=False, obs_text=True,
        world_cfg=WorldConfig(size_x=48 if small else 96, size_z=48 if small else 96,
                              size_y=64 if small else 96),
        sim_cfg=SimConfig(weather_enabled=False, wind_enabled=False, fire_enabled=False),
        render_cfg=RenderConfig(width=96 if small else 128, height=72 if small else 128,
                                max_dist=96 if small else 144),
    )


class SkylandsEnv(WildlandsEnv):
    commands = COMMANDS

    def __init__(self, cfg=None, render_mode="rgb_array", **overrides):
        cfg = deepcopy(cfg) if cfg is not None else default_config()
        super().__init__(cfg, render_mode, **overrides)
        if self.cfg.action_space != "multi":
            raise ValueError("Skylands uses the multi action space")
        if min(self.cfg.world_cfg.size_x, self.cfg.world_cfg.size_z) < 32 or self.cfg.world_cfg.size_y < 40:
            raise ValueError("Skylands requires at least 32 x 40 x 32 voxels (x, y, z)")
        self.cfg.starter_camp = False
        self.action_space["part"] = spaces.Discrete(MAX_PARTS + 1)
        self.action_space["build_move"] = spaces.Box(-1, 1, (3,), np.float32)
        self.observation_space["construction"] = spaces.Box(
            -np.inf, np.inf, (MAX_PARTS, len(PART_COLUMNS)), np.float32)
        self.observation_space["build_state"] = spaces.Box(-np.inf, np.inf, (len(BUILD_STATE_NAMES),), np.float32)

    def _generate_world(self):
        return generate(replace(self.cfg.world_cfg, seed=self._seed))

    def _reset_mechanics(self):
        self.construction = Construction(self.world, self.agent, self.cfg.sim_cfg, self.world.meta["parts"])
        self.layer_maps = np.zeros((3, *self.map_memory.shape), np.uint8)
        self.layer = self._layer()
        self.success = False
        self._beacon_requested = False

    def _layer(self):
        _, surface, sky = self.world.meta["layer_heights"]
        return 2 if self.agent.pos[1] >= sky - 4 else (1 if self.agent.pos[1] >= surface - 2 else 0)

    def _explore(self):
        layer, old = self._layer(), self.layer
        self.layer = layer
        self.map_memory = self.layer_maps[layer]
        fresh, events = super()._explore()
        if layer != old:
            events.append(dict(type="layer_changed", previous=LAYERS[old], layer=LAYERS[layer]))
        return fresh, events

    def _dispatch(self, command, ingredients, action):
        self._beacon_requested = False
        events = self.construction.dispatch(command, int(action["part"]), action["build_move"])
        if command == "ascend":
            return events + self._ascend()
        if command in BUILD_COMMANDS:
            return events
        if command == "interact":
            target = np.asarray(self.world.meta["beacon"]) + 0.5
            delta = target - self.agent.eye
            distance = np.linalg.norm(delta)
            hit = self.world.raycast(self.agent.eye, delta, distance)
            if distance <= 3 and (not hit.hit or hit.pos == self.world.meta["beacon"]):
                edge, _, _, gap = self.world.meta["bridge"]
                if self.bridge_complete() and self.agent.on_ground and self.agent.pos[0] >= edge + gap:
                    self._beacon_requested = True
                    return events
                reason = "cross_to_beacon" if self.bridge_complete() else "bridge_incomplete"
                return events + [dict(type="action_failed", command=command, reason=reason)]
        return events + super()._dispatch(command, ingredients, action)

    def _ascend(self):
        def fail(reason):
            return [dict(type="action_failed", command="ascend", reason=reason)]

        a, cfg, w = self.agent, self.cfg.sim_cfg, self.world
        if self.construction.held is not None:
            return fail("holding_part")
        if not a.alive or a.stamina < 20:
            return fail("insufficient_stamina")
        head = a.pos + (0, cfg.agent_height + 0.01, 0)
        hit = w.raycast(head, (0, 1, 0), 32)
        if not hit.hit:
            return fail("no_ceiling")
        x, bottom, z = hit.pos
        top = bottom
        while top < w.size_y and w.is_solid(x, top, z):
            top += 1
        if top - bottom > 8:
            return fail("ceiling_too_thick")
        feet = np.array((a.pos[0], top + 0.01, a.pos[2]), np.float32)
        center = feet + (0, cfg.agent_height / 2, 0)
        if top + cfg.agent_height >= w.size_y or collide.box_collides(w, center, self.controller.half):
            return fail("blocked_exit")
        touching = collide.box_block_ids(w, center, self.controller.half)
        if touching & {B.LAVA, B.FIRE, B.CAMPFIRE_LIT, B.WATER}:
            return fail("unsafe_exit")
        previous = LAYERS[self._layer()]
        a.pos, a.vel = feet, np.zeros(3, np.float32)
        a.spend_stamina(20)
        a.on_ground, a.gliding, a.climbing, a.in_water, a.submerged = True, False, False, False, False
        a.air_time = 0
        return [dict(type="ascended", previous_layer=previous, layer=LAYERS[self._layer()],
                     height=float(top))]

    def bridge_complete(self):
        """A walkable, face-connected path across the gap at the deck height."""
        edge, y, z, gap = self.world.meta["bridge"]

        def walkable(x, zz):
            return (self.world.is_solid(x, y, zz)
                    and not self.world.is_solid(x, y + 1, zz)
                    and not self.world.is_solid(x, y + 2, zz))

        todo = [(edge - 1, zz) for zz in range(z - 5, z + 6) if walkable(edge - 1, zz)]
        seen = set(todo)
        while todo:
            x, zz = todo.pop()
            if x == edge + gap:
                return True
            for nxt in ((x - 1, zz), (x + 1, zz), (x, zz - 1), (x, zz + 1)):
                if (nxt not in seen and edge - 1 <= nxt[0] <= edge + gap
                        and z - 5 <= nxt[1] <= z + 5 and walkable(*nxt)):
                    seen.add(nxt)
                    todo.append(nxt)
        return False

    def _task_outcome(self, events):
        if self._beacon_requested and self.agent.alive and self.bridge_complete():
            self.success = True
            self.world.set_block(*self.world.meta["beacon"], B.BEACON_LIT)
            events.append(dict(type="trial_complete", task="aster_bridge"))
            return True, 10.0
        return False, 0.0

    def _observation(self):
        obs = super()._observation()
        obs["construction"], obs["build_state"] = self.construction.observe(self.layer)
        if self.cfg.obs_text:
            obs["text"] = (
                "Aster bridge trial: join the two beams across the sky gap, cross, and interact with the beacon. "
                "Grab a nearby part; build_move shifts east/up/south by one voxel; rotate turns 90 degrees; "
                "place fixes it to a touching surface. Ascend passes through a thin ceiling overhead. "
                f"Layer: {LAYERS[self.layer]}. Held part: {self.construction.held or 0}. "
                f"Health {self.agent.health:.1f}; stamina {self.agent.stamina:.1f}."
            )
        return obs

    def _info(self, events):
        out = super()._info(events)
        out.update(success=self.success, task_progress=1.0 if self.success else 0.5 if self.bridge_complete() else 0.0,
                   layer=LAYERS[self.layer], held_part=self.construction.held,
                   bridge_complete=self.bridge_complete(), visited_cells=int(np.count_nonzero(self.layer_maps)),
                   visited_cells_by_layer={name: int(np.count_nonzero(self.layer_maps[i]))
                                           for i, name in enumerate(LAYERS)})
        return out

    def _extra_entities(self):
        return self.construction.render_list() + self.world.meta.get("scenery", [])

    def _extra_lights(self):
        sources = sorted(self.world.meta["crystal_lights"], key=lambda p: np.linalg.norm(self.agent.pos - p))
        return [dict(pos=p, radius=10.0, intensity=1.2, color=(0.3, 0.85, 0.8))
                for p in sources[:8] if np.linalg.norm(self.agent.pos - p) < 22
                and self.world.get_f(*p) == B.CRYSTAL]


def make_small(**kwargs):
    cfg = kwargs.pop("cfg", None)
    return SkylandsEnv(cfg=cfg if cfg is not None else default_config(small=True), **kwargs)
