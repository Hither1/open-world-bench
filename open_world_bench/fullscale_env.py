"""Full-scale Teyvat Task A as a native environment, route-preview scope.

The environment runs the phase-1 world (`fullscale_world`) under the solid-
terrain controller (`fullscale_physics`), with the ravine gate's collision and
two working lifts. Phase 2 adds the course mechanics (relays, camps, items,
locks). Until then the env offers a **route preview**: each milestone is a
sequence of checkpoints that completes when the character arrives at them in
order, and arriving at the authored points opens the same facts the task logic
will (``ferry_open``, ``corridor_open``, ``cap_installed``). Nothing is
teleported: the character walks, swims, glides, and rides lifts natively.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json

import numpy as np

from .camera import CameraConfig, OrbitCamera
from .character import MoveState
from .env import EnvConfig, OpenWorldEnv
from .events import Event
from .fullscale_physics import PHYSICS_VERSION, FullScaleCombat, FullScaleController
from .fullscale_world import CAP_INSTALLED, CORRIDOR_OPEN, FERRY_OPEN, SUITE_VERSION, build_world
from .party import Party
from .quests import ObjKind, Objective, Quest, QuestEngine, QuestGraph, Stage
from .render import RenderSettings
from .world import EntityKind, World, WorldConfig

PREVIEW_VERSION = "teyvat-fullscale-route-preview-v1"
LIFT_REACH = 4.0            # metres from a lift end to board it
LIFT_SPEED = 3.0            # metres per second along the cabin path
LIFT_CLEARANCE = 3.0        # cable height above the upper landing
ARRIVAL_RADIUS = 5.0        # checkpoint radius (horizontal)
ARRIVAL_HEIGHT = 4.0        # checkpoint tolerance (vertical)
BEACON_OFFSET = (0.0, -6.0)  # beacon position relative to the pillar centre


@dataclass(frozen=True)
class Checkpoint:
    name: str                   # layout anchor, or "beacon"
    label: str
    grants: str | None = None   # fact opened on arrival


@dataclass(frozen=True)
class PreviewMilestone:
    id: str
    title: str
    checkpoints: tuple


# The twelve Task A milestones with the places each one visits. Facts open
# where the task logic will open them: the mechanic installs the casing at
# the lower ferry, the quartermaster opens the ravine gate, and the cap goes
# into the service-lift socket.
ROUTE = (
    PreviewMilestone("A01", "Learn relay repair from the keeper",
                     (Checkpoint("keeper", "Relay keeper"), Checkpoint("yard_1", "Training relay 1"))),
    PreviewMilestone("A02", "Repair a second training relay", (Checkpoint("yard_2", "Training relay 2"),)),
    PreviewMilestone("A03", "Recover an overloaded training relay", (Checkpoint("yard_3", "Training relay 3"),)),
    PreviewMilestone("A04", "Survey the marsh supply route",
                     (Checkpoint("junction_1", "Reed post 1"), Checkpoint("junction_2", "Reed post 2"),
                      Checkpoint("junction_3", "Reed post 3"), Checkpoint("cache", "Supply cache (registered)"))),
    PreviewMilestone("A05", "Recover the ferry-lift casing", (Checkpoint("camp", "Raider camp"),)),
    PreviewMilestone("A06", "Repair the ferry lift",
                     (Checkpoint("ferry_low", "Lower ferry: casing installed", grants=FERRY_OPEN),
                      Checkpoint("ferry_high", "Upper landing: test lever"))),
    PreviewMilestone("A07", "Supply the highland outpost",
                     (Checkpoint("cache", "Supply cache (S1 return)"),
                      Checkpoint("outpost", "Quartermaster: ravine gate opens", grants=CORRIDOR_OPEN))),
    PreviewMilestone("A08", "Restore the first field relay", (Checkpoint("relay_1", "Field relay 1"),)),
    PreviewMilestone("A09", "Stabilize the second field relay", (Checkpoint("relay_2", "Field relay 2"),)),
    PreviewMilestone("A10", "Recover the summit service cap",
                     (Checkpoint("west_edge", "West edge of the plateau"), Checkpoint("ruin", "Ruin vault (S2)"))),
    PreviewMilestone("A11", "Reopen summit access",
                     (Checkpoint("summit_base", "Service lift: cap installed", grants=CAP_INSTALLED),
                      Checkpoint("summit_top", "Summit work platform"))),
    PreviewMilestone("A12", "Commission the summit beacon", (Checkpoint("beacon", "Summit beacon"),)),
)


@dataclass
class LiftState:
    id: str
    path: np.ndarray            # (n, 3) cabin path from the bottom to the top
    opens: str
    position: float = 0.0       # metres along the path where the cabin rests
    riding: int = 0             # +1 up, -1 down, 0 parked

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.path, axis=0), axis=1).sum())

    def point(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.length))
        for a, b in zip(self.path[:-1], self.path[1:]):
            seg = float(np.linalg.norm(b - a))
            if s <= seg or b is self.path[-1]:
                return a + (b - a) * (s / max(seg, 1e-9))
            s -= seg
        return self.path[-1].copy()


def lift_path(terrain, bottom, top) -> np.ndarray:
    """Rise beside the lower end, cross on a level cable, set down on the upper end."""
    bh = float(terrain.height_at(*bottom))
    th = float(terrain.height_at(*top))
    cable = th + LIFT_CLEARANCE
    return np.array([[bottom[0], bh, bottom[1]], [bottom[0], cable, bottom[1]],
                     [top[0], cable, top[1]], [top[0], th, top[1]]], dtype=np.float64)


@dataclass
class PreviewState:
    milestone: int = 0
    checkpoint: int = 0
    facts: set = field(default_factory=set)
    completed: list = field(default_factory=list)
    visited: list = field(default_factory=list)


class FullScaleSimWorld(World):
    """World with lifts, gate facts, and route-preview checkpoints."""

    def setup(self, fullscale, lifts):
        self.fullscale, self.lifts = fullscale, lifts
        self.preview = PreviewState()
        self.pending: list[Event] = []
        self.last_feedback = ""

    # ------------------------------------------------------------ positions
    def place(self, name) -> np.ndarray:
        if name == "beacon":
            sx, sz = self.fullscale.layout.anchors["summit_top"]
            x, z = sx + BEACON_OFFSET[0], sz + BEACON_OFFSET[1]
        else:
            x, z = self.fullscale.layout.anchors[name]
        return np.array([x, float(self.terrain.height_at(x, z)), z])

    def current_checkpoint(self):
        p = self.preview
        if p.milestone >= len(ROUTE):
            return None, None
        milestone = ROUTE[p.milestone]
        return milestone, milestone.checkpoints[p.checkpoint]

    def riding(self):
        return next((lift for lift in self.lifts if lift.riding), None)

    # --------------------------------------------------------------- events
    def emit(self, kind, **payload):
        self.pending.append(Event(kind, payload))

    def open_fact(self, fact):
        if fact not in self.fullscale.state.facts:
            self.fullscale.state.facts.add(fact)
            self.preview.facts = set(self.fullscale.state.facts)
            self.emit("fact_opened", fact=fact)

    # -------------------------------------------------------------- updates
    def interact(self, char):
        for lift in self.lifts:
            ends = (lift.point(0.0), lift.point(lift.length))
            for end, direction in zip(ends, (+1, -1)):
                if np.linalg.norm((char.pos - end)[[0, 2]]) < LIFT_REACH and abs(char.pos[1] - end[1]) < 3.0:
                    if lift.opens not in self.fullscale.state.facts:
                        self.last_feedback = "The lift is not running."
                        return [Event("lift_locked", {"lift": lift.id})]
                    lift.position = 0.0 if direction > 0 else lift.length
                    lift.riding = direction
                    self.last_feedback = "Riding the lift."
                    return [Event("lift_board", {"lift": lift.id, "direction": "up" if direction > 0 else "down"})]
        return super().interact(char)

    def update(self, char, dt, char_cfg):
        events = super().update(char, dt, char_cfg)
        lift = self.riding()
        if lift is not None:
            lift.position += lift.riding * LIFT_SPEED * dt
            done = lift.position <= 0.0 or lift.position >= lift.length
            lift.position = float(np.clip(lift.position, 0.0, lift.length))
            char.pos[:] = lift.point(lift.position)
            char.vel[:] = 0.0
            char.grounded = True
            char.state = MoveState.GROUNDED
            if done:
                self.emit("lift_arrived", lift=lift.id, end="top" if lift.riding > 0 else "bottom")
                lift.riding = 0
        self._check_arrival(char)
        events.extend(self.pending)
        self.pending.clear()
        return events

    def _check_arrival(self, char):
        milestone, checkpoint = self.current_checkpoint()
        if checkpoint is None or self.riding() is not None:
            return
        target = self.place(checkpoint.name)
        if (np.linalg.norm((char.pos - target)[[0, 2]]) > ARRIVAL_RADIUS
                or abs(char.pos[1] - target[1]) > ARRIVAL_HEIGHT):
            return
        p = self.preview
        p.visited.append(dict(milestone=milestone.id, checkpoint=checkpoint.name, time=round(self.elapsed, 6)))
        self.emit("checkpoint", milestone=milestone.id, checkpoint=checkpoint.name, label=checkpoint.label)
        self.last_feedback = checkpoint.label
        if checkpoint.grants:
            self.open_fact(checkpoint.grants)
        p.checkpoint += 1
        if p.checkpoint == len(milestone.checkpoints):
            p.completed.append(dict(id=milestone.id, time=round(self.elapsed, 6)))
            self.emit("activate", entity=milestone.id)
            self.emit("milestone_complete", milestone=milestone.id, title=milestone.title)
            p.milestone += 1
            p.checkpoint = 0


class TeyvatFullScaleEnv(OpenWorldEnv):
    """Native-action full-scale Task A world (route preview until phase 2)."""

    def __init__(self, seed=7, *, render_enabled=False, width=384, height=216, max_steps=40_000):
        self.render_enabled = render_enabled
        cfg = EnvConfig(seed=seed, text_obs=True, backend="soft", action_repeat=6, max_episode_steps=max_steps,
                        world=WorldConfig(prop_density=0, day_length=100000, start_time_of_day=.43),
                        camera=CameraConfig(distance=9, far=420),
                        render=RenderSettings(width=width, height=height, march_steps=48, refine_steps=4,
                                              shadow_steps=0, prop_draw_distance=90, max_view_distance=420,
                                              ground_cover=False, clouds=False))
        cfg.reward.shaping = 0
        cfg.reward.discovery = 0
        super().__init__(cfg)

    def _build(self, seed):
        c = self.cfg
        self.fullscale = fw = build_world(seed)
        self.world_digest = fw.digest()          # terrain and scenery are static
        self.rng = np.random.default_rng(seed)
        self.party = Party(c.party)
        wcfg = replace(c.world, seed=seed, terrain=fw.terrain.cfg)
        self.world = FullScaleSimWorld(wcfg, terrain=fw.terrain, props=fw.props, party=self.party)
        lifts = [LiftState(l.id, lift_path(fw.terrain, l.bottom, l.top), l.opens) for l in fw.lifts]
        self.world.setup(fw, lifts)
        self.world.architecture = fw.architecture
        self.world.combat = FullScaleCombat(self.world, self.party)
        self._spawn_sites()
        stages = [Stage([Objective(m.id, ObjKind.ACTIVATE, m.title, targets=(m.id,))]) for m in ROUTE]
        self.graph = QuestGraph([Quest("fullscale_a", "Task A", "Full-scale route preview", stages, reward=0)])
        self.quests = QuestEngine(self.graph)
        char_cfg = replace(c.character)
        char_cfg.max_hp = self.party.max_hp
        char_cfg.stamina_regen *= self.party.stamina_regen_multiplier
        for cost in ("sprint_cost", "climb_cost", "glide_cost", "swim_cost", "jump_cost"):
            setattr(char_cfg, cost, getattr(char_cfg, cost) * self.party.stamina_multiplier)
        self.controller = FullScaleController(fw.terrain, fw.props, char_cfg)
        self.camera_rig = OrbitCamera(fw.terrain, fw.props, c.camera)
        self.controller.architecture = self.camera_rig.architecture = fw.architecture
        sx, sz = fw.layout.anchors["spawn"]
        self.char = self.controller.spawn(sx, sz)
        self.cam = self.camera_rig.reset(self.char)
        self.cam.pitch = .34
        self.steps = 0
        self._prev_dist = None
        self._used_entities = set()
        self._episode_reward = 0.0
        self._last_events = []
        self._last_frame = None

    def _spawn_sites(self):
        """Static site fixtures so places read correctly; mechanics arrive in phase 2."""
        w = self.world
        anchors = self.fullscale.layout.anchors

        def relay(name, label, version=1):
            x, z = anchors[name]
            for role, (dx, dz) in zip(("source", "vane", "drain", "core"), ((-5, -4), (5, -4), (-5, 5), (5, 5))):
                kind = EntityKind.BEACON if role == "core" else EntityKind.PILLAR
                w.spawn_at(f"{name}:{role}", kind, x + dx, z + dz, radius=2.1,
                           label=f"{label} {role}", visual_role=role, visual_version=version)

        def single(name, kind, label, dx=0.0, dz=0.0, **kw):
            x, z = anchors[name]
            w.spawn_at(f"{name}:{kind.name.lower()}", kind, x + dx, z + dz, radius=2.1, label=label, **kw)

        single("keeper", EntityKind.NPC, "Relay keeper")
        for k, name in enumerate(("yard_1", "yard_2", "yard_3"), 1):
            relay(name, f"Training relay {k}")
        for k, name in enumerate(("junction_1", "junction_2", "junction_3"), 1):
            single(name, EntityKind.WAYPOINT, f"Reed post {k}", dx=3.0)
        single("cache", EntityKind.CHEST, "Supply cache", dx=3.0)
        single("camp", EntityKind.CHEST, "Ferry-lift casing", dx=3.0)
        single("ferry_low", EntityKind.NPC, "Ferry mechanic", dx=-5.0)
        single("ferry_high", EntityKind.PILLAR, "Upper test lever", dx=-5.0, visual_role="vane")
        single("outpost", EntityKind.NPC, "Quartermaster", dx=3.0)
        relay("relay_1", "Field relay 1", version=2)
        relay("relay_2", "Field relay 2", version=2)
        single("ruin", EntityKind.CHEST, "Ruin vault", dx=3.0)
        bx, bz = self.world.place("beacon")[[0, 2]]
        w.spawn_at("beacon:core", EntityKind.BEACON, bx, bz, radius=2.1, label="Summit beacon",
                   visual_role="core", visual_version=2)

    # -------------------------------------------------------- observations
    def _observe(self):
        if self.render_enabled:
            obs = super()._observe()
        else:
            obs = {"rgb": np.zeros((self.cfg.render.height, self.cfg.render.width, 3), np.uint8),
                   "proprio": self._proprio(), "task": self._task_vector(), "party": self.party.as_vector(),
                   "instruction": self.quests.instruction()}
        obs["local_text"] = self.world.last_feedback
        return obs

    def _info(self, events=None):
        result = super()._info(events)
        p = self.world.preview
        result["route_preview"] = dict(milestone_index=p.milestone, checkpoint_index=p.checkpoint,
                                       facts=sorted(self.fullscale.state.facts))
        return result

    def state_digest(self):
        p = self.world.preview
        state = {"suite": SUITE_VERSION, "physics": PHYSICS_VERSION, "preview": PREVIEW_VERSION,
                 "world": self.world_digest, "position": self.char.pos.tolist(),
                 "velocity": self.char.vel.tolist(), "stamina": float(self.char.stamina),
                 "move_state": int(self.char.state), "time": self.world.elapsed,
                 "party": self.party.summary(), "facts": sorted(self.fullscale.state.facts),
                 "milestone": p.milestone, "checkpoint": p.checkpoint,
                 "lifts": [(l.id, l.position, l.riding) for l in self.world.lifts],
                 "camera": [float(self.cam.yaw), float(self.cam.pitch)]}
        return hashlib.sha256(json.dumps(state, sort_keys=True, default=float, allow_nan=False).encode()).hexdigest()
