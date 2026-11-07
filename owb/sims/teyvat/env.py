"""Gymnasium environment: a Genshin-like open world with a long questline.

Observation (Dict):
    rgb        (H, W, 3) uint8   third-person cel-shaded view
    proprio    (24,) float32     body state, stamina, time of day, ...
    task       (15,) float32     egocentric direction to the current objective
    party      (30,) float32     four padded party slots, shield, swap readiness
    instruction  Text            current quest line (only if `text_obs=True`)

Action (Dict by default, flat Box with `flat_actions=True`):
    move     (2,) in [-1, 1]  camera-relative stick: (strafe, forward)
    look     (2,) in [-1, 1]  camera yaw / pitch rate
    buttons  MultiBinary(7)   jump, sprint, interact, attack, climb, skill, burst
    swap     Discrete(N+1)    0 keeps current member; 1..N selects a party slot

Flat actions have 15 entries: move(2), look(2), buttons(7), swap requests(4).
Positive swap requests select the largest value (ties use the first slot).
Set `party=None` for the original five-button / nine-entry action space.

Reward is sparse quest progress, with optional distance shaping toward the
current objective. The episode ends when the whole questline is finished.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
import string

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("open-world-bench needs gymnasium: pip install gymnasium") from exc

from . import math3d as mth
from .camera import CameraConfig, OrbitCamera
from .character import CharacterConfig, CharacterController, Command, MoveState
from .events import Event
from .party import Party, PartyConfig
from .quests import ObjKind, QuestEngine, build_questline
from .render import RenderSettings, make_renderer
from .scatter import scatter_props
from .terrain import Terrain, TerrainConfig
from .world import World, WorldConfig

BUTTONS = ("jump", "sprint", "interact", "attack", "climb")
PARTY_BUTTONS = BUTTONS + ("skill", "burst")
_OBJ_KINDS = list(ObjKind)

# Island construction dominates reset cost, so cache it per configuration.
_TERRAIN_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_CACHE_SIZE = 4


def _get_island(tcfg: TerrainConfig, seed: int, density: float):
    key = (seed, tcfg.size, tcfg.resolution, tcfg.max_height, tcfg.shore_falloff,
           tcfg.terrace_step, tcfg.terrace_strength, round(float(density), 4))
    if key in _TERRAIN_CACHE:
        _TERRAIN_CACHE.move_to_end(key)
        return _TERRAIN_CACHE[key]
    terrain = Terrain(TerrainConfig(**{**tcfg.__dict__, "seed": seed}))
    props = scatter_props(terrain, seed, density)
    _TERRAIN_CACHE[key] = (terrain, props)
    while len(_TERRAIN_CACHE) > _CACHE_SIZE:
        _TERRAIN_CACHE.popitem(last=False)
    return terrain, props


@dataclass
class RewardConfig:
    objective: float = 1.0        # scale on the quest-defined objective rewards
    quest: float = 1.0            # scale on quest completion rewards
    shaping: float = 0.02         # per metre of progress toward the objective
    shaping_clip: float = 0.5
    time_penalty: float = 0.001
    damage_penalty: float = 0.02  # per hit point lost
    death_penalty: float = 5.0
    finish_bonus: float = 25.0
    discovery: float = 0.25       # one-off for each new interactable used


@dataclass
class EnvConfig:
    seed: int = 0
    world: WorldConfig = field(default_factory=WorldConfig)
    character: CharacterConfig = field(default_factory=CharacterConfig)
    party: PartyConfig | None = field(default_factory=PartyConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    render: RenderSettings = field(default_factory=lambda: RenderSettings(width=128, height=96, march_steps=72))
    reward: RewardConfig = field(default_factory=RewardConfig)
    dt: float = 1.0 / 30.0
    action_repeat: int = 2
    max_episode_steps: int = 12_000
    randomize_world_per_episode: bool = False   # new island every reset
    randomize_spawn: bool = False
    text_obs: bool = False
    flat_actions: bool = False
    backend: str = "auto"


class OpenWorldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, cfg: EnvConfig | None = None, render_mode: str | None = "rgb_array", **overrides):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        for k, v in overrides.items():
            if not hasattr(self.cfg, k):
                raise TypeError(f"unknown EnvConfig field {k!r}")
            setattr(self.cfg, k, v)
        self.render_mode = render_mode
        self._renderer = make_renderer(self.cfg.render, self.cfg.backend)
        self._episode_seed = int(self.cfg.seed)
        self._last_frame: np.ndarray | None = None

        rs = self.cfg.render
        obs: dict[str, spaces.Space] = {
            "rgb": spaces.Box(0, 255, (rs.height, rs.width, 3), dtype=np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, (24,), dtype=np.float32),
            "task": spaces.Box(-np.inf, np.inf, (15,), dtype=np.float32),
        }
        if self.cfg.text_obs:
            obs["instruction"] = spaces.Text(max_length=256, charset=string.printable)
        if self.cfg.party is not None:
            obs["party"] = spaces.Box(0.0, np.inf, (30,), dtype=np.float32)
        self.observation_space = spaces.Dict(obs)

        if self.cfg.flat_actions:
            self.action_space = spaces.Box(-1.0, 1.0, (15 if self.cfg.party else 9,), dtype=np.float32)
        else:
            actions = {
                "move": spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
                "look": spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
                "buttons": spaces.MultiBinary(len(PARTY_BUTTONS if self.cfg.party else BUTTONS)),
            }
            if self.cfg.party is not None:
                actions["swap"] = spaces.Discrete(len(self.cfg.party.members) + 1)
            self.action_space = spaces.Dict(actions)

        self.world: World | None = None
        self._build(self._episode_seed)

    # ------------------------------------------------------------------ setup
    def _build(self, seed: int) -> None:
        c = self.cfg
        wcfg = WorldConfig(**{**c.world.__dict__, "seed": seed})
        terrain, props = _get_island(wcfg.terrain, seed, wcfg.prop_density)
        self.party = Party(c.party) if c.party is not None else None
        self.world = World(wcfg, terrain=terrain, props=props, party=self.party)
        self.rng = np.random.default_rng(seed)
        self.graph, spawn = build_questline(self.world, self.rng)
        self.quests = QuestEngine(self.graph)
        char_cfg = replace(c.character)
        if self.party is not None:
            char_cfg.max_hp = self.party.max_hp
            char_cfg.stamina_regen *= self.party.stamina_regen_multiplier
            for cost in ("sprint_cost", "climb_cost", "glide_cost", "swim_cost", "jump_cost"):
                setattr(char_cfg, cost, getattr(char_cfg, cost) * self.party.stamina_multiplier)
        self.controller = CharacterController(self.world.terrain, self.world.props, char_cfg)
        self.camera_rig = OrbitCamera(self.world.terrain, self.world.props, c.camera)
        if c.randomize_spawn:
            p = self.world.terrain.sample_land(self.rng, 1, min_height=4.0, max_slope_deg=25.0)[0]
            spawn = self.world.ground_pos(float(p[0]), float(p[1]))
        from .architecture import populate_architecture
        architecture = populate_architecture(self.world, spawn)
        self.controller.architecture = self.camera_rig.architecture = architecture
        self.char = self.controller.spawn(float(spawn[0]), float(spawn[2]))
        self.cam = self.camera_rig.reset(self.char)
        self.steps = 0
        self._prev_dist: float | None = None
        self._used_entities: set[str] = set()
        self._episode_reward = 0.0
        self._last_events: list[Event] = []
        self._last_frame = None

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._episode_seed = int(seed)
        elif self.cfg.randomize_world_per_episode:
            self._episode_seed = int(np.random.SeedSequence().generate_state(1)[0] % (2 ** 31))
        super().reset(seed=self._episode_seed)
        self._build(self._episode_seed)
        obs = self._observe()
        return obs, self._info()

    # ------------------------------------------------------------------- step
    def _decode(self, action) -> Command:
        swap = -1
        if self.cfg.flat_actions:
            a = np.asarray(action, dtype=np.float32).reshape(-1)
            sizes = (9, 15) if self.party else (9,)
            if a.size not in sizes or not np.isfinite(a).all():
                raise ValueError(f"flat action must have {sizes} finite entries")
            move, look, btn = a[0:2], a[2:4], a[4:11 if a.size == 15 else 9] > 0.0
            if a.size == 15:
                requests = a[11:11 + len(self.party.members)]
                if requests.max() > 0.0:
                    swap = int(np.argmax(requests))
        else:
            move = np.asarray(action["move"], dtype=np.float32)
            look = np.asarray(action["look"], dtype=np.float32)
            btn = np.asarray(action["buttons"]).reshape(-1)
            sizes = (5, 7) if self.party else (5,)
            if btn.size not in sizes or not np.isin(btn, (0, 1)).all():
                raise ValueError(f"buttons must contain {sizes} binary entries")
            if self.party is not None:
                requested = action.get("swap", 0)
                if not self.action_space["swap"].contains(requested):
                    raise ValueError("swap must be 0 (keep) or a one-based party slot")
                swap = int(requested) - 1
        if move.shape != (2,) or look.shape != (2,) or not np.isfinite(np.concatenate([move, look])).all():
            raise ValueError("move and look must be finite two-element vectors")
        cmd = Command(move=np.clip(move, -1.0, 1.0),
                      jump=bool(btn[0]), sprint=bool(btn[1]),
                      interact=bool(btn[2]), attack=bool(btn[3]), climb=bool(btn[4]),
                      skill=bool(btn[5]) if len(btn) > 5 else False,
                      burst=bool(btn[6]) if len(btn) > 6 else False, swap=swap)
        self._look = np.clip(look, -1.0, 1.0)
        return cmd

    def step(self, action):
        cmd = self._decode(action)
        c = self.cfg
        rc = c.reward
        events: list[Event] = []
        if cmd.swap >= 0:
            events += self.world.swap_character(self.char, cmd.swap)
        damage_taken = 0.0

        for k in range(max(1, c.action_repeat)):
            if self.char.hp <= 0.0:
                break
            if self.party is not None:
                self.controller.cfg.max_hp = self.party.max_hp
            # Interact / attack fire once per agent step, not once per substep.
            sub = cmd if k == 0 else Command(move=cmd.move, jump=cmd.jump, sprint=cmd.sprint, climb=cmd.climb)
            hp_before = self.char.hp
            move_events = self.controller.step(self.char, sub, self.cam.yaw, c.dt)
            damage_taken += max(0.0, hp_before - self.char.hp)
            events += [Event(f"move_{m}") for m in move_events]
            world_events = self.world.update(self.char, c.dt, self.controller.cfg)
            damage_taken += sum(e.get("damage", 0.0) for e in world_events if e.type == "player_hit")
            events += world_events
            if self.char.hp <= 0.0:
                break
            if k == 0 and cmd.interact:
                events += self.world.interact(self.char)
            if k == 0 and cmd.attack:
                events += self.world.attack(self.char)
            if k == 0 and cmd.skill:
                events += self.world.combat.cast(self.char, self.controller)
            if k == 0 and cmd.burst:
                events += self.world.combat.cast(self.char, self.controller, burst=True)
            self.camera_rig.update(self.cam, self.char, self._look, c.dt)

        if self.party is not None:
            self.party.active.hp = self.char.hp
            self.controller.cfg.max_hp = self.party.max_hp

        self.steps += 1
        ws = self.world.state_dict(self.char)
        quest_reward = self.quests.consume(events, ws)

        reward = quest_reward * rc.objective
        reward -= rc.time_penalty
        reward -= damage_taken * rc.damage_penalty
        if any(e.type == "player_defeated" for e in events):
            reward -= rc.death_penalty
        for e in events:
            if e.type in ("open_chest", "activate", "attune", "collect") and e.get("entity"):
                if e.get("entity") not in self._used_entities:
                    self._used_entities.add(e.get("entity"))
                    reward += rc.discovery

        reward += self._shaping_reward()

        terminated = bool(self.quests.all_complete or self.char.hp <= 0.0)
        if self.quests.all_complete:
            reward += rc.finish_bonus
        truncated = self.steps >= c.max_episode_steps

        self._last_events = events
        self._episode_reward += float(reward)
        obs = self._observe()
        return obs, float(reward), terminated, truncated, self._info(events)

    def _shaping_reward(self) -> float:
        rc = self.cfg.reward
        if rc.shaping <= 0.0:
            return 0.0
        target = self.quests.focus_target()
        if target is None:
            self._prev_dist = None
            return 0.0
        dist = float(np.linalg.norm((self.char.pos - target)[[0, 2]]))
        if self._prev_dist is None:
            self._prev_dist = dist
            return 0.0
        delta = self._prev_dist - dist
        self._prev_dist = dist
        return float(np.clip(delta * rc.shaping, -rc.shaping_clip, rc.shaping_clip))

    # ------------------------------------------------------------ observation
    def _observe(self) -> dict:
        frame = self._renderer.render(self.world, self.char, self.cam, fov_y=self.cfg.camera.fov_y)
        self._last_frame = frame
        out = {"rgb": frame, "proprio": self._proprio(), "task": self._task_vector()}
        if self.party is not None:
            out["party"] = self.party.as_vector()
        if self.cfg.text_obs:
            out["instruction"] = self.quests.instruction()
        return out

    def _proprio(self) -> np.ndarray:
        ch, w = self.char, self.world
        cy, sy = np.cos(-self.cam.yaw), np.sin(-self.cam.yaw)
        vx, vz = float(ch.vel[0]), float(ch.vel[2])
        # Velocity expressed in camera frame, so the policy can relate it to pixels.
        fwd_v = vx * sy + vz * cy
        side_v = vx * cy - vz * sy
        ground = float(w.terrain.height_at(ch.pos[0], ch.pos[2]))
        dyaw = (ch.yaw - self.cam.yaw + np.pi) % (2 * np.pi) - np.pi
        tod = w.time_of_day * 2.0 * np.pi
        hint = 1.0 if w.last_interact_hint else 0.0
        v = [
            side_v / 10.0, float(ch.vel[1]) / 10.0, fwd_v / 10.0,
            float(np.linalg.norm(ch.vel[[0, 2]])) / 10.0,
            ch.stamina / max(self.cfg.character.stamina_max, 1e-6),
            ch.hp / max(self.controller.cfg.max_hp, 1e-6),
            float(ch.state == MoveState.GROUNDED), float(ch.state == MoveState.AIRBORNE),
            float(ch.state == MoveState.GLIDING), float(ch.state == MoveState.CLIMBING),
            float(ch.state == MoveState.SWIMMING),
            float(ch.pos[1] - ground) / 20.0, float(ch.pos[1]) / 100.0,
            np.sin(dyaw), np.cos(dyaw), np.sin(tod), np.cos(tod), w.daylight,
            float(w.terrain.slope_at(ch.pos[0], ch.pos[2])) / (np.pi * 0.5),
            float(w.terrain.water_depth(ch.pos[0], ch.pos[2])) / 10.0,
            np.sin(ch.yaw), np.cos(ch.yaw), hint,
            self.steps / max(self.cfg.max_episode_steps, 1),
        ]
        return np.asarray(v, dtype=np.float32)

    def _task_vector(self) -> np.ndarray:
        out = np.zeros(15, dtype=np.float32)
        focus = self.quests.focus()
        if focus is None:
            return out
        quest, obj = focus
        kind_idx = _OBJ_KINDS.index(obj.kind)
        out[5 + kind_idx] = 1.0
        target = self.quests.focus_target()
        if target is None:
            return out
        d = target - self.char.pos
        dist = float(np.linalg.norm(d[[0, 2]]))
        # Direction to the goal, rotated into the camera frame.
        cy, sy = np.cos(-self.cam.yaw), np.sin(-self.cam.yaw)
        dirn = mth.norm(d)
        out[0] = dirn[0] * cy - dirn[2] * sy
        out[1] = dirn[1]
        out[2] = dirn[0] * sy + dirn[2] * cy
        out[3] = float(np.log1p(dist) / 7.0)
        out[4] = 1.0
        return out

    # ------------------------------------------------------------------- info
    def _info(self, events: list[Event] | None = None) -> dict:
        snap = self.quests.snapshot()
        return {
            "instruction": snap["instruction"],
            "quest_completion": snap["completion"],
            "quests_done": snap["completed_quests"],
            "objectives_done": snap["completed_objectives"],
            "inventory": snap["inventory"],
            "active_quests": snap["active"],
            "position": self.char.pos.copy(),
            "move_state": self.char.state.name,
            "stamina": float(self.char.stamina),
            "hp": float(self.char.hp),
            "time_of_day": float(self.world.time_of_day),
            "time_label": self.world.time_label(),
            "interact_hint": self.world.last_interact_hint,
            "events": [e.type for e in (events or [])],
            "event_details": [{"type": e.type, **{
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in e.payload.items()
            }} for e in (events or [])],
            "party": self.party.summary() if self.party else [],
            "active_character": self.party.active.spec.id if self.party else None,
            "episode_reward": self._episode_reward,
            "seed": self._episode_seed,
        }

    # ----------------------------------------------------------------- render
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._last_frame is None:
            self._last_frame = self._renderer.render(self.world, self.char, self.cam,
                                                     fov_y=self.cfg.camera.fov_y)
        return self._last_frame

    def render_hd(self, width: int = 960, height: int = 540, hud: bool = True,
                  *, renderer: str = "cycles", blender=None, samples: int = 48,
                  threads: int = 8) -> np.ndarray:
        """A Cycles presentation frame; CPU output requires renderer='cpu'."""
        from .render.hud import draw_hud
        from owb.render.presentation import PresentationRenderer
        with PresentationRenderer(backend=renderer, width=width, height=height,
                                  blender=blender, samples=samples, threads=threads) as presentation:
            img = presentation.render(self)
        if hud:
            img = draw_hud(img, self)
        return img

    def close(self) -> None:
        self._renderer.close()

    # ------------------------------------------------------------- utilities
    @property
    def instruction(self) -> str:
        return self.quests.instruction()

    def objective_target(self) -> np.ndarray | None:
        return self.quests.focus_target()


def make_env(**kwargs) -> OpenWorldEnv:
    """Convenience constructor: `make_env(seed=3, text_obs=True)`."""
    render_mode = kwargs.pop("render_mode", "rgb_array")
    return OpenWorldEnv(EnvConfig(**kwargs) if kwargs else None, render_mode=render_mode)


def _make_small(render_mode: str | None = "rgb_array", **kwargs):
    """A cheaper variant: smaller island, 64x48 frames, shorter episodes."""
    cfg = EnvConfig(
        world=WorldConfig(terrain=TerrainConfig(size=512.0, resolution=257, max_height=150.0),
                          prop_density=1.6),
        render=RenderSettings(width=64, height=48, march_steps=56, shadow_steps=0,
                              prop_draw_distance=140.0),
        max_episode_steps=4000,
    )
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return OpenWorldEnv(cfg, render_mode=render_mode)
