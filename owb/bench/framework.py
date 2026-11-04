"""Versioned agent interface shared by the independent OWB simulators.

Adapters translate controls and observations; native simulation code, rewards,
and task definitions remain owned by each environment.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
import string
from typing import Callable

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..env import INVENTORY_COLUMNS, SURVIVAL_NAMES

SCHEMA_VERSION = 1
BUTTONS = ("jump", "sprint", "climb", "glide", "dive", "crouch")
COMMANDS = ("none", "interact", "attack", "skill", "burst", "eat", "cook", "equip", "stow")
MODALITIES = ("party", "inventory", "survival", "minimap", "goal")
PROPRIO_NAMES = (
    "velocity_forward", "velocity_right", "velocity_up", "health_fraction", "stamina_fraction",
    "grounded", "airborne", "swimming", "climbing", "gliding", "day_sin", "day_cos",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 0
    variant: str = "small"
    image_size: tuple[int, int] = (96, 72)  # width, height
    max_steps: int = 3000
    decision_seconds: float = 0.2
    include_goal_hint: bool = False
    render_quality: str = "fast"

    def __post_init__(self):
        if self.variant not in ("small", "full"):
            raise ValueError("variant must be 'small' or 'full'")
        if self.render_quality not in ("fast", "presentation"):
            raise ValueError("render_quality must be 'fast' or 'presentation'")
        if (len(self.image_size) != 2 or any(not isinstance(n, int) or n <= 0 for n in self.image_size)
                or not isinstance(self.max_steps, int) or self.max_steps <= 0
                or not math.isfinite(self.decision_seconds) or self.decision_seconds <= 0):
            raise ValueError("image dimensions, max_steps, and decision_seconds must be positive")


def jsonable(value):
    """Detach simulator values and convert them to JSON-compatible data."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return deepcopy(value)


def neutral_action() -> dict:
    """A new, valid no-op action in the common schema."""
    return dict(move=np.zeros(2, np.float32), look=np.zeros(2, np.float32),
                buttons=np.zeros(len(BUTTONS), np.int8), command=0, swap=0, slot=0,
                ingredients=np.zeros(5, np.int64))


def _repeat(seconds, dt):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("native physics timestep must be positive")
    count = round(seconds / dt)
    if count < 1 or not math.isclose(count * dt, seconds, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f"decision_seconds={seconds} must be a positive multiple of native dt={dt}")
    return count


class NativeAdapter:
    """Implement this contract to add another simulator to the common pipeline."""
    commands = ("none", "interact", "attack")
    buttons = BUTTONS
    modalities: tuple[str, ...] = ()

    def __init__(self, native, config):
        self.native, self.config = native, config

    def translate(self, action):
        raise NotImplementedError

    def state(self):
        """Return velocity, view forward/right, health/cap, stamina/cap,
        grounded, airborne, swimming, climbing, gliding, and time of day."""
        raise NotImplementedError

    def project(self, observation):
        """Return common observation fields supplied by this backend."""
        raise NotImplementedError

    def metrics(self, info):
        """Return success (bool or None), task progress, and native metrics."""
        raise NotImplementedError

    def elapsed(self):
        raise NotImplementedError

    def events(self, info):
        return info.get("events", [])

    def swap_mask(self):
        return np.array([1, 0, 0, 0, 0], np.int8)


class TeyvatAdapter(NativeAdapter):
    buttons = BUTTONS[:-1]

    def __init__(self, native, config):
        super().__init__(native, config)
        self.commands = ("none", "interact", "attack") + (("skill", "burst") if native.party else ())
        self.modalities = (("party",) if native.party else ()) + (("goal",) if config.include_goal_hint else ())

    def translate(self, action):
        from open_world_bench.character import MoveState

        command = COMMANDS[int(action["command"])]
        jump, sprint, climb, glide, dive, _ = action["buttons"]
        state = self.native.char.state
        jump = bool(jump or (glide and state in (MoveState.AIRBORNE, MoveState.GLIDING))
                    or (dive and state == MoveState.SWIMMING))
        buttons = [jump, sprint, command == "interact", command == "attack", climb]
        if self.native.party:
            buttons += [command == "skill", command == "burst"]
        out = dict(move=np.asarray(action["move"])[::-1].copy(),
                   look=np.asarray(action["look"]) * np.array([1, -1], np.float32),
                   buttons=np.asarray(buttons, np.int8))
        if self.native.party:
            out["swap"] = int(action["swap"])
        return out

    def state(self):
        from open_world_bench.character import MoveState

        env, ch = self.native, self.native.char
        yaw = env.cam.yaw
        forward = np.array([np.sin(yaw), 0, -np.cos(yaw)])
        right = np.array([np.cos(yaw), 0, np.sin(yaw)])
        return (ch.vel, forward, right, ch.hp, env.controller.cfg.max_hp,
                ch.stamina, env.controller.cfg.stamina_max,
                ch.state == MoveState.GROUNDED, ch.state == MoveState.AIRBORNE,
                ch.state == MoveState.SWIMMING, ch.state == MoveState.CLIMBING,
                ch.state == MoveState.GLIDING, env.world.time_of_day)

    def project(self, observation):
        out = dict(instruction=self.native.instruction)
        if self.native.party:
            out["party"] = observation["party"]
        if self.config.include_goal_hint:
            out["goal"] = observation["task"]
        return out

    def metrics(self, info):
        return bool(self.native.quests.all_complete), info["quest_completion"], {
            key: info[key] for key in ("quests_done", "objectives_done")}

    def elapsed(self):
        return self.native.world.elapsed

    def events(self, info):
        return info.get("event_details", [])

    def swap_mask(self):
        mask = super().swap_mask()
        if self.native.party:
            mask[1:1 + len(self.native.party.members)] = 1
        return mask


class WildlandsAdapter(NativeAdapter):
    commands = ("none", "interact", "attack", "eat", "cook", "equip", "stow")
    modalities = ("inventory", "survival", "minimap")

    def translate(self, action):
        from ..env import COMMANDS as NATIVE_COMMANDS

        jump, sprint, climb, glide, dive, crouch = action["buttons"]
        return dict(move=np.asarray(action["move"]).copy(), look=np.asarray(action["look"]).copy(),
                    buttons=np.array([jump, sprint, climb, glide, bool(dive or crouch)], np.int8),
                    command=NATIVE_COMMANDS.index(COMMANDS[int(action["command"])]),
                    slot=int(action["slot"]), ingredients=np.asarray(action["ingredients"]).copy())

    def state(self):
        a = self.native.agent
        forward = np.array([np.cos(a.yaw), 0, np.sin(a.yaw)])
        right = np.array([-np.sin(a.yaw), 0, np.cos(a.yaw)])
        airborne = not (a.on_ground or a.in_water or a.climbing or a.gliding)
        return (a.vel, forward, right, a.health, a.max_health, a.stamina, a.stamina_cap,
                a.on_ground, airborne, a.in_water, a.climbing, a.gliding,
                self.native.elements.climate.time)

    def project(self, observation):
        return dict(instruction=observation["text"], inventory=observation["inventory"],
                    survival=observation["survival"], minimap=observation["minimap"])

    def metrics(self, info):
        # Survival has no task victory predicate yet. Do not label survival success.
        return None, None, {key: info[key] for key in ("visited_cells", "discovered_landmarks")}

    def elapsed(self):
        return self.native.elapsed


def _teyvat(config, native_config):
    from open_world_bench import EnvConfig, OpenWorldEnv, TerrainConfig, WorldConfig, RenderSettings

    if native_config is not None and not isinstance(native_config, EnvConfig):
        raise TypeError("teyvat native_config must be open_world_bench.EnvConfig")
    cfg = deepcopy(native_config) if native_config is not None else EnvConfig()
    if native_config is None and config.variant == "small":
        cfg.world = WorldConfig(terrain=TerrainConfig(size=512, resolution=257, max_height=150), prop_density=1.6)
        cfg.render = RenderSettings(march_steps=56, shadow_steps=0, prop_draw_distance=140)
    cfg.render = replace(cfg.render, width=config.image_size[0], height=config.image_size[1])
    if config.render_quality == "presentation":
        cfg.render = replace(cfg.render, march_steps=144, refine_steps=8, shadow_steps=6,
                             supersample=2, prop_draw_distance=280)
    cfg.seed, cfg.max_episode_steps = config.seed, config.max_steps
    cfg.action_repeat = _repeat(config.decision_seconds, cfg.dt)
    cfg.flat_actions, cfg.text_obs, cfg.randomize_world_per_episode = False, True, False
    return OpenWorldEnv(cfg)


def _wildlands(config, native_config):
    from ..core.config import EnvConfig, WorldConfig, RenderConfig
    from ..env import WildlandsEnv

    if native_config is not None and not isinstance(native_config, EnvConfig):
        raise TypeError("wildlands native_config must be owb.EnvConfig")
    cfg = deepcopy(native_config) if native_config is not None else EnvConfig()
    if native_config is None and config.variant == "small":
        cfg.world_cfg = WorldConfig(size_x=96, size_z=96, size_y=64, relief=16, mountain_relief=18)
        cfg.render_cfg = RenderConfig(max_dist=72)
    cfg.render_cfg = replace(cfg.render_cfg, width=config.image_size[0], height=config.image_size[1])
    if config.render_quality == "presentation":
        cfg.render_cfg = replace(cfg.render_cfg, supersample=2, shadows=True)
    cfg.seed, cfg.max_steps = config.seed, config.max_steps
    cfg.frame_skip = _repeat(config.decision_seconds, cfg.sim_cfg.dt)
    cfg.action_space = "multi"
    cfg.obs_rgb = cfg.obs_inventory = cfg.obs_minimap = cfg.obs_text = True
    cfg.obs_depth = cfg.obs_semantic = False
    cfg.minimap_size = 32
    return WildlandsEnv(cfg)


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    factory: Callable | None
    adapter: type[NativeAdapter] | None
    description: str

    @property
    def implemented(self):
        return self.factory is not None and self.adapter is not None


_REGISTRY = {
    "teyvat": EnvironmentSpec("teyvat", _teyvat, TeyvatAdapter, "Party combat, traversal, and island quests"),
    "wildlands": EnvironmentSpec("wildlands", _wildlands, WildlandsAdapter, "Voxel exploration, weather, and survival"),
    "skylands": EnvironmentSpec("skylands", None, None, "Planned: vertical worlds and construction"),
}


def environment_specs() -> tuple[EnvironmentSpec, ...]:
    return tuple(_REGISTRY.values())


def available_environments() -> tuple[str, ...]:
    return tuple(name for name, spec in _REGISTRY.items() if spec.implemented)


def register_environment(spec: EnvironmentSpec) -> None:
    """Add a backend, or supply an implementation for a planned entry."""
    if spec.name in _REGISTRY and _REGISTRY[spec.name].implemented:
        raise ValueError(f"environment {spec.name!r} is already implemented")
    if not spec.implemented or not issubclass(spec.adapter, NativeAdapter):
        raise ValueError("registration requires a native factory and a NativeAdapter subclass")
    _REGISTRY[spec.name] = spec


def make_env(environment: str, *, config: BenchmarkConfig | None = None,
             native_config=None, **overrides) -> "BenchmarkEnv":
    """Construct a common-schema environment without changing either native API."""
    if environment not in _REGISTRY:
        raise ValueError(f"unknown environment {environment!r}; choose from {tuple(_REGISTRY)}")
    spec = _REGISTRY[environment]
    if not spec.implemented:
        raise NotImplementedError(f"{environment} is planned but has no runnable backend yet")
    cfg = replace(config or BenchmarkConfig(), **overrides)
    native = spec.factory(cfg, native_config)
    try:
        return BenchmarkEnv(spec.name, native, spec.adapter(native, cfg), cfg)
    except Exception:
        native.close()
        raise


class BenchmarkEnv(gym.Env):
    """Identical spaces and episode bookkeeping across supported environments."""
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, name, native, adapter, config):
        super().__init__()
        self.name, self.native, self.adapter, self.config = name, native, adapter, config
        self.render_mode = "rgb_array"
        self.metadata = dict(self.metadata, render_fps=max(1, round(1 / config.decision_seconds)))
        self.action_space = spaces.Dict({
            "move": spaces.Box(-1, 1, (2,), np.float32),  # forward, right
            "look": spaces.Box(-1, 1, (2,), np.float32),  # right, up
            "buttons": spaces.MultiBinary(len(BUTTONS)),
            "command": spaces.Discrete(len(COMMANDS)),
            "swap": spaces.Discrete(5), "slot": spaces.Discrete(32),
            "ingredients": spaces.MultiDiscrete([33] * 5),
        })
        width, height = config.image_size
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(0, 255, (height, width, 3), np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, (len(PROPRIO_NAMES),), np.float32),
            "instruction": spaces.Text(max_length=1024, charset=string.printable),
            "party": spaces.Box(0, np.inf, (30,), np.float32),
            "inventory": spaces.Box(0, np.inf, (32, len(INVENTORY_COLUMNS)), np.float32),
            "survival": spaces.Box(-np.inf, np.inf, (len(SURVIVAL_NAMES),), np.float32),
            "minimap": spaces.Box(0, 255, (32, 32), np.uint8),
            "goal": spaces.Box(-np.inf, np.inf, (15,), np.float32),
            "modality_mask": spaces.MultiBinary(len(MODALITIES)),
            "command_mask": spaces.MultiBinary(len(COMMANDS)),
            "button_mask": spaces.MultiBinary(len(BUTTONS)),
            "swap_mask": spaces.MultiBinary(5),
        })
        self._needs_reset, self._seed = True, config.seed
        self.steps, self.episode_return = 0, 0.0

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("common reset options are not supported")
        self._needs_reset = True
        self._seed = self._seed if seed is None else int(seed)
        super().reset(seed=self._seed)
        self.action_space.seed(self._seed)
        observation, info = self.native.reset(seed=self._seed)
        self.steps, self.episode_return = 0, 0.0
        out = self._observation(observation)
        self._needs_reset = False
        return out, self._info(info, [], False, False)

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("call reset() before starting or after an episode ends")
        if not self.action_space.contains(action):
            raise ValueError("action is outside the common action_space")
        action = deepcopy(action)
        unavailable = []
        command = COMMANDS[int(action["command"])]
        if command not in self.adapter.commands:
            unavailable.append(dict(type="action_unavailable", command=command))
            action["command"] = 0
        for i, button in enumerate(BUTTONS):
            if action["buttons"][i] and button not in self.adapter.buttons:
                unavailable.append(dict(type="action_unavailable", button=button))
                action["buttons"][i] = 0
        if not self.adapter.swap_mask()[int(action["swap"])]:
            unavailable.append(dict(type="action_unavailable", swap=int(action["swap"])))
            action["swap"] = 0
        observation, reward, terminated, truncated, info = self.native.step(self.adapter.translate(action))
        self.steps += 1
        self.episode_return += float(reward)
        terminated = bool(terminated)
        truncated = bool((truncated or self.steps >= self.config.max_steps) and not terminated)
        self._needs_reset = terminated or truncated
        events = unavailable + self.adapter.events(info)
        return self._observation(observation), float(reward), terminated, truncated, self._info(
            info, events, terminated, truncated)

    def _observation(self, native_obs):
        out = {key: np.zeros(space.shape, dtype=space.dtype)
               for key, space in self.observation_space.spaces.items() if key != "instruction"}
        out["rgb"] = np.array(native_obs["rgb"], copy=True)
        out.update({key: deepcopy(value) for key, value in self.adapter.project(native_obs).items()})
        velocity, forward, right, hp, hp_max, stamina, stamina_max, *rest = self.adapter.state()
        grounded, airborne, swimming, climbing, gliding, tod = rest
        out["proprio"] = np.array([
            np.dot(velocity, forward), np.dot(velocity, right), velocity[1],
            hp / max(hp_max, 1e-6), stamina / max(stamina_max, 1e-6),
            grounded, airborne, swimming, climbing, gliding,
            np.sin(tod * 2 * np.pi), np.cos(tod * 2 * np.pi),
        ], np.float32)
        out["modality_mask"] = np.array([name in self.adapter.modalities for name in MODALITIES], np.int8)
        out["command_mask"] = np.array([name in self.adapter.commands for name in COMMANDS], np.int8)
        out["button_mask"] = np.array([name in self.adapter.buttons for name in BUTTONS], np.int8)
        out["swap_mask"] = self.adapter.swap_mask()
        return out

    def _info(self, native_info, events, terminated, truncated):
        success, progress, metrics = self.adapter.metrics(native_info)
        _, _, _, hp, _, stamina, *_ = self.adapter.state()
        reason = ("success" if success else "death") if terminated else ("time_limit" if truncated else None)
        info = dict(schema_version=SCHEMA_VERSION, environment=self.name, seed=self._seed,
                    steps=self.steps, simulated_seconds=float(self.adapter.elapsed()),
                    decision_seconds=self.config.decision_seconds, episode_return=self.episode_return,
                    health=float(hp), stamina=float(stamina), success=success, task_progress=progress,
                    success_defined=success is not None, task_progress_defined=progress is not None,
                    terminated=terminated, truncated=truncated, termination_reason=reason,
                    events=jsonable(events), metrics=jsonable(metrics), native=jsonable(native_info))
        if terminated or truncated:
            info["episode"] = {key: info[key] for key in (
                "environment", "seed", "steps", "simulated_seconds", "episode_return", "success",
                "task_progress", "success_defined", "task_progress_defined",
                "terminated", "truncated", "termination_reason")}
        return info

    def render(self):
        if self.steps == 0 and self._needs_reset:
            raise gym.error.ResetNeeded("call reset() before render()")
        return np.array(self.native.render(), copy=True)

    def close(self):
        self.native.close()
        self._needs_reset = True
