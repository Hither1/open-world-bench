"""open-world-bench: an anime-styled 3D open world for long-horizon agents."""
from __future__ import annotations

__version__ = "0.1.0"

from .camera import CameraConfig, CameraState, OrbitCamera
from .character import CharacterConfig, CharacterController, CharacterState, Command, MoveState
from .env import EnvConfig, OpenWorldEnv, RewardConfig, make_env
from .events import Event
from .elements import Element, ElementalStatus, Reaction
from .party import Party, PartyConfig, ROSTER
from .quests import QuestEngine, QuestGraph, build_questline
from .render import RenderSettings, make_renderer
from .terrain import Terrain, TerrainConfig
from .world import Entity, EntityKind, World, WorldConfig

__all__ = [
    "CameraConfig", "CameraState", "OrbitCamera",
    "CharacterConfig", "CharacterController", "CharacterState", "Command", "MoveState",
    "EnvConfig", "OpenWorldEnv", "RewardConfig", "make_env", "Event",
    "QuestEngine", "QuestGraph", "build_questline",
    "RenderSettings", "make_renderer", "Terrain", "TerrainConfig",
    "Entity", "EntityKind", "World", "WorldConfig", "register_envs",
    "Element", "ElementalStatus", "Reaction", "Party", "PartyConfig", "ROSTER",
]


def register_envs() -> None:
    """Register the gymnasium ids: OpenWorldBench/Genshin-v0 and -Small-v0."""
    from gymnasium.envs.registration import register, registry

    if "OpenWorldBench/Genshin-v0" in registry:
        return
    register(id="OpenWorldBench/Genshin-v0", entry_point="open_world_bench.env:OpenWorldEnv",
             max_episode_steps=None)
    register(id="OpenWorldBench/Genshin-Small-v0",
             entry_point="open_world_bench.env:_make_small", max_episode_steps=None)


def _unused() -> None:  # pragma: no cover
    pass


try:  # registering on import keeps `gym.make` working without ceremony
    register_envs()
except Exception:  # pragma: no cover - gymnasium missing or already registered
    pass
