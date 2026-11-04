"""Wildlands: a headless open-world exploration and survival environment."""

from gymnasium.envs.registration import register, registry

from .core.config import EnvConfig, RenderConfig, SimConfig, WorldConfig
from .env import WildlandsEnv


def register_envs():
    for name, entry in (("OpenWorldBench/Wildlands-v0", "owb.env:WildlandsEnv"),
                        ("OpenWorldBench/Wildlands-Small-v0", "owb.env:make_small")):
        if name not in registry:
            register(id=name, entry_point=entry)


register_envs()

__all__ = ["WildlandsEnv", "EnvConfig", "WorldConfig", "SimConfig", "RenderConfig", "register_envs"]
