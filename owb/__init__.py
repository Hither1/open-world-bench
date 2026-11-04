"""Open-World Bench: common pipeline plus native Wildlands APIs."""

from gymnasium.envs.registration import register, registry

from .core.config import EnvConfig, RenderConfig, SimConfig, WorldConfig
from .env import WildlandsEnv


def register_envs():
    for name, entry in (("OpenWorldBench/Wildlands-v0", "owb.env:WildlandsEnv"),
                        ("OpenWorldBench/Wildlands-Small-v0", "owb.env:make_small"),
                        ("OpenWorldBench/Genshin-v0", "open_world_bench.env:OpenWorldEnv"),
                        ("OpenWorldBench/Genshin-Small-v0", "open_world_bench.env:_make_small")):
        if name not in registry:
            register(id=name, entry_point=entry)


register_envs()

from .bench import BenchmarkConfig, available_environments, make_env

__all__ = ["WildlandsEnv", "EnvConfig", "WorldConfig", "SimConfig", "RenderConfig", "register_envs",
           "BenchmarkConfig", "available_environments", "make_env"]
