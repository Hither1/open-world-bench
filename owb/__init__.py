"""Open-World Bench: common pipeline plus native Wildlands APIs."""

from gymnasium.envs.registration import register, registry

from . import _compat

_compat.install()

from .sims.voxel.core.config import EnvConfig, RenderConfig, SimConfig, WorldConfig
from .sims.voxel.env import WildlandsEnv, SkylandsEnv


def register_envs():
    for name, entry in (("OpenWorldBench/Wildlands-v0", "owb.sims.voxel.env:WildlandsEnv"),
                        ("OpenWorldBench/Wildlands-Small-v0", "owb.sims.voxel.env:make_small"),
                        ("OpenWorldBench/Skylands-v0", "owb.sims.voxel.env:SkylandsEnv"),
                        ("OpenWorldBench/Skylands-Small-v0", "owb.sims.voxel.env.skylands:make_small"),
                        ("OpenWorldBench/Skylands-Tempest-v0", "owb.sims.voxel.env.skylands_tempest:SkylandsTempestEnv"),
                        ("OpenWorldBench/Skylands-Tempest-Extended-v0", "owb.sims.voxel.env.skylands_tempest:make_extended"),
                        ("OpenWorldBench/Genshin-v0", "owb.sims.teyvat.env:OpenWorldEnv"),
                        ("OpenWorldBench/Genshin-Small-v0", "owb.sims.teyvat.env:_make_small")):
        if name not in registry:
            register(id=name, entry_point=entry)


register_envs()

from .bench import BenchmarkConfig, available_environments, make_env

__all__ = ["WildlandsEnv", "SkylandsEnv", "EnvConfig", "WorldConfig", "SimConfig", "RenderConfig", "register_envs",
           "BenchmarkConfig", "available_environments", "make_env"]


def __getattr__(name):
    # ``import owb; owb.env.WildlandsEnv`` used to work because owb imported these eagerly.
    if f"owb.{name}" in _compat.ALIASES:
        import importlib
        return importlib.import_module(f"owb.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
