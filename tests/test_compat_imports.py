"""Deprecated import paths resolve to the canonical owb.sims modules."""
import importlib
import subprocess
import sys
import warnings

import gymnasium as gym
import pytest

import owb

OLD_TO_NEW = [
    ("open_world_bench.env", "owb.sims.teyvat.env"),
    ("open_world_bench.reference_tasks", "owb.sims.teyvat.reference_tasks"),
    ("open_world_bench.reference_policy", "owb.sims.teyvat.reference_policy"),
    ("open_world_bench.render.meshes", "owb.sims.teyvat.render.meshes"),
    ("open_world_bench.quests.schema", "owb.sims.teyvat.quests.schema"),
    ("open_world_bench.architecture", "owb.sims.teyvat.architecture"),
    ("open_world_bench.campaign_tasks", "owb.sims.teyvat.campaign_tasks"),
    ("open_world_bench.render.architecture", "owb.sims.teyvat.render.architecture"),
    ("owb.core.blocks", "owb.sims.voxel.core.blocks"),
    ("owb.core.config", "owb.sims.voxel.core.config"),
    ("owb.world.worldgen", "owb.sims.voxel.world.worldgen"),
    ("owb.physics.collide", "owb.sims.voxel.physics.collide"),
    ("owb.sim.construction", "owb.sims.voxel.sim.construction"),
    ("owb.entities.agent", "owb.sims.voxel.entities.agent"),
    ("owb.env", "owb.sims.voxel.env"),
    ("owb.env.wildlands_reference", "owb.sims.voxel.env.wildlands_reference"),
    ("owb.env.skylands_reference", "owb.sims.voxel.env.skylands_reference"),
    ("owb.env.skylands_campaign", "owb.sims.voxel.env.skylands_campaign"),
    ("owb.env.skylands_tempest", "owb.sims.voxel.env.skylands_tempest"),
    ("owb.world.tempest", "owb.sims.voxel.world.tempest"),
    ("owb.render.landscape", "owb.sims.voxel.render.landscape"),
    ("owb.render.raycast", "owb.sims.voxel.render.raycast"),
]

GYM_IDS = ["OpenWorldBench/Wildlands-v0", "OpenWorldBench/Wildlands-Small-v0",
           "OpenWorldBench/Skylands-v0", "OpenWorldBench/Skylands-Small-v0",
           "OpenWorldBench/Skylands-Tempest-v0", "OpenWorldBench/Skylands-Tempest-Extended-v0",
           "OpenWorldBench/Genshin-v0", "OpenWorldBench/Genshin-Small-v0"]


@pytest.mark.parametrize("old,new", OLD_TO_NEW)
def test_old_module_is_new_module(old, new):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert importlib.import_module(old) is importlib.import_module(new)
    assert sys.modules[old] is sys.modules[new]


def test_old_names_are_identical_objects():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import open_world_bench
        from open_world_bench.env import OpenWorldEnv
        from open_world_bench.quests.schema import QuestGraph
        from owb.env import WildlandsEnv, SkylandsEnv
        from owb.core.config import EnvConfig
        from owb.render import make_renderer
    import owb.sims.teyvat as teyvat
    from owb.sims.teyvat.quests import schema
    from owb.sims.voxel import env as voxel_env
    from owb.sims.voxel.core import config
    from owb.sims.voxel import render as voxel_render
    assert OpenWorldEnv is teyvat.env.OpenWorldEnv is open_world_bench.OpenWorldEnv
    assert QuestGraph is schema.QuestGraph
    assert WildlandsEnv is voxel_env.WildlandsEnv is owb.WildlandsEnv
    assert SkylandsEnv is voxel_env.SkylandsEnv
    assert EnvConfig is config.EnvConfig is owb.EnvConfig
    assert make_renderer is voxel_render.make_renderer
    assert set(open_world_bench.__all__) == set(teyvat.__all__)
    assert open_world_bench.__version__ == teyvat.__version__
    assert open_world_bench.terrain is teyvat.terrain


def _fresh(code):
    """Run ``code`` in a clean interpreter so each alias warns for the first time."""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return out.stdout.split()


def test_old_paths_warn_once_per_alias():
    code = """
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import owb, owb.bench, owb.scripts.benchmark
    import owb.sims.teyvat.env, owb.sims.voxel.env
    assert not caught, [str(w.message) for w in caught]
    import open_world_bench
    import open_world_bench.env, open_world_bench.render.meshes, open_world_bench.quests.schema
    import owb.env, owb.env.skylands, owb.core.blocks, owb.render.landscape
    from owb.render import make_renderer
print(*sorted(str(w.message).split()[0] for w in caught if w.category is DeprecationWarning))
"""
    assert _fresh(code) == ["open_world_bench", "owb.core", "owb.env", "owb.render.landscape",
                            "owb.render.make_renderer"]


def test_old_package_registers_gym_ids():
    code = """
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
import open_world_bench
from gymnasium.envs.registration import registry
print(*sorted(spec.entry_point for name, spec in registry.items() if name.startswith("OpenWorldBench/")))
"""
    entries = _fresh(code)
    assert len(entries) == len(GYM_IDS)
    assert all(e.startswith(("owb.sims.teyvat.env:", "owb.sims.voxel.env")) for e in entries), entries


@pytest.mark.parametrize("env_id", GYM_IDS)
def test_gym_make_registered_ids(env_id):
    env = gym.make(env_id, disable_env_checker=True)
    try:
        assert env.unwrapped.__class__.__module__.startswith("owb.sims.")
    finally:
        env.close()
