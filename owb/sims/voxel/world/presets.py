"""Named world presets.

A preset is a bundle of :class:`~owb.sims.voxel.core.config.WorldConfig` overrides.  Tasks
reference presets by name so a benchmark definition stays short and readable.
"""

from __future__ import annotations

from dataclasses import replace

from ..core.config import WorldConfig

PRESETS: dict[str, dict] = {
    # the default: everything in one map
    "wildlands": dict(),

    # small, gentle, no monsters -- for debugging and unit tests
    "flatlands": dict(size_x=96, size_z=96, size_y=48, relief=4.0,
                      mountain_relief=2.0, cliff_sharpness=1.0,
                      tree_density=0.4, monster_density=0.0, cave_density=0.0,
                      landmarks=False, river_density=0.0, snow_line=40),

    # starting plateau: compact, one tower, climbable cliffs all round
    "plateau": dict(size_x=160, size_z=160, size_y=72, relief=20.0,
                    mountain_relief=26.0, cliff_sharpness=2.0,
                    structure_density=1.4, monster_density=0.7),

    # tall cliff bands and chasms: traversal / glider tasks
    "crags": dict(size_x=192, size_z=192, size_y=96, relief=30.0,
                  mountain_relief=46.0, cliff_sharpness=2.6,
                  hill_scale=34.0, tree_density=0.5, river_density=0.6),

    # cold crossing: most of the map above the snow line
    "frostpass": dict(size_x=192, size_z=192, size_y=88, relief=26.0,
                      mountain_relief=34.0, snow_line=26,
                      temperature_scale=240.0, tree_density=0.5),

    # hot province: lava, basalt, fire monsters
    "cinderfell": dict(size_x=176, size_z=176, size_y=80, relief=22.0,
                       mountain_relief=30.0, moisture_scale=70.0,
                       temperature_scale=90.0, tree_density=0.2,
                       river_density=0.4),

    # water world: islands, cryo / swimming tasks
    "archipelago": dict(size_x=192, size_z=192, size_y=72, sea_level=26,
                        base_height=16.0, relief=24.0, mountain_relief=18.0,
                        river_density=1.4),

    # dense forest: resource gathering, cooking
    "greatwood": dict(size_x=176, size_z=176, moisture_scale=60.0,
                      tree_density=2.0, grass_density=1.6, relief=16.0,
                      mountain_relief=14.0),

    # canyon maze: long-horizon navigation
    "redcut": dict(size_x=208, size_z=208, size_y=88, relief=30.0,
                   mountain_relief=24.0, cliff_sharpness=3.0, hill_scale=28.0,
                   river_density=1.2, temperature_scale=180.0),

    # tiny arena for combat / physics unit tests
    "arena": dict(size_x=64, size_z=64, size_y=40, relief=2.0,
                  mountain_relief=0.0, cliff_sharpness=1.0, tree_density=0.0,
                  grass_density=0.3, cave_density=0.0, river_density=0.0,
                  landmarks=False, monster_density=0.0, ore_density=0.3),
}


def apply_preset(cfg: WorldConfig) -> WorldConfig:
    """Return a copy of ``cfg`` with its named preset applied.

    Fields the caller changed from the dataclass default win over the preset,
    so ``WorldConfig(preset="crags", seed=3, size_x=128)`` does what it looks
    like it does.
    """
    name = cfg.preset or "wildlands"
    over = PRESETS.get(name)
    if over is None:
        raise KeyError(f"unknown world preset {name!r}; "
                       f"known: {sorted(PRESETS)}")
    default = WorldConfig()
    changes = {}
    for k, v in over.items():
        if getattr(cfg, k) == getattr(default, k):   # user did not override it
            changes[k] = v
    return replace(cfg, **changes) if changes else cfg


def names() -> list[str]:
    return sorted(PRESETS)
