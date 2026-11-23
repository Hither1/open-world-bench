"""Configuration dataclasses for the world, simulator, renderer and env.

Coordinate convention (used everywhere):

* ``x`` east, ``y`` **up**, ``z`` south; 1 voxel = 1 metre.
* voxels live in ``world.vox[x, y, z]`` (``uint8`` block ids).
* ``yaw`` is radians, 0 = +x, increasing toward +z; ``pitch`` radians, + = up.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class WorldConfig:
    """Terrain generation parameters."""

    size_x: int = 256
    size_z: int = 256
    size_y: int = 80
    seed: int = 0

    sea_level: int = 18
    base_height: float = 31.0
    relief: float = 24.0           # metres from valley floor to typical ridge
    mountain_relief: float = 34.0  # extra metres on ridged peaks
    continent_scale: float = 140.0
    hill_scale: float = 46.0
    detail_scale: float = 13.0
    warp_strength: float = 6.0
    river_density: float = 1.0
    cliff_sharpness: float = 1.5   # >1 pushes slopes toward climbable cliffs

    #: fraction of the map covered by each biome driver
    temperature_scale: float = 150.0
    moisture_scale: float = 110.0
    snow_line: int = 66            # above this height -> snow
    desert_threshold: float = 0.45

    tree_density: float = 1.0
    grass_density: float = 1.0
    ore_density: float = 1.0
    cave_density: float = 1.0
    structure_density: float = 1.0
    monster_density: float = 1.0
    animal_density: float = 1.0

    #: generate the big set-piece landmarks (towers, trial chambers, camps...)
    landmarks: bool = True
    #: name of a preset in :mod:`owb.world.presets`, applied before the fields above
    preset: str = "wildlands"


@dataclass
class SimConfig:
    """Physics / gameplay tuning."""

    dt: float = 0.1                # seconds per simulator tick
    gravity: float = -18.0         # m/s^2 (snappier than Earth, like action games)
    max_fall_speed: float = -42.0

    walk_speed: float = 3.6
    run_speed: float = 6.4
    swim_speed: float = 2.4
    climb_speed: float = 1.9
    glide_speed: float = 9.0
    glide_sink: float = -2.2
    jump_speed: float = 7.2

    agent_radius: float = 0.35
    agent_height: float = 1.75
    eye_height: float = 1.6
    step_height: float = 0.65      # auto-step over small ledges

    max_health: float = 24.0       # 6 "hearts" x 4
    max_stamina: float = 100.0
    stamina_regen: float = 11.0
    climb_cost: float = 11.0       # per second
    sprint_cost: float = 9.0
    glide_cost: float = 5.5
    swim_cost: float = 7.0
    rain_climb_penalty: float = 1.9
    slip_chance_in_rain: float = 0.045

    fall_damage_threshold: float = -16.0
    fall_damage_scale: float = 0.9
    drown_damage: float = 3.0
    cold_damage: float = 0.55      # health per second when freezing
    heat_damage: float = 0.55
    lava_damage: float = 14.0
    fire_damage: float = 4.0
    shock_damage: float = 5.0

    #: real seconds of simulated time per in-world day
    day_length: float = 1200.0
    start_time: float = 0.30       # 0 = midnight, 0.5 = noon
    weather_enabled: bool = True
    thunder_prob: float = 0.22     # probability a rain front is a thunderstorm
    wind_enabled: bool = True

    fire_enabled: bool = True
    fire_spread_rate: float = 0.5
    fire_lifetime: float = 7.0
    updraft_strength: float = 7.5

    durability_enabled: bool = True
    hunger_enabled: bool = False   # off by default; survival tasks turn it on

    # Zelda-style recovery (opt-in; reference traces use permadeath).
    respawn_on_death: bool = False  # at 0 health, continue at the last milestone checkpoint
    max_continues: int = 3          # deaths that continue; the next death ends the episode
    void_out: bool = False          # out of stamina while swimming: back to last safe ground
    void_out_damage: float = 4.0    # one heart
    max_bodies: int = 256
    max_npcs: int = 96


@dataclass
class RenderConfig:
    """Visual renderer settings; voxel collisions are independent of style."""

    width: int = 128
    height: int = 128
    fov: float = 75.0              # degrees, horizontal
    max_dist: float = 112.0
    max_steps: int = 256
    #: out to this range every voxel is traced exactly; past it the renderer
    #: switches to a heightfield march with empty-space skipping
    exact_dist: float = 34.0
    far_steps: int = 96
    shadows: bool = True
    shadow_steps: int = 36
    water_depth_steps: int = 28
    clouds: bool = True
    fog: bool = True
    third_person: bool = False
    cam_distance: float = 4.0
    cam_pitch_offset: float = -0.12
    texture_noise: float = 0.10    # per-voxel colour jitter
    entity_impostors: bool = True
    hud: bool = False              # burn health/stamina bars into the RGB frame
    supersample: int = 1
    style: str = "landscape"       # "landscape" (cel shaded) | "voxel" (debug)
    quality: str = "balanced"      # "fast" | "balanced" | "high"; mesh/material detail
    ambient_occlusion: bool = True # local contact shading; RGB only


@dataclass
class EnvConfig:
    """Environment / observation / action settings."""

    #: "discrete" (named commands) | "multi" (movement + inventory Dict)
    action_space: str = "multi"
    frame_skip: int = 2
    max_steps: int = 3000
    obs_rgb: bool = True
    obs_depth: bool = False
    obs_semantic: bool = False
    obs_proprio: bool = True
    obs_inventory: bool = True
    obs_minimap: bool = True
    obs_text: bool = False
    minimap_size: int = 32
    minimap_scale: int = 4          # voxels per minimap pixel
    #: clickable map logo in the RGB view; see owb.sim.world_map
    world_map: bool = True
    map_cell: float = 2.0           # voxels per fog-of-war cell
    #: metres revealed around the agent's path; None scales with the world
    #: (8% of the shorter side, clamped to 8..24 m: 20 m on the default world)
    map_reveal_radius: float | None = None
    reward_mode: str = "milestone"  # "milestone" | "sparse" | "dense"
    exploration_bonus: float = 0.0
    time_penalty: float = 0.0
    death_penalty: float = 0.0
    seed: int = 0
    starter_camp: bool = True     # safe clearing, cooking pot, nearby ingredients
    render_cfg: RenderConfig = field(default_factory=RenderConfig)
    sim_cfg: SimConfig = field(default_factory=SimConfig)
    world_cfg: WorldConfig = field(default_factory=WorldConfig)


def to_dict(cfg: Any) -> dict:
    return asdict(cfg)


def update(cfg: Any, **kw) -> Any:
    """Return a copy of ``cfg`` with fields overridden (ignores unknown keys)."""
    import copy
    out = copy.deepcopy(cfg)
    for k, v in kw.items():
        if hasattr(out, k):
            setattr(out, k, v)
    return out
