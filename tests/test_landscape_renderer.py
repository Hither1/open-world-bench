"""Rendering invariants: edits/caves, water and mesh depth, reproducibility."""
import numpy as np
import pytest

from owb import EnvConfig, RenderConfig, WildlandsEnv, WorldConfig
from owb.sims.voxel.core import blocks as B
from owb.sims.voxel.render import make_renderer
from owb.sims.voxel.render.landscape import LandscapeRenderer
from owb.sims.voxel.world.voxel import VoxelWorld


def world():
    w = VoxelWorld(24, 20, 24, sea_level=7)
    w.fill_box((0, 0, 0), (23, 3, 23), B.GRASS)
    return w


def renderer(**overrides):
    return LandscapeRenderer(RenderConfig(width=33, height=25, max_dist=32,
                                           shadows=False, **overrides))


def center(frame, key):
    return frame[key][12, 16]


def test_edit_opens_visible_cave_without_changing_collision_grid():
    w, r = world(), renderer()
    w.fill_box((14, 4, 0), (20, 18, 23), B.STONE)
    before = w.vox.copy()
    closed = r.render(w, (8.5, 6.5, 12.5), 0, 0)
    assert 4.5 < center(closed, "depth") < 6.5
    np.testing.assert_array_equal(w.vox, before)
    w.fill_box((14, 5, 10), (20, 8, 14), B.AIR)
    opened = r.render(w, (8.5, 6.5, 12.5), 0, 0)
    assert center(opened, "depth") == 32
    assert center(opened, "semantic") == B.SEM_SKY


def test_water_surface_has_consistent_depth_and_class():
    w, r = world(), renderer()
    w.fill_box((4, 4, 4), (23, 7, 23), B.WATER)
    frame = r.render(w, (8.5, 10, 12.5), 0, -0.6)
    assert center(frame, "semantic") == B.SEM_WATER
    assert center(frame, "depth") == pytest.approx(2 / np.sin(0.6), abs=0.03)


def test_meshes_use_same_depth_convention_as_terrain():
    w, r = world(), renderer()
    e = dict(pos=(13, 6.5, 12.5), radius=0.6, height=1.2,
             color=(200, 90, 70), semantic=B.SEM_MONSTER)
    frame = r.render(w, (8.5, 6.5, 12.5), 0, 0, entities=[e])
    assert center(frame, "semantic") == B.SEM_MONSTER
    assert 3.8 < center(frame, "depth") < 4.6
    w.fill_box((10, 4, 10), (10, 10, 14), B.STONE)
    occluded = r.render(w, (8.5, 6.5, 12.5), 0, 0, entities=[e])
    assert center(occluded, "semantic") == B.SEM_ROCK
    assert center(occluded, "depth") < 2


def test_cache_distinguishes_worlds_with_identical_revision_numbers():
    a, b, r = world(), world(), renderer()
    a.fill_box((14, 4, 0), (20, 18, 23), B.STONE)
    b.fill_box((14, 4, 0), (20, 18, 23), B.AIR)
    assert a.rev == b.rev
    assert center(r.render(a, (8.5, 6.5, 12.5), 0, 0), "depth") < 7
    assert center(r.render(b, (8.5, 6.5, 12.5), 0, 0), "depth") == 32


@pytest.mark.parametrize("weather", ["clear", "rain", "snow"])
def test_deterministic_aligned_buffers_and_supersampling(weather):
    w, r = world(), renderer(supersample=2)
    a = r.render(w, (8.5, 6.5, 12.5), 0, -0.25, weather=weather, frame=50)
    b = r.render(w, (8.5, 6.5, 12.5), 0, -0.25, weather=weather, frame=50)
    assert a["rgb"].shape == (25, 33, 3)
    assert a["depth"].shape == a["semantic"].shape == (25, 33)
    assert np.isfinite(a["depth"]).all()
    assert a["depth"].min() >= 0 and a["depth"].max() <= 32
    assert a["semantic"].max() < B.NUM_SEMANTIC
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_rendering_is_independent_of_simulation_rng():
    cfg = EnvConfig(obs_rgb=False, action_space="discrete",
                    world_cfg=WorldConfig(preset="arena", size_x=32, size_z=32, size_y=40),
                    render_cfg=RenderConfig(width=33, height=25, max_dist=32, third_person=True))
    a, b = WildlandsEnv(cfg), WildlandsEnv(cfg)
    a.reset(seed=5)
    b.reset(seed=5)
    for _ in range(3):
        a.render()
        ra, rb = a.step(1), b.step(1)
        assert ra[1:] == rb[1:]
        for key in ra[0]:
            np.testing.assert_equal(ra[0][key], rb[0][key])


def test_voxel_style_remains_available_and_style_is_validated():
    from owb.sims.voxel.render.raycast import Renderer
    assert type(make_renderer(RenderConfig(style="voxel"))) is Renderer
    assert isinstance(make_renderer(RenderConfig()), LandscapeRenderer)
    with pytest.raises(ValueError, match="style"):
        make_renderer(RenderConfig(style="unknown"))
    with pytest.raises(ValueError, match="quality"):
        make_renderer(RenderConfig(quality="unknown"))


def test_mesh_shadow_direction_filter_and_map_boundaries():
    from owb.sims.voxel.render.shadows import MeshShadowMap
    roof = np.array([[[-1, 2, -1], [1, 2, -1], [1, 2, 1]],
                     [[-1, 2, -1], [1, 2, 1], [-1, 2, 1]]], np.float32)
    shadow = MeshShadowMap(roof, (0, 1, 0), (0, 0, 0), extent=4, resolution=64)
    points = np.array([[0, 0, 0], [0, 3, 0], [3, 0, 0], [-5, 0, 0]], np.float32)
    visibility = shadow.sample(points)
    assert visibility[0] < 0.3  # underneath the roof
    np.testing.assert_allclose(visibility[1:], 1)  # above, beside, outside the map
    assert np.all((shadow.sample(roof) >= 0) & (shadow.sample(roof) <= 1))


@pytest.mark.parametrize("quality", ["fast", "balanced", "high"])
def test_lighting_options_preserve_sensor_geometry(quality):
    w = world()
    w.fill_box((13, 4, 11), (13, 8, 11), B.LOG)
    w.fill_box((12, 8, 10), (14, 9, 12), B.LEAVES)
    cfg = RenderConfig(width=33, height=25, max_dist=32, quality=quality)
    r = LandscapeRenderer(cfg)
    snapshot = w.vox.copy()
    lit = r.render(w, (8.5, 6.5, 12.5), 0, -0.25)
    again = r.render(w, (8.5, 6.5, 12.5), 0, -0.25)
    np.testing.assert_array_equal(lit["rgb"], again["rgb"])
    cfg.shadows = cfg.ambient_occlusion = False
    unshadowed = r.render(w, (8.5, 6.5, 12.5), 0, -0.25)
    for key in ("depth", "semantic"):
        np.testing.assert_array_equal(lit[key], unshadowed[key])
    np.testing.assert_array_equal(w.vox, snapshot)
    assert np.isfinite(lit["depth"]).all()


@pytest.mark.parametrize("style", ["landscape", "voxel"])
def test_distant_floating_architecture_does_not_fill_space_below(style):
    w = VoxelWorld(64, 32, 24)
    w.meta["layered"] = True
    w.fill_box((42, 20, 10), (48, 21, 14), B.BRICK)
    r = make_renderer(RenderConfig(width=33, height=25, max_dist=60,
                                  exact_dist=8, max_steps=16, shadows=False, style=style))
    below = r.render(w, (8.5, 6.5, 12.5), 0, 0)
    assert center(below, "semantic") == B.SEM_SKY
    assert center(below, "depth") == 60
    roof = r.render(w, (8.5, 6.5, 12.5), 0, np.arctan2(14, 36))
    assert center(roof, "semantic") == B.SEM_STRUCTURE
    assert 34 < center(roof, "depth") < 43
