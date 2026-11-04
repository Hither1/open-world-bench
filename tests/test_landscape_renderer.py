"""Rendering invariants: edits/caves, water and mesh depth, reproducibility."""
import numpy as np
import pytest

from owb import EnvConfig, RenderConfig, WildlandsEnv, WorldConfig
from owb.core import blocks as B
from owb.render import make_renderer
from owb.render.landscape import LandscapeRenderer
from owb.world.voxel import VoxelWorld


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
    from owb.render.raycast import Renderer
    assert type(make_renderer(RenderConfig(style="voxel"))) is Renderer
    assert isinstance(make_renderer(RenderConfig()), LandscapeRenderer)
    with pytest.raises(ValueError, match="style"):
        make_renderer(RenderConfig(style="unknown"))
