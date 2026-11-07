"""Scene fidelity, geometry topology, and simulation isolation for offline export."""
from copy import deepcopy
import json

import gymnasium as gym
import numpy as np
import pytest

import owb
from owb.render.scene_export import capture_scene, SceneSnapshot, _add_batch, _grid_triangles, _isosurface, _voxel_faces


def test_heightfield_winding_and_extent():
    x, z = np.meshgrid([0, 2, 4], [-3, 0, 3], indexing="ij")
    triangles = _grid_triangles(np.stack((x, x * 0.3 + z * 0.1, z), axis=-1))
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.all(normal[:, 1] > 0)
    np.testing.assert_allclose(triangles.min(axis=(0, 1)), [0, -0.3, -3])
    np.testing.assert_allclose(triangles.max(axis=(0, 1)), [4, 1.5, 3])


def test_isosurface_preserves_enclosed_cave_and_outward_normals():
    density = np.ones((8, 8, 8), np.float32)
    density[2:6, 2:6, 2:6] = 0
    triangles = _isosurface(np.pad(density, 1))
    center = triangles.mean(axis=1)
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    inward = np.all((center > 1) & (center < 7), axis=1)
    assert inward.any() and (~inward).any()
    radial_dot = (normal * (center - 4)).sum(axis=1)
    assert np.all(radial_dot[inward] < 0)  # cave walls face into the empty cavity
    assert np.all(radial_dot[~inward] > 0)
    # Both the outer surface and cave boundary must be watertight.
    _, faces = np.unique(np.round(triangles.reshape(-1, 3), 5), axis=0, return_inverse=True)
    faces = faces.reshape(-1, 3)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert np.all(counts == 2)


def test_voxel_architecture_omits_internal_faces():
    voxels = np.ones((2, 1, 1), np.uint8)
    triangles, blocks = _voxel_faces(voxels, voxels != 0)
    assert triangles.shape == (20, 3, 3)
    assert np.all(blocks == 1)
    assert not np.any(np.all(triangles[:, :, 0] == 1, axis=1))


def test_export_repairs_inward_primitives_without_changing_native_meshes():
    from owb.sims.teyvat.render.meshes import MeshBatch
    batch = MeshBatch()
    batch.ellipsoid((0, 0, 0), (1, 2, 1), (0.4, 0.6, 0.2))
    original = batch.finish()[0].copy()
    scene = SceneSnapshot({})
    _add_batch(scene, batch)
    triangles = scene.meshes[0]["triangles"]
    normals = scene.meshes[0]["normals"]
    face = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.all((face * normals.mean(axis=1)).sum(axis=1) > 0)
    np.testing.assert_array_equal(batch.finish()[0], original)


@pytest.fixture(params=("teyvat", "wildlands"))
def env(request):
    if request.param == "teyvat":
        from owb.sims.teyvat import EnvConfig, WorldConfig, TerrainConfig, RenderSettings
        cfg = EnvConfig(world=WorldConfig(terrain=TerrainConfig(size=512, resolution=33), prop_density=0),
                        render=RenderSettings(march_steps=16, shadow_steps=0, draw_props=False))
    else:
        cfg = owb.EnvConfig(world_cfg=owb.WorldConfig(preset="arena", size_x=24, size_z=24, size_y=40),
                            render_cfg=owb.RenderConfig(max_dist=32, max_steps=32, shadows=False))
    with owb.make_env(request.param, native_config=cfg, image_size=(16, 12)) as instance:
        instance.reset(seed=7)
        yield instance


def test_capture_is_detached_and_does_not_advance_simulation(env, tmp_path):
    native = env.native
    actor = native.char if env.name == "teyvat" else native.agent
    before = actor.pos.copy()
    rng = deepcopy(native.np_random.bit_generator.state)
    simulation_rng = getattr(native, "rng", native.np_random)
    simulation_rng_before = deepcopy(simulation_rng.bit_generator.state)
    clock_before = native.world.elapsed if env.name == "teyvat" else native.ticks
    scene = capture_scene(env, width=320, height=180)
    np.testing.assert_array_equal(actor.pos, before)
    assert native.np_random.bit_generator.state == rng
    assert simulation_rng.bit_generator.state == simulation_rng_before
    assert (native.world.elapsed if env.name == "teyvat" else native.ticks) == clock_before
    assert scene.metadata["environment"] == env.name
    assert scene.metadata["seed"] == 7
    assert scene.metadata["size"] == [320, 180]
    assert len(scene.meshes) >= 2
    for mesh in scene.meshes:
        for name in ("triangles", "colors", "normals", "emission"):
            assert np.isfinite(mesh[name]).all()
    # A later change in the live environment cannot alter an exported snapshot.
    frozen = [mesh["triangles"].copy() for mesh in scene.meshes]
    actor.pos += (1, 0, 0)
    for mesh, expected in zip(scene.meshes, frozen):
        np.testing.assert_array_equal(mesh["triangles"], expected)
    path = scene.save(tmp_path / "snapshot.npz")
    with np.load(path, allow_pickle=False) as data:
        assert json.loads(str(data["metadata"]))["version"] == 1
        np.testing.assert_array_equal(data["mesh_0_triangles"], frozen[0])
    with pytest.raises(FileExistsError):
        scene.save(path)


def test_camera_uses_native_fov_convention(env):
    scene = capture_scene(env, width=320, height=180)
    fov_y = scene.metadata["camera"]["fov_y"]
    if env.name == "teyvat":
        assert fov_y == env.native.cfg.camera.fov_y
        np.testing.assert_allclose(scene.metadata["camera"]["eye"], env.native.cam.eye)
    else:
        horizontal = 2 * np.arctan(np.tan(np.radians(fov_y) / 2) * 320 / 180)
        assert np.degrees(horizontal) == pytest.approx(env.native.cfg.render_cfg.fov)


def test_requires_reset_and_valid_resolution():
    with owb.sims.voxel.env.WildlandsEnv() as env:
        with pytest.raises(gym.error.ResetNeeded):
            capture_scene(env)
        with pytest.raises(ValueError, match="positive integers"):
            capture_scene(env, width=0)
