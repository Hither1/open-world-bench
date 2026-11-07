"""Architecture stability, passage clearance, and structural collision checks."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from open_world_bench.architecture import Architecture, Building, KINDS, populate_architecture, solid_parts
from open_world_bench.character import CharacterState
from open_world_bench.reference_tasks import ReferenceTaskEnv
from open_world_bench.render.architecture import _template, add_architecture
from open_world_bench.render.meshes import MeshBatch
from owb.render.scene_export import SceneSnapshot, _add_batch


def fingerprint(arrays):
    return hashlib.sha256(b''.join(a.tobytes() for a in arrays)).hexdigest()


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('foundation_depth', [0.0, 3.2])
def test_architecture_meshes_are_stable_varied_and_exportable(kind, foundation_depth):
    before = deepcopy(np.random.get_state())
    first = _template(kind, 37, foundation_depth)
    digest = fingerprint(first)
    other = _template(kind, 91, foundation_depth)
    assert digest != fingerprint(other)
    building = Building('test', kind, (10, 12, -8), .7, 37, foundation_depth)
    batch = MeshBatch()
    add_architecture(batch, Architecture([building]), np.zeros(3), 100)
    assert len(batch.finish()[0]) == len(first[0])
    assert digest == fingerprint(_template(kind, 37, foundation_depth))
    np.testing.assert_array_equal(before[1], np.random.get_state()[1])
    assert before[2:] == np.random.get_state()[2:]
    scene = SceneSnapshot({})
    _add_batch(scene, batch)
    materials = {m['material'] for m in scene.meshes}
    assert 'stone' in materials and len(materials) >= 2
    for mesh in scene.meshes:
        assert np.isfinite(mesh['triangles']).all()
        assert np.isfinite(mesh['normals']).all()
        assert np.all((mesh['colors'] >= 0) & (mesh['colors'] <= 1))
        normal = np.cross(mesh['triangles'][:, 1]-mesh['triangles'][:, 0],
                          mesh['triangles'][:, 2]-mesh['triangles'][:, 0])
        assert np.all(np.linalg.norm(normal, axis=1) > 1e-8)
        assert np.all(np.sum(normal*mesh['normals'].mean(axis=1), axis=1) >= -1e-5)


@pytest.mark.parametrize('yaw', [0, .73, np.pi/2])
def test_solid_house_stops_walking_and_preserves_tangent_velocity(yaw):
    building = Building('house', 'timber_house', (3, 12, 7), yaw, 7)
    architecture = Architecture([building])
    rotation = building.rotation
    state = CharacterState()
    state.pos = (np.array((3.1, 0, 0)) @ rotation.T+building.pos).astype(np.float32)
    state.vel = (np.array((-2, 0, 1)) @ rotation.T).astype(np.float32)
    architecture.resolve(state, .4, 1.75)
    local = (state.pos-building.pos) @ rotation
    assert local[0] == pytest.approx(3.4, abs=2e-5)
    np.testing.assert_allclose(state.vel @ rotation, (0, 0, 1), atol=1e-6)
    assert not architecture.blocked([state.pos], margin=.39)[0]
    # Flying above the structure must not produce a ground-footprint collision.
    state.pos = np.array((3, 30, 7), np.float32)
    before = state.pos.copy()
    architecture.resolve(state, .4, 1.75)
    np.testing.assert_array_equal(state.pos, before)


@pytest.mark.parametrize('kind', ['pavilion', 'market_stall', 'ruin_gate'])
def test_open_architecture_keeps_its_central_passage(kind):
    architecture = Architecture([Building('passage', kind, (0, 12, 0), 0, 1)])
    state = CharacterState()
    state.pos = np.array((0, 12, 0), np.float32)
    before = state.pos.copy()
    architecture.resolve(state, .4, 1.75)
    np.testing.assert_array_equal(state.pos, before)
    assert not architecture.blocked([state.pos+(0, 1, 0)])[0]
    assert architecture.blocked([architecture.centers[0]])[0]


@pytest.mark.parametrize('task,seed', [('a', 7), ('b', 7), ('b', 19)])
def test_task_layout_is_stable_and_preserves_terrain_rng_and_interaction_zones(task, seed):
    with ReferenceTaskEnv(task, seed=seed, render_enabled=False) as env:
        env.reset(seed=seed)
        before = [b.metadata() for b in env.world.architecture.buildings]
        height = env.world.terrain.height.copy()
        rng = deepcopy(env.world.rng.bit_generator.state)
        course_rng = deepcopy(env.course.rng.bit_generator.state)
        state = env.state_digest()
        architecture = populate_architecture(env.world, env.char.pos)
        assert before == [b.metadata() for b in architecture.buildings]
        assert set(KINDS) <= {b.kind for b in architecture.buildings}
        np.testing.assert_array_equal(height, env.world.terrain.height)
        assert env.world.rng.bit_generator.state == rng
        assert env.course.rng.bit_generator.state == course_rng
        assert env.state_digest() == state
        for entity in env.world.entities.values():
            for angle in np.arange(8)*np.pi/4:
                point = entity.home+(np.cos(angle)*3, .8, np.sin(angle)*3)
                assert not architecture.blocked([point], margin=.4)[0]
        # The controller must resolve structures even with an empty PropField.
        building = next(b for b in architecture.buildings if b.kind == 'timber_house')
        env.char.pos = np.array(building.pos, np.float32)
        env.controller._resolve_props(env.char)
        assert not architecture.blocked([env.char.pos], margin=.39)[0]
        assert env.camera_rig.architecture is not None


def test_regular_island_gets_grounded_architecture_without_earthworks():
    from owb import make_env
    with make_env('teyvat', seed=7, image_size=(16, 12)) as env:
        env.reset(seed=7)
        world = env.native.world
        assert len(world.architecture.buildings) >= 6
        assert {'timber_house', 'windmill', 'ruin_gate'} <= {b.kind for b in world.architecture.buildings}
        terrain = world.terrain.height.copy()
        identities = [b.metadata() for b in world.architecture.buildings]
        populate_architecture(world, env.native.char.pos)
        np.testing.assert_array_equal(terrain, world.terrain.height)
        assert identities == [b.metadata() for b in world.architecture.buildings]
        for building in world.architecture.buildings:
            assert 0 <= building.foundation_depth <= 3.2
            # Foundations reach at least the lowest sampled ground elevation.
            parts = solid_parts(building.kind, building.foundation_depth)
            bottoms = [center[1]-size[1]/2 for center, size, _ in parts]
            assert min(bottoms) <= -building.foundation_depth
