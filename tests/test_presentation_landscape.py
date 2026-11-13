"""Presentation landscape: framing, set pieces, and ground detail never touch simulation."""
import json
from pathlib import Path

import numpy as np
import pytest

from owb.render import dressing
from owb.render.scene_export import capture_scene

ROOT = Path(__file__).resolve().parents[1]


def _trace(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return np.array([row["position_after"] for row in rows], float)[:, [0, 2]]


@pytest.fixture(scope="module", params=["a", "b"])
def teyvat_course(request):
    from open_world_bench.reference_tasks import ReferenceTaskEnv
    env = ReferenceTaskEnv(request.param, 7, render_enabled=False)
    env.reset(seed=7)
    yield request.param, env
    env.close()


def test_teyvat_set_pieces_clear_the_recorded_reference_trace(teyvat_course):
    task, env = teyvat_course
    layout = dressing.course_layout(env.world)
    kinds = dressing.layout_summary(layout)["kinds"]
    # Middle-scale locations, not a scatter of one kind of prop.
    assert len(layout.pieces) > 100 and len(kinds) >= 5
    positions = _trace(ROOT / f"assets/demos/teyvat-reference-cycles/task-{task}-seed-7/actions.jsonl")
    assert dressing.min_clearance(layout, positions) > 1.0


def test_wildlands_set_pieces_clear_the_recorded_reference_trace():
    from owb.env.wildlands_reference import WildlandsReferenceEnv
    for task in "ab":
        env = WildlandsReferenceEnv(task, 7)
        env.reset(seed=7)
        layout = dressing.wildlands_reference_layout(env)
        assert layout.pieces
        positions = _trace(ROOT / f"assets/demos/wildlands-reference/task-{task}-seed-7/actions.jsonl")
        assert dressing.min_clearance(layout, positions) > 1.0
        env.close()


def test_capture_with_landscape_leaves_simulation_and_rng_untouched(teyvat_course, tmp_path):
    _, env = teyvat_course
    before = env.state_digest()
    rng_state = env.rng.bit_generator.state
    course_rng = env.course.rng.bit_generator.state
    np.random.seed(5)
    global_state = np.random.get_state()[1].copy()
    scene = capture_scene(env, width=320, height=180)
    assert env.state_digest() == before
    assert env.rng.bit_generator.state == rng_state
    assert env.course.rng.bit_generator.state == course_rng
    np.testing.assert_array_equal(np.random.get_state()[1], global_state)
    summary = scene.metadata["presentation_landscape"]
    assert summary["collision"] is False and summary["observation"] is False
    assert summary["horizon_layers"] == 2
    path = scene.save(tmp_path / "scene.npz")
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        texture = data["texture_ground_detail"]
    assert texture.dtype == np.uint8 and texture.shape[2] == 4
    assert metadata["textures"]["ground_detail"]["channels"] == ["path", "soil", "paving", "rock"]
    prototypes = [m for m in metadata["meshes"] if m.get("prototype")]
    assert len(prototypes) == len(metadata["ground_cover"]["prototypes"])


def test_layout_is_a_stable_function_of_world_identity(teyvat_course):
    _, env = teyvat_course
    first = dressing.teyvat_course_layout(env.world)
    second = dressing.teyvat_course_layout(env.world)
    assert [(p.identity, p.kind, tuple(np.round(p.xz, 6))) for p in first.pieces] == \
           [(p.identity, p.kind, tuple(np.round(p.xz, 6))) for p in second.pieces]
    # Appearance follows the piece, not the camera: two views emit identical
    # geometry for a piece visible from both.
    piece = first.pieces[0]
    eye = np.array((piece.xz[0] - 20, 15, piece.xz[1]))
    from open_world_bench.render.meshes import MeshBatch
    a, b = MeshBatch(), MeshBatch()
    first.pieces, keep = [piece], first.pieces
    dressing.emit_pieces(first, a, eye, np.array((1.0, -.3, 0)))
    dressing.emit_pieces(first, b, eye + (3, 1, 2), np.array((1.0, -.2, .1)))
    first.pieces = keep
    np.testing.assert_array_equal(a.finish()[0], b.finish()[0])


def test_horizon_ridges_stay_outside_the_walkable_square():
    from open_world_bench.render.meshes import MeshBatch
    batch = MeshBatch()
    dressing.horizon_ridges(batch, center=(80, 80), half_extent=80, inner_gap=10, depth=200,
                            floor=6, peak=80, identity="test")
    vertices = np.concatenate(batch.verts).reshape(-1, 3)
    inside = (np.abs(vertices[:, 0] - 80) < 80) & (np.abs(vertices[:, 2] - 80) < 80)
    assert not inside.any()


def test_relay_parts_stand_on_foundations():
    from open_world_bench.render.assets import add_detailed_entity
    from open_world_bench.render.meshes import MeshBatch
    for role in ("source", "vane", "drain", "core"):
        batch = MeshBatch()
        add_detailed_entity(batch, "pillar", np.zeros(3), 0.0, False, 100, identity=("relay", role),
                            visual_role=role)
        vertices = batch.finish()[0].reshape(-1, 3)
        footing = vertices[vertices[:, 1] < .1]
        # A footing wider than the shaft, set at ground level.
        assert np.ptp(footing[:, 0]) > 1.4 and np.ptp(footing[:, 2]) > 1.4
