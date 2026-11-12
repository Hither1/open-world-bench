"""Detached scene snapshots for offline rendering, independent of agent controls.

The NPZ format contains only numeric arrays and a JSON manifest (no pickle).
Positions and normals stay in the simulators' right-handed, Y-up coordinates.
Blender converts these to Z-up; its glTF exporter converts back to Y-up.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from copy import deepcopy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np


def _presentation_batch():
    from open_world_bench.render.meshes import MeshBatch

    class PresentationBatch(MeshBatch):
        """Spend more triangles on the same shapes for offline presentation."""
        presentation_detail = True
        def ellipsoid(self, center, radii, color, yaw=0, obj_id=1, sides=8, rings=4):
            super().ellipsoid(center, radii, color, yaw, obj_id, max(sides, 18), max(rings, 8))

        def cone(self, base, radius, height, color, sides=7, obj_id=1, emissive=0, taper=0):
            super().cone(base, radius, height, color, max(sides, 16), obj_id, emissive, taper)

        def tube(self, start, end, radius, color, obj_id=1, end_radius=None, sides=6):
            super().tube(start, end, radius, color, obj_id, end_radius, max(sides, 12))

    return PresentationBatch()


@dataclass
class SceneSnapshot:
    metadata: dict
    meshes: list[dict] = field(default_factory=list)

    def add(self, name, triangles, colors, *, normals=None, emission=None, smooth=False, water=False, material="surface"):
        triangles = np.asarray(triangles, np.float32).reshape(-1, 3, 3)
        if not len(triangles):
            return
        colors = np.broadcast_to(colors, (len(triangles), 3)).astype(np.float32).copy()
        self.meshes.append(dict(name=name, triangles=triangles.copy(), colors=colors,
                                normals=np.zeros_like(triangles) if normals is None else np.asarray(normals, np.float32).copy(),
                                emission=np.zeros(len(triangles), np.float32) if emission is None else np.asarray(emission, np.float32).copy(),
                                smooth=bool(smooth), water=bool(water), material=material))

    def save(self, path):
        """Write a new snapshot, refusing to overwrite an existing artifact."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata, meshes=[])
        arrays = {}
        for i, mesh in enumerate(self.meshes):
            metadata["meshes"].append({k: mesh[k] for k in ("name", "smooth", "water", "material")})
            for key in ("triangles", "colors", "normals", "emission"):
                arrays[f"mesh_{i}_{key}"] = mesh[key]
        with path.open("xb") as handle:
            np.savez_compressed(handle, metadata=json.dumps(metadata), **arrays)
        return path


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def _grid_triangles(points):
    """Triangulate an X/Z grid, with winding toward +Y."""
    a, b, c, d = points[:-1, :-1], points[:-1, 1:], points[1:, 1:], points[1:, :-1]
    return np.stack((np.stack((a, b, c), -2), np.stack((a, c, d), -2)), -3).reshape(-1, 3, 3)


def _isosurface(density, level=0.5):
    """March tetrahedra at voxel centres; preserve caves and disconnected layers.

    Process only boundary cells so memory scales with surface area. Input has
    one padding sample on each side, matching LandscapeRenderer._density.
    """
    offsets = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
    shape = np.array(density.shape) - 1
    low = density[:-1, :-1, :-1].copy()
    high = low.copy()
    for off in offsets[1:]:
        values = density[tuple(slice(int(o), int(o + n)) for o, n in zip(off, shape))]
        np.minimum(low, values, out=low)
        np.maximum(high, values, out=high)
    cells = np.argwhere((low < level) & (high >= level))
    if not len(cells):
        return np.empty((0, 3, 3), np.float32)
    corners = cells[:, None, :] + offsets
    values = density[tuple(corners.transpose(2, 0, 1))]
    points = corners.astype(np.float32) - 0.5
    output = []
    # Six tetrahedra sharing the same body diagonal, consistent across cells.
    for tetra in ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
                  (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)):
        p, v = points[:, tetra], values[:, tetra]
        codes = ((v >= level) * (1 << np.arange(4))).sum(axis=1)
        for code in range(1, 15):
            select = codes == code
            if not select.any():
                continue
            pp, vv = p[select], v[select]
            inside = [i for i in range(4) if code & (1 << i)]
            outside = [i for i in range(4) if not code & (1 << i)]

            def edge(a, b):
                t = (level - vv[:, a]) / (vv[:, b] - vv[:, a])
                return pp[:, a] + t[:, None] * (pp[:, b] - pp[:, a])

            if len(inside) == 2:
                a, b = inside
                c, d = outside
                q = np.stack((edge(a, c), edge(b, c), edge(b, d), edge(a, d)), axis=1)
                tris = np.stack((q[:, [0, 1, 2]], q[:, [0, 2, 3]]), axis=1)
            else:
                single, others = (inside[0], outside) if len(inside) == 1 else (outside[0], inside)
                tris = np.stack([edge(single, j) for j in others], axis=1)[:, None]
            outward = pp[:, outside].mean(axis=1) - pp[:, inside].mean(axis=1)
            normal = np.cross(tris[:, :, 1] - tris[:, :, 0], tris[:, :, 2] - tris[:, :, 0])
            flip = (normal * outward[:, None]).sum(axis=-1) < 0
            tris[flip] = tris[flip][:, [0, 2, 1]]
            output.append(tris.reshape(-1, 3, 3))
    triangles = np.concatenate(output)
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    return triangles[area > 1e-8]


def _voxel_faces(voxels, occupied):
    """Exposed faces only, retaining architecture and water boundaries."""
    from open_world_bench.render.meshes import _CUBE_V, _CUBE_F
    output, blocks = [], []
    for axis, sign, face in ((2, -1, 0), (2, 1, 1), (0, -1, 2),
                             (0, 1, 3), (1, 1, 4), (1, -1, 5)):
        neighbor = np.roll(occupied, -sign, axis=axis)
        boundary = [slice(None)] * 3
        boundary[axis] = -1 if sign == 1 else 0
        neighbor[tuple(boundary)] = False
        indices = np.argwhere(occupied & ~neighbor)
        if len(indices):
            faces = _CUBE_V[_CUBE_F[face * 2:face * 2 + 2]] + 0.5
            output.append((indices[:, None, None] + faces).reshape(-1, 3, 3))
            blocks.append(np.repeat(voxels[tuple(indices.T)], 2))
    if not output:
        return np.empty((0, 3, 3), np.float32), np.empty(0, np.uint8)
    return np.concatenate(output).astype(np.float32), np.concatenate(blocks)


def _add_batch(scene, batch):
    triangles, colors, emission, _ = batch.finish()
    if len(triangles):
        normals = np.concatenate(getattr(batch, "smooth_normals", batch.normals))
        # Legacy primitives were shaded two-sided by the CPU rasterizer. Some
        # wind inward, which breaks ray-traced shading and glTF backface culling.
        # Each add() is one convex primitive or a planar patch. Orient convex
        # faces away from their local centre; authored normals take precedence.
        triangles, normals = triangles.copy(), normals.copy()
        start = 0
        for part in batch.verts:
            end = start + len(part)
            face = np.cross(part[:, 1] - part[:, 0], part[:, 2] - part[:, 0])
            authored = normals[start:end].mean(axis=1)
            outward = part.mean(axis=1) - part.reshape(-1, 3).mean(axis=0)
            reference = np.where(np.linalg.norm(authored, axis=1, keepdims=True) > 0, authored, outward)
            flip = (face * reference).sum(axis=1) < -1e-7
            triangles[start:end][flip] = part[flip][:, [0, 2, 1]]
            normals[start:end][flip] = normals[start:end][flip][:, [0, 2, 1]]
            start = end
        from open_world_bench.render.meshes import MATERIAL_NAMES
        material_ids=np.concatenate(batch.material_ids)
        for material_id in np.unique(material_ids):
            mask=material_ids==material_id
            name=MATERIAL_NAMES[int(material_id)]
            scene.add("Characters and props: "+name, triangles[mask], colors[mask],
                      normals=normals[mask], emission=emission[mask], material=name)


VIEWS = ("presentation", "first_person")


def teyvat_camera(native, view="presentation"):
    """Eye and target for a Teyvat presentation view; never changes the env.

    ``presentation`` is the policy's own orbit camera. ``first_person`` puts the
    eye at the active character's eye height, looking level along the
    character's heading (its facing, not the freely orbiting camera yaw).
    """
    if view == "presentation":
        return native.cam.eye.copy(), native.cam.pivot.copy()
    if view == "first_person":
        from open_world_bench.math3d import yaw_pitch_to_dir
        eye = (native.char.pos + np.array([0.0, native.cfg.character.eye_height, 0.0])).astype(np.float32)
        return eye, (eye + yaw_pitch_to_dir(native.char.yaw, 0.0)).astype(np.float32)
    raise ValueError(f"view must be one of {VIEWS}")


def _teyvat(native, size, view="presentation"):
    from open_world_bench.render.soft_renderer import SoftRenderer
    from open_world_bench.render.style import BIOME_ALBEDO, sky_keys_at
    from open_world_bench.terrain import MEADOW, FOREST, CLIFF, SNOW, BEACH
    from open_world_bench.math3d import smoothstep

    world, terrain = native.world, native.world.terrain
    eye, target = teyvat_camera(native, view)
    keys = sky_keys_at(world.time_of_day)
    scene = SceneSnapshot(dict(version=1, environment="teyvat", seed=native._episode_seed, view=view,
        time_of_day=float(world.time_of_day), coordinates="Y-up, metres", size=list(size),
        camera=dict(eye=eye.tolist(), target=target.tolist(), fov_y=native.cfg.camera.fov_y),
        lighting=dict(direction=world.sun_direction.tolist(), color=list(keys.sun),
                      strength=keys.sun_strength, sky=list(keys.zenith), horizon=list(keys.horizon))))
    axis = np.linspace(-terrain.cfg.size / 2, terrain.cfg.size / 2, terrain.cfg.resolution)
    x, z = np.meshgrid(axis, axis, indexing="ij")
    triangles = _grid_triangles(np.stack((x, terrain.height, z), axis=-1))
    p = triangles.mean(axis=1)
    normals = terrain.normal_at(p[:, 0], p[:, 2])
    slope = np.arccos(np.clip(normals[:, 1], -1, 1))
    wet = smoothstep(0.44, 0.66, terrain.moisture_at(p[:, 0], p[:, 2]))[:, None]
    color = BIOME_ALBEDO[MEADOW] * (1 - wet) + BIOME_ALBEDO[FOREST] * wet
    for weight, albedo in (
        (smoothstep(np.radians(30), np.radians(52), slope), BIOME_ALBEDO[CLIFF]),
        (smoothstep(terrain.cfg.snow_line - 30, terrain.cfg.snow_line + 4, p[:, 1])
         * np.clip(1 - slope / np.radians(56), 0, 1), BIOME_ALBEDO[SNOW]),
        (smoothstep(4.2, 0.6, p[:, 1]) * np.clip(1 - slope / np.radians(34), 0, 1), BIOME_ALBEDO[BEACH]),
    ):
        color = color * (1 - weight[:, None]) + albedo * weight[:, None]
    if hasattr(world, "campaign"):
        from open_world_bench.campaign_tasks import SUITE_VERSION
        scene.metadata["campaign"] = dict(version=SUITE_VERSION, task=world.campaign.task,
            geography_scale=native.geography_scale, state_digest=native.state_digest(),
            active_seals=deepcopy(world.architecture.active()))
        if world.campaign.task == "desert":
            weight = smoothstep(np.radians(30), np.radians(52), slope)[:,None]
            color = np.array((.66,.49,.29))*(1-weight)+np.array((.49,.36,.24))*weight
    scene.add("Island terrain", triangles, color, smooth=True, material="terrain")
    if world.architecture is not None:
        scene.metadata["architecture"] = dict(
            revision=1, collision="structural boxes; roofs are not traversal platforms",
            buildings=[b.metadata() for b in world.architecture.buildings])
    extent = terrain.cfg.size * 4
    plane = np.array([[[-extent, terrain.cfg.sea_level, -extent], [-extent, terrain.cfg.sea_level, extent]],
                      [[extent, terrain.cfg.sea_level, -extent], [extent, terrain.cfg.sea_level, extent]]])
    scene.add("Ocean", _grid_triangles(plane), (0.16, 0.61, 0.66), water=True)
    # Observation budgets must not reduce presentation geometry or vegetation.
    settings = replace(native.cfg.render, width=size[0], height=size[1],
                       draw_props=True, draw_entities=True, draw_character=view != "first_person",
                       ground_cover=True, prop_draw_distance=max(260, native.cfg.render.prop_draw_distance))
    _add_batch(scene, SoftRenderer(settings)._build_meshes(
        world, native.char, eye, _unit(target - eye), batch=_presentation_batch(), full_detail=True))
    return scene


def _wildlands(native, size):
    from .landscape import LandscapeRenderer, _NATURAL
    from .camera import Camera, resolve_third_person
    from . import palette as P
    from ..core import blocks as B

    world, cfg, agent = native.world, native.cfg, native.agent
    settings = replace(cfg.render_cfg, width=size[0], height=size[1], quality="high")
    renderer = LandscapeRenderer(settings)
    renderer._prepare(world)
    eye = agent.pos + (0, cfg.sim_cfg.eye_height, 0)
    cam = Camera(eye, agent.yaw, agent.pitch)
    if hasattr(native, "presentation_camera"):
        camera_eye, camera_target = native.presentation_camera()
        direction = np.asarray(camera_target) - np.asarray(camera_eye)
        cam = Camera(np.asarray(camera_eye, np.float32),
                     float(np.arctan2(direction[2], direction[0])),
                     float(np.arctan2(direction[1], np.linalg.norm(direction[[0, 2]]))))
    elif cfg.render_cfg.third_person:
        cam = resolve_third_person(world, eye, agent.yaw, agent.pitch,
                                   cfg.render_cfg.cam_distance, cfg.render_cfg.cam_pitch_offset)
    forward = cam.basis()[0]
    keys = P.sky_colors(native.elements.climate.time, native.elements.climate.weather)
    fov_y = np.degrees(2 * np.arctan(np.tan(np.radians(cfg.render_cfg.fov) / 2) * size[1] / size[0]))
    scene = SceneSnapshot(dict(version=1, environment="wildlands", seed=native._seed,
        time_of_day=float(native.elements.climate.time), coordinates="Y-up, metres", size=list(size),
        camera=dict(eye=cam.pos.tolist(), target=(cam.pos + forward).tolist(), fov_y=float(fov_y)),
        lighting=dict(direction=keys["sun"]["dir"].tolist(), color=keys["sun"]["color"].tolist(),
                      strength=keys["sun"]["intensity"], sky=keys["zenith"].tolist(), horizon=keys["horizon"].tolist())))
    triangles = _isosurface(renderer._density)
    p = triangles.mean(axis=1)
    normal = _unit(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]))
    block = world.blocks_at(p - normal * 0.6)
    block[~_NATURAL[block]] = B.STONE
    color = P.COLOR_F[block].copy()
    grass = np.clip((normal[:, 1] - 0.65) / 0.3, 0, 1)
    grass[~np.isin(block, (B.GRASS, B.DIRT, B.MOSS_STONE))] = 0
    color = color * (1 - grass[:, None]) + np.array((0.46, 0.59, 0.24)) * grass[:, None]
    scene.add("Terrain with caves", triangles, color, smooth=True, material="terrain")
    triangles, block = _voxel_faces(renderer._detail.vox, renderer._detail.vox != B.AIR)
    scene.add("Architecture", triangles, P.COLOR_F[block], material="stone")
    triangles, _ = _voxel_faces(world.vox, world.vox == B.WATER)
    scene.add("Water", triangles, (0.16, 0.61, 0.66), water=True)
    entities = native.npcs.render_list(agent.pos) + native.bodies.render_list() + native.gameplay.render_list()
    if world.meta.get("theme") == "skylands":
        entities += native._extra_entities()
        scene.metadata["environment"] = "skylands"
    if hasattr(native, "presentation_lights"):
        scene.metadata["point_lights"] = native.presentation_lights()
    if cfg.render_cfg.third_person:
        entities.append(dict(pos=tuple(agent.pos + (0, 0.9, 0)), radius=0.35, height=1.75,
            color=(59, 128, 147), semantic=B.SEM_SELF, yaw=agent.yaw,
            gliding=agent.gliding, phase=agent.distance_travelled * 2.5))
    _add_batch(scene, renderer._meshes(world, cam.pos, forward, entities, native.ticks))
    if hasattr(native, "presentation_meshes"):
        batch = _presentation_batch()
        native.presentation_meshes(batch)
        _add_batch(scene, batch)
        scene.metadata["reference_task"] = native.task
    return scene


def capture_scene(env, *, width=960, height=540, view="presentation"):
    """Capture a reset Teyvat or Wildlands env, including a shared-framework env.

    This is a camera-specific presentation snapshot, not a complete savegame.
    It doesn't advance physics or replace the renderer used for observations.
    """
    from open_world_bench.env import OpenWorldEnv
    from ..env.wildlands import WildlandsEnv

    if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
        raise ValueError("width and height must be positive integers")
    base = env.unwrapped
    native = getattr(base, "native", base)
    if getattr(native, "world", None) is None:
        raise gym.error.ResetNeeded("call reset() before exporting a scene")
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}")
    if isinstance(native, OpenWorldEnv):
        return _teyvat(native, (width, height), view)
    if view != "presentation":
        raise ValueError("first-person scene export is implemented for Teyvat here; "
                         "Skylands campaigns use owb.render.skylands_campaign")
    # Deliberately exclude subclasses with extra scene content until adapted.
    from ..env.wildlands_reference import WildlandsReferenceEnv
    from ..env.skylands import SkylandsEnv
    from ..env.skylands_reference import SkylandsReferenceEnv
    if type(native) in (WildlandsEnv, WildlandsReferenceEnv, SkylandsEnv, SkylandsReferenceEnv):
        return _wildlands(native, (width, height))
    raise ValueError("scene export currently supports teyvat, wildlands, and skylands")


def export_scene(env, path, *, width=960, height=540, view="presentation"):
    return capture_scene(env, width=width, height=height, view=view).save(path)
