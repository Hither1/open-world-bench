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
    textures: dict = field(default_factory=dict)

    def add_texture(self, name, pixels, **mapping):
        """A uint8 RGBA image plus its world mapping (e.g. ground detail)."""
        pixels = np.ascontiguousarray(pixels, np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError("textures must be (rows, cols, 4) uint8")
        self.textures[name] = (pixels, dict(mapping))

    def add(self, name, triangles, colors, *, normals=None, emission=None, smooth=False, water=False,
            material="surface", occluder=False, blend=False):
        """``blend`` averages the per-triangle colours onto shared vertices in
        the renderer, so natural terrain shades smoothly instead of flipping
        colour from one small triangle to the next."""
        triangles = np.asarray(triangles, np.float32).reshape(-1, 3, 3)
        if not len(triangles):
            return
        colors = np.broadcast_to(colors, (len(triangles), 3)).astype(np.float32).copy()
        self.meshes.append(dict(name=name, triangles=triangles.copy(), colors=colors,
                                normals=np.zeros_like(triangles) if normals is None else np.asarray(normals, np.float32).copy(),
                                emission=np.zeros(len(triangles), np.float32) if emission is None else np.asarray(emission, np.float32).copy(),
                                smooth=bool(smooth), water=bool(water), material=material,
                                occluder=bool(occluder), blend=bool(blend)))

    def save(self, path):
        """Write a new snapshot, refusing to overwrite an existing artifact."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata, meshes=[])
        arrays = {}
        if self.textures:
            metadata["textures"] = {}
            for name, (pixels, mapping) in self.textures.items():
                metadata["textures"][name] = mapping
                arrays[f"texture_{name}"] = pixels
        for i, mesh in enumerate(self.meshes):
            spec = {k: mesh[k] for k in ("name", "smooth", "water", "material")}
            for flag in ("occluder", "prototype", "blend"):
                if mesh.get(flag):
                    spec[flag] = True
            metadata["meshes"].append(spec)
            for key in ("triangles", "colors", "normals", "emission"):
                arrays[f"mesh_{i}_{key}"] = mesh[key]
        with path.open("xb") as handle:
            np.savez_compressed(handle, metadata=json.dumps(metadata), **arrays)
        return path


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def natural_terrain_colors(lookup, triangles):
    """Per-triangle colour of a smooth natural-terrain mesh.

    ``lookup(points)`` returns the block ids at world points. The block just
    inside the surface sets the colour; where that sample lands in air or a
    built block (common on smoothed one-block risers), deeper and straight-down
    samples are tried before falling back to stone, so risers do not show as
    grey seams. Level and moderately sloped soil reads as grass.
    """
    from .landscape import _NATURAL
    from . import palette as P
    from ..core import blocks as B
    p = triangles.mean(axis=1)
    n = _unit(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]))
    down = np.array((0.0, 1.0, 0.0), np.float32)
    block = np.asarray(lookup(p - n * 0.6)).copy()
    for probe in (lambda q, m: q[m] - n[m] * 1.4, lambda q, m: q[m] - down * 0.6, lambda q, m: q[m] - down * 1.6):
        missing = ~_NATURAL[block]
        if not missing.any():
            break
        block[missing] = lookup(probe(p, missing))
    block[~_NATURAL[block]] = B.STONE
    color = P.COLOR_F[block].copy()
    grass = np.clip((n[:, 1] - 0.5) / 0.35, 0, 1)
    grass[~np.isin(block, (B.GRASS, B.DIRT, B.MOSS_STONE))] = 0
    return color * (1 - grass[:, None]) + np.array((0.46, 0.59, 0.24)) * grass[:, None]


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


def _water_surface(world):
    """Smooth water surface following the voxel water exactly where it is
    open to the air, with rounded outlines; it extends a cell into adjacent
    ground so shorelines meet the terrain without seams."""
    from scipy import ndimage
    from ..core import blocks as B
    water = world.vox == B.WATER
    if not water.any():
        return np.empty((0, 3, 3), np.float32)
    solid = B.SOLID[world.vox]
    extended = water.copy()
    for axis in (0, 2):
        for shift in (1, -1):
            extended |= np.roll(water, shift, axis=axis) & solid
    extended |= np.roll(water, 1, axis=1) & solid   # bed under the water
    lo = np.maximum(np.argwhere(extended).min(0) - 3, 0)
    hi = np.minimum(np.argwhere(extended).max(0) + 4, extended.shape)
    sub = extended[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    density = ndimage.gaussian_filter(np.pad(sub, 1), sigma=(0.8, 0.3, 0.8))
    triangles = _isosurface(density) + lo.astype(np.float32)
    # Sit a few centimetres below the physical top so flush banks never
    # share a plane with the water.
    triangles[..., 1] -= 0.06
    return triangles.astype(np.float32)


def _add_batch(scene, batch, faded=(), fade_triangles=None):
    """Export a batch grouped by material. Triangles whose object id is in
    ``faded`` (or selected by ``fade_triangles(triangles, ids)``) become
    separate occluder meshes the presentation fades for camera rays only."""
    triangles, colors, emission, ids = batch.finish()
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
        fade=np.isin(ids, np.asarray(sorted(faded), np.int64)) if len(faded) else np.zeros(len(ids), bool)
        if fade_triangles is not None:
            fade|=fade_triangles(triangles, ids)
        for material_id in np.unique(material_ids):
            name=MATERIAL_NAMES[int(material_id)]
            for occluder in (False, True):
                mask=(material_ids==material_id)&(fade==occluder)
                if not mask.any():
                    continue
                label=("Faded occluders: " if occluder else "Characters and props: ")+name
                scene.add(label, triangles[mask], colors[mask], normals=normals[mask],
                          emission=emission[mask], material=name, occluder=occluder)


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
    color = _graded_meadow(color, p)
    scene.add("Island terrain", triangles, color, smooth=True, material="terrain")
    if world.architecture is not None:
        scene.metadata["architecture"] = dict(
            revision=1, collision="structural boxes; roofs are not traversal platforms",
            buildings=[b.metadata() for b in world.architecture.buildings])
    extent = terrain.cfg.size * 4
    plane = np.array([[[-extent, terrain.cfg.sea_level, -extent], [-extent, terrain.cfg.sea_level, extent]],
                      [[extent, terrain.cfg.sea_level, -extent], [extent, terrain.cfg.sea_level, extent]]])
    scene.add("Ocean", _grid_triangles(plane), (0.16, 0.61, 0.66), water=True)
    course = getattr(world, "course", None)
    layout = None
    if course is not None:
        from . import dressing
        layout = dressing.course_layout(world)
        scene.metadata["presentation_landscape"] = dressing.layout_summary(layout)
        scene.add_texture("ground_detail", layout.mask_image, **layout.mask)
        prototypes = []
        for k in range(dressing.CARPET_VARIANTS):
            vertices, colors, normals, _, _ = dressing.carpet_template(k)
            name = f"Meadow grass prototype {k}"
            scene.add(name, vertices, colors, normals=normals, material="foliage")
            scene.meshes[-1]["prototype"] = True
            prototypes.append(name)
        scene.metadata["ground_cover"] = dict(target="Island terrain", prototypes=prototypes,
                                              radius=40.0, density=10.0, seed=int(native._episode_seed))
        _add_ridges(scene, ("teyvat-course", native._episode_seed),
                    [dict(center=(0, 0), half_extent=300, inner_gap=40, depth=520, floor=-14, peak=92,
                          gaps=True, shoreline=terrain.cfg.sea_level + .4),
                     dict(center=(0, 0), half_extent=300, inner_gap=640, depth=900, floor=-10, peak=270,
                          gaps=False, snow=205)])
    else:
        half = terrain.cfg.size / 2
        _add_ridges(scene, ("teyvat-island", native._episode_seed),
                    [dict(center=(0, 0), half_extent=half, inner_gap=half * .55, depth=half * 1.4, floor=-16,
                          peak=150, gaps=True, shoreline=terrain.cfg.sea_level + .4, snow=230)])
    scene.metadata["haze"] = dict(start=70.0, distance=1100.0, amount=.85)
    # Observation budgets must not reduce presentation geometry or vegetation.
    settings = replace(native.cfg.render, width=size[0], height=size[1],
                       draw_props=True, draw_entities=True,
                       # A caller that draws its own character sets ``presentation_hides_character``.
                       draw_character=view != "first_person" and not getattr(native, "presentation_hides_character", False),
                       ground_cover=layout is None,
                       prop_draw_distance=max(260, native.cfg.render.prop_draw_distance))
    forward = _unit(target - eye)
    batch = SoftRenderer(settings)._build_meshes(
        world, native.char, eye, forward, batch=_presentation_batch(), full_detail=True)
    faded = set()
    head = native.char.pos + np.array((0.0, 1.25, 0.0))
    third_person = view == "presentation"
    if third_person:
        for n, entity in enumerate(world.entities.values()):
            if entity.alive and _occludes(eye, head, entity.pos[[0, 2]], entity.pos[1], 3.6, 1.15):
                faded.add(100 + n)
    if layout is not None:
        from . import dressing
        occludes = None
        if third_person:
            occludes = lambda xz, base, height, radius: _occludes(eye, head, xz, base, height, radius)
        scene.metadata["presentation_landscape"]["visible"] = dressing.emit_pieces(
            layout, batch, eye, forward, occludes=occludes)
        dressing.meadow_tufts(layout, batch, eye)
        faded.add(dressing.OBJ_FADED)
    if faded:
        scene.metadata["faded_occluders"] = len(faded)
    _add_batch(scene, batch, faded)
    return scene


def _graded_meadow(color, points):
    """Presentation grading: deeper, less saturated grass with restrained variation."""
    from .dressing import fbm
    luminance = color @ np.array((.3, .55, .15))
    graded = (color * .82 + luminance[:, None] * .18) * .8
    variation = fbm(points[:, 0], points[:, 2], 55, 3, salt=91)[:, None]
    dry = np.array((.40, .41, .23))
    greenness = np.clip((color[:, 1] - color[:, 0]) * 4, 0, 1)[:, None]
    graded = graded * (1 - .35 * variation * greenness) + dry * .35 * variation * greenness
    return np.clip(graded * (.94 + .12 * fbm(points[:, 0], points[:, 2], 9, 2, salt=92))[:, None], 0, 1)


def _occludes(eye, head, center_xz, base_y, height, radius):
    """Whether a vertical cylinder blocks the camera's line to the character."""
    t = np.linspace(.04, .9, 32)[:, None]
    line = eye + (head - eye) * t
    near = np.linalg.norm(line[:, [0, 2]] - np.asarray(center_xz), axis=1) < radius
    within = (line[:, 1] > base_y - .2) & (line[:, 1] < base_y + height)
    return bool(np.any(near & within))


_RIDGES = {}


def _add_ridges(scene, identity, layers):
    """Horizon ridges outside the walkable world; cached per world identity."""
    from .dressing import horizon_ridges
    from open_world_bench.render.meshes import MeshBatch
    key = (identity, repr(layers))
    if key not in _RIDGES:
        if len(_RIDGES) > 6:
            _RIDGES.clear()
        batch = MeshBatch()
        for n, layer in enumerate(layers):
            horizon_ridges(batch, identity=(identity, n), **layer)
        triangles, colors, _, _ = batch.finish()
        _RIDGES[key] = (triangles, colors, np.concatenate(batch.normals))
    triangles, colors, normals = _RIDGES[key]
    scene.add("Horizon ridges (presentation only)", triangles, colors, normals=normals, material="terrain")
    scene.metadata.setdefault("presentation_landscape", {})["horizon_layers"] = len(layers)


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
    color = natural_terrain_colors(world.blocks_at, triangles)
    scene.add("Terrain with caves", triangles, color, smooth=True, material="terrain", blend=True)
    triangles, block = _voxel_faces(renderer._detail.vox, renderer._detail.vox != B.AIR)
    scene.add("Architecture", triangles, P.COLOR_F[block], material="stone")
    # Presentation grade for dusk and night: a readable moonlit scene. Policy
    # observations and the simulation are unaffected.
    sun_elevation = float(P.sky_colors(native.elements.climate.time)["sun"]["elevation"])
    if getattr(native, "presentation_kind", "") == "wildlands_reference" and sun_elevation < 0.08:
        k = float(np.clip((0.08 - sun_elevation) / 0.25, 0, 1))
        scene.metadata["night_grade"] = dict(moon=1 + 2.0 * k, sky=1 + 3.5 * k, exposure=0.7 * k,
                                             tint=[0.62, 0.72, 1.0])
    triangles = _water_surface(world)
    scene.add("Water", triangles, (0.16, 0.61, 0.66), smooth=True, water=True)
    scene.metadata["water_volume"] = True
    from . import creatures
    living = native.npcs.render_list(agent.pos)
    drawn = [e for e in living if e.get("kind") in creatures.DRAW or e.get("kind") in creatures.MONSTERS
             or e.get("kind") in ("bird", "fish", "frog")]
    pickups = native.gameplay.render_list()
    apples = [e for e in pickups if e.get("item") == "apple"]
    entities = ([e for e in living if e not in drawn] + native.bodies.render_list()
                + [e for e in pickups if e.get("item") != "apple"])
    if world.meta.get("theme") == "skylands":
        entities += native._extra_entities()
        scene.metadata["environment"] = "skylands"
    if hasattr(native, "presentation_lights"):
        scene.metadata["point_lights"] = native.presentation_lights()
    riding = getattr(native, "riding", None)
    mounted = riding is not None and riding.mounted
    if cfg.render_cfg.third_person and not mounted:
        entities.append(dict(pos=tuple(agent.pos + (0, 0.9, 0)), radius=0.35, height=1.75,
            color=(59, 128, 147), semantic=B.SEM_SELF, yaw=agent.yaw,
            gliding=agent.gliding, phase=agent.distance_travelled * 2.5))
    batch = renderer._meshes(world, cam.pos, forward, entities, native.ticks)
    night = bool(native.elements.climate.is_night)
    seat = creatures.emit(batch, drawn, float(native.elapsed), night=night)
    if mounted and seat is not None and cfg.render_cfg.third_person:
        creatures.rider(batch, seat[0], seat[1], np.array((59, 128, 147)) / 255, agent.distance_travelled)
    dens = {tuple(n.data["den"]) for n in native.npcs.npcs if n.alive and n.data.get("den")}
    for den in sorted(dens):
        if np.linalg.norm(np.asarray(den) - cam.pos) < 150:
            creatures.den(batch, den, ("den", int(native._seed)) + tuple(np.round(den, 1)))
    orchard = getattr(native, "orchard", None)
    if orchard is not None:
        for tree in orchard.trees:
            if tree["apples"] and np.linalg.norm(np.asarray(tree["trunk"]) - cam.pos) < 150:
                creatures.apples_on_tree(batch, tree, tree["apples"])
    for e in apples:
        creatures.apple(batch, np.asarray(e["pos"]) - (0, 0.45, 0))
    _wildlands_landscape(scene, native, batch, cam.pos, forward)
    focus = agent.pos + np.array((0.0, 1.0, 0.0))
    fade = None
    if cfg.render_cfg.third_person or hasattr(native, "presentation_camera"):
        from .dressing import OBJ_FADED
        def fade(triangles, ids, eye=np.asarray(cam.pos, float), head=focus):
            # Foliage and props between the presentation camera and the traveler.
            centre = triangles.mean(axis=1).astype(float)
            line = head - eye
            t = np.clip(((centre - eye) @ line) / max(line @ line, 1e-9), 0, 1)
            near = np.linalg.norm(centre - (eye + t[:, None] * line), axis=1) < 2.2
            tall = np.isin(ids, (B.SEM_TREE, B.SEM_ROCK, B.SEM_STRUCTURE, 11))
            return (near & tall & (t > .02) & (t < .9)) | (ids == OBJ_FADED)
        scene.metadata["faded_occluders"] = "camera-to-traveler cylinder, camera rays only"
    _add_batch(scene, batch, fade_triangles=fade)
    if hasattr(native, "presentation_meshes"):
        batch = _presentation_batch()
        native.presentation_meshes(batch)
        _add_batch(scene, batch)
        scene.metadata["reference_task"] = native.task
    return scene


def _wildlands_landscape(scene, native, batch, eye, forward):
    """Presentation landscape around voxel worlds; never collision or observation."""
    from . import dressing
    world = native.world
    half = max(world.size_x, world.size_z) / 2
    center = (world.size_x / 2, world.size_z / 2)
    sea = float(world.sea_level)
    _add_ridges(scene, ("voxel", type(native).__name__, int(native._seed), world.size_x, world.size_z),
                [dict(center=center, half_extent=half, inner_gap=max(10, half * .12), depth=max(200, half * 2.5),
                      floor=sea - 6, peak=sea + max(55, half * .8), gaps=True, shoreline=sea + .4),
                 dict(center=center, half_extent=half, inner_gap=max(260, half * 3.2), depth=max(420, half * 5),
                      floor=sea - 4, peak=sea + max(170, half * 2.2), gaps=False, snow=sea + max(140, half * 1.8))])
    # Open sea around the square world (not under it: caves stay dry).
    far = half * 14
    x0, x1, z0, z1 = 0.0, float(world.size_x), 0.0, float(world.size_z)
    cx, cz = center
    quads = [((cx - far, cz - far), (x0, cz + far)), ((x1, cz - far), (cx + far, cz + far)),
             ((x0, cz - far), (x1, z0)), ((x0, z1), (x1, cz + far))]
    for (a, b), (c, d) in quads:
        plane = np.array([[[a, sea + .45, b], [a, sea + .45, d]], [[c, sea + .45, b], [c, sea + .45, d]]])
        scene.add("Open sea (presentation only)", _grid_triangles(plane), (0.16, 0.61, 0.66), water=True)
    scene.metadata["haze"] = dict(start=45.0, distance=max(520.0, half * 6), amount=.82)
    if type(native).__name__ == "WildlandsReferenceEnv":
        key = ("wildlands-reference", int(native._seed), native.task)
        layout = dressing._cached(key, lambda: dressing.wildlands_reference_layout(native))
        scene.add_texture("ground_detail", layout.mask_image, **layout.mask)
        dressing.emit_pieces(layout, batch, eye, forward, max_distance=260, near_detail=60)
        dressing.wildlands_course_trim(native, batch)
        summary = dressing.layout_summary(layout)
        scene.metadata["presentation_landscape"] = dict(scene.metadata.get("presentation_landscape", {}), **summary)


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
