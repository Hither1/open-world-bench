"""Painterly Wildlands renderer: continuous terrain and procedural mesh foliage.

The natural surface is the 0.5 isosurface of interpolated voxel occupancy, so
caves, overhangs and edits remain visible. Collision still uses the voxel grid;
rounding is a visual approximation within roughly one voxel of that grid.
Architectural blocks retain exact geometry. No image assets or GPU are needed.
"""
from __future__ import annotations

import numpy as np

from open_world_bench.render import raster

from ..core import blocks as B, noise as N
from ..world.biomes import BIOMES
from ..world.voxel import VoxelWorld
from . import models, palette as P
from .camera import Camera, ray_dirs
from .raycast import Renderer as VoxelRenderer
from .shadows import MeshShadowMap, contact_occlusion

_NATURAL = np.zeros(B.NUM_BLOCKS, bool)
_NATURAL[[B.STONE, B.DIRT, B.GRASS, B.SAND, B.SNOW, B.ICE, B.MUD, B.GRAVEL,
          B.ASH, B.SNOW_PACK, B.MOSS_STONE, B.BASALT, B.LAVA,
          B.ORE_IRON, B.ORE_FLINT]] = True
_MODELED = np.zeros(B.NUM_BLOCKS, bool)
_MODELED[[B.LOG, B.LEAVES, B.TALL_GRASS, B.FLOWER, B.BUSH, B.COOKPOT,
          B.CAMPFIRE, B.CAMPFIRE_LIT, B.WATER, B.FIRE, B.TORCH_WALL,
          B.CRYSTAL, B.VINE, B.GLASS]] = True


class LandscapeRenderer(VoxelRenderer):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        if self.cfg.quality not in ("fast", "balanced", "high"):
            raise ValueError("render_cfg.quality must be 'fast', 'balanced' or 'high'")
        self._quality = ("fast", "balanced", "high").index(self.cfg.quality)
        self._world, self._revision = None, -1
        self._mesh_shadow = None

    def _prepare(self, world):
        if world is self._world and world.rev == self._revision:
            return
        self._world, self._revision = world, world.rev
        self._tree_meshes = {}
        natural = _NATURAL[world.vox]
        density = natural.astype(np.float32)
        # A narrow filter rounds terrace corners without flattening caves.
        for axis in (0, 2):
            pad = [(0, 0)] * 3
            pad[axis] = (1, 1)
            padded = np.pad(density, pad, mode="edge")
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[axis], hi[axis] = slice(0, -2), slice(2, None)
            density = density * 0.84 + (padded[tuple(lo)] + padded[tuple(hi)]) * 0.08
        self._density = np.pad(density, 1)
        ys = np.arange(world.size_y)[None, :, None]
        self._height = np.where(natural, ys + 1, 0).max(axis=1).astype(np.float32)
        pad_x, pad_z = (-world.size_x) % 8, (-world.size_z) % 8
        h = np.pad(self._height, ((0, pad_x), (0, pad_z)), mode="edge")
        self._mip = h.reshape(h.shape[0] // 8, 8, h.shape[1] // 8, 8).max(axis=(1, 3))
        self._detail = VoxelWorld(*world.shape, sea_level=world.sea_level)
        self._detail.vox = np.where(natural | _MODELED[world.vox], B.AIR, world.vox).astype(np.uint8)
        self._detail.recompute_heightmaps()

        logs = world.vox == B.LOG
        below = np.zeros_like(logs)
        below[:, 1:] = logs[:, :-1]
        self._trees = []
        for x, y, z in np.argwhere(logs & ~below):
            top = int(y)
            while top < world.size_y and logs[x, top, z]:
                top += 1
            if top - y < 2:
                # Single placed logs retain geometry; do not silently hide them.
                self._detail.vox[x, y, z] = B.LOG
                continue
            kind = BIOMES[int(world.biome[x, z])].tree_kind
            if world.meta.get("theme") == "skylands" and y > world.meta["layer_heights"][1] + 6:
                kind = "golden"
            leaves = world.vox[max(0, x - 3):x + 4, max(y, top - 4):top + 4,
                               max(0, z - 3):z + 4] == B.LEAVES
            if not leaves.any():
                kind = "dead"
            self._trees.append(((x + 0.5, float(y), z + 0.5), top - y, kind))
        self._detail.recompute_heightmaps()
        self._props = np.argwhere(np.isin(world.vox, [B.BUSH, B.COOKPOT, B.CAMPFIRE, B.CAMPFIRE_LIT,
                                                     B.FIRE, B.TORCH_WALL, B.CRYSTAL, B.VINE, B.GLASS]))
        self._grass = np.argwhere(np.isin(world.vox, [B.TALL_GRASS, B.FLOWER]))
        # Stable, sub-voxel ground cover; density depends on the presentation tier.
        xx, zz = np.indices(self._height.shape)
        yy = np.maximum(self._height.astype(int) - 1, 0)
        cover = (world.vox[xx, yy, zz] == B.GRASS) & (N.hash_unit(xx, zz, 731) > 0.12)
        gx, gz = np.nonzero(cover)
        self._grass = np.concatenate((self._grass, np.stack((gx, yy[gx, gz] + 1, gz), axis=1)))
        self._grass_points = self._grass.astype(np.float32) + (0.5, 0, 0.5)
        repeats = (1, 2, 4)[self._quality]
        self._grass_points = np.repeat(self._grass_points, repeats, axis=0)
        self._grass_flowers = np.repeat(world.vox[tuple(self._grass.T)] == B.FLOWER, repeats)
        i = np.arange(len(self._grass_points))
        self._grass_variation = N.hash_unit(i, i, 12)
        self._grass_points[:, 0] += (N.hash_unit(i, i, 84) - 0.5) * 0.8
        self._grass_points[:, 2] += (N.hash_unit(i, i, 32) - 0.5) * 0.8
        # Set roots on the actual rendered isosurface, including rounded edges.
        lo, hi = self._grass_points.copy(), self._grass_points.copy()
        lo[:, 1] -= 1.2
        hi[:, 1] += 0.6
        supported = self._sample(lo) > 0.5
        for _ in range(7):
            mid = (lo + hi) * 0.5
            under = self._sample(mid) > 0.5
            lo[under], hi[~under] = mid[under], mid[~under]
        self._grass_points[supported, 1] = ((lo + hi) * 0.5)[supported, 1] - 0.03

    def _sample(self, points):
        p = points + 0.5  # voxel centres, including the one-cell padding
        low = np.floor(p).astype(np.int32)
        low = np.clip(low, 0, np.array(self._density.shape) - 2)
        frac = np.clip(p - low, 0, 1)
        x, y, z = low.T
        fx, fy, fz = frac.T
        d = self._density
        a = d[x, y, z] * (1 - fx) + d[x + 1, y, z] * fx
        b = d[x, y + 1, z] * (1 - fx) + d[x + 1, y + 1, z] * fx
        c = d[x, y, z + 1] * (1 - fx) + d[x + 1, y, z + 1] * fx
        e = d[x, y + 1, z + 1] * (1 - fx) + d[x + 1, y + 1, z + 1] * fx
        return (a * (1 - fy) + b * fy) * (1 - fz) + (c * (1 - fy) + e * fy) * fz

    def _surface(self, world, eye, directions, limits):
        # Slab intersection permits showcase cameras outside the world, too.
        safe = np.where(np.abs(directions) > 1e-8, directions, 1e-8)
        a, b = -eye / safe, (np.array(world.shape) - eye) / safe
        enter = np.maximum(np.minimum(a, b).max(axis=1), 0.03)
        leave = np.minimum(np.maximum(a, b).min(axis=1), limits)
        active = np.flatnonzero(enter < leave)
        travel = enter[active].astype(np.float32)
        previous = travel.copy()
        hits = np.full(len(directions), self.cfg.max_dist, np.float32)
        found = np.zeros(len(directions), bool)
        for _ in range(int(self.cfg.max_dist / 0.65) + 4):
            if not len(active):
                break
            d = directions[active]
            p = eye + d * travel[:, None]
            inside = self._sample(p) >= 0.5
            if inside.any():
                ids = active[inside]
                lo, hi = previous[inside].copy(), travel[inside].copy()
                for _ in range(6):
                    mid = (lo + hi) * 0.5
                    under = self._sample(eye + directions[ids] * mid[:, None]) >= 0.5
                    hi, lo = np.where(under, mid, hi), np.where(under, lo, mid)
                hits[ids], found[ids] = hi, True
            keep = ~inside
            active, d, p = active[keep], d[keep], p[keep]
            previous = travel[keep]
            if not len(active):
                break
            xi = np.clip(p[:, 0].astype(int), 0, world.size_x - 1)
            zi = np.clip(p[:, 2].astype(int), 0, world.size_z - 1)
            above = p[:, 1] - self._mip[xi // 8, zi // 8] - 1.5
            offset = p[:, (0, 2)] % 8
            horizontal = d[:, (0, 2)]
            tile_exit = np.min(np.where(horizontal > 0, 8 - offset, offset)
                               / np.maximum(np.abs(horizontal), 1e-6), axis=1)
            drop = np.where(d[:, 1] < 0, above / np.maximum(-d[:, 1], 1e-6), 1e6)
            stride = np.where(above > 0.7, np.maximum(0.65, np.minimum(tile_exit, drop)), 0.65)
            travel = previous + np.minimum(stride, 16)
            keep = travel < leave[active]
            active, travel, previous = active[keep], travel[keep], previous[keep]
        normal = np.zeros((len(directions), 3), np.float32)
        points = eye + directions[found] * hits[found, None]
        if len(points):
            for axis in range(3):
                offset = np.eye(3, dtype=np.float32)[axis] * 0.85
                normal[found, axis] = self._sample(points - offset) - self._sample(points + offset)
            normal[found] /= np.maximum(np.linalg.norm(normal[found], axis=1, keepdims=True), 1e-5)
        return hits, found, normal

    def _fog(self, color, distance, sky, weather):
        if not self.cfg.fog:
            return color
        reach = P.FOG_DISTANCE.get(weather, 150) * 1.35
        amount = 1 - np.exp(-(np.maximum(distance - 9, 0) / reach) ** 1.2)
        return color * (1 - amount[..., None]) + sky["horizon"] * amount[..., None]

    def _lighting(self, base, normal, sky, shadow=1):
        sun = sky["sun"]
        dot = np.sum(normal * sun["dir"], axis=-1)
        # Soft-edged bands instead of per-block face shading and texture noise.
        band = 0.14 + 0.46 * N.smoothstep(-0.15, 0.45, dot) + 0.40 * N.smoothstep(0.40, 0.90, dot)
        shade = np.asarray(shadow)
        direct = band[..., None] * sun["color"] * sun["intensity"] * shade[..., None] * 0.80
        hemi = 0.66 + 0.34 * np.clip(normal[..., 1], 0, 1)
        ambient = np.array((0.50, 0.64, 0.74)) * sun["ambient"] * hemi[..., None]
        return base * (direct + ambient)

    def _sky(self, dirs, sky, weather, time):
        dy = dirs[:, 1]
        blend = np.clip(dy * 2.2, 0, 1) ** 0.65
        color = sky["horizon"] * (1 - blend[:, None]) + sky["zenith"] * blend[:, None]
        alignment = np.clip(dirs @ sky["sun"]["dir"], 0, 1)
        glow = alignment ** 40 * 0.12 + alignment ** 1800 * 1.1
        color += glow[:, None] * sky["sun"]["color"]
        if self.cfg.clouds:
            above = dy > 0.02
            d = dirs[above]
            scale = np.maximum(d[:, 1] + 0.15, 0.05)
            n = N.fbm2(d[:, 0] / scale * 1.4 + time * 0.3,
                       d[:, 2] / scale * 1.4, 608, octaves=3)
            threshold = -0.10 if weather in ("cloud", "rain", "storm", "snow") else 0.035
            alpha = N.smoothstep(threshold, threshold + 0.15, n) * np.clip(d[:, 1] * 8, 0, 1)
            cloud = np.array((0.93, 0.94, 0.84)) * (0.38 + 0.62 * max(sky["sun"]["elevation"], 0) ** 0.3)
            underside = np.clip((n - threshold) * 1.8, 0, 0.14)
            cloud = cloud + underside[:, None]
            color[above] = color[above] * (1 - alpha[:, None]) + cloud * alpha[:, None]
        return color.astype(np.float32)

    def _terrain_shadow(self, points, normals, sun):
        occluded = np.zeros(len(points), bool)
        origins = points + normals * 0.85
        for distance in np.geomspace(1, 36, min(self.cfg.shadow_steps, 12)):
            p = origins + sun * distance
            occluded |= (self._sample(p) > 0.52) | self._detail.solid_at(p)
        return np.where(occluded, 0.30, 1).astype(np.float32)

    def _water_surface(self, world, eye, dirs):
        distance = np.full(len(dirs), self.cfg.max_dist, np.float32)
        thickness = np.zeros(len(dirs), np.float32)
        for level in np.unique(world.water_height):
            if level < 0:
                continue
            dy = np.where(np.abs(dirs[:, 1]) > 1e-6, dirs[:, 1], 1e-6)
            wt = (float(level) + 1 - eye[1]) / dy
            wp = eye + dirs * wt[:, None]
            xi, zi = np.floor(wp[:, 0]).astype(int), np.floor(wp[:, 2]).astype(int)
            valid = (wt > 0) & (wt < distance) & (xi >= 0) & (xi < world.size_x) & (zi >= 0) & (zi < world.size_z)
            indices = np.flatnonzero(valid)
            indices = indices[world.water_height[xi[indices], zi[indices]] == level]
            distance[indices] = wt[indices]
            thickness[indices] = float(level) + 1 - self._height[xi[indices], zi[indices]]
        return distance, thickness

    def _shade_landscape(self, world, eye, dirs, g, sky, weather, lights, time):
        color = self._sky(dirs, sky, weather, time)
        hit = g["blk"] != B.AIR
        if hit.any():
            idx = np.flatnonzero(hit)
            p = eye + dirs[idx] * g["t"][idx, None]
            n, block = g["nrm"][idx], g["blk"][idx]
            base = P.COLOR_F[block].copy()
            natural = _NATURAL[block]
            if natural.any():
                q, nn = p[natural], n[natural]
                x = np.clip(q[:, 0].astype(int), 0, world.size_x - 1)
                z = np.clip(q[:, 2].astype(int), 0, world.size_z - 1)
                bio = world.biome[x, z]
                meadow = np.tile((0.38, 0.52, 0.23), (len(q), 1))
                meadow[bio == 1] = (0.35, 0.48, 0.23)
                meadow[bio == 6] = (0.63, 0.62, 0.30)
                rock = np.tile((0.47, 0.46, 0.39), (len(q), 1))
                rock[bio == 10] = (0.28, 0.31, 0.32)
                rock[bio == 11] = (0.65, 0.43, 0.29)
                if world.meta.get("theme") == "skylands":
                    high = q[:, 1] > world.meta["layer_heights"][1] + 6
                    meadow[high] = (0.63, 0.66, 0.34)
                    rock[high] = (0.61, 0.59, 0.48)
                    low = q[:, 1] < world.meta["layer_heights"][1] - 2
                    rock[low] = (0.22, 0.29, 0.30)
                    meadow[low] = (0.22, 0.29, 0.30)
                grass_weight = N.smoothstep(0.72, 0.94, nn[:, 1])
                grass_weight *= np.clip((q[:, 1] - self._height[x, z] + 2) / 2, 0, 1)
                grass_weight[np.isin(bio, [3, 4, 5, 8, 9, 10, 11])] = 0
                surface = rock * (1 - grass_weight[:, None]) + meadow * grass_weight[:, None]
                sand = np.isin(block[natural], [B.SAND])
                snow = np.isin(block[natural], [B.SNOW, B.SNOW_PACK, B.ICE])
                surface[sand] = (0.77, 0.70, 0.49)
                surface[snow] = (0.84, 0.89, 0.88)
                variation = N.perlin2(q[:, 0] * 0.19, q[:, 2] * 0.19, 65)
                if self._quality:
                    # Continuous world-space materials: irregular moss, soil and
                    # weathered stone. High frequencies fade before aliasing.
                    uv = np.stack((q[:, 0] + q[:, 1] * 0.63,
                                   q[:, 2] + q[:, 1] * 0.47), axis=1)
                    fleck = N.perlin2(uv[:, 0] * 3.2, uv[:, 1] * 3.2, 184)
                    fine = N.perlin2(uv[:, 0] * 14, uv[:, 1] * 14, 936)
                    near = np.clip(1 - g["t"][idx[natural]] / 55, 0, 1)
                    soil = N.smoothstep(0.02, 0.32, variation + fleck * 0.24) * grass_weight * 0.65
                    soil[sand | snow] = 0
                    surface = surface * (1 - soil[:, None]) + np.array((0.32, 0.29, 0.18)) * soil[:, None]
                    cracks = 1 - N.smoothstep(0.012, 0.07, np.abs(N.perlin2(uv[:, 0] * 0.7, uv[:, 1] * 0.7, 442)))
                    surface *= (1 + fleck[:, None] * near[:, None] * 0.34
                                + fine[:, None] * near[:, None] * 0.22
                                - cracks[:, None] * (1 - grass_weight[:, None]) * 0.13)
                surface *= (1 + variation[:, None] * 0.34)
                surface *= (1 + 0.035 * np.sin(q[:, 1] * 2.4 + variation * 5))[:, None]
                base[natural] = surface
            shadow = np.ones(len(p), np.float32)
            if self.cfg.shadows:
                shadow = self._terrain_shadow(p, n, sky["sun"]["dir"])
                if self._mesh_shadow is not None:
                    shadow *= self._mesh_shadow.sample(p + n * 0.08)
            lit = self._lighting(base, n, sky, shadow)
            emission = np.clip(B.EMISSIVE[block], 0, 1)
            lit = lit * (1 - emission[:, None]) + base * emission[:, None]
            if lights:
                lit += base * self._point_lights(p, n, lights)
            lava = block == B.LAVA
            lit[lava] = (1, 0.43, 0.10)
            rim = (1 - np.clip(np.sum(n * -dirs[idx], axis=1), 0, 1)) ** 3
            lit += rim[:, None] * np.array((0.10, 0.10, 0.06)) * sky["sun"]["intensity"]
            color[idx] = self._fog(lit, g["t"][idx], sky, weather)
        return color

    def _meshes(self, world, eye, forward, entities, frame):
        batch = models.ModelBatch()
        reach = min(self.cfg.max_dist, 135)

        def visible(p, radius=2, limit=reach):
            delta = np.asarray(p) - eye
            return np.linalg.norm(delta) < limit + radius and np.dot(delta, forward) > -radius

        for tree_index, (p, height, kind) in enumerate(self._trees):
            distance = np.linalg.norm(np.asarray(p) - eye)
            if distance < 48 or visible(np.asarray(p) + (0, height / 2, 0), height):
                v = float(N.hash_unit(np.array([int(p[0])]), np.array([int(p[2])]), 83)[0])
                pixels = height * self.cfg.width / max(distance, 1)
                detail = min(self._quality, 2 if distance < 22 and pixels > 65
                             else 1 if distance < 55 and pixels > 24 else 0)
                key = (tree_index, detail)
                if key not in self._tree_meshes:
                    mesh = models.ModelBatch()
                    models.tree(mesh, p, height, kind, v, detail)
                    self._tree_meshes[key] = mesh
                batch.extend(self._tree_meshes[key])
        for x, y, z in self._props:
            p = (x + 0.5, y, z + 0.5)
            if visible(p):
                models.prop(batch, world.get(x, y, z), p, frame * 0.1)
        if len(self._grass_points):
            points = self._grass_points
            distance = np.linalg.norm(points - eye, axis=1)
            reach = (15, 25, 34)[self._quality]
            facing = (points - eye) @ forward
            indices = np.flatnonzero((distance < reach) & (facing > distance * 0.45 - 1))
            budget = min((180, 800, 1800)[self._quality],
                         max(50, self.cfg.width * self.cfg.height // 300))
            indices = indices[np.argsort(distance[indices])[:budget]]
            for index in indices:
                p = points[index]
                variation = float(self._grass_variation[index])
                models.grass(batch, p, variation, frame * 0.1,
                             self._grass_flowers[index], self._quality)
        if self.cfg.entity_impostors:
            for e in entities or ():
                if visible(e["pos"], e.get("height", 2)):
                    models.entity(batch, e, frame * 0.1)
        return batch

    def render(self, world, pos, yaw, pitch, *, time_of_day=0.5, weather="clear",
               lights=None, entities=None, flash=0, rain_intensity=0, frame=0):
        self._prepare(world)
        cfg = self.cfg
        ss = max(1, int(cfg.supersample))
        width, height = cfg.width * ss, cfg.height * ss
        cam = Camera(np.asarray(pos, np.float32), yaw, pitch, cfg.fov)
        forward, right, up = cam.basis()
        dirs = ray_dirs(cam, width, height)
        g = self._march(self._detail, cam.pos, dirs, cfg.max_dist, cfg.max_steps, exact=True)
        water_distance, water_depth = self._water_surface(world, cam.pos, dirs)
        distance, found, normal = self._surface(world, cam.pos, dirs, np.minimum(g["t"], water_distance))
        points = cam.pos + dirs[found] * distance[found, None]
        material = world.blocks_at(points - normal[found] * 0.6)
        material[~_NATURAL[material]] = B.STONE
        g["blk"][found], g["t"][found], g["nrm"][found] = material, distance[found], normal[found]
        sky = P.sky_colors(time_of_day, weather)
        # A clear cyan sky and warm cream horizon, softened by existing weather keys.
        if weather in ("clear", "cloud") and sky["sun"]["elevation"] > 0.15:
            sky["zenith"] = np.array((0.28, 0.60, 0.76), np.float32)
            sky["horizon"] = np.array((0.76, 0.83, 0.78), np.float32)
        if world.meta.get("theme") == "skylands" and cam.pos[1] < world.meta["layer_heights"][1] - 2:
            sky["zenith"] = np.array((0.025, 0.045, 0.065), np.float32)
            sky["horizon"] = np.array((0.05, 0.11, 0.12), np.float32)
            sky["sun"] = dict(sky["sun"], intensity=0.18, ambient=0.45,
                               color=np.array((0.40, 0.74, 0.80), np.float32))
        batch = self._meshes(world, cam.pos, forward, entities, frame)
        verts, colors, emission, mesh_ids = batch.finish()
        self._mesh_shadow = None
        if cfg.shadows and self._quality and len(verts):
            # Ground-cover shadows are covered by AO. Keep larger foliage and
            # characters, including trees behind the camera that cast into view.
            caster = (mesh_ids != B.SEM_PLANT) & (emission < 0.5)
            self._mesh_shadow = MeshShadowMap(verts[caster], sky["sun"]["dir"],
                                              cam.pos + forward * 14, extent=48,
                                              resolution=384 if self._quality == 1 else 768)
        color = self._shade_landscape(world, cam.pos, dirs, g, sky, weather, lights, time_of_day)
        semantic = B.SEMANTIC[g["blk"]].copy()
        wet = water_distance < g["t"]
        if wet.any():
            wt = water_distance[wet]
            p = cam.pos + dirs[wet] * wt[:, None]
            thickness = np.clip(water_depth[wet], 0, 14)
            deep = 1 - np.exp(-thickness * 0.22)
            water = np.array((0.26, 0.64, 0.62)) * (1 - deep[:, None]) + np.array((0.13, 0.42, 0.52)) * deep[:, None]
            fresnel = (1 - np.abs(dirs[wet, 1])) ** 4 * 0.4
            water = water * (1 - fresnel[:, None]) + sky["horizon"] * fresnel[:, None]
            waves = np.sin(p[:, 0] * 2.3 + p[:, 2] * 1.6 + frame * 0.08)
            foam = (thickness < 0.25) * (0.08 + 0.08 * np.maximum(waves, 0))
            ripple = np.stack((np.cos(p[:, 0] * 1.7 + frame * 0.1) * 0.09,
                               np.ones(len(p)), np.sin(p[:, 2] * 2.1 + frame * 0.07) * 0.09), axis=1)
            ripple /= np.linalg.norm(ripple, axis=1, keepdims=True)
            half = sky["sun"]["dir"] - dirs[wet]
            half /= np.maximum(np.linalg.norm(half, axis=1, keepdims=True), 1e-6)
            glint = np.maximum(np.sum(ripple * half, axis=1), 0) ** 120
            water += (np.maximum(waves - 0.97, 0) * 0.5 + foam + glint * 0.55)[:, None]
            if self._mesh_shadow is not None:
                water *= (0.7 + 0.3 * self._mesh_shadow.sample(p))[:, None]
            water *= 0.35 + 0.65 * min(sky["sun"]["intensity"], 1)
            color[wet] = self._fog(water, wt, sky, weather)
            g["t"][wet], semantic[wet] = wt, B.SEM_WATER
        # Rasterizer depth is along the camera axis; public depth is ray distance.
        cosine = dirs @ forward
        depth = (g["t"] * cosine).reshape(height, width).copy()
        color = color.reshape(height, width, 3).astype(np.float32)
        normal_buf = g["nrm"].reshape(height, width, 3).copy()
        ids = semantic.reshape(height, width).astype(np.int32)
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = (right, up, -forward)
        view[:3, 3] = -view[:3, :3] @ cam.pos
        tan = np.tan(np.deg2rad(cfg.fov) / 2)
        proj = np.zeros((4, 4), np.float32)
        proj[0, 0], proj[1, 1], proj[2, 2], proj[2, 3], proj[3, 2] = 1 / tan, width / height / tan, -1, -0.1, -1
        if len(verts):
            screen, wdepth, keep = raster.project(verts, view, proj, width, height)
            # Offscreen casters are needed in the shadow map, but should not
            # pay for per-vertex lighting, fog and terrain shadow tracing.
            verts, colors, emission, mesh_ids = verts[keep], colors[keep], emission[keep], mesh_ids[keep]
            smooth = np.concatenate(batch.smooth_normals)[keep]
            screen, wdepth = screen[keep], wdepth[keep]
            keep = np.ones(len(verts), bool)
            normals = raster.face_normals(verts, cam.pos)
            smooth = np.where(np.linalg.norm(smooth, axis=-1, keepdims=True) > 0,
                              smooth, normals[:, None, :])
            mesh_shadow = np.ones(verts.shape[:2], np.float32)
            if self._mesh_shadow is not None:
                mesh_shadow = self._mesh_shadow.sample(verts + smooth * 0.10, bias=0.16)
            if cfg.shadows and self._quality:
                terrain_shadow = self._terrain_shadow(verts.mean(axis=1), normals, sky["sun"]["dir"])
                mesh_shadow *= terrain_shadow[:, None]
            shaded = self._lighting(colors[:, None, :], smooth, sky, mesh_shadow)
            # Gentle transmission keeps thin foliage luminous against the sun.
            leaves = np.isin(mesh_ids, (B.SEM_TREE, B.SEM_PLANT))
            transmission = np.maximum(np.sum(-smooth * sky["sun"]["dir"], axis=-1), 0) * 0.14
            shaded += colors[:, None, :] * (leaves[:, None] * transmission * sky["sun"]["intensity"])[..., None]
            shaded = shaded * (1 - emission[:, None, None]) + colors[:, None, :] * emission[:, None, None]
            shaded = self._fog(shaded, np.linalg.norm(verts.mean(axis=1) - cam.pos, axis=1)[:, None], sky, weather)
            big, tiny = raster.split_by_screen_size(screen, keep, threshold=1.2)
            args = (shaded, normals, mesh_ids, color, depth, normal_buf, ids)
            raster.draw_triangles(screen, wdepth, big, *args)
            raster.draw_point_sprites(screen, wdepth, tiny, shaded.mean(axis=1), normals, mesh_ids,
                                      color, depth, normal_buf, ids)
        depth = np.clip(depth / cosine.reshape(height, width), 0, cfg.max_dist)
        if cfg.ambient_occlusion and self._quality:
            solid = (ids != B.SEM_SKY) & (ids != B.SEM_WATER)
            color = contact_occlusion(color, depth, dirs, normal_buf, solid, width / (2 * tan))
        if world.get_f(*cam.pos) == B.WATER:
            murk = 1 - np.exp(-depth * 0.10)
            color = color * (1 - 0.75 * murk[..., None]) + np.array((0.10, 0.31, 0.36)) * (0.75 * murk[..., None])
        # Light color grading, no voxel hash noise or screen-space blur.
        color = color * np.array((1.025, 1.015, 0.97))
        rgb = np.clip(color * 255, 0, 255).astype(np.uint8)
        if weather in ("rain", "storm", "snow") or rain_intensity > 0:
            self._weather_overlay(rgb, weather, rain_intensity, frame)
        if flash:
            rgb = np.clip(rgb.astype(np.float32) + 255 * flash, 0, 255).astype(np.uint8)
        if ss > 1:
            rgb = rgb.reshape(cfg.height, ss, cfg.width, ss, 3).mean(axis=(1, 3)).astype(np.uint8)
            depth = depth.reshape(cfg.height, ss, cfg.width, ss)[:, ss // 2, :, ss // 2]
            ids = ids.reshape(cfg.height, ss, cfg.width, ss)[:, ss // 2, :, ss // 2]
        return dict(rgb=np.ascontiguousarray(rgb), depth=np.ascontiguousarray(depth.astype(np.float32)),
                    semantic=np.ascontiguousarray(ids.astype(np.uint8)))
