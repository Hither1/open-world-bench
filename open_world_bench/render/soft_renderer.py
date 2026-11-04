"""Pure-numpy cel-shaded renderer: raymarched terrain, rasterized characters.

Why a software renderer at all: this environment has to produce pixel
observations on cluster nodes that have no GPU and no GL driver. The terrain is
a heightmap, so it can be raymarched per pixel in vectorized numpy, and the
handful of actual meshes (character, entities, near props) go through a small
z-buffered rasterizer. Everything is seeded and deterministic.
"""
from __future__ import annotations

import numpy as np

from .. import math3d as mth
from .. import noise as nz
from ..character import CharacterState, MoveState
from ..scatter import BUSH, GRASS, PROP_NAMES
from ..terrain import BEACH, CLIFF, FOREST, HIGHLAND, MEADOW, SNOW
from . import raster
from .base import Frame, Renderer, RenderSettings
from .meshes import MeshBatch, add_character, add_entity, add_prop
from .style import BIOME_ALBEDO, BIOME_SHADOW, ToonStyle, grade, sky_keys_at, toon_ramp

_SUN_COLOR_WARM = np.array([1.0, 0.94, 0.84], dtype=np.float32)
# Props that are drawn as geometry; ground cover is folded into terrain colour.
_MESH_PROPS = tuple(i for i, n in enumerate(PROP_NAMES) if i != GRASS)


class SoftRenderer(Renderer):
    def __init__(self, settings: RenderSettings | None = None):
        super().__init__(settings)
        self._ray_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ rays
    def _camera_plane(self, w: int, h: int, fov_y: float):
        key = (w, h, round(fov_y, 4))
        if key not in self._ray_cache:
            tan_y = float(np.tan(np.radians(fov_y) * 0.5))
            tan_x = tan_y * (w / max(h, 1))
            u = (2.0 * (np.arange(w, dtype=np.float32) + 0.5) / w - 1.0) * tan_x
            v = (1.0 - 2.0 * (np.arange(h, dtype=np.float32) + 0.5) / h) * tan_y
            self._ray_cache[key] = (u, v)
        return self._ray_cache[key]

    def _ray_dirs(self, view: np.ndarray, w: int, h: int, fov_y: float) -> np.ndarray:
        u, v = self._camera_plane(w, h, fov_y)
        right, up, fwd = view[0, :3], view[1, :3], -view[2, :3]
        d = (fwd[None, None, :]
             + u[None, :, None] * right[None, None, :]
             + v[:, None, None] * up[None, None, :])
        return (d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-8)).astype(np.float32)

    # -------------------------------------------------------------- marching
    def _march(self, terrain, eye: np.ndarray, dirs: np.ndarray):
        """Per-pixel heightmap raymarch with active-set compaction."""
        s = self.settings
        n = dirs.shape[0]
        t_hit = np.full(n, np.inf, dtype=np.float32)
        t_prev = np.zeros(n, dtype=np.float32)
        far = float(s.max_view_distance)
        top = float(terrain.height.max()) + 2.0
        active = np.arange(n, dtype=np.int64)
        # Rays leaving upward above the highest peak can never hit the ground.
        sky_up = (dirs[:, 1] > 0.0) & (eye[1] > top)
        active = active[~sky_up[active]]
        steps = max(8, int(s.march_steps))
        ramp = 0.25 + (np.arange(1, steps + 1, dtype=np.float32) / steps) ** 2.0 * far
        prev = 0.0
        for tk in ramp:
            if len(active) == 0:
                break
            d = dirs[active]
            p = eye[None, :] + d * tk
            gh = terrain.height_at(p[:, 0], p[:, 2])
            below = p[:, 1] <= gh
            if below.any():
                sel = active[below]
                t_hit[sel] = tk
                t_prev[sel] = prev
            # Drop rays that hit, or that are above the terrain ceiling going up.
            escaped = (~below) & (p[:, 1] > top) & (d[:, 1] > 0.0)
            active = active[~(below | escaped)]
            prev = float(tk)

        # Bisection refinement on the pixels that hit something.
        got = np.isfinite(t_hit)
        if got.any():
            lo = t_prev[got].copy()
            hi = t_hit[got].copy()
            d = dirs[got]
            for _ in range(max(0, int(s.refine_steps))):
                mid = (lo + hi) * 0.5
                p = eye[None, :] + d * mid[:, None]
                under = p[:, 1] <= terrain.height_at(p[:, 0], p[:, 2])
                hi = np.where(under, mid, hi)
                lo = np.where(under, lo, mid)
            t_hit[got] = hi
        return t_hit, got

    # ---------------------------------------------------------------- shading
    def _shade_terrain(self, world, pos: np.ndarray, dirs: np.ndarray, eye: np.ndarray,
                       sun: np.ndarray, keys, style: ToonStyle) -> np.ndarray:
        terrain = world.terrain
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        nrm = terrain.normal_at(x, z)
        biome = terrain.biome_at(x, z)
        shadow_tone = BIOME_SHADOW[biome]
        slope = np.arccos(np.clip(nrm[:, 1], -1.0, 1.0))
        moist = terrain.moisture_at(x, z)

        # Continuous material blend: grass -> forest -> rock -> snow -> sand.
        # Blending weights rather than snapping to a biome id keeps large
        # slopes from breaking up into speckle.
        meadow = BIOME_ALBEDO[MEADOW][None, :]
        forest = BIOME_ALBEDO[FOREST][None, :]
        wet = mth.smoothstep(0.44, 0.66, moist)[:, None]
        albedo = meadow * (1.0 - wet) + forest * wet

        rock_w = mth.smoothstep(np.radians(30.0), np.radians(52.0), slope)[:, None]
        albedo = albedo * (1.0 - rock_w) + BIOME_ALBEDO[CLIFF][None, :] * rock_w

        high = mth.smoothstep(terrain.cfg.tree_line - 30.0, terrain.cfg.tree_line + 25.0, y)[:, None]
        albedo = albedo * (1.0 - high * 0.75) + BIOME_ALBEDO[HIGHLAND][None, :] * (high * 0.75)

        snow_w = (mth.smoothstep(terrain.cfg.snow_line - 30.0, terrain.cfg.snow_line + 4.0, y)
                  * np.clip(1.0 - slope / np.radians(56.0), 0.0, 1.0))[:, None]
        albedo = albedo * (1.0 - snow_w) + BIOME_ALBEDO[SNOW][None, :] * snow_w

        sand_w = (mth.smoothstep(4.2, 0.6, y) * np.clip(1.0 - slope / np.radians(34.0), 0.0, 1.0))[:, None]
        albedo = albedo * (1.0 - sand_w) + BIOME_ALBEDO[BEACH][None, :] * sand_w

        # Gentle macro variation only -- enough to break up flat fields.
        var = nz.fbm(x / 34.0, z / 34.0, world.cfg.seed + 601, octaves=3)
        albedo = albedo * (1.0 + 0.11 * var)[:, None]

        # Terrace lips catch the light: a cheap stylized edge highlight.
        step = max(terrain.cfg.terrace_step, 1e-3)
        lip = 1.0 - np.abs(((y / step) % 1.0) - 0.5) * 2.0
        albedo *= (1.0 + 0.09 * np.clip(lip, 0.0, 1.0) * (slope > np.radians(26.0)))[:, None]

        ndl = np.sum(nrm * sun[None, :], axis=1)
        ramp = 0.55 * toon_ramp(ndl, style) + 0.45 * (0.5 + 0.5 * np.clip(ndl, 0, 1))
        shade = self._sun_shadow(terrain, pos, sun) if self.settings.shadow_steps else np.ones(len(pos), np.float32)
        ramp = ramp * (0.62 + 0.38 * shade)

        sun_col = np.asarray(keys.sun, dtype=np.float32) * keys.sun_strength
        ambient = np.asarray(keys.zenith, dtype=np.float32) * (keys.ambient + 0.25)
        sky_bounce = style.sky_bounce * np.clip(nrm[:, 1], 0.0, 1.0)[:, None] * np.asarray(keys.zenith, np.float32)

        indirect = 0.27 + keys.ambient * 0.35
        lit = albedo * (sun_col[None, :] * ramp[:, None] * 0.72 + indirect)
        lit += shadow_tone * ambient[None, :] * 0.18 + albedo * sky_bounce * 0.45
        # Anime rim light along silhouette edges facing away from the camera.
        view_dir = -dirs
        rim = np.clip(1.0 - np.sum(nrm * view_dir, axis=1), 0.0, 1.0) ** style.rim_power
        rim = rim * style.rim_strength * np.clip(ndl + 0.35, 0.0, 1.0)
        lit = lit + rim[:, None] * sun_col[None, :]
        return lit.astype(np.float32), nrm

    def _sun_shadow(self, terrain, pos: np.ndarray, sun: np.ndarray) -> np.ndarray:
        """Soft-ish cast shadows from a short march toward the sun.

        Only the first ~110 m matter: beyond that the shadow terms turn into
        high-frequency noise that reads as dirt rather than shape.
        """
        steps = int(self.settings.shadow_steps)
        if steps <= 0:
            return np.ones(len(pos), dtype=np.float32)
        if sun[1] <= 0.05:
            return np.full(len(pos), 0.35, dtype=np.float32)
        occluded = np.zeros(len(pos), dtype=np.float32)
        p0 = pos + np.array([0.0, 0.6, 0.0], dtype=np.float32)[None, :]
        dists = np.geomspace(2.0, 110.0, steps).astype(np.float32)
        for dist in dists:
            q = p0 + sun[None, :] * dist
            gap = terrain.height_at(q[:, 0], q[:, 2]) - q[:, 1]
            # Partial occlusion accumulates, so shadow edges are not binary.
            occluded = np.maximum(occluded, np.clip(gap / 3.0, 0.0, 1.0))
        return (1.0 - 0.85 * occluded).astype(np.float32)

    # ------------------------------------------------------------------- sky
    def _sky(self, world, dirs: np.ndarray, sun: np.ndarray, keys, style: ToonStyle) -> np.ndarray:
        dy = dirs[:, 1]
        t = mth.smoothstep(-0.02, 0.46, dy) ** 0.72
        zen = np.asarray(keys.zenith, dtype=np.float32)
        hor = np.asarray(keys.horizon, dtype=np.float32)
        col = hor[None, :] + (zen[None, :] - hor[None, :]) * t[:, None]

        cs = np.clip(np.sum(dirs * sun[None, :], axis=1), 0.0, 1.0)
        sun_col = np.asarray(keys.sun, dtype=np.float32)
        disk = mth.smoothstep(0.9975, 0.9992, cs)
        glow = cs ** 48.0 * 0.5 + cs ** 8.0 * 0.14
        col = col + sun_col[None, :] * (disk * 1.5 + glow * keys.sun_strength)[:, None]

        if self.settings.clouds:
            up = dy > 0.04
            if up.any():
                d = dirs[up]
                # Intersect a cloud deck 1100 m up; clamp so the horizon does
                # not alias into speckle.
                tt = np.minimum(1100.0 / np.maximum(d[:, 1], 0.04), 9000.0)
                cx = (d[:, 0] * tt) / 1700.0 + world.elapsed * 0.0015
                cz = (d[:, 2] * tt) / 1700.0
                n = nz.fbm(cx, cz, world.cfg.seed + 909, octaves=4) * 1.8 + 0.55
                cover = 1.0 - style.cloud_cover
                alpha = np.clip((n - cover) * style.cloud_sharpness * 3.0, 0.0, 1.0)
                alpha *= mth.smoothstep(0.04, 0.30, d[:, 1])
                lit = np.clip((n - cover) * 5.0, 0.0, 1.0)[:, None]
                shade_lo = np.asarray(keys.horizon, dtype=np.float32) * 0.78 + 0.08
                shade_hi = np.asarray(keys.sun, dtype=np.float32) * 0.92 + 0.08
                cloud = shade_lo[None, :] + (shade_hi[None, :] - shade_lo[None, :]) * lit
                col[up] = col[up] * (1.0 - alpha[:, None]) + cloud * alpha[:, None]
        return col.astype(np.float32)

    # ----------------------------------------------------------------- water
    def _water(self, world, eye: np.ndarray, dirs: np.ndarray, t_terrain: np.ndarray,
               color: np.ndarray, sun: np.ndarray, keys, style: ToonStyle):
        """Stylized water: banded depth tint, foam at the shore, sun glint."""
        sea = world.terrain.cfg.sea_level
        dy = dirs[:, 1]
        going_down = dy < -1e-4
        above = eye[1] > sea + 0.02
        if not above:
            return color, t_terrain, None
        tw = np.where(going_down, (sea - eye[1]) / np.where(going_down, dy, -1.0), np.inf)
        visible = going_down & (tw > 0.0) & (tw < np.minimum(t_terrain, self.settings.max_view_distance))
        if not visible.any():
            return color, t_terrain, None

        idx = np.nonzero(visible)[0]
        twv = tw[idx]
        p = eye[None, :] + dirs[idx] * twv[:, None]
        bed = world.terrain.height_at(p[:, 0], p[:, 2])
        depth = np.clip(sea - bed, 0.0, 14.0)

        # Ripples: two drifting noise fields perturb the surface normal.
        ph = world.elapsed * 0.45
        n1 = nz.fbm(p[:, 0] * 0.18 + ph, p[:, 2] * 0.18, world.cfg.seed + 1301, octaves=2)
        n2 = nz.fbm(p[:, 0] * 0.46 - ph * 0.7, p[:, 2] * 0.46, world.cfg.seed + 1409, octaves=2)
        nrm = np.stack([n1 * 0.14, np.ones_like(n1), n2 * 0.14], axis=1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-8)

        shallow = np.asarray(style.water_shallow, dtype=np.float32)
        deep = np.asarray(style.water_deep, dtype=np.float32)
        # Quantized depth bands read as painted water rather than a smooth fade.
        band = np.floor(np.clip(depth / 6.0, 0.0, 1.0) * 3.0 + 0.5) / 3.0
        base = shallow[None, :] + (deep[None, :] - shallow[None, :]) * band[:, None]
        base = base * (0.62 + 0.38 * keys.sun_strength)

        view = -dirs[idx]
        fres = np.clip(1.0 - np.sum(nrm * view, axis=1), 0.0, 1.0) ** 3.0
        refl = (0.10 + 0.40 * fres)[:, None]
        mirrored = mth.norm(dirs[idx] * np.array([1.0, -1.0, 1.0], dtype=np.float32))
        sky_col = self._sky(world, mirrored, sun, keys, style)
        col = base * (1.0 - refl) + sky_col * 0.9 * refl

        half = mth.norm(view + sun[None, :])
        spec = np.clip(np.sum(nrm * half, axis=1), 0.0, 1.0) ** 180.0
        col = col + np.asarray(keys.sun, dtype=np.float32)[None, :] * (spec * 0.6 * keys.sun_strength)[:, None]

        # Foam only where the bed comes close to the surface, plus a moving
        # shoreline band, which is what sells stylized water.
        shore = mth.smoothstep(0.9, 0.08, depth)
        ripple = 0.5 + 0.5 * np.sin(depth * 6.0 - world.elapsed * 2.2 + n2 * 4.0)
        foam = np.clip(shore * (0.45 + 0.55 * ripple), 0.0, 1.0)
        col = col * (1.0 - foam[:, None]) + np.asarray(style.water_foam, dtype=np.float32)[None, :] * foam[:, None]

        color[idx] = col
        t_out = t_terrain.copy()
        t_out[idx] = twv
        return color, t_out, (idx, nrm)

    # ------------------------------------------------------------------ fog
    def _apply_fog(self, color: np.ndarray, depth: np.ndarray, keys, style: ToonStyle,
                   height: np.ndarray | None = None) -> np.ndarray:
        f = mth.smoothstep(style.fog_start, style.fog_end, depth) * getattr(style, "fog_max", 1.0)
        if height is not None and style.fog_height_falloff > 0.0:
            f = f * np.exp(-np.maximum(height, 0.0) * style.fog_height_falloff)
        fog_col = np.asarray(keys.fog, dtype=np.float32)
        return color * (1.0 - f[:, None]) + fog_col[None, :] * f[:, None]

    # --------------------------------------------------------------- meshes
    def _build_meshes(self, world, char: CharacterState, eye: np.ndarray, fwd: np.ndarray) -> MeshBatch:
        s = self.settings
        batch = MeshBatch()
        if s.draw_props and s.ground_cover:
            self._ground_cover(batch, world, char.pos)
        if s.draw_props and len(world.props) > 0:
            props = world.props
            vis = props.visible_subset(eye, fwd, s.prop_draw_distance, fov_cos=-0.25)
            if len(vis):
                # Third-person foliage cutaway: do not put a trunk between the
                # camera and the avatar. Collision geometry remains unchanged.
                horizontal = fwd[[0, 2]] / max(np.linalg.norm(fwd[[0, 2]]), 1e-6)
                delta = props.pos[vis][:, [0, 2]] - eye[[0, 2]]
                along = delta @ horizontal
                across = np.abs(delta[:, 0] * horizontal[1] - delta[:, 1] * horizontal[0])
                boom = np.linalg.norm((char.pos - eye)[[0, 2]])
                occluding = (along > -1.0) & (along < boom - 0.3) & (across < props.radius[vis] + 0.65)
                vis = vis[~occluding]
                grass = vis[(props.kind[vis] == GRASS) & (np.linalg.norm(props.pos[vis] - char.pos, axis=1) < 18.0)][:48]
                for i in grass:
                    add_prop(batch, "grass", props.pos[i], float(props.yaw[i]), float(props.scale[i]), obj_id=10)
            if len(vis):
                vis = vis[np.isin(props.kind[vis], _MESH_PROPS)]
            if len(vis):
                d = np.linalg.norm(props.pos[vis][:, [0, 2]] - eye[[0, 2]][None, :], axis=1)
                order = np.argsort(d)
                vis, d = vis[order], d[order]
                near = vis[d <= 45.0][:52]
                far = vis[(d > 45.0)][:160]
                for i in near:
                    add_prop(batch, PROP_NAMES[int(props.kind[i])], props.pos[i],
                             float(props.yaw[i]), float(props.scale[i]), obj_id=10)
                for i in far:
                    _add_prop_lod(batch, PROP_NAMES[int(props.kind[i])], props.pos[i],
                                  float(props.scale[i]), obj_id=10)
        if s.draw_entities:
            from ..world import ENTITY_NAMES
            from ..elements import Element, ELEMENT_COLORS
            for n, e in enumerate(world.entities.values()):
                if not e.alive:
                    continue
                dxz = float(np.linalg.norm((e.pos - eye)[[0, 2]]))
                if dxz > s.prop_draw_distance * 1.6:
                    continue
                add_entity(batch, ENTITY_NAMES[int(e.kind)], e.pos, e.yaw, e.used,
                           obj_id=100 + n, bob=e.bob)
                aura = e.elemental.aura.element
                if e.elemental.frozen > 0.0:
                    aura = Element.FROST
                elif e.elemental.stunned > 0.0:
                    aura = Element.SPARK
                if aura != Element.NONE:
                    p = e.pos + np.array([0.0, 3.3 if int(e.kind) == 6 else 2.5, 0.0])
                    batch.cone(p, 0.25, 0.5, ELEMENT_COLORS[aura], sides=4,
                               obj_id=100 + n, emissive=0.8)
                if e.elemental.shield > 0.0:
                    self._element_ring(batch, e.pos, 1.2, e.elemental.shield_element, 100 + n)
        if s.draw_character:
            spec = world.party.active.spec if world.party else None
            add_character(batch, char, obj_id=2, spec=spec)
            if world.party and world.party.shield > 0.0:
                self._element_ring(batch, char.pos, 0.9, world.party.shield_element, 2)
        for effect in world.combat.effects:
            self._element_ring(batch, effect.pos, effect.radius * (1.0 - effect.remaining),
                               effect.element, 9000)
        return batch

    @staticmethod
    def _ground_cover(batch, world, center):
        """World-anchored decorative patches; never consume simulation RNG."""
        from .style import PROP_ALBEDO

        cx, cz = np.floor(center[[0, 2]] / 2.5).astype(int)
        ix, iz = np.meshgrid(np.arange(cx - 6, cx + 7), np.arange(cz - 6, cz + 7))
        phase = ix.ravel() * 127.1 + iz.ravel() * 311.7 + world.cfg.seed * 0.13
        noise = np.mod(np.sin(phase) * 43758.5453, 1.0)
        x = (ix.ravel() + noise) * 2.5
        z = (iz.ravel() + np.mod(noise * 7.3, 1.0)) * 2.5
        y = world.terrain.height_at(x, z)
        good = ((noise > 0.45) & (y > world.terrain.cfg.sea_level + 2)
                & (y < world.terrain.cfg.tree_line) & (world.terrain.slope_at(x, z) < 0.50))
        for i in np.flatnonzero(good):
            p = np.array([x[i], y[i] + 0.025, z[i]], np.float32)
            add_prop(batch, "grass", p, float(phase[i]), 0.7 + noise[i] * 0.4, obj_id=10)
            if noise[i] > 0.80:
                color = (0.98, 0.90, 0.57) if noise[i] > 0.93 else (0.86, 0.84, 0.98)
                for dx, dz in ((0, 0), (0.2, 0.1), (-0.1, 0.2)):
                    base = p + np.array([dx, 0, dz])
                    batch.quad(base + (-0.012, 0, 0), base + (0.012, 0, 0),
                               base + (0.012, 0.32, 0), base + (-0.012, 0.32, 0),
                               PROP_ALBEDO["grass"], 10)
                    batch.cone(base + (0, 0.32, 0), 0.09, 0.045, color, 5, 10, emissive=0.1)

    @staticmethod
    def _element_ring(batch, pos, radius, element, obj_id):
        from ..elements import ELEMENT_COLORS

        color = ELEMENT_COLORS[element]
        for a in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
            b = a + 2.0 * np.pi / 16
            p = np.asarray(pos) + np.array([0.0, 0.15, 0.0])
            d0 = np.array([np.cos(a), 0.0, np.sin(a)])
            d1 = np.array([np.cos(b), 0.0, np.sin(b)])
            batch.quad(p + d0 * radius, p + d1 * radius,
                       p + d1 * (radius + 0.12), p + d0 * (radius + 0.12),
                       color, obj_id, emissive=0.8)

    def _draw_meshes(self, batch: MeshBatch, view, proj, eye, sun, keys, style, buffers) -> None:
        verts, colors, emissive, ids = batch.finish()
        if len(verts) == 0:
            return
        color_buf, depth_buf, normal_buf, id_buf = buffers
        h, w = depth_buf.shape
        screen, wdepth, keep = raster.project(verts, view, proj, w, h)
        if not keep.any():
            return
        nrm = raster.face_normals(verts, eye)
        vertex_normals = np.concatenate(batch.normals)
        smooth = np.linalg.norm(vertex_normals, axis=-1).max(axis=1) > 0.5
        vertex_normals[~smooth] = nrm[~smooth, None, :]
        ndl = np.sum(vertex_normals * sun[None, None, :], axis=-1)
        ramp = toon_ramp(ndl, style)
        sun_col = np.asarray(keys.sun, dtype=np.float32) * keys.sun_strength
        amb = np.asarray(keys.zenith, dtype=np.float32) * (keys.ambient + 0.3)
        shaded = colors[:, None, :] * (sun_col[None, None, :] * ramp[..., None] * 0.68
                                       + amb[None, None, :] * 0.60 + 0.22)
        # Emissive surfaces ignore the light and glow on their own.
        shaded = shaded * (1.0 - emissive[:, None, None]) + colors[:, None, :] * (1.25 * emissive[:, None, None])
        centroid_depth = wdepth.mean(axis=1)
        fog = (mth.smoothstep(style.fog_start, style.fog_end, centroid_depth)
               * getattr(style, "fog_max", 1.0))[:, None]
        shaded = shaded * (1.0 - fog[:, :, None]) + np.asarray(keys.fog, dtype=np.float32)[None, None, :] * fog[:, :, None]

        big, tiny = raster.split_by_screen_size(screen, keep, threshold=2.5)
        raster.draw_triangles(screen, wdepth, big, shaded, nrm, ids,
                              color_buf, depth_buf, normal_buf, id_buf)
        raster.draw_point_sprites(screen, wdepth, tiny, shaded, nrm, ids,
                                  color_buf, depth_buf, normal_buf, id_buf)

    # ---------------------------------------------------------------- outline
    def _outline(self, frame: Frame, style: ToonStyle) -> np.ndarray:
        """Ink pass: silhouettes, creases and object boundaries.

        Uses the *second* difference of log depth rather than the gradient. A
        flat surface seen at a grazing angle has an enormous depth gradient but
        an almost zero Laplacian, so the sea stops being drawn as solid ink.
        """
        d = np.where(np.isfinite(frame.depth), frame.depth, 1e4)
        ld = np.log1p(d)
        lap = np.zeros(ld.shape, dtype=np.float32)
        lap[:, 1:-1] = np.abs(2.0 * ld[:, 1:-1] - ld[:, 2:] - ld[:, :-2])
        lap[1:-1, :] = np.maximum(lap[1:-1, :], np.abs(2.0 * ld[1:-1, :] - ld[2:, :] - ld[:-2, :]))
        depth_edge = mth.smoothstep(style.outline_depth_threshold,
                                    style.outline_depth_threshold * 4.0, lap)

        n = frame.normal
        nx = np.zeros(ld.shape, dtype=np.float32)
        ny = np.zeros(ld.shape, dtype=np.float32)
        nx[:, 1:-1] = 1.0 - np.sum(n[:, 2:] * n[:, :-2], axis=-1)
        ny[1:-1, :] = 1.0 - np.sum(n[2:, :] * n[:-2, :], axis=-1)
        normal_edge = mth.smoothstep(style.outline_normal_threshold,
                                     style.outline_normal_threshold + 0.35, np.maximum(nx, ny))
        edge = np.maximum(depth_edge, normal_edge * 0.2)

        if frame.ids is not None:
            ids = frame.ids
            id_edge = np.zeros(ld.shape, dtype=np.float32)
            id_edge[:, 1:-1] = (ids[:, 2:] != ids[:, :-2]).astype(np.float32)
            id_edge[1:-1, :] = np.maximum(id_edge[1:-1, :], (ids[2:, :] != ids[:-2, :]).astype(np.float32))
            edge = np.maximum(edge, id_edge)
            # Paint terrain with tonal changes, reserving fine ink for meshes.
            edge *= np.where(ids == 0, 0.06, 1.0)

        # Lines fade with distance so the horizon does not turn into mush.
        edge = edge * (1.0 - mth.smoothstep(style.fog_start * 1.6, style.fog_end, d))
        ink = edge[..., None] * style.outline_strength
        return frame.color * (1.0 - ink) + (frame.color * 0.48) * ink

    # ---------------------------------------------------------------- render
    def render_frame(self, world, char: CharacterState, cam, fov_y: float = 58.0) -> Frame:
        s = self.settings
        style = s.style
        ss = max(1, int(s.supersample))
        W, H = s.width * ss, s.height * ss
        eye = np.asarray(cam.eye, dtype=np.float32)
        target = np.asarray(cam.pivot, dtype=np.float32)
        view = mth.look_at(eye, target)
        proj = mth.perspective(fov_y, W / H, 0.06, s.max_view_distance * 1.6)
        dirs_img = self._ray_dirs(view, W, H, fov_y)
        dirs = dirs_img.reshape(-1, 3)

        sun = world.sun_direction
        keys = sky_keys_at(world.time_of_day)

        t_hit, got = self._march(world.terrain, eye, dirs)
        color = self._sky(world, dirs, sun, keys, style)
        normal = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (len(dirs), 1))
        depth = np.where(got, t_hit, np.inf).astype(np.float32)

        if got.any():
            idx = np.nonzero(got)[0]
            pos = eye[None, :] + dirs[idx] * t_hit[idx][:, None]
            lit, nrm = self._shade_terrain(world, pos, dirs[idx], eye, sun, keys, style)
            # Soft grounding beneath the avatar; does not modify physics.
            horizontal = pos[:, [0, 2]] - char.pos[[0, 2]]
            altitude = max(0.0, float(char.pos[1] - world.terrain.height_at(char.pos[0], char.pos[2])))
            contact = np.exp(-np.sum(horizontal ** 2, axis=1) / (0.38 + altitude * 0.3))
            lit *= (1.0 - contact * 0.24 * np.exp(-altitude * 0.3))[:, None]
            lit = self._apply_fog(lit, t_hit[idx], keys, style)
            color[idx] = lit
            normal[idx] = nrm

        if s.water:
            color, depth_w, water = self._water(world, eye, dirs, depth, color, sun, keys, style)
            if water is not None:
                widx, wnrm = water
                # Note: the ripple normal is deliberately *not* stored -- the
                # outline pass would turn every wavelet into an ink line.
                normal[widx] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                color[widx] = self._apply_fog(color[widx], depth_w[widx], keys, style)
                depth = depth_w

        # Rasterized mesh depths are camera-axis distances, whereas the terrain
        # march produces ray distances. Compare them in the same depth space.
        forward_depth = dirs @ (-view[2, :3])
        depth *= forward_depth
        frame = Frame(
            color=color.reshape(H, W, 3),
            depth=depth.reshape(H, W),
            normal=normal.reshape(H, W, 3),
            ids=np.zeros((H, W), dtype=np.int32),
        )
        fwd = mth.norm(target - eye)
        batch = self._build_meshes(world, char, eye, fwd)
        self._draw_meshes(batch, view, proj, eye, sun, keys, style,
                          (frame.color, frame.depth, frame.normal, frame.ids))
        frame.depth /= forward_depth.reshape(H, W)

        if s.outlines:
            frame.color = self._outline(frame, style)

        # Underwater: everything behind a thick blue-green veil.
        if eye[1] < world.terrain.cfg.sea_level:
            tint = np.array([0.10, 0.38, 0.46], dtype=np.float32)
            murk = np.clip(mth.smoothstep(0.0, 42.0, np.where(np.isfinite(frame.depth), frame.depth, 42.0)), 0.0, 1.0)
            frame.color = frame.color * (1.0 - 0.75 * murk[..., None]) + tint[None, None, :] * (0.75 * murk[..., None])

        frame.color = grade(frame.color, style)
        if style.vignette > 0.0:
            yy = np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None]
            xx = np.linspace(-1.0, 1.0, W, dtype=np.float32)[None, :]
            r = np.sqrt(xx * xx + yy * yy) / 1.4142
            frame.color *= (1.0 - style.vignette * r[..., None] ** 2.4)

        if ss > 1:
            c = frame.color.reshape(s.height, ss, s.width, ss, 3).mean(axis=(1, 3))
            d = frame.depth.reshape(s.height, ss, s.width, ss).min(axis=(1, 3))
            n = frame.normal.reshape(s.height, ss, s.width, ss, 3).mean(axis=(1, 3))
            i = frame.ids.reshape(s.height, ss, s.width, ss)[:, 0, :, 0]
            frame = Frame(color=c.astype(np.float32), depth=d.astype(np.float32),
                          normal=n.astype(np.float32), ids=i)
        return frame

    def render(self, world, char: CharacterState, cam, fov_y: float = 58.0) -> np.ndarray:
        return self.render_frame(world, char, cam, fov_y).to_uint8()


def _add_prop_lod(batch: MeshBatch, kind_name: str, pos: np.ndarray, scale: float, obj_id: int) -> None:
    """A 5-6 triangle stand-in used beyond the near LOD radius."""
    from .style import PROP_ALBEDO
    s = float(scale)
    if kind_name in ("broadleaf", "sakura"):
        batch.cone(pos, 0.22 * s, 3.4 * s, PROP_ALBEDO["trunk"], 4, obj_id, taper=0.5)
        batch.ellipsoid(pos + np.array([0.0, 4.4 * s, 0.0]), (2.5 * s, 1.9 * s, 2.2 * s),
                        PROP_ALBEDO[kind_name], obj_id=obj_id, sides=6, rings=3)
    elif kind_name in ("pine", "bamboo"):
        h = {"pine": 9.0, "broadleaf": 7.5, "sakura": 6.5, "bamboo": 5.5}[kind_name]
        col = PROP_ALBEDO[kind_name]
        batch.cone(pos, 1.5 * s, h * s, col, 4, obj_id, taper=0.15 if kind_name == "bamboo" else 0.0)
    elif kind_name in ("rock", "boulder"):
        r = (0.55 if kind_name == "rock" else 1.5) * s
        batch.cone(pos, r, r * 1.4, PROP_ALBEDO[kind_name], 4, obj_id, taper=0.4)
    elif kind_name == "crystal":
        batch.cone(pos, 0.45 * s, 2.2 * s, PROP_ALBEDO["crystal"], 4, obj_id, emissive=0.45)
    elif kind_name == "bush":
        batch.ellipsoid(pos + np.array([0.0, 0.4 * s, 0.0]), (0.8 * s, 0.6 * s, 0.7 * s),
                        PROP_ALBEDO["bush"], obj_id=obj_id, sides=5, rings=3)
