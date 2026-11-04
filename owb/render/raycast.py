"""Vectorised voxel raycasting renderer.

One NumPy DDA march over all pixels at once, with active-ray compaction so the
cost tracks the number of rays still in flight rather than ``max_steps``.  No
GPU, no display, no native extension -- which is the point: it renders the same
on a laptop and on a headless cluster node, which is where RL rollouts live.

Features: alpha-accumulated foliage, volumetric water and lava, directional
sun/moon with optional hard shadows, point lights (torches, campfires, glowing
crystals), height fog, procedural sky with sun disk / clouds / stars, weather
overlays, billboard impostors for creatures and items, and aligned depth +
semantic-segmentation buffers.
"""

from __future__ import annotations

import numpy as np

from ..core import blocks as B
from ..core.config import RenderConfig
from . import palette as P
from .camera import Camera, ray_dirs

_INF = np.float32(1e9)


class Renderer:
    """Renders :class:`~owb.world.voxel.VoxelWorld` views for an agent."""

    def __init__(self, cfg: RenderConfig | None = None):
        self.cfg = cfg or RenderConfig()

    # ------------------------------------------------------------------
    def render(self, world, pos, yaw: float, pitch: float, *,
               time_of_day: float = 0.5, weather: str = "clear",
               lights: list | None = None, entities: list | None = None,
               flash: float = 0.0, rain_intensity: float = 0.0,
               frame: int = 0) -> dict:
        """Render one view.

        Returns a dict with ``rgb`` ``(H, W, 3) uint8``, ``depth``
        ``(H, W) float32`` in metres and ``semantic`` ``(H, W) uint8``.
        """
        cfg = self.cfg
        ss = max(1, int(cfg.supersample))
        W, H = cfg.width * ss, cfg.height * ss
        cam = Camera(np.asarray(pos, dtype=np.float32), float(yaw), float(pitch),
                     cfg.fov)
        dirs = ray_dirs(cam, W, H)
        o = cam.pos.astype(np.float32)

        g = self._march(world, o, dirs, cfg.max_dist, cfg.max_steps)
        sky = P.sky_colors(time_of_day, weather)
        rgb = self._shade(world, o, dirs, g, sky, weather, lights, time_of_day,
                          frame)
        depth = np.where(g["water_t"] >= 0,
                         np.minimum(g["water_t"], g["t"]), g["t"])
        sem = B.SEMANTIC[g["blk"]].astype(np.uint8)
        sem[g["blk"] == B.AIR] = B.SEM_SKY
        watery = (g["water_t"] >= 0) & (g["water_len"] > 0.6)
        sem[watery] = B.SEM_WATER

        rgb = rgb.reshape(H, W, 3)
        depth = depth.reshape(H, W)
        sem = sem.reshape(H, W)

        if cfg.entity_impostors and entities:
            self._draw_entities(cam, entities, rgb, depth, sem, W, H,
                                time_of_day)
        if weather in ("rain", "storm") or rain_intensity > 0:
            self._weather_overlay(rgb, weather, rain_intensity, frame)
        elif weather == "snow":
            self._weather_overlay(rgb, "snow", 1.0, frame)
        if flash > 0:
            rgb = np.clip(rgb.astype(np.float32) + 255.0 * float(flash), 0,
                          255).astype(np.uint8)

        if ss > 1:
            rgb = rgb.reshape(cfg.height, ss, cfg.width, ss, 3).mean(
                axis=(1, 3)).astype(np.uint8)
            depth = depth.reshape(cfg.height, ss, cfg.width, ss).mean(axis=(1, 3))
            sem = sem.reshape(cfg.height, ss, cfg.width, ss)[:, 0, :, 0]
        return dict(rgb=np.ascontiguousarray(rgb),
                    depth=np.ascontiguousarray(depth.astype(np.float32)),
                    semantic=np.ascontiguousarray(sem.astype(np.uint8)))

    # ------------------------------------------------------------------
    # the march: exact voxel DDA near, heightfield march far
    # ------------------------------------------------------------------
    def _march(self, world, o: np.ndarray, dirs: np.ndarray, max_dist: float,
               max_steps: int, *, exact=False) -> dict:
        n = dirs.shape[0]
        g = dict(
            blk=np.zeros(n, dtype=np.uint8),
            t=np.full(n, np.float32(max_dist), dtype=np.float32),
            nrm=np.zeros((n, 3), dtype=np.float32),
            vox=np.zeros((n, 3), dtype=np.int32),
            acc_rgb=np.zeros((n, 3), dtype=np.float32),
            acc_a=np.zeros(n, dtype=np.float32),
            water_len=np.zeros(n, dtype=np.float32),
            water_t=np.full(n, np.float32(-1.0), dtype=np.float32),
            lava_len=np.zeros(n, dtype=np.float32),
        )
        # Heightfields fill the space underneath floating geometry. Layered
        # worlds and the landscape's sparse architecture need full voxel DDA.
        exact = exact or world.meta.get("layered", False)
        near = float(max_dist) if exact else min(float(self.cfg.exact_dist), float(max_dist))
        if exact:
            max_steps = max(max_steps, int(np.ceil(max_dist * np.sqrt(3))) + 3)
        alive, t_alive = self._march_near(world, o, dirs, near, max_steps, g)
        if alive.size and max_dist > near + 0.5:
            self._march_far(world, o, dirs, alive, t_alive, float(max_dist), g)
        return g

    def _march_near(self, world, o, dirs, max_dist: float, max_steps: int, g):
        """Exact voxel DDA. Returns the rays that survived to ``max_dist``."""
        n = dirs.shape[0]
        vox_flat = world.vox.reshape(-1)
        SX, SY, SZ = world.size_x, world.size_y, world.size_z
        blk_out, t_out, nrm_out, vox_out = g["blk"], g["t"], g["nrm"], g["vox"]
        acc_rgb, acc_a = g["acc_rgb"], g["acc_a"]
        water_len, water_t, lava_len = g["water_len"], g["water_t"], g["lava_len"]

        idx = np.arange(n, dtype=np.int64)
        d = dirs
        v = np.floor(o[None, :]).astype(np.int32) + np.zeros((n, 3), dtype=np.int32)
        step = np.sign(d).astype(np.int32)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(np.abs(d) > 1e-9, 1.0 / d, _INF).astype(np.float32)
        t_delta = np.abs(inv)
        t_max = ((v + (step > 0).astype(np.int32)).astype(np.float32)
                 - o[None, :]) * inv
        t_max = t_max.astype(np.float32)
        t_max[np.abs(d) <= 1e-9] = _INF
        t_cur = np.zeros(n, dtype=np.float32)
        last_axis = np.full(n, -1, dtype=np.int64)

        alpha_tab, col_tab = P.ALPHA, P.COLOR_F
        md = np.float32(max_dist)
        surv: list[np.ndarray] = []      # rays handed over to the far pass

        for _ in range(int(max_steps)):
            m = idx.size
            if m == 0:
                break
            rows = np.arange(m, dtype=np.int64)
            vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
            oob = ((vx < 0) | (vx >= SX) | (vz < 0) | (vz >= SZ) | (vy >= SY)
                   | (vy < 0))
            lin = (vx.astype(np.int64) * SY + vy) * SZ + vz
            np.clip(lin, 0, vox_flat.size - 1, out=lin)
            blk = vox_flat[lin]
            if oob.any():
                blk = np.where(oob, np.uint8(B.AIR), blk)

            axis = np.argmin(t_max, axis=1)
            t_next = t_max[rows, axis]
            seg = np.minimum(t_next, md) - t_cur
            np.maximum(seg, 0.0, out=seg)

            alpha = alpha_tab[blk]
            is_water = blk == B.WATER
            if is_water.any():
                gi = idx[is_water]
                water_len[gi] += seg[is_water]
                fresh = water_t[gi] < 0
                if fresh.any():
                    water_t[gi[fresh]] = t_cur[is_water][fresh]
            is_lava = blk == B.LAVA
            if is_lava.any():
                lava_len[idx[is_lava]] += seg[is_lava]

            fol = (alpha > 0.0) & (alpha < 0.999)
            if fol.any():
                fi = np.nonzero(fol)[0]
                gi = idx[fi]
                contrib = (1.0 - acc_a[gi]) * alpha[fi]
                acc_rgb[gi] += contrib[:, None] * col_tab[blk[fi]]
                acc_a[gi] += contrib

            opq = alpha >= 0.999
            if opq.any():
                oi = np.nonzero(opq)[0]
                gi = idx[oi]
                blk_out[gi] = blk[oi]
                t_out[gi] = t_cur[oi]
                vox_out[gi] = v[oi]
                la = last_axis[oi]
                nn = np.zeros((oi.size, 3), dtype=np.float32)
                first = la < 0
                if first.any():
                    la = la.copy()
                    la[first] = np.argmax(np.abs(d[oi][first]), axis=1)
                r2 = np.arange(oi.size, dtype=np.int64)
                nn[r2, la] = -step[oi][r2, la].astype(np.float32)
                nrm_out[gi] = nn

            stop = opq | (acc_a[idx] > 0.97) | oob
            reached = (t_next >= md) & ~stop
            if reached.any():
                surv.append(idx[reached])
            done = stop | reached
            keep = ~done
            if not keep.any():
                idx = idx[:0]
                break
            kk = np.nonzero(keep)[0]
            ax = axis[kk]
            r2 = np.arange(kk.size, dtype=np.int64)
            v = v[kk]
            t_max = t_max[kk]
            t_delta = t_delta[kk]
            step = step[kk]
            v[r2, ax] += step[r2, ax]
            t_max[r2, ax] += t_delta[r2, ax]
            t_cur = t_next[kk]
            last_axis = ax
            d = d[kk]
            idx = idx[kk]

        # Rays that crossed the exact/far boundary without being stopped are
        # handed to the heightfield pass, which resumes them at that boundary.
        if idx.size:
            surv.append(idx)
        if not surv:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        alive = np.concatenate(surv)
        return alive, np.full(alive.size, md, dtype=np.float32)

    def _march_far(self, world, o, dirs, idx, t_start, max_dist: float, g):
        """Heightfield ray march with max-height tile skipping.

        Past the exact range the world is treated as a heightfield: one gather
        and compare per step instead of a full DDA, and any ray flying above a
        tile's maximum column height jumps straight across that tile.  Caves
        and overhangs are not represented out here, which is invisible at range
        and is what makes a 112 m view distance affordable.
        """
        cfg = self.cfg
        tile = 8
        shift = 3
        H = world.height
        WH = world.water_height
        mip = world.height_mip(tile)
        vox = world.vox
        SX, SZ, SY = world.size_x, world.size_z, world.size_y

        d = dirs[idx]
        t = t_start.astype(np.float32) + 0.001
        p = o[None, :] + d * t[:, None]
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        adx = np.maximum(np.abs(dx), 1e-4)
        adz = np.maximum(np.abs(dz), 1e-4)
        md = np.float32(max_dist)

        for _ in range(int(cfg.far_steps)):
            if idx.size == 0:
                break
            xi = p[:, 0].astype(np.int32)
            zi = p[:, 2].astype(np.int32)
            py = p[:, 1]
            oob = (xi < 0) | (xi >= SX) | (zi < 0) | (zi >= SZ) | (py >= SY) \
                | (py < 0) | (t >= md)
            live = ~oob
            if not live.any():
                break
            xi = np.clip(xi, 0, SX - 1)
            zi = np.clip(zi, 0, SZ - 1)
            h = H[xi, zi].astype(np.float32)
            wh = WH[xi, zi].astype(np.float32)

            hit = live & (py <= h + 1.0)
            wet = live & ~hit & (py <= wh + 1.0)
            if hit.any() or wet.any():
                fin = hit | wet
                fi = np.nonzero(fin)[0]
                gi = idx[fi]
                g["t"][gi] = t[fi]
                xb = xi[fi]
                zb = zi[fi]
                yb = np.clip(np.where(hit[fi], h[fi], wh[fi]), 0,
                             SY - 1).astype(np.int32)
                g["vox"][gi] = np.stack([xb, yb, zb], axis=1)
                blk = vox[xb, yb, zb]
                blk = np.where(wet[fi], np.uint8(B.WATER), blk)
                blk = np.where(blk == B.AIR, np.uint8(B.STONE), blk)
                g["blk"][gi] = blk
                # smooth normal from the heightmap gradient
                xl = np.clip(xb - 1, 0, SX - 1)
                xr = np.clip(xb + 1, 0, SX - 1)
                zl = np.clip(zb - 1, 0, SZ - 1)
                zr = np.clip(zb + 1, 0, SZ - 1)
                nx = (H[xl, zb] - H[xr, zb]).astype(np.float32)
                nz = (H[xb, zl] - H[xb, zr]).astype(np.float32)
                nn = np.stack([nx, np.full(fi.size, 2.4, dtype=np.float32), nz],
                              axis=1)
                nn[wet[fi]] = (0.0, 1.0, 0.0)
                nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-5)
                g["nrm"][gi] = nn

            cont = live & ~hit & ~wet
            if not cont.any():
                break
            ci = np.nonzero(cont)[0]

            # how far can this ray safely jump?
            pxc, pzc, pyc = p[ci, 0], p[ci, 2], p[ci, 1]
            tm = mip[xi[ci] >> shift, zi[ci] >> shift].astype(np.float32)
            above = pyc - (tm + 1.0)
            fx = pxc - np.floor(pxc / tile) * tile
            fz = pzc - np.floor(pzc / tile) * tile
            ex = np.where(dx[ci] > 0, tile - fx, fx) / adx[ci]
            ez = np.where(dz[ci] > 0, tile - fz, fz) / adz[ci]
            tile_exit = np.minimum(ex, ez) + 0.02
            dyc = dy[ci]
            drop = np.where(dyc < -1e-4, above / np.maximum(-dyc, 1e-4), _INF)
            jump = np.minimum(tile_exit, drop)
            stepk = np.where(above > 0.5, np.clip(jump, 0.6, 24.0),
                             np.float32(1.0)).astype(np.float32)

            p = p[ci] + d[ci] * stepk[:, None]
            t = t[ci] + stepk
            d = d[ci]
            dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
            adx, adz = adx[ci], adz[ci]
            idx = idx[ci]

    # ------------------------------------------------------------------
    # occlusion march (shadows)
    # ------------------------------------------------------------------
    @staticmethod
    def _occluded(world, origins: np.ndarray, sun: np.ndarray, max_steps: int,
                  max_dist: float = 40.0) -> np.ndarray:
        n = origins.shape[0]
        out = np.zeros(n, dtype=bool)
        if n == 0:
            return out
        vox = world.vox
        SX, SY, SZ = world.size_x, world.size_y, world.size_z
        d = np.broadcast_to(sun.astype(np.float32), (n, 3)).copy()
        v = np.floor(origins).astype(np.int32)
        step = np.sign(d).astype(np.int32)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(np.abs(d) > 1e-9, 1.0 / d, _INF)
            t_delta = np.abs(inv).astype(np.float32)
            t_max = ((v + (step > 0).astype(np.int32)).astype(np.float32)
                     - origins) * inv
        t_max = t_max.astype(np.float32)
        t_max[np.abs(d) <= 1e-9] = _INF
        idx = np.arange(n, dtype=np.int64)
        opaque = B.SOLID.copy()
        opaque[B.LEAVES] = True
        for _ in range(int(max_steps)):
            m = idx.size
            if m == 0:
                break
            vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
            oob = (vx < 0) | (vx >= SX) | (vz < 0) | (vz >= SZ) | (vy >= SY) | (vy < 0)
            blk = np.zeros(m, dtype=np.uint8)
            inb = ~oob
            if inb.any():
                blk[inb] = vox[vx[inb], vy[inb], vz[inb]]
            hit = opaque[blk] & inb
            if hit.any():
                out[idx[hit]] = True
            ar = np.arange(m, dtype=np.int64)
            axis = np.argmin(t_max, axis=1)
            t_next = t_max[ar, axis]
            keep = ~(hit | oob | (t_next > np.float32(max_dist)))
            if not keep.any():
                break
            kk = np.nonzero(keep)[0]
            v = v[kk]
            t_max = t_max[kk]
            t_delta = t_delta[kk]
            step = step[kk]
            ax = axis[kk]
            rows = np.arange(kk.size, dtype=np.int64)
            v[rows, ax] += step[rows, ax]
            t_max[rows, ax] += t_delta[rows, ax]
            idx = idx[kk]
        return out

    # ------------------------------------------------------------------
    # shading
    # ------------------------------------------------------------------
    def _shade(self, world, o, dirs, g, sky, weather, lights, tod, frame):
        cfg = self.cfg
        n = dirs.shape[0]
        sun = sky["sun"]
        sdir = sun["dir"]
        scol = sun["color"] * sun["intensity"]
        amb = sun["ambient"]
        blk = g["blk"]
        t = g["t"]
        out = np.zeros((n, 3), dtype=np.float32)

        # ---------------- sky ------------------------------------------
        miss = blk == B.AIR
        if miss.any():
            out[miss] = self._sky(dirs[miss], sky, weather, tod)

        # ---------------- surfaces --------------------------------------
        hit = ~miss
        if hit.any():
            hi = np.nonzero(hit)[0]
            bk = blk[hi]
            base = P.COLOR_F[bk]
            vv = g["vox"][hi]
            jitter = P.voxel_hash(vv[:, 0], vv[:, 1], vv[:, 2]) * cfg.texture_noise
            base = base * (1.0 + jitter[:, None])
            nrm = g["nrm"][hi]
            ndl_raw = np.maximum((nrm * sdir[None, :]).sum(axis=1), 0.0)
            # wrap lighting: faces turned away from the sun keep a little
            # shaping instead of collapsing to flat ambient
            ndl = (ndl_raw + 0.30) / 1.30

            shadow = np.ones(hi.size, dtype=np.float32)
            if cfg.shadows and sun["intensity"] > 0.25:
                lit = ndl_raw > 0.01
                if lit.any():
                    pts = (o[None, :] + dirs[hi][lit] * t[hi][lit][:, None]
                           + nrm[lit] * 0.04).astype(np.float32)
                    occ = self._occluded(world, pts, sdir, cfg.shadow_steps)
                    s = np.ones(int(lit.sum()), dtype=np.float32)
                    s[occ] = 0.45
                    shadow[lit] = s

            # Sky dome ambient + a dim opposite-side fill + a ground bounce.
            # Without the fill, any face turned away from the sun renders black,
            # which hides most of the geometry an agent needs to read.
            skytint = sky["zenith"] * 0.55 + sky["horizon"] * 0.45
            ground = np.array([0.34, 0.30, 0.24], dtype=np.float32)
            ny = nrm[:, 1]
            ambient = (amb * (0.62 + 0.38 * np.maximum(ny, 0.0)))[:, None] \
                * skytint[None, :]
            bounce = (amb * 0.30 * np.maximum(-ny, 0.0))[:, None] * ground[None, :]
            fill_dir = -sdir
            fill = (np.maximum((nrm * fill_dir[None, :]).sum(axis=1), 0.0)
                    * 0.15 * sun["intensity"])[:, None] * skytint[None, :]
            direct = (ndl * shadow)[:, None] * scol[None, :]
            light = ambient + bounce + fill + direct

            # point lights (torches, fires, crystals)
            if lights:
                pts = (o[None, :] + dirs[hi] * t[hi][:, None]).astype(np.float32)
                light = light + self._point_lights(pts, nrm, lights)

            col = base * light
            em = B.EMISSIVE[bk]
            glow = em > 0
            if glow.any():
                col[glow] = col[glow] * 0.35 + base[glow] * (0.95 * em[glow])[:, None]

            # specular
            gl = B.GLOSS[bk]
            sp = gl > 0
            if sp.any():
                vdir = -dirs[hi][sp]
                hvec = vdir + sdir[None, :]
                hvec /= np.maximum(np.linalg.norm(hvec, axis=1, keepdims=True), 1e-6)
                spec = np.power(np.maximum((nrm[sp] * hvec).sum(axis=1), 0.0), 24.0)
                col[sp] += (spec * gl[sp] * shadow[sp])[:, None] * scol[None, :]
            out[hi] = col

        # ---------------- foliage alpha over the top ---------------------
        a = g["acc_a"]
        fo = a > 0.002
        if fo.any():
            fl = amb * 0.9 + 0.75 * sun["intensity"]
            fcol = g["acc_rgb"][fo] * fl
            out[fo] = out[fo] * (1.0 - a[fo])[:, None] + fcol

        # ---------------- water / lava volumes ---------------------------
        wl = g["water_len"]
        inw = wl > 0.01
        if inw.any():
            f = 1.0 - np.exp(-P.WATER_EXTINCTION * wl[inw])
            out[inw] = out[inw] * (1.0 - f)[:, None] \
                + P.WATER_TINT[None, :] * f[:, None] * (0.45 + 0.75 * sun["intensity"])
            # surface sheen: fresnel-ish rim plus a sun glint
            wt = g["water_t"][inw]
            surf = wt >= 0
            if surf.any():
                dd = dirs[inw][surf]
                up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                fres = np.power(1.0 - np.abs(dd[:, 1]), 4.0) * 0.55
                hvec = -dd + sdir[None, :]
                hvec /= np.maximum(np.linalg.norm(hvec, axis=1, keepdims=True), 1e-6)
                glint = np.power(np.maximum((hvec * up[None, :]).sum(axis=1), 0.0),
                                 64.0) * 0.8
                wi = np.nonzero(inw)[0][surf]
                out[wi] += (fres[:, None] * sky["horizon"][None, :]
                            + (glint * sun["intensity"])[:, None] * sun["color"][None, :])
        ll = g["lava_len"]
        inl = ll > 0.01
        if inl.any():
            f = 1.0 - np.exp(-1.6 * ll[inl])
            out[inl] = out[inl] * (1.0 - f)[:, None] \
                + P.LAVA_TINT[None, :] * (f * 1.15)[:, None]

        # ---------------- fog --------------------------------------------
        if cfg.fog:
            fogd = P.FOG_DISTANCE.get(weather, 140.0)
            tt = np.where(miss, cfg.max_dist, t)
            f = 1.0 - np.exp(-np.maximum(tt, 0.0) / fogd)
            f = np.where(miss, 0.0, f)        # the sky already is the fog colour
            fogcol = sky["horizon"] * (0.85 + 0.3 * sun["intensity"] * 0.3)
            out = out * (1.0 - f)[:, None] + fogcol[None, :] * f[:, None]

        return np.clip(out * 255.0, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    @staticmethod
    def _point_lights(pts: np.ndarray, nrm: np.ndarray, lights: list) -> np.ndarray:
        acc = np.zeros_like(pts)
        for lt in lights[:10]:
            lp = np.asarray(lt["pos"], dtype=np.float32)
            rad = float(lt.get("radius", 10.0))
            col = np.asarray(lt.get("color", (1.0, 0.75, 0.42)), dtype=np.float32)
            inten = float(lt.get("intensity", 1.0))
            dv = lp[None, :] - pts
            dist = np.linalg.norm(dv, axis=1)
            near = dist < rad
            if not near.any():
                continue
            dv = dv[near] / np.maximum(dist[near][:, None], 1e-5)
            atten = np.clip(1.0 - dist[near] / rad, 0.0, 1.0) ** 2
            ndl = np.maximum((nrm[near] * dv).sum(axis=1), 0.0) * 0.8 + 0.2
            acc[near] += (atten * ndl * inten)[:, None] * col[None, :]
        return acc

    # ------------------------------------------------------------------
    def _sky(self, d: np.ndarray, sky: dict, weather: str, tod: float) -> np.ndarray:
        from ..core import noise as N
        dy = d[:, 1]
        k = np.clip(dy * 1.7, 0.0, 1.0) ** 0.65
        col = sky["horizon"][None, :] * (1.0 - k)[:, None] \
            + sky["zenith"][None, :] * k[:, None]
        sun = sky["sun"]
        cosang = np.clip((d * sun["dir"][None, :]).sum(axis=1), -1.0, 1.0)
        if sun["elevation"] > -0.12:
            disk = np.power(np.maximum(cosang, 0.0), 2200.0)
            halo = np.power(np.maximum(cosang, 0.0), 22.0) * 0.32
            col += (disk * 2.4 + halo)[:, None] * sky["glow"][None, :]
        else:
            moon = np.power(np.maximum(cosang, 0.0), 2600.0)
            col += (moon * 1.6)[:, None] * np.array([0.9, 0.93, 1.0],
                                                    dtype=np.float32)[None, :]
            # stars
            up = dy > 0.02
            if up.any():
                sx = d[up, 0] / np.maximum(dy[up], 1e-3) * 18.0
                sz = d[up, 2] / np.maximum(dy[up], 1e-3) * 18.0
                h = N.hash_unit(np.floor(sx * 3).astype(np.int64),
                                np.floor(sz * 3).astype(np.int64), 991)
                star = (h > 0.9975).astype(np.float32) * (0.6 + 0.4 * h)
                col[up] += star[:, None] * 0.9
        if self.cfg.clouds and weather != "fog":
            up = dy > 0.035
            if up.any():
                px = d[up, 0] / dy[up] * 0.030
                pz = d[up, 2] / dy[up] * 0.030
                drift = tod * 26.0
                c = N.fbm2(px + drift * 0.01, pz, 4242, octaves=4)
                cover = {"clear": 0.16, "cloud": 0.42, "rain": 0.62,
                         "storm": 0.72, "snow": 0.58}.get(weather, 0.2)
                cl = np.clip((c + cover - 0.10) * 3.2, 0.0, 1.0)
                cl *= np.clip((dy[up] - 0.035) * 3.4, 0.0, 1.0)
                shade = 0.55 + 0.45 * max(sun["elevation"], 0.0)
                ccol = (np.array([1.0, 0.99, 0.97], dtype=np.float32) * shade
                        if weather in ("clear", "cloud")
                        else np.array([0.62, 0.64, 0.68], dtype=np.float32) * shade)
                col[up] = col[up] * (1.0 - cl)[:, None] + ccol[None, :] * cl[:, None]
        return np.clip(col, 0.0, 1.6)

    # ------------------------------------------------------------------
    def _draw_entities(self, cam: Camera, entities: list, rgb, depth, sem,
                       W: int, H: int, tod: float) -> None:
        f, r, u = cam.basis()
        tan_half = float(np.tan(np.deg2rad(cam.fov) * 0.5))
        aspect = W / max(H, 1)
        sun = P.sun_state(tod)
        # far entities first so near ones paint over them
        ents = sorted(entities,
                      key=lambda e: -float(np.dot(np.asarray(e["pos"], dtype=np.float32)
                                                  - cam.pos, f)))
        for e in ents:
            p = np.asarray(e["pos"], dtype=np.float32) - cam.pos
            along = float(np.dot(p, f))
            if along < 0.25 or along > self.cfg.max_dist:
                continue
            sx = float(np.dot(p, r)) / along / tan_half
            sy = float(np.dot(p, u)) / along / (tan_half / aspect)
            cx = (sx * 0.5 + 0.5) * W
            cy = (0.5 - sy * 0.5) * H
            rad = float(e.get("radius", 0.45))
            hgt = float(e.get("height", rad * 2.2))
            pw = rad / along / tan_half * (W * 0.5)
            ph = hgt * 0.5 / along / (tan_half / aspect) * (H * 0.5)
            if pw < 0.4 or ph < 0.4:
                pw, ph = max(pw, 0.5), max(ph, 0.5)
            x0, x1 = int(np.floor(cx - pw)), int(np.ceil(cx + pw))
            y0, y1 = int(np.floor(cy - ph)), int(np.ceil(cy + ph))
            if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
                continue
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W - 1, x1), min(H - 1, y1)
            xs = np.arange(x0, x1 + 1, dtype=np.float32)
            ys = np.arange(y0, y1 + 1, dtype=np.float32)
            nx = (xs[None, :] - cx) / max(pw, 1e-3)
            ny = (ys[:, None] - cy) / max(ph, 1e-3)
            rr = nx * nx + ny * ny
            mask = (rr <= 1.0) & (depth[y0:y1 + 1, x0:x1 + 1] > along)
            if not mask.any():
                continue
            col = np.asarray(e.get("color", (200, 90, 90)), dtype=np.float32) / 255.0
            # fake sphere normal for a bit of shape
            nz = np.sqrt(np.maximum(1.0 - rr, 0.0))
            nvec = (nx[None, ...] * r[:, None, None]
                    + (-ny)[None, ...] * u[:, None, None]
                    + nz[None, ...] * (-f)[:, None, None])
            ndl = np.clip((nvec * sun["dir"][:, None, None]).sum(axis=0), 0.0, 1.0)
            shade = sun["ambient"] * 0.9 + sun["intensity"] * 0.85 * ndl
            if e.get("emissive"):
                shade = np.maximum(shade, 1.0)
            px = np.clip(col[None, None, :] * shade[..., None] * 255.0, 0, 255)
            tgt = rgb[y0:y1 + 1, x0:x1 + 1]
            tgt[mask] = px[mask].astype(np.uint8)
            depth[y0:y1 + 1, x0:x1 + 1][mask] = along
            sem[y0:y1 + 1, x0:x1 + 1][mask] = np.uint8(e.get("semantic",
                                                             B.SEM_MONSTER))

    # ------------------------------------------------------------------
    @staticmethod
    def _weather_overlay(rgb, weather: str, intensity: float, frame: int) -> None:
        H, W, _ = rgb.shape
        rng = np.random.default_rng(1234 + int(frame))
        if weather in ("rain", "storm"):
            n = int(W * H * 0.012 * max(intensity, 0.5))
            if n <= 0:
                return
            xs = rng.integers(0, W, n)
            ys = rng.integers(0, H - 4, n)
            for dy in range(3):
                yy = np.clip(ys + dy, 0, H - 1)
                cur = rgb[yy, xs].astype(np.float32)
                rgb[yy, xs] = np.clip(cur * 0.55 + 150.0, 0, 255).astype(np.uint8)
        else:  # snow
            n = int(W * H * 0.010)
            xs = rng.integers(0, W, n)
            ys = rng.integers(0, H, n)
            cur = rgb[ys, xs].astype(np.float32)
            rgb[ys, xs] = np.clip(cur * 0.3 + 215.0, 0, 255).astype(np.uint8)
