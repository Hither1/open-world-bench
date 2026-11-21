"""Hamlets, farmsteads and working buildings for the organic Skylands campaigns.

Walls, floors, roofs, chimneys and doorways are collision voxels authored at
reset, before any reference trace is collected; cladding, shingles and props
live in :mod:`owb.render.settlements`. Plots sit off the trails, clear of every
site, fixture, waterway and shaft, on natural ground that is graded to the
layer's walking level. Plot choice and appearance are keyed to stable world
identity (seed, task and building name), never to the simulation RNG.

Functional buildings reuse the Tempest envelopes (unchanged, so Tempest's
published traces keep their geometry); cottages are the residential kind.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from ..core import blocks as B
from . import organic as O
from .tempest import envelope as tempest_envelope


#: Building kind for each site kind, per layer (0 depths, 1 surface, 2 sky).
SITE_BUILDINGS = {
    1: {"clue": "archive", "raft": "boathouse", "ascend": "workshop", "stage": "boathouse", "return": "boathouse",
        "hoist": "gantry", "cache": "shelter", "take": "workshop", "reverse": "gantry", "station": "pump_house",
        "update": "archive", "retrieve": "workshop", "compose": "foundry", "delivery": "boathouse"},
    0: {"dock": "workshop", "ascend": "shelter", "extract": "archive", "raft": "boathouse", "freight": "foundry",
        "outpost": "shelter", "station": "pump_house"},
    2: {"calibrate": "observatory", "clue": "signal_tower", "reverse": "archive", "ascend_station": "observatory"},
}
#: Extra structures on sky islands, which have room for a second landmark.
SKY_EXTRA = dict(calibrate="signal_tower", ascend_station="cistern")
#: Cottages beside each surface site; the landing hub is a larger hamlet.
SITE_COTTAGES, HUB_COTTAGES = 2, 4
FARMSTEADS = dict(a=10, b=16)
TALL = ("signal_tower", "observatory", "foundry")


def appearance(seed, task, name):
    return int.from_bytes(hashlib.sha256(f"skylands-settlement:{seed}:{task}:{name}".encode()).digest()[:4], "little")


def cottage_dims(record):
    """Half-width ``a`` (along the ridge), half-depth ``c`` and wall block."""
    v = record["appearance"]
    return 4+(v % 2), 3+(v >> 1) % 2, (B.PLANK, B.BRICK)[(v >> 2) % 2]


def cottage_cells(record):
    """Integer closed boxes in the cottage's local frame (door facing -z)."""
    a, c, wall = cottage_dims(record)
    chimney = a-1 if (record["appearance"] >> 3) % 2 else -(a-1)
    parts = []

    def box(lo, hi, block):
        parts.append((tuple(lo), tuple(hi), block))

    box((-a-1, -2, -c-1), (a+1, -1, c+1), B.BRICK)            # plinth and front stoop
    for x in (-a, a):
        box((x, 0, -c), (x, 4, c), wall)
    for z in (-c, c):
        box((-a, 0, z), (a, 4, z), wall)
    for dz in range(-c, c+1):                                   # gable triangles
        top = 4+(c+1-abs(dz))
        for x in (-a, a):
            box((x, 5, dz), (x, top-1, dz), wall)
    for dz in range(-c-1, c+2):                                 # stepped 45 degree roof
        y = 4+(c+1-abs(dz))
        box((-a-1, y, dz), (a+1, y, dz), B.PLANK)
    box((0, 0, -c), (0, 1, -c), B.AIR)                          # door
    for x in (-(a-2), a-2):                                     # front and back windows
        for z in (-c, c):
            box((x, 1, z), (x, 2, z), B.AIR)
    for x in (-a, a):                                           # side windows
        box((x, 1, 0), (x, 2, 0), B.AIR)
    box((chimney, 0, c-1), (chimney, c+6, c-1), B.BRICK)
    return parts


def rotate_cell(i, j, k):
    for _ in range(k % 4):
        i, j = -j, i
    return i, j


def local_parts(record):
    if record["kind"] == "cottage":
        parts = cottage_cells(record)
    else:
        parts = tempest_envelope(record["kind"])
    k = record.get("rotation", 0)
    out = []
    for lo, hi, block in parts:
        a = rotate_cell(lo[0], lo[2], k); b = rotate_cell(hi[0], hi[2], k)
        out.append(((min(a[0], b[0]), lo[1], min(a[1], b[1])), (max(a[0], b[0]), hi[1], max(a[1], b[1])), block))
    return out


def footprint(kind, record=None):
    """Half extents (x, z) of the plot, including props and yard lamps, and the
    clearance height."""
    if kind == "cottage":
        a, c, _ = cottage_dims(record) if record else (5, 4, None)
        r = max(a, c)+3
        return r, r, c+9
    return 10, 11, 24


def door_normal(k):
    n = (0, -1)
    for _ in range(k % 4):
        n = (-n[1], n[0])
    return np.array(n, float)


class Planner:
    """Finds plots in a reset campaign world and installs buildings there."""

    def __init__(self, env):
        self.env = env
        w = self.w = env.world
        self.levels = env.levels
        sx, sz = w.size_x, w.size_z
        self.road_distance = distance_transform_edt(~env._road_mask)
        wet1 = (w.vox[:, 30:83, :] == B.WATER).any(axis=1)
        wet0 = (w.vox[:, 1:31, :] == B.WATER).any(axis=1)
        self.water_distance = {0: distance_transform_edt(~wet0), 1: distance_transform_edt(~wet1),
                               2: np.full((sx, sz), 1e9)}
        self.top = O.surface_top(w, 30, 82)
        self.reserved = self._reserved()
        self.occupied = {layer: np.zeros((sx, sz), bool) for layer in range(3)}
        self.records = []
        self.mask = np.zeros((sx, sz), bool)          # surface plots, for scenery and creatures

    def _reserved(self):
        env, w = self.env, self.w
        sx, sz = w.size_x, w.size_z
        xx, zz = np.ogrid[:sx, :sz]
        m = np.zeros((sx, sz), bool)

        def disc(p, r):
            x, z = float(p[0]), float(p[-1])
            lo0, hi0 = max(0, int(x-r-1)), min(sx, int(x+r+2))
            lo1, hi1 = max(0, int(z-r-1)), min(sz, int(z+r+2))
            m[lo0:hi0, lo1:hi1] |= (xx[lo0:hi0]-x)**2+(zz[:, lo1:hi1]-z)**2 <= r*r

        def segment(a, b, r):
            a = np.asarray(a, float)[[0, 2]]; b = np.asarray(b, float)[[0, 2]]
            for t in np.linspace(0, 1, max(2, int(np.linalg.norm(b-a)/2)+1)):
                disc(a+(b-a)*t, r)

        for site in env.milestones:
            disc(site.pos[[0, 2]], 15)
        for obj in env.targets.values():
            disc(np.asarray(obj["pos"])[[0, 2]], 8)
        for r in env.rafts.values():
            points = [r["pos"], *r.get("route", []), r["destination"]]
            for p, q in zip(points[:-1], points[1:]):
                segment(p, q, 7)
            if r.get("receiver") is not None:
                disc(np.asarray(r["receiver"])[[0, 2]], 7)
        for mv in env.movers.values():
            disc(np.asarray(mv["pos"])[[0, 2]], 10)
        for p in getattr(env, "refuge_docks", []):
            disc(np.asarray(p)[[0, 2]], 10)
        dx, dz = env.descent_xy
        disc(env.vertical_xy, 22); disc((dx, dz), 15); disc((dx+19, dz), 15)
        disc(env.point(60, 180), 10); disc(env.point(90, 90), 12)
        if env.task == "a":
            disc(env.point(460, 520), 14)
        for lo, hi in getattr(env, "_protect_boxes", []):
            m[max(0, lo[0]-6):hi[0]+7, max(0, lo[2]-6):hi[2]+7] = True
        m |= binary_dilation(env._protected(2), iterations=6)
        return m

    # -- plot tests ------------------------------------------------------
    def fits(self, kind, x, z, layer, record):
        w, y = self.w, self.levels[layer]
        hx, hz, height = footprint(kind, record)
        if not (hx+16 <= x < w.size_x-hx-16 and hz+16 <= z < w.size_z-hz-16):
            return False
        x0, x1, z0, z1 = x-hx, x+hx+1, z-hz, z+hz+1
        if self.reserved[x0:x1, z0:z1].any() or self.occupied[layer][x0-4:x1+4, z0-4:z1+4].any():
            return False
        if self.env._road_mask[x0-3:x1+3, z0-3:z1+3].any():
            return False
        if self.water_distance[layer][x0:x1, z0:z1].min() < (16 if layer == 1 else 6):
            return False
        clear = w.vox[x0:x1, y:y+height+1, z0:z1]
        if layer == 1:
            top = self.top[x0-2:x1+2, z0-2:z1+2]
            if (top < y-7).any() or (top > y+4).any():
                return False
            # Only natural ground is cut, and whole trees are never sliced.
            if (~O.CUTTABLE[clear] & (clear != B.AIR)).any():
                return False
            band = w.vox[x0-2:x1+2, y-8:y+height+1, z0-2:z1+2]
            if np.isin(band, (B.LOG, B.LEAVES)).any():
                return False
        else:
            if not B.SOLID[w.vox[x0-1:x1+1, y-1, z0-1:z1+1]].all() or (clear != B.AIR).any():
                return False
            if layer == 2 and np.isin(w.vox[x0-2:x1+2, y:y+height+1, z0-2:z1+2], (B.LOG, B.LEAVES)).any():
                return False
        return True

    def rotation(self, x, z):
        """Turn the cottage door toward the nearest trail."""
        d = self.road_distance
        g = np.array((d[min(x+3, d.shape[0]-1), z]-d[max(x-3, 0), z], d[x, min(z+3, d.shape[1]-1)]-d[x, max(z-3, 0)]))
        toward = -g if np.linalg.norm(g) > 1e-6 else np.array((0., -1.))
        return int(max(range(4), key=lambda k: float(door_normal(k) @ toward)))

    def place(self, name, kind, anchor, layer, radii, phase):
        env = self.env
        record = dict(id=name, kind=kind, layer=layer, appearance=appearance(env._seed, env.task, name), rotation=0)
        angles = (np.arange(16)+phase)*np.pi/8
        for radius in radii:
            for angle in angles:
                x = int(round(anchor[0]+radius*np.cos(angle))); z = int(round(anchor[1]+radius*np.sin(angle)))
                if self.fits(kind, x, z, layer, record):
                    if kind == "cottage":
                        record["rotation"] = self.rotation(x, z)
                    record["pos"] = [x, self.levels[layer], z]
                    hx, hz, _ = footprint(kind, record)
                    self.occupied[layer][x-hx:x+hx+1, z-hz:z+hz+1] = True
                    self.records.append(record)
                    return record
        return None

    # -- installation -----------------------------------------------------
    def install(self, record):
        env, w = self.env, self.w
        x, y, z = record["pos"]
        hx, hz, height = footprint(record["kind"], record)
        if record["layer"] == 1:
            # A graded yard at the walking level, blended into the hillside.
            shape = (w.size_x, w.size_z)
            mask = O.rounded_box(shape, (x-hx, z-hz), (x+hx+1, z+hz+1), 2.5)
            xs, zs = np.nonzero(mask)
            for dy in range(height+1):
                col = w.vox[xs, y+dy, zs]
                w.vox[xs, y+dy, zs] = np.where(O.CUTTABLE[col], B.AIR, col)
            w.vox[xs, y-1, zs] = B.GRASS
            env._fill_under(w, xs, zs, y)
            env._surface_grade(w, mask, y-1)
            self.mask[x-hx-3:x+hx+4, z-hz-3:z+hz+4] = True
        p = np.array((x, y, z))
        for lo, hi, block in local_parts(record):
            w.fill_box(p+lo, p+hi, block)


def plan(env):
    """Author every settlement in ``env``; returns (records, surface mask)."""
    P = Planner(env)
    rng = np.random.default_rng(O.stable_seed("skylands-settlements", env._seed, env.task))
    hub = np.asarray(env.point(90, 90), float)
    groups = {}
    for m in env.milestones:
        groups.setdefault((m.layer, int(m.x), int(m.z)), []).append(m)
    for (layer, x, z), sites in sorted(groups.items(), key=lambda g: (g[1][0].id)):
        site = sites[0]
        kind = SITE_BUILDINGS[layer].get(site.kind, "workshop")
        if layer == 1 and site.kind == "raft" and "boathouse" in [r["kind"] for r in P.records[-6:]]:
            kind = "workshop"
        anchor = (x, z)
        radii = (18, 22, 27, 32, 38) if layer == 2 else (20, 26, 32, 40, 50, 62, 76)
        phase = float(rng.uniform(0, 16))
        P.place(f"{site.id}-{kind}", kind, anchor, layer, radii, phase)
        if layer == 2 and site.kind in SKY_EXTRA:
            extra = SKY_EXTRA[site.kind]
            P.place(f"{site.id}-{extra}", extra, anchor, layer, radii, phase+5)
        if layer == 1:
            count = HUB_COTTAGES if np.hypot(*(np.asarray(anchor)-hub)) < 1 else SITE_COTTAGES
            for k in range(count):
                P.place(f"{site.id}-cottage-{k}", "cottage", anchor, 1, (16, 20, 24, 29, 35, 42, 50), phase+3+4.5*k)
    # Farmsteads along the surface trails, away from the sites.
    road = env._road_mask & np.isin(env.world.vox[:, env.levels[1]-1, :], (B.GRAVEL, B.DIRT))
    cells = np.argwhere(road)
    sites = [np.asarray(m.pos)[[0, 2]] for m in env.milestones]
    steads = []
    for i in rng.permutation(len(cells)):
        if len(steads) >= FARMSTEADS[env.task]:
            break
        c = cells[i].astype(float)
        if min(np.hypot(*(c-s)) for s in sites) < 60 or any(np.hypot(*(c-s)) < 80 for s in steads):
            continue
        name = f"farmstead-{len(steads):02}"
        first = P.place(f"{name}-cottage-0", "cottage", c, 1, (12, 15, 18, 22), float(rng.uniform(0, 16)))
        if first is None:
            continue
        steads.append(c)
        home = np.asarray(first["pos"], float)[[0, 2]]
        P.place(f"{name}-cottage-1", "cottage", home, 1, (17, 20, 24), float(rng.uniform(0, 16)))
        if len(steads) % 3 == 0:
            P.place(f"{name}-shelter", "shelter", home, 1, (22, 26, 31), float(rng.uniform(0, 16)))
    for record in P.records:
        P.install(record)
    env.world.recompute_heightmaps()
    return P.records, P.mask


def lights(records):
    out = []
    for r in records:
        p = np.asarray(r["pos"], float)
        if r["kind"] == "cottage":
            a, c, _ = cottage_dims(r)
            i, j = rotate_cell(-1, -c-1, r["rotation"])     # lantern beside the door
            out.append(dict(pos=(p+(i+.5, 2.6, j+.5)).tolist(), color=[1., .70, .38], power=260., radius=.25))
            i, j = rotate_cell(0, 0, r["rotation"])          # warm interior
            out.append(dict(pos=(p+(i+.5, 2.4, j+.5)).tolist(), color=[1., .66, .36], power=420., radius=.6))
        else:
            out.append(dict(pos=(p+(-9.5, 6., -10.)).tolist(), color=[.95, .72, .42],
                            power=4500. if p[1] < 30 else 1800., radius=2.))
            if r["kind"] not in ("gantry", "signal_tower", "cistern"):
                for x in (-2.7, 3.7):
                    out.append(dict(pos=(p+(x, 3.1, -7.)).tolist(), color=[1., .68, .32], power=750., radius=.3))
    return out
