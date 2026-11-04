"""Original procedural foliage, camp props and traveler meshes for Wildlands."""
from __future__ import annotations

import numpy as np

from open_world_bench.render.meshes import MeshBatch

from ..core import blocks as B


class ModelBatch(MeshBatch):
    """MeshBatch plus optional smooth normals for rounded organic shapes."""
    def __init__(self):
        super().__init__()
        self.smooth_normals = []

    def add(self, tris, color, obj_id=1, emissive=0.0, normals=None):
        tris = np.asarray(tris, np.float32).reshape(-1, 3, 3)
        if not len(tris):
            return
        super().add(tris, color, obj_id, emissive)
        self.smooth_normals.append(np.zeros_like(tris) if normals is None else normals)

# Subdivided octahedron: a rounded silhouette with only 32 faces.
_V = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
               (0, 0, 1), (0, 0, -1)], np.float32)
_F = np.array([(2, 0, 4), (2, 4, 1), (2, 1, 5), (2, 5, 0),
               (3, 4, 0), (3, 1, 4), (3, 5, 1), (3, 0, 5)])
_TRIS = []
for a, b, c in _V[_F]:
    ab, bc, ca = (a + b) / np.sqrt(2), (b + c) / np.sqrt(2), (c + a) / np.sqrt(2)
    _TRIS.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
_SPHERE = np.asarray(_TRIS, np.float32)


def ellipsoid(batch, pos, radii, color, semantic, yaw=0):
    vertices = _SPHERE * np.asarray(radii, np.float32)
    normals = _SPHERE / np.maximum(np.asarray(radii, np.float32), 1e-6)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-6)
    if yaw:
        c, s = np.cos(yaw), np.sin(yaw)
        vertices = vertices @ np.array(((c, 0, -s), (0, 1, 0), (s, 0, c)), np.float32)
        normals = normals @ np.array(((c, 0, -s), (0, 1, 0), (s, 0, c)), np.float32)
    batch.add(vertices + np.asarray(pos, np.float32), color, semantic, normals=normals)


def branch(batch, start, end, radius, color, semantic, taper=0.7):
    start, end = np.asarray(start), np.asarray(end)
    axis = end - start
    axis = axis / max(np.linalg.norm(axis), 1e-6)
    right = np.cross(axis, (0, 0, 1))
    if np.linalg.norm(right) < 0.01:
        right = np.cross(axis, (1, 0, 0))
    right /= max(np.linalg.norm(right), 1e-6)
    other = np.cross(axis, right)
    angles = np.arange(6) * (np.pi / 3)
    ring = (np.cos(angles)[:, None] * right + np.sin(angles)[:, None] * other) * radius
    low, high = start + ring, end + ring * taper
    for i in range(6):
        j = (i + 1) % 6
        batch.quad(low[i], low[j], high[j], high[i], color, semantic)


def tree(batch, pos, height, kind, variation):
    p = np.asarray(pos, np.float32)
    trunk = (0.32, 0.27, 0.20)
    leaf = np.array((0.32, 0.49, 0.19)) * (0.88 + variation * 0.22)
    sid = B.SEM_TREE
    branch(batch, p, p + (0, height, 0), 0.36, trunk, sid, taper=0.4)
    if kind == "pine":
        for level in range(3):
            y = height * (0.18 + level * 0.22)
            batch.cone(p + (0, y, 0), height * (0.30 - level * 0.065),
                       height * 0.40, (0.20 + level * 0.015, 0.37 + level * 0.035, 0.23),
                       sides=9, obj_id=sid)
    elif kind == "dead":
        for side in (-1, 1):
            branch(batch, p + (0, height * 0.65, 0), p + (side * 1.2, height * 0.9, 0.4),
                   0.16, trunk, sid, taper=0.3)
    elif kind == "palm":
        for angle in np.arange(7) * (2 * np.pi / 7):
            d = np.array((np.cos(angle), 0, np.sin(angle)))
            r = np.array((-d[2], 0, d[0])) * 0.55
            base = p + (0, height, 0)
            tip = base + d * 3.1 - (0, 0.7, 0)
            mid = base + d * 1.5 + (0, 0.5, 0)
            batch.quad(base, mid + r, tip, mid - r, leaf, sid)
    else:
        radius = min(3.2, height * 0.48 + 0.4)
        for offset, scale, tint in (((0, 0.3, 0), 1.0, 1), ((-0.65, -0.22, 0.15), 0.74, 0.91),
                                     ((0.6, -0.15, 0.25), 0.76, 1.08)):
            center = p + (0, height, 0) + np.asarray(offset) * radius
            ellipsoid(batch, center, np.array((radius, radius * 0.72, radius)) * scale,
                      leaf * tint, sid, variation * 4)


def grass(batch, pos, variation, time, flower=False):
    p = np.asarray(pos, np.float32)
    bend = np.sin(time * 1.8 + p[0] * 0.35 + p[2] * 0.2) * 0.12
    for k in range(4):
        angle = variation * 6.28 + k * 2.4
        offset = np.array((np.cos(angle), 0, np.sin(angle))) * 0.25
        right = np.array((-np.sin(angle), 0, np.cos(angle))) * 0.075
        base = p + offset
        mid = base + (bend * 0.3, 0.24, 0)
        tip = base + (bend, 0.48 + variation * 0.28, 0.12)
        col = np.array((0.48, 0.60, 0.23)) * (0.86 + 0.05 * k)
        batch.add([[base - right, base + right, mid + right * 0.6],
                   [base - right, mid + right * 0.6, tip]], col, B.SEM_PLANT)
    if flower:
        ellipsoid(batch, p + (0, 0.6, 0), (0.13, 0.10, 0.13), (0.92, 0.80, 0.47), B.SEM_PLANT)


def traveler(batch, feet, yaw, height=1.75, color=(0.22, 0.51, 0.62),
             semantic=B.SEM_SELF, phase=0, gliding=False):
    """Tapered cloth, rounded head, articulated limbs, scarf and equipment."""
    feet, scale = np.asarray(feet), height / 1.75
    forward = np.array((np.cos(yaw), 0, np.sin(yaw)))
    right = np.array((-forward[2], 0, forward[0]))

    def p(x, y, z=0):
        return feet + scale * (right * x + np.array((0, y, 0)) + forward * z)

    boot, skin, leather = (0.25, 0.23, 0.19), (0.82, 0.66, 0.46), (0.37, 0.27, 0.17)
    for sign in (-1, 1):
        swing = np.sin(phase) * 0.16 * sign
        branch(batch, p(sign * 0.13, 0.75), p(sign * 0.15, 0.10, swing),
               0.105 * scale, boot, semantic, taper=0.8)
        ellipsoid(batch, p(sign * 0.15, 0.08, 0.08 + swing), (0.105 * scale, 0.08 * scale, 0.18 * scale),
                  boot, semantic, -yaw)
        shoulder = p(sign * 0.27, 1.30)
        hand = p(sign * (0.65 if gliding else 0.33), 1.58 if gliding else 0.86, -swing)
        branch(batch, shoulder, hand, 0.09 * scale, color, semantic)
        ellipsoid(batch, hand, (0.075 * scale,) * 3, skin, semantic)
    batch.cone(p(0, 0.68), 0.30 * scale, 0.60 * scale, color, sides=8, obj_id=semantic, taper=0.78)
    batch.cone(p(0, 0.85), 0.25 * scale, 0.07 * scale, leather, sides=8, obj_id=semantic, taper=1)
    ellipsoid(batch, p(0, 1.52, 0.015), np.array((0.18, 0.22, 0.17)) * scale, skin, semantic)
    ellipsoid(batch, p(0, 1.66, -0.025), np.array((0.19, 0.15, 0.18)) * scale,
              (0.68, 0.53, 0.27), semantic)
    # A short ochre scarf, satchel and diagonal scabbard distinguish the traveler.
    batch.quad(p(-0.20, 1.35, -0.15), p(0.20, 1.35, -0.15),
               p(0.25, 0.84, -0.28), p(-0.15, 0.92, -0.32), (0.70, 0.53, 0.27), semantic)
    ellipsoid(batch, p(-0.25, 0.82, -0.05), np.array((0.12, 0.15, 0.12)) * scale, leather, semantic)
    branch(batch, p(-0.25, 0.65, -0.27), p(0.30, 1.42, -0.27), 0.04 * scale, boot, semantic)
    if gliding:
        for sign in (-1, 1):
            batch.quad(p(0, 2.4, -0.45), p(sign * 1.7, 2.05, -0.25),
                       p(sign * 1.25, 2.15, 1.15), p(0, 2.4, 0.85), (0.81, 0.72, 0.48), semantic)
            branch(batch, p(sign * 0.65, 1.58), p(sign * 1.2, 2.16, 0.15),
                   0.015, leather, semantic)


def prop(batch, block, pos, time=0):
    p = np.asarray(pos, np.float32)
    stone, sid = (0.40, 0.43, 0.39), B.SEM_INTERACTIVE
    if block == B.COOKPOT:
        ellipsoid(batch, p + (0, 0.35, 0), (0.48, 0.34, 0.48), (0.22, 0.26, 0.25), sid)
        batch.cone(p + (0, 0.52, 0), 0.43, 0.07, (0.12, 0.16, 0.15), 12, sid, taper=1)
        for x, z in ((-0.3, -0.2), (0.3, -0.2), (0, 0.3)):
            branch(batch, p + (x, 0, z), p + (x * 0.8, 0.3, z * 0.8), 0.06, stone, sid)
    elif block in (B.CAMPFIRE, B.CAMPFIRE_LIT):
        for angle in np.arange(8) * np.pi / 4:
            ellipsoid(batch, p + (np.cos(angle) * 0.42, 0.08, np.sin(angle) * 0.42),
                      (0.16, 0.12, 0.15), stone, sid)
        for side in (-1, 1):
            branch(batch, p + (-0.32, 0.15, side * 0.24), p + (0.32, 0.15, -side * 0.24),
                   0.10, (0.35, 0.25, 0.14), sid)
        if block == B.CAMPFIRE_LIT:
            batch.cone(p + (0, 0.12, 0), 0.3, 0.7 + 0.12 * np.sin(time * 6),
                       (1, 0.57, 0.17), 7, B.SEM_FIRE, emissive=1)
    elif block == B.BUSH:
        ellipsoid(batch, p + (0, 0.4, 0), (0.65, 0.55, 0.6), (0.36, 0.49, 0.20), B.SEM_PLANT)
    elif block in (B.FIRE, B.TORCH_WALL):
        if block == B.TORCH_WALL:
            branch(batch, p, p + (0, 0.65, 0), 0.07, (0.32, 0.24, 0.13), B.SEM_FIRE)
        batch.cone(p + (0, 0.25, 0), 0.25, 0.8 + 0.1 * np.sin(time * 7),
                   (1, 0.57, 0.16), 7, B.SEM_FIRE, emissive=1)
    elif block == B.CRYSTAL:
        batch.cone(p, 0.4, 1.3, (0.47, 0.79, 0.82), 5, B.SEM_ORE, emissive=0.35)
    elif block == B.VINE:
        for side in (-1, 1):
            branch(batch, p + (side * 0.18, 0, 0), p + (side * 0.12, 1, 0.06),
                   0.035, (0.35, 0.49, 0.18), B.SEM_PLANT)
    elif block == B.GLASS:
        batch.box(p + (0, 0.5, 0), (1, 1, 1), (0.51, 0.72, 0.73), obj_id=B.SEM_STRUCTURE)


def entity(batch, e, time):
    p = np.asarray(e["pos"], np.float32)
    height, radius = e.get("height", 1), e.get("radius", 0.4)
    sid = e.get("semantic", B.SEM_MONSTER)
    col = np.asarray(e.get("color", (150, 160, 90))) / 255
    if sid in (B.SEM_SELF, B.SEM_NPC) or (sid == B.SEM_MONSTER and height > 1.3):
        traveler(batch, p - (0, height / 2, 0), e.get("yaw", 0), height, col, sid,
                 e.get("phase", 0), e.get("gliding", False))
    elif e.get("kind") == "chest":
        batch.box(p, (0.9, 0.5, 0.65), (0.40, 0.27, 0.13), obj_id=sid)
        ellipsoid(batch, p + (0, 0.25, 0), (0.46, 0.22, 0.33), (0.55, 0.42, 0.20), sid)
        for x in (-0.28, 0.28):
            batch.box(p + (x, 0.12, 0), (0.06, 0.58, 0.69), (0.70, 0.59, 0.30), obj_id=sid)
    elif e.get("kind") == "plant":
        branch(batch, p - (0, 0.4, 0), p + (0, 0.1, 0), 0.035, (0.29, 0.43, 0.13), sid)
        ellipsoid(batch, p + (0.08, 0, 0), (0.16, 0.20, 0.16), (0.79, 0.28, 0.13), sid)
        ellipsoid(batch, p + (-0.11, -0.12, 0), (0.19, 0.07, 0.12), (0.43, 0.58, 0.20), sid)
    else:
        ellipsoid(batch, p, (radius, height / 2, radius), col, sid)
