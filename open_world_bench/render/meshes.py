"""Procedural mesh primitives, material tags, and character articulation.

Meshes are plain triangle soups: `verts` is (T, 3, 3) and `colors` is (T, 3).
Organic assets include authored normals. The same asset families feed policy
observations and the denser Cycles presentation geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import math3d as mth
from ..character import CharacterState, MoveState
from .style import CHARACTER_PALETTE

MATERIAL_NAMES = ("surface", "stone", "bark", "foliage", "metal", "cloth", "skin", "terrain")

# Unit cube triangles (centred, side 1), wound counter-clockwise outward.
_CUBE_V = np.array([
    [-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
    [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5],
], dtype=np.float32)
_CUBE_F = np.array([
    [0, 2, 1], [0, 3, 2],      # -z
    [4, 5, 6], [4, 6, 7],      # +z
    [0, 4, 7], [0, 7, 3],      # -x
    [1, 2, 6], [1, 6, 5],      # +x
    [3, 7, 6], [3, 6, 2],      # +y
    [0, 1, 5], [0, 5, 4],      # -y
], dtype=np.int32)


@dataclass
class MeshBatch:
    """Accumulates triangles with per-triangle colour, emission and object id."""

    verts: list[np.ndarray] = field(default_factory=list)
    colors: list[np.ndarray] = field(default_factory=list)
    emissive: list[np.ndarray] = field(default_factory=list)
    ids: list[np.ndarray] = field(default_factory=list)
    normals: list[np.ndarray] = field(default_factory=list)
    material_ids: list[np.ndarray] = field(default_factory=list)
    current_material: int = 0

    def add(self, tris: np.ndarray, color, obj_id: int = 1, emissive: float = 0.0,
            vertex_normals=None, material=None) -> None:
        tris = np.asarray(tris, dtype=np.float32).reshape(-1, 3, 3)
        if len(tris) == 0:
            return
        self.verts.append(tris)
        self.colors.append(np.broadcast_to(np.asarray(color, dtype=np.float32), (len(tris), 3)).copy())
        self.emissive.append(np.full(len(tris), float(emissive), dtype=np.float32))
        self.ids.append(np.full(len(tris), int(obj_id), dtype=np.int32))
        self.normals.append(np.zeros_like(tris) if vertex_normals is None
                            else np.asarray(vertex_normals, np.float32).reshape(-1, 3, 3))
        tag = self.current_material if material is None else material
        self.material_ids.append(np.broadcast_to(np.asarray(tag, np.int8), (len(tris),)).copy())

    def box(self, center, size, color, yaw: float = 0.0, obj_id: int = 1, emissive: float = 0.0) -> None:
        size = np.asarray(size, dtype=np.float32)
        v = _CUBE_V * size[None, :]
        if yaw:
            c, s = np.cos(yaw), np.sin(yaw)
            r = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
            v = v @ r.T
        v = v + np.asarray(center, dtype=np.float32)[None, :]
        self.add(v[_CUBE_F], color, obj_id, emissive)

    def cone(self, base, radius: float, height: float, color, sides: int = 7,
             obj_id: int = 1, emissive: float = 0.0, taper: float = 0.0) -> None:
        base = np.asarray(base, dtype=np.float32)
        ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False, dtype=np.float32)
        ring = np.stack([np.cos(ang) * radius, np.zeros_like(ang), np.sin(ang) * radius], axis=1) + base
        top_r = radius * taper
        top_ring = np.stack([np.cos(ang) * top_r, np.zeros_like(ang), np.sin(ang) * top_r], axis=1)
        top_ring = top_ring + base + np.array([0.0, height, 0.0], dtype=np.float32)
        tris = []
        for i in range(sides):
            j = (i + 1) % sides
            if taper <= 1e-4:
                tris.append([ring[i], ring[j], top_ring[0]])
            else:
                tris.append([ring[i], ring[j], top_ring[j]])
                tris.append([ring[i], top_ring[j], top_ring[i]])
        self.add(np.asarray(tris, dtype=np.float32), color, obj_id, emissive)

    def quad(self, p0, p1, p2, p3, color, obj_id: int = 1, emissive: float = 0.0) -> None:
        tris = np.asarray([[p0, p1, p2], [p0, p2, p3]], dtype=np.float32)
        self.add(tris, color, obj_id, emissive)

    def ellipsoid(self, center, radii, color, yaw=0.0, obj_id=1, sides=8, rings=4):
        """Rounded silhouette with a small, deterministic triangle budget."""
        latitude = np.linspace(0, np.pi, rings + 1)
        longitude = np.linspace(0, 2 * np.pi, sides, endpoint=False)
        vertices = np.stack(np.broadcast_arrays(
            np.sin(latitude[:, None]) * np.cos(longitude),
            np.cos(latitude[:, None]) * np.ones(sides),
            np.sin(latitude[:, None]) * np.sin(longitude)), axis=-1)
        normals = vertices / np.maximum(np.asarray(radii), 1e-8)
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
        vertices *= np.asarray(radii)
        c, s = np.cos(yaw), np.sin(yaw)
        vertices = vertices @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]]).T
        normals = normals @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]]).T
        vertices += np.asarray(center)
        triangles, smooth = [], []
        for row in range(rings):
            for col in range(sides):
                nxt = (col + 1) % sides
                if row > 0:
                    triangles.append([vertices[row, col], vertices[row + 1, col], vertices[row, nxt]])
                    smooth.append([normals[row, col], normals[row + 1, col], normals[row, nxt]])
                if row < rings - 1:
                    triangles.append([vertices[row, nxt], vertices[row + 1, col], vertices[row + 1, nxt]])
                    smooth.append([normals[row, nxt], normals[row + 1, col], normals[row + 1, nxt]])
        self.add(np.asarray(triangles), color, obj_id, vertex_normals=smooth)

    def tube(self, start, end, radius, color, obj_id=1, end_radius=None, sides=6):
        """Tapered connected limb, trunk, or branch in arbitrary orientation."""
        a, b = np.asarray(start), np.asarray(end)
        axis = mth.norm(b - a)
        helper = np.array([1, 0, 0]) if abs(axis[1]) > 0.9 else np.array([0, 1, 0])
        right = mth.norm(np.cross(axis, helper))
        up = np.cross(axis, right)
        angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
        radial = np.cos(angles)[:, None] * right + np.sin(angles)[:, None] * up
        lo = a + radial * radius
        hi = b + radial * (radius if end_radius is None else end_radius)
        triangles = []
        for i in range(sides):
            j = (i + 1) % sides
            triangles.extend([[lo[i], hi[i], hi[j]], [lo[i], hi[j], lo[j]],
                              [a, lo[j], lo[i]], [b, hi[i], hi[j]]])
        self.add(np.asarray(triangles), color, obj_id)

    def finish(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.verts:
            z3 = np.zeros((0, 3, 3), dtype=np.float32)
            return z3, np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32)
        return (np.concatenate(self.verts), np.concatenate(self.colors),
                np.concatenate(self.emissive), np.concatenate(self.ids))


# --------------------------------------------------------------------- props
def add_prop(batch: MeshBatch, kind_name: str, pos: np.ndarray, yaw: float, scale: float, obj_id: int = 1) -> None:
    """Stable per-instance botanical and geological variations."""
    if kind_name not in ("pine","broadleaf","sakura","bamboo","rock","boulder","bush","grass","crystal"):
        return
    from .assets import add_detailed_prop
    add_detailed_prop(batch,kind_name,np.asarray(pos),yaw,scale,obj_id)


def add_entity(batch: MeshBatch, kind_name: str, pos: np.ndarray, yaw: float, used: bool,
               obj_id: int, bob: float = 0.0, *, identity=None, visual_role="", visual_version=1) -> None:
    from .assets import add_detailed_entity
    add_detailed_entity(batch,kind_name,pos,yaw,used,obj_id,bob,identity=identity,
                        visual_role=visual_role,visual_version=visual_version)


# ----------------------------------------------------------------- character
def _humanoid(batch: MeshBatch, feet: np.ndarray, yaw: float, obj_id: int,
              tunic, hair, phase: float, scale: float = 1.0, lean: float = 0.0,
              arms_out: bool = False, skin=None, hair_style: int = 0) -> None:
    """An original adventurer silhouette with fitted clothing and jointed limbs."""
    pal = dict(CHARACTER_PALETTE)
    if skin is not None:
        pal["skin"] = skin
    s = scale
    c, sn = np.cos(yaw), np.sin(yaw)

    def local(x, y, z):
        """Body-local (right, up, forward) -> world, with a forward lean."""
        y2 = y * np.cos(lean) - z * np.sin(lean)
        z2 = y * np.sin(lean) + z * np.cos(lean)
        return np.asarray(feet, dtype=np.float32) + np.array(
            [x * c - z2 * sn, y2, x * sn + z2 * c], dtype=np.float32) * np.float32(s)

    swing = np.sin(phase * 2.0 * np.pi) * 0.22
    gold = (0.88, 0.73, 0.43)
    cloth = np.asarray(tunic)
    for side in (-1, 1):
        batch.current_material = 5
        hip = local(side * 0.12, 0.88, 0)
        knee = local(side * 0.13, 0.49, side * swing)
        ankle = local(side * 0.14, 0.12, -side * swing * 0.5)
        batch.tube(hip, knee, 0.09 * s, (0.31, 0.34, 0.43), obj_id, end_radius=0.07 * s)
        batch.tube(knee, ankle, 0.075 * s, pal["boots"], obj_id, end_radius=0.055 * s)
        batch.ellipsoid(ankle + (0, -0.05 * s, 0), (0.08 * s, 0.065 * s, 0.14 * s),
                        pal["boots"], yaw, obj_id, sides=6, rings=3)
        shoulder = local(side * 0.23, 1.35, 0)
        elbow = local(side * (0.48 if arms_out else 0.29), 1.23 if arms_out else 1.08, -side * swing)
        hand = local(side * (0.70 if arms_out else 0.30), 1.35 if arms_out else 0.88, -side * swing * 1.3)
        batch.tube(shoulder, elbow, 0.095 * s, tunic, obj_id, end_radius=0.065 * s)
        batch.tube(elbow, hand, 0.06 * s, pal["trim"], obj_id, end_radius=0.045 * s)
        batch.current_material = 6
        batch.ellipsoid(hand, (0.055 * s, 0.075 * s, 0.05 * s), pal["skin"], yaw, obj_id, sides=6, rings=3)

    batch.current_material = 5
    batch.ellipsoid(local(0, 1.20, 0), (0.225 * s, 0.30 * s, 0.14 * s), tunic, yaw, obj_id)
    batch.cone(local(0, 0.70, 0), 0.29 * s, 0.32 * s, cloth * 0.85, 8, obj_id, taper=0.60)
    batch.tube(local(0, 0.97, 0), local(0, 1.02, 0), 0.19 * s, gold, obj_id, sides=8)
    batch.current_material = 6
    batch.tube(local(0, 1.42, 0), local(0, 1.53, 0), 0.065 * s, pal["skin"], obj_id)
    batch.ellipsoid(local(0, 1.67, 0), (0.155 * s, 0.20 * s, 0.145 * s), pal["skin"], yaw, obj_id)
    batch.current_material = 5
    batch.ellipsoid(local(0, 1.78, 0.025), (0.18 * s, 0.145 * s, 0.17 * s), hair, yaw, obj_id)
    # Layered locks, swept fringe, and a small clasp remain legible from behind.
    for side in (-1, 1):
        if hair_style == 0:
            batch.tube(local(side * 0.14, 1.77, 0.04), local(side * 0.16, 1.43, 0.12),
                       0.075 * s, hair, obj_id, end_radius=0.018 * s)
        batch.tube(local(side * 0.09, 1.82, -0.10), local(side * 0.12, 1.67, -0.14),
                   0.055 * s, hair, obj_id, end_radius=0.006 * s)
        batch.ellipsoid(local(side * 0.058, 1.68, -0.135), (0.026 * s, 0.035 * s, 0.012 * s),
                        (0.16, 0.24, 0.33), yaw, obj_id, sides=5, rings=3)
    batch.ellipsoid(local(0, 1.54, 0.14), (0.045 * s, 0.065 * s, 0.035 * s), gold, yaw, obj_id, sides=5, rings=3)
    if hair_style == 2:
        batch.tube(local(0,1.75,.14),local(.04,1.36,.20),.07*s,hair,obj_id,end_radius=.025*s)
    elif hair_style == 3:
        batch.ellipsoid(local(0,1.82,0),(.23*s,.10*s,.21*s),cloth*.72,yaw,obj_id)
        batch.box(local(0,1.80,-.14),(.36*s,.035*s,.24*s),cloth*.72,yaw,obj_id)
    batch.current_material = 0


def add_character(batch: MeshBatch, st: CharacterState, obj_id: int = 2, spec=None) -> None:
    """Build the player avatar, posed for its current movement state."""
    pal = dict(CHARACTER_PALETTE)
    if spec is not None:
        pal.update(tunic=spec.tunic, hair=spec.hair)
        pal["cape"] = tuple(np.asarray(spec.tunic) * 0.25 + np.array([0.96, 0.94, 0.82]) * 0.75)
    gliding = st.state == MoveState.GLIDING
    climbing = st.state == MoveState.CLIMBING
    swimming = st.state == MoveState.SWIMMING
    lean = 0.55 if gliding else (0.25 if climbing else 0.0)
    phase = st.anim_phase if st.state in (MoveState.GROUNDED, MoveState.CLIMBING) else 0.25
    feet = st.pos.copy()
    if swimming:
        feet = feet + np.array([0.0, -0.35, 0.0], dtype=np.float32)

    _humanoid(batch, feet, st.yaw, obj_id, tunic=pal["tunic"], hair=pal["hair"],
              phase=phase, scale=1.0, lean=lean, arms_out=gliding or swimming)

    if gliding:
        batch.current_material = 5
        # A wind-glider canopy: two panels in a shallow V, with a coloured
        # underside offset below the top skin so the two do not z-fight.
        c, s = np.cos(st.yaw), np.sin(st.yaw)
        right = np.array([c, 0.0, s], dtype=np.float32)
        fwd = np.array([-s, 0.0, c], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        top = feet + up * 2.75
        tip_drop = up * 0.55
        for side in (-1.0, 1.0):
            inner_b = top - fwd * 0.45
            inner_f = top + fwd * 1.45
            outer_b = top + right * (2.15 * side) - fwd * 0.25 - tip_drop
            outer_f = top + right * (1.35 * side) + fwd * 1.35 - tip_drop
            batch.quad(inner_b, outer_b, outer_f, inner_f, pal["glider"], obj_id)
            drop = up * 0.09
            batch.quad(inner_f - drop, outer_f - drop, outer_b - drop, inner_b - drop, pal["cape"], obj_id)
        # Rigging line from the canopy down to the character.
        batch.box(feet + up * 2.1, (0.06, 0.9, 0.06), pal["trim"], st.yaw, obj_id)
    elif not swimming:
        batch.current_material = 5
        # A short cape so the silhouette reads from behind.
        c, s = np.cos(st.yaw), np.sin(st.yaw)
        right = np.array([c, 0.0, s], dtype=np.float32)
        back = np.array([-s, 0.0, c], dtype=np.float32)
        top = feet + np.array([0.0, 1.5, 0.0], dtype=np.float32)
        flap = 0.12 * np.sin(st.anim_phase * 6.28)
        for side in (-1, 1):
            a = top + right * (side * 0.24) + back * 0.12
            b = top + right * (side * 0.03) + back * 0.18
            c0 = top + right * (side * 0.07) + back * (0.30 + flap) - (0, 0.88, 0)
            d = top + right * (side * 0.36) + back * (0.38 + flap) - (0, 0.77, 0)
            batch.quad(a, b, c0, d, pal["cape"], obj_id)
            batch.tube(d, c0, 0.012, (0.88, 0.74, 0.45), obj_id, sides=4)
    batch.current_material = 0
