"""Low-poly mesh building for the character, entities and near props.

Meshes are plain triangle soups: `verts` is (T, 3, 3) and `colors` is (T, 3).
Flat per-triangle normals are what give the cel-shaded faceted look, so no
vertex normals are kept.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import math3d as mth
from ..character import CharacterState, MoveState
from .style import CHARACTER_PALETTE, ENTITY_ALBEDO, ENTITY_EMISSIVE, PROP_ALBEDO

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

    def add(self, tris: np.ndarray, color, obj_id: int = 1, emissive: float = 0.0) -> None:
        tris = np.asarray(tris, dtype=np.float32).reshape(-1, 3, 3)
        if len(tris) == 0:
            return
        self.verts.append(tris)
        self.colors.append(np.tile(np.asarray(color, dtype=np.float32), (len(tris), 1)))
        self.emissive.append(np.full(len(tris), float(emissive), dtype=np.float32))
        self.ids.append(np.full(len(tris), int(obj_id), dtype=np.int32))

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

    def finish(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.verts:
            z3 = np.zeros((0, 3, 3), dtype=np.float32)
            return z3, np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32)
        return (np.concatenate(self.verts), np.concatenate(self.colors),
                np.concatenate(self.emissive), np.concatenate(self.ids))


# --------------------------------------------------------------------- props
def add_prop(batch: MeshBatch, kind_name: str, pos: np.ndarray, yaw: float, scale: float, obj_id: int = 1) -> None:
    """A low-poly stand-in for one scattered prop."""
    trunk = PROP_ALBEDO["trunk"]
    p = np.asarray(pos, dtype=np.float32)
    s = float(scale)
    if kind_name == "pine":
        batch.box(p + (0.0, 1.1 * s, 0.0), (0.42 * s, 2.2 * s, 0.42 * s), trunk, yaw, obj_id)
        for k, (h, r) in enumerate(((1.8, 2.3), (4.0, 1.7), (6.0, 1.0))):
            batch.cone(p + (0.0, h * s, 0.0), r * s, 3.6 * s, PROP_ALBEDO["pine"], 7, obj_id)
    elif kind_name == "broadleaf":
        batch.box(p + (0.0, 1.5 * s, 0.0), (0.5 * s, 3.0 * s, 0.5 * s), trunk, yaw, obj_id)
        batch.cone(p + (0.0, 2.9 * s, 0.0), 2.6 * s, 2.4 * s, PROP_ALBEDO["broadleaf"], 8, obj_id, taper=0.55)
        batch.cone(p + (0.0, 4.6 * s, 0.0), 1.7 * s, 1.6 * s, PROP_ALBEDO["broadleaf"], 7, obj_id)
    elif kind_name == "sakura":
        batch.box(p + (0.0, 1.3 * s, 0.0), (0.45 * s, 2.6 * s, 0.45 * s), trunk, yaw, obj_id)
        batch.cone(p + (0.0, 2.4 * s, 0.0), 2.4 * s, 1.9 * s, PROP_ALBEDO["sakura"], 9, obj_id, taper=0.7)
        batch.cone(p + (0.0, 3.8 * s, 0.0), 1.8 * s, 1.5 * s, PROP_ALBEDO["sakura"], 8, obj_id)
    elif kind_name == "bamboo":
        for k in range(3):
            off = np.array([np.cos(yaw + k * 2.1) * 0.35 * s, 0.0, np.sin(yaw + k * 2.1) * 0.35 * s], dtype=np.float32)
            batch.box(p + off + (0.0, 2.6 * s, 0.0), (0.16 * s, 5.2 * s, 0.16 * s), PROP_ALBEDO["bamboo"], yaw, obj_id)
    elif kind_name in ("rock", "boulder"):
        r = (0.55 if kind_name == "rock" else 1.5) * s
        batch.cone(p, r, r * 1.5, PROP_ALBEDO[kind_name], 6, obj_id, taper=0.35)
    elif kind_name == "bush":
        batch.cone(p, 0.75 * s, 0.9 * s, PROP_ALBEDO["bush"], 6, obj_id, taper=0.6)
    elif kind_name == "grass":
        batch.cone(p, 0.3 * s, 0.55 * s, PROP_ALBEDO["grass"], 4, obj_id)
    elif kind_name == "crystal":
        batch.cone(p, 0.45 * s, 2.4 * s, PROP_ALBEDO["crystal"], 5, obj_id, emissive=0.45)


# ------------------------------------------------------------------ entities
def add_entity(batch: MeshBatch, kind_name: str, pos: np.ndarray, yaw: float, used: bool,
               obj_id: int, bob: float = 0.0) -> None:
    p = np.asarray(pos, dtype=np.float32)
    base = ENTITY_ALBEDO.get(kind_name, (0.8, 0.8, 0.8))
    lit = ENTITY_EMISSIVE.get(kind_name)
    glow = 0.0
    color = base
    if used and lit is not None:
        color, glow = lit, 0.8

    if kind_name == "npc":
        _humanoid(batch, p, yaw, obj_id, tunic=base, hair=(0.25, 0.20, 0.24), phase=0.0, scale=1.0)
    elif kind_name == "chest":
        batch.box(p + (0.0, 0.35, 0.0), (1.1, 0.7, 0.8), (0.46, 0.33, 0.21), yaw, obj_id)
        lid_y = 0.78 if not used else 1.0
        batch.box(p + (0.0, lid_y, 0.0 if not used else -0.3), (1.1, 0.26, 0.8), base, yaw, obj_id, 0.15 if used else 0.0)
    elif kind_name == "beacon":
        batch.cone(p, 0.9, 3.2, (0.42, 0.40, 0.46), 6, obj_id, taper=0.45)
        flame_h = 1.1 + 0.18 * np.sin(bob * 4.0)
        batch.cone(p + (0.0, 3.1, 0.0), 0.55, flame_h, color, 6, obj_id, emissive=1.0 if used else 0.1)
    elif kind_name == "pillar":
        batch.box(p + (0.0, 2.0, 0.0), (0.9, 4.0, 0.9), (0.60, 0.58, 0.64), yaw, obj_id)
        batch.cone(p + (0.0, 4.1, 0.0), 0.5, 0.9, color, 5, obj_id, emissive=1.0 if used else 0.1)
    elif kind_name == "collectible":
        h = 0.3 + 0.05 * np.sin(bob * 2.0)
        batch.cone(p + (0.0, h, 0.0), 0.22, 0.5, (0.75, 0.9, 1.0), 5, obj_id, emissive=0.6)
        batch.box(p + (0.0, 0.15, 0.0), (0.08, 0.3, 0.08), (0.35, 0.55, 0.35), yaw, obj_id)
    elif kind_name in ("enemy", "boss"):
        scale = 1.0 if kind_name == "enemy" else 1.7
        _humanoid(batch, p, yaw, obj_id, tunic=base, hair=(0.18, 0.12, 0.2), phase=bob * 1.3, scale=scale)
        batch.cone(p + (0.0, 1.9 * scale, 0.0), 0.3 * scale, 0.7 * scale,
                   (0.95, 0.75, 0.35) if kind_name == "boss" else (0.7, 0.6, 0.9), 5, obj_id, emissive=0.5)
    elif kind_name == "altar":
        batch.box(p + (0.0, 0.5, 0.0), (2.4, 1.0, 2.4), (0.72, 0.70, 0.62), yaw, obj_id)
        batch.cone(p + (0.0, 1.0, 0.0), 0.8, 1.4, color, 6, obj_id, emissive=glow)
    elif kind_name == "waypoint":
        batch.box(p + (0.0, 0.3, 0.0), (2.0, 0.6, 2.0), (0.55, 0.60, 0.62), yaw, obj_id)
        batch.cone(p + (0.0, 0.6, 0.0), 0.7, 3.4, base, 6, obj_id, taper=0.2, emissive=0.35 + glow * 0.5)
        batch.cone(p + (0.0, 4.0, 0.0), 0.45, 0.8, (0.9, 1.0, 0.95), 5, obj_id, emissive=0.9)


# ----------------------------------------------------------------- character
def _humanoid(batch: MeshBatch, feet: np.ndarray, yaw: float, obj_id: int,
              tunic, hair, phase: float, scale: float = 1.0, lean: float = 0.0,
              arms_out: bool = False) -> None:
    """A deliberately simple chibi-proportioned figure: head, torso, 4 limbs."""
    pal = CHARACTER_PALETTE
    s = scale
    c, sn = np.cos(yaw), np.sin(yaw)

    def local(x, y, z):
        """Body-local (right, up, forward) -> world, with a forward lean."""
        y2 = y * np.cos(lean) - z * np.sin(lean)
        z2 = y * np.sin(lean) + z * np.cos(lean)
        return np.asarray(feet, dtype=np.float32) + np.array(
            [x * c - z2 * sn, y2, x * sn + z2 * c], dtype=np.float32) * np.float32(s)

    swing = np.sin(phase * 2.0 * np.pi) * 0.42
    lift = abs(np.cos(phase * 2.0 * np.pi)) * 0.1
    # Legs
    batch.box(local(-0.17, 0.42 + lift, swing * 0.35), (0.19 * s, 0.84 * s, 0.22 * s), pal["boots"], yaw, obj_id)
    batch.box(local(0.17, 0.42 + lift, -swing * 0.35), (0.19 * s, 0.84 * s, 0.22 * s), pal["boots"], yaw, obj_id)
    # Torso + trim
    batch.box(local(0.0, 1.16, 0.0), (0.52 * s, 0.76 * s, 0.33 * s), tunic, yaw, obj_id)
    batch.box(local(0.0, 0.84, 0.0), (0.54 * s, 0.14 * s, 0.35 * s), pal["trim"], yaw, obj_id)
    # Arms
    # `arms_out` only nudges the arms clear of the torso; a large offset
    # detaches them from the shoulders.
    ax = 0.08 if arms_out else 0.0
    arm_y = 1.2 + (0.22 if arms_out else 0.0)
    batch.box(local(-0.36 - ax, arm_y, -swing * 0.4), (0.15 * s, 0.68 * s, 0.17 * s), tunic, yaw, obj_id)
    batch.box(local(0.36 + ax, arm_y, swing * 0.4), (0.15 * s, 0.68 * s, 0.17 * s), tunic, yaw, obj_id)
    # Head + hair
    batch.box(local(0.0, 1.69, 0.02), (0.36 * s, 0.38 * s, 0.34 * s), pal["skin"], yaw, obj_id)
    batch.box(local(0.0, 1.86, -0.02), (0.42 * s, 0.22 * s, 0.4 * s), hair, yaw, obj_id)
    batch.box(local(0.0, 1.72, -0.2), (0.38 * s, 0.42 * s, 0.1 * s), hair, yaw, obj_id)


def add_character(batch: MeshBatch, st: CharacterState, obj_id: int = 2, spec=None) -> None:
    """Build the player avatar, posed for its current movement state."""
    pal = dict(CHARACTER_PALETTE)
    if spec is not None:
        pal.update(tunic=spec.tunic, hair=spec.hair)
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
        # A short cape so the silhouette reads from behind.
        c, s = np.cos(st.yaw), np.sin(st.yaw)
        right = np.array([c, 0.0, s], dtype=np.float32)
        back = np.array([s, 0.0, -c], dtype=np.float32)
        top = feet + np.array([0.0, 1.5, 0.0], dtype=np.float32)
        flap = 0.12 * np.sin(st.anim_phase * 6.28)
        batch.quad(top - right * 0.3 + back * 0.18, top + right * 0.3 + back * 0.18,
                   top + right * 0.34 + back * (0.42 + flap) - np.array([0.0, 0.8, 0.0], dtype=np.float32),
                   top - right * 0.34 + back * (0.42 + flap) - np.array([0.0, 0.8, 0.0], dtype=np.float32),
                   pal["cape"], obj_id)
