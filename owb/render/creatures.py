"""Presentation-only creature, rider, den and fruit models for Cycles exports.

Policy observations keep using ``models.entity``; these meshes are added only
to offline presentation scenes. Appearance variation (coat, antlers, size) is
keyed to the creature's stable id, and gait phase to simulated time and
speed, so a creature looks the same from every camera position. Designs are
original: generic goblins, a heavy brute, gel slimes and a skeletal stalker.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..core import blocks as B

SURFACE, STONE, BARK, FOLIAGE, METAL, CLOTH, SKIN = range(7)


def _u(*identity):
    return int.from_bytes(hashlib.sha256(repr(identity).encode()).digest()[:8], "little") / 2.0 ** 64


class _Frame:
    """Local frame: x forward, y up, z right of a creature at ``feet``."""

    def __init__(self, feet, yaw, scale=1.0):
        self.o = np.asarray(feet, float)
        self.f = np.array((np.cos(yaw), 0.0, np.sin(yaw)))
        self.r = np.array((-np.sin(yaw), 0.0, np.cos(yaw)))
        self.yaw, self.s = yaw, scale

    def p(self, x, y, z=0.0):
        return self.o + self.s * (self.f * x + np.array((0.0, y, 0.0)) + self.r * z)


def _blob(batch, fr, x, y, z, radii, color, sid, material, pitch=0.0):
    batch.current_material = material
    radii = np.asarray(radii, float) * fr.s
    if pitch:
        # Approximate a pitched ellipsoid with two overlapping ones.
        d = radii[0] * 0.45
        for k in (-1, 1):
            c = fr.p(x + k * d / fr.s * np.cos(pitch), y + k * d / fr.s * np.sin(pitch), z)
            batch.ellipsoid(c, (radii[0] * 0.62, radii[1], radii[2]), color, fr.yaw, sid, 16, 8)
    else:
        batch.ellipsoid(fr.p(x, y, z), radii, color, fr.yaw, sid, 16, 8)


def _limb(batch, a, b, r0, r1, color, sid, material):
    batch.current_material = material
    batch.tube(a, b, r0, color, sid, r1, 10)


# ---------------------------------------------------------------------------
# Quadrupeds
# ---------------------------------------------------------------------------

SPECIES = {
    #        length, leg, body radii (x, y, z), neck (len, angle), head (len, r), coat, belly, legs
    "horse": dict(L=1.95, leg=0.88, body=(0.98, 0.38, 0.32), neck=(0.78, 0.95), head=(0.56, 0.15),
                  coats=((0.42, 0.27, 0.16), (0.28, 0.18, 0.11), (0.62, 0.48, 0.32), (0.80, 0.78, 0.74)),
                  belly=0.92, legs=(0.18, 0.13, 0.10)),
    "deer": dict(L=1.35, leg=0.78, body=(0.66, 0.27, 0.22), neck=(0.50, 1.05), head=(0.34, 0.10),
                 coats=((0.60, 0.43, 0.26), (0.54, 0.38, 0.23)), belly=1.25, legs=None),
    "boar": dict(L=1.15, leg=0.34, body=(0.60, 0.30, 0.26), neck=(0.18, 0.25), head=(0.38, 0.15),
                 coats=((0.30, 0.22, 0.17), (0.24, 0.19, 0.16)), belly=1.0, legs=(0.16, 0.13, 0.11)),
    "goat": dict(L=1.0, leg=0.52, body=(0.50, 0.26, 0.21), neck=(0.32, 0.95), head=(0.28, 0.10),
                 coats=((0.86, 0.83, 0.76), (0.62, 0.55, 0.47), (0.36, 0.31, 0.27)), belly=1.05, legs=None),
}


def quadruped(batch, e, time, *, saddled=False):
    kind, nid = e["kind"], int(e.get("npc", 0))
    sp = SPECIES[kind]
    size = 0.92 + 0.16 * _u(kind, nid, "size")
    feet = np.asarray(e["pos"], float) - (0, e.get("height", 1.0) / 2, 0)
    fr = _Frame(feet, float(e.get("yaw", 0.0)), size)
    sid = B.SEM_ANIMAL
    coat = np.asarray(sp["coats"][int(_u(kind, nid, "coat") * len(sp["coats"]))])
    belly = np.clip(coat * sp["belly"], 0, 1)
    legc = coat * 0.9 if sp["legs"] is None else np.asarray(sp["legs"])
    L, leg = sp["L"], sp["leg"]
    bx, by, bz = sp["body"]
    speed = float(e.get("speed", 0.0))
    stride = 0.9 + 0.5 * leg
    phase = 2 * np.pi * (time * speed / max(stride, 1e-3)) + nid * 1.7
    swing = np.clip(speed / 3.0, 0, 1) * 0.55
    graze = speed < 0.15 and not e.get("ridden") and _u(kind, nid, int(time // 9)) < 0.5 and kind != "horse"
    back = leg + by * 1.6
    # Barrel, chest and haunch.
    _blob(batch, fr, 0, back - by * 0.2, 0, (bx, by, bz), coat, sid, CLOTH)
    _blob(batch, fr, bx * 0.55, back - by * 0.1, 0, (bx * 0.48, by * 1.05, bz * 0.95), coat, sid, CLOTH)
    _blob(batch, fr, -bx * 0.55, back - by * 0.05, 0, (bx * 0.5, by * 1.0, bz * 1.0), coat, sid, CLOTH)
    _blob(batch, fr, 0, back - by * 0.75, 0, (bx * 0.8, by * 0.35, bz * 0.8), belly, sid, CLOTH)
    # Legs: two segments, diagonal pairs in phase, hooves.
    for i, (lx, side, off) in enumerate(((0.72, -1, 0.0), (0.72, 1, np.pi), (-0.72, -1, np.pi), (-0.72, 1, 0.0))):
        a = np.sin(phase + off) * swing
        hip = fr.p(lx * bx, back - by * 0.55, side * bz * 0.62)
        knee = fr.p(lx * bx + a * leg * 0.45, leg * 0.52, side * bz * 0.6)
        lift = max(0.0, np.sin(phase + off + 1.2)) * swing * 0.18
        hoof = fr.p(lx * bx + a * leg * 0.9, 0.05 + lift, side * bz * 0.6)
        thick = 0.075 if kind in ("horse", "boar") else 0.05
        _limb(batch, hip, knee, thick * 1.5 * size, thick * size, coat if lx > 0 or kind != "horse" else coat * 0.95, sid, CLOTH)
        _limb(batch, knee, hoof, thick * size, thick * 0.75 * size, legc, sid, CLOTH)
        _blob(batch, fr, lx * bx + a * leg * 0.9, 0.04 + lift, side * bz * 0.6, (0.07, 0.05, 0.06), (0.12, 0.1, 0.08), sid, STONE)
    # Neck and head.
    nl, na = sp["neck"]
    if graze:
        na = -0.9
    base = np.array((bx * 0.9, back + by * 0.1))
    tip = base + nl * np.array((np.cos(na) * 0.6 + 0.4, np.sin(na)))
    _limb(batch, fr.p(base[0], base[1]), fr.p(tip[0], tip[1]), by * 0.55 * size, by * 0.38 * size, coat, sid, CLOTH)
    hl, hr = sp["head"]
    hp = -0.6 if not graze else -1.25
    hx, hy = tip[0] + hl * 0.42 * np.cos(hp + 0.6), tip[1] + hl * 0.42 * np.sin(hp + 0.6)
    _blob(batch, fr, tip[0] + 0.05, tip[1], 0, (hr * 1.1, hr * 1.05, hr * 0.95), coat, sid, CLOTH)
    _blob(batch, fr, hx, hy, 0, (hl * 0.5, hr * 0.8, hr * 0.75), coat, sid, CLOTH, pitch=hp)
    muzzle = (hx + hl * 0.32 * np.cos(hp), hy + hl * 0.32 * np.sin(hp))
    _blob(batch, fr, muzzle[0], muzzle[1], 0, (hr * 0.55, hr * 0.6, hr * 0.62), np.clip(coat * 0.6, 0, 1), sid, SKIN)
    for side in (-1, 1):
        _blob(batch, fr, tip[0] + 0.12, tip[1] + hr * 0.25, side * hr * 0.85, (0.025, 0.03, 0.02), (0.03, 0.025, 0.02), sid, SKIN)
        ear = fr.p(tip[0] - 0.02, tip[1] + hr * 0.9, side * hr * 0.6)
        _limb(batch, ear, ear + np.array((0, 0.13 * size, 0)) + fr.r * side * 0.04, 0.035 * size, 0.005, coat, sid, CLOTH)
    # Tail.
    rump = fr.p(-bx * 1.0, back)
    if kind == "horse":
        for k in range(5):
            _limb(batch, rump, fr.p(-bx * 1.15 - 0.05 * k, back - 0.75 + 0.04 * k, (k - 2) * 0.03), 0.05 * size, 0.02,
                  legc, sid, CLOTH)
        # Mane along the neck.
        for k in range(7):
            t = k / 6
            q = base + (tip - base) * t
            _limb(batch, fr.p(q[0], q[1] + by * 0.35), fr.p(q[0] - 0.12, q[1] + by * 0.35 - 0.18, 0.04), 0.035 * size, 0.01,
                  legc, sid, CLOTH)
    elif kind == "deer":
        _blob(batch, fr, -bx * 1.02, back - 0.02, 0, (0.07, 0.1, 0.07), (0.92, 0.9, 0.85), sid, CLOTH)
    else:
        _limb(batch, rump, fr.p(-bx * 1.15, back - 0.25), 0.03 * size, 0.01, coat, sid, CLOTH)
    # Species features.
    crown = (tip[0] + 0.02, tip[1] + hr * 0.95)
    if kind == "deer" and _u("antlers", nid) < 0.55:
        for side in (-1, 1):
            a = fr.p(crown[0], crown[1], side * 0.05)
            b = fr.p(crown[0] - 0.1, crown[1] + 0.36, side * 0.2)
            _limb(batch, a, b, 0.02, 0.012, (0.72, 0.64, 0.5), sid, BARK)
            for k, t in enumerate((0.45, 0.8)):
                q = a + (b - a) * t
                _limb(batch, q, q + np.array((0, 0.18, 0)) + fr.f * 0.1 + fr.r * side * 0.04, 0.012, 0.006, (0.72, 0.64, 0.5), sid, BARK)
    if kind == "goat":
        for side in (-1, 1):
            prev = fr.p(crown[0], crown[1], side * 0.05)
            for k in range(1, 5):
                ang = k * 0.55
                q = fr.p(crown[0] - 0.12 * np.sin(ang), crown[1] + 0.1 * np.cos(ang) + 0.06, side * (0.05 + 0.025 * k))
                _limb(batch, prev, q, 0.03 - 0.005 * k, 0.025 - 0.005 * k, (0.55, 0.5, 0.42), sid, BARK)
                prev = q
        chin = fr.p(muzzle[0] - 0.05, muzzle[1] - 0.1)
        _limb(batch, chin, chin - np.array((0, 0.14, 0)), 0.03, 0.005, coat * 0.85, sid, CLOTH)
    if kind == "boar":
        for side in (-1, 1):
            t0 = fr.p(muzzle[0], muzzle[1] - 0.03, side * 0.06)
            _limb(batch, t0, t0 + fr.f * 0.09 + np.array((0, 0.08, 0)) + fr.r * side * 0.03, 0.02, 0.004, (0.93, 0.9, 0.82), sid, STONE)
        for k in range(8):
            q = fr.p(bx * (0.6 - 0.18 * k), back + by * 0.85)
            _limb(batch, q, q + np.array((0, 0.09, 0)), 0.025, 0.004, (0.12, 0.1, 0.09), sid, CLOTH)
    if kind == "horse" and saddled:
        _blob(batch, fr, 0.05, back + by * 0.72, 0, (0.34, 0.07, bz * 1.05), (0.36, 0.22, 0.12), sid, CLOTH)
        _blob(batch, fr, 0.05, back + by * 0.55, 0, (0.38, 0.1, bz * 1.12), (0.55, 0.16, 0.12), sid, CLOTH)
        for side in (-1, 1):
            s0 = fr.p(0.05, back + by * 0.6, side * bz * 1.05)
            _limb(batch, s0, s0 - np.array((0, 0.55 * size, 0)), 0.01, 0.01, (0.2, 0.14, 0.08), sid, CLOTH)
    batch.current_material = 0
    return dict(seat=fr.p(0.05, back + by * 0.85), frame=fr)


def rider(batch, seat, yaw, color, phase=0.0):
    """The traveler seated astride, legs along the flanks, hands on the reins."""
    fr = _Frame(seat, yaw)
    sid = B.SEM_SELF
    boot, skin, leather, cloth = (0.25, 0.23, 0.19), (0.82, 0.66, 0.46), (0.37, 0.27, 0.17), np.asarray(color)
    bob = 0.03 * np.sin(phase * 2)
    for side in (-1, 1):
        hip = fr.p(0.0, 0.08 + bob, side * 0.14)
        knee = fr.p(0.22, -0.12 + bob, side * 0.36)
        foot = fr.p(0.12, -0.55 + bob, side * 0.38)
        _limb(batch, hip, knee, 0.09, 0.08, (0.63, 0.61, 0.46), sid, CLOTH)
        _limb(batch, knee, foot, 0.08, 0.07, boot, sid, CLOTH)
        _blob(batch, fr, 0.2, -0.58 + bob, side * 0.38, (0.16, 0.07, 0.09), boot, sid, CLOTH)
        sh = fr.p(0.0, 0.52 + bob, side * 0.26)
        el = fr.p(0.25, 0.32 + bob, side * 0.28)
        hand = fr.p(0.45, 0.28 + bob, side * 0.12)
        _limb(batch, sh, el, 0.08, 0.07, cloth, sid, CLOTH)
        _limb(batch, el, hand, 0.065, 0.055, leather, sid, CLOTH)
        _blob(batch, fr, 0.45, 0.28 + bob, side * 0.12, (0.06, 0.06, 0.06), skin, sid, SKIN)
    _blob(batch, fr, 0.0, 0.38 + bob, 0, (0.2, 0.3, 0.24), cloth, sid, CLOTH)
    _blob(batch, fr, -0.02, 0.12 + bob, 0, (0.22, 0.12, 0.25), cloth * 0.9, sid, CLOTH)
    _blob(batch, fr, 0.03, 0.82 + bob, 0, (0.15, 0.19, 0.15), skin, sid, SKIN)
    _blob(batch, fr, 0.0, 0.95 + bob, 0, (0.17, 0.13, 0.17), (0.61, 0.46, 0.23), sid, CLOTH)
    for side in (-1, 1):
        _blob(batch, fr, 0.15, 0.84 + bob, side * 0.065, (0.012, 0.027, 0.023), (0.15, 0.26, 0.27), sid, SKIN)
    # Ochre scarf trailing behind.
    batch.current_material = CLOTH
    batch.quad(fr.p(-0.12, 0.66 + bob, -0.18), fr.p(-0.12, 0.66 + bob, 0.18), fr.p(-0.55, 0.45 + bob, 0.12),
               fr.p(-0.5, 0.48 + bob, -0.15), (0.70, 0.53, 0.27), sid)
    batch.current_material = 0


# ---------------------------------------------------------------------------
# Small wildlife
# ---------------------------------------------------------------------------

def bird(batch, e, time):
    nid = int(e.get("npc", 0))
    feet = np.asarray(e["pos"], float) - (0, e.get("height", 0.5) / 2, 0)
    fr = _Frame(feet, float(e.get("yaw", 0.0)), 1.0)
    sid = B.SEM_ANIMAL
    tone = np.array(((0.85, 0.83, 0.78), (0.35, 0.33, 0.36), (0.55, 0.42, 0.3))[int(_u("bird", nid) * 3)])
    flying = float(e.get("speed", 0)) > 1.0
    _blob(batch, fr, 0, 0.22, 0, (0.14, 0.09, 0.08), tone, sid, CLOTH)
    _blob(batch, fr, 0.12, 0.3, 0, (0.06, 0.06, 0.055), tone, sid, CLOTH)
    batch.current_material = SKIN
    batch.cone(fr.p(0.17, 0.3), 0.018, 0.06, (0.9, 0.62, 0.2), 8, sid, taper=0)
    flap = np.sin(time * 18 + nid) * 0.6 if flying else -0.5
    for side in (-1, 1):
        root = fr.p(0.0, 0.25, side * 0.06)
        tip = root + fr.r * side * 0.28 + np.array((0, 0.28 * flap, 0)) - fr.f * 0.06
        batch.current_material = CLOTH
        batch.add([[root, tip, fr.p(-0.12, 0.24, side * 0.05)]], tone * 0.85, sid)
        batch.add([[root, fr.p(-0.12, 0.24, side * 0.05), tip]], tone * 0.85, sid)
        if not flying:
            _limb(batch, fr.p(0, 0.15, side * 0.03), fr.p(0.01, 0.0, side * 0.03), 0.008, 0.006, (0.8, 0.55, 0.2), sid, SKIN)
    batch.current_material = 0


def fish(batch, e, time):
    nid = int(e.get("npc", 0))
    centre = np.asarray(e["pos"], float)
    fr = _Frame(centre - (0, 0.1, 0), float(e.get("yaw", 0.0)), 0.8 + 0.5 * _u("fish", nid))
    sid = B.SEM_ANIMAL
    tone = np.array(((0.55, 0.62, 0.66), (0.42, 0.5, 0.36), (0.72, 0.55, 0.42))[int(_u("fish-tone", nid) * 3)])
    wiggle = np.sin(time * 9 + nid) * 0.08
    _blob(batch, fr, 0, 0.1, wiggle * 0.3, (0.2, 0.065, 0.045), tone, sid, SKIN)
    _blob(batch, fr, 0.03, 0.08, 0, (0.15, 0.04, 0.04), np.clip(tone * 1.3, 0, 1), sid, SKIN)
    batch.current_material = SKIN
    tail = fr.p(-0.2, 0.1, wiggle)
    batch.add([[tail, fr.p(-0.32, 0.18, wiggle * 2), fr.p(-0.32, 0.02, wiggle * 2)]], tone * 0.8, sid)
    batch.add([[tail, fr.p(-0.32, 0.02, wiggle * 2), fr.p(-0.32, 0.18, wiggle * 2)]], tone * 0.8, sid)
    batch.add([[fr.p(0.02, 0.16), fr.p(-0.1, 0.2), fr.p(-0.1, 0.15)]], tone * 0.8, sid)
    batch.add([[fr.p(0.02, 0.16), fr.p(-0.1, 0.15), fr.p(-0.1, 0.2)]], tone * 0.8, sid)
    batch.current_material = 0


def frog(batch, e, time):
    feet = np.asarray(e["pos"], float) - (0, e.get("height", 0.4) / 2, 0)
    fr = _Frame(feet, float(e.get("yaw", 0.0)), 1.0)
    sid = B.SEM_ANIMAL
    tone = (0.34, 0.52, 0.24)
    _blob(batch, fr, 0, 0.08, 0, (0.11, 0.07, 0.09), tone, sid, SKIN)
    for side in (-1, 1):
        _blob(batch, fr, 0.07, 0.14, side * 0.045, (0.025, 0.025, 0.025), (0.85, 0.75, 0.2), sid, SKIN)
        _blob(batch, fr, -0.06, 0.04, side * 0.09, (0.06, 0.03, 0.035), tone, sid, SKIN)


# ---------------------------------------------------------------------------
# Monsters
# ---------------------------------------------------------------------------

def goblin(batch, e, time, night):
    """Lanky, hunched forest goblin: long ears, tusked jaw, hide wrap, club."""
    nid = int(e.get("npc", 0))
    height = float(e.get("height", 1.6))
    feet = np.asarray(e["pos"], float) - (0, height / 2, 0)
    fr = _Frame(feet, float(e.get("yaw", 0.0)), height / 1.6 * (0.92 + 0.14 * _u("gob", nid)))
    sid = B.SEM_MONSTER
    skin = np.array(((0.55, 0.32, 0.22), (0.45, 0.36, 0.24), (0.36, 0.4, 0.28))[int(_u("gob-skin", nid) * 3)])
    hide, bone = (0.38, 0.29, 0.2), (0.86, 0.82, 0.7)
    speed = float(e.get("speed", 0.0))
    phase = 2 * np.pi * time * speed / 1.1 + nid
    swing = min(1.0, speed / 3) * 0.5
    attack = e.get("state") in ("attack", "chase")
    for side in (-1, 1):
        a = np.sin(phase + (0 if side < 0 else np.pi)) * swing
        hip = fr.p(-0.05, 0.72, side * 0.13)
        knee = fr.p(0.12 + a * 0.2, 0.4, side * 0.16)
        foot = fr.p(a * 0.35, 0.04, side * 0.15)
        _limb(batch, hip, knee, 0.08, 0.06, skin, sid, SKIN)
        _limb(batch, knee, foot, 0.06, 0.045, skin, sid, SKIN)
        _blob(batch, fr, a * 0.35 + 0.08, 0.04, side * 0.15, (0.12, 0.04, 0.06), skin * 0.8, sid, SKIN)
    _blob(batch, fr, -0.02, 0.8, 0, (0.18, 0.16, 0.2), hide, sid, CLOTH)
    _blob(batch, fr, 0.06, 1.05, 0, (0.2, 0.24, 0.21), skin, sid, SKIN, pitch=0.5)
    _blob(batch, fr, 0.04, 1.02, 0, (0.18, 0.08, 0.23), hide, sid, CLOTH)
    head = (0.3, 1.32)
    _blob(batch, fr, head[0], head[1], 0, (0.15, 0.14, 0.14), skin, sid, SKIN)
    _blob(batch, fr, head[0] + 0.1, head[1] - 0.07, 0, (0.09, 0.07, 0.11), skin * 0.95, sid, SKIN)
    eye = (1.0, 0.75, 0.2) if night else (0.95, 0.8, 0.3)
    batch.current_material = SKIN
    for side in (-1, 1):
        batch.ellipsoid(fr.p(head[0] + 0.12, head[1] + 0.03, side * 0.06), (0.025, 0.02, 0.02), eye, fr.yaw, sid, 10, 5)
        ear0 = fr.p(head[0] - 0.02, head[1] + 0.05, side * 0.12)
        _limb(batch, ear0, fr.p(head[0] - 0.12, head[1] + 0.16, side * 0.36), 0.05, 0.006, skin, sid, SKIN)
        tusk = fr.p(head[0] + 0.17, head[1] - 0.1, side * 0.05)
        _limb(batch, tusk, tusk + np.array((0, 0.07, 0)) + fr.f * 0.02, 0.014, 0.003, bone, sid, STONE)
    if night:
        for side in (-1, 1):
            batch.current_material = SKIN
            batch.ellipsoid(fr.p(head[0] + 0.125, head[1] + 0.03, side * 0.06), (0.018, 0.014, 0.014), (1.0, 0.7, 0.2), fr.yaw, sid, 8, 4)
            batch.emissive[-1][:] = 1.4
    # Arms: long, one carrying a knotted club raised when chasing.
    for side in (-1, 1):
        sh = fr.p(0.12, 1.15, side * 0.22)
        el = fr.p(0.2, 0.85, side * 0.3)
        hand = fr.p(0.3, 0.6, side * 0.28)
        if side > 0 and attack:
            el, hand = fr.p(0.15, 1.35, side * 0.3), fr.p(0.1, 1.6, side * 0.25)
        _limb(batch, sh, el, 0.06, 0.05, skin, sid, SKIN)
        _limb(batch, el, hand, 0.05, 0.04, skin, sid, SKIN)
        if side > 0:
            top = hand + (hand - el) / max(np.linalg.norm(hand - el), 1e-6) * 0.65
            _limb(batch, hand, top, 0.035, 0.07, (0.4, 0.28, 0.16), sid, BARK)
            for k in range(3):
                q = hand + (top - hand) * (0.6 + 0.15 * k)
                _limb(batch, q, q + fr.r * 0.07 + fr.f * 0.03 * (k - 1), 0.012, 0.002, bone, sid, STONE)
    # Bone necklace.
    for k in range(5):
        ang = (k - 2) * 0.35
        q = fr.p(0.2 + 0.06 * np.cos(ang), 1.15, 0.16 * np.sin(ang))
        _blob(batch, fr, 0.2 + 0.06 * np.cos(ang), 1.12, 0.16 * np.sin(ang), (0.025, 0.04, 0.02), bone, sid, STONE)
    batch.current_material = 0


def brute(batch, e, time, night):
    """Heavy-shouldered brute with a bone-crested helm and a long spear."""
    nid = int(e.get("npc", 0))
    height = float(e.get("height", 2.4))
    feet = np.asarray(e["pos"], float) - (0, height / 2, 0)
    fr = _Frame(feet, float(e.get("yaw", 0.0)), height / 2.4)
    sid = B.SEM_MONSTER
    skin = np.array(((0.42, 0.38, 0.33), (0.34, 0.3, 0.36))[int(_u("brute", nid) * 2)])
    fur, bone, iron = (0.26, 0.2, 0.15), (0.84, 0.8, 0.68), (0.42, 0.42, 0.44)
    speed = float(e.get("speed", 0.0))
    phase = 2 * np.pi * time * speed / 1.6 + nid
    swing = min(1.0, speed / 3) * 0.4
    for side in (-1, 1):
        a = np.sin(phase + (0 if side < 0 else np.pi)) * swing
        hip = fr.p(0, 1.0, side * 0.22)
        knee = fr.p(0.1 + a * 0.25, 0.55, side * 0.26)
        foot = fr.p(a * 0.45, 0.05, side * 0.26)
        _limb(batch, hip, knee, 0.15, 0.12, skin, sid, SKIN)
        _limb(batch, knee, foot, 0.12, 0.1, skin, sid, SKIN)
        _blob(batch, fr, a * 0.45 + 0.1, 0.06, side * 0.26, (0.18, 0.07, 0.1), skin * 0.8, sid, SKIN)
    _blob(batch, fr, 0, 1.1, 0, (0.3, 0.22, 0.32), fur, sid, CLOTH)
    _blob(batch, fr, 0.05, 1.55, 0, (0.34, 0.38, 0.42), skin, sid, SKIN)
    _blob(batch, fr, 0.02, 1.85, 0, (0.3, 0.16, 0.5), fur, sid, CLOTH)
    _blob(batch, fr, 0.2, 2.05, 0, (0.17, 0.16, 0.16), skin, sid, SKIN)
    _blob(batch, fr, 0.18, 2.14, 0, (0.19, 0.1, 0.18), iron, sid, METAL)
    for side in (-1, 1):
        crest = fr.p(0.14, 2.18, side * 0.14)
        _limb(batch, crest, crest + np.array((0, 0.25, 0)) - fr.f * 0.12 + fr.r * side * 0.12, 0.04, 0.004, bone, sid, STONE)
        batch.current_material = SKIN
        batch.ellipsoid(fr.p(0.35, 2.06, side * 0.07), (0.022, 0.018, 0.018), (1, 0.55, 0.2) if night else (0.9, 0.8, 0.4), fr.yaw, sid, 8, 4)
        if night:
            batch.emissive[-1][:] = 1.4
        sh = fr.p(0.05, 1.8, side * 0.44)
        el = fr.p(0.2, 1.35, side * 0.5)
        hand = fr.p(0.42, 1.15, side * 0.4)
        _limb(batch, sh, el, 0.12, 0.1, skin, sid, SKIN)
        _limb(batch, el, hand, 0.1, 0.08, skin, sid, SKIN)
        _blob(batch, fr, 0.05, 1.82, side * 0.44, (0.16, 0.12, 0.16), iron, sid, METAL)
    base, tip = fr.p(-0.6, 0.6, 0.4), fr.p(1.2, 2.2, 0.4)
    _limb(batch, base, tip, 0.035, 0.03, (0.42, 0.3, 0.17), sid, BARK)
    batch.current_material = METAL
    axis = (tip - base) / np.linalg.norm(tip - base)
    batch.tube(tip, tip + axis * 0.32, 0.06, iron, sid, 0.005, 8)
    batch.current_material = 0


def slime(batch, e, time, night):
    """Gel slime: a wobbling translucent-looking dome with a darker core."""
    nid = int(e.get("npc", 0))
    feet = np.asarray(e["pos"], float) - (0, e.get("height", 1.1) / 2, 0)
    col = np.asarray(e.get("color", (126, 196, 150)), float) / 255
    fr = _Frame(feet, float(e.get("yaw", 0.0)), 1.0)
    sid = B.SEM_MONSTER
    squash = 1 + 0.12 * np.sin(time * 5 + nid)
    _blob(batch, fr, 0, 0.38 / squash, 0, (0.5 * squash ** .5, 0.38 / squash, 0.5 * squash ** .5), col, sid, SKIN)
    _blob(batch, fr, 0, 0.62 / squash, 0, (0.3, 0.2 / squash, 0.3), np.clip(col * 1.2, 0, 1), sid, SKIN)
    _blob(batch, fr, 0.05, 0.32, 0, (0.18, 0.14, 0.18), col * 0.55, sid, SKIN)
    for side in (-1, 1):
        batch.current_material = SKIN
        batch.ellipsoid(fr.p(0.42, 0.5 / squash, side * 0.13), (0.03, 0.06, 0.05), (0.06, 0.06, 0.07), fr.yaw, sid, 10, 5)
        batch.ellipsoid(fr.p(0.445, 0.53 / squash, side * 0.12), (0.012, 0.02, 0.015), (1, 1, 1), fr.yaw, sid, 6, 3)
        if night:
            batch.emissive[-1][:] = 1.0
    batch.current_material = 0


def stalker(batch, e, time, night):
    """Skeletal night stalker: bone limbs, ribcage, glowing sockets."""
    nid = int(e.get("npc", 0))
    height = float(e.get("height", 1.6))
    feet = np.asarray(e["pos"], float) - (0, height / 2, 0)
    fr = _Frame(feet, float(e.get("yaw", 0.0)), height / 1.6)
    sid = B.SEM_MONSTER
    bone, dark = (0.84, 0.81, 0.72), (0.2, 0.18, 0.17)
    speed = float(e.get("speed", 0.0))
    phase = 2 * np.pi * time * speed / 1.2 + nid
    swing = min(1.0, speed / 3) * 0.5
    for side in (-1, 1):
        a = np.sin(phase + (0 if side < 0 else np.pi)) * swing
        hip, knee, foot = fr.p(0, 0.85, side * 0.1), fr.p(0.1 + a * 0.2, 0.45, side * 0.12), fr.p(a * 0.35, 0.03, side * 0.12)
        _limb(batch, hip, knee, 0.03, 0.025, bone, sid, STONE)
        _limb(batch, knee, foot, 0.025, 0.02, bone, sid, STONE)
        _blob(batch, fr, 0.1 + a * 0.2, 0.45, side * 0.12, (0.04, 0.04, 0.04), bone, sid, STONE)
    _blob(batch, fr, 0, 0.88, 0, (0.08, 0.06, 0.13), bone, sid, STONE)
    spine0, spine1 = fr.p(0, 0.88), fr.p(0.06, 1.38)
    _limb(batch, spine0, spine1, 0.025, 0.022, bone, sid, STONE)
    for k in range(5):
        y = 1.0 + 0.08 * k
        for side in (-1, 1):
            _limb(batch, fr.p(0.03, y), fr.p(0.1, y - 0.04, side * 0.14), 0.012, 0.01, bone, sid, STONE)
            _limb(batch, fr.p(0.1, y - 0.04, side * 0.14), fr.p(0.16, y - 0.06, side * 0.03), 0.01, 0.008, bone, sid, STONE)
    _blob(batch, fr, 0.08, 1.55, 0, (0.12, 0.13, 0.11), bone, sid, STONE)
    _blob(batch, fr, 0.15, 1.47, 0, (0.06, 0.04, 0.08), bone, sid, STONE)
    for side in (-1, 1):
        batch.current_material = SKIN
        batch.ellipsoid(fr.p(0.17, 1.57, side * 0.045), (0.022, 0.025, 0.022), dark, fr.yaw, sid, 8, 4)
        batch.ellipsoid(fr.p(0.185, 1.57, side * 0.045), (0.012, 0.012, 0.012), (0.55, 0.95, 1.0), fr.yaw, sid, 6, 3)
        batch.emissive[-1][:] = 2.0 if night else 0.6
        sh, el, hand = fr.p(0.05, 1.36, side * 0.17), fr.p(0.15, 1.1, side * 0.22), fr.p(0.32, 0.95, side * 0.2)
        _limb(batch, sh, el, 0.022, 0.02, bone, sid, STONE)
        _limb(batch, el, hand, 0.02, 0.016, bone, sid, STONE)
    batch.current_material = 0


# ---------------------------------------------------------------------------
# Dens and fruit
# ---------------------------------------------------------------------------

def den(batch, pos, identity):
    """Earth-and-rock burrow mound with a dark mouth, facing a stable direction."""
    fr = _Frame(np.asarray(pos, float) - (0, 1.0, 0), 2 * np.pi * _u("den", *identity), 1.0)
    sid = B.SEM_ROCK
    soil, rock = (0.36, 0.3, 0.22), (0.45, 0.43, 0.39)
    _blob(batch, fr, -0.6, 0.35, 0, (1.6, 0.8, 1.4), soil, sid, STONE)
    for k in range(7):
        a = 2 * np.pi * _u("den-rock", *identity, k)
        r = 1.2 + 0.6 * _u("den-r", *identity, k)
        _blob(batch, fr, -0.6 + r * np.cos(a), 0.25, r * np.sin(a),
              (0.35 + 0.3 * _u("den-s", *identity, k), 0.25 + 0.2 * _u("den-h", *identity, k), 0.3 + 0.2 * _u("den-w", *identity, k)),
              rock, sid, STONE)
    _blob(batch, fr, 0.75, 0.4, 0, (0.18, 0.42, 0.55), (0.03, 0.025, 0.02), sid, STONE)
    for side in (-1, 1):
        q = fr.p(0.85, 0.0, side * 0.6)
        _limb(batch, q, q + np.array((0, 1.1, 0)) + fr.r * -side * 0.2, 0.06, 0.04, (0.32, 0.24, 0.15), sid, BARK)
    _blob(batch, fr, 1.0, 1.15, 0, (0.08, 0.1, 0.07), (0.85, 0.82, 0.72), sid, STONE)
    batch.current_material = 0


def apples_on_tree(batch, tree, count):
    x, y, z = tree["trunk"]
    crown = tree["crown"]
    for k in range(int(count)):
        a = 2 * np.pi * _u("apple-hang", tree["id"], k)
        r = 1.0 + 1.0 * _u("apple-r", tree["id"], k)
        h = crown + 0.4 + 1.6 * _u("apple-h", tree["id"], k)
        p = (x + r * np.cos(a), h, z + r * np.sin(a))
        batch.current_material = SKIN
        batch.ellipsoid(p, (0.09, 0.085, 0.09), (0.72, 0.1, 0.08), 0.0, B.SEM_PLANT, 12, 6)
    batch.current_material = 0


def apple(batch, pos):
    p = np.asarray(pos, float)
    batch.current_material = SKIN
    batch.ellipsoid(p + (0, 0.08, 0), (0.09, 0.085, 0.09), (0.72, 0.1, 0.08), 0.0, B.SEM_ITEM, 12, 6)
    batch.current_material = BARK
    batch.tube(p + (0, 0.15, 0), p + (0.01, 0.21, 0), 0.008, (0.3, 0.22, 0.12), B.SEM_ITEM, 0.006, 6)
    batch.current_material = FOLIAGE
    batch.add([[p + (0.01, 0.2, 0), p + (0.08, 0.23, 0.02), p + (0.04, 0.18, 0.03)]], (0.3, 0.5, 0.2), B.SEM_ITEM)
    batch.current_material = 0


DRAW = {"horse": quadruped, "deer": quadruped, "boar": quadruped, "goat": quadruped}
MONSTERS = {"bokoblin": goblin, "moblin": brute, "stalker": stalker,
            "chuchu": slime, "fire_chuchu": slime, "ice_chuchu": slime, "electro_chuchu": slime}


def emit(batch, entities, time, *, night=False, rider_info=None):
    """Draw creatures; returns the rider seat if the agent is mounted."""
    seat = None
    for e in entities:
        kind = e.get("kind")
        if kind in DRAW:
            out = quadruped(batch, e, time, saddled=kind == "horse")
            if e.get("ridden"):
                seat = (out["seat"], float(e.get("yaw", 0.0)))
        elif kind == "bird":
            bird(batch, e, time)
        elif kind == "fish":
            fish(batch, e, time)
        elif kind == "frog":
            frog(batch, e, time)
        elif kind in MONSTERS:
            MONSTERS[kind](batch, e, time, night)
    return seat
