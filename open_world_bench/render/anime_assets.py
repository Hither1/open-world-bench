"""Stylised presentation assets: rounded trees and an articulated hero.

These are drawn only in presentation renders that select the anime style. They
read simulation state (position, heading, velocity, movement state, the gait
phase) and never write it, so the same action trace always yields the same
frames. Collision is unchanged: a stylised tree stands on the same trunk.
"""
from functools import lru_cache
import zlib

import numpy as np

from ..character import MoveState
from .assets import BARK, CLOTH, FOLIAGE, METAL, SKIN, _pack
from .meshes import MeshBatch

UP = np.array((0.0, 1.0, 0.0))


# ----------------------------------------------------------------- primitives
def limb(batch, a, b, r0, r1, color, sides=10, obj_id=1):
    """Tapered limb with smooth radial normals and rounded ends."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return
    axis /= length
    helper = np.array((1.0, 0, 0)) if abs(axis[1]) > 0.9 else UP
    right = np.cross(axis, helper); right /= np.linalg.norm(right)
    up = np.cross(axis, right)
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    radial = np.cos(angles)[:, None] * right + np.sin(angles)[:, None] * up
    lo, hi = a + radial * r0, b + radial * r1
    cap_a, cap_b = a - axis * r0 * .6, b + axis * r1 * .6
    triangles, normals = [], []
    for i in range(sides):
        j = (i + 1) % sides
        triangles += [[lo[i], hi[i], hi[j]], [lo[i], hi[j], lo[j]], [cap_a, lo[j], lo[i]], [cap_b, hi[i], hi[j]]]
        normals += [[radial[i], radial[i], radial[j]], [radial[i], radial[j], radial[j]],
                    [-axis, radial[j], radial[i]], [axis, radial[i], radial[j]]]
    batch.add(np.asarray(triangles), color, obj_id, vertex_normals=np.asarray(normals))


def blob(batch, centre, radii, color, yaw=0.0, obj_id=1, sides=12, rings=6, normal_from=None, soften=0.0):
    """Ellipsoid; ``normal_from`` bends its normals toward a shared centre, so a cluster shades as one mass."""
    batch.ellipsoid(centre, radii, color, yaw, obj_id, sides=sides, rings=rings)
    if normal_from is not None and soften > 0.0:
        triangles = batch.verts[-1]
        shared = triangles - np.asarray(normal_from, np.float32)
        shared /= np.maximum(np.linalg.norm(shared, axis=-1, keepdims=True), 1e-6)
        mixed = batch.normals[-1] * (1.0 - soften) + shared * soften
        batch.normals[-1] = (mixed / np.maximum(np.linalg.norm(mixed, axis=-1, keepdims=True), 1e-6)).astype(np.float32)


def panel(batch, a, b, c, d, color, away_from, obj_id=1):
    """Cloth quad with one authored normal facing away from ``away_from``, so both triangles shade alike."""
    quad = np.array([np.asarray(a, float), np.asarray(b, float), np.asarray(c, float), np.asarray(d, float)])
    normal = np.cross(quad[1] - quad[0], quad[2] - quad[0])
    normal /= max(np.linalg.norm(normal), 1e-8)
    if normal @ (quad.mean(axis=0) - np.asarray(away_from, float)) < 0:
        normal = -normal
    batch.add(np.array([[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]), color, obj_id,
              vertex_normals=np.broadcast_to(normal, (2, 3, 3)))


def two_bone(root, target, upper, lower, pole):
    """Middle joint of a two-segment limb from ``root`` to ``target``, bent toward ``pole``."""
    root, target = np.asarray(root, float), np.asarray(target, float)
    reach = target - root
    distance = float(np.clip(np.linalg.norm(reach), 1e-4, upper + lower - 1e-4))
    direction = reach / max(np.linalg.norm(reach), 1e-6)
    target = root + direction * distance
    along = (upper ** 2 - lower ** 2 + distance ** 2) / (2 * distance)
    height = np.sqrt(max(upper ** 2 - along ** 2, 0.0))
    pole = np.asarray(pole, float)
    side = pole - direction * float(pole @ direction)
    side /= max(np.linalg.norm(side), 1e-6)
    return root + direction * along + side * height, target


# ---------------------------------------------------------------------- trees
_LEAF = {
    "broadleaf": ((.20, .46, .17), (.40, .66, .22), (.62, .80, .30)),
    "sakura": ((.80, .42, .58), (.95, .62, .74), (1.0, .82, .88)),
    "pine": ((.10, .32, .26), (.17, .46, .32), (.34, .62, .38)),
    "bamboo": ((.30, .52, .18), (.48, .70, .24), (.70, .84, .36)),
    "bush": ((.20, .44, .18), (.36, .62, .22), (.58, .78, .30)),
}
_TRUNK = {"broadleaf": (.42, .28, .18), "sakura": (.36, .24, .22), "pine": (.38, .24, .16), "bamboo": (.52, .62, .26),
          "bush": (.3, .22, .15)}
STYLISED_KINDS = tuple(_LEAF)


@lru_cache(maxsize=256)
def stylised_plant(kind, variant, detail=1):
    """Rounded, clumped foliage in three tones: shade below, light on top.

    Crown blobs share normals bent toward the crown's centre, so the toon light falls
    across the whole crown as on a painted tree rather than on every leaf."""
    rng = np.random.default_rng(zlib.crc32(f"{kind}|{variant}".encode()))
    dark, mid, light = (np.asarray(c) for c in _LEAF[kind])
    trunk = np.asarray(_TRUNK[kind])
    sides, rings = (12, 6) if detail else (7, 4)
    batch = MeshBatch()
    if kind in ("broadleaf", "sakura"):
        height = rng.uniform(5.6, 7.6) if kind == "broadleaf" else rng.uniform(4.6, 6.2)
        fork = height * rng.uniform(.4, .5)
        lean = rng.uniform(-.35, .35, 2)
        top = np.array((lean[0], fork, lean[1]))
        batch.current_material = BARK
        limb(batch, (0, -.2, 0), top * (.5, .55, .5), .30, .22, trunk, sides=8)
        limb(batch, top * (.5, .55, .5), top, .22, .16, trunk * 1.06, sides=8)
        batch.cone((0, -.25, 0), .46, .5, trunk * .9, sides=8, taper=.6)
        crown = top + (0, height * .3, 0)
        spread = height * rng.uniform(.36, .44)
        batch.current_material = BARK
        branches = int(rng.integers(3, 5))
        for k in range(branches):
            a = 2 * np.pi * k / branches + rng.uniform(-.4, .4)
            tip = crown + (np.cos(a) * spread * .7, rng.uniform(-.25, .1) * height * .3, np.sin(a) * spread * .7)
            limb(batch, top, tip, .12, .05, trunk * 1.04, sides=6)
        batch.current_material = FOLIAGE
        count = int(rng.integers(7, 10)) if detail else 5
        for k in range(count):
            a = 2 * np.pi * k / count + rng.uniform(-.3, .3)
            ring = 0.0 if k == 0 else spread * rng.uniform(.5, .8)
            lift = (rng.uniform(.35, .6) if k == 0 else rng.uniform(-.25, .3)) * spread
            centre = crown + (np.cos(a) * ring, lift, np.sin(a) * ring)
            size = spread * (rng.uniform(.62, .78) if k == 0 else rng.uniform(.42, .6))
            tone = light if lift > spread * .28 else (mid if lift > -spread * .08 else dark)
            blob(batch, centre, (size, size * rng.uniform(.72, .86), size), tone * rng.uniform(.94, 1.06), sides=sides,
                 rings=rings, normal_from=crown - (0, spread * .5, 0), soften=.7)
    elif kind == "pine":
        height = rng.uniform(8.0, 11.0)
        batch.current_material = BARK
        limb(batch, (0, -.2, 0), (0, height * .78, 0), .27, .07, trunk, sides=8)
        batch.cone((0, -.25, 0), .4, .5, trunk * .9, sides=8, taper=.6)
        batch.current_material = FOLIAGE
        tiers = 6 if detail else 4
        for k in range(tiers):
            t = k / (tiers - 1)
            base = height * (.2 + .62 * t)
            radius = height * (.27 - .2 * t) * rng.uniform(.92, 1.08)
            tone = dark + (light - dark) * (.15 + .75 * t)
            # A soft skirt: a low dome over a short cone, so each tier reads as a rounded mass.
            batch.cone((rng.uniform(-.06, .06), base, rng.uniform(-.06, .06)), radius, height * .23, tone * rng.uniform(.95, 1.05),
                       sides=10 if detail else 7)
            blob(batch, (0, base + height * .02, 0), (radius * .92, height * .045, radius * .92), tone * .86, sides=sides,
                 rings=3, normal_from=(0, base - 1.0, 0), soften=.5)
    elif kind == "bamboo":
        batch.current_material = BARK
        for k in range(6 if detail else 4):
            a = 2 * np.pi * k / 6 + rng.uniform(-.3, .3)
            x, z = .34 * np.cos(a), .34 * np.sin(a)
            height = rng.uniform(4.6, 6.4)
            sway = rng.uniform(-.7, .7, 2)
            tip = np.array((x + sway[0], height, z + sway[1]))
            batch.current_material = BARK
            limb(batch, (x, 0, z), tip * (1, .55, 1) + (0, 0, 0), .055, .045, trunk, sides=6)
            limb(batch, tip * (1, .55, 1), tip, .045, .02, trunk * 1.08, sides=6)
            batch.current_material = FOLIAGE
            for leaf in range(3):
                at = tip * (1, .62 + .16 * leaf, 1)
                blob(batch, at + (rng.uniform(-.2, .2), 0, rng.uniform(-.2, .2)), (.5, .34, .5),
                     (mid if leaf < 2 else light) * rng.uniform(.94, 1.06), sides=8, rings=4, normal_from=at - (0, .5, 0), soften=.6)
    elif kind == "bush":
        batch.current_material = FOLIAGE
        for k in range(5 if detail else 3):
            a = 2 * np.pi * k / 5 + rng.uniform(-.4, .4)
            ring = 0.0 if k == 0 else rng.uniform(.3, .5)
            size = rng.uniform(.5, .66) if k == 0 else rng.uniform(.34, .46)
            centre = np.array((np.cos(a) * ring, size * (.8 if k == 0 else .62), np.sin(a) * ring))
            blob(batch, centre, (size, size * .8, size), (light if k == 0 else mid) * rng.uniform(.94, 1.06), sides=sides,
                 rings=rings, normal_from=(0, -.1, 0), soften=.75)
    else:
        raise ValueError(kind)
    return _pack(batch)


def add_stylised_plant(batch, kind, pos, yaw, scale, obj_id=10, detail=1):
    variant = int(zlib.crc32(f"{pos[0]:.2f}|{pos[2]:.2f}".encode()) % (12 if detail else 5))
    vertices, colors, normals, materials, emission = stylised_plant(kind, variant, detail)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array(((c, 0, -s), (0, 1, 0), (s, 0, c)))
    batch.add(vertices @ rotation * scale + np.asarray(pos, float), colors, obj_id, vertex_normals=normals @ rotation,
              material=materials)
    batch.emissive[-1] = emission.copy()


# ----------------------------------------------------------------------- hero
SKIN_TONE = np.array((1.0, .87, .78))
ELEMENT_GEM = {"EMBER": (1.0, .45, .2), "TIDE": (.25, .65, 1.0), "GALE": (.35, .95, .75), "FROST": (.7, .92, 1.0),
               "SPARK": (.75, .5, 1.0), "STONE": (.95, .78, .3)}
HAIR_STYLES = {"EMBER": "ponytail", "TIDE": "long", "GALE": "short", "FROST": "long", "SPARK": "twin", "STONE": "short"}


def hero_pose(state, phase, speed):
    """Joint targets in body-local metres (right, up, forward) for a movement state.

    Returns (pelvis height, lean radians, foot targets L/R, hand targets L/R, knee pole, elbow pole)."""
    cycle = 2 * np.pi * phase
    feet, hands = [], []
    knee_pole, elbow_pole = np.array((0, 0, 1.0)), np.array((0, -.3, -1.0))
    if state == MoveState.CLIMBING:
        for k, side in enumerate((-1, 1)):
            t = np.sin(cycle + np.pi * k)
            feet.append(np.array((side * .13, .36 + .14 * t, .2)))
            hands.append(np.array((side * .2, 1.62 - .16 * t, .3)))
        return .86, .1, feet, hands, np.array((0, .3, 1.0)), np.array((0, -1.0, .2))
    if state == MoveState.GLIDING:
        for side in (-1, 1):
            feet.append(np.array((side * .1, .34, -.42)))
            hands.append(np.array((side * .34, 1.98, .16)))
        return .9, .42, feet, hands, knee_pole, np.array((0, -.2, -1.0))
    if state == MoveState.SWIMMING:
        for k, side in enumerate((-1, 1)):
            t = np.sin(cycle + np.pi * k)
            feet.append(np.array((side * .1, .12 + .1 * t, -.2)))
            hands.append(np.array((side * (.26 + .1 * t), 1.5 + .25 * t, .42)))
        return .9, 1.15, feet, hands, knee_pole, elbow_pole
    if state == MoveState.AIRBORNE:
        for side in (-1, 1):
            feet.append(np.array((side * .11, .36 + .05 * side, .1 * side)))
            hands.append(np.array((side * .46, 1.24, .04)))
        return .92, .08, feet, hands, knee_pole, elbow_pole
    pace = float(np.clip(speed / 6.0, 0.0, 1.0))
    sprint = float(np.clip((speed - 6.2) / 2.5, 0.0, 1.0))
    stride = .38 * pace + .14 * sprint
    lift = .05 + .13 * pace + .08 * sprint
    for k, side in enumerate((-1, 1)):
        a = cycle + np.pi * k
        feet.append(np.array((side * .09, lift * max(0.0, np.sin(a)), stride * np.cos(a))))
        swing = -np.cos(a) * (.26 * pace + .12 * sprint)
        hands.append(np.array((side * (.25 - .04 * pace), .98 + .1 * abs(swing) + .14 * sprint, swing + .02)))
    bob = .012 * np.sin(cycle) if pace < .05 else .028 * pace * np.cos(2 * cycle)
    return .9 + bob - .03 * pace, .05 * pace + .2 * sprint, feet, hands, knee_pole, elbow_pole


def add_hero(batch, st, spec=None, obj_id=2):
    """Draw the player's character from simulation state. Returns the index of its first part in ``batch``."""
    first = len(batch.verts)
    element = spec.element.name if spec is not None else "GALE"
    tunic = np.asarray(spec.tunic if spec is not None else (.33, .52, .86))
    hair = np.asarray(spec.hair if spec is not None else (.96, .84, .52))
    trim, dark, gold = np.array((.96, .95, .92)), np.array((.2, .22, .3)), np.array((.9, .74, .4))
    speed = float(np.hypot(st.vel[0], st.vel[2]))
    pelvis_y, lean, feet, hands, knee_pole, elbow_pole = hero_pose(st.state, st.anim_phase, speed)
    yaw = float(st.yaw)
    c, sn = np.cos(yaw), np.sin(yaw)
    origin = np.asarray(st.pos, float) + ((0, -.35, 0) if st.state == MoveState.SWIMMING else (0, 0, 0))

    def world(p, pivot=.9):
        """Body-local (right, up, forward) to world, leaning forward about the hips."""
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        y0 = y - pivot
        y2 = pivot + y0 * np.cos(lean) - z * np.sin(lean)
        z2 = y0 * np.sin(lean) + z * np.cos(lean)
        return origin + np.array((x * c - z2 * sn, y2, x * sn + z2 * c))

    def direction(p):
        return world(p, 0.0) - world((0, 0, 0), 0.0)

    lift = pelvis_y - .9
    blow = float(np.clip(speed / 9.0, 0.0, 1.0)) + (.6 if st.state == MoveState.GLIDING else 0.0)
    # Legs: leggings above the knee, tall boots below.
    for k, side in enumerate((-1, 1)):
        hip = np.array((side * .085, pelvis_y, 0))
        knee, foot = two_bone(hip, feet[k] + (0, .07, 0), .43, .43, knee_pole)
        batch.current_material = CLOTH
        limb(batch, world(hip), world(knee), .082, .06, dark, obj_id=obj_id)
        limb(batch, world(knee), world(foot), .064, .046, trim * .96, obj_id=obj_id)
        limb(batch, world(knee - (0, .02, 0)), world(knee + (0, .03, 0)), .07, .07, gold, obj_id=obj_id)
        toe = foot + (0, -.035, .1)
        blob(batch, world(toe), (.055, .045, .12), dark, yaw, obj_id, sides=8, rings=4)
    # Torso, waist and belt.
    batch.current_material = CLOTH
    chest, waist, neck = np.array((0, 1.3 + lift, 0)), np.array((0, 1.02 + lift, 0)), np.array((0, 1.47 + lift, 0))
    blob(batch, world(chest), (.165, .19, .115), tunic, yaw, obj_id)
    limb(batch, world(waist - (0, .04, 0)), world(chest - (0, .08, 0)), .125, .15, trim, obj_id=obj_id)
    limb(batch, world(waist - (0, .06, 0)), world(waist - (0, .01, 0)), .14, .14, gold * .85, obj_id=obj_id)
    blob(batch, world((0, pelvis_y + .02, 0)), (.145, .11, .12), dark, yaw, obj_id)
    # Coat skirt: panels flare from the waist and trail with speed; the back is longer.
    panels = 8
    for k in range(panels):
        a0, a1 = 2 * np.pi * k / panels, 2 * np.pi * (k + 1) / panels
        back = max(0.0, -np.cos((a0 + a1) / 2))                       # 1 behind, 0 in front
        if np.cos((a0 + a1) / 2) > .75:
            continue                                                   # open at the front
        length = .30 + .22 * back
        flare = .24 + .1 * back
        sway = .03 * np.sin(2 * np.pi * st.anim_phase * 2 + k) * (.3 + blow)
        top = [np.array((np.sin(a) * .14, waist[1] - .03, np.cos(a) * .12)) for a in (a0, a1)]
        hem = [np.array((np.sin(a) * flare + sway, waist[1] - length, np.cos(a) * flare - blow * (.1 + .25 * back)))
               for a in (a0, a1)]
        color = tunic * (.92 if k % 2 else 1.0)
        quad = [world(top[0]), world(top[1]), world(hem[1]), world(hem[0])]
        core = world((0, waist[1] - length / 2, 0))
        panel(batch, quad[0], quad[1], quad[2], quad[3], color, core, obj_id)
        panel(batch, quad[3], quad[2], world(hem[1] + (0, -.035, 0)), world(hem[0] + (0, -.035, 0)), trim, core, obj_id)
    # Arms: sleeve, glove, hand.
    for k, side in enumerate((-1, 1)):
        shoulder = np.array((side * .185, 1.4 + lift, 0))
        elbow, hand = two_bone(shoulder, hands[k] + (0, lift, 0), .27, .26, elbow_pole + (side * .5, 0, 0))
        batch.current_material = CLOTH
        blob(batch, world(shoulder), (.075, .07, .075), tunic * 1.05, yaw, obj_id, sides=8, rings=4)
        limb(batch, world(shoulder), world(elbow), .058, .048, tunic, obj_id=obj_id)
        limb(batch, world(elbow), world(hand), .05, .04, trim, obj_id=obj_id)
        limb(batch, world(elbow + (hand - elbow) * .72), world(elbow + (hand - elbow) * .8), .05, .05, gold, obj_id=obj_id)
        batch.current_material = SKIN
        blob(batch, world(hand + (hand - elbow) / .26 * .045), (.04, .05, .035), SKIN_TONE, yaw, obj_id, sides=8, rings=4)
    # Scarf tails from the collar, streaming behind.
    batch.current_material = CLOTH
    accent = np.asarray(ELEMENT_GEM.get(element, (.9, .9, .9)))
    limb(batch, world(neck - (0, .05, 0)), world(neck + (0, .01, 0)), .085, .078, accent * .9, obj_id=obj_id)
    for side in (-1, 1):
        root = neck + (side * .05, -.03, -.07)
        previous, width = root, .05
        for seg in range(4):
            t = (seg + 1) / 4
            wave = .05 * np.sin(2 * np.pi * (st.anim_phase * 2 + .22 * seg + .15 * side)) * (.4 + blow)
            nxt = root + (side * (.04 + .05 * t) + wave, -.34 * t * (1 - .6 * min(blow, 1)), -.12 * t - .5 * t * blow)
            a, b = world(previous - (width, 0, 0)), world(previous + (width, 0, 0))
            d, e = world(nxt + (width * .8, 0, 0)), world(nxt - (width * .8, 0, 0))
            panel(batch, a, b, d, e, accent * (.95 if seg % 2 else .8), world(chest), obj_id)
            previous, width = nxt, width * .8
    # Neck, head and face.
    head = np.array((0, 1.615 + lift, .01))
    batch.current_material = SKIN
    limb(batch, world(neck - (0, .02, 0)), world(head - (0, .08, 0)), .045, .042, SKIN_TONE * .96, obj_id=obj_id)
    blob(batch, world(head), (.118, .135, .118), SKIN_TONE, yaw, obj_id, sides=14, rings=8)
    for side in (-1, 1):
        eye = head + (side * .047, -.012, .104)
        batch.current_material = METAL
        blob(batch, world(eye), (.024, .034, .012), dark * .7, yaw, obj_id, sides=8, rings=4)
        blob(batch, world(eye + (0, -.008, .006)), (.015, .019, .008), accent * .8, yaw, obj_id, sides=6, rings=3)
        blob(batch, world(eye + (side * -.008, .014, .01)), (.007, .008, .004), (1, 1, 1), yaw, obj_id, sides=5, rings=3)
    # Hair: a cap, a fringe of locks, side locks and a style at the back.
    batch.current_material = CLOTH
    blob(batch, world(head + (0, .03, -.022)), (.132, .132, .136), hair, yaw, obj_id, sides=14, rings=8)
    for k, x in enumerate(np.linspace(-.095, .095, 6)):
        root = head + (x, .1, .075)
        tip = head + (x * 1.18 + .012 * (-1) ** k, .012 - .012 * (k % 2), .128)
        limb(batch, world(root), world(tip), .034, .008, hair * (1.04 if k % 2 else .96), sides=6, obj_id=obj_id)
    for side in (-1, 1):
        limb(batch, world(head + (side * .11, .04, .03)), world(head + (side * .125, -.2, .05 - .06 * blow)), .036, .008,
             hair * .95, sides=6, obj_id=obj_id)
    style = HAIR_STYLES.get(element, "short")
    sway = .04 * np.sin(2 * np.pi * st.anim_phase * 2)
    if style in ("ponytail", "long"):
        root = head + (0, .06 if style == "ponytail" else -.02, -.12)
        length = .5 if style == "ponytail" else .62
        previous, radius = root, .06 if style == "ponytail" else .1
        if style == "ponytail":
            blob(batch, world(root), (.045, .045, .045), gold, yaw, obj_id, sides=8, rings=4)
        for seg in range(4):
            t = (seg + 1) / 4
            nxt = root + (sway * t * (1 + blow), -length * t * (1 - .55 * min(blow, 1)), -.06 * t - .42 * t * blow)
            limb(batch, world(previous), world(nxt), radius, radius * .68, hair * (1.0 - .05 * seg), sides=8, obj_id=obj_id)
            previous, radius = nxt, radius * .68
    elif style == "twin":
        for side in (-1, 1):
            root = head + (side * .1, .05, -.07)
            tip = root + (side * .08 + sway, -.42 * (1 - .5 * min(blow, 1)), -.08 - .35 * blow)
            limb(batch, world(root), world((root + tip) / 2 + (side * .03, 0, 0)), .05, .042, hair, sides=8, obj_id=obj_id)
            limb(batch, world((root + tip) / 2 + (side * .03, 0, 0)), world(tip), .042, .01, hair * .94, sides=8, obj_id=obj_id)
    else:
        for k in range(7):
            a = np.pi * (.15 + .7 * k / 6)
            root = head + (np.cos(a) * .1, .03, -.1 * np.sin(a) - .02)
            tip = head + (np.cos(a) * .15, -.1 - .03 * (k % 2), -.15 * np.sin(a) - .04 - .05 * blow)
            limb(batch, world(root), world(tip), .04, .006, hair * (1.03 if k % 2 else .95), sides=6, obj_id=obj_id)
        limb(batch, world(head + (0, .15, 0)), world(head + (.03 + sway, .23, -.02)), .014, .003, hair, sides=5, obj_id=obj_id)
    # Element gem at the collar.
    batch.current_material = METAL
    blob(batch, world(chest + (0, .1, .112)), (.03, .036, .016), accent, yaw, obj_id, sides=8, rings=4)
    batch.emissive[-1][:] = 1.2
    if st.state == MoveState.GLIDING:
        _glider(batch, world, tunic, trim, gold, lift, obj_id)
    return first


def _glider(batch, world, tunic, trim, gold, lift, obj_id):
    """A framed wing: a keel, two swept spars, ribs, and cloth panels in the hero's colours."""
    keel_front, keel_back = np.array((0, 2.12 + lift, .95)), np.array((0, 2.02 + lift, -.75))
    batch.current_material = BARK
    limb(batch, world(keel_front), world(keel_back), .022, .018, gold * .7, sides=6, obj_id=obj_id)
    limb(batch, world((-.36, 1.98 + lift, .16)), world((.36, 1.98 + lift, .16)), .018, .018, gold * .7, sides=6, obj_id=obj_id)
    for side in (-1, 1):
        tip = np.array((side * 2.1, 1.86 + lift, -.35))
        limb(batch, world(keel_front), world(tip), .02, .012, gold * .7, sides=6, obj_id=obj_id)
        batch.current_material = CLOTH
        ribs = 5
        front = [keel_front + (tip - keel_front) * (k / ribs) for k in range(ribs + 1)]
        back = [keel_back + (np.array((side * 1.75, 1.84 + lift, -.95)) - keel_back) * (k / ribs) + (0, .05 * np.sin(np.pi * k / ribs), 0)
                for k in range(ribs + 1)]
        for k in range(ribs):
            color = tunic if k % 2 == 0 else trim
            quad = [world(front[k]), world(front[k + 1]), world(back[k + 1]), world(back[k])]
            panel(batch, quad[0], quad[1], quad[2], quad[3], color, quad[0] - (0, 5.0, 0), obj_id)
            batch.current_material = BARK
            limb(batch, quad[1], quad[2], .01, .008, gold * .7, sides=5, obj_id=obj_id)
            batch.current_material = CLOTH
    batch.current_material = 0


def outline_hull(parts, thickness=0.018):
    """Each convex part pushed outward from its own centre: drawn back-face-only as an ink line."""
    hulls = []
    for part in parts:
        if len(part) <= 2:                 # flat cloth panels have no volume to outline
            continue
        points = part.reshape(-1, 3)
        centre = points.mean(axis=0)
        offset = part - centre
        distance = np.maximum(np.linalg.norm(offset, axis=-1, keepdims=True), 1e-5)
        hull = part + offset / distance * thickness
        # Wind every triangle outward: the ink shows on back faces only.
        face = np.cross(hull[:, 1] - hull[:, 0], hull[:, 2] - hull[:, 0])
        inward = (face * (hull.mean(axis=1) - centre)).sum(axis=1) < 0
        hull[inward] = hull[inward][:, [0, 2, 1]]
        hulls.append(hull)
    return np.concatenate(hulls) if hulls else np.zeros((0, 3, 3), np.float32)


# -------------------------------------------------------------- other figures
class _Figure:
    """The fields ``add_hero`` reads, for characters that are not the player."""

    def __init__(self, pos, yaw, phase, speed, state=MoveState.GROUNDED):
        self.pos, self.yaw, self.anim_phase, self.state = np.asarray(pos, float), float(yaw), float(phase), state
        self.vel = np.array((np.sin(yaw) * speed, 0.0, -np.cos(yaw) * speed))


class _Look:
    def __init__(self, tunic, hair, element):
        self.tunic, self.hair = tunic, hair
        self.element = type("Element", (), {"name": element})()


_NPC_LOOKS = (((.80, .56, .20), (.42, .28, .18), "STONE"), ((.26, .42, .66), (.86, .86, .9), "FROST"),
              ((.30, .62, .40), (.55, .36, .2), "GALE"), ((.62, .36, .62), (.2, .18, .24), "SPARK"),
              ((.70, .30, .26), (.92, .8, .5), "EMBER"), ((.24, .56, .62), (.3, .3, .36), "TIDE"))
RAIDER_LOOK = ((.36, .12, .13), (.16, .13, .14), "STONE")


def add_person(batch, identity, pos, yaw, phase=0.0, speed=0.0, look=None, obj_id=3):
    """A townsperson or raider in the hero's style; ``identity`` picks a stable look."""
    tunic, hair, element = look or _NPC_LOOKS[zlib.crc32(str(identity).encode()) % len(_NPC_LOOKS)]
    return add_hero(batch, _Figure(pos, yaw, phase, speed), _Look(tunic, hair, element), obj_id=obj_id)


SENTINEL_GLOW = {"shield": (.25, .85, .85), "exposed": (.55, 1.0, .95), "windup": (1.0, .42, .16), "vulnerable": (1.0, .9, .55)}


def add_sentinel(batch, pos, yaw, phase, live, time, variant=0, obj_id=4):
    """A stone construct whose posture is its phase.

    Shield: four slabs closed around the core. Exposed: the slabs swing open. Wind-up: they
    rise and spin while the core burns orange. Vulnerable: they lie fallen and the core is
    cracked and pale. A dormant sentinel is closed and dark."""
    pos = np.asarray(pos, float)
    rng = np.random.default_rng(1000 + variant)
    stone = np.array(((.56, .55, .52), (.42, .47, .40), (.50, .46, .52))[variant % 3])
    glow = np.asarray(SENTINEL_GLOW[phase]) if live else np.array((.2, .22, .24))
    strength = (0.0 if not live else {"shield": .5, "exposed": 2.2, "windup": 3.0, "vulnerable": 1.0}[phase])
    pulse = 1.0 + (.35 * np.sin(time * 9.0) if live and phase == "windup" else 0.0)
    c, s = np.cos(yaw), np.sin(yaw)

    def at(x, y, z):
        return pos + np.array((x * c - z * s, y, x * s + z * c))

    from .assets import STONE
    batch.current_material = STONE
    # Plinth and rune ring: the ring carries the phase colour at ground level.
    batch.cone(at(0, -.2, 0), 1.5, .5, stone * .8, sides=10, obj_id=obj_id, taper=.86)
    batch.current_material = METAL
    for k in range(12):
        a = 2 * np.pi * k / 12
        batch.box(at(2.6 * np.cos(a), .12, 2.6 * np.sin(a)), (.5, .1, .16), glow, yaw=-a + np.pi / 2 - yaw, obj_id=obj_id,
                  emissive=strength * .5)
    hover = 1.9 + (.12 * np.sin(time * 2.0) if live else -.5)
    core = at(0, hover, 0)
    # Core: an orb in a stone cage.
    blob(batch, core, (.42 * pulse, .42 * pulse, .42 * pulse), glow, obj_id=obj_id, sides=12, rings=6)
    batch.emissive[-1][:] = strength * pulse
    batch.current_material = STONE
    for k in range(3):
        a = 2 * np.pi * k / 3 + time * (.4 if live else 0.0)
        limb(batch, core + (np.cos(a) * .6, -.5, np.sin(a) * .6), core + (np.cos(a) * .6, .5, np.sin(a) * .6), .07, .07, stone * .7,
             sides=6, obj_id=obj_id)
    # Head with an eye slit, and two floating fists.
    blob(batch, at(0, hover + 1.05, 0), (.36, .3, .34), stone, yaw, obj_id, sides=8, rings=4)
    batch.current_material = METAL
    batch.box(at(0, hover + 1.07, -.31), (.4, .07, .08), glow, yaw=-yaw, obj_id=obj_id, emissive=strength)
    if variant % 3 == 1:                                    # the hollow's sentinel wears a crest
        batch.current_material = STONE
        for side in (-1, 1):
            limb(batch, at(side * .2, hover + 1.25, 0), at(side * .5, hover + 1.75, .1), .09, .02, stone * .75, sides=6, obj_id=obj_id)
    batch.current_material = STONE
    for side in (-1, 1):
        sway = .1 * np.sin(time * 2.0 + side) if live else 0.0
        fist = at(side * 1.25, hover - .25 + sway - (.9 if phase == "vulnerable" or not live else 0.0), -.1)
        blob(batch, fist, (.26, .3, .26), stone * .9, yaw, obj_id, sides=8, rings=4)
    # Shield slabs.
    for k in range(4):
        a = np.pi / 4 + np.pi / 2 * k
        if not live or phase == "shield":
            radius, height, tilt, spin = .95, hover, 0.0, 0.0
        elif phase == "exposed":
            radius, height, tilt, spin = 1.7, hover - .1, .5, 0.0
        elif phase == "windup":
            radius, height, tilt, spin = 1.45, hover + 1.1, .2, time * 3.0
        else:
            radius, height, tilt, spin = 1.9 + .2 * rng.uniform(), .22, 1.45, rng.uniform(-.4, .4)
        a += spin
        centre = at(radius * np.cos(a), height, radius * np.sin(a))
        outward = np.array((np.cos(a + yaw), 0.0, np.sin(a + yaw)))
        tangent = np.array((-outward[2], 0.0, outward[0]))
        up = UP * np.cos(tilt) + outward * np.sin(tilt)
        half = np.array((.62, .95, .12))
        corners = [centre + tangent * sx * half[0] * (1.0 - .25 * (sy > 0)) + up * sy * half[1] + outward * sz * half[2]
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        faces = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
        triangles = [[corners[f[0]], corners[f[1]], corners[f[2]]] for f in faces] + [[corners[f[0]], corners[f[2]], corners[f[3]]] for f in faces]
        batch.add(np.asarray(triangles), stone * rng.uniform(.9, 1.08), obj_id)
        batch.current_material = METAL
        batch.box(centre + outward * .14, (.1, .9, .05), glow, yaw=-(a + yaw) + np.pi / 2, obj_id=obj_id, emissive=strength * .6)
        batch.current_material = STONE
    batch.current_material = 0
