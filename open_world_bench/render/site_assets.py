"""Meshes for the full-scale site pieces (``fullscale_sites``).

Each template is built once per (kind, appearance, foundation depth) from the
same primitives as the shared architecture, around the piece's colliding boxes.
Appearance is keyed to the piece's stable identity, never to the camera.
"""
from functools import lru_cache
import zlib

import numpy as np

from ..fullscale_sites import PIECE_RADII, piece_parts
from .architecture import (GOLD, STONE_COLOR, WOOD, _beam, _box, _gable, _hip_roof, _masonry, _tile, _window)
from .assets import BARK, CLOTH, FOLIAGE, METAL, STONE, _pack
from .meshes import MeshBatch

# One cloth colour per site, so banners and awnings tell the sites apart from a distance.
_SITE_CLOTH = ((.70, .16, .13), (.14, .36, .62), (.80, .62, .14), (.20, .50, .30), (.52, .22, .55), (.85, .42, .12),
               (.12, .52, .55), (.62, .12, .32))
THATCH = np.array((.62, .52, .28))
CANVAS = np.array((.80, .74, .60))
DARK_STONE = np.array((.36, .36, .34))
TIMBER = np.array((.43, .30, .18))      # weathered boards: lighter than the shared frame wood


def site_cloth(site: str):
    return np.asarray(_SITE_CLOTH[zlib.crc32(site.encode()) % len(_SITE_CLOTH)])


def _plinth(batch, half_x, half_z, depth, color=STONE_COLOR):
    _box(batch, (0, -.2 - depth / 2, 0), (2 * half_x + .3, .6 + depth, 2 * half_z + .3), np.asarray(color) * .92)


def _planks(batch, rng, centre, size, wood, axis=0, count=8):
    """A wall of vertical planks with slight tone variation."""
    cx, cy, cz = centre
    for k in range(count):
        t = (k + .5) / count - .5
        offset = (t * size[0], 0, 0) if axis == 0 else (0, 0, t * size[2])
        dims = (size[0] / count - .03, size[1], size[2]) if axis == 0 else (size[0], size[1], size[2] / count - .03)
        _box(batch, (cx + offset[0], cy, cz + offset[2]), dims, np.asarray(wood) * rng.uniform(.85, 1.18), BARK)


def _rough_block(batch, rng, centre, size, color, pieces=5):
    """An irregular standing stone: stacked, slightly turned and tapered blocks."""
    cx, cy, cz = centre
    base = cy - size[1] / 2
    step = size[1] / pieces
    batch.current_material = STONE
    for k in range(pieces):
        taper = 1.0 - .28 * (k / max(pieces - 1, 1)) ** 1.4
        w, d = size[0] * taper * rng.uniform(.9, 1.08), size[2] * taper * rng.uniform(.9, 1.08)
        batch.box((cx + rng.uniform(-.06, .06), base + (k + .5) * step, cz + rng.uniform(-.06, .06)),
                  (w, step * 1.04, d), np.asarray(color) * rng.uniform(.82, 1.12), yaw=rng.uniform(-.22, .22))


@lru_cache(maxsize=512)
def _template(kind, appearance, foundation_depth=0.0, cloth=(.7, .16, .13)):
    rng = np.random.default_rng(appearance)
    batch = MeshBatch()
    wood = np.asarray(WOOD) * (.85 + .1 * (appearance % 4))
    roof = np.asarray(((.18, .30, .34), (.40, .19, .12), (.24, .31, .22))[appearance % 3])
    plaster = np.asarray(((.83, .74, .55), (.77, .79, .68), (.88, .77, .62))[appearance % 3])
    cloth = np.asarray(cloth)
    timber = TIMBER * (.9 + .07 * (appearance % 4))
    d = foundation_depth
    if kind == "granary":
        # Tall stone store with a timber upper floor, a steep roof and a loading crane.
        _box(batch, (0, 3.6, 0), (7.0, 7.2 + .01, 7.0), STONE_COLOR)
        _masonry(batch, rng, 3.5, 7.2, 3.5)
        _plinth(batch, 3.5, 3.5, d)
        _box(batch, (0, 9.6, 0), (7.5, 4.8, 7.5), plaster)
        for x in (-3.78, 0, 3.78):
            for z in (-3.78, 3.78):
                _box(batch, (x, 9.6, z), (.22, 4.9, .22), wood, BARK)
        for z in (-3.78, 3.78):
            for y in (7.3, 9.6, 11.95):
                _box(batch, (0, y, z), (7.7, .2, .2), wood, BARK)
            for sx in (-1, 1):
                _beam(batch, (sx * .2, 7.4, z), (sx * 3.6, 11.8, z), .09, wood)
        for x in (-3.78, 3.78):
            for y in (7.3, 9.6, 11.95):
                _box(batch, (x, y, 0), (.2, .2, 7.7), wood, BARK)
        _gable(batch, rng, 4.5, 4.3, 12.0, 4.4, roof)
        for z in (-3.82, 3.82):
            batch.add(np.array([[[-3.75, 12.0, z], [3.75, 12.0, z], [0, 16.2, z]]]), plaster, material=STONE)
        _box(batch, (0, 1.5, -3.56), (2.4, 3.0, .16), wood, BARK)
        for x in np.linspace(-1.0, 1.0, 7):
            _box(batch, (x, 1.5, -3.66), (.2, 2.9, .05), wood * 1.2, BARK)
        for y in (9.2, 10.8):
            _window(batch, -1.9, y, -3.8, .8, 1.0)
        _box(batch, (1.4, 9.9, -3.82), (1.5, 2.2, .14), wood * .8, BARK)       # loading hatch
        # Crane: jib out over the front, braced, with a hanging hook block.
        jib_root, jib_tip = np.array((1.4, 12.6, -3.6)), np.array((1.4, 13.4, -8.4))
        _beam(batch, jib_root, jib_tip, .16, wood)
        _beam(batch, (1.4, 10.9, -3.9), jib_root + (jib_tip - jib_root) * .62, .1, wood)
        _beam(batch, (.6, 12.3, -3.7), (2.2, 12.3, -3.7), .12, wood)
        _beam(batch, jib_tip, jib_tip + (0, -6.2, 0), .03, (.55, .5, .4), CLOTH)
        _box(batch, jib_tip + (0, -6.5, 0), (.34, .5, .26), DARK_STONE * .7, METAL)
        _box(batch, jib_tip + (0, -7.3, 0), (1.0, .9, 1.0), wood * 1.3, BARK)   # the load
        batch.current_material = METAL
        batch.cone((0, 16.3, 0), .14, 1.3, GOLD, sides=6)
        _tile(batch, [(0, 17.5, 0), (0, 17.5, 1.5), (0, 16.9, 1.5), (0, 16.9, 0)], cloth, CLOTH, .03)
    elif kind == "storehouse":
        _plinth(batch, 5.5, 2.75, d)
        _box(batch, (0, .45, 0), (11.0, .9, 5.5), STONE_COLOR)
        _planks(batch, rng, (0, 2.75, -2.7), (10.9, 3.7, .12), timber, axis=0, count=24)
        _planks(batch, rng, (0, 2.75, 2.7), (10.9, 3.7, .12), timber, axis=0, count=24)
        _planks(batch, rng, (-5.45, 2.75, 0), (.12, 3.7, 5.3), timber, axis=2, count=12)
        _planks(batch, rng, (5.45, 2.75, 0), (.12, 3.7, 5.3), timber, axis=2, count=12)
        for x in (-5.45, -1.8, 1.8, 5.45):
            for z in (-2.72, 2.72):
                _box(batch, (x, 2.5, z), (.24, 4.4, .24), wood * .8, BARK)
        _gable(batch, rng, 3.5, 6.1, 4.6, 2.3, roof)
        for x in (-5.5, 5.5):
            pass
        for z in (-2.74, 2.74):
            _box(batch, (0, 4.55, z), (11.2, .2, .2), wood * .8, BARK)
        for x in (-5.5, 5.5):
            batch.add(np.array([[[x, 4.6, -2.75], [x, 4.6, 2.75], [x, 6.9, 0]]]), timber * .92, material=BARK)
        _box(batch, (0, 2.0, -2.8), (2.6, 3.0, .14), wood * .7, BARK)           # double door
        _box(batch, (0, 2.0, -2.88), (.06, 3.0, .05), GOLD, METAL)
        for x in (-3.6, 3.6):
            _window(batch, x, 2.9, -2.78, .8, .9)
    elif kind == "workshop":
        # Open timber frame under a roof: benches and a relay-part rack inside.
        for x in (-4.2, 0.0, 4.2):
            for z in (-2.6, 2.6):
                _box(batch, (x, -.2 - d / 2, z), (.7, .6 + d, .7), STONE_COLOR)
                _box(batch, (x, 2.3, z), (.34, 4.6, .34), wood, BARK)
                _beam(batch, (x, 3.5, z), (x, 4.5, z - np.sign(z) * 1.0), .08, wood)
        for z in (-2.6, 2.6):
            _box(batch, (0, 4.55, z), (9.0, .26, .26), wood, BARK)
        for x in (-4.2, 0.0, 4.2):
            _box(batch, (x, 4.55, 0), (.26, .26, 5.5), wood, BARK)
        _gable(batch, rng, 3.5, 5.0, 4.7, 2.0, roof)
        _box(batch, (0, .85, 1.6), (3.0, .12, .9), wood * 1.25, BARK)
        for x in (-1.3, 1.3):
            _box(batch, (x, .42, 1.6), (.14, .84, .8), wood, BARK)
        batch.current_material = METAL
        batch.cone((-.6, .92, 1.6), .22, .3, (.42, .44, .46), sides=8, taper=.8)
        _box(batch, (.6, 1.02, 1.6), (.5, .2, .3), (.30, .32, .35), METAL)
        for x in np.linspace(-3.4, -1.2, 4):                                    # rack of planks
            _box(batch, (x, 1.5, -2.2), (.18, 3.0, .08), wood * rng.uniform(1.0, 1.4), BARK)
    elif kind == "well":
        batch.current_material = STONE
        batch.cone((0, -.3 - d, 0), .95, 1.25 + d, STONE_COLOR, sides=10, taper=.94)
        batch.cone((0, .9, 0), .62, .06, (.08, .16, .2), sides=10, taper=1.0)
        for x in (-.8, .8):
            _box(batch, (x, 1.6, 0), (.14, 2.2, .14), wood, BARK)
        _beam(batch, (-.9, 2.3, 0), (.9, 2.3, 0), .09, wood)
        _gable(batch, rng, 1.2, 1.0, 2.6, .7, roof)
        _beam(batch, (0, 2.3, 0), (0, 1.2, 0), .02, (.55, .5, .4), CLOTH)
        batch.current_material = BARK
        batch.cone((0, 1.0, 0), .2, .32, wood * 1.3, sides=8, taper=.85)
    elif kind == "fence":
        for x in (-2.1, 0, 2.1):
            _box(batch, (x, .5 - d / 2, 0), (.16, 1.25 + d, .16), wood, BARK)
        for y in (.45, .9):
            _beam(batch, (-2.1, y, 0), (2.1, y + rng.uniform(-.04, .04), 0), .05, wood * 1.15)
    elif kind == "palisade":
        for k, x in enumerate(np.linspace(-1.95, 1.95, 9)):
            h = 3.2 + rng.uniform(-.25, .3)
            batch.current_material = BARK
            batch.cone((x, -.3 - d, rng.uniform(-.05, .05)), .24, h + .3 + d, timber * rng.uniform(.75, 1.1), sides=6, taper=.8)
            batch.cone((x, h, 0), .19, .42, timber * 1.2, sides=6)
        for y in (1.0, 2.3):
            _box(batch, (0, y, .24), (4.2, .16, .1), wood * .75, BARK)
    elif kind == "tent":
        ridge = 2.9
        for side in (-1, 1):
            for k in range(6):
                z0, z1 = -1.7 + 3.4 * k / 6, -1.7 + 3.4 * (k + 1) / 6
                _tile(batch, [(0, ridge, z0), (0, ridge, z1), (side * 1.75, 0, z1), (side * 1.75, 0, z0)],
                      CANVAS * rng.uniform(.9, 1.06), CLOTH, .04)
        batch.add(np.array([[[-1.7, 0, 1.7], [1.7, 0, 1.7], [0, ridge, 1.7]]]), CANVAS * .86, material=CLOTH)
        for z in (-1.7, 1.7):
            _beam(batch, (0, 0, z), (0, ridge + .25, z), .06, wood)
        _beam(batch, (0, ridge, -1.75), (0, ridge, 1.75), .05, wood)
        for side in (-1, 1):
            _tile(batch, [(side * .08, ridge, -1.72), (side * .08, .0, -1.72), (side * 1.7, 0, -1.72), (side * .9, 1.6, -1.72)],
                  cloth * rng.uniform(.9, 1.05), CLOTH, .03)
    elif kind == "standing_stone":
        _rough_block(batch, rng, (0, 1.4 - d / 2, 0), (1.05, 3.3 + d, .82), STONE_COLOR * .9, pieces=5)
        _box(batch, (0, 2.0, -.44), (.34, .5, .05), (.25, .62, .60), METAL)
        batch.emissive[-1][:] = .35
    elif kind == "signal_mast":
        _box(batch, (0, 2.0, 0), (3.4, 4.4, 3.4), STONE_COLOR)
        _masonry(batch, rng, 1.7, 4.2, 1.7)
        _plinth(batch, 1.7, 1.7, d)
        _box(batch, (0, 4.35, 0), (3.9, .3, 3.9), STONE_COLOR * .85)
        batch.current_material = BARK
        batch.cone((0, 4.4, 0), .34, 15.5, wood, sides=8, taper=.35)
        for y, span in ((12.0, 3.4), (15.5, 2.4), (18.2, 1.4)):
            _beam(batch, (-span, y, 0), (span, y, 0), .09, wood)
            for sx in (-1, 1):
                _beam(batch, (sx * span, y, 0), (sx * span * .35, y - 1.6, 0), .03, (.5, .47, .4), CLOTH)
                _box(batch, (sx * span, y + .28, 0), (.3, .44, .3), (.80, .86, .82), METAL)
                batch.emissive[-1][:] = .5
        for a in np.arange(4) * np.pi / 2 + np.pi / 4:
            _beam(batch, (1.6 * np.cos(a), 4.4, 1.6 * np.sin(a)), (0, 11.0, 0), .025, (.5, .47, .4), CLOTH)
        batch.current_material = METAL
        batch.cone((0, 19.9, 0), .12, 1.1, GOLD, sides=6)
        _tile(batch, [(0, 19.6, 0), (1.9, 19.4, 0), (1.9, 18.6, 0), (0, 18.8, 0)], cloth, CLOTH, .03)
    elif kind == "lift_pylon":
        _box(batch, (0, -.2 - d / 2, 0), (2.0, .7 + d, 2.0), STONE_COLOR)
        for sx in (-1, 1):
            for sz in (-1, 1):
                _beam(batch, (sx * .68, .1, sz * .68), (sx * .34, 10.6, sz * .34), .11, wood)
        for y in (2.2, 4.8, 7.4, 9.9):
            w = .68 - (.34 * y / 10.6)
            for sx, sz, ex, ez in ((-1, -1, 1, -1), (1, -1, 1, 1), (1, 1, -1, 1), (-1, 1, -1, -1)):
                _beam(batch, (sx * w, y, sz * w), (ex * w, y, ez * w), .055, wood * 1.15)
        _box(batch, (0, 10.8, 0), (2.6, .28, 1.1), wood, BARK)
        for sx in (-1, 1):
            batch.current_material = METAL
            batch.ellipsoid((sx * 1.05, 10.45, 0), (.36, .36, .12), (.32, .34, .36))
        _hip_roof(batch, rng, 1.5, 11.0, .9, roof)
    elif kind == "beacon_tower":
        _box(batch, (0, 4.2, 0), (2.5, 8.6, 2.5), STONE_COLOR)
        _masonry(batch, rng, 1.25, 8.4, 1.25)
        _plinth(batch, 1.25, 1.25, d)
        _box(batch, (0, 8.6, 0), (3.2, .3, 3.2), STONE_COLOR * .85)
        for x in (-1.35, 1.35):
            for z in (-1.35, 1.35):
                _box(batch, (x, 9.6, z), (.2, 1.8, .2), DARK_STONE)
        _hip_roof(batch, rng, 2.0, 10.5, 1.2, roof)
        batch.current_material = METAL
        batch.cone((0, 8.75, 0), .55, .5, (.24, .22, .2), sides=8, taper=1.25)
        batch.ellipsoid((0, 9.6, 0), (.42, .55, .42), (1.0, .72, .3))
        batch.emissive[-1][:] = 2.2
        _window(batch, 0, 5.2, -1.3, .4, 1.1)
        _box(batch, (0, 1.1, -1.3), (1.0, 2.2, .12), wood, BARK)
    elif kind == "shrine_hall":
        _box(batch, (0, -.1 - d / 2, 0), (8.0, .5 + d, 6.0), STONE_COLOR * .95)
        _box(batch, (0, .22, -3.3), (4.0, .22, .7), STONE_COLOR * .88)
        for x in (-3.2, 3.2):
            for z in (-2.2, 2.2):
                batch.current_material = BARK
                batch.cone((x, .15, z), .22, 4.4, (.52, .14, .10), sides=10, taper=.9)
                _box(batch, (x, 4.5, z), (.7, .2, .7), wood, BARK)
        for z in (-2.2, 2.2):
            _box(batch, (0, 4.3, z), (7.4, .3, .3), (.52, .14, .10), BARK)
        for x in (-3.2, 3.2):
            _box(batch, (x, 4.3, 0), (.3, .3, 4.9), (.52, .14, .10), BARK)
        _box(batch, (0, 1.75, 2.3), (6.4, 3.3, .32), plaster)
        _box(batch, (0, 1.0, 1.6), (2.2, 1.0, .9), STONE_COLOR * 1.05)          # altar
        _box(batch, (0, 1.75, 1.6), (.6, .5, .4), (.25, .62, .60), METAL)
        batch.emissive[-1][:] = .6
        _hip_roof(batch, rng, 4.9, 4.55, 2.6, (.12, .33, .30))
        for x in (-2.6, 2.6):
            _beam(batch, (x, 4.3, -2.3), (x, 3.3, -2.3), .02, GOLD, METAL)
            _box(batch, (x, 3.1, -2.3), (.3, .44, .3), cloth, CLOTH)
    elif kind == "crates":
        for (x, y, z), s, turn in (((-.5, .45, 0), .9, .1), ((.55, .4, .2), .8, -.25), ((0, 1.15, .1), .7, .5)):
            batch.current_material = BARK
            batch.box((x, y - d / 2 if y < .6 else y, z), (s, s + (d if y < .6 else 0), s), timber * rng.uniform(.95, 1.25), yaw=turn)
            for e in (-1, 1):
                batch.box((x, y + e * s * .46, z), (s * 1.04, .06, s * 1.04), wood * .7, yaw=turn)
        batch.current_material = BARK
        batch.cone((.2, 0, -.55), .34, .9, wood * 1.1, sides=10, taper=.85)
        batch.current_material = METAL
        for y in (.2, .7):
            batch.cone((.2, y, -.55), .36, .05, (.28, .28, .3), sides=10, taper=1.0)
    elif kind == "cart":
        _box(batch, (0, .78, 0), (2.5, .14, 1.3), timber, BARK)
        for z in (-.65, .65):
            _planks(batch, rng, (0, 1.1, z), (2.5, .55, .07), timber, axis=0, count=6)
        _box(batch, (-1.25, 1.1, 0), (.07, .55, 1.3), wood, BARK)
        for z in (-.78, .78):
            batch.current_material = BARK
            for a in np.linspace(0, 2 * np.pi, 10, endpoint=False):
                p0 = np.array((.2 + .62 * np.cos(a), .62 + .62 * np.sin(a), z))
                p1 = np.array((.2 + .62 * np.cos(a + 2 * np.pi / 10), .62 + .62 * np.sin(a + 2 * np.pi / 10), z))
                batch.tube(p0, p1, .05, wood * .8, sides=5)
                batch.tube((.2, .62, z), p0, .03, wood, sides=4)
        for z in (-.4, .4):
            _beam(batch, (1.25, .8, z), (2.9, .35, z), .05, wood)
        for x, s in ((-.6, .55), (.3, .45)):
            _box(batch, (x, .85 + s / 2, 0), (s * 1.3, s, s * 1.2), CANVAS * rng.uniform(.8, 1.0), CLOTH)
    elif kind == "banner":
        _box(batch, (0, -.1 - d / 2, 0), (.6, .4 + d, .6), STONE_COLOR)
        batch.current_material = BARK
        batch.cone((0, 0, 0), .09, 5.4, wood, sides=6, taper=.7)
        _beam(batch, (0, 5.0, 0), (1.25, 5.0, 0), .035, wood)
        for k in range(5):
            y0, y1 = 4.95 - .6 * k, 4.95 - .6 * (k + 1)
            sway = .06 * np.sin(k * 1.3 + appearance % 7)
            _tile(batch, [(.08, y0, sway), (1.2, y0, sway), (1.2 - (.5 if k == 4 else 0), y1, sway * 1.4), (.08, y1, sway * 1.4)],
                  cloth * (1.0 if k % 2 == 0 else .82), CLOTH, .02)
        batch.current_material = METAL
        batch.cone((0, 5.4, 0), .08, .3, GOLD, sides=6)
    elif kind == "ruin_wall":
        length, n = 5.0, 9
        heights = 1.2 + 2.0 * np.clip(rng.uniform(.1, 1.0, n) * np.linspace(1.0, .45, n)[rng.permutation(n) % n], 0, 1)
        heights[[0, -1]] *= .6
        for k in range(n):
            x = -length / 2 + (k + .5) * length / n
            rows = int(heights[k] / .5) + 1
            for row in range(rows):
                h = min(.5, heights[k] - row * .5)
                if h <= .05:
                    break
                batch.current_material = STONE
                batch.box((x + (row % 2) * .12 - .06, row * .5 + h / 2 - (d if row == 0 else 0) / 2, rng.uniform(-.04, .04)),
                          (length / n - .03, h + (d if row == 0 else 0), .9 * rng.uniform(.92, 1.05)),
                          STONE_COLOR * rng.uniform(.72, 1.05))
        for k in range(3):
            _rough_block(batch, rng, (rng.uniform(-2.2, 2.2), .22, rng.choice((-1, 1)) * rng.uniform(.9, 1.5)),
                         (.6, .44, .5), STONE_COLOR * .8, pieces=1)
    elif kind == "broken_column":
        batch.current_material = STONE
        batch.cone((0, -.2 - d, 0), .62, .5 + d, STONE_COLOR * .9, sides=10, taper=.92)
        h = 0.3
        for k in range(4):
            seg = rng.uniform(.5, .72)
            batch.cone((rng.uniform(-.03, .03), h, rng.uniform(-.03, .03)), .43 - .01 * k, seg, STONE_COLOR * rng.uniform(.85, 1.1),
                       sides=10, taper=.98)
            h += seg
        batch.cone((0, h, 0), .42, .28, STONE_COLOR * .8, sides=10, taper=.55)
        drum = np.array((1.0, .3, .5))
        batch.tube(drum, drum + (.9, .0, .5), .36, STONE_COLOR * .85, sides=10)
    elif kind == "haystack":
        batch.current_material = CLOTH
        batch.ellipsoid((0, .75, 0), (1.15, 1.0, 1.15), THATCH * rng.uniform(.9, 1.1), sides=10, rings=5)
        batch.cone((0, 1.3, 0), .7, .75, THATCH * 1.08, sides=10)
        batch.cone((0, -.2 - d, 0), 1.1, .5 + d, THATCH * .8, sides=10, taper=1.0)
    else:
        raise ValueError(f"unknown site piece: {kind}")
    return _pack(batch)


def add_site_pieces(batch, pieces, eye, max_distance, obj_base=24000):
    """Add every piece within ``max_distance`` of the eye (plan distance)."""
    eye = np.asarray(eye, float)
    for index, piece in enumerate(pieces):
        if np.hypot(piece.pos[0] - eye[0], piece.pos[2] - eye[2]) > max_distance + PIECE_RADII[piece.kind]:
            continue
        vertices, colors, normals, materials, emission = _template(
            piece.kind, piece.appearance, piece.foundation_depth, tuple(site_cloth(piece.site).tolist()))
        rotation = piece.rotation
        batch.add(vertices @ rotation.T + np.asarray(piece.pos), colors, obj_base + index,
                  vertex_normals=normals @ rotation.T, material=materials)
        batch.emissive[-1] = emission.copy()


@lru_cache(maxsize=8)
def far_peaks(seed, inner=1250.0, outer=2400.0, count=34, sea=-4.0):
    """Mountains standing in the sea beyond the island: presentation only.

    Each massif is a few overlapping peaks. A peak rises from a broad foot in stepped tiers
    with lobed outlines and level shoulders, so the skyline reads as layered rock from any
    bearing. Nothing here is reachable or collides."""
    rng = np.random.default_rng([int(seed), 0xFA12])
    batch = MeshBatch()
    batch.current_material = STONE
    sides = 18
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    for k in range(count):
        bearing = 2 * np.pi * (k + rng.uniform(-.4, .4)) / count
        distance = rng.uniform(inner, outer)
        centre = np.array((distance * np.cos(bearing), sea, distance * np.sin(bearing)))
        scale = .75 + .9 * (distance - inner) / (outer - inner)
        for peak in range(int(rng.integers(2, 5))):
            spread = 170.0 * scale * (peak > 0)
            base = centre + np.array((rng.uniform(-spread, spread), 0, rng.uniform(-spread, spread)))
            radius = rng.uniform(120, 230) * scale * (1.0 if peak == 0 else rng.uniform(.5, .8))
            height = rng.uniform(150, 330) * scale * (1.0 if peak == 0 else rng.uniform(.45, .8))
            phase = rng.uniform(0, 2 * np.pi, 3)
            lobes = 1 + .2 * np.sin(2 * angles + phase[0]) + .13 * np.sin(3 * angles + phase[1]) + .07 * np.sin(7 * angles + phase[2])
            tiers = int(rng.integers(4, 7))
            y, r = 0.0, radius
            rings = [(0.0, r)]
            for tier in range(tiers):
                rise = height / tiers * rng.uniform(.7, 1.3)
                rings.append((y + rise, r * rng.uniform(.78, .9)))            # steep face, leaning in
                y += rise
                r *= rng.uniform(.6, .78)
                rings.append((y + rng.uniform(2, 8), r))                      # shoulder
            drift = rng.uniform(-.12, .12, 2) * radius                        # the summit sits off-centre
            points = [base + np.stack((np.cos(angles) * rr * lobes + drift[0] * yy / max(y, 1),
                                       np.full(sides, yy), np.sin(angles) * rr * lobes + drift[1] * yy / max(y, 1)), axis=1)
                      for yy, rr in rings]
            triangles = []
            for lower, upper in zip(points, points[1:]):
                for i in range(sides):
                    j = (i + 1) % sides
                    triangles += [[lower[i], upper[i], upper[j]], [lower[i], upper[j], lower[j]]]
            top = points[-1]
            apex = top.mean(axis=0) + (0, rng.uniform(8, 26) * scale, 0)
            triangles += [[top[i], apex, top[(i + 1) % sides]] for i in range(sides)]
            batch.add(np.asarray(triangles), (.58, .56, .52))
    vertices, colors, _, _ = batch.finish()
    return vertices, colors


GRASS_VARIANTS = 8


@lru_cache(maxsize=16)
def grass_clump(variant):
    """A lawn clump for the presentation carpet: broad blades, dark at the root and bright at the tip."""
    rng = np.random.default_rng(9100 + variant)
    batch = MeshBatch()
    batch.current_material = FOLIAGE
    warm = rng.uniform(0, 1)
    root = np.array((.25, .50, .16)) * (1 - .2 * warm) + np.array((.30, .48, .16)) * .2 * warm
    tip = np.array((.50, .74, .28)) * (1 - .35 * warm) + np.array((.64, .78, .34)) * .35 * warm
    for _ in range(int(rng.integers(12, 18))):
        a = rng.uniform(0, 2 * np.pi)
        h = rng.uniform(.10, .26)
        out = np.array((np.cos(a), 0, np.sin(a)))
        side = np.array((-np.sin(a), 0, np.cos(a)))
        base = out * rng.uniform(0, .2)
        lean = out * rng.uniform(.04, .16)
        w = rng.uniform(.022, .04)
        p0, p1, p2 = base, base + lean * .4 + (0, h * .6, 0), base + lean + (0, h, 0)
        mid = root + (tip - root) * .55
        batch.add([[p0 - side * w, p0 + side * w, p1 + side * w * .65],
                   [p0 - side * w, p1 + side * w * .65, p1 - side * w * .65]], np.clip(mid * rng.uniform(.9, 1.08), 0, 1))
        batch.add([[p1 - side * w * .65, p1 + side * w * .65, p2]], np.clip(tip * rng.uniform(.92, 1.08), 0, 1))
    if variant == 3:                              # an occasional flower
        petal = ((1.0, .96, .9), (1.0, .82, .3), (.62, .7, 1.0))[variant % 3]
        batch.tube((0, 0, 0), (0, .3, 0), .008, root, sides=4)
        batch.ellipsoid((0, .31, 0), (.05, .018, .05), petal, sides=6, rings=2)
    batch.current_material = 0
    return _pack(batch)


NEAR_PLANTS = (55.0, 45)      # full presentation detail: (metres, most plants)
MID_PLANTS = (130.0, 150)     # the shared observation-detail templates


@lru_cache(maxsize=256)
def _distant_template(kind, variant):
    """A plant for the far field: the detailed template's size and colours in a few dozen triangles."""
    from .assets import FOLIAGE, _prop_template
    vertices, colors, _, materials, _ = _prop_template(kind, variant, 0)
    points = vertices.reshape(-1, 3)
    leaf = np.repeat(materials == FOLIAGE, 3) if (materials == FOLIAGE).any() else np.ones(len(points), bool)
    leaf_color = colors[materials == FOLIAGE].mean(axis=0) if (materials == FOLIAGE).any() else colors.mean(axis=0)
    wood_color = colors[materials != FOLIAGE].mean(axis=0) if (materials != FOLIAGE).any() else leaf_color * .5
    top = float(points[:, 1].max())
    crown = points[leaf]
    lo, hi = float(np.percentile(crown[:, 1], 4)), float(crown[:, 1].max())
    width = float(np.percentile(np.hypot(crown[:, 0], crown[:, 2]), 92))
    rng = np.random.default_rng(variant)
    batch = MeshBatch()
    if kind == "pine":
        batch.current_material = BARK
        batch.cone((0, 0, 0), .22, top * .5, wood_color, sides=5, taper=.6)
        batch.current_material = FOLIAGE
        tiers = 5
        for k in range(tiers):
            t = k / tiers
            base = lo + (hi - lo) * t * .92
            batch.cone((rng.uniform(-.08, .08), base, rng.uniform(-.08, .08)), width * (1.0 - .78 * t),
                       (hi - lo) * (.34 if k < tiers - 1 else .3), leaf_color * rng.uniform(.88, 1.1), sides=7)
    elif kind in ("broadleaf", "sakura"):
        batch.current_material = BARK
        batch.cone((0, 0, 0), .26, lo + (hi - lo) * .35, wood_color, sides=5, taper=.55)
        batch.current_material = FOLIAGE
        mid = (lo + hi) / 2
        for k in range(5):
            a = 2 * np.pi * k / 5 + rng.uniform(-.3, .3)
            r = width * (.42 if k else 0.0)
            batch.ellipsoid((r * np.cos(a), mid + rng.uniform(-.12, .2) * (hi - lo) + (.18 * (hi - lo) if k == 0 else 0),
                             r * np.sin(a)), (width * .62, (hi - lo) * .36, width * .62),
                            leaf_color * rng.uniform(.86, 1.12), sides=7, rings=4)
    elif kind == "bamboo":
        for k in range(5):
            a = 2 * np.pi * k / 5
            x, z = .35 * np.cos(a), .35 * np.sin(a)
            lean = rng.uniform(-.5, .5, 2)
            batch.current_material = BARK
            batch.tube((x, 0, z), (x + lean[0], top * .95, z + lean[1]), .05, wood_color, sides=4)
            batch.current_material = FOLIAGE
            batch.ellipsoid((x + lean[0], top * .8, z + lean[1]), (.5, top * .16, .5), leaf_color * rng.uniform(.9, 1.1),
                            sides=6, rings=3)
    elif kind == "bush":
        batch.current_material = FOLIAGE
        for k in range(3):
            a = 2 * np.pi * k / 3
            batch.ellipsoid((width * .35 * np.cos(a), top * .45, width * .35 * np.sin(a)),
                            (width * .7, top * .5, width * .7), leaf_color * rng.uniform(.88, 1.1), sides=6, rings=3)
    else:
        return None
    return _pack(batch)


def _add_distant(batch, kind, pos, yaw, scale, obj_id):
    template = _distant_template(kind, int(zlib.crc32(f"{pos[0]:.2f}|{pos[2]:.2f}".encode()) % 6))
    if template is None:
        return False
    vertices, colors, normals, materials, emission = template
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array(((c, 0, -s), (0, 1, 0), (s, 0, c)))
    batch.add(vertices @ rotation * scale + pos, colors, obj_id, vertex_normals=normals @ rotation, material=materials)
    batch.emissive[-1] = emission.copy()
    return True


def add_vegetation(batch, props, eye, forward, focus, max_distance=420.0, budget=900, obj_id=10, stylised=False):
    """Presentation pass for groves: every mesh plant in view up to ``budget``, nearest first.

    The shared renderer draws at most 212 plants a frame, an observation budget that thins a
    grove to a few trees. Here the nearest plants use the full presentation templates,
    the middle distance the shared observation templates, and the far
    field compact stand-ins with the same size and colours. Plants standing between the eye
    and ``focus`` (the character) are left out, as in the shared third-person cutaway;
    collision is unchanged."""
    from ..scatter import GRASS, PROP_NAMES
    from .meshes import add_prop
    eye, focus = np.asarray(eye, float), np.asarray(focus, float)
    delta = props.pos[:, [0, 2]] - eye[[0, 2]]
    dist = np.hypot(delta[:, 0], delta[:, 1])
    horizontal = np.asarray(forward, float)[[0, 2]]
    horizontal = horizontal / max(np.linalg.norm(horizontal), 1e-6)
    along = delta @ horizontal
    across = np.abs(delta[:, 0] * horizontal[1] - delta[:, 1] * horizontal[0])
    boom = float(np.linalg.norm((focus - eye)[[0, 2]]))
    occluding = (along > -1.0) & (along < boom - 0.3) & (across < props.radius + 0.65) & (props.radius > 0)
    near_grass = (props.kind == GRASS) & (np.linalg.norm(props.pos - focus, axis=1) < 22.0)
    visible = (dist < max_distance) & (along > -0.25 * dist - 6.0) & ~occluding & ((props.kind != GRASS) | near_grass)
    chosen = np.flatnonzero(visible)
    chosen = chosen[np.argsort(dist[chosen])][:budget]
    near = mid = 0
    if stylised:
        from .anime_assets import STYLISED_KINDS, add_stylised_plant
    for i in chosen:
        name = PROP_NAMES[int(props.kind[i])]
        args = (props.pos[i], float(props.yaw[i]), float(props.scale[i]))
        if stylised and name in STYLISED_KINDS:
            add_stylised_plant(batch, name, *args, obj_id=obj_id, detail=int(dist[i] <= MID_PLANTS[0] * 1.6))
            continue
        if dist[i] <= NEAR_PLANTS[0] and near < NEAR_PLANTS[1]:
            batch.presentation_detail = True
            add_prop(batch, name, *args, obj_id=obj_id)
            batch.presentation_detail = False
            near += 1
        elif dist[i] <= MID_PLANTS[0] and mid < MID_PLANTS[1]:
            add_prop(batch, name, *args, obj_id=obj_id)
            mid += 1
        elif not _add_distant(batch, name, np.asarray(args[0], float), args[1], args[2], obj_id):
            add_prop(batch, name, *args, obj_id=obj_id)       # rocks, boulders, crystals: already small
    return len(chosen)


__all__ = ["add_site_pieces", "add_vegetation", "site_cloth", "piece_parts"]
