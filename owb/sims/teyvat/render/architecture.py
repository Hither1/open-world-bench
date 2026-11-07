"""Original timber, courtyard, and ruin architecture shared by both renderers."""
from functools import lru_cache

import numpy as np

from ..architecture import RADII, solid_parts
from .assets import _pack, STONE, BARK, METAL, CLOTH
from .meshes import MeshBatch


WOOD = (.22, .105, .055)
STONE_COLOR = np.array((.49, .48, .40))
GOLD = (.72, .48, .18)


def _box(batch, center, size, color, material=STONE):
    batch.current_material = material
    batch.box(center, size, color)


def _beam(batch, start, end, radius=.09, color=WOOD, material=BARK):
    batch.current_material = material
    batch.tube(start, end, radius, color, sides=6)


def _tile(batch, corners, color, material=STONE, thickness=.09):
    """Closed tile slab; top, underside and edges all survive glTF export."""
    top = np.asarray(corners)
    normal = np.cross(top[1]-top[0], top[2]-top[0])
    normal /= max(np.linalg.norm(normal), 1e-8)
    bottom = top-normal*thickness
    vertices = np.concatenate((top, bottom))
    faces = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]]
    for i in range(4):
        j = (i+1) % 4
        faces.extend(([i, j, j+4], [i, j+4, i+4]))
    batch.add(vertices[faces], color, material=material)


def _gable(batch, rng, width, depth, eave, rise, color):
    for side in (-1, 1):
        for row in range(7):
            t0, t1 = row/7, (row+1)/7
            for col in range(14):
                z0, z1 = -depth+2*depth*col/14, -depth+2*depth*(col+1)/14-.018
                def point(t, z):
                    return (side*width*t, eave+rise*(1-t), z)
                corners = [point(t0, z0), point(t0, z1), point(t1, z1), point(t1, z0)]
                _tile(batch, corners, np.asarray(color)*rng.uniform(.88, 1.12))
        for z in (-depth, depth):
            _beam(batch, (0, eave+rise, z), (side*width, eave, z), .13)
        _beam(batch, (side*width, eave, -depth), (side*width, eave, depth), .13)
    _beam(batch, (0, eave+rise+.06, -depth-.1), (0, eave+rise+.06, depth+.1), .12,
          color=(.31, .29, .24), material=STONE)


def _hip_roof(batch, rng, width, eave, rise, color):
    # Four curved tiled slopes with raised corners and gold hip ribs.
    def point(side, t, u):
        edge = np.array((width*t, width*t*u))
        angle = side*np.pi/2
        x, z = edge @ np.array(((np.cos(angle), np.sin(angle)), (-np.sin(angle), np.cos(angle))))
        y = eave+rise*(1-t)**1.6+.38*t**6*abs(u)**3
        return (x, y, z)
    for side in range(4):
        for row in range(8):
            t0, t1 = .06+.94*row/8, .06+.94*(row+1)/8
            for col in range(10):
                u0, u1 = -1+2*col/10, -1+2*(col+1)/10-.008
                _tile(batch, [point(side, t0, u0), point(side, t0, u1),
                              point(side, t1, u1), point(side, t1, u0)],
                      np.asarray(color)*rng.uniform(.88, 1.12))
        for t0, t1 in zip(np.linspace(.06, 1, 10)[:-1], np.linspace(.06, 1, 10)[1:]):
            _beam(batch, point(side, t0, 1), point(side, t1, 1), .065, GOLD, METAL)
    batch.current_material = METAL
    batch.cone((0, eave+rise-.15, 0), .19, .65, GOLD, sides=8)


def _window(batch, x, y, z, width=.95, height=1.2):
    _box(batch, (x, y, z), (width, height, .07), (.09, .19, .20), METAL)
    for dx in (-width/2, 0, width/2):
        _box(batch, (x+dx, y, z-.07), (.07, height+.12, .12), WOOD, BARK)
    for dy in (-height/2, 0, height/2):
        _box(batch, (x, y+dy, z-.07), (width+.12, .07, .12), WOOD, BARK)
    for side in (-1, 1):
        _box(batch, (x+side*(width/2+.25), y, z), (.35, height, .12), (.26, .36, .31), BARK)
        for dy in (-.4, .35):
            _box(batch, (x+side*(width/2+.25), y+dy, z-.07), (.36, .055, .06), WOOD, BARK)


def _masonry(batch, rng, half_width, height, half_depth):
    # Individual shallow facing stones with staggered joints on all four walls.
    for side in range(4):
        angle = side*np.pi/2
        c, s = np.cos(angle), np.sin(angle)
        width, depth = (half_width, half_depth) if side % 2 == 0 else (half_depth, half_width)
        for row in range(int(height/.48)):
            for col in range(int(2*width/.8)+1):
                x = -width+.4+col*.8-(row % 2)*.4
                lo, hi = max(-width, x-.38), min(width, x+.38)
                if hi-lo < .06:
                    continue
                local = np.array(((lo+hi)/2, row*.48+.23, -depth-.025))
                center = (local[0]*c+local[2]*s, local[1], -local[0]*s+local[2]*c)
                batch.current_material = STONE
                batch.box(center, (hi-lo, .44, .075), np.asarray(STONE_COLOR)*rng.uniform(.83, 1.15), yaw=angle)


@lru_cache(maxsize=256)
def _template(kind, appearance, foundation_depth=0.0):
    rng = np.random.default_rng(appearance)
    batch = MeshBatch()
    plaster = np.asarray(((.83, .74, .55), (.77, .79, .68), (.88, .77, .62))[appearance % 3])
    roof = np.asarray(((.18, .30, .34), (.40, .19, .12), (.24, .31, .22))[appearance % 3])
    wood = np.asarray(WOOD)*(.85+.1*(appearance % 4))
    palette = dict(plaster=(plaster, STONE), stone=(STONE_COLOR, STONE), wood=(wood, BARK))
    for center, size, material in solid_parts(kind, foundation_depth):
        color, tag = palette[material]
        _box(batch, center, size, color, tag)
    if kind == "timber_house":
        _box(batch, (0, -.35-foundation_depth/2, 0), (6.12, 1.3+foundation_depth, 5.12), STONE_COLOR)
        for x in (-2.92, 0, 2.92):
            for z in (-2.53, 2.53):
                _box(batch, (x, 2, z), (.18, 4, .18), wood, BARK)
        for y in (.4, 2.65, 3.92):
            for z in (-2.54, 2.54):
                _box(batch, (0, y, z), (6.15, .17, .16), wood, BARK)
            for x in (-3.03, 3.03):
                _box(batch, (x, y, 0), (.16, .17, 5.1), wood, BARK)
        for z in (-2.56, 2.56):
            for side in (-1, 1):
                _beam(batch, (side*.15, 2.7, z), (side*2.85, 3.85, z), .085, wood)
            batch.add(np.array([[[-3, 4, z], [3, 4, z], [0, 6.1, z]]]), plaster, material=STONE)
            for side in (-1, 1):
                _beam(batch, (0, 6.1, z), (side*3, 4, z), .12, wood)
        _box(batch, (0, 1.1, -2.57), (1.25, 2.2, .14), wood, BARK)
        for x in np.linspace(-.5, .5, 6):
            _box(batch, (x, 1.1, -2.66), (.12, 2.1, .05), wood*1.2, BARK)
        _box(batch, (.4, 1.1, -2.72), (.1, .12, .07), GOLD, METAL)
        for x in (-1.9, 1.9):
            _window(batch, x, 1.7, -2.58)
        _gable(batch, rng, 3.6, 3.05, 3.9, 2.3, roof)
        _box(batch, (1.8, 5.3, 1.2), (.75, 2.5, .8), STONE_COLOR)
        _box(batch, (1.8, 6.6, 1.2), (.95, .2, 1.0), STONE_COLOR*.8)
    elif kind == "windmill":
        _masonry(batch, rng, 2.3, 3.1, 2.3)
        _box(batch, (0, 5.1, 0), (4.68, 3.9, 4.68), plaster)
        for x in (-2.36, 2.36):
            for z in (-2.36, 2.36):
                _box(batch, (x, 5.1, z), (.18, 4, .18), wood, BARK)
        _gable(batch, rng, 2.95, 2.8, 7.05, 1.9, roof)
        _window(batch, 0, 1.7, -2.37, 1.0, 1.4)
        hub = np.array((0, 6.25, -2.95))
        batch.current_material = METAL
        batch.ellipsoid(hub, (.32, .32, .25), GOLD)
        for angle in np.arange(4)*np.pi/2+.32:
            direction = np.array((np.cos(angle), np.sin(angle), 0))
            across = np.array((-np.sin(angle), np.cos(angle), 0))
            _beam(batch, hub, hub+direction*4.6, .11, wood)
            for length in np.linspace(1.3, 4.5, 8):
                _beam(batch, hub+direction*length-across*.1,
                      hub+direction*length+across*.9, .035, wood)
            _tile(batch, [hub+direction*1.3, hub+direction*4.5,
                          hub+direction*4.5+across*.85, hub+direction*1.3+across*.65],
                  (.84, .79, .62), CLOTH, thickness=.025)
            _beam(batch, hub+direction*1.3+across*.9, hub+direction*4.5+across*.9, .055, wood)
    elif kind == "pavilion":
        for x in (-2.6, 2.6):
            for z in (-2.6, 2.6):
                _box(batch, (x, -.35-foundation_depth/2, z), (.62, 1.25+foundation_depth, .62), STONE_COLOR)
                for dx, dz in ((-np.sign(x)*.8, 0), (0, -np.sign(z)*.8)):
                    _beam(batch, (x, 2.65, z), (x+dx, 3.65, z+dz), .12, wood)
                _box(batch, (x, 3.55, z), (.8, .18, .8), wood, BARK)
        _hip_roof(batch, rng, 3.5, 3.85, 2.0, (.12, .33, .30))
        for x in (-2, 2):
            _beam(batch, (x, 3.5, -2.3), (x, 2.9, -2.3), .02, GOLD, METAL)
            _box(batch, (x, 2.75, -2.3), (.28, .4, .28), (.78, .37, .15), CLOTH)
    elif kind == "market_stall":
        for stripe in range(12):
            x0, x1 = -2.2+4.4*stripe/12, -2.2+4.4*(stripe+1)/12
            color = (.72, .28, .15) if stripe % 2 else (.88, .80, .60)
            for row in range(6):
                z0, z1 = -1.9+3.8*row/6, -1.9+3.8*(row+1)/6
                def p(x, z):
                    return (x, 3.25-.17*abs(z)-.12*(1-(x/2.2)**2), z)
                _tile(batch, [p(x0, z0), p(x1, z0), p(x1, z1), p(x0, z1)], color, CLOTH, .025)
            _tile(batch, [(x0, 2.92, -1.9), (x1, 2.92, -1.9),
                          (x1, 2.65, -1.9), (x0, 2.65, -1.9)], color, CLOTH, .025)
        for x in (-1.2, -.4, .4, 1.2):
            _box(batch, (x, 1.15, 1), (.65, .25, .55), wood*1.3, BARK)
            batch.current_material = STONE
            batch.cone((x, 1.28, 1), .16, .35, (.65, .30, .15), sides=10, taper=.65)
        for x in np.linspace(-1.6, 1.6, 12):
            _box(batch, (x, .5, .68), (.22, 1.0, .06), wood*1.15, BARK)
    elif kind == "ruin_gate":
        for x in (-3.1, 3.1):
            for y in np.arange(.2, 4, .5):
                _box(batch, (x, y, -.93), (1.4, .44, .09), np.asarray(STONE_COLOR)*rng.uniform(.85, 1.1))
            _box(batch, (x, .05, 0), (1.85, .35, 2.1), STONE_COLOR)
            _box(batch, (x, 3.8, 0), (1.85, .32, 2.15), STONE_COLOR)
        for i in range(13):
            a0, a1 = i*np.pi/13+.012, (i+1)*np.pi/13-.012
            corners = [(r*np.cos(a), 4.6+r*np.sin(a), -.92)
                       for r, a in ((1.1, a0), (1.1, a1), (1.8, a1), (1.8, a0))]
            # Decorative sun arch above the solid lintel; the passage stays open.
            _tile(batch, corners, np.asarray(STONE_COLOR)*rng.uniform(.85, 1.1), thickness=.16)
        _box(batch, (0, 4.35, -1.01), (.65, .45, .15), (.18, .40, .41), METAL)
        for x in (-3.1, 3.1):
            _box(batch, (x, 2.7, -.99), (.2, .7, .09), (.25, .34, .24))
    elif kind == "watchtower":
        _masonry(batch, rng, 2.25, 5.4, 2.25)
        for y in (.4, 4.95):
            _box(batch, (0, y, 0), (4.85, .25, 4.85), STONE_COLOR*.9)
        for x in (-1.9, 0, 1.9):
            for z in (-2.15, 2.15):
                h = rng.uniform(.65, 1.05)
                _box(batch, (x, 5.4+h/2, z), (.72, h, .65), STONE_COLOR)
        for x in (-2.15, 2.15):
            _box(batch, (x, 5.85, 0), (.65, .9, .72), STONE_COLOR)
        _window(batch, 0, 3.8, -2.32, .45, 1.3)
        _box(batch, (0, 1.15, -2.32), (1.3, 2.3, .14), wood, BARK)
        for x in (-.45, 0, .45):
            _box(batch, (x, 1.15, -2.42), (.05, 2.3, .06), (.20, .23, .24), METAL)
    return _pack(batch)


def add_architecture(batch, architecture, eye, max_distance):
    if architecture is None:
        return
    for index, building in enumerate(architecture.buildings):
        if np.linalg.norm((np.asarray(building.pos)-eye)[[0, 2]]) > max_distance+RADII[building.kind]:
            continue
        vertices, colors, normals, materials, emission = _template(building.kind, building.appearance, building.foundation_depth)
        rotation = building.rotation
        batch.add(vertices @ rotation.T+building.pos, colors, 20000+index,
                  vertex_normals=normals @ rotation.T, material=materials)
        batch.emissive[-1] = emission.copy()
