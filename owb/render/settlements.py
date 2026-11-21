"""Presentation trim for campaign settlements, keyed solely by building identity.

Every mesh follows the collision envelope from :mod:`owb.world.settlements`:
cladding sits on the outer faces of real wall voxels, shingles cover the real
stepped roof, frames surround the real door and window openings. Functional
buildings reuse the Tempest trim.
"""
from __future__ import annotations

import numpy as np

from open_world_bench.render.assets import STONE, BARK, METAL, CLOTH, FOLIAGE
from open_world_bench.render.meshes import MeshBatch

from ..core import blocks as B
from ..world.settlements import cottage_cells, cottage_dims
from .tempest import _building as functional_building, WOOD, IRON, LIMESTONE


ROOFS = np.array(((.23, .26, .30), (.55, .24, .14), (.44, .37, .21), (.31, .20, .15)))
ACCENTS = np.array(((.20, .36, .50), (.47, .17, .13), (.22, .40, .26), (.58, .46, .20)))
MASONRY = np.array(((.62, .58, .50), (.55, .31, .22)))
BLOOMS = np.array(((.86, .22, .25), (.95, .78, .25), (.62, .38, .80), (.95, .95, .90)))


def _grid(record):
    a, c, _ = cottage_dims(record)
    lo = np.array((-a-1, -2, -c-1)); shape = (2*a+3, c+11, 2*c+3)
    g = np.zeros(shape, np.int8)
    for l, h, block in cottage_cells(record):
        g[l[0]-lo[0]:h[0]-lo[0]+1, l[1]-lo[1]:h[1]-lo[1]+1, l[2]-lo[2]:h[2]-lo[2]+1] = block
    return g, lo


def _frame(b, center, normal, width, height, wood):
    """Jambs, head and sill around a ``width`` x ``height`` opening whose outer
    face centre is ``center``; ``normal`` is the outward (x or z) axis sign."""
    axis, sign = normal
    along = np.array((0., 0., 1.)) if axis == 0 else np.array((1., 0., 0.))
    out = np.zeros(3); out[axis] = sign
    for s in (-1, 1):
        b.box(center+along*s*(width/2+.06)+out*.04, np.abs(along)*.13+np.array((0, height, 0))+np.abs(out)*.12, wood)
    b.box(center+np.array((0, height/2+.07, 0))+out*.05, np.abs(along)*(width+.38)+np.array((0, .15, 0))+np.abs(out)*.14, wood)
    return along, out


def _cottage(record):
    a, c, wall = cottage_dims(record)
    v = record["appearance"]
    rng = np.random.default_rng(v)
    b = MeshBatch()
    wood = WOOD*rng.uniform(.95, 1.3)
    board = np.array(((.52, .38, .24), (.40, .30, .22), (.62, .55, .44)))[(v >> 8) % 3]
    stone = MASONRY[(v >> 10) % 2]
    roof = ROOFS[(v >> 4) % 4]
    accent = ACCENTS[(v >> 6) % 4]
    chimney = a-1 if (v >> 3) % 2 else -(a-1)
    g, lo = _grid(record)
    ridge = c+6
    # 1. Cladding on every outer wall face: weatherboards or coursed masonry.
    sides = (((-1, 0, 0), 0, -1), ((1, 0, 0), 0, 1), ((0, 0, -1), 2, -1), ((0, 0, 1), 2, 1))
    for i, y, j in np.argwhere(g == wall):
        x, yy, z = i+lo[0], y+lo[1], j+lo[2]
        if yy < 0 or (x == chimney and z == c-1) or yy == 4+(c+1-abs(z)):
            continue
        for d, axis, sign in sides:
            n = (i+d[0], y, j+d[2])
            inside = all(0 <= n[k] < g.shape[k] for k in range(3))
            if inside and g[n] != B.AIR:
                continue
            # Outer faces only: the neighbour must lie outside the wall ring.
            nx, nz = x+d[0], z+d[2]
            if abs(nx) < a and abs(nz) < c:
                continue
            face = np.array((x+.5, yy, z+.5), float); face[axis] = (x if axis == 0 else z)+(sign > 0)+sign*.03
            size_along = np.array((0, 0, 1.)) if axis == 0 else np.array((1., 0, 0))
            thin = np.zeros(3); thin[axis] = .06
            for k in range(2):
                center = face+(0, .25+.5*k, 0)
                if wall == B.PLANK:
                    b.current_material = BARK
                    b.box(center+thin*.3*sign*(k == 1), size_along*1.0+(0, .47, 0)+thin, board*rng.uniform(.82, 1.12))
                else:
                    b.current_material = STONE
                    if (yy*2+k+x+z) % 2:
                        b.box(center, size_along*.95+(0, .44, 0)+thin, stone*rng.uniform(.8, 1.15))
                    else:
                        for s in (-.25, .25):
                            b.box(center+size_along*s, size_along*.46+(0, .44, 0)+thin, stone*rng.uniform(.8, 1.15))
    # Timber corner posts, sill and wall plate.
    b.current_material = BARK
    for x in (-a-.04, a+1.04):
        for z in (-c-.04, c+1.04):
            b.box((x, 2.5, z), (.22, 5., .22), wood*.6)
    for z in (-c-.06, c+1.06):
        for y in (.1, 4.9):
            b.box((.5, y, z), (2*a+1.3, .18, .14), wood*.62)
    for x in (-a-.06, a+1.06):
        for y in (.1, 4.9):
            b.box((x, y, .5), (.14, .18, 2*c+1.3), wood*.62)
    # 2. Door: frame, an open leaf against the jamb, step stone and lantern.
    _frame(b, np.array((.5, 1., -c-.02)), (2, -1), 1., 2., wood*.55)
    b.box((.06, 1., -c+1.5), (.07, 1.95, .95), accent*.85)
    b.current_material = METAL
    b.box((.14, 1., -c+1.9), (.06, .08, .06), IRON)
    b.current_material = STONE
    b.box((.5, .04, -c-.55), (1.5, .09, .9), LIMESTONE*.8)
    b.current_material = METAL
    b.tube((-.5, 2.85, -c), (-.5, 2.85, -c-.48), .03, IRON)
    b.box((-.5, 2.6, -c-.5), (.22, .32, .22), (1., .74, .42), emissive=1.6)
    b.box((-.5, 2.8, -c-.5), (.3, .06, .3), IRON)
    # 3. Windows: frames, mullions, warm panes, shutters, flower boxes.
    windows = [((x, -c), (2, -1)) for x in (-(a-2), a-2)]+[((x, c), (2, 1)) for x in (-(a-2), a-2)]
    windows += [((x, 0), (0, -1 if x < 0 else 1)) for x in (-a, a)]
    for (x, z), (axis, sign) in windows:
        center = np.array((x+.5, 2., z+.5))
        center[axis] = (x if axis == 0 else z)+(sign > 0)+sign*.02
        b.current_material = BARK
        along, out = _frame(b, center, (axis, sign), 1., 2., wood*.55)
        b.box(center+(0, -1.05, 0)+out*.08, np.abs(along)*1.3+(0, .1, 0)+np.abs(out)*.22, wood*.55)
        pane = center-out*.5
        b.current_material = METAL
        b.box(pane, np.abs(along)*.98+(0, 1.98, 0)+np.abs(out)*.03, (.86, .63, .36), emissive=.35)
        b.current_material = BARK
        b.box(pane+out*.03, np.abs(along)*.07+(0, 1.98, 0)+np.abs(out)*.05, wood*.5)
        b.box(pane+out*.03+(0, .2, 0), np.abs(along)*.98+(0, .07, 0)+np.abs(out)*.05, wood*.5)
        for s in (-1, 1):
            b.box(center+along*s*.86+out*.05, np.abs(along)*.5+(0, 1.9, 0)+np.abs(out)*.05, accent*rng.uniform(.85, 1.1))
            for y in (-.55, .1, .75):
                b.box(center+along*s*.86+out*.09+(0, y, 0), np.abs(along)*.46+(0, .07, 0)+np.abs(out)*.03, accent*.7)
        if axis == 2 and sign < 0:
            box = center+(0, -1.3, 0)+out*.22
            b.box(box, np.abs(along)*1.15+(0, .26, 0)+np.abs(out)*.3, wood*.75)
            b.current_material = FOLIAGE
            for k in range(5):
                p = box+along*(-.45+.225*k)+(0, .2, 0)
                b.ellipsoid(p, (.14, .13, .14), np.array((.18, .36, .14))*rng.uniform(.8, 1.2))
                b.ellipsoid(p+(0, .12, 0), (.07, .06, .07), BLOOMS[int(rng.integers(0, 4))])
    # 4. Shingled roof on the real stepped roof, with fascia, verges and ridge.
    for side in (-1, 1):
        z0, z1 = (-c-1.35, .5) if side < 0 else (.5, c+2.35)
        yp = (lambda z: ridge+z+.06) if side < 0 else (lambda z: ridge+1-z+.06)
        n = int(np.ceil((z1-z0)/.42))
        edges = np.linspace(z0, z1, n+1)
        b.current_material = STONE
        for k in range(n):
            za, zb = edges[k], edges[k+1]
            offset = .45*(k % 2)
            xs = np.concatenate(([-a-1.3], np.arange(-a-1+offset, a+2, .9), [a+2.3]))
            xs = np.unique(np.clip(xs, -a-1.3, a+2.3))
            for xa, xb in zip(xs[:-1], xs[1:]):
                col = roof*rng.uniform(.78, 1.16)
                b.quad((xa, yp(za), za), (xb, yp(za), za), (xb, yp(zb), zb), (xa, yp(zb), zb), col)
        b.current_material = BARK
        ze = -c-1.03 if side < 0 else c+2.03
        b.box((.5, 4.55, ze), (2*a+3.6, .9, .07), wood*.55)
        for x in (-a-1.05, a+2.05):
            b.quad((x, yp(z0)+.12, z0), (x, yp(z1)+.12, z1), (x, yp(z1)-1.12, z1), (x, yp(z0)-1.12, z0), wood*.55)
    b.current_material = STONE
    b.box((.5, ridge+.6, .5), (2*a+3.7, .2, .36), roof*.7)
    # 5. Chimney stack above the roof: coursed stone, cap and pot.
    b.current_material = STONE
    for yy in range(5, c+7):
        for k in range(2):
            for d in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                cx = chimney+.5+d[0]*.53; cz = c-.5+d[1]*.53
                size = (.07, .46, 1.04) if d[0] else (1.04, .46, .07)
                b.box((cx, yy+.25+.5*k, cz), size, stone*rng.uniform(.75, 1.1))
    b.box((chimney+.5, c+7.1, c-.5), (1.35, .2, 1.35), stone*.7)
    b.current_material = METAL
    b.tube((chimney+.5, c+7.2, c-.5), (chimney+.5, c+7.75, c-.5), .2, (.45, .24, .15), end_radius=.16, sides=10)
    # 6. Yard props: woodpile under the gable, bench, water butt.
    sx = -1 if chimney > 0 else 1
    face = -a if sx < 0 else a+1
    b.current_material = BARK
    for row in range(3):
        for k in range(4-row):
            x = face+sx*(.17+.145*row+.29*k); y = .15+.26*row
            b.tube((x, y, -c+.4), (x, y, -.4), .14, wood*rng.uniform(.9, 1.4), sides=8)
    b.box((2.5, .46, -c-.7), (1.4, .08, .38), wood*.8)
    for x in (1.95, 3.05):
        b.box((x, .22, -c-.7), (.08, .44, .32), wood*.6)
    bx, bz = (-a+.55, -c-.5)
    b.tube((bx, 0., bz), (bx, .95, bz), .34, wood*.9, sides=14)
    b.current_material = METAL
    for y in (.18, .78):
        b.tube((bx, y, bz), (bx, y+.06, bz), .36, IRON, sides=14)
    # Rotate about the centre cell, then place the asset in the world.
    k = record.get("rotation", 0) % 4
    p = np.asarray(record["pos"], np.float32)
    for i in range(len(b.verts)):
        v = b.verts[i]
        u, w = v[..., 0]-.5, v[..., 2]-.5
        for _ in range(k):
            u, w = -w, u
        b.verts[i] = np.stack((u+.5+p[0], v[..., 1]+p[1], w+.5+p[2]), axis=-1).astype(np.float32)
        nrm = b.normals[i]
        nu, nw = nrm[..., 0], nrm[..., 2]
        for _ in range(k):
            nu, nw = -nw, nu
        b.normals[i] = np.stack((nu, nrm[..., 1], nw), axis=-1).astype(np.float32)
    return b


def settlement_meshes(records, batch):
    for record in records:
        b = _cottage(record) if record["kind"] == "cottage" else functional_building(record)
        for name in ("verts", "colors", "emissive", "ids", "normals", "material_ids"):
            getattr(batch, name).extend(getattr(b, name))
