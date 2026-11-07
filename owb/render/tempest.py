"""Detailed original industrial assets, keyed solely by world/entity identity."""
from __future__ import annotations

import numpy as np

from owb.sims.teyvat.render.assets import STONE, BARK, METAL, CLOTH
from owb.sims.teyvat.render.meshes import MeshBatch


WOOD = np.array((.29, .16, .085))
IRON = np.array((.19, .26, .28))
COPPER = np.array((.55, .31, .15))
PATINA = np.array((.15, .40, .37))
LIMESTONE = np.array((.61, .59, .49))


def ring(b, center, radius, tube, color, *, normal=(0, 1, 0), segments=32):
    center = np.asarray(center, float)
    n = np.asarray(normal, float); n /= np.linalg.norm(n)
    u = np.cross(n, (1, 0, 0))
    if np.linalg.norm(u) < .1:
        u = np.cross(n, (0, 0, 1))
    u /= np.linalg.norm(u); v = np.cross(n, u)
    points = [center+radius*(u*np.cos(a)+v*np.sin(a)) for a in np.linspace(0, 2*np.pi, segments+1)]
    for start, end in zip(points, points[1:]):
        b.tube(start, end, tube, color, sides=8)


def wheel(b, p, radius, *, phase=0, color=COPPER, normal=(0, 0, 1)):
    p = np.asarray(p, float)
    ring(b, p, radius, .09, color, normal=normal)
    # Wheels on workshop frontages are mounted in the X/Y plane.
    for angle in np.arange(10)*np.pi/5+phase:
        b.tube(p, p+(np.cos(angle)*radius, np.sin(angle)*radius, 0), .055, color)
    b.tube(p+(0, 0, -.2), p+(0, 0, .2), .19, IRON)


def pipe(b, points, radius=.18, color=PATINA):
    b.current_material = METAL
    for a, z in zip(points, points[1:]):
        a, z = np.asarray(a, float), np.asarray(z, float)
        b.tube(a, z, radius, color, sides=12)
        axis = (z-a)/np.linalg.norm(z-a)
        for t in (.05, .95):
            c = a+(z-a)*t
            b.tube(c-axis*.045, c+axis*.045, radius*1.35, IRON, sides=12)


def crate(b, p, width, height, variant):
    p = np.asarray(p)
    b.current_material = BARK
    color = WOOD*(.9+variant*.08)
    b.box(p+(0, height/2, 0), (width, height, width), color)
    for h in (.12, height-.12):
        for z in (-width/2-.01, width/2+.01):
            b.box(p+(0, h, z), (width+.04, .1, .06), WOOD*.65)
    for x in (-width*.35, width*.35):
        b.current_material = METAL
        b.box(p+(x, height/2, -width/2-.04), (.07, height, .04), IRON)


def _building(record):
    rng = np.random.default_rng(record["appearance"])
    kind = record["kind"]
    b = MeshBatch()
    wood = WOOD*rng.uniform(.85, 1.15)
    roof = np.array(((.20, .32, .34), (.40, .22, .14), (.22, .34, .27))[record["appearance"] % 3])
    height = 8 if kind in ("foundry", "observatory", "archive") else 5
    # Every foundation has distinct stone courses and capped corners.
    b.current_material = STONE
    for side in (-1, 1):
        for row in range(4):
            for col in range(18):
                x = -8+.45+col*.94
                b.box((x, -1.8+row*.46, side*7.52), (.90, .42, .07), LIMESTONE*rng.uniform(.76, 1.12))
    if kind in ("gantry", "signal_tower"):
        h = 17 if kind == "signal_tower" else 12
        b.current_material = METAL
        for x in (-5.9, 6.9):
            for level in range(0, h-1, 3):
                b.tube((x, level, -4.9), (x, level+3, 5.9), .10, IRON)
                b.tube((x, level, 5.9), (x, level+3, -4.9), .10, IRON)
        for z in (-4.9, 5.9):
            for x in np.arange(-5, 6, 2):
                b.tube((x, h+.12, z), (x+2, h+1.8, z), .13, COPPER)
        b.tube((.5, h, .5), (.5, 2, .5), .045, IRON)
        wheel(b, (.5, h-.8, -.15), .9)
        ring(b, (.5, 1.7, .5), .4, .10, COPPER, normal=(0, 0, 1))
        if kind == "signal_tower":
            b.tube((.5, h+1, .5), (.5, h+7, .5), .13, COPPER)
            for dy in (2.5, 4., 5.5):
                ring(b, (.5, h+dy, .5), 2-dy*.18, .08, PATINA)
            for x in (-2, 2):
                b.tube((.5, h+6, .5), (x, h+3, .5), .04, COPPER)
    elif kind == "cistern":
        b.current_material = METAL
        pipe(b, [(-5, 4.5, -5.5), (-5, 7, -5.5), (4, 7, -5.5), (4, 3, -5.5)], .28)
        for x in np.arange(-5.5, 6.5, 1.):
            b.box((x, 6.15, -.5), (.1, .12, 10), IRON)
        wheel(b, (4, 3, -5.9), .65)
        b.current_material = STONE
        for z in (-5.5, 6.5):
            b.box((.5, 6.02, z), (14, .15, 1.1), LIMESTONE)
    else:
        # Match the solid envelope; stone joints and timber frames decorate it.
        masonry = kind in ("foundry", "pump_house", "archive", "observatory")
        for side in (-1, 1):
            z = -6.04 if side < 0 else 7.04
            b.current_material = STONE if masonry else BARK
            for row in range((height+1)*2):
                for col in range(15):
                    x = -6.55+col*.95
                    if -2.1 < x < 3.1 and row < 8:
                        continue
                    color = LIMESTONE if masonry else wood
                    b.box((x, row*.5+.23, z), (.90, .45, .07), color*rng.uniform(.82, 1.12))
            b.current_material = BARK
            for x in (-6.8, -2.3, 3.3, 7.0):
                b.box((x, height/2, z+side*.1), (.20, height, .17), wood*.65)
            for y in (.3, 4., height+.2):
                b.box((.5, y, z+side*.1), (14.3, .17, .18), wood*.7)
            for x in (-4.8, 5.8):
                b.tube((x-1.3, 4.1, z+side*.13), (x+1.3, height-.1, z+side*.13), .08, wood)
        # Finish all elevations, including the real side-window openings.
        for x in (-7.04, 8.04):
            b.current_material = STONE if masonry else BARK
            for row in range((height+1)*2):
                for col in range(13):
                    z = -5.55+col*.95
                    y = row*.5+.23
                    if 2 <= y < 4 and -2.45 < z < 3.45:
                        continue
                    b.box((x, y, z), (.07, .45, .90),
                          (LIMESTONE if masonry else wood)*rng.uniform(.82, 1.12))
            b.current_material = BARK
            for z in (-5.8, -2.2, 3.2, 6.8):
                b.box((x, (height+1)/2, z), (.17, height+1, .18), wood*.65)
            for y in (.3, 1.95, 4.05, height+.8):
                b.box((x, y, .5), (.19, .15, 13.5), wood*.7)
            for z in (-1.9, -.3, 1.3, 2.9):
                b.box((x, 3., z), (.14, 2., .09), COPPER)
        # Hundreds of individually colored plates follow the actual stepped roof.
        if kind != "observatory":
            b.current_material = METAL if kind == "foundry" else STONE
            for x in range(-8, 9):
                y = height+2+(8-abs(x))//2
                for col in range(18):
                    b.box((x+.5, y+.035, -6.57+col*.82), (.96, .07, .78), roof*rng.uniform(.82, 1.15))
                for z in (-7.04, 8.04):
                    b.current_material = BARK
                    b.box((x+.5, y-.5, z), (.96, .94, .08), wood*.75)
                    b.current_material = METAL if kind == "foundry" else STONE
            b.current_material = METAL
            for z in (-7.1, 8.1):
                b.tube((-8, height+1, z), (8.9, height+1, z), .09, COPPER)
        else:
            b.current_material = METAL
            center = np.array((.5, height+2., .5))
            for angle in np.arange(16)*np.pi/8:
                previous = center+(6*np.cos(angle), 0, 5*np.sin(angle))
                for t in np.linspace(.15, np.pi/2, 10):
                    point = center+(6*np.cos(angle)*np.cos(t), 5*np.sin(t), 5*np.sin(angle)*np.cos(t))
                    b.tube(previous, point, .085, PATINA); previous = point
            ring(b, center, 5.7, .16, COPPER)
            b.tube((.5, height+1, .5), (.5, height+5, .5), .3, IRON)
            b.tube((-2, height+3, .5), (4, height+7, .5), .8, PATINA, end_radius=.55, sides=24)
            ring(b, (4, height+7, .5), .58, .1, COPPER, normal=(6, 4, 0))
        if kind in ("pump_house", "foundry"):
            b.current_material = METAL
            pipe(b, [(-5, .7, -7), (-5, 3, -7), (-3.7, 3, -7), (-3.7, 6, -7)], .25)
            wheel(b, (-5, 2.2, -7.4), .85)
            for x in (4.4, 5.8):
                b.tube((x, .4, -7), (x, 3.9, -7), .43, PATINA, sides=20)
                ring(b, (x, 1., -7), .46, .07, COPPER)
                ring(b, (x, 3.2, -7), .46, .07, COPPER)
            for y in np.arange(.5, height+8.5, .6):
                b.box((5.6, y, 4.55), (.07, .07, .5), IRON)
        elif kind == "boathouse":
            b.current_material = BARK
            for x in np.linspace(-5.7, -3.5, 8):
                b.tube((x, 1., -5), (x-.2, 2.6, -3), .07, wood)
                b.tube((x-.2, 2.6, -3), (x, 1., -1), .07, wood)
            b.tube((-5.5, 1.1, -5), (-3.5, 1.1, -5), .15, wood)
            for x in (-5, 5):
                ring(b, (x, 2.4, -6.4), .65, .11, (.76, .51, .25), normal=(0, 0, 1))
        elif kind == "archive":
            b.current_material = BARK
            for x in (-5.6, 5.6):
                for h in (1., 2., 3.):
                    b.box((x, h, 3.5), (1.7, .12, 3.4), wood)
                    for k in range(8):
                        b.box((x, h+.35, 2.05+k*.38), (.9, .6, .26), np.array((.35, .25, .17))*rng.uniform(.6, 1.6))
        elif kind == "workshop":
            b.current_material = BARK
            b.box((-4.8, 1.25, .5), (2.8, .24, 6.5), wood*1.2)
            for z in (-2.1, 3.1):
                for x in (-5.8, -3.8):
                    b.box((x, .6, z), (.16, 1.2, .16), wood*.7)
            b.current_material = METAL
            b.tube((-4.8, 1.7, -.8), (-4.8, 1.7, 2.), .24, PATINA)
            wheel(b, (-4.8, 1.7, -1.), .55)
            for i in range(7):
                x = -6.1+i*.45
                b.tube((x, 2.5, 5.7), (x, 3.6+(i % 3)*.15, 5.7), .035, IRON)
                b.box((x, 3.6+(i % 3)*.15, 5.7), (.25, .14, .12), COPPER)
        elif kind == "shelter":
            b.current_material = BARK
            for x in (-5.3, 5.8):
                for h in (.6, 2.1):
                    b.box((x, h, .5), (2.1, .18, 3.8), wood)
                    b.current_material = CLOTH
                    b.box((x, h+.2, .5), (1.8, .25, 3.4), (.40, .47, .32))
                    b.box((x, h+.38, 1.7), (1.3, .2, .5), (.80, .75, .59))
                    b.current_material = BARK
            b.current_material = METAL
            b.box((-4.7, 1., 4.6), (1.2, 1.5, 1.), IRON)
            pipe(b, [(-4.7, 1.8, 4.6), (-4.7, 4., 4.6), (-6.3, 4., 4.6)], .14, color=IRON)
        for k, x in enumerate((-5.3, 4.8)):
            crate(b, (x, 0, 4.5), 1.2, 1.+.3*k, k)
        # Regional fabric emblem, gutter brackets, lamps and shop tools.
        b.current_material = CLOTH
        color = np.array(((.27, .49, .49), (.67, .36, .19), (.55, .52, .31))[record["appearance"] % 3])
        b.quad((-1, 4.8, -6.4), (1.8, 4.8, -6.4), (1.6, 4.1, -6.6), (-1.1, 4.2, -6.6), color)
        b.current_material = METAL
        for x in (-2.7, 3.7):
            b.tube((x, 3.4, -6.2), (x, 3.4, -7), .045, IRON)
            b.box((x, 3.1, -7), (.24, .45, .24), (.95, .67, .32), emissive=1.3)
    # A modeled yard lamp corresponds to an actual presentation light source.
    b.current_material = METAL
    b.tube((-9.5, 0, -10), (-9.5, 6.2, -10), .09, IRON)
    b.box((-9.5, 6., -10), (.7, .35, .7), (.95, .72, .42), emissive=2.)
    b.box((-9.5, 6.3, -10), (.9, .12, .9), PATINA)
    for dx, dz in ((-.32, -.32), (-.32, .32), (.32, -.32), (.32, .32)):
        b.tube((-9.5+dx, 5.8, -10+dz), (-9.5+dx, 6.3, -10+dz), .025, IRON)
    # Move the entire local asset without changing its stable generated details.
    p = np.asarray(record["pos"], np.float32)
    for i in range(len(b.verts)):
        b.verts[i] = b.verts[i]+p
    return b


def architecture_meshes(records, batch):
    for record in records:
        b = _building(record)
        for name in ("verts", "colors", "emissive", "ids", "normals", "material_ids"):
            getattr(batch, name).extend(getattr(b, name))


def engine_meshes(env, b):
    p = env.engine_pos
    b.current_material = METAL
    stopped = env.engine_disabled
    phase = env.engine_phase()
    color = (.18, .68, .46) if stopped else (.95, .32, .12) if phase == "discharge" else (.32, .64, .92)
    # A turbine cage, concentric conductors, articulated shutters and exposed ribs.
    for y, radius in ((4, 8.2), (9, 8.7), (15.7, 10.2), (20, 5.5)):
        ring(b, p+(0, y, 0), radius, .18, COPPER)
    for a in np.arange(16)*np.pi/8:
        lower = p+(8.2*np.cos(a), 1, 8.2*np.sin(a))
        upper = p+(5.5*np.cos(a), 20, 5.5*np.sin(a))
        b.tube(lower, upper, .13, PATINA)
        for h in (5, 10, 16):
            b.box(p+(8.9*np.cos(a), h, 8.9*np.sin(a)), (.22, .6, .22), color, emissive=.7)
    angle = 0 if stopped else env.elapsed*1.4
    for a in np.arange(8)*np.pi/4+angle:
        c, s = np.cos(a), np.sin(a)
        b.quad(p+(c*2, 18, s*2), p+(c*9-s*.6, 18, s*9+c*.6),
               p+(c*9+s*.6, 19.2, s*9-c*.6), p+(c*2, 19, s*2), PATINA)
    for dx, dz in ((-23, -23), (-23, 23), (23, -23), (23, 23)):
        q = p+(dx, 0, dz)
        for side in (-2.06, 3.06):
            for y in np.arange(1., 16., 1.2):
                for offset in np.arange(-1.5, 3., 1.):
                    b.box(q+(offset, y, side), (.93, 1.10, .08), PATINA*.7)
                    b.box(q+(side, y, offset), (.08, 1.10, .93), PATINA*.7)
            for offset in (-1.7, 2.7):
                b.tube(q+(offset, .2, side), q+(offset, 16.5, side), .085, COPPER)
                b.tube(q+(side, .2, offset), q+(side, 16.5, offset), .085, COPPER)
        for y in range(2, 16, 2):
            ring(b, q+(.5, y, .5), 3.1, .10, IRON)
        pipe(b, [q+(0, 16.7, 0), q+(0, 19, 0), p+(dx*.55, 19, dz*.55), p+(dx*.55, 13, dz*.55)], .35)
        for a in np.arange(4)*np.pi/2:
            b.tube(q+(np.cos(a)*2.7, 1, np.sin(a)*2.7), q+(np.cos(a)*2.7, 16, np.sin(a)*2.7), .11, COPPER)
    for side in (-5.05, 6.05):
        for y in (2., 5., 8., 11.):
            for offset in (-3.5, -.5, 2.5, 5.):
                b.box(p+(offset, y, side), (2.3, 2.6, .09), IRON)
                b.box(p+(side, y, offset), (.09, 2.6, 2.3), IRON)
        for offset in (-4., 0., 4.):
            b.tube(p+(offset, .5, side), p+(offset, 12., side), .075, COPPER)
    for i, q in enumerate(env.lock_positions):
        b.box(q+(0, -.45, 0), (1.4, 1.1, .8), IRON)
        status = (.2, .8, .5) if env.locks[i] else color
        wheel(b, q+(0, .35, -.3), .4, color=COPPER)
        for k in range(4):
            b.box(q+(-.45+k*.3, .2, -.46), (.11, .13, .04), status, emissive=1.)
    c = env.cargo["grounding_cartridge"]["pos"]
    for x in (-.6, .6):
        b.tube(c+(x, .3, -.55), c+(x, 1.7, -.55), .18, COPPER)
        for y in np.arange(.5, 1.6, .16):
            ring(b, c+(x, y, -.55), .22, .035, PATINA, segments=16)
    b.current_material = STONE
