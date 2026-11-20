"""Usable ingredients for the lemon-tart kitchen: `LemonTart(usable_ingredients=True)`.

The default task only moves ingredients about. With this option:

- flour, sugar and egg are open containers of loose grains that pour out when
  the container is tipped;
- butter and lemon are two halves welded together; bringing a knife blade down
  into the seam releases the weld. `cut_rule="lenient"` (default) asks for a
  rough cut, `cut_rule="strict"` for a blade that follows the seam closely;
- a knife lies in a randomly chosen drawer, and the knife block is removed
  from the counter;
- the tart pan is a fluted metal tin, instead of the plate used by the
  default task.

Nothing here changes the scored milestones. Run this file to rebuild the
generated assets in `owb/assets/lemon_tart/`.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "lemon_tart"

# ---- loose grains ----
GRAIN_RADIUS = 0.006
GRAIN_PITCH = 0.0125  # spacing of the grains when a container is filled at reset
GRAIN_MASS = 0.0005
WALL = 0.004
# name -> container model, grains per axis (x, y, layers), grain colour
GRAINS = {
    "flour": dict(mjcf="flour_bag_open.xml", grid=(4, 4, 3), rgba="0.95 0.91 0.80 1"),
    "sugar": dict(mjcf="sugar_bag_open.xml", grid=(3, 3, 4), rgba="0.85 0.58 0.25 1"),
    "egg": dict(mjcf="egg_cup_open.xml", grid=(3, 3, 3), rgba="0.98 0.74 0.10 1"),
}

# ---- things that can be cut in half ----
# The seam is the local xz plane through the object's origin; half A (y < 0) is
# the object itself, half B (y > 0) is a second free body welded to it.
BUTTER = (0.0135, 0.066, 0.015)  # half-sizes of the whole stick
LEMON = (0.030, 0.045, 0.0275)  # semi-axes of the whole lemon
CUTTABLE = {
    "butter": dict(mjcf="butter_halves.xml", height=2 * BUTTER[2]),
    "lemon": dict(mjcf="lemon_halves.xml", height=2 * LEMON[2]),
}
# Collision bits. The blade (contype 1, conaffinity 0) passes through a half
# (contype 3, conaffinity 2) until the cut, when the half's conaffinity becomes 3.
# Everything else in the kitchen is (1, 1) or the fingers' (0, 1), and collides
# with both the blade and the halves throughout.
HALF_CONTYPE, HALF_UNCUT, HALF_CUT = 3, 2, 3

# ---- knife ----
KNIFE_MJCF = "knife.xml"
TART_PAN_MJCF = "tart_pan.xml"
# outer radius, wall height, bottom thickness. The whole tin is shorter than
# FLAT_DISH_HEIGHT in lemon_tart_env, so the dishwasher still props it.
TART_PAN_RADIUS = 0.068
TART_PAN_HEIGHT = 0.024
TART_PAN_BOTTOM = 0.003
HANDLE = (0.011, 0.05, 0.014)  # half-sizes; the handle is centred on the origin
BLADE = (0.001, 0.065, 0.014)  # half-sizes; the blade continues along +y
BLADE_Y = HANDLE[1] + BLADE[1]
# the cutting edge, in the knife's frame
EDGE = (np.array([0.0, HANDLE[1], -BLADE[2]]), np.array([0.0, HANDLE[1] + 2 * BLADE[1], -BLADE[2]]))

# Two versions of the cutting rule, chosen with `LemonTart(cut_rule=...)`.
# "lenient" (the default): the blade has to be at the object, roughly along the
# seam, and come down into it. "strict": it has to follow the seam closely and
# go nearly all the way down.
CUT_RULES = {
    "lenient": dict(
        angle_deg=45.0,  # blade plane within this angle of the seam plane
        offset_mm=25.0,  # blade edge within this distance of the seam plane
        centre_mm=30.0,  # the edge passes within this distance of the object's axis
        height_frac=0.6,  # the edge is down to this fraction of the object's height
    ),
    "strict": dict(angle_deg=20.0, offset_mm=8.0, centre_mm=12.0, height_frac=0.3),
}
DEFAULT_CUT_RULE = "lenient"

# centre of the second half (B) in its own frame; the first half mirrors it in y
HALF_CENTRE = {"butter": np.array([0.0, BUTTER[1] / 2, 0.0]), "lemon": np.array([0.0, 0.375 * LEMON[1], 0.0])}


def mjcf(name):
    return str(ASSETS / name)


def is_prep_asset(groups):
    return isinstance(groups, str) and groups.startswith(str(ASSETS))


class PrepKit:
    """Builds and steps the usable ingredients for one LemonTart environment."""

    def __init__(self, env, cut_rule=DEFAULT_CUT_RULE):
        if cut_rule not in CUT_RULES:
            raise ValueError(f"cut_rule must be one of {tuple(CUT_RULES)}, got {cut_rule!r}")
        self.env = env
        self.cut_rule = cut_rule
        self.limits = CUT_RULES[cut_rule]
        self.cut = {name: False for name in CUTTABLE}
        self.cut_step = {}

    # ---------- model ----------
    def sources(self):
        """Ingredient name -> model file that replaces the sampled RoboCasa object."""
        out = {name: mjcf(spec["mjcf"]) for name, spec in GRAINS.items()}
        out.update({name: mjcf(spec["mjcf"]) for name, spec in CUTTABLE.items()})
        out["tart_pan"] = mjcf(TART_PAN_MJCF)
        return out

    def remove_knife_block(self):
        """Take the knife block out of the scene.

        The fixture stays in the environment's bookkeeping, so that fixture and
        object sampling draw the same random numbers as the default task; its
        body is parked under the floor, where nothing can see or touch it.
        """
        env = self.env
        roots = {f.root_body for n, f in env.fixtures.items() if n.startswith("knife_block")}
        for body in env.model.worldbody.iter("body"):
            if body.get("name") in roots:
                body.set("pos", "0 0 -5")

    def add_bodies(self):
        """Grains, second halves and their welds, appended to the task model."""
        self.remove_knife_block()
        model = self.env.model
        world = model.worldbody
        inertia = 0.4 * GRAIN_MASS * GRAIN_RADIUS**2
        for name, spec in GRAINS.items():
            for i in range(int(np.prod(spec["grid"]))):
                body = ET.SubElement(world, "body", name=f"{name}_grain_{i}", pos="0 0 0")
                # Three slides, no rotation, frictionless: a pile of these
                # flows like a liquid and costs the solver far less than
                # rolling spheres with friction. The damping brings a spilled
                # grain to rest.
                for axis, vec in zip("xyz", ("1 0 0", "0 1 0", "0 0 1")):
                    ET.SubElement(
                        body, "joint", name=f"{name}_grain_{i}_{axis}", type="slide", axis=vec, damping="0.002"
                    )
                # robosuite only counts group-0 geoms towards inertia, and
                # these are drawn (group 1), so the inertia is given explicitly
                ET.SubElement(
                    body, "inertial", pos="0 0 0", mass=str(GRAIN_MASS), diaginertia=f"{inertia} {inertia} {inertia}"
                )
                ET.SubElement(
                    body,
                    "geom",
                    name=f"{name}_grain_{i}_g",
                    type="sphere",
                    size=str(GRAIN_RADIUS),
                    rgba=spec["rgba"],
                    group="1",
                    condim="1",
                    priority="1",
                    solref="0.006 1",
                )
        ET.SubElement(model.asset, "mesh", name="lemon_half_b_mesh", file=mjcf("lemon_half_b.obj"))
        half = dict(contype=str(HALF_CONTYPE), conaffinity=str(HALF_UNCUT), density="100", friction="0.95 0.3 0.1")
        for name in CUTTABLE:
            body = ET.SubElement(world, "body", name=f"{name}_half_b", pos="0 0 0", gravcomp="0")
            ET.SubElement(body, "freejoint", name=f"{name}_half_b_joint")
            if name == "butter":
                size = f"{BUTTER[0]} {BUTTER[1] / 2} {BUTTER[2]}"
                pos = f"0 {BUTTER[1] / 2} 0"
                ET.SubElement(body, "geom", name="butter_half_b_col", type="box", size=size, pos=pos, group="0", **half)
                ET.SubElement(
                    body, "geom", type="box", size=size, pos=pos, group="1", contype="0", conaffinity="0",
                    rgba=BUTTER_RGBA, mass="0",
                )
            else:
                ET.SubElement(
                    body, "geom", name="lemon_half_b_col", type="mesh", mesh="lemon_half_b_mesh", group="0",
                    condim="6", **half,
                )
                ET.SubElement(
                    body, "geom", type="mesh", mesh="lemon_half_b_mesh", group="1", contype="0", conaffinity="0",
                    rgba=LEMON_RGBA, mass="0",
                )
                ET.SubElement(
                    body, "geom", type="cylinder", size=f"{LEMON[2] - 0.002} 0.0004", pos="0 0.0002 0",
                    quat="0.7071068 0.7071068 0 0", group="1", contype="0", conaffinity="0", rgba=LEMON_FLESH_RGBA,
                    mass="0",
                )
            ET.SubElement(
                model.equality,
                "weld",
                name=f"{name}_weld",
                body1=self.env.objects[name].root_body,
                body2=f"{name}_half_b",
                relpose="0 0 0 1 0 0 0",
                # switched on by reset(), once the second half has been put in
                # place; before that it would drag the object to the origin
                active="false",
            )

    # ---------- reset ----------
    def _raw(self):
        sim = self.env.sim
        return sim.model._model, sim.data._data

    def _pose(self, name):
        data = self.env.sim.data
        bid = self.env.obj_body_id[name]
        return np.array(data.body_xpos[bid]), np.array(data.body_xmat[bid]).reshape(3, 3)

    def _half_geoms(self, name):
        model = self.env.sim.model
        root = model.body_name2id(self.env.objects[name].root_body)
        other = model.body_name2id(f"{name}_half_b")
        return [
            g
            for g in range(model.ngeom)
            if model.geom_contype[g] == HALF_CONTYPE
            and (model.body_rootid[model.geom_bodyid[g]] == root or model.geom_bodyid[g] == other)
        ]

    def _set_cut(self, name, cut):
        import mujoco

        model, data = self._raw()
        eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{name}_weld")
        data.eq_active[eq] = 0 if cut else 1
        for g in self._half_geoms(name):
            model.geom_conaffinity[g] = HALF_CUT if cut else HALF_UNCUT
        self.cut[name] = bool(cut)

    def reset(self):
        """Fill the containers and join the halves. Runs at reset only."""
        import robosuite.utils.transform_utils as T

        sim = self.env.sim
        model, data = sim.model, sim.data
        sim.forward()
        for name, spec in GRAINS.items():
            obj = self.env.objects[name]
            pos, rot = self._pose(name)
            nx, ny, nz = spec["grid"]
            floor = obj.bottom_offset[2] + WALL + GRAIN_RADIUS + 0.001
            i = 0
            for k in range(nz):
                for a in range(nx):
                    for b in range(ny):
                        local = np.array(
                            [
                                (a - 0.5 * (nx - 1)) * GRAIN_PITCH,
                                (b - 0.5 * (ny - 1)) * GRAIN_PITCH,
                                floor + k * GRAIN_PITCH,
                            ]
                        )
                        world = pos + rot @ local
                        for axis, value in zip("xyz", world):
                            joint = f"{name}_grain_{i}_{axis}"
                            data.qpos[model.get_joint_qpos_addr(joint)] = value
                            data.qvel[model.get_joint_qvel_addr(joint)] = 0.0
                        i += 1
        for name in CUTTABLE:
            pos, rot = self._pose(name)
            adr = model.get_joint_qpos_addr(f"{name}_half_b_joint")[0]
            vadr = model.get_joint_qvel_addr(f"{name}_half_b_joint")[0]
            data.qpos[adr : adr + 3] = pos
            data.qpos[adr + 3 : adr + 7] = T.convert_quat(T.mat2quat(rot), to="wxyz")
            data.qvel[vadr : vadr + 6] = 0.0
            self._set_cut(name, False)
        self.cut_step = {}
        sim.forward()

    # ---------- every step ----------
    def blade_edge(self):
        """World end points of the knife's cutting edge, and the blade's normal."""
        pos, rot = self._pose("knife")
        return pos + rot @ EDGE[0], pos + rot @ EDGE[1], rot[:, 0]

    def cut_report(self, name):
        """How the blade sits relative to the seam of `name`, and the limits it has to meet."""
        a, b, normal = self.blade_edge()
        spec = CUTTABLE[name]
        pos, rot = self._pose(name)
        seam = rot[:, 1]
        ab = b - a
        t = float(((pos - a)[:2] @ ab[:2]) / max(ab[:2] @ ab[:2], 1e-12))
        q = a + np.clip(t, 0.0, 1.0) * ab  # the point of the edge nearest the object's axis, seen from above
        return {
            "angle_deg": float(np.degrees(np.arccos(np.clip(abs(seam @ normal), 0.0, 1.0)))),
            "max_angle_deg": self.limits["angle_deg"],
            "offset_mm": 1000 * float(max(abs((a - pos) @ seam), abs((b - pos) @ seam))),
            "max_offset_mm": self.limits["offset_mm"],
            "along_edge": t,  # 0 at the heel of the blade, 1 at its tip; must be inside 0.05..0.95
            "centre_mm": 1000 * float(np.linalg.norm((q - pos)[:2])),
            "max_centre_mm": self.limits["centre_mm"],
            "height_frac": float((q[2] - (pos[2] - 0.5 * spec["height"])) / spec["height"]),
            "max_height_frac": self.limits["height_frac"],
        }

    def update(self):
        """Release a weld once the blade has come down through its seam."""
        if all(self.cut.values()):
            return
        for name in CUTTABLE:
            if self.cut[name]:
                continue
            r = self.cut_report(name)
            if (
                r["angle_deg"] <= r["max_angle_deg"]
                and r["offset_mm"] <= r["max_offset_mm"]
                and 0.05 < r["along_edge"] < 0.95
                and r["centre_mm"] <= r["max_centre_mm"]
                and r["height_frac"] <= r["max_height_frac"]
            ):
                self._set_cut(name, True)
                self.cut_step[name] = int(self.env.timestep)

    # ---------- reading ----------
    def grain_positions(self):
        model, data = self.env.sim.model, self.env.sim.data
        return {
            name: np.array(
                [data.body_xpos[model.body_name2id(f"{name}_grain_{i}")] for i in range(int(np.prod(spec["grid"])))]
            )
            for name, spec in GRAINS.items()
        }

    def half_poses(self, name):
        """(position, rotation) of the two halves of a cuttable ingredient."""
        data = self.env.sim.data
        bid = self.env.sim.model.body_name2id(f"{name}_half_b")
        return self._pose(name), (np.array(data.body_xpos[bid]), np.array(data.body_xmat[bid]).reshape(3, 3))

    def state(self):
        gaps = {}
        for name in CUTTABLE:
            (pa, _), (pb, _) = self.half_poses(name)
            gaps[name] = float(np.linalg.norm(pa - pb))
        return {"cut": dict(self.cut), "cut_step": dict(self.cut_step), "half_gap": gaps, "cut_rule": self.cut_rule}


# ---------- generated assets ----------
BUTTER_RGBA = "0.98 0.90 0.50 1"
LEMON_RGBA = "0.98 0.86 0.12 1"
LEMON_FLESH_RGBA = "0.99 0.96 0.70 1"

HEAD = """<mujoco model="model">
  <!-- Generated by owb/tasks/lemon_tart_prep.py for LemonTart(usable_ingredients=True). -->
  <compiler angle="radian"/>
  <asset>{asset}</asset>
  <default>
    <default class="visual">
      <geom conaffinity="0" contype="0" group="1"/>
    </default>
    <default class="collision">
      <geom group="0" rgba="0.5 0 0 0.5"/>
    </default>
    <default class="region">
      <geom group="1" conaffinity="0" contype="0" rgba="0 1 0 0" mass="0"/>
    </default>
  </default>
  <worldbody>
    <body>
      <body name="object">
{body}
      </body>
    </body>
  </worldbody>
  <actuator/>
</mujoco>
"""


def _fmt(values):
    return " ".join(f"{v:.5f}" for v in values)


def _write(name, rows, asset=""):
    (ASSETS / name).write_text(HEAD.format(asset=asset, body="\n".join("        " + row for row in rows)))


def _bbox(half, pos=(0, 0, 0)):
    return f'<geom class="region" name="reg_bbox" type="box" pos="{_fmt(pos)}" size="{_fmt(half)}"/>'


# condim 4 adds torsional friction, so that a container or the knife pinched
# between two fingers does not swing about the line between them
GRIP = ' condim="4"'


def _box(pos, half, rgba, extra=GRIP):
    return [
        f'<geom class="visual" type="box" pos="{_fmt(pos)}" size="{_fmt(half)}" rgba="{rgba}"/>',
        f'<geom class="collision" type="box" pos="{_fmt(pos)}" size="{_fmt(half)}"{extra}/>',
    ]


def _container(name, width, height, rgba, band):
    hw, hh = width / 2, height / 2
    walls = [
        ((0, 0, -hh + WALL / 2), (hw, hw, WALL / 2)),
        ((hw - WALL / 2, 0, 0), (WALL / 2, hw, hh)),
        ((-hw + WALL / 2, 0, 0), (WALL / 2, hw, hh)),
        ((0, hw - WALL / 2, 0), (hw - WALL, WALL / 2, hh)),
        ((0, -hw + WALL / 2, 0), (hw - WALL, WALL / 2, hh)),
    ]
    rows = [_bbox((hw, hw, hh))]
    for pos, half in walls:
        rows += _box(pos, half, rgba)
    for pos, half in walls[1:]:  # a coloured band round the outside, visual only
        out = [c + np.sign(c) * 0.0006 for c in pos[:2]]
        grow = (0.0004 if half[0] > half[1] else 0.0, 0.0004 if half[1] > half[0] else 0.0)
        rows.append(
            f'<geom class="visual" type="box" pos="{_fmt((out[0], out[1], 0.0))}" '
            f'size="{_fmt((half[0] + grow[0], half[1] + grow[1], hh * 0.3))}" rgba="{band}"/>'
        )
    _write(name, rows)


def _lemon_mesh(sign, rings=8, segments=20):
    """Half an ellipsoid on the +y (sign=1) or -y side, closed by a flat face at y = 0."""
    a, b, c = LEMON
    verts = [(0.0, sign * b, 0.0)]
    for i in range(1, rings + 1):
        theta = 0.5 * np.pi * i / rings
        for j in range(segments):
            phi = 2 * np.pi * j / segments
            verts.append((a * np.sin(theta) * np.cos(phi), sign * b * np.cos(theta), c * np.sin(theta) * np.sin(phi)))
    verts.append((0.0, 0.0, 0.0))
    centre = len(verts) - 1
    ring = lambda i, j: 1 + (i - 1) * segments + j % segments
    faces = []
    for j in range(segments):
        faces.append((0, ring(1, j), ring(1, j + 1)))
        for i in range(1, rings):
            faces.append((ring(i, j), ring(i + 1, j), ring(i + 1, j + 1)))
            faces.append((ring(i, j), ring(i + 1, j + 1), ring(i, j + 1)))
        faces.append((centre, ring(rings, j + 1), ring(rings, j)))
    v = np.array(verts)
    volume = sum(np.dot(v[f[0]], np.cross(v[f[1]], v[f[2]])) for f in faces)
    if volume < 0:  # make every face point outwards
        faces = [(f[0], f[2], f[1]) for f in faces]
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}" for f in faces]
    return "\n".join(lines) + "\n"


def build_assets():
    ASSETS.mkdir(parents=True, exist_ok=True)
    _container("flour_bag_open.xml", 0.060, 0.090, "0.82 0.70 0.50 1", "0.55 0.25 0.15 1")
    _container("sugar_bag_open.xml", 0.052, 0.075, "0.92 0.94 0.98 1", "0.20 0.40 0.75 1")
    _container("egg_cup_open.xml", 0.054, 0.065, "0.80 0.88 0.92 1", "0.95 0.60 0.10 1")

    half = f' contype="{HALF_CONTYPE}" conaffinity="{HALF_UNCUT}"'
    rows = [_bbox(BUTTER)]
    rows += _box((0, -BUTTER[1] / 2, 0), (BUTTER[0], BUTTER[1] / 2, BUTTER[2]), BUTTER_RGBA, GRIP + half)
    _write("butter_halves.xml", rows)

    (ASSETS / "lemon_half_a.obj").write_text(_lemon_mesh(-1.0))
    (ASSETS / "lemon_half_b.obj").write_text(_lemon_mesh(1.0))
    rows = [
        _bbox(LEMON),
        f'<geom class="visual" type="mesh" mesh="lemon_half_a" rgba="{LEMON_RGBA}"/>',
        f'<geom class="visual" type="cylinder" size="{LEMON[2] - 0.002:.5f} 0.0004" pos="0 -0.0002 0" '
        f'quat="0.7071068 0.7071068 0 0" rgba="{LEMON_FLESH_RGBA}"/>',
        f'<geom class="collision" type="mesh" mesh="lemon_half_a" condim="6"{half}/>',
    ]
    _write("lemon_halves.xml", rows, asset='\n    <mesh file="lemon_half_a.obj" name="lemon_half_a"/>\n  ')

    rows = [_bbox((HANDLE[0], HANDLE[1] + BLADE[1], HANDLE[2]), (0, BLADE[1], 0))]
    rows += _box((0, 0, 0), HANDLE, "0.20 0.13 0.08 1")
    rows += _box((0, BLADE_Y, 0), BLADE, "0.80 0.82 0.85 1", GRIP + ' contype="1" conaffinity="0"')
    _write(KNIFE_MJCF, rows)
    _write(TART_PAN_MJCF, _tart_pan_rows())


def _tart_pan_rows(flutes=20):
    """A round tin: a disk floor and a wall of overlapping flutes."""
    tin = "0.72 0.74 0.78 1"
    inside = "0.55 0.57 0.60 1"
    r = TART_PAN_RADIUS
    h = TART_PAN_HEIGHT
    b = TART_PAN_BOTTOM
    z0 = -h / 2
    rows = [_bbox((r + 0.004, r + 0.004, h / 2))]
    rows.append(
        f'<geom class="visual" type="cylinder" pos="0 0 {z0 + b / 2:.5f}" '
        f'size="{r - 0.006:.5f} {b / 2:.5f}" rgba="{inside}"/>'
    )
    rows.append(
        f'<geom class="collision" type="cylinder" pos="0 0 {z0 + b / 2:.5f}" '
        f'size="{r - 0.006:.5f} {b / 2:.5f}" rgba="{inside}"{GRIP}/>'
    )
    wall_h = (h - b) / 2
    wall_z = z0 + b + wall_h
    thick = 0.0035
    half_arc = np.pi * (r - thick) / flutes + 0.001
    for i in range(flutes):
        a = 2 * np.pi * i / flutes
        pos = ((r - thick) * np.cos(a), (r - thick) * np.sin(a), wall_z)
        quat = (np.cos(a / 2), 0.0, 0.0, np.sin(a / 2))
        half = (thick, half_arc, wall_h)
        rows.append(
            f'<geom class="visual" type="box" pos="{_fmt(pos)}" quat="{_fmt(quat)}" '
            f'size="{_fmt(half)}" rgba="{tin}"/>'
        )
        rows.append(
            f'<geom class="collision" type="box" pos="{_fmt(pos)}" quat="{_fmt(quat)}" '
            f'size="{_fmt(half)}" rgba="{tin}"{GRIP}/>'
        )
    return rows


if __name__ == "__main__":
    build_assets()
    print(f"wrote {sorted(p.name for p in ASSETS.iterdir())}")
