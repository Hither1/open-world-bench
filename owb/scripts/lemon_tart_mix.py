"""Lemon tart: put the dry ingredients into the mixing bowl.

The scored task only checks where things are, so the reference trajectory
never uses an ingredient. This one does: the robot takes the bowl, the flour
and the sugar out of their drawers, drops the sugar cube into the bowl and
tips the flour bag over it.

Same rules as `lemon_tart_reference`: nothing but the robot's own 12-dim
actions. The flour bag is one rigid body, so no flour comes out of it; the
script reports how far the bag was tipped while its mouth was over the bowl.
The sugar cube really ends up inside the bowl.

    python -m owb.scripts.lemon_tart_mix --seed 1

With `--usable` the environment is built with
`LemonTart(usable_ingredients=True)` (see `owb/tasks/lemon_tart_prep.py`):
flour, sugar and egg are open containers of loose grains, butter and lemon can
be cut in half, and a knife lies in one of the drawers. The robot pours the
sugar, the flour and the egg into the bowl, then cuts the butter and the lemon.

    python -m owb.scripts.lemon_tart_mix --seed 1 --usable
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from owb.scripts.lemon_tart_reference import (
    REPO,
    Chef,
    Driver,
    Failed,
    FollowCamera,
    VideoSink,
    grip_R,
    rotvec,
)
from owb.tasks import lemon_tart_prep as prep
from owb.tasks.lemon_tart_env import LemonTart


def axis_rot(axis, angle):
    """Rotation by `angle` about the unit vector `axis` (Rodrigues)."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


class MixChef(Chef):
    # on the counter above the dishwasher: the widest free stretch of worktop
    BOWL_SPOT = (1.95, -0.45)
    FLOUR_REST = (2.14, -0.52)  # where the bag is put down after pouring
    STANCE_Y = -0.95
    POUR_ROLL = np.deg2rad(140.0)
    POUR_HEIGHT = 0.21  # bag centre above the counter while pouring

    # usable variant: both bags are gripped from the side, and the hand is
    # 20 cm wide, so they stand further apart and are put down one behind the other
    POUR_SPOTS = {
        "flour": (0.76, -0.50),
        "sugar": (0.60, -0.56),
        # right-hand counter: the egg cup, then the two things to cut, with
        # room to their left for the hand that holds the knife
        "egg": (3.60, -0.52),
        "lemon": (3.80, -0.50),
        "butter": (4.05, -0.52),
    }
    # the emptied containers stand in a column, first poured at the back
    POUR_REST = {"sugar": (2.10, -0.36), "flour": (2.10, -0.48), "egg": (2.10, -0.60)}
    KNIFE_SPOT = (0.60, -0.55)  # handle; the blade points along +x
    KNIFE_REST = (3.56, -0.61)
    EDGE_MID = np.array([0.0, 0.115, -0.014])  # middle of the cutting edge, knife frame

    def __init__(self, driver, usable=False):
        super().__init__(driver)
        self.usable = usable
        self.pour = {name: {"max_tilt_deg": 0.0, "steps_past_90": 0} for name in ("flour", "sugar", "egg")}
        self.taking = None
        self.lifted = {"butter": 0.0, "lemon": 0.0}  # how high one half of each was lifted, metres
        if usable:
            self.SPOTS = dict(self.SPOTS, **self.POUR_SPOTS)

    # ---------- measurements ----------
    def points_in_bowl(self, points):
        """Which points are inside the bowl's rim circle and between its base and rim."""
        d = self.d
        bowl = self.env.objects["bowl"]
        rel = (np.atleast_2d(points) - d.obj_pos("bowl")) @ d.obj_R("bowl")
        radius = 0.5 * bowl.size[0]
        return (np.hypot(rel[:, 0], rel[:, 1]) < radius - 0.01) & (rel[:, 2] > -0.5 * bowl.size[2]) & (
            rel[:, 2] < 0.5 * bowl.size[2] + 0.01
        )

    def in_bowl(self, name):
        return bool(self.points_in_bowl(self.d.obj_pos(name))[0])

    def grains_in_bowl(self):
        """{ingredient: (grains in the bowl, grains in total)} for the usable variant."""
        return {
            name: (int(self.points_in_bowl(points).sum()), len(points))
            for name, points in self.env.grain_positions().items()
        }

    def mouth(self, name):
        """Centre of the bag's top face, and the bag's tilt from upright."""
        d = self.d
        R = d.obj_R(name)
        top = d.obj_pos(name) + R @ np.asarray(self.env.objects[name].top_offset, float)
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))
        return top, tilt

    def watch_pour(self, name="flour"):
        top, tilt = self.mouth(name)
        bowl = self.d.obj_pos("bowl")
        if np.hypot(top[0] - bowl[0], top[1] - bowl[1]) < 0.5 * self.env.objects["bowl"].size[0] and top[2] > bowl[2]:
            self.pour[name]["max_tilt_deg"] = max(self.pour[name]["max_tilt_deg"], tilt)
            if tilt > 90.0:
                self.pour[name]["steps_past_90"] += 1

    def drawer_heading(self, drawer, pos, half_hand=0.105):
        """The open bags are square: grip one squarely, along whichever of its
        own axes points most nearly away from the robot."""
        if self.taking is None:
            return super().drawer_heading(drawer, pos, half_hand)
        Ro = self.d.obj_R(self.taking)
        if self.taking == "knife":  # along the knife, so the fingers straddle the handle
            return (Ro[0, 1], Ro[1, 1])
        return tuple(max((s * Ro[:2, i] for i in (0, 1) for s in (1.0, -1.0)), key=lambda v: v[1]))

    # ---------- stages ----------
    def stage_bowl(self):
        """Bowl out of its drawer and onto the counter."""
        d = self.d
        drawer = self.env.evaluator_state()["homes"]["bowl"]
        self.open_drawer(drawer, pull=0.52)
        handle, _ = self.drawer_state(drawer)
        self.home()
        d.base_goto([handle[0], handle[1] - 0.20], np.pi / 2)
        # pinch the wall of the bowl on its -x side, from straight above
        off = d.obj_R("bowl") @ np.array([-0.092, 0.0, 0.024])
        self.pick("bowl", pitch_deg=90, heading=(0.0, 1.0), offset=tuple(off), lift=0.05, back=0.10, hold_tol=0.16)
        d.arm_line(d.eef_pos() + np.array([0, -0.06, 0.06]), speed=0.012, max_steps=30, tol=0.02, quiet=True)
        # the bowl hangs from its rim, up to 13 cm below the fingers: carry it
        # high enough to clear the counter's edge
        d.arm_line([d.eef_pos()[0], d.eef_pos()[1], 1.24], speed=0.015, max_steps=90, tol=0.02, quiet=True)
        if d.obj_pos("bowl")[2] < 0.95:
            raise Failed("lost the bowl while lifting it out of the drawer")
        self.grasp_off = d.obj_pos("bowl") - d.eef_pos()
        self.carry(fwd=0.40)
        # the bowl hangs to the hand's +x side: stand so that it lands on the spot
        self.drive(self.BOWL_SPOT[0] - self.grasp_off[0], self.STANCE_Y)
        self.grasp_off = d.obj_pos("bowl") - d.eef_pos()
        z = self.COUNTER_Z - self.env.objects["bowl"].bottom_offset[2] + 0.004
        self.place("bowl", [self.BOWL_SPOT[0], self.BOWL_SPOT[1], z], above=0.14, retreat=(0, 0, 0.12))
        d.wait(10)
        up = d.obj_R("bowl")[2, 2]
        if d.obj_pos("bowl")[2] < 0.93 or up < 0.98:
            raise Failed(f"the bowl is not sitting upright on the counter (up {up:.2f})")
        self.home()
        self.close_drawer(drawer)

    def stage_dry_out(self):
        """Flour and sugar out of their drawers, onto the counter beside them."""
        homes = self.env.evaluator_state()["homes"]
        for name in ("flour", "sugar"):
            self.taking = name if self.usable else None
            self.take_from_drawer(homes[name], name, self.SPOTS[name])
        self.taking = None

    def stage_sugar(self):
        """Carry the sugar cube over and let it go inside the bowl."""
        if self.usable:
            return self.pour_bag("sugar", self.POUR_REST["sugar"])
        d = self.d
        p = d.obj_pos("sugar")
        d.base_goto([p[0] + 0.30, self.STANCE_Y], np.pi / 2)
        self.pick("sugar", pitch_deg=90, heading=(0.0, 1.0), lift=0.12, back=0.10)
        d.arm_line([d.eef_pos()[0], d.eef_pos()[1], 1.12], speed=0.015, max_steps=40, tol=0.02, quiet=True)
        self.carry(fwd=0.40)
        bowl = d.obj_pos("bowl")
        self.drive(bowl[0], self.STANCE_Y)
        if not self.held("sugar"):
            raise Failed("dropped the sugar on the way to the bowl")
        bowl = d.obj_pos("bowl")
        off = d.obj_pos("sugar") - d.eef_pos()
        over = np.array([bowl[0], bowl[1], bowl[2] + 0.16]) - off
        d.arm_to(over, self.grasp_R, max_steps=150, tol=0.015)
        # fingertips just under the rim, so the cube cannot bounce out
        d.arm_line(over - np.array([0, 0, 0.115]), self.grasp_R, speed=0.008, max_steps=50, tol=0.008)
        d.set_gripper(False)
        d.wait(15)
        d.arm_line(d.eef_pos() + np.array([0, 0, 0.16]), self.grasp_R, speed=0.012, max_steps=40, tol=0.02, quiet=True)
        if not self.in_bowl("sugar"):
            raise Failed(f"the sugar is not in the bowl: {np.round(d.obj_pos('sugar'), 3)}")
        d.say(f"sugar is in the bowl at {np.round(d.obj_pos('sugar'), 3)}")
        self.home()

    def stage_flour(self):
        if self.usable:
            return self.pour_bag("flour", self.POUR_REST["flour"])
        return self.pour_bag("flour", self.FLOUR_REST)

    def pour_bag(self, name, rest):
        """Grip the bag from the side, tip it over the bowl with the wrist, put it down at `rest`."""
        d = self.d
        p = d.obj_pos(name)
        # the usable containers are gripped a little above their middle, which
        # is easier from slightly closer in
        self.drive(p[0], max(-1.15, p[1] - 0.60) if self.usable else -1.15)
        # come in along whichever horizontal axis of the bag points away from the robot
        Ro = d.obj_R(name)
        heading = max((s * Ro[:2, i] for i in (0, 1) for s in (1.0, -1.0)), key=lambda v: v[1])
        R0 = self.pick(name, pitch_deg=30, heading=tuple(heading))
        self.carry(fwd=0.42, z=1.18)
        bowl = d.obj_pos("bowl")
        self.drive(bowl[0], self.STANCE_Y)
        if not self.held(name, 0.08):
            raise Failed(f"dropped the {name} on the way to the bowl")
        bowl = d.obj_pos("bowl")
        approach = R0[:, 2]
        # rolling the wrist about the approach axis tips the bag's top to one
        # side; start with the bag on the other side of the bowl's centre
        side = -1.0
        R1 = axis_rot(approach, side * self.POUR_ROLL) @ R0
        up1 = R1 @ R0.T @ np.array([0.0, 0.0, 1.0])  # where the bag's up axis ends
        reach = float(self.env.objects[name].top_offset[2])
        centre = np.array([bowl[0], bowl[1], self.COUNTER_Z + self.POUR_HEIGHT])
        centre[:2] -= up1[:2] * reach
        off0 = d.obj_pos(name) - d.eef_pos()
        start = centre - off0
        d.arm_to(start, R0, max_steps=150, tol=0.015)
        steps = 14 if self.usable else 7
        for k in range(1, steps + 1):
            Rk = axis_rot(approach, side * self.POUR_ROLL * k / steps) @ R0
            pk = centre - Rk @ R0.T @ off0
            d.arm_to(pk, Rk, max_steps=40, tol=0.015, rot_tol=0.12, quiet=True)
            self.watch_pour(name)
        for _ in range(60 if self.usable else 30):
            d.step()
            self.watch_pour(name)
        tilt = self.pour[name]["max_tilt_deg"]
        d.say(f"{name} tipped {tilt:.0f} deg over the bowl; grains in the bowl: {self.grains_in_bowl()}")
        if not self.held(name, 0.08):
            raise Failed(f"the {name} bag slipped out while pouring")
        for k in range(steps - 1, -1, -1):
            Rk = axis_rot(approach, side * self.POUR_ROLL * k / steps) @ R0
            pk = centre - Rk @ R0.T @ off0
            d.arm_to(pk, Rk, max_steps=40, tol=0.015, rot_tol=0.12, quiet=True)
        if tilt < 100.0:
            raise Failed(f"the {name} bag only tipped {tilt:.0f} deg")
        self.grasp_R = R0
        self.grasp_off = d.obj_pos(name) - d.eef_pos()
        z = self.COUNTER_Z - self.env.objects[name].bottom_offset[2] + 0.003
        self.place(name, [rest[0], rest[1], z])
        self.carry(z=1.15, fwd=0.40)
        self.home()


    # ---------- usable variant only ----------
    def pick(self, name, pitch_deg=30.0, heading=(0.0, 1.0), **kw):
        if heading == "long" and name in ("flour", "sugar", "egg"):
            # a square container has no long axis: come in along whichever of
            # its own axes points most nearly away from the robot
            Ro = self.d.obj_R(name)
            heading = tuple(max((s * Ro[:2, i] for i in (0, 1) for s in (1.0, -1.0)), key=lambda v: v[1]))
            # hold it above its middle, so that it hangs rather than balances
            off = np.asarray(kw.get("offset", (0.0, 0.0, 0.0)), float)
            if off[2] == 0.0:
                kw["offset"] = (off[0], off[1], 0.012)
            # The arm does not always get down to the grasp height. A grip on
            # the rim lets the container slip out later, so let go and try again.
            want = kw["offset"][2]
            for attempt in range(3):
                start = self.d.obj_pos(name).copy()
                R = super().pick(name, pitch_deg, heading, **kw)
                high = self.d.eef_pos()[2] - self.d.obj_pos(name)[2] - want
                if high < 0.015:
                    return R
                self.d.say(f"{name} gripped {1000 * high:.0f} mm too high, putting it back")
                self.place(name, start + np.array([0, 0, 0.003]), above=0.04)
                self.carry(fwd=0.40, z=max(self.d.eef_pos()[2], 1.05))
                self.home()
            raise Failed(f"could not get a low enough grip on the {name}")
        return super().pick(name, pitch_deg, heading, **kw)

    def fridge_take(self, name):
        if name != "egg" or not self.usable:
            return super().fridge_take(name)
        # The cup is on the top shelf. Go up for it, and come down gently
        # once it is out, so that the egg stays in the cup.
        d = self.d
        self.drive(min(4.48, d.obj_pos(name)[0] - 0.02), -1.12)
        d.torso_to(0.2)
        self.pick(name, self.FRIDGE_PITCH[name], heading="long")
        self.carry(fwd=0.42)
        for height in (0.15, 0.10, 0.05, 0.0):
            d.torso_to(height)
            d.wait(4)
        if d.obj_R(name)[2, 2] < 0.9:
            raise Failed("the egg cup tipped over in the hand")
        spot = self.SPOTS[name]
        self.drive(spot[0], -1.15)
        z = self.COUNTER_Z - self.env.objects[name].bottom_offset[2] + 0.002
        self.place(name, [spot[0], spot[1], z])
        self.carry(z=1.15, fwd=0.40)

    def stage_fridge_out(self):
        """Egg cup, butter and lemon from the fridge to the right-hand counter."""
        self.open_fridge()
        self.carry(z=1.15)
        for name in ("lemon", "butter", "egg"):
            self.fridge_take(name)
        self.close_fridge()
        # the hand is still at the door: bring it back before driving off, or it
        # drags the door open again
        self.home()

    def stage_egg(self):
        return self.pour_bag("egg", self.POUR_REST["egg"])

    def stage_knife(self):
        """Knife out of its drawer and onto the counter, so the drawer can be closed."""
        self.taking = "knife"
        self.take_from_drawer(self.env.knife_home, "knife", self.KNIFE_SPOT)
        self.taking = None

    def knife_dir(self):
        """World direction the blade points in, flattened onto the counter."""
        v = self.d.obj_R("knife")[:2, 1]
        return v / np.linalg.norm(v)

    def stage_cut(self):
        """Pick the knife up by its handle and cut the butter, then the lemon."""
        d = self.d
        p = d.obj_pos("knife")
        d.base_goto([p[0] + 0.30, self.STANCE_Y], np.pi / 2)
        self.pick("knife", pitch_deg=90, heading=tuple(self.knife_dir()), lift=0.12, back=0.10)
        d.arm_line([d.eef_pos()[0], d.eef_pos()[1], 1.12], speed=0.015, max_steps=40, tol=0.02, quiet=True)
        self.carry(fwd=0.40)
        for name in ("butter", "lemon"):
            self.cut(name)
        # put the knife down in front of the lemon
        self.grasp_off = d.obj_pos("knife") - d.eef_pos()
        self.drive(self.KNIFE_REST[0], self.STANCE_Y)
        z = self.COUNTER_Z - self.env.objects["knife"].bottom_offset[2] + 0.003
        self.place("knife", [self.KNIFE_REST[0], self.KNIFE_REST[1], z], R=d.goal_R, retreat=(0, 0, 0.12))
        self.home()

    def cut(self, name):
        """Bring the blade down through the seam of `name`, then push the far half aside."""
        d = self.d
        env = self.env
        pos = d.obj_pos(name)
        seam = d.obj_R(name)[:, 1]
        # the blade lies in the seam plane: horizontal, at right angles to the seam normal
        along = np.array([-seam[1], seam[0], 0.0])
        along /= np.linalg.norm(along)
        if along[:2] @ self.knife_dir() < 0:
            along = -along
        # stand so that the hand, which is behind the blade, is straight ahead
        hand_xy = pos[:2] - along[:2] * self.EDGE_MID[1]
        self.drive(hand_xy[0], self.STANCE_Y)
        if not self.held("knife", 0.05):
            raise Failed(f"dropped the knife on the way to the {name}")

        def hand_for(edge_mid):
            """Hand pose that puts the middle of the cutting edge at `edge_mid`, blade along `along`."""
            Rk = d.obj_R("knife")
            now = Rk[:, 1]
            yaw = np.arctan2(along[1], along[0]) - np.arctan2(now[1], now[0])
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
            turn = axis_rot([0.0, 0.0, 1.0], yaw)
            knife_pos = np.asarray(edge_mid, float) - turn @ Rk @ self.EDGE_MID
            return knife_pos + turn @ (d.eef_pos() - d.obj_pos("knife")), turn @ d.eef_R()

        for attempt in range(4):
            pos = d.obj_pos(name)
            floor = pos[2] - 0.5 * env.objects[name].size[2]
            over = np.array([pos[0], pos[1], floor + 0.10])
            p, R = hand_for(over)
            d.arm_to(p, R, max_steps=150, tol=0.012)
            for _ in range(2):  # again, now that the wrist has turned
                p, R = hand_for(over)
                d.arm_to(p, R, max_steps=60, tol=0.005, quiet=True)
            # straight down, steering the edge onto the seam, until the environment reports the cut
            seam2 = seam[:2] / np.linalg.norm(seam[:2])
            for _ in range(160):
                edge = d.obj_pos("knife") + d.obj_R("knife") @ self.EDGE_MID
                if env.prep.cut[name] or edge[2] < floor + 0.003:
                    break
                err = pos[:2] - edge[:2]
                # across the seam matters most; along the blade there is plenty of edge
                fix = seam2 * np.clip(err @ seam2, -0.003, 0.003) + np.clip(err - seam2 * (err @ seam2), -0.002, 0.002)
                d.goal_pos = d.eef_pos() + np.array([fix[0], fix[1], -0.004])
                d.step()
            d.hold_here()
            d.wait(3)
            if env.prep.cut[name]:
                break
            report = {k: round(v, 2) for k, v in env.prep.cut_report(name).items()}
            d.say(f"{name} not cut on try {attempt + 1}: {report}")
            d.arm_line(d.eef_pos() + np.array([0, 0, 0.10]), speed=0.012, max_steps=40, tol=0.02, quiet=True)
        if not env.prep.cut[name]:
            raise Failed(f"the blade went down but the {name} was not cut")
        d.say(f"{name} cut at step {d.steps}")
        # the blade now sits between the halves: slide it sideways to part them
        (pa, _), (pb, _) = env.prep.half_poses(name)
        side = seam if (pb - pa) @ seam >= 0 else -seam
        side = np.array([side[0], side[1], 0.0])
        d.arm_line(d.eef_pos() + side * 0.03, speed=0.004, max_steps=40, tol=0.004, quiet=True)
        d.arm_line(d.eef_pos() - side * 0.01, speed=0.006, max_steps=20, tol=0.004, quiet=True)
        d.arm_line(d.eef_pos() + np.array([0, 0, 0.14]), speed=0.012, max_steps=40, tol=0.02, quiet=True)
        d.say(f"{name} halves are {1000 * env.prep_state()['half_gap'][name]:.0f} mm apart")


    def watch_halves(self):
        """For replay: how far the loose half of each cut ingredient has risen since the cut."""
        state = self.env.prep_state()
        for name, cut in state["cut"].items():
            if not cut:
                continue
            _, (pb, Rb) = self.env.prep.half_poses(name)
            z = float((pb + Rb @ prep.HALF_CENTRE[name])[2])
            base = self.__dict__.setdefault("_cut_z", {}).setdefault(name, z)
            self.lifted[name] = max(self.lifted[name], z - base)

    def watch_halves(self):
        """For replay: how far the loose half of each cut ingredient has risen since the cut."""
        state = self.env.prep_state()
        for name, cut in state["cut"].items():
            if not cut:
                continue
            _, (pb, Rb) = self.env.prep.half_poses(name)
            z = float((pb + Rb @ prep.HALF_CENTRE[name])[2])
            base = self.__dict__.setdefault("_cut_z", {}).setdefault(name, z)
            self.lifted[name] = max(self.lifted[name], z - base)

    def stage_halves(self):
        """Pick up one half of the butter and one half of the lemon, each on its own."""
        for name in ("butter", "lemon"):
            self.lift_half(name)
        self.home()

    def lift_half(self, name, height=0.10):
        """Pinch the loose half from above, lift it clear of the other half, set it back down."""
        d = self.d
        env = self.env

        def half():
            (pa, _), (pb, Rb) = env.prep.half_poses(name)
            return pa, pb + Rb @ prep.HALF_CENTRE[name], Rb

        pa, c, Rb = half()
        self.drive(c[0], self.STANCE_Y)
        pa, c, Rb = half()
        # fingers across the half's width: come down with the hand turned along its length
        h = Rb[:2, 1] if abs(Rb[2, 1]) < 0.8 else Rb[:2, 0]
        h = h / np.linalg.norm(h)
        if h[1] < 0:
            h = -h
        R = grip_R([0.0, 0.0, -1.0], np.cross([0.0, 0.0, 1.0], [h[0], h[1], 0.0]))
        z0 = c[2]
        d.set_gripper(False, steps=6)
        d.arm_to(c + np.array([0, 0, 0.10]), R, max_steps=150, tol=0.015)
        d.arm_line(c + np.array([0, 0, 0.002]), R, speed=0.008, max_steps=50, tol=0.006)
        d.set_gripper(True)
        d.arm_line(d.eef_pos() + np.array([0, 0, height]), R, speed=0.008, max_steps=50, tol=0.01)
        d.wait(10)
        pa2, c2, _ = half()
        self.lifted[name] = float(c2[2] - z0)
        d.say(f"{name}: one half lifted {100 * self.lifted[name]:.1f} cm; the other moved {1000 * np.linalg.norm(pa2 - pa):.0f} mm")
        d.arm_line(d.eef_pos() - np.array([0, 0, height - 0.004]), R, speed=0.008, max_steps=50, tol=0.006)
        d.set_gripper(False)
        d.wait(3)
        d.arm_line(d.eef_pos() + np.array([0, 0, 0.12]), R, speed=0.012, max_steps=40, tol=0.02, quiet=True)
        if self.lifted[name] < 0.05:
            raise Failed(f"could not lift half of the {name}")


STAGES = (
    ("bowl_out", "Bowl from the drawer onto the counter", "stage_bowl"),
    ("dry_out", "Flour and sugar out of the drawers", "stage_dry_out"),
    ("sugar_in_bowl", "Sugar cube into the bowl", "stage_sugar"),
    ("flour_poured", "Flour bag tipped over the bowl", "stage_flour"),
)
USABLE_TITLES = {
    "dry_out": "Flour bag and sugar bag out of the drawers",
    "sugar_in_bowl": "Pour the sugar into the bowl",
    "flour_poured": "Pour the flour into the bowl",
}
USABLE_STAGES = STAGES + (
    ("fridge_out", "Egg, butter, and lemon out of the fridge", "stage_fridge_out"),
    ("egg_poured", "Pour the egg into the bowl", "stage_egg"),
    ("knife_out", "Take the knife from its drawer", "stage_knife"),
    ("cut", "Cut the butter and the lemon", "stage_cut"),
    ("halves_lifted", "Lift one half of each cut ingredient", "stage_halves"),
)
MIN_GAP = 0.008  # how far apart the two halves must end up, metres
MIN_POURED = 0.7  # share of each bag's grains that must end up in the bowl


def measure(chef):
    d = chef.d
    env = chef.env

    def upright(name):
        return bool(d.obj_R(name)[2, 2] > 0.95 and d.obj_pos(name)[2] > 0.93)

    common = {
        "flour_max_tilt_deg_over_bowl": round(chef.pour["flour"]["max_tilt_deg"], 1),
        "flour_steps_past_90_deg": chef.pour["flour"]["steps_past_90"],
        "flour_upright_on_counter": upright("flour"),
        "bowl_upright_on_counter": bool(d.obj_R("bowl")[2, 2] > 0.98 and d.obj_pos("bowl")[2] > 0.93),
        "drawers_closed": all(drawer.is_closed(env) for drawer in env.drawers),
    }
    if not chef.usable:
        return {"sugar_in_bowl": chef.in_bowl("sugar"), **common}
    grains = chef.grains_in_bowl()
    state = env.prep_state()
    out = {}
    for name in ("sugar", "flour", "egg"):
        out[f"{name}_grains_in_bowl"] = grains[name][0]
        out[f"{name}_grains_total"] = grains[name][1]
    for name in ("butter", "lemon"):
        out[f"{name}_cut"] = bool(state["cut"][name])
        out[f"{name}_halves_apart_mm"] = round(1000 * state["half_gap"][name], 1)
        out[f"{name}_half_lifted_cm"] = round(100 * chef.lifted[name], 1)
    out.update(
        sugar_upright_on_counter=upright("sugar"),
        egg_upright_on_counter=upright("egg"),
        knife_on_counter=bool(d.obj_pos("knife")[2] > 0.92),
        fridge_closed="fridge" not in env._sense()["open"],
    )
    return {**out, **common}


def passed(checks, usable):
    ok = [checks["flour_upright_on_counter"], checks["bowl_upright_on_counter"], checks["drawers_closed"]]
    if usable:
        ok += [checks[f"{n}_grains_in_bowl"] >= MIN_POURED * checks[f"{n}_grains_total"] for n in ("sugar", "flour", "egg")]
        ok += [checks[f"{n}_cut"] and checks[f"{n}_halves_apart_mm"] >= 1000 * MIN_GAP for n in ("butter", "lemon")]
        ok += [checks[f"{n}_half_lifted_cm"] >= 5.0 for n in ("butter", "lemon")]
        ok += [checks[k] for k in ("sugar_upright_on_counter", "egg_upright_on_counter", "knife_on_counter", "fridge_closed")]
    else:
        ok += [checks["sugar_in_bowl"], checks["flour_max_tilt_deg_over_bowl"] >= 100.0]
    return all(ok)


def run(seed=1, video=None, every=4, verbose=True, actions_path=None, after_stage=None, usable=False, clip_dir=None):
    render = video is not None or clip_dir is not None
    kwargs = dict(has_renderer=False, has_offscreen_renderer=render, use_camera_obs=False, seed=seed)
    if usable:
        kwargs["usable_ingredients"] = True
    if render:
        kwargs["renderer"] = "mujoco"
    env = LemonTart(**kwargs)
    env.reset()
    sink = None
    recorder = None
    if video is not None:
        sink = VideoSink(video, 1280, 720, fps=20)
        recorder = FollowCamera(env, every=every, width=1280, height=720, sink=sink)
    driver = Driver(env, recorder, verbose=verbose)
    chef = MixChef(driver, usable=usable)
    clipper = None
    if clip_dir is not None and usable and video is None:
        clipper = ClipRecorder(env, chef, clip_dir, every=every)
        driver.recorder = clipper
    if sink is not None:

        def status(drv):
            if usable:
                g = chef.grains_in_bowl()
                cut = chef.env.prep_state()["cut"]
                head = (
                    f"in bowl: sugar {g['sugar'][0]}/{g['sugar'][1]}  flour {g['flour'][0]}/{g['flour'][1]}"
                    f"  egg {g['egg'][0]}/{g['egg'][1]}   cut: butter {'yes' if cut['butter'] else 'no'}"
                    f"  lemon {'yes' if cut['lemon'] else 'no'}"
                )
            else:
                head = (
                    f"sugar in bowl: {'yes' if chef.in_bowl('sugar') else 'no'}"
                    f"   flour bag max tilt {chef.pour['flour']['max_tilt_deg']:.0f} deg"
                )
            return (
                f"{head}   step {drv.steps}   sim {int(drv.steps / 20 // 60)}:{int(drv.steps / 20 % 60):02d}"
                "   robot's own actions"
            )

        sink.status = status
    actions = []
    if actions_path is not None:
        step = env.step

        def logged(action):
            actions.append(np.array(action, dtype=np.float64))
            return step(action)

        env.step = logged
    stages = []
    error = None
    started = time.time()
    try:
        driver.wait(10)
        for label, title, method in USABLE_STAGES if usable else STAGES:
            if sink is not None:
                sink.title = USABLE_TITLES.get(label, title) if usable else title
            if clipper is not None:
                clipper.stage = label
            t0 = driver.steps
            driver.say(f"=== {label}")
            getattr(chef, method)()
            stages.append({"stage": label, "steps": driver.steps - t0, "ends_at": driver.steps})
            if after_stage is not None:
                after_stage(label, driver, chef)
        driver.wait(20)
    except Failed as exc:
        error = str(exc)
        driver.say(f"FAILED: {error}")
    if clipper is not None:
        clipper.finish(driver)
    checks = measure(chef)
    success = error is None and passed(checks, usable)
    summary = {
        "seed": seed,
        "usable": usable,
        "steps": driver.steps,
        "simulated_seconds": driver.steps / 20.0,
        "wall_seconds": round(time.time() - started, 1),
        "stages": stages,
        "checks": checks,
        "success": bool(success),
        "error": error,
        "note": (
            "Flour, sugar and egg are open containers of loose grains; the checks count the grains inside the "
            "bowl. Butter and lemon are two welded halves that the environment releases when the knife blade "
            "comes down through the seam."
            if usable
            else "The flour bag is a single rigid body: it is tipped over the bowl but nothing comes out of it."
        ),
        "renderer": "MuJoCo native offscreen renderer (no Cycles exporter for RoboCasa)" if video else None,
    }
    if sink is not None:
        done_title = (
            "Done: sugar, flour, and egg in the bowl; butter and lemon cut in half"
            if usable
            else "Done: sugar in the bowl, flour bag tipped"
        )
        sink.title = done_title if success else f"Stopped: {error or 'checks failed'}"
        frame = recorder.render()
        for _ in range(60):
            sink(frame, driver)
        sink.close()
        summary["video"] = str(video)
        summary["video_frames"] = sink.frames
    if clipper is not None:
        summary["clips"] = clipper.written
    if actions_path is not None:
        np.savez_compressed(actions_path, actions=np.stack(actions), seed=seed, usable=usable)
        summary["actions"] = str(actions_path)
    env.close()
    return summary


def replay(actions_path):
    """Feed a saved action log to a fresh environment; only `env.step` runs."""
    data = np.load(actions_path)
    seed = int(data["seed"])
    usable = bool(data["usable"]) if "usable" in data else False
    kwargs = dict(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, seed=seed)
    if usable:
        kwargs["usable_ingredients"] = True
    env = LemonTart(**kwargs)
    env.reset()
    chef = MixChef(Driver(env, None, verbose=False), usable=usable)
    for action in data["actions"]:
        env.step(action)
        chef.watch_pour("flour")
        if usable:
            chef.watch_halves()
        if usable:
            chef.watch_halves()
    checks = measure(chef)
    result = {
        "seed": seed,
        "usable": usable,
        "steps": int(len(data["actions"])),
        "checks": checks,
        "success": passed(checks, usable),
    }
    env.close()
    return result


# Close-up clips of the usable-ingredients demo: name -> (stages, caption,
# object that has to be near the target before the camera rolls, target, camera).
CLIPS = {
    "flour-pour": (("flour_poured",), "Flour pours out", "flour", "bowl"),
    "egg-pour": (("egg_poured",), "Egg is a liquid and pours out", "egg", "bowl"),
    "sugar-pour": (("sugar_in_bowl",), "Sugar is a bag and pours out", "sugar", "bowl"),
    "cut": (("cut", "halves_lifted"), "Lemon and butter can be cut", "knife", "butter"),
}


class ClipRecorder:
    """Writes one close-up video per usable ingredient while the controller runs.

    It stands in for the Driver's recorder: `stage` is set by `run`, and the
    camera rolls once the thing being used is next to its target."""

    def __init__(self, env, chef, out_dir, every=2):
        from owb.scripts.lemon_tart_reference import Recorder

        self.env, self.chef, self.out_dir, self.every = env, chef, Path(out_dir), every
        self.recorder = Recorder(env, every=every, width=1280, height=720)
        self.stage = None
        self.active = None  # (name, sink)
        self.written = {}

    def _status(self, clip):
        def status(drv):
            if clip == "cut":
                state = self.env.prep_state()
                return (
                    f"butter: {'cut' if state['cut']['butter'] else 'whole'}, gap {1000 * state['half_gap']['butter']:.0f} mm"
                    f"   lemon: {'cut' if state['cut']['lemon'] else 'whole'}, gap {1000 * state['half_gap']['lemon']:.0f} mm"
                    f"   step {drv.steps}   robot's own actions"
                )
            key = clip.split("-")[0]
            g = self.chef.grains_in_bowl()
            return f"{key} grains in bowl: {g[key][0]}/{g[key][1]}   step {drv.steps}   robot's own actions"

        return status

    def _close(self, driver):
        name, sink = self.active
        frame = self.recorder.render()
        for _ in range(40):
            sink(frame, driver)
        sink.close()
        self.active = None

    def tick(self, driver):
        name = next((n for n, spec in CLIPS.items() if self.stage in spec[0]), None)
        if self.active is not None and self.active[0] != name:
            self._close(driver)
        if name is None:
            return
        stages, caption, mover, target = CLIPS[name]
        if self.active is None:
            near = np.linalg.norm((driver.obj_pos(mover) - driver.obj_pos(target))[:2]) < 0.45
            if not near or name in self.written:
                return
            sink = VideoSink(self.out_dir / f"lemon-tart-usable-{name}.mp4", 1280, 720, fps=20)
            sink.title = caption
            sink.status = self._status(name)
            self.active = (name, sink)
            self.written[name] = str(sink.path)
        if driver.steps % self.every:
            return
        if name == "cut":
            self.recorder.aim([3.93, -1.02, 1.95], [3.93, -0.50, 0.93], 42.0)
        else:
            bowl = driver.obj_pos("bowl")
            self.recorder.aim([bowl[0] - 0.62, bowl[1] - 0.50, 1.50], [bowl[0] + 0.03, bowl[1], 1.02], 45.0)
        self.active[1](self.recorder.render(), driver)

    def finish(self, driver):
        if self.active is not None:
            self._close(driver)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", type=Path, default=None, help="replay a saved action log instead of running the controller")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--usable", action="store_true", help="ingredients that pour and cut (LemonTart(usable_ingredients=True)), and the longer demo")
    parser.add_argument("--out", type=Path, default=None, help="default: videos/lemon-tart-mix[-usable].mp4")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--every", type=int, default=2, help="render every N-th control step")
    parser.add_argument("--summary", type=Path, default=None, help="default: videos/lemon-tart-mix[-usable].json")
    parser.add_argument("--actions", type=Path, default=None, help="save the action log (.npz)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--clips", type=Path, default=None, help="with --usable --no-video: write one close-up video per usable ingredient into this directory")
    args = parser.parse_args()
    if args.replay is not None:
        result = replay(args.replay)
        print(json.dumps(result, indent=2))
        return 0 if result["success"] else 1
    stem = "lemon-tart-mix-usable" if args.usable else "lemon-tart-mix"
    args.out = args.out or REPO / "videos" / f"{stem}.mp4"
    args.summary = args.summary or REPO / "videos" / f"{stem}.json"
    summary = run(
        seed=args.seed,
        usable=args.usable,
        video=None if args.no_video else args.out,
        every=args.every,
        verbose=not args.quiet,
        actions_path=args.actions,
        clip_dir=args.clips,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("steps", "success", "error", "checks", "stages")}, indent=2, ensure_ascii=False))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
