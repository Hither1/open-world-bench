"""Lemon tart, full flow of docs/mobile-manipulation-tasks.md (work in progress).

Builds on the earlier scripts: `MixChef` (pouring, cutting, lifting halves) and
`LockChef` (oven lock, knob order). This file adds what the new flow needs on
top of them, starting with the overhead cabinets.
"""
from __future__ import annotations

import numpy as np

from owb.scripts.lemon_tart_mix import MixChef, axis_rot
from owb.scripts.lemon_tart_oven_lock import LockChef
from owb.scripts.lemon_tart_reference import Failed, grip_R


class Cabinet:
    """Geometry of one overhead cabinet door, read from the simulator."""

    def __init__(self, driver, name, hinge_joint=None, handle_geom=None):
        self.d = driver
        m = driver.sim.model
        self.name = name
        self.joint = hinge_joint or f"{name}_doorhinge"
        self.handle = handle_geom or f"{name}_door_handle_g1"
        jid = m.joint_name2id(self.joint)
        self.hinge = np.array(driver.sim.data.xanchor[jid])
        lo, hi = m.jnt_range[jid]
        self.sign = -1.0 if lo < -0.1 else 1.0  # joint angle grows in this sense as the door opens
        self.limit = float(max(abs(lo), abs(hi)))
        h0 = np.array(driver.geom_pos(self.handle))
        self.handle_rel = h0 - self.hinge  # at the current angle
        self.handle_rel = axis_rot([0, 0, 1.0], -self.angle_raw()) @ self.handle_rel  # at angle 0
        body = m.body_name2id(f"{name}_main")
        self.centre = np.array(driver.sim.data.body_xpos[body])
        self.floor = float(driver.geom_pos(f"{name}_bottom")[2] + m.geom_size[m.geom_name2id(f"{name}_bottom")][2])

    def angle_raw(self):
        return self.d.joint(self.joint)

    def opening(self):
        """How far the door is open, rad, 0 = shut."""
        return self.sign * self.angle_raw()

    def point(self, opening, rel):
        """World position of a point fixed to the door (`rel` from the hinge at angle 0) at this opening."""
        return self.hinge + axis_rot([0, 0, 1.0], self.sign * opening) @ np.asarray(rel, float)

    def out(self, opening):
        """Outward normal of the door's front face."""
        return axis_rot([0, 0, 1.0], self.sign * opening) @ np.array([0.0, -1.0, 0.0])

    def edge_dir(self, opening):
        """From the hinge towards the door's free edge."""
        x = 1.0 if self.handle_rel[0] > 0 else -1.0
        return axis_rot([0, 0, 1.0], self.sign * opening) @ np.array([x, 0.0, 0.0])


class FullChef(MixChef, LockChef):
    CAB_TORSO = 0.20
    CAB_OPEN = 1.30  # rad; far enough to reach in, and the outer face can still be reached to close it

    def cabinet(self, name):
        return Cabinet(self.d, name)

    def open_cabinet(self, name):
        """Pinch the handle bar, pull the door part of the way, then push it wide from behind."""
        d = self.d
        cab = self.cabinet(name)
        self.home()
        handle = cab.point(0.0, cab.handle_rel)
        self.drive(handle[0] + 0.02, -1.08)
        d.torso_to(self.CAB_TORSO)
        z = handle[2]

        def grip(opening):
            n = cab.out(opening)
            return grip_R(-n, cab.edge_dir(opening))

        R = grip(0.0)
        d.set_gripper(False, steps=6)
        d.arm_to(handle + cab.out(0.0) * 0.10, R, max_steps=160, tol=0.012)
        d.arm_line(handle - cab.out(0.0) * 0.006, R, speed=0.008, max_steps=50, tol=0.006)
        w = d.set_gripper(True)
        d.say(f"cabinet {name}: handle gripped, finger width {w:.3f}")
        for _ in range(320):
            cur = cab.opening()
            if cur >= 0.75:
                break
            th = min(cur + 0.35, 0.90)
            d.goal_pos = cab.point(th, cab.handle_rel) - cab.out(th) * 0.006
            d.goal_R = grip(th)
            d.step()
        d.hold_here()
        d.say(f"cabinet {name}: pulled to {np.degrees(cab.opening()):.0f} deg")
        d.set_gripper(False)
        cur = cab.opening()
        d.arm_line(d.eef_pos() + cab.out(cur) * 0.09, speed=0.012, max_steps=40, tol=0.02, quiet=True)
        # behind the door: push its inner face the rest of the way
        d.set_gripper(True, steps=6)
        R = grip_R([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0])
        r = np.linalg.norm(cab.handle_rel[:2]) - 0.10

        def inner(opening):
            p = cab.hinge + cab.edge_dir(opening) * r - cab.out(opening) * 0.075
            p[2] = z
            return p

        # go round the free edge: out in front of it, past it, in behind it
        cur = cab.opening()
        past = cab.hinge + cab.edge_dir(cur) * (np.linalg.norm(cab.handle_rel[:2]) + 0.19)
        past[2] = z
        d.arm_to(past + cab.out(cur) * 0.09, R, max_steps=120, tol=0.02, quiet=True)
        d.arm_line(past - cab.out(cur) * 0.09, R, speed=0.012, max_steps=60, tol=0.015, quiet=True)
        d.arm_line(inner(cur), R, speed=0.012, max_steps=60, tol=0.015, quiet=True)
        d.say(f"cabinet {name}: behind the door at {np.degrees(cab.opening()):.0f} deg")
        for _ in range(240):
            cur = cab.opening()
            if cur >= self.CAB_OPEN:
                break
            d.goal_pos = inner(min(cur + 0.25, self.CAB_OPEN + 0.1))
            d.step()
        d.hold_here()
        cur = cab.opening()
        d.arm_line(d.eef_pos() - cab.out(cur) * 0.08 + np.array([0, -0.05, 0]), R, speed=0.012, max_steps=40, tol=0.02, quiet=True)
        d.say(f"cabinet {name}: open {np.degrees(cab.opening()):.0f} deg")
        if cab.opening() < self.CAB_OPEN - 0.2:
            raise Failed(f"cabinet {name} only opened {np.degrees(cab.opening()):.0f} deg")
        return cab

    def lower_torso(self, name=None):
        """Come down in steps, so that a container in the hand keeps its contents."""
        d = self.d
        for height in (0.15, 0.10, 0.05, 0.0):
            if d.torso() > height + 0.01:
                d.torso_to(height)
                d.wait(3)

    def take_from_cabinet(self, cab, name, spot, pitch_deg=12.0):
        """Reach into the open cabinet, bring `name` out and set it on the counter at `spot` (x, y)."""
        d = self.d
        p = d.obj_pos(name)
        d.base_goto([p[0] + 0.02, -1.0], np.pi / 2)
        d.torso_to(self.CAB_TORSO)
        self.pick(name, pitch_deg, heading="long", back=0.14, lift=0.025)
        # straight out of the cabinet before anything else
        d.arm_line([d.eef_pos()[0], -0.62, d.eef_pos()[2]], speed=0.012, max_steps=60, tol=0.02, quiet=True)
        self.carry(fwd=0.42)
        if not self.held(name, 0.08):
            raise Failed(f"lost the {name} while taking it out of the cabinet")
        self.lower_torso()
        d.arm_line([d.eef_pos()[0], d.eef_pos()[1], 1.15], speed=0.012, max_steps=60, tol=0.02, quiet=True)
        self.drive(spot[0], -1.15)
        z = self.COUNTER_Z - self.env.objects[name].bottom_offset[2] + 0.003
        self.place(name, [spot[0], spot[1], z])
        self.carry(z=1.15, fwd=0.40)

    def put_in_cabinet(self, cab, name, xy):
        """Pick `name` up from the counter and set it down inside the open cabinet at `xy`."""
        d = self.d
        p = d.obj_pos(name)
        self.drive(p[0], max(-1.15, p[1] - 0.60))
        self.pick(name, 30.0, heading="long")
        self.carry(fwd=0.42, z=1.18)
        self.drive(xy[0] + 0.02, -1.0)
        d.torso_to(self.CAB_TORSO)
        d.wait(3)
        z = cab.floor - self.env.objects[name].bottom_offset[2] + 0.006
        self.grasp_off = d.obj_pos(name) - d.eef_pos()
        tgt = np.array([xy[0], xy[1], z]) - self.grasp_off
        d.arm_to([tgt[0], -0.62, tgt[2] + 0.04], self.grasp_R, max_steps=160, tol=0.015)
        d.arm_line(tgt + np.array([0, 0, 0.03]), self.grasp_R, speed=0.012, max_steps=70, tol=0.012, quiet=True)
        d.arm_line(tgt + np.array([0, 0, 0.004]), self.grasp_R, speed=0.006, max_steps=30, tol=0.006, quiet=True)
        d.set_gripper(False)
        d.wait(3)
        d.arm_line([d.eef_pos()[0], -0.64, d.eef_pos()[2] + 0.02], self.grasp_R, speed=0.012, max_steps=60, tol=0.02, quiet=True)
        d.say(f"{name} in the cabinet at {np.round(d.obj_pos(name), 3)}")
        self.carry(fwd=0.40)
        self.lower_torso()

    def close_cabinet(self, cab):
        """Push the door's outer face round until it is nearly shut, then press it home from the front."""
        d = self.d
        name = cab.name
        self.home()
        radius = float(np.linalg.norm(cab.handle_rel[:2]))
        cur = cab.opening()
        edge = cab.hinge + cab.edge_dir(cur) * radius
        # stand back and to the open side, so the hand can get to the outer face
        self.drive(float(np.clip(edge[0] + 0.30, 0.55, 4.4)), -1.38)
        d.torso_to(self.CAB_TORSO)
        z = cab.point(0.0, cab.handle_rel)[2] - 0.02
        R = grip_R([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0])
        d.set_gripper(True, steps=6)

        def outer(opening, r=radius - 0.12, off=0.075):
            p = cab.hinge + cab.edge_dir(opening) * r + cab.out(opening) * off
            p[2] = z
            return p

        cur = cab.opening()
        d.arm_to(outer(cur, off=0.16) + np.array([0, -0.06, 0]), R, max_steps=160, tol=0.02)
        d.arm_line(outer(cur), R, speed=0.012, max_steps=60, tol=0.015, quiet=True)
        for _ in range(300):
            cur = cab.opening()
            if cur < 0.12:
                break
            d.goal_pos = outer(max(cur - 0.25, -0.05), off=0.06)
            d.step()
        d.hold_here()
        d.say(f"cabinet {name}: pushed to {np.degrees(cab.opening()):.0f} deg")
        # the last bit: press on the door's face near the handle
        d.arm_line(d.eef_pos() + np.array([0, -0.06, 0]), R, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        front = cab.point(0.0, cab.handle_rel) + np.array([-0.07 * np.sign(cab.handle_rel[0]), -0.09, -0.02])
        d.arm_to(front, R, max_steps=100, tol=0.015, quiet=True)
        for _ in range(100):
            if cab.opening() < 0.01:
                break
            d.goal_pos = np.array([front[0], d.eef_pos()[1] + 0.008, front[2]])
            d.step()
        d.hold_here()
        d.arm_line(d.eef_pos() + np.array([0, -0.10, 0]), R, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        d.say(f"cabinet {name}: closed to {np.degrees(cab.opening()):.1f} deg")
        self.carry(fwd=0.40)
        self.lower_torso()
        if cab.opening() > 0.03:
            raise Failed(f"cabinet {name} did not close ({np.degrees(cab.opening()):.0f} deg)")
