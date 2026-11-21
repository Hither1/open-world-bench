"""Lemon-tart reference trajectory.

python -m owb.scripts.lemon_tart_reference --seed 1 --horizon 24000

A scripted controller drives the PandaOmron through the nine lemon-tart
milestones. It reads simulator poses and joints, and `evaluator_state()` for
the ingredient homes, but everything it does goes through `env.step(action)`
with the ordinary 12-dim action vector. `run()` never writes qpos, never calls
`open_door` / `set_*`, and never touches the tracker. (`Driver.snapshot` and
`Driver.restore` exist for developing one stage at a time; `run()` does not
use them.)

Milestones are confirmed from `info["lemon_tart"]`, not from a parallel check.

Seed 1 takes 22402 control steps, more than the environment's 20000-step
limit, so the recorded run passes `--horizon 24000`. With the default limit
the episode is cut off after seven of the nine milestones.
Results and the things that differ from the original plan are in
`docs/lemon_tart_reference.md`.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import robosuite.utils.transform_utils as T

from owb.tasks.lemon_tart import MILESTONES
from owb.tasks.lemon_tart_env import LemonTart

REPO = Path(__file__).resolve().parents[2]
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

POS_STEP = 0.05  # OSC output_max, metres per unit action
ROT_STEP = 0.5  # rad per unit action


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rotvec(R):
    """Rotation matrix -> axis-angle vector."""
    quat = T.mat2quat(R)
    return T.quat2axisangle(quat)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


class Failed(RuntimeError):
    pass


class Driver:
    """Thin layer over env.step: world-frame arm goals, base velocity, gripper."""

    def __init__(self, env, recorder=None, verbose=True):
        self.env = env
        self.sim = env.sim
        self.robot = env.robots[0]
        self.cc = self.robot.composite_controller
        self.arm = self.cc.part_controllers["right"]
        self.recorder = recorder
        self.verbose = verbose
        self.steps = 0
        self.info = {}
        self.done = False
        self.grip = -1.0  # commanded gripper, -1 open .. +1 closed
        self.goal_pos = None  # world
        self.goal_R = None  # world
        self.stall_after = 40  # arm_to gives up once the arm has stopped moving ...
        self.stall_window = 25  # ... for this many steps
        self.lead = 0.05  # how far the commanded goal may lead the achieved pose
        self.rot_lead = 0.5
        self.latched = []
        self.log = []
        self.label = ""
        m = self.sim.model
        self.center_sid = m.site_name2id(self.robot.robot_model.base.correct_naming("center"))
        self.eef_sid = self.robot.eef_site_id["right"]
        self.finger_qadr = [
            m.get_joint_qpos_addr(j) for j in ("gripper0_right_finger_joint1", "gripper0_right_finger_joint2")
        ]
        self.torso_qadr = m.get_joint_qpos_addr("mobilebase0_joint_torso_height")
        self.yaw0 = self.base_yaw()  # yaw at reset
        self.torso_goal = self.torso()
        self.hold_here()

    # ---- reading ----
    def base_pos(self):
        return np.array(self.sim.data.site_xpos[self.center_sid])

    def base_R(self):
        return np.array(self.sim.data.site_xmat[self.center_sid]).reshape(3, 3)

    def base_yaw(self):
        R = self.base_R()
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def eef_pos(self):
        return np.array(self.sim.data.site_xpos[self.eef_sid])

    def eef_R(self):
        return np.array(self.sim.data.site_xmat[self.eef_sid]).reshape(3, 3)

    def finger_width(self):
        q = self.sim.data.qpos
        return float(q[self.finger_qadr[0]] - q[self.finger_qadr[1]])

    def torso(self):
        return float(self.sim.data.qpos[self.torso_qadr])

    def obj_pos(self, name):
        return np.array(self.sim.data.body_xpos[self.env.obj_body_id[name]])

    def obj_R(self, name):
        return np.array(self.sim.data.body_xmat[self.env.obj_body_id[name]]).reshape(3, 3)

    def joint(self, name):
        return float(self.sim.data.qpos[self.sim.model.get_joint_qpos_addr(name)])

    def geom_pos(self, name):
        return np.array(self.sim.data.geom_xpos[self.sim.model.geom_name2id(name)])

    def milestones(self):
        return self.info.get("lemon_tart", {}).get("milestones", {})

    # ---- acting ----
    def hold_here(self):
        self.goal_pos = self.eef_pos()
        self.goal_R = self.eef_R()

    def step(self, base=(0.0, 0.0, 0.0), torso=None):
        """One control step. The arm chases self.goal_pos / self.goal_R (world)."""
        if self.done:
            if self.info.get("lemon_tart", {}).get("success"):
                return self.info  # the environment ends the episode on success
            raise Failed("episode already ended")
        if torso is None:
            # the torso controller is a delta on the achieved height, so it
            # creeps under load; hold the height we last asked for
            e = self.torso_goal - self.torso()
            torso = float(np.clip(e / 0.02, -1, 1)) if abs(e) > 0.003 else 0.0
        Rb = self.base_R()
        pb = self.base_pos()
        cur_p = self.eef_pos()
        cur_R = self.eef_R()
        # the controller is always in "desired" mode (base_mode=+1): its goal is
        # last goal + delta, expressed in the base frame. Lead-limit the goal so
        # contact forces stay bounded.
        err = self.goal_pos - cur_p
        n = np.linalg.norm(err)
        tgt_p = cur_p + err * min(1.0, self.lead / n) if n > 1e-9 else cur_p
        rv = rotvec(self.goal_R @ cur_R.T)
        a = np.linalg.norm(rv)
        tgt_R = (T.quat2mat(T.axisangle2quat(rv * min(1.0, self.rot_lead / a))) @ cur_R) if a > 1e-9 else cur_R
        tgt_p_b = Rb.T @ (tgt_p - pb)
        tgt_R_b = Rb.T @ tgt_R
        if self.arm.goal_pos is None:
            ctrl_p = Rb.T @ (cur_p - pb)
            ctrl_R = Rb.T @ cur_R
        else:
            ctrl_p = np.array(self.arm.goal_pos)
            ctrl_R = np.array(self.arm.goal_ori)
        dp = np.clip((tgt_p_b - ctrl_p) / POS_STEP, -1, 1)
        dr = np.clip(rotvec(tgt_R_b @ ctrl_R.T) / ROT_STEP, -1, 1)
        # The base controller maps its (forward, left) input through the yaw
        # change since reset with the wrong sign. Pre-rotate so that `base` is
        # really expressed in the current base frame.
        base = np.asarray(base, dtype=float).copy()
        if base[0] != 0.0 or base[1] != 0.0:
            dyaw = _wrap(self.base_yaw() - self.yaw0)
            c2, s2 = np.cos(2 * dyaw), np.sin(2 * dyaw)
            base[0], base[1] = c2 * base[0] - s2 * base[1], s2 * base[0] + c2 * base[1]
        action = self.robot.create_action_vector(
            {
                "right": np.concatenate([dp, dr]),
                "right_gripper": np.array([self.grip]),
                "torso": np.array([torso]),
                "base": np.asarray(base, dtype=float),
                "base_mode": 1,
            }
        )
        _, reward, done, info = self.env.step(action)
        self.steps += 1
        self.info = info
        self.done = bool(done)
        now = [k for k, v in info["lemon_tart"]["milestones"].items() if v]
        for name in now:
            if name not in self.latched:
                self.latched.append(name)
                self.say(f"*** milestone {len(self.latched)}/9 {name} at step {self.steps}")
                self.log.append({"step": self.steps, "milestone": name})
        if self.recorder is not None:
            self.recorder.tick(self)
        return info

    def say(self, *args):
        if self.verbose:
            print(f"[{self.steps:6d}]", *args, flush=True)

    def wait(self, n):
        for _ in range(n):
            self.step()

    # ---- arm ----
    def arm_to(self, pos=None, R=None, tol=0.01, rot_tol=0.08, max_steps=200, settle=3, speed=None, quiet=False):
        """Servo the gripper site to a world pose. Returns True when within tol."""
        if pos is not None:
            final = np.asarray(pos, dtype=float)
        else:
            final = self.goal_pos.copy()
        if R is not None:
            self.goal_R = np.asarray(R, dtype=float)
        ok = 0
        trail = []
        for _ in range(max_steps):
            if speed is None:
                self.goal_pos = final
            else:
                d = final - self.goal_pos
                n = np.linalg.norm(d)
                self.goal_pos = final if n <= speed else self.goal_pos + d * speed / n
            self.step()
            perr = np.linalg.norm(self.eef_pos() - final)
            rerr = np.linalg.norm(rotvec(self.goal_R @ self.eef_R().T))
            if perr < tol and rerr < rot_tol:
                ok += 1
                if ok >= settle:
                    return True
            else:
                ok = 0
            # stalled: the goal has arrived but the arm no longer moves
            trail.append((self.eef_pos(), self.eef_R()))
            if len(trail) > self.stall_after and np.linalg.norm(self.goal_pos - final) < 1e-6:
                moved = np.linalg.norm(trail[-1][0] - trail[-self.stall_window][0])
                turned = np.linalg.norm(rotvec(trail[-1][1] @ trail[-self.stall_window][1].T))
                if moved < 0.002 and turned < 0.01:
                    break
        if not quiet:
            self.say(f"   arm_to timeout: pos err {np.round(self.eef_pos() - final, 3)} rot err {rerr:.2f}")
        return False

    def arm_line(self, pos, R=None, speed=0.01, **kw):
        """Straight-line move at `speed` metres per step."""
        self.goal_pos = self.eef_pos()
        return self.arm_to(pos, R, speed=speed, **kw)

    def set_gripper(self, closed, steps=14):
        self.grip = 1.0 if closed else -1.0
        last = None
        for i in range(steps + 10):
            self.step()
            w = self.finger_width()
            if i >= steps and last is not None and abs(w - last) < 2e-4:
                break
            last = w
        return self.finger_width()

    def torso_to(self, height, max_steps=120):
        self.torso_goal = height
        for _ in range(max_steps):
            e = height - self.torso()
            if abs(e) < 0.004:
                break
            self.step(torso=float(np.clip(e / 0.02, -1, 1)))
        self.step()

    # ---- base ----
    def base_goto(self, xy, yaw, tol=0.02, yaw_tol=0.03, max_steps=600, vmax=1.0, carry=True):
        """Drive the base to a stance with velocity commands, then stop."""
        xy = np.asarray(xy, dtype=float)
        # the arm goal rides with the base
        rel_p = self.base_R().T @ (self.goal_pos - self.base_pos())
        rel_R = self.base_R().T @ self.goal_R
        ok = 0
        hist = []
        for it in range(max_steps):
            pb = self.base_pos()
            Rb = self.base_R()
            hist.append((pb[0], pb[1], self.base_yaw()))
            if it >= 60 and ok == 0:
                old = hist[-60]
                if np.hypot(pb[0] - old[0], pb[1] - old[1]) < 0.01 and abs(_wrap(self.base_yaw() - old[2])) < 0.02:
                    self.say(f"base stuck at {np.round(pb[:2], 3)} heading for {np.round(xy, 3)}")
                    break
            e_w = np.array([xy[0] - pb[0], xy[1] - pb[1], 0.0])
            e_b = Rb.T @ e_w
            ey = _wrap(yaw - self.base_yaw())
            if np.linalg.norm(e_b[:2]) < tol and abs(ey) < yaw_tol:
                ok += 1
                if ok >= 3:
                    break
                cmd = (0.0, 0.0, 0.0)
            else:
                ok = 0
                # the base actuators have a deadband near |cmd| = 0.2
                n = np.linalg.norm(e_b[:2])
                mag = min(vmax, 0.27 + 2.5 * n) if n > tol * 0.5 else 0.0
                v = e_b[:2] / max(n, 1e-9) * mag
                w = float(np.sign(ey) * min(vmax, 0.1 + 1.5 * abs(ey))) if abs(ey) > yaw_tol * 0.5 else 0.0
                cmd = (float(v[0]), float(v[1]), w)
            if carry:
                self.goal_pos = pb + Rb @ rel_p
                self.goal_R = Rb @ rel_R
            self.step(base=cmd)
        if carry:
            self.goal_pos = self.base_pos() + self.base_R() @ rel_p
            self.goal_R = self.base_R() @ rel_R
        self.wait(2)
        pb = self.base_pos()
        return np.linalg.norm(pb[:2] - xy) < 2 * tol + 0.02


    # ---- snapshots (development only; the recorded run never restores) ----
    def snapshot(self):
        oven = self.env.toaster_oven
        return {
            "sim": np.array(self.sim.get_state().flatten()),
            "tracker": copy.deepcopy(self.env.tracker),
            "arm_goal": (copy.deepcopy(self.arm.goal_pos), copy.deepcopy(self.arm.goal_ori)),
            "parts": {
                k: copy.deepcopy(getattr(c, "goal_qpos", None)) for k, c in self.cc.part_controllers.items()
            },
            "oven_t": getattr(oven, "_last_time_update", None),
            "drv": (self.steps, self.grip, self.goal_pos.copy(), self.goal_R.copy(), list(self.latched)),
            "timestep": self.env.timestep,
            "grip_action": copy.deepcopy(getattr(self.robot.gripper["right"], "current_action", None)),
        }

    def restore(self, snap):
        self.sim.set_state_from_flattened(snap["sim"])
        self.sim.forward()
        self.env.tracker = copy.deepcopy(snap["tracker"])
        self.arm.goal_pos, self.arm.goal_ori = copy.deepcopy(snap["arm_goal"])
        for k, c in self.cc.part_controllers.items():
            if snap["parts"].get(k) is not None:
                c.goal_qpos = copy.deepcopy(snap["parts"][k])
        self.env.toaster_oven._last_time_update = snap["oven_t"]
        self.steps, self.grip, self.goal_pos, self.goal_R, latched = snap["drv"]
        self.goal_pos = self.goal_pos.copy()
        self.goal_R = self.goal_R.copy()
        self.latched = list(latched)
        self.env.timestep = snap["timestep"]
        if snap["grip_action"] is not None:
            self.robot.gripper["right"].current_action = copy.deepcopy(snap["grip_action"])
        self.env.done = False
        self.done = False
        self.env.update_state()


def grip_R(approach, finger_axis):
    """Gripper-site rotation: site z along `approach`, fingers open along site x."""
    z = np.asarray(approach, dtype=float)
    z = z / np.linalg.norm(z)
    x = np.asarray(finger_axis, dtype=float)
    x = x - z * (x @ z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


class Chef:
    """The lemon-tart script, built from Driver primitives."""

    def __init__(self, driver):
        self.d = driver
        self.env = driver.env
        # gripper pose at reset, relative to the base
        self.home_rel = np.array([0.2535, -0.0123, 0.5895])
        self.home_R = np.array([[0.04, 0.97, 0.25], [1.0, -0.03, -0.02], [-0.01, 0.25, -0.97]])
        u, _, vt = np.linalg.svd(self.home_R)
        self.home_R = u @ vt

    # ---------- the script ----------
    # where things are set down on the counters (x, y)
    SPOTS = {
        "lemon": (3.92, -0.50),
        "butter": (4.08, -0.52),
        "egg": (3.76, -0.50),
        "flour": (0.70, -0.52),
        "sugar": (0.60, -0.58),
    }
    FRIDGE_PITCH = {"lemon": 30, "butter": 30, "egg": 25}

    def stage_ingredients_out(self):
        d = self.d
        homes = self.env.evaluator_state()["homes"]
        self.open_fridge()
        self.carry(z=1.15)
        for name in ("lemon", "butter", "egg"):
            self.fridge_take(name)
        self.close_fridge()
        for name in ("flour", "sugar"):
            self.take_from_drawer(homes[name], name, self.SPOTS[name])

    def fridge_take(self, name):
        d = self.d
        self.drive(min(4.48, d.obj_pos(name)[0] - 0.02), -1.12)
        if name == "egg":
            d.torso_to(0.12)  # the top shelf needs a little more height
        self.pick(name, self.FRIDGE_PITCH[name], heading="long")
        self.carry(fwd=0.42)
        if name == "egg":
            d.torso_to(0.0)
        spot = self.SPOTS[name]
        self.drive(spot[0], -1.15)
        z = self.COUNTER_Z - self.env.objects[name].bottom_offset[2] + 0.002
        self.place(name, [spot[0], spot[1], z])
        self.carry(z=1.15, fwd=0.40)

    # ---------- fridge ----------
    FRIDGE_HINGE = "fridge_main_group_fridge_door_joint"
    HANDLE_PITCH = np.deg2rad(25)

    def fridge_geometry(self):
        d = self.d
        m = d.sim.model
        jid = m.joint_name2id(self.FRIDGE_HINGE)
        hinge = np.array(d.sim.data.xanchor[jid])
        handle = d.geom_pos("fridge_main_group_fridge_door_handle_main")
        theta = d.joint(self.FRIDGE_HINGE)
        return hinge, handle, theta

    def fridge_handle_pose(self, theta, z, hinge, handle0):
        """World pose of the gripper holding the handle bar at door angle theta."""
        rel = handle0[:2] - hinge[:2]
        c, s = np.cos(theta), np.sin(theta)
        xy = hinge[:2] + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
        cp, sp = np.cos(self.HANDLE_PITCH), np.sin(self.HANDLE_PITCH)
        approach = np.array([-s * cp, c * cp, -sp])
        R = grip_R(approach, [-c, -s, 0.0])
        return np.array([xy[0], xy[1], z]) + approach * 0.002, R

    def open_fridge(self):
        d = self.d
        hinge, handle, theta = self.fridge_geometry()
        # geometry of the closed door: remove current angle from the handle position
        rel = handle[:2] - hinge[:2]
        c, s = np.cos(-theta), np.sin(-theta)
        handle0 = np.array([hinge[0] + c * rel[0] - s * rel[1], hinge[1] + s * rel[0] + c * rel[1], handle[2]])
        z = 1.30
        d.say("fridge: drive to the door")
        d.base_goto([handle0[0] - 0.08, handle0[1] - 0.72], np.pi / 2)
        d.set_gripper(False)
        p, R = self.fridge_handle_pose(theta, z, hinge, handle0)
        approach = R[:, 2]
        d.say("fridge: reach the handle")
        d.arm_to(p - approach * 0.10, R, max_steps=150)
        d.arm_line(p, R, speed=0.008, max_steps=60)
        w = d.set_gripper(True)
        d.say(f"fridge: handle gripped, finger width {w:.3f}")
        # pull along the door arc
        target = np.deg2rad(42)
        th = theta
        for _ in range(400):
            cur = d.joint(self.FRIDGE_HINGE)
            if cur >= target:
                break
            th = min(cur + np.deg2rad(4.0), target + np.deg2rad(2))
            p, R = self.fridge_handle_pose(th, z, hinge, handle0)
            d.goal_pos, d.goal_R = p, R
            d.step()
        d.say(f"fridge: pulled to {np.rad2deg(d.joint(self.FRIDGE_HINGE)):.1f} deg")
        d.hold_here()
        d.set_gripper(False)
        # let go, back away from the handle, and tuck the arm top-down
        d.arm_line(d.eef_pos() - d.eef_R()[:, 2] * 0.10, speed=0.012, max_steps=40)
        back = d.eef_pos()
        back[1] = min(back[1], d.base_pos()[1] + 0.20)
        back[2] = 1.05
        d.arm_to(back, self.DOWN_Y, max_steps=120, tol=0.02)
        # push the door the rest of the way from its inner face
        self.fridge_push(open_to=np.deg2rad(80), hinge=hinge)
        return d.joint(self.FRIDGE_HINGE)

    DOWN_Y = grip_R([0, 0, -1.0], [0, -1.0, 0])  # top-down, fingers open along world y
    DOWN_X = grip_R([0, 0, -1.0], [-1.0, 0, 0])  # top-down, fingers open along world x

    def tuck(self, z=1.05, fwd=0.42, R=None):
        """Bring the gripper to a compact top-down pose in front of the base."""
        d = self.d
        Rb = d.base_R()
        p = d.base_pos() + Rb @ np.array([fwd, 0.0, 0.0])
        p[2] = z
        if R is None:
            R = Rb @ grip_R([0, 0, -1.0], [0, 1.0, 0])
        return d.arm_to(p, R, max_steps=120, tol=0.02)

    def home(self, max_steps=90):
        """Return the empty arm to its reset pose over the base. Starting every
        manipulation from here keeps the joint configuration repeatable."""
        d = self.d
        self.unlock_elbow()
        Rb = d.base_R()
        p = d.base_pos() + Rb @ self.home_rel
        return d.arm_to(p, Rb @ self.home_R, max_steps=max_steps, tol=0.02, rot_tol=0.1, quiet=True)

    def elbow(self):
        return float(self.d.sim.data.qpos[self.d.robot._ref_joint_pos_indexes[3]])

    def unlock_elbow(self):
        """A long reach can carry the elbow through full extension onto its
        joint stop. From there the task-space controller pushes it further into
        the stop whenever the hand is asked to come back. Reach outwards to pass
        back through full extension, then pull in."""
        d = self.d
        if self.elbow() < -0.45:
            return True
        d.say(f"elbow at its stop ({self.elbow():.2f} rad), unlocking")
        shoulder = d.base_pos() + np.array([0, 0, 0.333])
        lead = d.lead
        for n_out in (8, 25, 14, 20, 30, 10):
            u = d.eef_pos() - shoulder
            u = u / np.linalg.norm(u)
            p0 = d.eef_pos().copy()
            d.lead = 0.1 if n_out < 20 else 0.2
            for _ in range(n_out):
                d.goal_pos = p0 + u * 0.3
                d.step()
            for _ in range(50):
                d.goal_pos = shoulder + u * 0.45
                d.step()
                if self.elbow() < -1.3:
                    break
            if self.elbow() < -1.0:
                break
        d.lead = lead
        d.hold_here()
        if self.elbow() >= -1.0:
            raise Failed("elbow stayed on its stop")
        return True

    def carry(self, z=None, fwd=0.45, speed=0.015):
        """Pull the gripper straight back to a point in front of the base, keeping its orientation."""
        d = self.d
        p = d.base_pos() + d.base_R() @ np.array([fwd, 0.0, 0.0])
        p[2] = d.eef_pos()[2] if z is None else z
        return d.arm_line(p, d.goal_R, speed=speed, max_steps=120, tol=0.02)

    LANE_Y = -1.45  # the base travels along this line, clear of counters and open doors

    def drive(self, x, y, yaw=np.pi / 2, lane=None):
        """Back out to the lane, travel along it, then approach the stance."""
        d = self.d
        lane = self.LANE_Y if lane is None else lane
        pb = d.base_pos()
        if abs(pb[0] - x) > 0.04:
            if pb[1] > lane + 0.06:
                d.base_goto([pb[0], lane], d.base_yaw(), tol=0.05, yaw_tol=0.1)
            d.base_goto([x, min(lane, y)], yaw, tol=0.05, yaw_tol=0.06)
        return d.base_goto([x, y], yaw)

    def fridge_face_point(self, theta, hinge, r, off, z):
        """Point at radius r on the fridge door, `off` metres to the inside (+) or outside (-)."""
        c, s = np.cos(theta), np.sin(theta)
        xy = hinge[:2] + r * np.array([-c, -s]) + off * np.array([-s, c])
        return np.array([xy[0], xy[1], z])

    def fridge_push(self, open_to, hinge):
        d = self.d
        z = 0.86
        theta = d.joint(self.FRIDGE_HINGE)
        d.say("fridge: move round the door edge to push it from inside")
        d.base_goto([4.30, -1.65], np.pi / 2)
        edge = hinge[:2] + 0.78 * np.array([-np.cos(theta), -np.sin(theta)])
        R = self.DOWN_Y
        w1 = np.array([edge[0] - 0.10, edge[1] - 0.06, 1.0])
        d.arm_to(w1, R, max_steps=120, tol=0.02)
        d.arm_to([w1[0], w1[1], z], R, max_steps=60, tol=0.02)
        inside = self.fridge_face_point(theta, hinge, 0.60, 0.12, z)
        d.arm_to([w1[0], inside[1] + 0.05, z], R, max_steps=80, tol=0.02)
        d.arm_to(inside, R, max_steps=80, tol=0.02)
        for _ in range(300):
            cur = d.joint(self.FRIDGE_HINGE)
            if cur >= open_to:
                break
            d.goal_pos = self.fridge_face_point(min(cur + np.deg2rad(8), open_to + 0.1), hinge, 0.60, 0.03, z)
            d.step()
        d.say(f"fridge: pushed open to {np.rad2deg(d.joint(self.FRIDGE_HINGE)):.1f} deg")
        # leave the door alone: step back inside, then lift
        cur = d.joint(self.FRIDGE_HINGE)
        d.arm_to(self.fridge_face_point(cur, hinge, 0.60, 0.14, z), R, max_steps=60, tol=0.02)
        d.arm_to(self.fridge_face_point(cur, hinge, 0.60, 0.14, 1.05), R, max_steps=60, tol=0.02)

    def close_fridge(self):
        d = self.d
        hinge, _, theta = self.fridge_geometry()
        z = 1.15
        r = 0.55
        R = self.DOWN_Y
        if theta > np.deg2rad(50):
            d.say("fridge: go round to the outside of the door")
            self.drive(4.75, -2.0)
            d.arm_to([4.75, -1.62, z], R, max_steps=80, tol=0.03)
            d.arm_to([5.13, -1.72, z], R, max_steps=80, tol=0.03)
            d.arm_to([5.13, hinge[1] - r, z], R, max_steps=80, tol=0.03)
            self.fridge_sweep_closed(hinge, r, z, np.deg2rad(40))
            cur = d.joint(self.FRIDGE_HINGE)
            d.arm_to(self.fridge_face_point(cur, hinge, r, -0.12, z), R, max_steps=40, tol=0.03)
            self.carry(z=z, fwd=0.40)
        d.base_goto([4.42, -1.55], np.pi / 2)
        cur = d.joint(self.FRIDGE_HINGE)
        d.arm_to(self.fridge_face_point(cur, hinge, r, -0.10, z), R, max_steps=80, tol=0.03)
        self.fridge_sweep_closed(hinge, r, z, 0.0)
        # lean on it until the hinge is at its stop
        for _ in range(60):
            if d.joint(self.FRIDGE_HINGE) < 0.004:
                break
            d.goal_pos = self.fridge_face_point(0.0, hinge, r, 0.02, z)
            d.step()
        d.say(f"fridge: closed to {np.rad2deg(d.joint(self.FRIDGE_HINGE)):.2f} deg")
        d.arm_to(self.fridge_face_point(0.0, hinge, r, -0.15, z), R, max_steps=40, tol=0.03)
        return d.joint(self.FRIDGE_HINGE) < 0.007

    def fridge_sweep_closed(self, hinge, r, z, stop):
        d = self.d
        for _ in range(300):
            cur = d.joint(self.FRIDGE_HINGE)
            if cur <= stop + 0.003:
                break
            d.goal_pos = self.fridge_face_point(max(cur - np.deg2rad(8), stop - 0.05), hinge, r, -0.035, z)
            d.step()

    # ---------- drawers ----------
    def drawer_state(self, name):
        """(handle position, metres pulled out)."""
        d = self.d
        return d.geom_pos(f"{name}_door_handle_g1"), -d.joint(f"{name}_slidejoint")

    def drawer_pitch(self, handle_z):
        # low handles need a steeper wrist; under the counter lip a flatter one
        return np.deg2rad(50 if handle_z > 0.7 else (62 if handle_z > 0.5 else 75))

    def hook_R(self, pitch):
        return grip_R([0.0, np.cos(pitch), -np.sin(pitch)], [-1.0, 0.0, 0.0])

    def open_drawer(self, name, pull=0.45, stance_gap=0.55):
        """Hook the closed fingertips behind the handle bar and back the base away."""
        d = self.d
        d.say(f"drawer {name}: open")
        handle, out = self.drawer_state(name)
        self.home()
        if handle[2] < 0.5:
            stance_gap = 0.46  # low handles are only reachable close in
        self.drive(handle[0], handle[1] - stance_gap)
        handle, out = self.drawer_state(name)
        pitch = self.drawer_pitch(handle[2])
        R = self.hook_R(pitch)
        approach = R[:, 2]
        d.set_gripper(True, steps=6)
        tip = handle + np.array([0.0, 0.020, -0.004])
        d.arm_to(tip - approach * 0.09, R, max_steps=130, tol=0.015, quiet=True)
        seated = d.arm_line(tip, R, speed=0.008, max_steps=40, tol=0.008)
        if not seated:
            d.say(f"drawer {name}: hook not seated, err {np.round(d.eef_pos() - tip, 3)}")
        # back the base away until the drawer itself says it is out far enough
        pb = d.base_pos()
        rel_p = d.base_R().T @ (d.goal_pos - pb)
        rel_R = d.base_R().T @ d.goal_R
        for _ in range(220):
            _, out = self.drawer_state(name)
            if out >= pull or d.base_pos()[1] < pb[1] - pull - 0.12:
                break
            d.goal_pos = d.base_pos() + d.base_R() @ rel_p
            d.goal_R = d.base_R() @ rel_R
            ey = _wrap(np.pi / 2 - d.base_yaw())
            d.step(base=(-0.45, 0.0, float(np.clip(2.0 * ey, -0.3, 0.3))))
        d.wait(2)
        handle, out = self.drawer_state(name)
        d.say(f"drawer {name}: pulled out {out:.3f} m")
        # unhook: forward a little, then up and back
        d.arm_line(d.eef_pos() + np.array([0, 0.012, 0.0]), R, speed=0.006, max_steps=10, tol=0.01, quiet=True)
        d.arm_line(d.eef_pos() - approach * 0.10, R, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        if out < pull - 0.08:
            raise Failed(f"drawer {name} only opened {out:.3f}")
        return out

    def close_drawer(self, name, stance_gap=0.50):
        """Push the handle bar back in with the closed fingertips, base driving forward."""
        d = self.d
        d.say(f"drawer {name}: close")
        handle, out = self.drawer_state(name)
        self.home()
        self.drive(handle[0], handle[1] - stance_gap)
        handle, out = self.drawer_state(name)
        pitch = self.drawer_pitch(handle[2]) if handle[2] > 0.5 else np.pi / 2
        R = self.hook_R(pitch)
        d.set_gripper(True, steps=6)
        if handle[2] > 0.5:
            tip = handle + np.array([0.0, -0.030, 0.0])
        else:
            # low drawer: push the panel itself, beside the handle, so the wrist
            # stays clear of the handle of the drawer above
            tip = handle + np.array([0.17, 0.037 - 0.012, -0.022])
        # come down in front of the drawer panel, never across it
        d.arm_to(tip + np.array([0, -0.04, 0.18]), R, max_steps=130, tol=0.03, quiet=True)
        d.arm_line(tip + np.array([0, -0.02, 0.0]), R, speed=0.012, max_steps=40, tol=0.01, quiet=True)
        d.arm_line(tip, R, speed=0.008, max_steps=20, tol=0.008, quiet=True)
        pb = d.base_pos()
        d.base_goto([pb[0], pb[1] + out], np.pi / 2, vmax=0.5, tol=0.015)
        for _ in range(60):
            _, out = self.drawer_state(name)
            if out < 0.001:
                break
            d.goal_pos = np.array([tip[0], d.eef_pos()[1] + 0.04, tip[2]])
            d.step()
        d.hold_here()
        d.arm_line(d.eef_pos() + np.array([0, -0.08, 0.05]), R, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        _, out = self.drawer_state(name)
        d.say(f"drawer {name}: closed, {out * 1000:.1f} mm out")
        return out < 0.003

    def take_from_drawer(self, drawer, name, spot, side_x=None):
        """Open the drawer, lift `name` out onto the counter at `spot` (x, y), close the drawer.

        The counter is out of reach from in front of an open drawer, so the
        robot carries the object round to stand beside the drawer.
        """
        d = self.d
        self.open_drawer(drawer, pull=0.52)
        handle, _ = self.drawer_state(drawer)
        self.home()
        d.base_goto([handle[0], handle[1] - 0.20], np.pi / 2)
        obj = self.env.objects[name]
        R = self.pick(
            name,
            pitch_deg=90,
            heading=self.drawer_heading(drawer, d.obj_pos(name)),
            offset=(0, 0, 0.3 * obj.size[2]) if obj.size[2] > 0.05 else (0, 0, 0),
            lift=0.05,
            back=0.10,
        )
        # out from under the drawers above, then up over the counter height
        d.arm_line(d.eef_pos() + np.array([0, -0.06, 0.06]), speed=0.012, max_steps=30, tol=0.02, quiet=True)
        d.arm_line([d.eef_pos()[0], d.eef_pos()[1], 1.08], speed=0.015, max_steps=70, tol=0.02)
        if not self.held(name, 0.09):
            raise Failed(f"lost {name} while lifting it out of the drawer")
        self.carry(fwd=0.40)
        side_x = handle[0] + 0.55 if side_x is None else side_x
        d.base_goto([side_x, d.base_pos()[1]], np.pi / 2, tol=0.04)
        d.base_goto([side_x, -0.92], np.pi / 2)
        z = self.COUNTER_Z - obj.bottom_offset[2] + 0.002
        self.place(name, [spot[0], spot[1], z], retreat=(0, 0, 0.12))
        self.home()
        self.close_drawer(drawer)

    def drawer_heading(self, drawer, pos, half_hand=0.105):
        """Top-down heading whose finger axis keeps the 21 cm wide hand off the drawer walls."""
        fixture = self.env._drawer(drawer)
        p0, px, py, _ = [np.asarray(q, float) for q in fixture.get_int_sites(relative=False)["int"]]
        best, best_clear = (0.0, 1.0), -1.0
        for deg in (0, 45, -45, 90):
            a = np.deg2rad(deg)
            h = np.array([np.sin(a), np.cos(a)])
            f = np.array([-h[1], h[0]])  # finger axis = z x heading
            clear = min(
                pos[0] - half_hand * abs(f[0]) - p0[0],
                px[0] - pos[0] - half_hand * abs(f[0]),
                pos[1] - half_hand * abs(f[1]) - p0[1],
                py[1] - pos[1] - half_hand * abs(f[1]),
            )
            if clear > best_clear + 0.005:
                best, best_clear = (h[0], h[1]), clear
        return best

    COUNTER_Z = 0.92

    # ---------- dishwasher ----------
    DW_DOOR = "dishwasher_main_group_door_joint"
    DW_RACK = "dishwasher_main_group_rack1_joint"

    def dw_handle_pose(self, theta, hinge, radius, x, pitch0):
        """Gripper pose pinching the door handle when the door is open by theta."""
        # the door drops forward about a hinge along x at its bottom edge
        y = hinge[1] - radius * np.sin(theta + self.dw_phi0)
        z = hinge[2] + radius * np.cos(theta + self.dw_phi0)
        pitch = pitch0 + theta
        approach = np.array([0.0, np.cos(pitch), -np.sin(pitch)])
        finger = np.array([0.0, np.sin(pitch), np.cos(pitch)])
        return np.array([x, y, z]), grip_R(approach, finger)

    def open_dishwasher(self):
        d = self.d
        m = d.sim.model
        hinge = np.array(d.sim.data.xanchor[m.joint_name2id(self.DW_DOOR)])
        handle = d.geom_pos("dishwasher_main_group_door_handle_main")
        theta = d.joint(self.DW_DOOR)
        rel = handle - hinge
        radius = float(np.hypot(rel[1], rel[2]))
        self.dw_phi0 = float(np.arctan2(-rel[1], rel[2])) - theta
        d.say("dishwasher: open the door")
        self.home()
        self.drive(handle[0], -1.45)
        pitch0 = np.deg2rad(30)
        p, R = self.dw_handle_pose(theta, hinge, radius, handle[0], pitch0)
        d.set_gripper(False, steps=6)
        d.arm_to(p - R[:, 2] * 0.10, R, max_steps=130, tol=0.02)
        d.arm_line(p, R, speed=0.008, max_steps=40, tol=0.008)
        w = d.set_gripper(True)
        d.say(f"dishwasher: handle gripped, width {w:.3f}")
        top = m.jnt_range[m.joint_name2id(self.DW_DOOR)][1]
        d.lead = 0.09
        for _ in range(300):
            cur = d.joint(self.DW_DOOR)
            if cur >= top - 0.02:
                break
            p, R = self.dw_handle_pose(min(cur + 0.12, top + 0.05), hinge, radius, handle[0], pitch0)
            d.goal_pos, d.goal_R = p, R
            d.step()
        d.lead = 0.05
        d.hold_here()
        d.set_gripper(False)
        d.say(f"dishwasher: door at {d.joint(self.DW_DOOR):.3f} of {top:.3f} rad")
        d.arm_line(d.eef_pos() - d.eef_R()[:, 2] * 0.10, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        return d.joint(self.DW_DOOR) >= top - 0.05

    def pull_rack(self, hook_x=None):
        """Slide the upper rack out. The opening is too low for the wrist to
        reach deep, so the fingertips catch the top of the rack's front wall
        and keep pressing down while pulling."""
        d = self.d
        m = d.sim.model
        d.say("dishwasher: pull the rack out")
        wall = d.geom_pos("dishwasher_main_group_g64")  # front wall of the rack
        top = m.jnt_range[m.joint_name2id(self.DW_RACK)][1]
        if hook_x is None:
            # the free stretch of wall, left of the propped flat dishes
            lefts = [
                d.obj_pos(n)[0] - 0.5 * max(self.env.objects[n].size[:2])
                for n in ("tart_pan", "baking_sheet")
                if abs(d.obj_pos(n)[1] - wall[1]) < 0.12
            ]
            hook_x = (min(lefts) - 0.125) if lefts else 2.02
            hook_x = max(hook_x, 1.925)
        self.home()
        d.base_goto([hook_x + 0.02, wall[1] - 0.75], np.pi / 2)
        a = np.deg2rad(45)
        R = grip_R([0, np.cos(a), -np.sin(a)], [-1.0, 0, 0])
        d.set_gripper(True, steps=6)
        over = np.array([hook_x, wall[1] + 0.024, wall[2] + 0.076])
        tip = over + np.array([0, 0, -0.06])
        d.arm_to(over - R[:, 2] * 0.12, R, max_steps=130, tol=0.02, quiet=True)
        d.arm_line(over, R, speed=0.008, max_steps=50, tol=0.01, quiet=True)
        d.arm_line(tip, R, speed=0.006, max_steps=30, tol=0.01, quiet=True)
        start = d.eef_pos().copy()
        for i in range(35):
            d.goal_pos = np.array([start[0], start[1] - 0.004 * (i + 1) - 0.02, start[2] - 0.04])
            d.step()
        d.say(f"dishwasher: rack out {d.joint(self.DW_RACK):.3f} m after the first pull")
        d.hold_here()
        d.arm_line(d.eef_pos() + np.array([0, 0.01, 0.10]), R, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        # the wall is now clear of the dishwasher ceiling: hook it properly
        for _ in range(2):
            if d.joint(self.DW_RACK) >= top - 0.01:
                break
            wall = d.geom_pos("dishwasher_main_group_g64")
            d.base_goto([hook_x + 0.02, wall[1] - 0.68], np.pi / 2)
            wall = d.geom_pos("dishwasher_main_group_g64")
            a = np.deg2rad(65)
            R = grip_R([0, np.cos(a), -np.sin(a)], [-1.0, 0, 0])
            tip = np.array([hook_x, wall[1] + 0.022, wall[2] + 0.005])
            d.arm_to(tip + np.array([0, 0.0, 0.10]), R, max_steps=130, tol=0.02, quiet=True)
            d.arm_line(tip, R, speed=0.008, max_steps=40, tol=0.01, quiet=True)
            start = d.eef_pos().copy()
            need = top - d.joint(self.DW_RACK) + 0.03
            for i in range(120):
                if d.joint(self.DW_RACK) >= top - 0.004:
                    break
                d.goal_pos = np.array([start[0], start[1] - min(0.005 * (i + 1), need), start[2]])
                d.step()
            d.hold_here()
            d.arm_line(d.eef_pos() + np.array([0, 0.012, 0.0]), R, speed=0.006, max_steps=10, tol=0.01, quiet=True)
            d.arm_line(d.eef_pos() + np.array([0, 0.0, 0.14]), R, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        d.say(f"dishwasher: rack out {d.joint(self.DW_RACK):.3f} of {top:.3f} m")
        return d.joint(self.DW_RACK) >= top - 0.05

    # ---------- toaster oven ----------
    OVEN_DOOR = "toaster_oven_main_group_door_joint"
    OVEN_RACK = "toaster_oven_main_group_rack0_joint"
    OVEN_X = 2.44

    def oven_door_geom(self):
        d = self.d
        m = d.sim.model
        hinge = np.array(d.sim.data.xanchor[m.joint_name2id(self.OVEN_DOOR)])
        top = m.jnt_range[m.joint_name2id(self.OVEN_DOOR)][1]
        return hinge, top

    def open_oven(self):
        """Hook the door handle from above, swing the door down, then press it flat."""
        d = self.d
        hinge, top = self.oven_door_geom()
        d.say("oven: open the door")
        self.home()
        self.drive(self.OVEN_X, -1.13)
        handle = d.geom_pos("toaster_oven_main_group_door_handle_main")
        R = self.hook_R(np.deg2rad(60))
        approach = R[:, 2]
        d.set_gripper(True, steps=6)
        tip = handle + np.array([0.0, 0.021, -0.006])
        d.arm_to(tip - approach * 0.09, R, max_steps=130, tol=0.015, quiet=True)
        d.arm_line(tip, R, speed=0.008, max_steps=40, tol=0.008)
        rel = tip - hinge
        radius = float(np.hypot(rel[1], rel[2]))
        phi0 = float(np.arctan2(-rel[1], rel[2])) - d.joint(self.OVEN_DOOR)
        for _ in range(200):
            cur = d.joint(self.OVEN_DOOR)
            if cur >= 1.0:
                break
            th = cur + 0.25
            d.goal_pos = np.array([tip[0], hinge[1] - radius * np.sin(th + phi0) - 0.01, hinge[2] + radius * np.cos(th + phi0)])
            d.step()
        d.say(f"oven: door pulled to {d.joint(self.OVEN_DOOR):.2f} rad")
        # slip out from behind the bar, then press the door down from above
        d.hold_here()
        d.arm_line(d.eef_pos() + np.array([0, 0.0, 0.10]), R, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        for _ in range(3):
            cur = d.joint(self.OVEN_DOOR)
            if cur >= top - 0.03:
                break
            # a point on the upper face of the door, two thirds of the way out
            ang = cur + phi0
            p = np.array([tip[0], hinge[1] - 0.13 * np.sin(ang), hinge[2] + 0.13 * np.cos(ang)])
            d.arm_to(p + np.array([0, 0.03, 0.07]), R, max_steps=60, tol=0.02, quiet=True)
            for _ in range(60):
                cur = d.joint(self.OVEN_DOOR)
                if cur >= top - 0.02:
                    break
                ang = min(cur + 0.3, top + 0.1) + phi0
                d.goal_pos = np.array([tip[0], hinge[1] - 0.13 * np.sin(ang) + 0.01, hinge[2] + 0.13 * np.cos(ang) + 0.012])
                d.step()
            d.hold_here()
            d.arm_line(d.eef_pos() + np.array([0, -0.02, 0.12]), R, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        d.say(f"oven: door at {d.joint(self.OVEN_DOOR):.2f} of {top:.2f} rad")
        return d.joint(self.OVEN_DOOR) >= top - 0.1

    def oven_tray(self):
        """(front-edge centre of the oven tray, metres slid out, travel)."""
        d = self.d
        m = d.sim.model
        g = d.geom_pos("toaster_oven_main_group_g42")
        half = m.geom_size[m.geom_name2id("toaster_oven_main_group_g42")]
        top = m.jnt_range[m.joint_name2id(self.OVEN_RACK)][1]
        return np.array([g[0], g[1] - half[1], g[2] + half[2]]), d.joint(self.OVEN_RACK), top

    TRAY_STANCE = (2.50, -1.13)

    def slide_oven_tray(self, out, x=2.50):
        """Slide the tray out (True) or in (False). It has no handle: going out,
        the fingertips press down on it and drag; going in, they push its front edge."""
        d = self.d
        self.home()
        self.drive(*self.TRAY_STANCE)
        edge, cur, top = self.oven_tray()
        # straight in, wrist pitched 35 degrees: flat enough to pass under the
        # knob panel, steep enough to keep the forearm off the open door
        h = np.array([0.0, 1.0, 0.0])
        R = self.hook_R(np.deg2rad(35))
        d.set_gripper(True, steps=6)
        if out:
            d.say("oven: drag the tray out")
            press = np.array([x, edge[1] + 0.03, edge[2] + 0.004])
            # line up outside the oven mouth first, then go straight in along the wrist axis
            d.arm_to(press - h * 0.30 + np.array([0, 0, 0.12]), R, max_steps=130, tol=0.015, quiet=True)
            d.arm_line(press + np.array([0, 0, 0.05]) - h * 0.02, R, speed=0.01, max_steps=50, tol=0.012, quiet=True)
            d.arm_line(press, R, speed=0.006, max_steps=30, tol=0.006, quiet=True)
            start = d.eef_pos().copy()
            y0 = d.joint(self.OVEN_RACK)
            for i in range(140):
                moved = d.joint(self.OVEN_RACK) - y0
                if d.joint(self.OVEN_RACK) >= top - 0.004:
                    break
                # lean on the tray, and only lead the pull a little so the tips do not slip off
                d.goal_pos = np.array([start[0], start[1] - moved - 0.03, press[2] - 0.035])
                d.step()
        else:
            d.say("oven: push the tray in")
            push = np.array([x, edge[1] - 0.014, edge[2] - 0.006])
            d.arm_to(push - h * 0.25 + np.array([0, 0, 0.12]), R, max_steps=130, tol=0.015, quiet=True)
            d.arm_line(push + np.array([0, -0.03, 0.05]), R, speed=0.01, max_steps=40, tol=0.012, quiet=True)
            d.arm_line(push + np.array([0, -0.015, 0.0]), R, speed=0.006, max_steps=30, tol=0.006, quiet=True)
            start = d.eef_pos().copy()
            y0 = d.joint(self.OVEN_RACK)
            for i in range(140):
                moved = y0 - d.joint(self.OVEN_RACK)
                if d.joint(self.OVEN_RACK) <= 0.003:
                    break
                d.goal_pos = np.array([start[0], start[1] + moved + 0.04, start[2]])
                d.step()
        d.hold_here()
        d.arm_line(d.eef_pos() + np.array([0, -0.03, 0.10]), R, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        _, cur, top = self.oven_tray()
        d.say(f"oven: tray {cur:.3f} of {top:.3f} m out")
        return cur

    KNOB_JOINT = {
        "function": "toaster_oven_main_group_knob_function_joint",
        "temperature": "toaster_oven_main_group_knob_temp_joint",
        "time": "toaster_oven_main_group_knob_time_joint",
    }
    KNOB_GEOM = {
        "function": "toaster_oven_main_group_knob_function_main",
        "temperature": "toaster_oven_main_group_knob_temp_main",
        "time": "toaster_oven_main_group_knob_time_main",
    }
    KNOB_STANCE = (2.47, -1.0)

    def knob_value(self, knob):
        """Normalised knob position, the same number the tracker thresholds."""
        d = self.d
        m = d.sim.model
        lo, hi = m.jnt_range[m.joint_name2id(self.KNOB_JOINT[knob])]
        return (d.joint(self.KNOB_JOINT[knob]) - lo) / (hi - lo)

    def turn_knob(self, knob, target, max_grasps=6):
        """Pinch the knob's fin and twist the wrist about the knob axis until the
        knob passes `target` (normalised). target > current turns it up, else down."""
        d = self.d
        m = d.sim.model
        lo, hi = m.jnt_range[m.joint_name2id(self.KNOB_JOINT[knob])]
        up = target > self.knob_value(knob)
        pitch = np.deg2rad(30)
        approach = np.array([0.0, np.cos(pitch), -np.sin(pitch)])
        R0 = grip_R(approach, [-1.0, 0, 0])
        for _ in range(max_grasps):
            val = self.knob_value(knob)
            if (up and val >= target) or (not up and val <= target):
                break
            theta = d.joint(self.KNOB_JOINT[knob])
            # the fin looks the same every half turn: grasp it at the equivalent
            # angle nearest the wrist's neutral roll, biased away from the turn
            g = (theta + np.pi / 2) % np.pi - np.pi / 2
            if up and g > 0.6:
                g -= np.pi
            if not up and g < -0.6:
                g += np.pi
            Rg = rot_y(g) @ R0
            c = d.geom_pos(self.KNOB_GEOM[knob])
            d.set_gripper(False, steps=6)
            # get in front of the knob with a neutral wrist, then roll in place:
            # asking for a rolled pose from far away sends the arm through a singularity
            pre = c - approach * 0.07
            d.arm_to(pre, R0, max_steps=100, tol=0.012, rot_tol=0.06, quiet=True)
            n_roll = int(abs(g) / 0.05) + 1
            for i in range(n_roll):
                d.goal_pos, d.goal_R = pre, rot_y(g * (i + 1) / n_roll) @ R0
                d.step()
            d.arm_to(pre, Rg, max_steps=40, tol=0.012, rot_tol=0.06, quiet=True)
            d.arm_line(c + approach * 0.002, Rg, speed=0.006, max_steps=30, tol=0.005, quiet=True)
            d.set_gripper(True)
            want = (target - val) * (hi - lo) + (0.12 if up else -0.12)
            swing = float(np.clip(want, -1.3, 1.3))
            start = d.joint(self.KNOB_JOINT[knob])
            for i in range(90):
                done = d.joint(self.KNOB_JOINT[knob]) - start
                if abs(done) >= abs(swing) - 0.01:
                    break
                lead = done + np.sign(swing) * 0.35
                lead = np.clip(lead, -abs(swing) - 0.15, abs(swing) + 0.15)
                d.goal_R = rot_y(g + lead) @ R0
                d.goal_pos = c + approach * 0.002
                d.step()
            d.hold_here()
            d.set_gripper(False)
            d.arm_line(d.eef_pos() - approach * 0.07, speed=0.012, max_steps=20, tol=0.02, quiet=True)
            # unwind the wrist before going anywhere else
            d.arm_to(pre, R0, max_steps=50, tol=0.02, rot_tol=0.1, quiet=True)
        val = self.knob_value(knob)
        d.say(f"knob {knob}: {val:.3f} (target {target:.2f})")
        return (up and val >= target) or (not up and val <= target)

    # ---------- pick and place ----------
    def held(self, name, tol=0.06):
        return np.linalg.norm(self.d.obj_pos(name) - self.d.eef_pos()) < tol

    def pick(self, name, pitch_deg=30.0, heading=(0.0, 1.0), offset=(0.0, 0.0, 0.0), back=0.12, lift=0.03, tries=4, long_ref=None, hold_tol=0.06):
        """Pinch `name` at its centre. The gripper comes in along `heading`
        (world xy), pitched down by pitch_deg, fingers horizontal."""
        d = self.d
        if heading == "long":
            # come in along the object's long horizontal axis so the fingers
            # straddle its narrow side
            size = np.asarray(self.env.objects[name].size)
            axis = d.obj_R(name)[:, int(np.argmax(size[:2]))]
            ref = np.array([0.0, 1.0]) if long_ref is None else np.asarray(long_ref, float)
            if axis[:2] @ ref < 0:
                axis = -axis
            heading = axis[:2]
        h = np.array([heading[0], heading[1], 0.0])
        h = h / np.linalg.norm(h)
        a = np.deg2rad(pitch_deg)
        approach = h * np.cos(a) + np.array([0, 0, -np.sin(a)])
        finger = np.cross([0, 0, 1.0], h)
        R = grip_R(approach, finger)
        for attempt in range(tries):
            c = d.obj_pos(name) + np.asarray(offset, dtype=float)
            pre = c - approach * back
            d.set_gripper(False, steps=6)
            d.arm_to(pre, R, max_steps=130, tol=0.03, quiet=True)
            d.arm_line(c + approach * 0.004, R, speed=0.008, max_steps=50, tol=0.006)
            w = d.set_gripper(True)
            up = d.eef_pos() + np.array([0, 0, lift])
            d.arm_line(up, R, speed=0.006, max_steps=30)
            if self.held(name, hold_tol) and w > 0.004:
                d.say(f"picked {name} (finger width {w:.3f})")
                self.grasp_R = R
                self.grasp_off = d.obj_pos(name) - d.eef_pos()
                d.arm_line(pre + np.array([0, 0, lift]), R, speed=0.012, max_steps=50, tol=0.02, quiet=True)
                return R
            d.say(f"pick {name} missed (width {w:.3f}), retry")
            d.set_gripper(False)
            d.arm_line(pre, R, speed=0.012, max_steps=40, tol=0.02, quiet=True)
            # the task-space controller keeps whatever elbow/wrist configuration
            # it drifted into; come back in front of the base to shake it out,
            # a different way each time
            if attempt % 2 == 0:
                self.carry(fwd=0.40, z=max(pre[2], 1.05))
                d.arm_to(d.goal_pos, R, max_steps=60, tol=0.03, quiet=True)
            else:
                self.carry(fwd=0.40, z=max(pre[2], 1.05))
                self.home()
        raise Failed(f"could not pick {name}")

    def pick_rim(self, name, radius, toward=(0.0, -1.0), pitch_deg=70.0, inset=0.015, x_shift=0.0, under=0.0, tries=2):
        """Tongs grasp for a flat dish: one finger over the rim, one under it.

        `toward` is the world xy direction from the dish centre to the rim point
        to take (the side facing the robot). The gripper comes in over that rim,
        pitched down, with the fingers opening in the vertical plane.
        """
        d = self.d
        t = np.array([toward[0], toward[1], 0.0])
        t = t / np.linalg.norm(t)
        h = -t  # horizontal approach direction, rim -> centre
        a = np.deg2rad(pitch_deg)
        approach = h * np.cos(a) + np.array([0, 0, -np.sin(a)])
        finger = h * np.sin(a) + np.array([0, 0, np.cos(a)])
        R = grip_R(approach, finger)
        for attempt in range(tries):
            Ro = d.obj_R(name)
            # rim point: project the wanted direction into the dish plane
            n = Ro[:, 2]
            u = t - n * (t @ n)
            u = u / np.linalg.norm(u)
            side = np.cross(n, u)
            c = d.obj_pos(name) + u * (radius - inset) + side * x_shift + np.array([0, 0, -under])
            pre = c - approach * 0.11
            d.set_gripper(False, steps=6)
            d.arm_to(pre, R, max_steps=130, tol=0.02, quiet=True)
            d.arm_line(c, R, speed=0.007, max_steps=50, tol=0.006)
            w = d.set_gripper(True)
            d.arm_line(d.eef_pos() + np.array([0, 0, 0.06]), R, speed=0.006, max_steps=30, quiet=True)
            if self.held(name, tol=radius + 0.05) and w > 0.003:
                d.say(f"picked {name} by the rim (finger width {w:.3f})")
                self.grasp_R = R
                self.grasp_off = d.obj_pos(name) - d.eef_pos()
                return R
            d.say(f"rim pick {name} missed (width {w:.3f}), retry")
            d.set_gripper(False)
            d.arm_line(pre, R, speed=0.012, max_steps=40, tol=0.02, quiet=True)
        raise Failed(f"could not pick {name} by the rim")

    def nudge(self, name, radius, delta, pitch_deg=60.0):
        """Slide a flat dish along `delta` (world xy) by pushing its rim with the closed fingertips."""
        d = self.d
        delta = np.array([delta[0], delta[1], 0.0])
        dist = np.linalg.norm(delta)
        u = delta / dist
        a = np.deg2rad(pitch_deg)
        R = grip_R(np.array([0, np.cos(a), -np.sin(a)]), [-1.0, 0, 0])
        c = d.obj_pos(name)
        start = c - u * (radius + 0.03)
        d.set_gripper(True, steps=6)
        d.arm_to(start + np.array([0, 0, 0.08]), R, max_steps=130, tol=0.02, quiet=True)
        d.arm_line(start, R, speed=0.008, max_steps=40, tol=0.01, quiet=True)
        d.arm_line(start + u * (dist + 0.03), R, speed=0.005, max_steps=60, tol=0.01, quiet=True)
        d.arm_line(d.eef_pos() - u * 0.02 + np.array([0, 0, 0.10]), R, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        d.say(f"nudged {name} by {np.round(d.obj_pos(name) - c, 3)}")

    POT_GRIP = np.array([0.0, -0.14, 0.052])  # on the saucepan handle, in the pan's frame

    def pick_pot(self, name="saucepan"):
        """Pinch the saucepan's handle from the handle end, wrist pitched down."""
        d = self.d
        Ro = d.obj_R(name)
        grip = Ro @ self.POT_GRIP
        h = np.array([Ro[0, 1], Ro[1, 1], 0.0])
        h = h / np.linalg.norm(h)
        a = np.deg2rad(60)
        approach = h * np.cos(a) + np.array([0, 0, -np.sin(a)])
        R = grip_R(approach, np.cross([0, 0, 1.0], h))
        c = d.obj_pos(name) + grip
        d.set_gripper(False, steps=6)
        d.arm_to(c - approach * 0.12, R, max_steps=130, tol=0.02, quiet=True)
        d.arm_line(c + approach * 0.004, R, speed=0.008, max_steps=50, tol=0.006, quiet=True)
        w = d.set_gripper(True)
        z0 = d.obj_pos(name)[2]
        d.arm_line(d.eef_pos() + np.array([0, 0, 0.10]), R, speed=0.006, max_steps=40, quiet=True)
        if d.obj_pos(name)[2] < z0 + 0.05:
            raise Failed(f"{name} did not come up with the handle (width {w:.3f})")
        d.say(f"picked {name} by the handle (finger width {w:.3f})")
        self.grasp_R = R
        return R

    def set_down(self, name, xy, floor_z, stop=None, max_drop=0.35):
        """Lower the held object over world (x, y) until it rests, then let go.

        `floor_z` is where its bounding-box bottom should end up; `stop` is an
        optional callable that ends the descent early (a milestone flag)."""
        d = self.d
        obj = self.env.objects[name]
        off = d.obj_pos(name) - d.eef_pos()
        start = np.array([xy[0], xy[1], floor_z + obj.size[2] + 0.06]) - off
        d.arm_to(start, max_steps=130, tol=0.02, quiet=True)
        last = d.obj_pos(name)[2]
        still = 0
        for i in range(int(max_drop / 0.004)):
            d.goal_pos = d.goal_pos + np.array([0, 0, -0.004])
            d.step()
            z = d.obj_pos(name)[2]
            still = still + 1 if z > last - 0.001 else 0
            last = z
            if still >= 6 or (stop is not None and stop()):
                break
        d.hold_here()
        d.set_gripper(False)
        d.wait(4)
        d.arm_line(d.eef_pos() - d.eef_R()[:, 2] * 0.10, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        d.say(f"set {name} down at {np.round(d.obj_pos(name), 3)}")

    def stage_curd(self):
        """Saucepan from the dishwasher rack onto the stove."""
        d = self.d
        self.home()
        grip = d.obj_R("saucepan") @ self.POT_GRIP
        p = d.obj_pos("saucepan") + grip
        d.base_goto([p[0], p[1] - 0.52], np.pi / 2)
        self.pick_pot()
        # out from under the counter edge before going up
        d.arm_line(d.eef_pos() + np.array([0, -0.16, 0.02]), speed=0.01, max_steps=40, tol=0.02, quiet=True)
        d.arm_line(d.eef_pos() + np.array([0, 0.0, 0.22]), speed=0.012, max_steps=40, tol=0.02, quiet=True)
        self.carry(fwd=0.40, z=1.30)  # the pan hangs 17 cm below the hand: keep it above the cooktop
        if d.obj_pos("saucepan")[2] < 0.8:
            raise Failed("dropped the saucepan")
        self.drive(self.STOVE_SPOT[0], -1.17)
        self.set_down("saucepan", self.STOVE_SPOT, 0.95)
        if d.obj_pos("saucepan")[2] < 0.9:
            raise Failed("the saucepan is not on the stove")
        self.home()

    STOVE_SPOT = (3.02, -0.48)

    DISH_X = 2.385  # where a dish sits on the oven tray: left of the stretch the fingers push
    PAN_SPOT = (3.60, -0.52)  # where the tart ends up on the counter
    KNOB_ON = 0.32
    KNOB_OFF = 0.03

    def rim_half(self, name, toward=(0.0, -1.0)):
        """Half-extent of a flat dish along a world direction."""
        d = self.d
        size = np.asarray(self.env.objects[name].size)
        if abs(size[0] - size[1]) < 1e-3:
            return 0.5 * size[0]
        t = np.array([toward[0], toward[1], 0.0])
        Ro = d.obj_R(name)
        return 0.5 * float(abs(Ro[:, 0] @ t) * size[0] + abs(Ro[:, 1] @ t) * size[1])

    def dish_to_tray(self, name):
        """Take a propped dish off the dishwasher rack and set it on the pulled-out oven tray."""
        d = self.d
        self.home()
        d.base_goto([d.obj_pos(name)[0], -1.25], np.pi / 2)
        half = self.rim_half(name)
        self.pick_rim(name, half)
        d.arm_line(d.eef_pos() + np.array([0, -0.05, 0.25]), speed=0.012, max_steps=60, tol=0.02, quiet=True)
        self.carry(fwd=0.42, z=1.25)
        d.base_goto([self.DISH_X + 0.015, -1.25], np.pi / 2)
        edge, _, _ = self.oven_tray()
        # front rim 3 cm past the tray edge, so it can be taken by the rim again
        self.lay_dish(name, half, self.DISH_X, edge[1] - 0.03, edge[2])
        d.say(f"{name} on the oven tray at {np.round(d.obj_pos(name), 3)}")

    def lay_dish(self, name, half, x, y_front, floor_z):
        """Set a rim-held dish down flat with its front rim at y_front.

        A small dish swings down and hangs from the pinch. Its far edge is put
        down first, then the hand swings back and down about that edge."""
        d = self.d
        g = d.eef_pos()
        cvec = d.obj_pos(name) - g
        dist = float(np.linalg.norm(cvec))
        beta = float(np.arctan2(-cvec[2], cvec[1]))  # how far the dish hangs below horizontal
        if beta < np.deg2rad(30):
            off = d.obj_pos(name) - g
            tgt = np.array([x, y_front + half, floor_z + 0.012])
            d.arm_to(tgt - off + np.array([0, 0, 0.06]), max_steps=120, tol=0.015, quiet=True)
            d.arm_line(tgt - off + np.array([0, 0, 0.004]), speed=0.006, max_steps=40, tol=0.008, quiet=True)
        else:
            reach = dist + half  # pinch point to far edge
            far = np.array([x, y_front + 2 * half, floor_z + 0.010])
            R0 = d.goal_R.copy()

            def hand(b):
                return far + reach * np.array([0.0, -np.cos(b), np.sin(b)])

            d.arm_to(hand(beta) + np.array([0, 0, 0.06]), max_steps=120, tol=0.015, quiet=True)
            d.arm_line(hand(beta), speed=0.006, max_steps=40, tol=0.008, quiet=True)
            end = np.deg2rad(32)
            n = max(1, int((beta - end) / np.deg2rad(2.0)))
            for i in range(n):
                b = beta - (beta - end) * (i + 1) / n
                d.goal_pos = hand(b)
                d.goal_R = rot_x(beta - b) @ R0
                d.step()
            d.wait(4)
        d.hold_here()
        d.set_gripper(False)
        d.wait(6)
        d.arm_line(d.eef_pos() - d.eef_R()[:, 2] * 0.10, speed=0.012, max_steps=30, tol=0.02, quiet=True)

    def dish_from_tray(self, name):
        """Take a dish off the pulled-out oven tray by its overhanging front rim."""
        d = self.d
        self.home()
        self.drive(d.obj_pos(name)[0] + 0.015, -1.25)
        self.pick_rim(name, self.rim_half(name), inset=0.02, under=0.012)
        d.arm_line(d.eef_pos() + np.array([0, -0.08, 0.12]), speed=0.012, max_steps=40, tol=0.02, quiet=True)
        self.carry(fwd=0.42, z=1.25)

    def knobs(self, on):
        self.home()
        self.drive(*self.KNOB_STANCE)
        for knob in ("function", "temperature", "time"):  # time last: it runs down by itself
            if not self.turn_knob(knob, self.KNOB_ON if on else self.KNOB_OFF):
                raise Failed(f"knob {knob} did not reach its target")

    def stage_crust(self):
        """Tart pan from the dishwasher into the oven, three knobs up, then back off."""
        self.open_oven()
        self.slide_oven_tray(True)
        self.open_dishwasher()
        self.pull_rack()
        self.dish_to_tray("tart_pan")
        self.slide_oven_tray(False)
        self.knobs(True)
        self.knobs(False)

    def stage_crumble(self):
        """Tart pan out to the counter, baking sheet in, the same three knobs up."""
        d = self.d
        self.slide_oven_tray(True)
        self.dish_from_tray("tart_pan")
        self.drive(self.PAN_SPOT[0], -1.15)
        self.set_down("tart_pan", self.PAN_SPOT, self.COUNTER_Z)
        self.dish_to_tray("baking_sheet")
        self.slide_oven_tray(False)
        self.knobs(True)

    def stage_assemble(self):
        """Lemon onto the tart pan on the counter, hand away, so the contact is scored."""
        d = self.d
        self.home()
        pan = d.obj_pos("tart_pan")
        self.drive(0.5 * (pan[0] + d.obj_pos("lemon")[0]), -1.15)
        self.pick("lemon", 30, heading="long")
        z = pan[2] + 0.006 - self.env.objects["lemon"].bottom_offset[2] + 0.004
        self.place("lemon", [pan[0], pan[1], z])
        self.home()
        d.wait(5)

    FRIDGE_SLOTS = {"lemon": (4.60, -0.47), "butter": (4.44, -0.50), "egg": (4.74, -0.50)}
    SHELF_Z = 1.078

    def stage_restore(self):
        """Every ingredient back where it came from, doors shut."""
        d = self.d
        homes = self.env.evaluator_state()["homes"]
        self.open_fridge()
        self.carry(z=1.15)
        for name in ("lemon", "butter", "egg"):
            p = d.obj_pos(name)
            self.drive(p[0], -1.15)
            self.pick(name, 30, heading="long")
            self.carry(fwd=0.42, z=1.18)
            slot = self.FRIDGE_SLOTS[name]
            self.drive(min(4.48, slot[0] - 0.02), -1.12)
            z = self.SHELF_Z - self.env.objects[name].bottom_offset[2] + 0.004
            self.place(name, [slot[0], slot[1], z])
            self.carry(fwd=0.42, z=1.18)
        self.close_fridge()
        for name in ("flour", "sugar"):
            self.put_in_drawer(homes[name], name)

    def put_in_drawer(self, drawer, name):
        """Open the drawer, fetch `name` from the counter, lower it in, close the drawer."""
        d = self.d
        self.open_drawer(drawer, pull=0.52)
        handle, _ = self.drawer_state(drawer)
        self.home()
        side_x = handle[0] + 0.55
        d.base_goto([side_x, d.base_pos()[1]], np.pi / 2, tol=0.04)
        d.base_goto([side_x, -0.92], np.pi / 2)
        obj = self.env.objects[name]
        self.pick(name, pitch_deg=90, heading=(0.0, 1.0), lift=0.12, back=0.10)
        d.arm_line([d.eef_pos()[0], d.eef_pos()[1], 1.10], speed=0.015, max_steps=40, tol=0.02, quiet=True)
        self.carry(fwd=0.40)
        d.base_goto([side_x, handle[1] - 0.20], np.pi / 2, tol=0.04)
        d.base_goto([handle[0], handle[1] - 0.20], np.pi / 2)
        handle, _ = self.drawer_state(drawer)
        fixture = self.env._drawer(drawer)
        p0, px, py, _ = [np.asarray(q, float) for q in fixture.get_int_sites(relative=False)["int"]]
        target = np.array([0.5 * (p0[0] + px[0]), handle[1] + 0.20, p0[2] - obj.bottom_offset[2] + 0.012])
        off = d.obj_pos(name) - d.eef_pos()
        d.arm_to([target[0] - off[0], target[1] - off[1], d.eef_pos()[2]], max_steps=80, tol=0.02, quiet=True)
        d.arm_line(target - off, speed=0.012, max_steps=80, tol=0.01, quiet=True)
        d.set_gripper(False)
        d.wait(3)
        d.arm_line(d.eef_pos() + np.array([0, -0.05, 0.30]), speed=0.015, max_steps=50, tol=0.03, quiet=True)
        d.say(f"{name} in the drawer at {np.round(d.obj_pos(name), 3)}")
        self.home()
        self.close_drawer(drawer)

    # where things go on the pulled-out dishwasher rack (x, y)
    RACK_POT = (1.93, -0.68)
    RACK_BOWL = (2.17, -0.67)
    RACK_FLOOR = 0.52

    def rack_lower(self, name, xy, min_z=None):
        """Bring a held object in under the counter edge and lower it onto the rack."""
        d = self.d
        off = d.obj_pos(name) - d.eef_pos()
        top = self.env.objects[name].size[2]
        # in front of the rack, low enough to pass under the counter lip
        z_in = self.RACK_FLOOR + top + 0.10
        d.arm_to(np.array([xy[0], -0.92, z_in + 0.10]) - off, max_steps=130, tol=0.02, quiet=True)
        d.arm_line(np.array([xy[0], -0.92, z_in]) - off, speed=0.012, max_steps=40, tol=0.02, quiet=True)
        d.arm_line(np.array([xy[0], xy[1], z_in]) - off, speed=0.01, max_steps=50, tol=0.015, quiet=True)
        last = d.obj_pos(name)[2]
        still = 0
        for _ in range(60):
            d.goal_pos = d.goal_pos + np.array([0, 0, -0.004])
            d.step()
            z = d.obj_pos(name)[2]
            still = still + 1 if z > last - 0.001 else 0
            last = z
            if still >= 6 or (min_z is not None and z <= min_z):
                break
        d.hold_here()
        d.set_gripper(False)
        d.wait(4)
        d.arm_line(d.eef_pos() - d.eef_R()[:, 2] * 0.12, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        d.say(f"{name} on the rack at {np.round(d.obj_pos(name), 3)}")

    def wash_saucepan(self):
        d = self.d
        self.home()
        grip = d.obj_R("saucepan") @ self.POT_GRIP
        p = d.obj_pos("saucepan") + grip
        self.drive(p[0], p[1] - 0.55)
        self.pick_pot()
        d.arm_line(d.eef_pos() + np.array([0, -0.10, 0.15]), speed=0.012, max_steps=40, tol=0.02, quiet=True)
        self.carry(fwd=0.40, z=1.30)
        self.drive(self.RACK_POT[0], -1.27)
        self.rack_lower("saucepan", self.RACK_POT)

    def wash_sheet(self):
        """Baking sheet out of the oven and into the saucepan standing in the rack.

        Lying on the tines it would sit below the rack region; standing in the
        pan keeps it inside the rack."""
        d = self.d
        self.slide_oven_tray(True)
        self.dish_from_tray("baking_sheet")
        self.drive(d.obj_pos("saucepan")[0], -1.27)
        pot = d.obj_pos("saucepan")
        name = "baking_sheet"
        off = d.obj_pos(name) - d.eef_pos()
        hang = float(np.linalg.norm(off)) + 0.06
        clear = pot[2] + 0.065 + hang + 0.02  # sheet centre height that clears the pan's rim
        d.arm_to(np.array([pot[0], -0.90, clear]) - off, max_steps=130, tol=0.02, quiet=True)
        d.arm_line(np.array([pot[0], pot[1] - 0.005, clear]) - off, speed=0.01, max_steps=50, tol=0.012, quiet=True)
        last = d.obj_pos(name)[2]
        still = 0
        for _ in range(70):
            d.goal_pos = d.goal_pos + np.array([0, 0, -0.004])
            d.step()
            z = d.obj_pos(name)[2]
            still = still + 1 if z > last - 0.001 else 0
            last = z
            if still >= 5 or z < pot[2] + 0.005:
                break
        d.hold_here()
        d.set_gripper(False)
        d.wait(6)
        d.arm_line(d.eef_pos() - d.eef_R()[:, 2] * 0.14, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        d.say(f"baking sheet in the saucepan at {np.round(d.obj_pos(name), 3)}")

    def wash_from_drawer(self, drawer, name, rack_xy, offset, heading, hold_tol):
        """Open the drawer, lift the tool out, close the drawer one-handed is not
        possible, so: take it to the rack first, then come back and close."""
        d = self.d
        self.open_drawer(drawer, pull=0.52)
        handle, _ = self.drawer_state(drawer)
        self.home()
        d.base_goto([handle[0], handle[1] - 0.20], np.pi / 2)
        Ro = d.obj_R(name)
        off = Ro @ np.asarray(offset, float)
        if heading == "long":
            size = np.asarray(self.env.objects[name].size)
            axis = Ro[:, int(np.argmax(size[:2]))]
            heading = (axis[0], axis[1]) if axis[1] > 0 else (-axis[0], -axis[1])
        self.pick(name, pitch_deg=90, heading=heading, offset=tuple(off), lift=0.05, back=0.10, hold_tol=hold_tol)
        d.arm_line(d.eef_pos() + np.array([0, -0.06, 0.06]), speed=0.012, max_steps=30, tol=0.02, quiet=True)
        d.arm_line([d.eef_pos()[0], d.eef_pos()[1], 1.12], speed=0.015, max_steps=70, tol=0.02, quiet=True)
        if d.obj_pos(name)[2] < 0.95:
            raise Failed(f"lost {name} while lifting it out of the drawer")
        self.carry(fwd=0.40)
        self.drive(rack_xy[0], -1.27)
        self.rack_lower(name, rack_xy)
        self.home()
        self.close_drawer(drawer)

    def stage_wash(self):
        """Saucepan, baking sheet, bowl and whisk into the dishwasher; rack in; door shut."""
        d = self.d
        homes = self.env.evaluator_state()["homes"]
        self.wash_saucepan()
        self.wash_sheet()
        self.wash_from_drawer(homes["bowl"], "bowl", self.RACK_BOWL, (-0.092, 0.0, 0.024), (0.0, 1.0), 0.16)
        bowl = d.obj_pos("bowl")
        self.wash_from_drawer(homes["whisk"], "whisk", (bowl[0], bowl[1]), (0.0, -0.07, 0.0), "long", 0.12)
        self.push_rack_in()
        self.close_dishwasher()
        self.home()

    def push_rack_in(self):
        """Fingertips against the rack's front wall, base driving forward."""
        d = self.d
        d.say("dishwasher: push the rack in")
        self.home()
        wall = d.geom_pos("dishwasher_main_group_g64")
        x = 2.045  # between the saucepan handle and the bowl
        d.base_goto([x, wall[1] - 0.64], np.pi / 2)
        wall = d.geom_pos("dishwasher_main_group_g64")
        R = self.hook_R(np.deg2rad(50))
        d.set_gripper(True, steps=6)
        tip = np.array([x, wall[1] - 0.022, wall[2] - 0.005])
        d.arm_to(tip + np.array([0, -0.05, 0.10]), R, max_steps=130, tol=0.02, quiet=True)
        d.arm_line(tip, R, speed=0.008, max_steps=40, tol=0.01, quiet=True)
        rel = d.base_R().T @ (d.eef_pos() - d.base_pos()) + np.array([0.03, 0.0, 0.0])
        for _ in range(200):
            if d.joint(self.DW_RACK) <= 0.004 or d.base_pos()[1] > -1.145:
                break
            d.goal_pos = d.base_pos() + d.base_R() @ rel
            d.step(base=(0.35, 0.0, 0.0))
        # the base cannot go under the open door: finish with the arm
        start = d.eef_pos().copy()
        r0 = d.joint(self.DW_RACK)
        for _ in range(80):
            r = d.joint(self.DW_RACK)
            if r <= 0.004:
                break
            d.goal_pos = np.array([start[0], start[1] + (r0 - r) + 0.04, start[2]])
            d.step()
        d.hold_here()
        d.say(f"dishwasher: rack {d.joint(self.DW_RACK):.3f} m out")
        d.arm_line(d.eef_pos() + np.array([0, -0.10, 0.06]), R, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        d.arm_line(d.eef_pos() + np.array([0, -0.20, 0.25]), R, speed=0.015, max_steps=40, tol=0.03, quiet=True)
        return d.joint(self.DW_RACK) < 0.02

    def close_dishwasher(self):
        """Push the door up from outside, pressing on the handle bar, until the hinge is at its stop."""
        d = self.d
        m = d.sim.model
        d.say("dishwasher: close the door")
        hinge = np.array(d.sim.data.xanchor[m.joint_name2id(self.DW_DOOR)])
        self.home()
        handle = d.geom_pos("dishwasher_main_group_door_handle_main")
        self.drive(handle[0], -1.55)
        R = self.hook_R(np.deg2rad(50))
        d.set_gripper(True, steps=6)

        def press_point(extra):
            # on the flat outer face below the handle: pressing the bar itself
            # lets the fingertips slip behind it and drag the door open again
            th = d.joint(self.DW_DOOR)
            u = np.array([0.0, -np.sin(th), np.cos(th)])  # hinge -> top edge
            n_out = np.array([0.0, -np.cos(th), -np.sin(th)])  # outward normal of the door
            # just under the handle bar, which stops the fingertips sliding up the face
            p = hinge + u * 0.655 + n_out * (0.011 + 0.03 - extra)
            p[0] = handle[0]
            return p

        d.arm_to(press_point(-0.08) + np.array([0, 0, 0.03]), R, max_steps=130, tol=0.02, quiet=True)
        d.arm_line(press_point(0.0), R, speed=0.008, max_steps=30, tol=0.01, quiet=True)
        d.lead = 0.10
        for _ in range(260):
            if d.joint(self.DW_DOOR) <= 0.0012:
                break
            target = press_point(0.09)
            d.goal_pos = target
            # follow the door with the base so the press point stays 0.58 m ahead
            gap = (press_point(0.0)[1] - d.base_pos()[1]) - 0.58
            fwd = float(np.clip(0.27 + 3.0 * gap, 0.0, 0.7)) if gap > 0.015 else 0.0
            d.step(base=(fwd, 0.0, 0.0))
        d.lead = 0.05
        d.hold_here()
        d.say(f"dishwasher: door at {d.joint(self.DW_DOOR):.4f} rad")
        d.arm_line(d.eef_pos() + np.array([0, -0.12, 0.05]), R, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        return d.joint(self.DW_DOOR) <= 0.003

    def place(self, name, pos, R=None, above=0.07, drop=0.004, retreat=None):
        """Set the held object down with its origin at `pos`, then let go."""
        d = self.d
        R = self.grasp_R if R is None else R
        tgt = np.asarray(pos, dtype=float) - self.grasp_off
        d.arm_to(tgt + np.array([0, 0, above]), R, max_steps=150, tol=0.015)
        d.arm_line(tgt + np.array([0, 0, drop]), R, speed=0.007, max_steps=40, tol=0.006)
        d.set_gripper(False)
        d.wait(3)
        back = d.eef_pos() - R[:, 2] * 0.10 if retreat is None else d.eef_pos() + np.asarray(retreat, float)
        d.arm_line(back, R, speed=0.012, max_steps=30, tol=0.02)
        d.say(f"placed {name} at {np.round(d.obj_pos(name), 3)}")


def look_quat(pos, lookat):
    import mujoco

    forward = np.asarray(lookat, dtype=float) - np.asarray(pos, dtype=float)
    forward = forward / np.linalg.norm(forward)
    z_axis = -forward
    x_axis = np.cross(np.array([0.0, 0.0, 1.0]), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.stack([x_axis, y_axis, z_axis], axis=1).flatten())
    return quat


class Recorder:
    """Renders every `every`-th control step from a free camera."""

    CAMERA = "robot0_agentview_center"

    def __init__(self, env, every=4, width=1280, height=720, sink=None):
        self.env = env
        self.every = every
        self.width = width
        self.height = height
        self.sink = sink  # callable(frame) or None -> keep in memory
        self.frames = []
        self.pos = np.array([2.75, -2.55, 2.05])
        self.look = np.array([2.7, -0.45, 1.0])
        self.fovy = 62.0
        self.caption = ""
        self.count = 0

    def aim(self, pos, look, fovy=62.0):
        self.pos, self.look, self.fovy = np.asarray(pos, float), np.asarray(look, float), fovy

    def render(self):
        m = self.env.sim.model
        cid = m.camera_name2id(self.CAMERA)
        m.cam_bodyid[cid] = 0
        m.cam_pos[cid] = self.pos
        m.cam_quat[cid] = look_quat(self.pos, self.look)
        m.cam_fovy[cid] = self.fovy
        # camera pose only; kinematics of the scene are untouched
        import mujoco

        mujoco.mj_camlight(m._model, self.env.sim.data._data)
        return np.flipud(self.env.sim.render(height=self.height, width=self.width, camera_name=self.CAMERA))

    def tick(self, driver):
        if driver.steps % self.every:
            return
        frame = self.render()
        self.count += 1
        if self.sink is not None:
            self.sink(frame, driver)
        else:
            self.frames.append((driver.steps, frame))


STAGES = (
    ("ingredients_out", "Ingredients out, doors closed", "stage_ingredients_out"),
    ("crust_baked / knobs_zeroed", "Crust: tart pan in the oven, knobs on, then off", "stage_crust"),
    ("curd_started", "Curd: saucepan on the stove", "stage_curd"),
    ("crumble_baked", "Crumble: tart pan out, baking sheet in, knobs on again", "stage_crumble"),
    ("assembled", "Assemble: lemon on the tart pan", "stage_assemble"),
    ("ingredients_restored", "Ingredients returned", "stage_restore"),
    ("tools_washed / served", "Tools in the dishwasher, tart left on the counter", "stage_wash"),
)


class VideoSink:
    """Pipes captioned frames to ffmpeg (H.264, yuv420p, faststart)."""

    def __init__(self, path, width, height, fps):
        from PIL import ImageFont

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.font = ImageFont.truetype(FONT, 26)
        self.small = ImageFont.truetype(FONT, 20)
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", "-preset", "slow",
                "-movflags", "+faststart", str(self.path),
            ],
            stdin=subprocess.PIPE,
        )
        self.title = ""
        self.status = None  # optional: driver -> second caption line
        self.frames = 0

    def __call__(self, frame, driver):
        from PIL import Image, ImageDraw

        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 70), fill=(0, 0, 0))
        draw.text((14, 4), self.title, font=self.font, fill=(255, 255, 255))
        done = len(driver.latched)
        last = driver.latched[-1] if driver.latched else "-"
        seconds = driver.steps / 20.0
        if self.status is not None:
            draw.text((14, 40), self.status(driver), font=self.small, fill=(220, 220, 220))
            self.proc.stdin.write(np.ascontiguousarray(np.asarray(image)).tobytes())
            self.frames += 1
            return
        draw.text(
            (14, 40),
            f"milestones {done}/9 (latest: {last})   step {driver.steps}   sim {int(seconds // 60)}:{int(seconds % 60):02d}"
            "   old reference: the robot's own actions",
            font=self.small,
            fill=(220, 220, 220),
        )
        self.proc.stdin.write(np.ascontiguousarray(np.asarray(image)).tobytes())
        self.frames += 1

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


class FollowCamera(Recorder):
    """A free camera that stays to one side of the robot and looks at its hand."""

    def tick(self, driver):
        if driver.steps % self.every:
            return
        base = driver.base_pos()
        hand = driver.eef_pos()
        side = -1.0 if base[0] > 3.6 else 1.0
        want_pos = np.array([np.clip(base[0] + side * 1.25, 0.9, 5.0), np.clip(base[1] - 0.85, -2.7, -1.9), 2.0])
        want_look = np.array([0.5 * (base[0] + hand[0]), -0.55, 0.5 * (hand[2] + 0.95)])
        if not hasattr(self, "_p"):
            self._p, self._l = want_pos, want_look
        self._p = 0.92 * self._p + 0.08 * want_pos
        self._l = 0.85 * self._l + 0.15 * want_look
        self.aim(self._p, self._l, 58.0)
        super().tick(driver)


def run(seed=1, horizon=None, video=None, every=4, verbose=True, actions_path=None):
    kwargs = dict(
        has_renderer=False,
        has_offscreen_renderer=video is not None,
        use_camera_obs=False,
        seed=seed,
    )
    if video is not None:
        kwargs["renderer"] = "mujoco"
    if horizon is not None:
        kwargs["horizon"] = horizon
    env = LemonTart(**kwargs)
    env.reset()
    sink = None
    recorder = None
    if video is not None:
        sink = VideoSink(video, 1280, 720, fps=20)
        recorder = FollowCamera(env, every=every, width=1280, height=720, sink=sink)
    driver = Driver(env, recorder, verbose=verbose)
    actions = []
    if actions_path is not None:
        step = env.step

        def logged(action):
            actions.append(np.array(action, dtype=np.float64))  # float32 is not enough: replay must be bit-exact
            return step(action)

        env.step = logged
    chef = Chef(driver)
    stages = []
    error = None
    started = time.time()
    try:
        driver.wait(10)
        for label, title, method in STAGES:
            if sink is not None:
                sink.title = title
            t0 = driver.steps
            driver.say(f"=== {label}")
            getattr(chef, method)()
            stages.append({"stage": label, "steps": driver.steps - t0, "ends_at": driver.steps})
        driver.wait(20)
    except Failed as exc:
        error = str(exc)
        driver.say(f"FAILED: {error}")
    state = env.evaluator_state()
    summary = {
        "seed": seed,
        "horizon": env.horizon,
        "steps": driver.steps,
        "simulated_seconds": driver.steps / 20.0,
        "wall_seconds": round(time.time() - started, 1),
        "milestones": driver.log,
        "stages": stages,
        "success": bool(state["success"]),
        "homes": state["homes"],
        "opens_during_restore": state["opens_during_restore"],
        "wrong_opens": state["wrong_opens"],
        "error": error,
        "renderer": "MuJoCo native offscreen renderer (no Cycles exporter for RoboCasa)" if video else None,
    }
    if sink is not None:
        # the environment stops stepping on success: hold the last view so the
        # final caption is on screen
        sink.title = "Done: all nine old milestones" if summary["success"] else f"Stopped: {error or 'horizon'}"
        frame = recorder.render()
        for _ in range(60):
            sink(frame, driver)
        sink.close()
        summary["video"] = str(video)
        summary["video_frames"] = sink.frames
    if actions_path is not None:
        np.savez_compressed(actions_path, actions=np.stack(actions), seed=seed)
        summary["actions"] = str(actions_path)
    env.close()
    return summary


def replay(actions_path, horizon=None):
    """Feed a saved action log to a fresh environment and report what it scores.

    Nothing but `env.step(action)` runs here, so this checks that the result
    comes from the actions alone."""
    data = np.load(actions_path)
    seed = int(data["seed"])
    kwargs = dict(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, seed=seed)
    if horizon is not None:
        kwargs["horizon"] = horizon
    env = LemonTart(**kwargs)
    env.reset()
    reached = {}
    info = {}
    steps = 0
    for action in data["actions"]:
        _, _, done, info = env.step(action)
        steps += 1
        for name, value in info["lemon_tart"]["milestones"].items():
            if value and name not in reached:
                reached[name] = steps
        if done:
            break
    result = {
        "seed": seed,
        "steps": steps,
        "milestones": reached,
        "success": bool(info.get("lemon_tart", {}).get("success")),
    }
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=None, help="replay a saved action log instead of running the controller")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=None, help="override the episode limit (default: the env's 20000)")
    parser.add_argument("--out", type=Path, default=REPO / "videos" / "lemon-tart" / "lemon-tart-reference.mp4")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--every", type=int, default=4, help="render every N-th control step")
    parser.add_argument("--summary", type=Path, default=REPO / "videos" / "lemon-tart" / "lemon-tart-reference.json")
    parser.add_argument("--actions", type=Path, default=None, help="save the action log (.npz)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.replay is not None:
        result = replay(args.replay, args.horizon)
        print(json.dumps(result, indent=2))
        return 0 if result["success"] else 1
    summary = run(
        seed=args.seed,
        horizon=args.horizon,
        video=None if args.no_video else args.out,
        every=args.every,
        verbose=not args.quiet,
        actions_path=args.actions,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("steps", "success", "error", "stages")}, indent=2, ensure_ascii=False))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
