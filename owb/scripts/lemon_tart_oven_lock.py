"""Lemon tart: opening the toaster oven when its door has a lock.

Runs in `LemonTart(usable_ingredients=True, oven_lock="recipe")`. The robot

1. pulls the handle: the door is locked and does not move;
2. presses the button on the front panel: the lamp turns green;
3. opens the door;
4. closes it again: the door locks itself, the lamp turns red;
5. pulls the handle once more to show that it is locked;
6. presses the button and opens the door a second time.

Only the robot's own 12-dim actions are used.

    python -m owb.scripts.lemon_tart_oven_lock --seed 1

`--demo knobs` shows the other oven option, `oven_knob_order`: the three knobs
turn up only in an order drawn at reset, and back down only in the reverse
order. The robot first tries the knob that comes second, which does not move,
then turns all three up in order; then it tries to turn the first one back,
which does not move either, and turns all three down in reverse order.

    python -m owb.scripts.lemon_tart_oven_lock --seed 1 --demo knobs
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from owb.scripts.lemon_tart_reference import REPO, Chef, Driver, Failed, Recorder, VideoSink, grip_R
from owb.tasks import lemon_tart_lock as lock
from owb.tasks.lemon_tart_env import LemonTart


class LockChef(Chef):
    HANDLE = "toaster_oven_main_group_door_handle_main"

    def lock_state(self):
        return self.env.lock_state()

    def pull_handle(self, give_up=45):
        """Hook the handle from above and pull the door down. Returns the door angle reached."""
        d = self.d
        hinge, top = self.oven_door_geom()
        self.home()
        self.drive(self.OVEN_X, -1.13)
        handle = d.geom_pos(self.HANDLE)
        R = self.hook_R(np.deg2rad(60))
        approach = R[:, 2]
        d.set_gripper(True, steps=6)
        tip = handle + np.array([0.0, 0.021, -0.006])
        d.arm_to(tip - approach * 0.09, R, max_steps=130, tol=0.015, quiet=True)
        d.arm_line(tip, R, speed=0.008, max_steps=40, tol=0.008)
        rel = tip - hinge
        self._radius = float(np.hypot(rel[1], rel[2]))
        self._phi0 = float(np.arctan2(-rel[1], rel[2])) - d.joint(self.OVEN_DOOR)
        self._tip_x = float(tip[0])
        best, since = d.joint(self.OVEN_DOOR), 0
        for _ in range(200):
            cur = d.joint(self.OVEN_DOOR)
            if cur >= 1.0:
                break
            if cur > best + 0.01:
                best, since = cur, 0
            else:
                since += 1
                if since > give_up:  # it is not coming: stop pulling
                    break
            th = cur + 0.25
            d.goal_pos = np.array(
                [tip[0], hinge[1] - self._radius * np.sin(th + self._phi0) - 0.01, hinge[2] + self._radius * np.cos(th + self._phi0)]
            )
            d.step()
        d.hold_here()
        angle = d.joint(self.OVEN_DOOR)
        d.say(f"oven: pulled the handle, door at {angle:.2f} rad, lock {self.lock_state()}")
        d.arm_line(d.eef_pos() + np.array([0, 0.0, 0.10]), R, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        return angle

    def press_flat(self):
        """After a pull: press the half-open door down from above until it lies flat."""
        d = self.d
        hinge, top = self.oven_door_geom()
        R = self.hook_R(np.deg2rad(60))
        for _ in range(3):
            cur = d.joint(self.OVEN_DOOR)
            if cur >= top - 0.03:
                break
            ang = cur + self._phi0
            p = np.array([self._tip_x, hinge[1] - 0.13 * np.sin(ang), hinge[2] + 0.13 * np.cos(ang)])
            d.arm_to(p + np.array([0, 0.03, 0.07]), R, max_steps=60, tol=0.02, quiet=True)
            for _ in range(60):
                cur = d.joint(self.OVEN_DOOR)
                if cur >= top - 0.02:
                    break
                ang = min(cur + 0.3, top + 0.1) + self._phi0
                d.goal_pos = np.array([self._tip_x, hinge[1] - 0.13 * np.sin(ang) + 0.01, hinge[2] + 0.13 * np.cos(ang) + 0.012])
                d.step()
            d.hold_here()
            d.arm_line(d.eef_pos() + np.array([0, -0.02, 0.12]), R, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        d.say(f"oven: door at {d.joint(self.OVEN_DOOR):.2f} of {top:.2f} rad")
        return d.joint(self.OVEN_DOOR) >= top - 0.1

    def button_pos(self):
        m = self.d.sim.model
        return np.array(self.d.sim.data.body_xpos[m.body_name2id("oven_lock_button")])

    def press_button(self):
        """Push the button in with the closed fingertips."""
        d = self.d
        self.home()
        button = self.button_pos()
        self.drive(button[0], -1.08)
        pitch = np.deg2rad(20)
        approach = np.array([0.0, np.cos(pitch), -np.sin(pitch)])
        R = grip_R(approach, [-1.0, 0.0, 0.0])
        d.set_gripper(True, steps=6)
        face = button + np.array([0.0, -lock.BUTTON_HALF[1], 0.0])
        d.arm_to(face - approach * 0.08, R, max_steps=140, tol=0.012)
        before = self.lock_state()["unlock_presses"]
        for _ in range(160):
            if self.env.oven_lock.button_travel() > lock.PRESSED + 0.002 or d.eef_pos()[1] > face[1] + 0.02:
                break
            # straight in, staying on the button's centre line
            d.goal_pos = np.array([button[0], d.eef_pos()[1] + 0.008, button[2]])
            d.step()
        d.hold_here()
        d.wait(4)
        d.arm_line(d.eef_pos() - approach * 0.10, R, speed=0.012, max_steps=40, tol=0.02, quiet=True)
        state = self.lock_state()
        d.say(f"oven: pressed the button, lock {state}")
        if state["unlock_presses"] == before:
            raise Failed("the button press did not unlock the door")

    def close_door(self):
        """Lift the open door from under its outer edge, then push it shut from the front."""
        d = self.d
        hinge, top = self.oven_door_geom()
        self.home()
        self.drive(self.OVEN_X, -1.13)
        R = self.hook_R(np.deg2rad(60))
        d.set_gripper(True, steps=6)
        x = self.OVEN_X + 0.17  # right of the handle bar, under the door's corner, away from the button
        length = 0.205

        def edge(theta, r=length):
            return np.array([x, hinge[1] - r * np.sin(theta), hinge[2] + r * np.cos(theta)])

        def outward(theta):  # normal of the door's outer face
            return np.array([0.0, -np.cos(theta), -np.sin(theta)])

        cur = d.joint(self.OVEN_DOOR)
        start = edge(cur) + outward(cur) * 0.03 + np.array([0, -0.05, 0.0])
        d.arm_to(start + np.array([0, -0.04, 0.10]), R, max_steps=140, tol=0.02, quiet=True)
        d.arm_line(start, R, speed=0.01, max_steps=60, tol=0.01, quiet=True)
        for _ in range(260):
            cur = d.joint(self.OVEN_DOOR)
            if cur < 0.03:
                break
            lead = max(cur - 0.30, -0.12)
            d.goal_pos = edge(lead, r=0.19) + outward(lead) * 0.006
            d.step()
        # the last few degrees: back off, then press the middle of the handle bar home
        d.hold_here()
        bar = d.geom_pos(self.HANDLE)
        d.arm_line(d.eef_pos() + np.array([0, -0.06, 0.02]), R, speed=0.012, max_steps=30, tol=0.02, quiet=True)
        self.home()  # the lift leaves the arm stretched out; start the push from a fresh pose
        d.set_gripper(True, steps=4)
        bar = d.geom_pos(self.HANDLE)
        front = np.array([bar[0], bar[1] - 0.07, bar[2] + 0.005])
        d.arm_to(front, R, max_steps=160, tol=0.012)
        for _ in range(120):
            if d.joint(self.OVEN_DOOR) < 0.004:
                break
            d.goal_pos = np.array([bar[0], d.eef_pos()[1] + 0.008, front[2]])
            d.step()
        d.hold_here()
        d.wait(6)
        angle = d.joint(self.OVEN_DOOR)
        d.arm_line(d.eef_pos() + np.array([0, -0.10, 0.04]), R, speed=0.012, max_steps=40, tol=0.02, quiet=True)
        d.say(f"oven: door pushed to {angle:.3f} rad, lock {self.lock_state()}")
        if d.joint(self.OVEN_DOOR) > lock.SHUT:
            raise Failed(f"the door did not close ({d.joint(self.OVEN_DOOR):.2f} rad)")

    # ---------- the demo ----------
    def stage_locked(self):
        if self.pull_handle() > 0.1:
            raise Failed("the door opened although it should be locked")

    def stage_unlock_open(self):
        self.press_button()
        if self.pull_handle() < 1.0 or not self.press_flat():
            raise Failed("the door did not open after unlocking")

    def stage_close(self):
        self.close_door()
        if not self.lock_state()["locked"]:
            raise Failed("the door closed but did not lock")

    def stage_done(self):
        self.home()

    def knob_value(self, knob):
        # the environment narrows a held knob's range; measure on the full one
        if self.env.knob_order is not None:
            return self.env.knob_order.value(knob)
        return super().knob_value(knob)

    # ---------- knob order demo ----------
    def knob_state(self):
        return self.env.knob_state()

    def stage_wrong_knob(self):
        """Try the knob that comes second in the order: it must not turn."""
        d = self.d
        knob = self.knob_state()["order"][1]
        self.home()
        self.drive(*self.KNOB_STANCE)
        self.turn_knob(knob, self.KNOB_ON, max_grasps=1)
        state = self.knob_state()
        d.say(f"knobs: tried {knob} out of turn, {state}")
        if state["values"][knob] >= self.env.knob_order.on_level:
            raise Failed(f"the {knob} knob turned although it is not its turn")

    def stage_right_order(self):
        d = self.d
        order = self.knob_state()["order"]
        self.home()
        self.drive(*self.KNOB_STANCE)
        for i, knob in enumerate(order):
            # the time knob runs down by itself: give it a head start unless it is last
            target = self.KNOB_ON + (0.25 if knob == "time" and i < 2 else 0.0)
            if not self.turn_knob(knob, target):
                raise Failed(f"knob {knob} did not turn in its turn: {self.knob_state()}")
        d.say(f"knobs: all three up, {self.knob_state()}")

    def stage_wrong_back(self):
        """Try to turn the first knob back while the others are still up: it must not move."""
        d = self.d
        knob = self.knob_state()["order"][0]
        self.turn_knob(knob, self.KNOB_OFF, max_grasps=1)
        state = self.knob_state()
        d.say(f"knobs: tried to turn {knob} back out of turn, {state}")
        if state["values"][knob] < self.env.knob_order.on_level:
            raise Failed(f"the {knob} knob turned back although the later knobs are still on")

    def stage_reverse_order(self):
        d = self.d
        for knob in reversed(self.knob_state()["order"]):
            if not self.turn_knob(knob, self.KNOB_OFF):
                raise Failed(f"knob {knob} did not turn back in its turn: {self.knob_state()}")
        d.say(f"knobs: all three back down, {self.knob_state()}")


STAGES = (
    ("locked_pull", "Pull the handle: the door is locked and stays shut", "stage_locked"),
    ("unlock_open", "Press the button left of the knobs, then open the door", "stage_unlock_open"),
    ("close_relock", "Close the door: it locks again", "stage_close"),
    ("locked_pull_again", "Pull again: it is locked again", "stage_locked"),
    ("unlock_open_again", "Press the button again, then open the door", "stage_unlock_open"),
    ("done", "Done", "stage_done"),
)


KNOB_STAGES = (
    ("wrong_knob", "Turn the second knob first: it does not move", "stage_wrong_knob"),
    ("right_order", "Turn them in this episode's order: all three come on", "stage_right_order"),
    ("wrong_back", "Try to turn the first one back first: it does not move", "stage_wrong_back"),
    ("reverse_order", "Turn them back in reverse order: all three off", "stage_reverse_order"),
    ("done", "Done", "stage_done"),
)
KNOB_WORDS = {"function": "function", "temperature": "temperature", "time": "time"}


def run(seed=1, video=None, every=2, verbose=True, actions_path=None, mode="recipe", after_stage=None, demo="lock"):
    env = LemonTart(
        has_renderer=False, has_offscreen_renderer=video is not None, use_camera_obs=False, seed=seed,
        usable_ingredients=True, oven_lock=mode, oven_knob_order=mode if demo == "knobs" else None,
        **({"renderer": "mujoco"} if video is not None else {}),
    )
    env.reset()
    sink = recorder = None
    if video is not None:
        sink = VideoSink(video, 1280, 720, fps=20)
        recorder = Recorder(env, every=every, width=1280, height=720, sink=sink)
        recorder.aim([3.25, -1.30, 1.80], [2.50, -0.50, 1.10], 48.0)

        def status(drv):
            s = env.lock_state()
            if demo == "knobs":
                k = env.knob_state()
                def mark(n):
                    if not k["free"][n] and not k["free_down"][n]:
                        return " (stuck)"
                    return " (not up)" if not k["free"][n] else " (not down)" if not k["free_down"][n] else ""

                parts = "  ".join(f"{KNOB_WORDS[n]} {k['values'][n]:.2f}{mark(n)}" for n in k["order"])
                return (
                    f"order: {' -> '.join(KNOB_WORDS[n] for n in k['order'])}   {parts}"
                    f"   blocked turns {sum(k['blocked_turns'].values())}   step {drv.steps}"
                )
            return (
                f"lock: {'locked' if s['locked'] else 'unlocked'}   door {np.degrees(s['door_angle']):.0f} deg"
                f"   unlock presses {s['unlock_presses']}   pulls while locked {s['locked_pulls']}   relocks {s['relocks']}"
                f"   step {drv.steps}"
            )

        sink.status = status
    driver = Driver(env, recorder, verbose=verbose)
    chef = LockChef(driver)
    actions = []
    if actions_path is not None:
        step = env.step

        def logged(action):
            actions.append(np.array(action, dtype=np.float64))
            return step(action)

        env.step = logged
    stages, error, started = [], None, time.time()
    try:
        driver.wait(10)
        for label, title, method in KNOB_STAGES if demo == "knobs" else STAGES:
            if sink is not None:
                sink.title = title
            t0 = driver.steps
            driver.say(f"=== {label}")
            getattr(chef, method)()
            stages.append({"stage": label, "steps": driver.steps - t0, "ends_at": driver.steps, "lock": env.lock_state()})
            if after_stage is not None:
                after_stage(label, driver, chef)
        driver.wait(20)
    except Failed as exc:
        error = str(exc)
        driver.say(f"FAILED: {error}")
    state = env.lock_state()
    success = error is None and state["unlock_presses"] == 2 and state["relocks"] == 1 and state["door_angle"] > 1.2
    knobs = env.knob_state()
    if demo == "knobs":
        level = env.knob_order.on_level
        success = (
            error is None
            and all(v <= env.knob_order.off_level for v in knobs["values"].values())
            and knobs["blocked_turns"][knobs["order"][1]] >= 1
            and knobs["blocked_turns"][knobs["order"][0]] >= 1
        )
    summary = {
        "seed": seed,
        "demo": demo,
        "knobs": knobs,
        "oven_lock": mode,
        "steps": driver.steps,
        "simulated_seconds": driver.steps / 20.0,
        "wall_seconds": round(time.time() - started, 1),
        "stages": stages,
        "lock": state,
        "success": bool(success),
        "error": error,
        "renderer": "MuJoCo native offscreen renderer (no Cycles exporter for RoboCasa)" if video else None,
    }
    if sink is not None:
        done = (
            "Done: knobs on in order, then back off in reverse"
            if demo == "knobs"
            else "Done: both openings were unlocked first"
        )
        sink.title = done if success else f"Stopped: {error or 'checks failed'}"
        frame = recorder.render()
        for _ in range(60):
            sink(frame, driver)
        sink.close()
        summary["video"] = str(video)
    if actions_path is not None:
        np.savez_compressed(actions_path, actions=np.stack(actions), seed=seed, mode=mode)
        summary["actions"] = str(actions_path)
    env.close()
    return summary


def replay(actions_path):
    """Feed a saved action log to a fresh environment; only `env.step` runs."""
    data = np.load(actions_path)
    env = LemonTart(
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, seed=int(data["seed"]),
        usable_ingredients=True, oven_lock=str(data["mode"]),
    )
    env.reset()
    for action in data["actions"]:
        env.step(action)
    state = env.lock_state()
    env.close()
    return {"steps": int(len(data["actions"])), "lock": state}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", choices=lock.MODES, default="recipe")
    parser.add_argument("--demo", choices=("lock", "knobs"), default="lock")
    parser.add_argument("--out", type=Path, default=None, help="default: videos/lemon-tart-oven-<demo>.mp4")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--summary", type=Path, default=None, help="default: videos/lemon-tart-oven-<demo>.json")
    parser.add_argument("--actions", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.replay is not None:
        print(json.dumps(replay(args.replay), indent=2))
        return 0
    args.out = args.out or REPO / "videos" / f"lemon-tart-oven-{args.demo}.mp4"
    args.summary = args.summary or REPO / "videos" / f"lemon-tart-oven-{args.demo}.json"
    summary = run(
        seed=args.seed, video=None if args.no_video else args.out, every=args.every, verbose=not args.quiet,
        actions_path=args.actions, mode=args.mode, demo=args.demo,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("steps", "success", "error", "lock", "knobs")}, indent=2, ensure_ascii=False))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
