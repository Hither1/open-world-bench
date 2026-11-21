"""Render the lemon-tart milestone states.

This is a scored-state walkthrough: object and door poses are set directly so
each chapter shows what the evaluator looks for. It is not a reference
trajectory. A reference trajectory has to reach those states through the
robot's own actions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from owb.tasks.lemon_tart import DISHWASHER, FRIDGE, INGREDIENTS, WASH_TOOLS
from owb.tasks.lemon_tart_env import LemonTart

import robocasa.utils.env_utils as EnvUtils

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def _box(fixture):
    sites = fixture.get_int_sites(relative=False)
    if not sites:
        return None
    key = "int" if "int" in sites else ("rack0" if "rack0" in sites else next(iter(sites)))
    points = np.stack([np.asarray(point, dtype=float) for point in sites[key][:4]])
    return points.mean(axis=0), float(points[:, 2].min()), points


def _fixture(env, name):
    if name == FRIDGE:
        return env.fridge
    if name == DISHWASHER:
        return env.dishwasher
    if name == "toaster_oven":
        return env.toaster_oven
    if name == "stove":
        return env.stove
    return env._drawer(name)


def _put(env, name, pos):
    obj = env.objects[name]
    qpos = np.array(env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float)
    origin = np.array(pos, dtype=float)
    origin[2] -= float(obj.bottom_offset[2])
    env.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([origin, qpos[3:7]]))


def _put_in(env, name, fixture, slot, count):
    found = _box(fixture)
    if found is None:
        center = np.array(fixture.pos, dtype=float)
        center[2] += float(fixture.size[2]) / 2
        _put(env, name, center)
        return
    center, bottom, points = found
    axis = points[1] - points[0]
    norm = np.linalg.norm(axis)
    axis = axis / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    pos = center + axis * ((slot - (count - 1) / 2.0) * 0.08)
    pos = np.array(pos, dtype=float)
    pos[2] = bottom + 0.01
    _put(env, name, pos)


def _open(env, fixture, open_door):
    try:
        if open_door:
            fixture.open_door(env, min=0.95, max=1.0, partial_open=False)
        else:
            fixture.close_door(env)
    except TypeError:
        if open_door:
            fixture.open_door(env, min=0.95, max=1.0)
        else:
            fixture.close_door(env)


def _doors(env, open_door):
    for fixture in (env.fridge, env.dishwasher, env.toaster_oven, *env.drawers):
        _open(env, fixture, open_door)


def _knobs(env, value):
    setters = {
        "function": env.toaster_oven.set_function,
        "temperature": env.toaster_oven.set_temperature,
        "time": env.toaster_oven.set_time,
    }
    present = set(env.sim.model.joint_names)
    for name, setter in setters.items():
        joint = env.toaster_oven._joint_names.get(name)
        if joint in present:
            setter(env, value)


def _home_place(env, names):
    groups = {}
    for name in names:
        groups.setdefault(env.homes[name], []).append(name)
    for home, members in groups.items():
        fixture = _fixture(env, home)
        for slot, name in enumerate(members):
            _put_in(env, name, fixture, slot, len(members))


def _surface(fixture):
    pos = np.array(fixture.pos, dtype=float)
    pos[2] = float(fixture.pos[2]) + float(fixture.size[2]) / 2.0
    pos[1] = float(fixture.pos[1]) - 0.08
    return pos


def _spread(env, names, origin, gap=0.12):
    for slot, name in enumerate(names):
        pos = np.array(origin, dtype=float)
        pos[0] += (slot - (len(names) - 1) / 2.0) * gap
        _put(env, name, pos)


def _counter_place(env, names):
    _spread(env, names, _surface(env.counter))


def _look_quat(pos, lookat):
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


def _park(env):
    EnvUtils.set_robot_to_position(env, np.array([2.4, -1.9, 0.0]))
    yaw = env.sim.model.get_joint_qpos_addr("mobilebase0_joint_mobile_yaw")
    env.sim.data.qpos[yaw] = np.pi / 2
    env.sim.forward()


def _render(env, width, height):
    camera_id = env.sim.model.camera_name2id("robot0_agentview_center")
    position = np.array([2.75, -2.55, 2.05])
    env.sim.model.cam_bodyid[camera_id] = 0
    env.sim.model.cam_pos[camera_id] = position
    env.sim.model.cam_quat[camera_id] = _look_quat(position, np.array([2.7, -0.45, 1.0]))
    env.sim.model.cam_fovy[camera_id] = 62
    env.sim.forward()
    return np.flipud(env.sim.render(height=height, width=width, camera_name="robot0_agentview_center"))


def _caption(frame, title, note):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 32)
    small = ImageFont.truetype(FONT, 22)
    draw.rectangle((0, 0, image.width, 86), fill=(0, 0, 0))
    draw.text((16, 8), title, font=font, fill=(255, 255, 255))
    draw.text((16, 48), note, font=small, fill=(220, 220, 220))
    return np.array(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/yirantao/long-horizon/open-world-bench/videos/lemon-tart/lemon-tart-state-walkthrough.mp4"),
    )
    args = parser.parse_args()
    env = LemonTart(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        renderer="mujoco",
        seed=args.seed,
    )
    env.reset()
    tools = [name for name in ("bowl", "whisk", "saucepan", "baking_sheet", "tart_pan")]

    def show_start():
        _doors(env, False)
        env.sim.forward()
        _home_place(env, list(env.homes))
        _knobs(env, 0.0)

    def show_ingredients_out():
        _doors(env, False)
        env.sim.forward()
        _counter_place(env, list(INGREDIENTS))
        _home_place(env, tools)
        _knobs(env, 0.0)

    def show_crust():
        show_ingredients_out()
        _open(env, env.toaster_oven, True)
        env.sim.forward()
        _put_in(env, "tart_pan", env.toaster_oven, 0, 1)
        _knobs(env, 0.6)

    def show_zero():
        show_crust()
        _knobs(env, 0.0)

    def show_curd():
        show_zero()
        _spread(env, ["saucepan"], _surface(env.stove))

    def show_crumble():
        show_curd()
        _put_in(env, "baking_sheet", env.toaster_oven, 0, 1)
        _spread(env, ["tart_pan"], _surface(env.counter))
        _knobs(env, 0.6)

    def show_assembled():
        show_crumble()
        _knobs(env, 0.0)
        _open(env, env.toaster_oven, False)
        env.sim.forward()
        _spread(env, ["tart_pan"], _surface(env.counter))
        pan = env.sim.data.get_joint_qpos(env.objects["tart_pan"].joints[0])
        _put(env, "lemon", np.array(pan[:3]) + np.array([0.0, 0.0, 0.03]))

    def show_restored():
        show_assembled()
        _doors(env, False)
        env.sim.forward()
        _home_place(env, list(INGREDIENTS))
        _spread(env, ["tart_pan"], _surface(env.counter))

    def show_done():
        show_restored()
        _open(env, env.dishwasher, True)
        env.sim.forward()
        for slot, name in enumerate(WASH_TOOLS):
            _put_in(env, name, env.dishwasher, slot, len(WASH_TOOLS))
        env.sim.forward()
        _spread(env, ["tart_pan"], _surface(env.counter))

    chapters = [
        ("start", "Every door is closed. Objects are in the fridge, drawers, and dishwasher.", show_start),
        ("ingredients out", "The five ingredients are on the counter. The tools are still where they started.", show_ingredients_out),
        ("first bake", "The tart pan is in the toaster oven and the three knobs are on.", show_crust),
        ("knobs off", "The three knobs are back off, so the second bake cannot copy them.", show_zero),
        ("saucepan on the stove", "This stretch is the gap between the two bakes.", show_curd),
        ("second bake", "The baking sheet is in the oven and the three knobs are on again.", show_crumble),
        ("assembled", "The lemon is on the tart pan on the counter, then it goes back to the fridge.", show_assembled),
        ("ingredients home", "The five ingredients are back in the fridge or their drawers.", show_restored),
        ("cleanup", "Bowl, whisk, saucepan, and baking sheet are in the dishwasher. The tart stays on the counter.", show_done),
    ]

    frames = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stills = args.out.parent / "lemon-tart-walkthrough"
    stills.mkdir(parents=True, exist_ok=True)
    for index, (title, note, arrange) in enumerate(chapters, start=1):
        arrange()
        env.sim.forward()
        _park(env)
        frame = _caption(_render(env, 1280, 720), f"{index}/9  {title}", note + "  State walkthrough, not robot actions.")
        frames.extend([frame] * 10)
        Image.fromarray(frame).save(stills / f"{index:02d}-{title}.png")
        print(title, flush=True)
    env.close()

    tmp = stills / "frames"
    tmp.mkdir(exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(tmp / f"{index:04d}.png")
    import subprocess
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-framerate", "8", "-i", str(tmp / "%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-crf", "18", str(args.out),
        ]
    )
    print(args.out)


if __name__ == "__main__":
    main()
