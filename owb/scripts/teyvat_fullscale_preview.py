"""Full-scale Teyvat Task A route preview: collect, verify, render.

The route preview is labelled as such everywhere: a privileged reference agent
visits each milestone's places in order with native movement (walking,
swimming, sprinting, gliding, riding lifts). Relay, combat, and item mechanics
arrive in phase 2, so milestones here are checkpoint visits, and each gate fact
opens when the agent reaches the place where the task logic will open it.

    # run the agent, then replay its actions in a fresh env and compare
    python -B -m owb.scripts.teyvat_fullscale_preview collect --seed 7 --output runs/fullscale-preview/seed-7
    python -B -m owb.scripts.teyvat_fullscale_preview verify runs/fullscale-preview/seed-7
    # replay again, render sampled states with Cycles on GPUs, and assemble the video
    python -B -m owb.scripts.teyvat_fullscale_preview render runs/fullscale-preview/seed-7 \
        --cache .cache/teyvat-fullscale-preview/seed-7 --gpus 0 1

Rendering resumes from the cache: each frame records its replay-state digest,
the render settings, and its image hash, and a frame is reused only if all match.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Simulation sources a recorded trace depends on; replay refuses if any changed.
SOURCES = ("open_world_bench/fullscale_env.py", "open_world_bench/fullscale_policy.py",
           "open_world_bench/fullscale_nav.py", "open_world_bench/fullscale_world.py",
           "open_world_bench/fullscale_physics.py", "open_world_bench/fullscale_reach.py")
# Presentation sources, recorded with each video.
PRESENTATION_SOURCES = ("owb/scripts/teyvat_fullscale_preview.py", "owb/render/scene_export.py",
                        "owb/render/blender_scene.py")
SCENE_SIZE = (1280, 720)
CANVAS = (1600, 900)
FPS = 10
STRIDE = dict(walk=8, lift=5, glide=2)          # 16x, 10x, 4x at 10 fps and 0.2 s per action
HOLDS = dict(checkpoint=10, milestone_complete=15, fact_opened=15, lift_board=6, lift_arrived=6,
             move_glide_start=8, move_land=6)
LABEL = "ROUTE PREVIEW"
DISCLAIMER = ("A privileged reference agent visits each milestone's places with native movement. Relay, combat, "
              "and item mechanics are not implemented yet; each gate opens when the agent reaches the place "
              "where the task logic will open it.")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in (sorted(value) if isinstance(value, set) else value)]
    return value


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_rows(directory):
    with (Path(directory) / "actions.jsonl").open() as stream:
        return [json.loads(line) for line in stream]


def native_action(row_action):
    return {k: (np.asarray(v, np.int8 if k == "buttons" else np.float32) if k in ("move", "look", "buttons", "pointer")
                else v) for k, v in row_action.items()}


def make_env(seed):
    from open_world_bench.fullscale_env import TeyvatFullScaleEnv
    env = TeyvatFullScaleEnv(seed, render_enabled=False)
    env.reset(seed=seed)
    return env


def snapshot(env, explanation=""):
    world, char = env.world, env.char
    milestone, checkpoint = world.current_checkpoint()
    riding = world.riding()
    return dict(milestone=milestone.id if milestone else None, title=milestone.title if milestone else None,
                checkpoint=checkpoint.name if checkpoint else None,
                checkpoint_label=checkpoint.label if checkpoint else None,
                explanation=explanation, time=round(world.elapsed, 6), position=char.pos.tolist(),
                move_state=char.state.name, stamina=float(char.stamina),
                stamina_max=float(env.controller.cfg.stamina_max), hp=float(char.hp),
                max_hp=float(env.controller.cfg.max_hp), facts=sorted(env.fullscale.state.facts),
                riding=riding.id if riding else None)


# ---------------------------------------------------------------- collect
def collect(seed, directory, max_steps=20_000):
    from open_world_bench.fullscale_env import PREVIEW_VERSION, ROUTE
    from open_world_bench.fullscale_physics import PHYSICS_VERSION
    from open_world_bench.fullscale_policy import RoutePreviewPolicy
    from open_world_bench.fullscale_world import SUITE_VERSION

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    env = make_env(seed)
    policy = RoutePreviewPolicy(env)
    initial = env.state_digest()
    started = time.time()
    travel, terminated = 0.0, False
    with (directory / "actions.jsonl.partial").open("w") as stream:
        for step in range(max_steps):
            before = env.char.pos.copy()
            action = policy.act()
            _, reward, terminated, truncated, info = env.step(action)
            travel += float(np.linalg.norm((env.char.pos - before)[[0, 2]]))
            row = dict(step=step, action=jsonable(action), position_before=before.tolist(),
                       position_after=env.char.pos.tolist(), events=jsonable(info["event_details"]),
                       reward=float(reward), terminated=bool(terminated), truncated=bool(truncated),
                       digest=env.state_digest(), **snapshot(env, policy.explanation))
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            if terminated or truncated:
                break
    (directory / "actions.jsonl.partial").replace(directory / "actions.jsonl")
    preview = env.world.preview
    success = len(preview.completed) == len(ROUTE) and env.char.hp > 0
    manifest = dict(
        kind="route-preview", preview_version=PREVIEW_VERSION, suite=SUITE_VERSION, physics=PHYSICS_VERSION,
        seed=seed, world_digest=env.world_digest, initial_digest=initial, final_digest=env.state_digest(),
        success=bool(success), terminated=bool(terminated), actions=step + 1,
        simulated_seconds=round(env.world.elapsed, 3), travel_metres=round(travel, 1),
        min_hp=min(r["hp"] for r in read_rows(directory)), final_hp=float(env.char.hp),
        milestones=[dict(id=m.id, title=m.title, checkpoints=[c.name for c in m.checkpoints],
                         completed=next((c["time"] for c in preview.completed if c["id"] == m.id), None))
                    for m in ROUTE],
        checkpoints=preview.visited, replans=policy._replans, decision_seconds=0.2,
        controller="RoutePreviewPolicy: privileged planner on the phase-1 under-model graph; native actions only",
        limitations=[DISCLAIMER, "Seeds 7 and 19 were run; other seeds are untested.",
                     "Lifts carry the character along an authored cabin path (no rigid-body cabin)."],
        sources={p: sha256(ROOT / p) for p in SOURCES}, collected_wall_seconds=round(time.time() - started, 1))
    save_json(directory / "manifest.json", manifest)
    env.close()
    print(f"seed {seed}: {'success' if success else 'FAILED'}: {manifest['actions']} actions, "
          f"{manifest['simulated_seconds'] / 60:.1f} simulated minutes, {travel / 1000:.2f} km", flush=True)
    return manifest


def verify(directory):
    """Replay every action in a fresh env and compare positions, events, and digests."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = read_rows(directory)
    for path, digest in manifest["sources"].items():
        if sha256(ROOT / path) != digest:
            raise SystemExit(f"Replay source changed: {path}")
    env = make_env(manifest["seed"])
    assert env.world_digest == manifest["world_digest"], "world digest differs"
    assert env.state_digest() == manifest["initial_digest"], "initial state differs"
    for row in rows:
        assert env.char.pos.tolist() == row["position_before"], f"position before step {row['step']}"
        _, reward, terminated, truncated, info = env.step(native_action(row["action"]))
        assert env.char.pos.tolist() == row["position_after"], f"position after step {row['step']}"
        assert jsonable(info["event_details"]) == row["events"], f"events at step {row['step']}"
        assert reward == row["reward"] and terminated == row["terminated"] and truncated == row["truncated"]
        assert env.state_digest() == row["digest"], f"state digest at step {row['step']}"
    assert env.state_digest() == manifest["final_digest"], "final state differs"
    result = dict(ok=True, actions=len(rows), success=manifest["success"],
                  checked=["positions before and after every action", "event streams", "rewards and flags",
                           "state digest after every action", "initial and final digests", "world digest",
                           "source hashes"],
                  actions_sha256=sha256(directory / "actions.jsonl"), sources=manifest["sources"])
    save_json(directory / "verification.json", result)
    env.close()
    print(f"verified {len(rows)} actions", flush=True)
    return result


# ------------------------------------------------------------------ plan
def mode_of(row):
    if row["riding"]:
        return "lift"
    if row["move_state"] in ("GLIDING", "AIRBORNE") or row["explanation"].startswith(("Glide", "Walk off", "Rest at")):
        return "glide"
    return "walk"


def frame_plan(rows):
    """Steps to render and how many video frames each holds (sampled real states)."""
    plan = []
    for i, row in enumerate(rows):
        kinds = {e["type"] for e in row["events"]}
        mode = mode_of(row)
        if mode != "glide":
            kinds.discard("move_land")          # small hops while walking are not events
        hold = max([HOLDS[k] for k in kinds if k in HOLDS], default=0)
        if i == 0 or i == len(rows) - 1 or hold or row["step"] % STRIDE[mode] == 0:
            plan.append(dict(step=row["step"], repeats=max(1, hold), mode=mode,
                             speed=round(0.2 * STRIDE[mode] * FPS), events=sorted(kinds & set(HOLDS))))
    plan[-1]["repeats"] = 30
    return plan


# ----------------------------------------------------------------- scene
def fullscale_scene(env, width, height):
    """Cycles snapshot of the actual state plus the gate and lift structures."""
    from open_world_bench.render.assets import BARK, CLOTH, METAL
    from open_world_bench.render.meshes import MeshBatch
    from owb.render.scene_export import _add_batch, capture_scene

    scene = capture_scene(env, width=width, height=height)
    batch = MeshBatch()
    for i, gate in enumerate(env.fullscale.architecture.active()):
        centre = np.asarray(gate["position"], float) + (0.0, gate["size"][1] / 2, 0.0)
        batch.current_material = METAL
        batch.box(centre, gate["size"], (.19, .40, .51), obj_id=30000 + i, emissive=.12)
        for x in np.linspace(-gate["size"][0] / 2 + 2, gate["size"][0] / 2 - 2, 7):
            batch.box(centre + (x, 0, -gate["size"][2] / 2 - .05), (.08, gate["size"][1] - .5, .06),
                      (.55, .84, .84), obj_id=30000 + i, emissive=.4)
    steel, timber, canvas = (.30, .31, .33), (.46, .31, .18), (.58, .20, .13)
    for k, lift in enumerate(env.world.lifts):
        bottom, rise, cross, top = lift.path
        obj = 31000 + k
        batch.current_material = METAL
        for dx, dz in ((-1.9, -1.9), (1.9, -1.9), (-1.9, 1.9), (1.9, 1.9)):
            batch.tube(bottom + (dx, -1.0, dz), rise + (dx * .55, 1.5, dz * .55), .16, steel, obj_id=obj)
        for y in np.arange(bottom[1] + 6.0, rise[1], 8.0):
            batch.box((bottom[0], y, bottom[2]), (4.2, .18, .18), steel, obj_id=obj)
            batch.box((bottom[0], y, bottom[2]), (.18, .18, 4.2), steel, obj_id=obj)
        for dx, dz in ((-1.4, -1.4), (1.4, -1.4), (-1.4, 1.4), (1.4, 1.4)):
            batch.tube(top + (dx, -.5, dz), cross + (dx, 1.2, dz), .14, steel, obj_id=obj)
        heading = (cross - rise)[[0, 2]] / max(np.linalg.norm((cross - rise)[[0, 2]]), 1e-6)
        side = np.array([-heading[1], 0.0, heading[0]])
        for s in (-.7, .7):
            batch.tube(rise + (0, 1.6, 0) + side * s, cross + (0, 1.6, 0) + side * s, .05, (.12, .12, .13),
                       obj_id=obj)
        cabin = lift.point(lift.position)
        batch.current_material = BARK
        batch.box(cabin + (0, -.15, 0), (2.8, .3, 2.8), timber, obj_id=obj)
        for dx, dz in ((-1.25, -1.25), (1.25, -1.25), (-1.25, 1.25), (1.25, 1.25)):
            batch.box(cabin + (dx, 1.2, dz), (.14, 2.4, .14), timber, obj_id=obj)
        batch.box(cabin + (0, .55, -1.32), (2.8, .1, .08), timber, obj_id=obj)
        batch.box(cabin + (0, .55, 1.32), (2.8, .1, .08), timber, obj_id=obj)
        batch.current_material = CLOTH
        batch.box(cabin + (0, 2.45, 0), (3.1, .12, 3.1), canvas, obj_id=obj)
        batch.current_material = METAL
        batch.tube(cabin + (0, 2.5, 0), cabin + (0, 3.4, 0), .06, (.12, .12, .13), obj_id=obj)
    _add_batch(scene, batch)
    scene.metadata["fullscale"] = dict(kind="route-preview", state_digest=env.state_digest(),
                                       facts=sorted(env.fullscale.state.facts),
                                       active_gates=[g["id"] for g in env.fullscale.architecture.active()],
                                       lifts={l.id: dict(position=l.position, riding=l.riding)
                                              for l in env.world.lifts},
                                       note="Lift towers, cables, and cabins are presentation geometry; "
                                            "the gate box matches its collision.")
    return scene


# ---------------------------------------------------------------- render
class CyclesFarm:
    """Render snapshots with one Blender process per GPU slot."""

    def __init__(self, blender, gpus, per_gpu, samples, threads, device):
        self.blender, self.samples, self.threads, self.device = blender, samples, threads, device
        self.slots = queue.Queue()
        for gpu in gpus:
            for _ in range(per_gpu):
                self.slots.put(str(gpu))
        self.workers = len(gpus) * per_gpu
        self.errors = []
        self.lock = threading.Lock()

    def render(self, scene_path, output, log):
        gpu = self.slots.get()
        try:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            command = [self.blender, "--background", "--factory-startup", "--python-exit-code", "1",
                       "--python", str(ROOT / "owb/render/blender_scene.py"), "--", "--scene", str(scene_path),
                       "--output", str(output), "--samples", str(self.samples), "--threads", str(self.threads),
                       "--device", self.device]
            with open(log, "w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            if result.returncode or not Path(output).is_file():
                raise RuntimeError(f"Cycles failed for {scene_path}: see {log}")
        finally:
            self.slots.put(gpu)


def blender_version(blender):
    out = subprocess.run([blender, "--version"], capture_output=True, text=True).stdout.splitlines()
    return out[0].strip() if out else "unknown"


def render(directory, cache, *, gpus=(0,), per_gpu=1, samples=64, threads=8, device="OPTIX",
           blender=None, max_frames=None):
    from owb.render.presentation import resolve_blender

    directory, cache = Path(directory), Path(cache)
    manifest = json.loads((directory / "manifest.json").read_text())
    verification = json.loads((directory / "verification.json").read_text())
    assert verification["ok"] and verification["actions_sha256"] == sha256(directory / "actions.jsonl"), \
        "render only verified traces"
    rows = read_rows(directory)
    plan = frame_plan(rows)
    blender = resolve_blender(blender)
    settings = dict(renderer="cycles", blender=blender_version(blender), device=device, samples=samples,
                    size=list(SCENE_SIZE), denoising=True, max_bounces=6, view_transform="AgX")
    frames_dir = cache / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    work = cache / "work"
    work.mkdir(exist_ok=True)
    farm = CyclesFarm(blender, gpus, per_gpu, samples, threads, device)
    by_step = {p["step"]: p for p in plan}
    env = make_env(manifest["seed"])
    started, rendered, reused = time.time(), 0, 0
    pending = []

    def finish(step, digest, scene_path, output, log):
        farm.render(scene_path, output, log)
        record = dict(step=step, digest=digest, settings=settings, png_sha256=sha256(output))
        save_json(frames_dir / f"{step:06d}.json", record)
        scene_path.unlink(missing_ok=True)
        log.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=farm.workers) as pool:
        for row in rows:
            env.step(native_action(row["action"]))
            assert env.state_digest() == row["digest"], f"replay diverged at step {row['step']}"
            if row["step"] not in by_step:
                continue
            step = row["step"]
            png, meta = frames_dir / f"{step:06d}.png", frames_dir / f"{step:06d}.json"
            if png.exists() and meta.exists():
                record = json.loads(meta.read_text())
                if (record["digest"] == row["digest"] and record["settings"] == settings
                        and record["png_sha256"] == sha256(png)):
                    reused += 1
                    continue
            if max_frames is not None and rendered >= max_frames:
                continue
            while len([f for f in pending if not f.done()]) >= 2 * farm.workers:
                time.sleep(0.2)
            for f in [f for f in pending if f.done()]:
                f.result()
                pending.remove(f)
            scene_path = work / f"{step:06d}.npz"
            scene_path.unlink(missing_ok=True)
            fullscale_scene(env, *SCENE_SIZE).save(scene_path)
            pending.append(pool.submit(finish, step, row["digest"], scene_path, png, work / f"{step:06d}.log"))
            rendered += 1
            if rendered % 25 == 0:
                rate = (time.time() - started) / rendered
                print(f"rendered {rendered} new frames ({reused} reused), {rate:.1f} s/frame wall, "
                      f"step {step}/{rows[-1]['step']}", flush=True)
        for f in pending:
            f.result()
    env.close()
    missing = [p["step"] for p in plan if not (frames_dir / f"{p['step']:06d}.png").exists()]
    print(f"render pass done: {rendered} new, {reused} reused, {len(missing)} missing, "
          f"{time.time() - started:.0f} s", flush=True)
    return dict(settings=settings, missing=missing, plan=plan)


# --------------------------------------------------------------- overlay
def load_font(size, bold=False):
    from owb.scripts.teyvat_reference import load_font as legacy
    return legacy(size, bold)


class Overlay:
    def __init__(self, env, rows, manifest, settings):
        from PIL import Image
        from open_world_bench.architecture import Architecture
        from open_world_bench.world_map import terrain_surface

        self.manifest, self.settings = manifest, settings
        self.font = {n: load_font(n) for n in (13, 15, 17, 19, 22)}
        self.bold = {n: load_font(n, True) for n in (15, 17, 19, 24, 30)}
        fw = env.fullscale
        rgb, self.bounds = terrain_surface(fw.terrain, fw.props, Architecture(fw.buildings), 256)
        self.map_size = 296
        self.base_map = Image.fromarray(np.ascontiguousarray(rgb.transpose(1, 0, 2))).resize(
            (self.map_size, self.map_size), Image.Resampling.LANCZOS)
        self.trace = np.array([r["position_after"] for r in rows])[:, [0, 2]]
        self.lifts = [(l.path[0][[0, 2]], l.path[-1][[0, 2]]) for l in env.world.lifts]
        self.gate = fw.gates[0]

    def _map_xy(self, xz):
        x0, z0, x1, z1 = self.bounds
        return ((xz[0] - x0) / (x1 - x0) * self.map_size, (xz[1] - z0) / (z1 - z0) * self.map_size)

    def compose(self, scene_rgb, row, entry, rows_index):
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", CANVAS, (14, 18, 21))
        canvas.paste(Image.fromarray(scene_rgb), (0, 0))
        draw = ImageDraw.Draw(canvas)
        accent, text, muted, good = (92, 184, 196), (232, 238, 240), (140, 156, 162), (120, 200, 140)
        # Right panel: milestone list.
        x = 1296
        draw.text((x, 16), "TEYVAT · FULL-SCALE TASK A", font=self.bold[17], fill=text)
        draw.rounded_rectangle((x, 44, x + 150, 68), 5, fill=(196, 120, 40))
        draw.text((x + 10, 47), LABEL, font=self.bold[15], fill=(20, 16, 12))
        draw.text((x + 160, 47), f"seed {self.manifest['seed']}", font=self.font[15], fill=muted)
        done = {m["id"] for m in self.manifest["milestones"]
                if m["completed"] is not None and m["completed"] <= row["time"] + 1e-9}
        current = row["milestone"]
        y = 82
        for m in self.manifest["milestones"]:
            if m["id"] in done:
                colour, mark = good, "✓"
            elif m["id"] == current:
                colour, mark = accent, "▸"
            else:
                colour, mark = muted, " "
            title = m["title"] if len(m["title"]) <= 27 else m["title"][:26] + "…"
            font = self.bold[15] if m["id"] == current else self.font[15]
            draw.text((x, y), f"{mark} {m['id']}  {title}", font=font, fill=colour)
            y += 24
        # Route map.
        mx, my = 1296, 404
        canvas.paste(self.base_map, (mx, my))
        pts = [self._map_xy(p) for p in self.trace[::4]]
        draw.line([(mx + u, my + v) for u, v in pts], fill=(255, 255, 255), width=1)
        k = max(1, rows_index // 4)
        if k > 1:
            draw.line([(mx + u, my + v) for u, v in pts[:k]], fill=(255, 196, 64), width=3)
        for a, b in self.lifts:
            (u0, v0), (u1, v1) = self._map_xy(a), self._map_xy(b)
            draw.line([(mx + u0, my + v0), (mx + u1, my + v1)], fill=(250, 220, 60), width=3)
        gx, _, gz = self.gate["position"]
        u, v = self._map_xy((gx, gz))
        draw.rectangle((mx + u - 4, my + v - 4, mx + u + 4, my + v + 4),
                       fill=(120, 200, 140) if "corridor_open" in row["facts"] else (220, 60, 60))
        u, v = self._map_xy((row["position"][0], row["position"][2]))
        draw.ellipse((mx + u - 6, my + v - 6, mx + u + 6, my + v + 6), fill=(255, 80, 60), outline=(255, 255, 255))
        draw.text((mx, my + self.map_size + 2), "Route map (diagnostic shading)", font=self.font[13], fill=muted)
        # Bottom bar.
        y0 = SCENE_SIZE[1] + 12
        title = f"{current} · {row['title']}" if current else "Route complete"
        draw.text((24, y0), title, font=self.bold[30], fill=text)
        now = row["explanation"] or ""
        if row["checkpoint_label"] and current and row["checkpoint_label"] not in now:
            now = f"{now}   →  next checkpoint: {row['checkpoint_label']}"
        draw.text((24, y0 + 42), now, font=self.font[19], fill=accent)
        minutes, seconds = divmod(int(row["time"]), 60)
        speed = "paused on event" if entry["repeats"] > 1 else f"timelapse {entry['speed']}×"
        stats = (f"sim {minutes:02d}:{seconds:02d}   {speed}   height {row['position'][1]:.0f} m   "
                 f"HP {row['hp']:.0f}/{row['max_hp']:.0f}   stamina")
        draw.text((24, y0 + 74), stats, font=self.font[17], fill=text)
        sx = 24 + draw.textlength(stats, font=self.font[17]) + 10
        frac = max(0.0, min(1.0, row["stamina"] / max(row["stamina_max"], 1e-6)))
        draw.rounded_rectangle((sx, y0 + 78, sx + 140, y0 + 92), 4, outline=muted)
        draw.rounded_rectangle((sx + 2, y0 + 80, sx + 2 + 136 * frac, y0 + 90), 3, fill=(120, 200, 140))
        facts = ", ".join(row["facts"]) or "none"
        draw.text((sx + 160, y0 + 74), f"open gates: {facts}", font=self.font[17], fill=muted)
        for k, line in enumerate(self._wrap(draw, f"{LABEL}: {DISCLAIMER}", self.font[13], 1240)[:2]):
            draw.text((24, y0 + 106 + 18 * k), line, font=self.font[13], fill=muted)
        s = self.settings
        draw.text((24, y0 + 146), f"{s['blender']} Cycles · {s['device']} GPU · {s['size'][0]}×{s['size'][1]} · "
                  f"{s['samples']} spp · denoised · every action replay-verified", font=self.font[13], fill=muted)
        # Event banner.
        banner = self._banner(row, entry["mode"])
        if banner and entry["repeats"] > 1:
            w = draw.textlength(banner, font=self.bold[24]) + 40
            draw.rounded_rectangle((640 - w / 2, 20, 640 + w / 2, 66), 8, fill=(14, 18, 21))
            draw.text((640 - w / 2 + 20, 29), banner, font=self.bold[24], fill=text)
        return np.asarray(canvas)

    @staticmethod
    def _wrap(draw, text, font, width):
        lines, line = [], ""
        for word in text.split():
            trial = f"{line} {word}".strip()
            if draw.textlength(trial, font=font) > width and line:
                lines.append(line)
                line = word
            else:
                line = trial
        return lines + ([line] if line else [])

    @staticmethod
    def _banner(row, mode):
        kinds = {e["type"]: e for e in row["events"]}
        if mode != "glide":
            kinds.pop("move_land", None)
        if "milestone_complete" in kinds:
            e = kinds["milestone_complete"]
            return f"✓ {e['milestone']} reached · {e['title']}"
        if "fact_opened" in kinds:
            return {"ferry_open": "Ferry lift running", "corridor_open": "Ravine gate opens",
                    "cap_installed": "Service lift running"}.get(kinds["fact_opened"]["fact"], "Gate opens")
        if "checkpoint" in kinds:
            return f"Checkpoint · {kinds['checkpoint']['label']}"
        if "lift_board" in kinds:
            return f"Board the lift ({kinds['lift_board']['direction']})"
        if "lift_arrived" in kinds:
            return "Lift arrived"
        if "move_glide_start" in kinds:
            return "Glider open"
        if "move_land" in kinds:
            return "Landed"
        return None


def assemble(directory, cache, settings, plan):
    import cv2
    from PIL import Image
    from owb.render.video import H264Writer, resolve_ffmpeg

    directory, cache = Path(directory), Path(cache)
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = read_rows(directory)
    index = {r["step"]: i for i, r in enumerate(rows)}
    env = make_env(manifest["seed"])
    overlay = Overlay(env, rows, manifest, settings)
    env.close()
    output = directory / "walkthrough.partial.mp4"
    writer = H264Writer(output, size=CANVAS, fps=FPS)
    frames, chapters, records = 0, [], []
    try:
        for entry in plan:
            png = cache / "frames" / f"{entry['step']:06d}.png"
            record = json.loads((cache / "frames" / f"{entry['step']:06d}.json").read_text())
            row = rows[index[entry["step"]]]
            assert record["digest"] == row["digest"] and record["settings"] == settings
            assert record["png_sha256"] == sha256(png), f"frame {png} changed"
            with Image.open(png) as image:
                scene = np.asarray(image.convert("RGB"))
            frame = overlay.compose(scene, row, entry, index[entry["step"]])
            if not chapters or chapters[-1]["id"] != row["milestone"]:
                if chapters:
                    chapters[-1]["end_seconds"] = frames / FPS
                chapters.append(dict(id=row["milestone"], title=row["title"], start_seconds=frames / FPS,
                                     simulation_start=row["time"], action_start=row["step"]))
            for _ in range(entry["repeats"]):
                writer.append_data(frame)
            records.append(dict(step=entry["step"], digest=row["digest"], png_sha256=record["png_sha256"],
                                video_start=frames, video_frames=entry["repeats"], mode=entry["mode"]))
            frames += entry["repeats"]
            if row["milestone"] == "A06" and not (directory / "preview.png").exists() and row["riding"]:
                Image.fromarray(frame).save(directory / "preview.png")
        chapters[-1]["end_seconds"] = frames / FPS
    finally:
        writer.close()
    cap = cv2.VideoCapture(str(output))
    try:
        assert cap.isOpened(), "cannot decode the video"
        decoded = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            decoded += 1
        assert decoded == frames, f"decoded {decoded} of {frames} frames"
    finally:
        cap.release()
    final = directory / "walkthrough.mp4"
    output.replace(final)
    ffmpeg = resolve_ffmpeg()
    save_json(directory / "chapters.json", chapters)
    save_json(directory / "frames.json", records)
    save_json(directory / "video-manifest.json", dict(
        kind="route-preview", label=LABEL, disclaimer=DISCLAIMER, presentation=settings,
        canvas=list(CANVAS), fps=FPS, frames=frames, seconds=frames / FPS, rendered_states=len(plan),
        timelapse=dict(walk=f"{STRIDE['walk'] * 2}x", lift=f"{STRIDE['lift'] * 2}x", glide=f"{STRIDE['glide'] * 2}x",
                       note="each rendered frame is a real replay state; event holds repeat it"),
        decoded_frames=decoded, video_sha256=sha256(final), actions_sha256=sha256(directory / "actions.jsonl"),
        encoder=dict(ffmpeg=subprocess.run([ffmpeg, "-version"], capture_output=True, text=True).stdout.splitlines()[0],
                     codec="libx264 baseline, yuv420p, faststart"),
        sources={p: sha256(ROOT / p) for p in SOURCES + PRESENTATION_SOURCES}))
    print(f"wrote {final} ({frames} frames, {frames / FPS:.1f} s)", flush=True)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("collect", help="run the route-preview agent and verify it by replay")
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--output", type=Path, required=True)
    v = commands.add_parser("verify", help="replay a recorded run in a fresh env")
    v.add_argument("directory", type=Path)
    r = commands.add_parser("render", help="render sampled states with Cycles and assemble the video")
    r.add_argument("directory", type=Path)
    r.add_argument("--cache", type=Path, required=True)
    r.add_argument("--gpus", type=int, nargs="+", default=[0])
    r.add_argument("--per-gpu", type=int, default=1)
    r.add_argument("--samples", type=int, default=64)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--device", choices=("OPTIX", "CUDA"), default="OPTIX")
    r.add_argument("--blender")
    r.add_argument("--max-frames", type=int, help="render at most this many new frames, then stop")
    args = parser.parse_args(argv)
    if args.command == "collect":
        collect(args.seed, args.output)
        verify(args.output)
    elif args.command == "verify":
        verify(args.directory)
    else:
        result = render(args.directory, args.cache, gpus=args.gpus, per_gpu=args.per_gpu, samples=args.samples,
                        threads=args.threads, device=args.device, blender=args.blender, max_frames=args.max_frames)
        if result["missing"]:
            raise SystemExit(f"{len(result['missing'])} frames still missing; run render again to continue")
        assemble(args.directory, args.cache, result["settings"], result["plan"])


if __name__ == "__main__":
    main()
