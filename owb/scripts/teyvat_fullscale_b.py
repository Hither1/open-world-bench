"""Full-scale Teyvat Task B: world check, reference collection, replay verification, Cycles walkthrough.

    python -B -m owb.scripts.teyvat_fullscale_b world-check --seed 7 --output runs/fullscale-b/world-7
    python -B -m owb.scripts.teyvat_fullscale_b collect --seed 7 --output runs/fullscale-b/seed-7
    python -B -m owb.scripts.teyvat_fullscale_b verify runs/fullscale-b/seed-7
    python -B -m owb.scripts.teyvat_fullscale_b render runs/fullscale-b/seed-7 \\
        --cache .cache/teyvat-fullscale-b/seed-7 --gpus 0 1 --per-gpu 2

The reference controller is privileged (it reads the hidden rule assignments
and positions) and acts only through native actions. Every collected action is
replayed in a fresh environment before anything is rendered; the render pass
replays it again and checks the state digest at every step.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from owb.scripts.teyvat_fullscale_preview import (CyclesFarm, blender_version, jsonable, load_font, native_action,
                                                  read_rows, save_json, sha256)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("open_world_bench/fullscale_b_env.py", "open_world_bench/fullscale_b_policy.py",
           "open_world_bench/working_memory_probes.py", "owb/core/working_memory.py",
           "open_world_bench/fullscale_b_world.py", "open_world_bench/fullscale_mechanics.py",
           "open_world_bench/fullscale_nav.py", "open_world_bench/fullscale_physics.py",
           "open_world_bench/fullscale_reach.py", "open_world_bench/fullscale_sites.py",
           "open_world_bench/fullscale_world.py",
           # shared simulation code the course builds on
           "open_world_bench/architecture.py", "open_world_bench/camera.py", "open_world_bench/campaign_env.py",
           "open_world_bench/character.py", "open_world_bench/elements.py", "open_world_bench/env.py",
           "open_world_bench/party.py", "open_world_bench/quests/engine.py", "open_world_bench/scatter.py",
           "open_world_bench/terrain.py", "open_world_bench/world.py")
PRESENTATION_SOURCES = ("owb/scripts/teyvat_fullscale_b.py", "owb/scripts/teyvat_fullscale_preview.py",
                        "open_world_bench/render/site_assets.py", "open_world_bench/render/anime_assets.py",
                        "open_world_bench/render/soft_renderer.py",
                        "owb/render/scene_export.py", "owb/render/blender_scene.py")
SCENE_SIZE = (1280, 720)
VEGETATION_RANGE, VEGETATION_BUDGET = 420.0, 900      # presentation: plants drawn per frame
STRUCTURE_RANGE = 700.0                                # presentation: site pieces drawn per frame
FIGURE_RANGE = 220.0                                   # presentation: people and sentinels drawn per frame
PRESENTATION_STYLE = "anime"                           # soft toon shading, lawn carpet, painted sky
GRASS_RADIUS, GRASS_DENSITY = 60.0, 14.0               # presentation: lawn carpet near the camera
CANVAS = (1600, 900)
FPS = 10
# Actions per rendered frame; at 0.2 s per action and 10 fps, speed = 2 x stride.
STRIDE = dict(walk=8, swim=8, rest=8, glide=2, vent=2, mechanism=2, fight=2, done=1)
HOLDS = dict(milestone_complete=16, fact_opened=12, item_acquired=10, housing_unlocked=8, relay_interrupted=8,
             vent_launch=5, sentinel_interrupted=5, sentinel_reformed=5, sentinel_slam=6, talk=12,
             designed_context_boundary=10, relay_active=6, statue_heal=6, wm_cue_read=12, wm_attempt=8)
LABEL = "REFERENCE SOLUTION"
DISCLAIMER = ("A privileged reference controller reads the hidden rule assignments and positions and acts only "
              "through native actions (move, look, jump/glide, sprint, interact, attack, skill, party swap). It shows "
              "that the course is solvable; it is not a model evaluation.")


def make_env(seed, working_memory=True, wm_control=False):
    from open_world_bench.fullscale_b_env import TeyvatExpeditionEnv
    env = TeyvatExpeditionEnv(seed, render_enabled=False, working_memory=working_memory, wm_control=wm_control)
    env.reset(seed=seed)
    return env


def env_options(manifest):
    """Course options a run was collected with; runs from before the working-memory probes
    (no ``working_memory`` entry) replay on the legacy course."""
    wm = manifest.get("working_memory")
    return dict(working_memory=bool(wm and wm.get("enabled")), wm_control=bool(wm and wm.get("control")))


def snapshot(env, policy):
    c, char = env.course, env.char
    m, op = c.current, c.op
    near = None
    if op is not None and op.kind == "sentinel":
        u = c.sentinels[op.target]
        near = dict(label=u.label, phase=u.posture() if c.live_sentinel(u.id) else "inert",
                    hp=float(env.world.get(u.entity).hp), max_hp=float(env.world.get(u.entity).max_hp))
    if op is not None and op.kind == "relay":
        near = dict(label=c.relays[op.target].label, cue=c.relays[op.target].cue())
    return dict(milestone=m.id if m else None, title=m.title if m else None,
                op=op.kind if op else None, explanation=policy.explanation, mode=policy.mode,
                time=round(env.world.elapsed, 6), position=char.pos.tolist(), move_state=char.state.name,
                stamina=float(char.stamina), stamina_max=float(env.controller.cfg.stamina_max), hp=float(char.hp),
                max_hp=float(env.party.max_hp), active=env.party.active.spec.name, facts=sorted(c.facts),
                pack=dict(c.pack), focus=near, local_text=env.course.local_text(char))


# ---------------------------------------------------------------- world check
def world_check(seeds, output, workers=9):
    from open_world_bench.fullscale_b_check import check_world_parallel
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for seed in seeds:
        started = time.time()
        report = check_world_parallel(seed, workers=workers)
        report["wall_seconds"] = round(time.time() - started, 1)
        report["sources"] = {p: sha256(ROOT / p) for p in SOURCES + ("open_world_bench/fullscale_b_check.py",)}
        save_json(output / f"report-{seed}.json", report)
        reports[seed] = report["ok"]
        print(f"seed {seed}: {'ok' if report['ok'] else 'FAILED'} in {report['wall_seconds']} s", flush=True)
    return reports


# ---------------------------------------------------------------- collect
def collect(seed, directory, max_steps=60_000, *, wm_control=False):
    from open_world_bench.fullscale_b_policy import ExpeditionReference
    from open_world_bench.fullscale_b_world import SUITE_VERSION
    from open_world_bench.fullscale_physics import PHYSICS_VERSION

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    env = make_env(seed, wm_control=wm_control)
    policy = ExpeditionReference(env)
    initial = env.state_digest()
    started = time.time()
    travel, terminated, falls = 0.0, False, []
    with (directory / "actions.jsonl.partial").open("w") as stream:
        for step in range(max_steps):
            before = env.char.pos.copy()
            hp_before = float(env.char.hp)
            action = policy.act()
            _, reward, terminated, truncated, info = env.step(action)
            travel += float(np.linalg.norm((env.char.pos - before)[[0, 2]]))
            if any(e["type"] == "move_fall_damage" for e in info["event_details"]):
                falls.append(dict(step=step, damage=round(hp_before - float(env.char.hp), 2)))
            row = dict(step=step, action=jsonable(action), position_before=before.tolist(),
                       position_after=env.char.pos.tolist(), events=jsonable(info["event_details"]),
                       reward=float(reward), terminated=bool(terminated), truncated=bool(truncated),
                       digest=env.state_digest(), **snapshot(env, policy))
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            if terminated or truncated or env.course.current is None:
                break
    (directory / "actions.jsonl.partial").replace(directory / "actions.jsonl")
    c = env.course
    verdict = c.verify()
    rows = read_rows(directory)
    times = {m["id"]: m["time"] for m in c.completed}
    starts = {}
    previous = 0.0
    for m in c.milestones:
        starts[m.id] = previous
        previous = times.get(m.id, previous)
    modes = {}
    for r in rows:
        modes[r["mode"]] = modes.get(r["mode"], 0) + 1
    manifest = dict(
        kind="reference-solution", course=c.version, suite=SUITE_VERSION, physics=PHYSICS_VERSION, seed=seed,
        world_digest=env.world_digest, initial_digest=initial, final_digest=env.state_digest(),
        success=bool(verdict["success"]), verdict=verdict, terminated=bool(terminated), actions=step + 1,
        simulated_seconds=round(env.world.elapsed, 3), travel_metres=round(travel, 1),
        min_hp=round(min(r["hp"] for r in rows), 2), deaths=c.deaths, falls=falls, modes=modes,
        rules=dict(relay={k: v.name.lower() for k, v in c.relay_elements.items()},
                   sentinel=dict(breaker=c.guard_elements[0].name.lower(), interrupter=c.guard_elements[1].name.lower()),
                   vent=dict(element=c.vent_element.name.lower(), safe_beat=c.safe_beat),
                   outpost_version=c.outpost_version, note="private; never shown to the agent"),
        milestones=[dict(id=m.id, title=m.title, start=starts[m.id], completed=times.get(m.id),
                         ops=[dict(kind=o.kind, target=o.target) for o in m.ops]) for m in c.milestones],
        relays={k: dict(version=u.version, applications=u.applications, invalid=u.invalid,
                        unnecessary_drains=u.unnecessary_drains, interrupted=u.interrupted)
                for k, u in c.relays.items()},
        sentinels={k: dict(cycles=u.cycles, wrong_casts=u.wrong_casts, interrupts=u.interrupts,
                           missed_windows=u.missed_windows, pulses=u.pulses, slams=u.slams)
                   for k, u in c.sentinels.items()},
        vents={k: dict(route=v.route, leg=v.leg, launches=v.launches, sputters=v.sputters, charges=v.charges)
               for k, v in c.vents.items()},
        working_memory=dict(enabled=True, control=bool(wm_control), contract="docs/working-memory-probes.md",
                            probes=c.wm_summary(),
                            note="evaluator-only; probe content reaches the policy only through the cue while it "
                                 "is in reach (5 s) or, under wm_control, at the use site"),
        replans=policy.replans, route_restarts=policy.route_restarts, decision_seconds=0.2,
        controller="ExpeditionReference: privileged; plans on the under-model graph; native actions only",
        limitations=[DISCLAIMER, "Seed 7 only in this pass; other seeds are untested.",
                     "Vent launches are kinematic rises to the column top; sentinels are stationary constructs."],
        sources={p: sha256(ROOT / p) for p in SOURCES}, collected_wall_seconds=round(time.time() - started, 1))
    save_json(directory / "manifest.json", manifest)
    env.close()
    print(f"seed {seed}: {'success' if manifest['success'] else 'FAILED'}: {manifest['actions']} actions, "
          f"{manifest['simulated_seconds'] / 60:.1f} simulated minutes, {travel / 1000:.2f} km, "
          f"min HP {manifest['min_hp']}, deaths {c.deaths}, falls {len(falls)}", flush=True)
    return manifest


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = read_rows(directory)
    for path, digest in manifest["sources"].items():
        if sha256(ROOT / path) != digest:
            raise SystemExit(f"Replay source changed: {path}")
    env = make_env(manifest["seed"], **env_options(manifest))
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
    assert env.course.verify()["success"] == manifest["success"]
    result = dict(ok=True, actions=len(rows), success=manifest["success"], course_options=env_options(manifest),
                  working_memory=env.course.wm_summary() if env.course.working_memory else None,
                  checked=["positions before and after every action", "event streams", "rewards and flags",
                           "state digest after every action", "initial and final digests", "world digest",
                           "course verdict", "source hashes"],
                  actions_sha256=sha256(directory / "actions.jsonl"), sources=manifest["sources"])
    save_json(directory / "verification.json", result)
    env.close()
    print(f"verified {len(rows)} actions", flush=True)
    return result


# ------------------------------------------------------------------ plan
def mode_of(row):
    if row["explanation"].startswith(("Rest", "Wait for the", "Walk to the statue")):
        return "rest"
    return row["mode"] if row["mode"] in STRIDE else "walk"


def frame_plan(rows):
    plan = []
    for i, row in enumerate(rows):
        kinds = {e["type"] for e in row["events"]}
        mode = mode_of(row)
        hold = max([HOLDS[k] for k in kinds if k in HOLDS], default=0)
        if i == 0 or i == len(rows) - 1 or hold or row["step"] % STRIDE[mode] == 0:
            plan.append(dict(step=row["step"], repeats=max(1, hold), mode=mode,
                             speed=round(0.2 * STRIDE[mode] * FPS), events=sorted(kinds & set(HOLDS))))
    plan[-1]["repeats"] = 40
    return plan


# ----------------------------------------------------------------- scene
PHASE_COLOURS = dict(shield=(.25, .55, .95), exposed=(.95, .9, .5), windup=(1.0, .55, .15), vulnerable=(.95, .2, .2))


def expedition_scene(env, width, height):
    """Cycles snapshot of the actual state plus presentation cues for gates, vents and sentinels."""
    from open_world_bench.render.assets import METAL, STONE
    from open_world_bench.render.meshes import MeshBatch
    from owb.render.scene_export import _add_batch, capture_scene

    from open_world_bench.render.site_assets import add_site_pieces, add_vegetation
    from open_world_bench.scatter import PropField

    # The shared capture draws at most 212 plants, which thins the groves. Capture without
    # plants, then add every plant in view here. Props are static; the field is restored.
    props = env.world.props
    env.world.props = PropField(props.pos[:0], props.yaw[:0], props.scale[:0], props.kind[:0])
    from open_world_bench.world import EntityKind
    figures = [e for e in env.world.entities.values()
               if e.alive and e.kind in (EntityKind.NPC, EntityKind.ENEMY, EntityKind.BOSS)] if PRESENTATION_STYLE else []
    env.presentation_hides_character = bool(PRESENTATION_STYLE)
    env.world.presentation_hidden = {e.id for e in figures}
    try:
        scene = capture_scene(env, width=width, height=height)
    finally:
        env.world.props = props
        env.presentation_hides_character = False
        env.world.presentation_hidden = ()
    if PRESENTATION_STYLE:
        from open_world_bench.render.site_assets import GRASS_VARIANTS, grass_clump
        scene.metadata["style"] = dict(name=PRESENTATION_STYLE)
        names = []
        for k in range(GRASS_VARIANTS):
            vertices, colors, normals, _, _ = grass_clump(k)
            names.append(f"Lawn clump {k}")
            scene.add(names[-1], vertices, colors, normals=normals, material="foliage")
            scene.meshes[-1]["prototype"] = True
        from open_world_bench.render.site_assets import far_peaks
        vertices, colors = far_peaks(int(env.fullscale.seed))
        scene.add("Far peaks", vertices, colors, material="terrain")
        scene.metadata["ground_cover"] = dict(target="Island terrain", prototypes=names, radius=GRASS_RADIUS,
                                              density=GRASS_DENSITY, seed=int(env.fullscale.seed), min_up=0.9,
                                              min_height=2.4, clump_floor=0.2)
    eye = np.asarray(scene.metadata["camera"]["eye"], float)
    forward = np.asarray(scene.metadata["camera"]["target"], float) - eye
    batch = MeshBatch()
    plants = add_vegetation(batch, props, eye, forward, env.char.pos, max_distance=VEGETATION_RANGE,
                            budget=VEGETATION_BUDGET, stylised=bool(PRESENTATION_STYLE))
    if PRESENTATION_STYLE:
        from open_world_bench.render.anime_assets import add_hero, outline_hull
        from open_world_bench.render.anime_assets import RAIDER_LOOK, add_person, add_sentinel
        first = add_hero(batch, env.char, env.party.active.spec)
        sentinel_of = {unit.entity: (k, unit) for k, unit in enumerate(env.course.sentinels.values())}
        for e in figures:
            offset = env.char.pos - e.pos
            distance = float(np.hypot(offset[0], offset[2]))
            if float(np.hypot(e.pos[0] - eye[0], e.pos[2] - eye[2])) > FIGURE_RANGE:
                continue
            facing = float(np.arctan2(offset[0], -offset[2]))
            if e.id in sentinel_of:
                k, unit = sentinel_of[e.id]
                add_sentinel(batch, e.pos, facing, unit.phase, env.course.live_sentinel(unit.id), env.world.elapsed, variant=k)
            elif e.kind == EntityKind.ENEMY:
                chasing = 1.9 < distance < e.aggro_radius and not e.elemental.disabled
                add_person(batch, e.id, e.pos, e.yaw if chasing else facing, phase=(e.bob * 1.7) % 1.0,
                           speed=3.2 if chasing else 0.0, look=RAIDER_LOOK)
            else:
                add_person(batch, e.id, e.pos, facing if distance < 14.0 else e.yaw, phase=(e.bob * .22) % 1.0)
        scene.add("Figure outlines", outline_hull(batch.verts[first:]), (0.1, 0.08, 0.14), material="outline")
    add_site_pieces(batch, env.fullscale.pieces, eye, STRUCTURE_RANGE)
    c = env.course
    for i, gate in enumerate(env.fullscale.architecture.active()):
        centre = np.asarray(gate["position"], float) + (0.0, gate["size"][1] / 2, 0.0)
        batch.current_material = METAL
        batch.box(centre, gate["size"], (.19, .40, .51), obj_id=30000 + i, emissive=.12)
        for x in np.linspace(-gate["size"][0] / 2 + 2, gate["size"][0] / 2 - 2, 7):
            batch.box(centre + (x, 0, gate["size"][2] / 2 + .05), (.08, gate["size"][1] - .5, .06),
                      (.55, .84, .84), obj_id=30000 + i, emissive=.4)
    now = env.world.elapsed
    eye = env.char.pos
    for k, vent in enumerate(c.vents.values()):
        if np.hypot(vent.pad[0] - eye[0], vent.pad[1] - eye[2]) > 260:
            continue
        base = np.array([vent.pad[0], vent.pad_height + 0.05, vent.pad[1]])
        powered = c.vent_powered(vent)
        bright = vent.bright(now)
        colour = (.55, .95, .9) if powered else (.25, .27, .3)
        glow = (.9 if bright else .25) if powered else 0.0
        batch.current_material = STONE
        batch.box(base + (0, -.1, 0), (3.6, .25, 3.6), (.38, .37, .35), obj_id=32000 + k)
        batch.current_material = METAL
        for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
            p = base + (1.5 * np.cos(a), .2, 1.5 * np.sin(a))
            batch.box(p, (.35, .18, .35), colour, obj_id=32000 + k, emissive=glow)
        if powered and (vent.charged or (c.launch is not None and c.launch[0] == vent.id)):
            top = vent.apex - vent.pad_height
            for s in range(0, int(top), 3):
                batch.box(base + (0, s + 1.5, 0), (.6 + .2 * np.sin(s), .12, .6), (.6, 1.0, .95),
                          obj_id=32000 + k, emissive=1.2)
    for k, unit in enumerate(c.sentinels.values()):
        e = env.world.get(unit.entity)
        if PRESENTATION_STYLE or not e.alive or np.linalg.norm(e.pos - eye) > 120:
            continue
        colour = PHASE_COLOURS[unit.phase] if c.live_sentinel(unit.id) else (.3, .3, .32)
        ring_glow = .9 if c.live_sentinel(unit.id) else 0.0
        batch.current_material = METAL
        for r, y in ((2.6, .4), (2.0, 2.6)):
            for a in np.linspace(0, 2 * np.pi, 13)[:-1]:
                batch.box(e.pos + (r * np.cos(a), y, r * np.sin(a)), (.3, .25, .3), colour, obj_id=33000 + k,
                          emissive=ring_glow)
    _add_batch(scene, batch)
    scene.metadata["fullscale_b"] = dict(kind="reference-solution", state_digest=env.state_digest(),
                                         facts=sorted(c.facts), plants_drawn=int(plants),
                                         note="Vent pads, columns and sentinel phase rings are presentation cues; "
                                              "gate boxes match their collision.")
    return scene


def render(directory, cache, *, gpus=(0,), per_gpu=1, samples=64, threads=8, device="OPTIX", blender=None,
           max_frames=None, steps=None):
    """Render planned states not yet in the cache. ``steps=(start, end)`` limits this pass to a
    step range, so several passes can share one cache; each still replays from step 0."""
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
                    size=list(SCENE_SIZE), denoising=True, max_bounces=6,
                    view_transform="Standard" if PRESENTATION_STYLE else "AgX", style=PRESENTATION_STYLE or "default",
                    plants_per_frame=VEGETATION_BUDGET, lawn_radius=GRASS_RADIUS)
    frames_dir = cache / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    work = cache / "work"
    work.mkdir(exist_ok=True)
    farm = CyclesFarm(blender, gpus, per_gpu, samples, threads, device)
    by_step = {p["step"]: p for p in plan}
    env = make_env(manifest["seed"], **env_options(manifest))
    started, rendered, reused = time.time(), 0, 0
    pending = []

    def finish(step, digest, scene_path, output, log):
        farm.render(scene_path, output, log)
        save_json(frames_dir / f"{step:06d}.json", dict(step=step, digest=digest, settings=settings,
                                                         png_sha256=sha256(output)))
        scene_path.unlink(missing_ok=True)
        log.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=farm.workers) as pool:
        for row in rows:
            env.step(native_action(row["action"]))
            assert env.state_digest() == row["digest"], f"replay diverged at step {row['step']}"
            if steps is not None and row["step"] >= steps[1]:
                break
            if row["step"] not in by_step or (steps is not None and row["step"] < steps[0]):
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
            png.unlink(missing_ok=True)          # a stale image without a matching record is redone
            expedition_scene(env, *SCENE_SIZE).save(scene_path)
            pending.append(pool.submit(finish, step, row["digest"], scene_path, png, work / f"{step:06d}.log"))
            rendered += 1
            if rendered % 50 == 0:
                rate = (time.time() - started) / rendered
                print(f"rendered {rendered} new frames ({reused} reused), {rate:.2f} s/frame wall, "
                      f"step {step}/{rows[-1]['step']}", flush=True)
        for f in pending:
            f.result()
    env.close()
    missing = [p["step"] for p in plan if not (frames_dir / f"{p['step']:06d}.png").exists()]
    print(f"render pass done: {rendered} new, {reused} reused, {len(missing)} missing, "
          f"{time.time() - started:.0f} s", flush=True)
    return dict(settings=settings, missing=missing, plan=plan)


# --------------------------------------------------------------- overlay
class Overlay:
    def __init__(self, env, rows, manifest, settings):
        from PIL import Image
        from open_world_bench.architecture import Architecture
        from open_world_bench.world_map import terrain_surface

        self.manifest, self.settings = manifest, settings
        self.font = {n: load_font(n) for n in (12, 13, 14, 15, 17, 19)}
        self.bold = {n: load_font(n, True) for n in (13, 15, 17, 24, 28)}
        fw = env.fullscale
        rgb, self.bounds = terrain_surface(fw.terrain, fw.props, Architecture(fw.buildings), 256)
        self.map_size = 250
        self.base_map = Image.fromarray(np.ascontiguousarray(rgb.transpose(1, 0, 2))).resize(
            (self.map_size, self.map_size), Image.Resampling.LANCZOS)
        self.trace = np.array([r["position_after"] for r in rows])[:, [0, 2]]
        self.vents = [(v.pad, v.target_xz) for v in fw.vents]
        self.gates = fw.gates

    def _map_xy(self, xz):
        x0, z0, x1, z1 = self.bounds
        return ((xz[0] - x0) / (x1 - x0) * self.map_size, (xz[1] - z0) / (z1 - z0) * self.map_size)

    def compose(self, scene_rgb, row, entry, rows_index):
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", CANVAS, (14, 18, 21))
        canvas.paste(Image.fromarray(scene_rgb), (0, 0))
        draw = ImageDraw.Draw(canvas)
        accent, text, muted, good, warn = (92, 184, 196), (232, 238, 240), (140, 156, 162), (120, 200, 140), (236, 170, 80)
        x = 1294
        draw.text((x, 12), "TEYVAT · FULL-SCALE TASK B", font=self.bold[15], fill=text)
        draw.text((x, 32), "The Expedition of Three Arts", font=self.font[13], fill=muted)
        draw.rounded_rectangle((x, 52, x + 176, 74), 5, fill=(196, 120, 40))
        draw.text((x + 8, 55), LABEL, font=self.bold[13], fill=(20, 16, 12))
        draw.text((x + 186, 55), f"seed {self.manifest['seed']}", font=self.font[13], fill=muted)
        done = {m["id"] for m in self.manifest["milestones"]
                if m["completed"] is not None and m["completed"] <= row["time"] + 1e-9}
        current = row["milestone"]
        y = 84
        for m in self.manifest["milestones"]:
            if m["id"] in done:
                colour, mark = good, "✓"
            elif m["id"] == current:
                colour, mark = accent, "▸"
            else:
                colour, mark = muted, " "
            title = m["title"] if len(m["title"]) <= 31 else m["title"][:30] + "…"
            font = self.bold[13] if m["id"] == current else self.font[12]
            draw.text((x, y), f"{mark} {m['id']} {title}", font=font, fill=colour)
            y += 16
        mx, my = 1294 + 27, 488
        canvas.paste(self.base_map, (mx, my))
        pts = [self._map_xy(p) for p in self.trace[::6]]
        draw.line([(mx + u, my + v) for u, v in pts], fill=(255, 255, 255), width=1)
        k = max(1, rows_index // 6)
        if k > 1:
            draw.line([(mx + u, my + v) for u, v in pts[:k]], fill=(255, 196, 64), width=2)
        for gate in self.gates:
            u, v = self._map_xy((gate["position"][0], gate["position"][2]))
            draw.rectangle((mx + u - 3, my + v - 3, mx + u + 3, my + v + 3),
                           fill=good if gate["opens"] in row["facts"] else (220, 60, 60))
        u, v = self._map_xy((row["position"][0], row["position"][2]))
        draw.ellipse((mx + u - 5, my + v - 5, mx + u + 5, my + v + 5), fill=(255, 80, 60), outline=(255, 255, 255))
        draw.text((mx, my + self.map_size + 2), "Route map (diagnostic shading)", font=self.font[12], fill=muted)
        pack = ", ".join(row["pack"]) or "empty"
        draw.text((x, my + self.map_size + 22), f"Pack: {pack}"[:44], font=self.font[13], fill=text)
        facts = ", ".join(f.replace("_", " ") for f in row["facts"]) or "none"
        for k2, line in enumerate(self._wrap(draw, f"Open: {facts}", self.font[12], 300)[:3]):
            draw.text((x, my + self.map_size + 42 + 16 * k2), line, font=self.font[12], fill=muted)
        # Bottom bar.
        y0 = SCENE_SIZE[1] + 8
        title = f"{current} · {row['title']}" if current else "Expedition complete: 24 / 24"
        draw.text((24, y0), title, font=self.bold[28], fill=text)
        draw.text((24, y0 + 38), row["explanation"] or "", font=self.font[19], fill=accent)
        minutes, seconds = divmod(int(row["time"]), 60)
        speed = "paused on event" if entry["repeats"] > 1 else f"timelapse {entry['speed']}×"
        stats = (f"sim {minutes:02d}:{seconds:02d}   {speed}   height {row['position'][1]:.0f} m   "
                 f"{row['active']} HP {row['hp']:.0f}/{row['max_hp']:.0f}   stamina")
        draw.text((24, y0 + 68), stats, font=self.font[17], fill=text)
        sx = 24 + draw.textlength(stats, font=self.font[17]) + 10
        frac = max(0.0, min(1.0, row["stamina"] / max(row["stamina_max"], 1e-6)))
        draw.rounded_rectangle((sx, y0 + 72, sx + 140, y0 + 86), 4, outline=muted)
        draw.rounded_rectangle((sx + 2, y0 + 74, sx + 2 + 136 * frac, y0 + 84), 3, fill=good)
        focus = row.get("focus")
        if focus and "phase" in focus:
            draw.text((sx + 160, y0 + 68), f"{focus['label']}: {focus['phase']}  core {focus['hp']:.0f}/{focus['max_hp']:.0f}",
                      font=self.font[17], fill=warn)
        elif focus and "cue" in focus:
            draw.text((sx + 160, y0 + 68), focus["cue"][:60], font=self.font[17], fill=warn)
        seen = row.get("local_text", "")
        for k2, line in enumerate(self._wrap(draw, f"Agent's local_text: {seen}", self.font[13], 1250)[:2]):
            draw.text((24, y0 + 96 + 17 * k2), line, font=self.font[13], fill=(196, 206, 210))
        draw.text((24, y0 + 134), self._wrap(draw, f"{LABEL}: {DISCLAIMER}", self.font[12], 1560)[0],
                  font=self.font[12], fill=muted)
        s = self.settings
        draw.text((24, y0 + 152), f"{s['blender']} Cycles · {s['device']} GPU · {s['size'][0]}×{s['size'][1]} · "
                  f"{s['samples']} spp · denoised · every action replay-verified · vent columns and sentinel "
                  f"rings are presentation cues", font=self.font[12], fill=muted)
        banner = self._banner(row)
        if banner and entry["repeats"] > 1:
            w = draw.textlength(banner, font=self.bold[24]) + 40
            draw.rounded_rectangle((640 - w / 2, 18, 640 + w / 2, 62), 8, fill=(14, 18, 21))
            draw.text((640 - w / 2 + 20, 26), banner, font=self.bold[24], fill=text)
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
        return lines + ([line] if line else [""])

    @staticmethod
    def _banner(row):
        kinds = {e["type"]: e for e in row["events"]}
        if "milestone_complete" in kinds:
            e = kinds["milestone_complete"]
            return f"✓ {e['milestone']} · {e['title']}"
        if "fact_opened" in kinds:
            return {"causeway_open": "Causeway gate opens", "reserve_power": "Reserve power restored",
                    "skylift_power": "Sky-lift vents powered", "crater_gate_open": "Crater gate opens",
                    "crown_signal": "Crown landing signal lit", "summit_power": "Spire vents powered",
                    "outpost_clear": "Outpost vents powered"}.get(kinds["fact_opened"]["fact"], "Gate opens")
        if "item_acquired" in kinds:
            from open_world_bench.fullscale_b_env import ITEM_NAMES
            return f"Took the {ITEM_NAMES.get(kinds['item_acquired']['item'], kinds['item_acquired']['item'])}"
        if "designed_context_boundary" in kinds:
            return f"Fresh-context point before {kinds['designed_context_boundary']['milestone']}"
        if "housing_unlocked" in kinds:
            return "Housing unlocked"
        if "relay_interrupted" in kinds:
            return "Surge! Relay overloaded after priming"
        if "relay_active" in kinds:
            return "Relay running"
        if "vent_launch" in kinds:
            return "Vent launch"
        if "sentinel_interrupted" in kinds:
            return "Wind-up interrupted: core cracked"
        if "sentinel_reformed" in kinds:
            return "Shield reforms"
        if "sentinel_slam" in kinds:
            return "Slam!"
        if "talk" in kinds:
            return "Lesson / lead"
        if "statue_heal" in kinds:
            return "Party restored at the statue"
        return None


def assemble(directory, cache, settings, plan):
    import cv2
    from PIL import Image
    from owb.render.video import H264Writer, resolve_ffmpeg

    directory, cache = Path(directory), Path(cache)
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = read_rows(directory)
    index = {r["step"]: i for i, r in enumerate(rows)}
    env = make_env(manifest["seed"], **env_options(manifest))
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
            if (row["milestone"] == "B20" and entry["mode"] == "vent" and not (directory / "preview.png").exists()):
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
        kind="reference-solution", label=LABEL, disclaimer=DISCLAIMER, presentation=settings,
        canvas=list(CANVAS), fps=FPS, frames=frames, seconds=frames / FPS, rendered_states=len(plan),
        timelapse={k: f"{v * 2}x" for k, v in STRIDE.items()},
        timelapse_note="each rendered frame is a real replay state; event holds repeat it",
        decoded_frames=decoded, video_sha256=sha256(final), actions_sha256=sha256(directory / "actions.jsonl"),
        encoder=dict(ffmpeg=subprocess.run([ffmpeg, "-version"], capture_output=True, text=True).stdout.splitlines()[0],
                     codec="libx264 baseline, yuv420p, faststart"),
        sources={p: sha256(ROOT / p) for p in SOURCES + PRESENTATION_SOURCES}))
    print(f"wrote {final} ({frames} frames, {frames / FPS:.1f} s)", flush=True)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    w = commands.add_parser("world-check", help="gate proofs and vent flights")
    w.add_argument("--seed", type=int, nargs="+", default=[7])
    w.add_argument("--output", type=Path, required=True)
    w.add_argument("--workers", type=int, default=9)
    c = commands.add_parser("collect", help="run the reference controller and verify it by replay")
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--wm-control", action="store_true",
                   help="matched control condition: the working-memory cue is also posted at each use site")
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
    r.add_argument("--max-frames", type=int)
    r.add_argument("--steps", type=int, nargs=2, metavar=("START", "END"),
                   help="render only planned states in this step range (parallel passes share the cache)")
    args = parser.parse_args(argv)
    if args.command == "world-check":
        ok = world_check(args.seed, args.output, args.workers)
        if not all(ok.values()):
            raise SystemExit("world check failed")
    elif args.command == "collect":
        collect(args.seed, args.output, wm_control=args.wm_control)
        verify(args.output)
    elif args.command == "verify":
        verify(args.directory)
    else:
        result = render(args.directory, args.cache, gpus=args.gpus, per_gpu=args.per_gpu, samples=args.samples,
                        threads=args.threads, device=args.device, blender=args.blender, max_frames=args.max_frames,
                        steps=args.steps)
        if args.steps is not None:
            print(f"range {args.steps} done; {len(result['missing'])} planned states still missing overall")
            return
        if result["missing"]:
            raise SystemExit(f"{len(result['missing'])} frames still missing; run render again to continue")
        assemble(args.directory, args.cache, result["settings"], result["plan"])


if __name__ == "__main__":
    main()
