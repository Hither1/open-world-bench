"""Render many presentation snapshots in one Blender process.

Run by ``owb.render.cycles_batch.CyclesBatchRenderer``. Jobs arrive as JSON
lines on stdin; one acknowledgement per job is written to ``--ack-fd``.

Every frame starts from Blender's factory settings and then runs
``blender_scene.main`` unchanged, so each frame is built exactly as a fresh
``blender --factory-startup`` process would build it. What is saved is process
start-up, Python/NumPy import, and GPU device and kernel initialisation.
"""
import json
import os
from pathlib import Path
import sys
import time
import traceback

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_scene  # noqa: E402

DENOISER = "OPENIMAGEDENOISE"


def option(args, name):
    return args[args.index(name) + 1]


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    ack = os.fdopen(int(option(args, "--ack-fd")), "w", buffering=1)
    samples, threads, device = option(args, "--samples"), option(args, "--threads"), option(args, "--device")
    executable = sys.argv[0]
    for line in sys.stdin:
        if not line.strip():
            continue
        job = json.loads(line)
        output = Path(job["output"])
        partial = output.with_name(output.stem + ".partial.png")
        marks = {}
        try:
            started = time.monotonic()
            bpy.ops.wm.read_factory_settings(use_empty=False)
            # Handlers are cleared by the factory reset, so register per frame.
            bpy.app.handlers.render_pre.append(lambda *_: marks.setdefault("render", time.monotonic()))
            partial.unlink(missing_ok=True)
            sys.argv = [executable, "--", "--scene", job["scene"], "--output", str(partial),
                        "--samples", samples, "--threads", threads, "--device", device]
            blender_scene.main()
            denoiser = bpy.context.scene.cycles.denoiser
            if denoiser != DENOISER or not bpy.context.scene.cycles.use_denoising:
                raise RuntimeError(f"Unexpected denoiser {denoiser}; refusing to change the presentation look")
            os.replace(partial, output)
            finished = time.monotonic()
            render_start = marks.get("render", finished)
            ack.write(json.dumps(dict(id=job["id"], ok=True, build_seconds=render_start - started,
                                      render_seconds=finished - render_start)) + "\n")
        except Exception:
            ack.write(json.dumps(dict(id=job["id"], ok=False, error=traceback.format_exc()[-4000:])) + "\n")
            raise
    ack.close()


if __name__ == "__main__":
    main()
