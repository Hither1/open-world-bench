"""Presentation frames from live simulator state; Cycles is the default.

The policy's observation renderer is independent. Presentation must never change
the simulated world or silently fall back to a different rendering backend.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .scene_export import export_scene


def resolve_blender(executable=None):
    """Resolve an explicit path, OWB_BLENDER, PATH, or local repo configuration."""
    configured = executable or os.environ.get("OWB_BLENDER")
    if configured is None:
        configured = shutil.which("blender")
    if configured is None:
        local = Path(__file__).resolve().parents[2] / ".owb-render.local.json"
        if local.exists():
            configured = json.loads(local.read_text()).get("blender")
    resolved = shutil.which(str(configured)) if configured else None
    if resolved is None:
        raise RuntimeError(
            "Blender Cycles is required for presentation output. Set OWB_BLENDER "
            "or pass --blender /path/to/blender. Use --renderer cpu only when "
            "a lightweight render is explicitly wanted; there is no automatic fallback."
        )
    return resolved


class PresentationRenderer:
    """Render actual states at presentation resolution without stepping the env."""

    def __init__(self, *, backend="cycles", width=960, height=540,
                 blender=None, samples=48, threads=8, device=None, view="presentation"):
        from .scene_export import VIEWS
        if backend not in ("cycles", "cpu"):
            raise ValueError("renderer must be 'cycles' or 'cpu'")
        if any(not isinstance(n, int) or n <= 0 for n in (width, height, samples, threads)):
            raise ValueError("render dimensions, samples, and threads must be positive integers")
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}")
        if view != "presentation" and backend != "cycles":
            raise ValueError("first-person presentation is only implemented for Cycles")
        self.view = view
        self.backend = backend
        self.width, self.height = width, height
        self.samples, self.threads = samples, threads
        self.device = device or os.environ.get("OWB_CYCLES_DEVICE", "CPU")
        if self.device not in ("CPU", "CUDA", "OPTIX"):
            raise ValueError("Cycles device must be CPU, CUDA, or OPTIX")
        self.blender = resolve_blender(blender) if backend == "cycles" else None
        self._workspace = None

    @property
    def metadata(self):
        result = dict(backend=self.backend, width=self.width, height=self.height)
        if self.view != "presentation":
            result["view"] = self.view
        if self.backend == "cycles":
            result.update(samples=self.samples, threads=self.threads, device=self.device)
        return result

    def render(self, env):
        native = getattr(env, "native", getattr(env, "unwrapped", env))
        if self.backend == "cpu":
            return self._cpu_frame(native)
        from PIL import Image
        if self._workspace is None:
            self._workspace = tempfile.TemporaryDirectory(prefix="owb-cycles-")
        directory = Path(self._workspace.name)
        snapshot, output, log = (directory / name for name in ("scene.npz", "frame.png", "blender.log"))
        snapshot.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        export_scene(native, snapshot, width=self.width, height=self.height, view=self.view)
        script = Path(__file__).with_name("blender_scene.py")
        command = [self.blender, "--background", "--factory-startup", "--python-exit-code", "1",
                   "--python", str(script), "--", "--scene", str(snapshot), "--output", str(output),
                   "--samples", str(self.samples), "--threads", str(self.threads),
                   "--device", self.device]
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode or not output.is_file():
            tail = log.read_text(errors="replace")[-2500:]
            raise RuntimeError(f"Blender Cycles render failed (exit {result.returncode}):\n{tail}")
        with Image.open(output) as image:
            frame = np.asarray(image.convert("RGB")).copy()
        if frame.shape != (self.height, self.width, 3):
            raise RuntimeError(f"Unexpected Cycles frame dimensions: {frame.shape}")
        return frame

    def _cpu_frame(self, native):
        from open_world_bench.env import OpenWorldEnv
        if isinstance(native, OpenWorldEnv):
            from open_world_bench.render import make_renderer
            settings = replace(native.cfg.render, width=self.width, height=self.height)
            renderer = make_renderer(settings, native.cfg.backend)
            try:
                return renderer.render(native.world, native.char, native.cam, fov_y=native.cfg.camera.fov_y)
            finally:
                renderer.close()
        # Native RGB is an explicit diagnostic option for the other environments.
        from PIL import Image
        frame = native.render()
        return np.asarray(Image.fromarray(frame).resize((self.width, self.height), Image.Resampling.LANCZOS))

    def close(self):
        if self._workspace is not None:
            self._workspace.cleanup()
            self._workspace = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def add_render_arguments(parser, *, backend_flag="--renderer", default="cycles"):
    parser.add_argument(backend_flag, choices=("cycles", "cpu"), default=default,
                        help="presentation renderer; Cycles is the default for Teyvat/Wildlands")
    parser.add_argument("--blender", help="Blender executable (or set OWB_BLENDER)")
    parser.add_argument("--samples", type=int, default=48, help="Cycles samples per pixel")
    parser.add_argument("--threads", type=int, default=8, help="Cycles CPU threads")
