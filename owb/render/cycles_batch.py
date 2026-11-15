"""Cycles presentation frames from one persistent Blender process.

Same snapshots, scene builder, and settings as ``PresentationRenderer``; only
the process model differs. ``submit`` exports the current state and queues it;
``result`` waits for that frame. Callers can export the next state while
Blender renders the previous one.
"""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .presentation import PresentationRenderer
from .scene_export import export_scene


def blender_version(executable):
    out = subprocess.run([executable, "--version"], capture_output=True, text=True, check=True).stdout
    return out.splitlines()[0].strip()


class CyclesBatchRenderer:
    def __init__(self, *, width=960, height=540, blender=None, samples=48, threads=8,
                 device=None, view="presentation"):
        # Reuse the single-frame renderer's validation, Blender resolution, and metadata.
        base = PresentationRenderer(backend="cycles", width=width, height=height, blender=blender,
                                    samples=samples, threads=threads, device=device, view=view)
        self.width, self.height, self.view = base.width, base.height, base.view
        self.samples, self.threads, self.device, self.blender = base.samples, base.threads, base.device, base.blender
        self.metadata = dict(base.metadata, process="persistent", blender=blender_version(self.blender),
                             denoiser="OPENIMAGEDENOISE")
        self.timings = []
        self._workspace = None
        self._process = None
        self._acks = None
        self._pending = deque()
        self._next = 0

    def _start(self):
        # NFS may briefly keep silly-renamed files after Blender exits; never fail a finished render on that.
        self._workspace = tempfile.TemporaryDirectory(prefix="owb-cycles-batch-", ignore_cleanup_errors=True)
        directory = Path(self._workspace.name)
        self._log = directory / "blender.log"
        read_fd, write_fd = os.pipe()
        script = Path(__file__).with_name("blender_batch.py")
        command = [self.blender, "--background", "--factory-startup", "--python-exit-code", "1",
                   "--python", str(script), "--", "--ack-fd", str(write_fd), "--samples", str(self.samples),
                   "--threads", str(self.threads), "--device", self.device]
        with self._log.open("w") as stream:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stream,
                                             stderr=subprocess.STDOUT, pass_fds=(write_fd,), text=True)
        os.close(write_fd)
        self._acks = os.fdopen(read_fd, "r")

    def _failure(self, message):
        tail = self._log.read_text(errors="replace")[-2500:] if self._log.exists() else ""
        return RuntimeError(f"{message}\n{tail}")

    def submit(self, env):
        native = getattr(env, "native", getattr(env, "unwrapped", env))
        if self._process is None:
            self._start()
        if self._process.poll() is not None:
            raise self._failure(f"Blender batch process exited ({self._process.returncode})")
        ticket = self._next
        self._next += 1
        directory = Path(self._workspace.name)
        scene, output = directory / f"scene-{ticket:06}.npz", directory / f"frame-{ticket:06}.png"
        export_scene(native, scene, width=self.width, height=self.height, view=self.view)
        try:
            self._process.stdin.write(json.dumps(dict(id=ticket, scene=str(scene), output=str(output))) + "\n")
            self._process.stdin.flush()
        except BrokenPipeError:
            raise self._failure("Blender batch process stopped accepting frames") from None
        self._pending.append((ticket, scene, output))
        return ticket

    def result(self, ticket):
        from PIL import Image
        if not self._pending or self._pending[0][0] != ticket:
            raise ValueError("Results must be collected in submission order")
        _, scene, output = self._pending.popleft()
        line = self._acks.readline()
        if not line:
            self._process.wait()
            raise self._failure(f"Blender Cycles batch render failed (exit {self._process.returncode})")
        ack = json.loads(line)
        if ack["id"] != ticket or not ack["ok"]:
            raise self._failure(f"Blender Cycles render failed for frame {ticket}:\n{ack.get('error', '')}")
        self.timings.append(dict(build_seconds=ack["build_seconds"], render_seconds=ack["render_seconds"]))
        with Image.open(output) as image:
            frame = np.asarray(image.convert("RGB")).copy()
        scene.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        if frame.shape != (self.height, self.width, 3):
            raise RuntimeError(f"Unexpected Cycles frame dimensions: {frame.shape}")
        return frame

    def render(self, env):
        return self.result(self.submit(env))

    def close(self):
        if self._process is not None:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                self._process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._acks.close()
            self._process = None
        if self._workspace is not None:
            self._workspace.cleanup()
            self._workspace = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
