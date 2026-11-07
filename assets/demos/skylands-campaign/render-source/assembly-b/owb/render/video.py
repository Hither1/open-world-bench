"""Browser-compatible H.264 MP4 encoding for presentation recordings."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


def resolve_ffmpeg():
    path = os.environ.get("IMAGEIO_FFMPEG_EXE") or shutil.which("ffmpeg")
    if path is None:
        local = Path(__file__).resolve().parents[2] / ".owb-render.local.json"
        if local.exists():
            path = json.loads(local.read_text()).get("ffmpeg")
    if path is None:
        try:
            import imageio_ffmpeg
            path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError('H.264 encoding requires FFmpeg or pip install -e ".[video]"') from exc
    resolved = shutil.which(str(path))
    if resolved is None:
        raise RuntimeError(f"FFmpeg executable was not found: {path}")
    return resolved


class H264Writer:
    def __init__(self, path, *, size, fps):
        self.width, self.height = size
        if min(*size, fps) <= 0 or self.width % 2 or self.height % 2:
            raise ValueError("H.264 needs positive fps and even, positive dimensions")
        self.frames = 0
        self._closed = False
        self._log = tempfile.TemporaryFile(mode="w+")
        command = [resolve_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.width}x{self.height}",
                   "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                   "-profile:v", "baseline", "-pix_fmt", "yuv420p", "-preset", "fast",
                   "-crf", "20", "-threads", "2", "-movflags", "+faststart", str(path)]
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                             stderr=self._log)
        except Exception:
            self._log.close()
            raise

    def append_data(self, rgb):
        frame = np.asarray(rgb)
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError("Expected an RGB uint8 frame matching the encoder dimensions")
        if self._closed:
            raise RuntimeError("The video encoder is closed")
        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            self._log.seek(0)
            raise RuntimeError(f"H.264 encoding failed: {self._log.read()[-2000:]}") from exc
        self.frames += 1

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
            result = self._process.wait()
            if result:
                self._log.seek(0)
                raise RuntimeError(f"H.264 encoding failed: {self._log.read()[-2000:]}")
        finally:
            self._log.close()
