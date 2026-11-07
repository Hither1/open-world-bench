"""Camera model and screen-ray generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    pos: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0
    fov: float = 75.0

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward / right / up unit vectors.

        ``yaw`` 0 looks toward +x and increases toward +z, so increasing yaw
        turns the camera to its right.
        """
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        f = np.array([cp * cy, sp, cp * sy], dtype=np.float32)
        f /= np.linalg.norm(f)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        r = np.cross(f, world_up)
        n = np.linalg.norm(r)
        if n < 1e-5:                      # looking straight up or down
            r = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            r = r / n
        u = np.cross(r, f)
        return f, r.astype(np.float32), u.astype(np.float32)


_SCREEN_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def screen_offsets(width: int, height: int, fov_deg: float):
    """Cached per-pixel (sx, sy) tangent-space offsets, flattened row-major."""
    key = (width, height, round(float(fov_deg), 3))
    hit = _SCREEN_CACHE.get(key)
    if hit is not None:
        return hit
    tan_half = float(np.tan(np.deg2rad(fov_deg) * 0.5))
    aspect = width / max(height, 1)
    i = (np.arange(width, dtype=np.float32) + 0.5) / width
    j = (np.arange(height, dtype=np.float32) + 0.5) / height
    sx = (2.0 * i - 1.0) * tan_half
    sy = (1.0 - 2.0 * j) * tan_half / aspect
    SX = np.broadcast_to(sx[None, :], (height, width)).ravel().astype(np.float32)
    SY = np.broadcast_to(sy[:, None], (height, width)).ravel().astype(np.float32)
    out = (np.ascontiguousarray(SX), np.ascontiguousarray(SY))
    if len(_SCREEN_CACHE) < 64:
        _SCREEN_CACHE[key] = out
    return out


def ray_dirs(cam: Camera, width: int, height: int) -> np.ndarray:
    """``(N, 3)`` unit ray directions in world space, row-major over pixels."""
    f, r, u = cam.basis()
    sx, sy = screen_offsets(width, height, cam.fov)
    d = (f[None, :] + sx[:, None] * r[None, :] + sy[:, None] * u[None, :])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d.astype(np.float32)


def resolve_third_person(world, eye: np.ndarray, yaw: float, pitch: float,
                         distance: float, pitch_offset: float = -0.12):
    """Pull the camera back behind the agent, stopping short of geometry."""
    cam = Camera(eye, yaw, pitch + pitch_offset)
    f, _, _ = cam.basis()
    back = -f
    hit = world.raycast(eye, back, distance + 0.6)
    d = distance
    if hit.hit:
        d = max(0.6, hit.dist - 0.45)
    return Camera((eye + back * d).astype(np.float32), yaw, pitch + pitch_offset,
                  cam.fov)
