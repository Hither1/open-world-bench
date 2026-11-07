"""Minimal right-handed 3D math helpers (Y-up, OpenGL conventions).

All functions are pure numpy and work on single vectors or batches where noted.
"""
from __future__ import annotations

import numpy as np

Vec3 = np.ndarray
Mat4 = np.ndarray

UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vec3:
    return np.array([x, y, z], dtype=np.float32)


def norm(v: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Normalize a vector or a batch of vectors along `axis`."""
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def length(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(np.asarray(v, dtype=np.float32), axis=axis)


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def smoothstep(e0, e1, x):
    """Hermite interpolation between two edges.

    Edges may be reversed (``e0 > e1``), which flips the ramp; that form is
    used all over the renderer ("fade out as depth grows"), so the denominator
    must keep its sign rather than being clamped to a positive epsilon.
    """
    span = np.asarray(e1, dtype=np.float32) - np.asarray(e0, dtype=np.float32)
    span = np.where(np.abs(span) < 1e-8, np.where(span < 0.0, -1e-8, 1e-8), span)
    t = clamp((x - e0) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def damp(current, target, smoothing: float, dt: float):
    """Exponential smoothing that is stable across variable dt.

    `smoothing` is the fraction of the remaining gap left after one second.
    """
    smoothing = float(np.clip(smoothing, 1e-6, 1.0 - 1e-6))
    return target + (current - target) * (smoothing ** dt)


def yaw_pitch_to_dir(yaw: float, pitch: float) -> Vec3:
    """Forward direction for a yaw/pitch pair, both in radians.

    yaw = 0 looks down -Z; positive yaw turns toward +X.
    """
    cp = np.cos(pitch)
    return np.array([np.sin(yaw) * cp, np.sin(pitch), -np.cos(yaw) * cp], dtype=np.float32)


def look_at(eye: Vec3, target: Vec3, up: Vec3 = UP) -> Mat4:
    eye = np.asarray(eye, dtype=np.float32)
    f = norm(np.asarray(target, dtype=np.float32) - eye)
    s = norm(np.cross(f, up))
    if not np.isfinite(s).all() or np.linalg.norm(s) < 1e-6:
        s = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> Mat4:
    f = 1.0 / np.tan(np.radians(fov_y_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def rotation_y(angle: float) -> Mat4:
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def trs(translation: Vec3, yaw: float = 0.0, scale=1.0) -> Mat4:
    """Translate * RotateY * Scale, the only transform composition we need."""
    m = rotation_y(yaw)
    s = np.asarray(scale, dtype=np.float32)
    if s.ndim == 0:
        s = np.repeat(s, 3)
    m[:3, :3] *= s[None, :]
    m[:3, 3] = np.asarray(translation, dtype=np.float32)
    return m


def signed_angle_to(forward_xz: np.ndarray, target_xz: np.ndarray) -> float:
    """Yaw delta (radians) that rotates `forward_xz` onto `target_xz`.

    Matches the yaw convention of :func:`yaw_pitch_to_dir`: a positive result
    means the target lies to the right of the current heading.
    """
    f = norm(np.asarray([forward_xz[0], 0.0, forward_xz[1]], dtype=np.float32))
    t = norm(np.asarray([target_xz[0], 0.0, target_xz[1]], dtype=np.float32))
    dot = float(np.clip(f[0] * t[0] + f[2] * t[2], -1.0, 1.0))
    cross_y = f[0] * t[2] - f[2] * t[0]
    return float(np.sign(cross_y) * np.arccos(dot))
