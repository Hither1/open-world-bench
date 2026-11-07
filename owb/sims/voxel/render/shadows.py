"""Orthographic mesh shadows, sampled in world space with a soft pixel filter.

This supplements the occupancy-based terrain shadows. It never writes sensor
depth or semantic buffers and needs no graphics context or external assets.
"""
from __future__ import annotations

import numpy as np


class MeshShadowMap:
    def __init__(self, triangles, sun, center, extent=48.0, resolution=384):
        self.sun = np.asarray(sun, np.float32)
        self.right = np.cross(self.sun, (0, 1, 0))
        if np.linalg.norm(self.right) < 1e-5:
            self.right = np.array((1, 0, 0), np.float32)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.sun)
        self.center = np.asarray(center, np.float32)
        self.scale = resolution / (2 * extent)
        self.size = resolution
        self.depth = np.full((resolution, resolution), -np.inf, np.float32)
        if not len(triangles):
            return
        delta = triangles - self.center
        uv = np.stack((np.sum(delta * self.right, axis=-1), np.sum(delta * self.up, axis=-1)), axis=-1) * self.scale + resolution / 2
        depth = np.sum(delta * self.sun, axis=-1)
        low = np.maximum(np.floor(uv.min(axis=1)).astype(int), 0)
        high = np.minimum(np.ceil(uv.max(axis=1)).astype(int) + 1, resolution)
        e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
        areas = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        # Subpixel leaf triangles are represented by their larger crown lobes.
        active = np.flatnonzero(np.all(high > low, axis=1) & (np.abs(areas) > 0.5))
        for i in active:
            x0, y0 = low[i]
            x1, y1 = high[i]
            ax, ay = uv[i, 0]
            bx, by = uv[i, 1]
            cx, cy = uv[i, 2]
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if abs(area) < 1e-7:
                continue
            xx = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5
            yy = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5
            a = ((bx - xx) * (cy - yy) - (by - yy) * (cx - xx)) / area
            b = ((cx - xx) * (ay - yy) - (cy - yy) * (ax - xx)) / area
            c = 1 - a - b
            inside = (a >= 0) & (b >= 0) & (c >= 0)
            z = a * depth[i, 0] + b * depth[i, 1] + c * depth[i, 2]
            target = self.depth[y0:y1, x0:x1]
            np.maximum(target, np.where(inside, z, -np.inf), out=target)

    def sample(self, points, bias=0.12):
        shape = points.shape[:-1]
        delta = np.asarray(points).reshape(-1, 3) - self.center
        uv = np.stack((np.sum(delta * self.right, axis=-1), np.sum(delta * self.up, axis=-1)), axis=-1) * self.scale + self.size / 2 - 0.5
        low = np.floor(uv).astype(int)
        frac = uv - low
        depth = np.sum(delta * self.sun, axis=-1) + bias
        visibility = np.ones(len(delta), np.float32)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            xy = low + (dx, dy)
            valid = ((xy >= 0) & (xy < self.size)).all(axis=1)
            ids = np.flatnonzero(valid)
            blocked = self.depth[xy[ids, 1], xy[ids, 0]] > depth[ids]
            weight = (frac[ids, 0] if dx else 1 - frac[ids, 0]) * (frac[ids, 1] if dy else 1 - frac[ids, 1])
            visibility[ids] -= blocked * weight * 0.82
        return visibility.reshape(shape)


def contact_occlusion(color, depth, directions, normals, valid, focal_length):
    """Small world-space-radius AO kernel; sky/water and image borders excluded."""
    h, w = depth.shape
    points = directions.reshape(h, w, 3) * depth[..., None]
    yy, xx = np.indices((h, w))
    radius = np.clip(focal_length * 0.75 / np.maximum(depth, 0.5), 1, 18)
    total = np.zeros((h, w), np.float32)
    for a in np.arange(8) * np.pi / 4 + 0.2:
        x = xx + np.rint(np.cos(a) * radius).astype(int)
        y = yy + np.rint(np.sin(a) * radius).astype(int)
        inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
        delta = points[y, x] - points
        length = np.linalg.norm(delta, axis=-1)
        above = np.sum(delta * normals, axis=-1) / np.maximum(length, 1e-4)
        total += np.maximum(above - 0.12, 0) * np.clip(1 - length / 1.5, 0, 1) * inside * valid[y, x]
    amount = np.clip(total * 0.13, 0, 0.35) * valid
    return color * (1 - amount[..., None])
