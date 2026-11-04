"""A small z-buffered triangle rasterizer in numpy.

It only has to draw a few thousand flat-shaded triangles per frame (character,
entities, nearby props), so one python loop over triangles with vectorized
inner spans is fast enough and keeps the dependency list empty.
"""
from __future__ import annotations

import numpy as np


def project(verts: np.ndarray, view: np.ndarray, proj: np.ndarray, width: int, height: int):
    """Project (T, 3, 3) world triangles to screen space.

    Returns (screen_xy, w, clipped_mask) where `w` is positive view depth in
    metres and `clipped_mask` marks triangles that can be skipped entirely.
    """
    T = len(verts)
    if T == 0:
        return (np.zeros((0, 3, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32),
                np.zeros(0, dtype=bool))
    flat = verts.reshape(-1, 3)
    homo = np.concatenate([flat, np.ones((len(flat), 1), dtype=np.float32)], axis=1)
    clip = homo @ (proj @ view).T
    w = clip[:, 3].reshape(T, 3)
    safe_w = np.where(np.abs(w) < 1e-6, 1e-6, w)
    ndc = (clip[:, :3].reshape(T, 3, 3) / safe_w[..., None])
    sx = (ndc[..., 0] * 0.5 + 0.5) * width
    sy = (1.0 - (ndc[..., 1] * 0.5 + 0.5)) * height
    screen = np.stack([sx, sy], axis=-1).astype(np.float32)

    behind = w <= 0.05
    keep = ~behind.any(axis=1)                      # drop anything crossing the eye plane
    keep &= (sx.max(axis=1) >= 0) & (sx.min(axis=1) < width)
    keep &= (sy.max(axis=1) >= 0) & (sy.min(axis=1) < height)
    return screen, w, keep


def face_normals(verts: np.ndarray, eye: np.ndarray) -> np.ndarray:
    """Flat normals, flipped to face the camera so thin quads shade correctly."""
    e1 = verts[:, 1] - verts[:, 0]
    e2 = verts[:, 2] - verts[:, 0]
    n = np.cross(e1, e2)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
    centroid = verts.mean(axis=1)
    to_eye = eye[None, :] - centroid
    flip = (np.sum(n * to_eye, axis=1) < 0.0)[:, None]
    return np.where(flip, -n, n).astype(np.float32)


def draw_triangles(
    screen: np.ndarray,
    w: np.ndarray,
    keep: np.ndarray,
    shaded: np.ndarray,
    normals: np.ndarray,
    ids: np.ndarray,
    color_buf: np.ndarray,
    depth_buf: np.ndarray,
    normal_buf: np.ndarray,
    id_buf: np.ndarray,
) -> int:
    """Z-buffer the triangles into the target buffers. Returns pixels written."""
    H, W = depth_buf.shape
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        return 0
    # Draw nearest-first so the depth test rejects most work early.
    order = idx[np.argsort(w[idx].min(axis=1))]
    written = 0
    for t in order:
        p = screen[t]
        x0 = max(int(np.floor(p[:, 0].min())), 0)
        x1 = min(int(np.ceil(p[:, 0].max())) + 1, W)
        y0 = max(int(np.floor(p[:, 1].min())), 0)
        y1 = min(int(np.ceil(p[:, 1].max())) + 1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        ys = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5
        xs = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5
        ax, ay = p[0]
        bx, by = p[1]
        cx, cy = p[2]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-9:
            continue
        inv = 1.0 / area
        l0 = ((bx - xs) * (cy - ys) - (by - ys) * (cx - xs)) * inv
        l1 = ((cx - xs) * (ay - ys) - (cy - ys) * (ax - xs)) * inv
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-5) & (l1 >= -1e-5) & (l2 >= -1e-5)
        if not inside.any():
            continue
        # Perspective-correct depth from the three vertex depths.
        iw = l0 / w[t, 0] + l1 / w[t, 1] + l2 / w[t, 2]
        depth = np.where(np.abs(iw) > 1e-9, 1.0 / np.maximum(iw, 1e-9), np.inf)
        sub_depth = depth_buf[y0:y1, x0:x1]
        hit = inside & (depth < sub_depth) & (depth > 0.02)
        if not hit.any():
            continue
        sub_depth[hit] = depth[hit]
        color_buf[y0:y1, x0:x1][hit] = shaded[t]
        normal_buf[y0:y1, x0:x1][hit] = normals[t]
        id_buf[y0:y1, x0:x1][hit] = ids[t]
        written += int(hit.sum())
    return written


def draw_point_sprites(
    screen: np.ndarray,
    w: np.ndarray,
    keep: np.ndarray,
    shaded: np.ndarray,
    normals: np.ndarray,
    ids: np.ndarray,
    color_buf: np.ndarray,
    depth_buf: np.ndarray,
    normal_buf: np.ndarray,
    id_buf: np.ndarray,
) -> int:
    """Fully vectorized path for triangles smaller than a pixel or two.

    Distant vegetation generates thousands of sub-pixel triangles; looping over
    them costs far more than the pixels they cover, so they are splatted as
    depth-tested points instead.
    """
    H, W = depth_buf.shape
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        return 0
    cx = screen[idx, :, 0].mean(axis=1)
    cy = screen[idx, :, 1].mean(axis=1)
    depth = w[idx].mean(axis=1)
    px = np.floor(cx).astype(np.int32)
    py = np.floor(cy).astype(np.int32)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (depth > 0.02)
    if not ok.any():
        return 0
    idx, px, py, depth = idx[ok], px[ok], py[ok], depth[ok]
    # Scatter far-to-near so the nearest sprite ends up as the final write.
    order = np.argsort(-depth)
    idx, px, py, depth = idx[order], px[order], py[order], depth[order]
    tmp_depth = np.full((H, W), np.inf, dtype=np.float32)
    tmp_color = np.zeros((H, W, 3), dtype=np.float32)
    tmp_normal = np.zeros((H, W, 3), dtype=np.float32)
    tmp_id = np.zeros((H, W), dtype=np.int32)
    tmp_depth[py, px] = depth
    tmp_color[py, px] = shaded[idx]
    tmp_normal[py, px] = normals[idx]
    tmp_id[py, px] = ids[idx]
    hit = tmp_depth < depth_buf
    if not hit.any():
        return 0
    depth_buf[hit] = tmp_depth[hit]
    color_buf[hit] = tmp_color[hit]
    normal_buf[hit] = tmp_normal[hit]
    id_buf[hit] = tmp_id[hit]
    return int(hit.sum())


def split_by_screen_size(screen: np.ndarray, keep: np.ndarray, threshold: float = 2.5):
    """Split kept triangles into (large, tiny) masks by screen bounding box."""
    if len(screen) == 0:
        return keep, keep
    span_x = screen[:, :, 0].max(axis=1) - screen[:, :, 0].min(axis=1)
    span_y = screen[:, :, 1].max(axis=1) - screen[:, :, 1].min(axis=1)
    tiny = (span_x < threshold) & (span_y < threshold)
    return keep & ~tiny, keep & tiny
