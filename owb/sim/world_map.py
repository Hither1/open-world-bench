"""Clickable world map with fog of war, shared by Teyvat, Wildlands and Skylands.

Every policy RGB frame carries a small map logo in its top-right corner. A
``click`` whose ``pointer`` lands on the logo opens a north-up map of the whole
world in place of the 3-D view. Regions the agent has not explored are covered
in shadow that shows no world content. Clicking the logo again closes the map;
any other click does nothing. Layered worlds (Skylands) add tabs on the left
edge of the open map that switch between the sky, surface and depths maps.

While the map is open the simulation is paused: each decision still counts
toward the step budget, but no simulated time passes and other inputs are
ignored. The step whose click closes the map simulates normally.

Conventions: +x (east) points right and +z (south) points down, as in the
Wildlands minimap. Pointer coordinates are normalised screen coordinates in
[0, 1], ``u`` to the right and ``v`` downward, so a click is independent of
the observation resolution.

Exploration is a disc around the agent's path, widened with altitude where the
environment supports it. Visual texture is keyed to integer world coordinates;
it never reads a simulation RNG.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

SHADOW = np.array((26, 30, 44), np.float32)
SCREEN = np.array((14, 16, 22), np.float32)
VOID = np.array((74, 96, 126), np.float32)       # no ground in this layer
PARCHMENT = np.array((232, 218, 176), np.float32)
ACCENT = np.array((255, 196, 64), np.float32)
AGENT = np.array((255, 244, 214), np.float32)
OUTLINE = np.array((22, 20, 18), np.float32)
LANDMARK = np.array((255, 166, 58), np.float32)


# --------------------------------------------------------------- utilities
def _hash(ix, iz, salt: int) -> np.ndarray:
    """Stable integer-lattice hash in [0, 1)."""
    x = np.asarray(ix, np.int64).astype(np.uint32)
    z = np.asarray(iz, np.int64).astype(np.uint32)
    h = (x * np.uint32(0x9E3779B1)) ^ (z * np.uint32(0x85EBCA77)) ^ np.uint32((salt * 0xC2B2AE3D) & 0xFFFFFFFF)
    h ^= h >> np.uint32(15)
    h = h * np.uint32(0x27D4EB2F)
    h ^= h >> np.uint32(13)
    return (h & np.uint32(0xFFFF)).astype(np.float32) / 65536.0


def value_noise(x, z, scale: float, salt: int = 0) -> np.ndarray:
    """Smooth lattice noise in [0, 1] over world coordinates."""
    gx, gz = np.broadcast_arrays(np.asarray(x, np.float64) / scale, np.asarray(z, np.float64) / scale)
    i, j = np.floor(gx), np.floor(gz)
    tx, tz = gx - i, gz - j
    tx, tz = tx * tx * (3 - 2 * tx), tz * tz * (3 - 2 * tz)
    a, b = _hash(i, j, salt), _hash(i + 1, j, salt)
    c, d = _hash(i, j + 1, salt), _hash(i + 1, j + 1, salt)
    return ((a + (b - a) * tx) * (1 - tz) + (c + (d - c) * tx) * tz).astype(np.float32)


def reveal_radius(base: float, height_above_ground: float) -> float:
    """Seeing farther from high ground: up to twice the base radius."""
    return float(base) * min(2.0, 1.0 + max(0.0, float(height_above_ground)) / (2.0 * float(base)))


def parse_click(click, pointer) -> tuple[bool, float, float]:
    """Validate a (click, pointer) pair; returns (pressed, u, v)."""
    pressed = bool(int(np.asarray(click).reshape(-1)[0]))
    p = np.asarray(pointer, np.float64).reshape(-1)
    if p.size != 2:
        raise ValueError("pointer must have two entries (u, v)")
    if pressed and not np.isfinite(p).all():
        raise ValueError("pointer must be finite")
    return pressed, float(np.clip(np.nan_to_num(p[0]), 0, 1)), float(np.clip(np.nan_to_num(p[1]), 0, 1))


# --------------------------------------------------------------- fog of war
class FogOfWar:
    """Explored cells over an axis-aligned world rectangle (x0, z0, x1, z1)."""

    def __init__(self, bounds, cell: float, radius: float):
        x0, z0, x1, z1 = map(float, bounds)
        if not (x1 > x0 and z1 > z0 and cell > 0 and radius > 0):
            raise ValueError("fog bounds, cell and radius must be positive")
        self.bounds, self.cell, self.radius = (x0, z0, x1, z1), float(cell), float(radius)
        self.explored = np.zeros((max(1, math.ceil((x1 - x0) / cell)),
                                  max(1, math.ceil((z1 - z0) / cell))), bool)
        self._last: np.ndarray | None = None

    def reset_trail(self) -> None:
        """Forget the previous position, e.g. after leaving this layer."""
        self._last = None

    def reveal(self, x: float, z: float, radius: float | None = None) -> int:
        """Reveal a capsule from the previous position; returns new cells."""
        r = float(self.radius if radius is None else radius)
        p = np.array((x, z), np.float64)
        before = int(self.explored.sum())
        if self._last is None or np.linalg.norm(p - self._last) > 6 * r:
            points = p[None]
        else:
            n = int(np.ceil(np.linalg.norm(p - self._last) / (0.5 * r))) + 1
            points = np.linspace(self._last, p, n)
        for q in points:
            self._stamp(q, r)
        self._last = p
        return int(self.explored.sum()) - before

    def _stamp(self, q, r):
        x0, z0 = self.bounds[:2]
        c, (nx, nz) = self.cell, self.explored.shape
        i0, i1 = max(0, int((q[0] - r - x0) // c)), min(nx, int((q[0] + r - x0) // c) + 1)
        j0, j1 = max(0, int((q[1] - r - z0) // c)), min(nz, int((q[1] + r - z0) // c) + 1)
        if i0 >= i1 or j0 >= j1:
            return
        cx = x0 + (np.arange(i0, i1) + 0.5) * c - q[0]
        cz = z0 + (np.arange(j0, j1) + 0.5) * c - q[1]
        self.explored[i0:i1, j0:j1] |= cx[:, None] ** 2 + cz[None, :] ** 2 <= r * r

    def is_explored(self, x: float, z: float) -> bool:
        x0, z0 = self.bounds[:2]
        i, j = int((x - x0) // self.cell), int((z - z0) // self.cell)
        return 0 <= i < self.explored.shape[0] and 0 <= j < self.explored.shape[1] and bool(self.explored[i, j])

    @property
    def fraction(self) -> float:
        return float(self.explored.mean())


# --------------------------------------------------------------- layout
def _unit(width, height):
    return min(width, height)


def logo_box(width: int, height: int) -> tuple[int, int, int, int]:
    """Pixel box (col0, row0, col1, row1), half-open, of the map logo."""
    s = min(width, height, max(9, round(0.11 * _unit(width, height))))
    m = max(2, round(0.025 * _unit(width, height)))
    return width - m - s, m, width - m, m + s


def logo_center(width: int, height: int) -> tuple[float, float]:
    """Normalised (u, v) of the logo centre."""
    c0, r0, c1, r1 = logo_box(width, height)
    return (c0 + c1) / 2 / width, (r0 + r1) / 2 / height


def tab_boxes(width: int, height: int, count: int) -> list[tuple[int, int, int, int]]:
    """Layer tabs on the left edge, listed top to bottom."""
    t = max(7, round(0.08 * _unit(width, height)))
    m = max(2, round(0.025 * _unit(width, height)))
    return [(m, m + i * (t + m), m + t, m + i * (t + m) + t) for i in range(count)]


def _inside(box, u, v, width, height, pad=1.0) -> bool:
    c0, r0, c1, r1 = box
    px, py = u * width, v * height
    return c0 - pad <= px < c1 + pad and r0 - pad <= py < r1 + pad


# --------------------------------------------------------------- drawing
def _blend(region, mask, color, alpha=1.0):
    region[mask] = region[mask] * (1 - alpha) + np.asarray(color, np.float32) * alpha


def _rounded(u, v, corner, inset=0.0):
    half = 0.5 - inset
    du = np.maximum(np.abs(u - 0.5) - (half - corner), 0)
    dv = np.maximum(np.abs(v - 0.5) - (half - corner), 0)
    return (du * du + dv * dv <= corner * corner) & (np.abs(u - 0.5) < half) & (np.abs(v - 0.5) < half)


def draw_logo(img: np.ndarray, active: bool) -> None:
    """A folded paper map with a pin on a dark rounded badge (in place).

    The open-map state uses a warm badge and a thicker gold rim.
    """
    height, width = img.shape[:2]
    c0, r0, c1, r1 = logo_box(width, height)
    s = c1 - c0
    v, u = np.mgrid[0:s, 0:s].astype(np.float32)
    u, v = (u + 0.5) / s, (v + 0.5) / s
    rim_width = (2.0 if active else 1.0) / s
    badge = _rounded(u, v, 0.24)
    rim = badge & ~_rounded(u, v, 0.24 - rim_width, rim_width)
    region = img[r0:r1, c0:c1].astype(np.float32)
    _blend(region, badge, (58, 50, 34) if active else (34, 40, 56), 0.86)
    # Three folded panels, alternating light and shade, with a zig-zag edge.
    panel = np.clip(((u - 0.2) / 0.2).astype(int), 0, 2)
    lift = np.where(panel % 2 == 1, 0.05, 0.0)
    paper = (u >= 0.2) & (u < 0.8) & (v >= 0.24 + lift) & (v < 0.76 + lift)
    _blend(region, paper & (panel % 2 == 0), PARCHMENT)
    _blend(region, paper & (panel % 2 == 1), (198, 182, 140))
    route = paper & (np.abs((v - 0.62) + 0.35 * (u - 0.3)) < max(0.045, 0.6 / s))
    _blend(region, route, (120, 150, 96))
    pin = (u - 0.6) ** 2 + (v - 0.42) ** 2 <= max(0.09, 0.9 / s) ** 2
    _blend(region, pin, (214, 70, 58))
    _blend(region, rim, ACCENT if active else (210, 200, 170))
    img[r0:r1, c0:c1] = region.clip(0, 255).astype(img.dtype)


def _fill_polygon(img, points, color, alpha=1.0):
    """Fill a convex polygon given in (col, row) screen coordinates."""
    pts = np.asarray(points, np.float64)
    h, w = img.shape[:2]
    c0, r0 = np.floor(pts.min(0)).astype(int)
    c1, r1 = np.ceil(pts.max(0)).astype(int) + 1
    c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, w), min(r1, h)
    if c0 >= c1 or r0 >= r1:
        return
    y, x = np.mgrid[r0:r1, c0:c1].astype(np.float64) + 0.5
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
    sign = 1.0 if e1[0] * e2[1] - e1[1] * e2[0] >= 0 else -1.0
    inside = np.ones(x.shape, bool)
    for k in range(len(pts)):
        a, b = pts[k], pts[(k + 1) % len(pts)]
        inside &= sign * ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])) >= 0
    _blend(img[r0:r1, c0:c1], inside, color, alpha)


def _draw_agent(img, c, heading, size):
    d = np.asarray(heading if heading is not None else (0.0, -1.0), np.float64)[:2]
    d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-9 else np.array((0.0, -1.0))
    n = np.array((-d[1], d[0]))
    # Concave arrowhead drawn as two triangles, outline first.
    for k, color in ((1.45, OUTLINE), (1.0, AGENT)):
        tip, back = c + d * size * k, c - d * 0.25 * size * k
        wing_l = c - d * 0.6 * size * k + n * 0.6 * size * k
        wing_r = c - d * 0.6 * size * k - n * 0.6 * size * k
        _fill_polygon(img, [tip, wing_l, back], color)
        _fill_polygon(img, [tip, back, wing_r], color)


def _draw_diamond(img, c, r, color):
    _fill_polygon(img, [c + (0, -r - 1), c + (r + 1, 0), c + (0, r + 1), c + (-r - 1, 0)], OUTLINE)
    _fill_polygon(img, [c + (0, -r), c + (r, 0), c + (0, r), c + (-r, 0)], color)


def _nice_spacing(extent):
    raw = extent / 5.0
    p = 10 ** math.floor(math.log10(max(raw, 1e-9)))
    return min((k * p for k in (1, 2, 5, 10)), key=lambda s: abs(s - raw))


def _fog_coverage(fog: FogOfWar, wx, wz):
    """Soft explored coverage in [0, 1] sampled at world points."""
    e = np.pad(fog.explored.astype(np.float32), 1, mode="edge")
    n, m = fog.explored.shape
    blur = sum(e[1 + a:1 + a + n, 1 + b:1 + b + m] for a in (-1, 0, 1) for b in (-1, 0, 1)) / 9.0
    # Keep explored cells at least half covered and unexplored at most half.
    blur = np.where(fog.explored, np.maximum(blur, 0.5), np.minimum(blur, 0.5))
    if n == 1 or m == 1:
        blur = np.pad(blur, ((0, n == 1), (0, m == 1)), mode="edge")
    fx0, fz0 = fog.bounds[:2]
    gi = np.clip((wx - fx0) / fog.cell - 0.5, 0, blur.shape[0] - 1.001)
    gj = np.clip((wz - fz0) / fog.cell - 0.5, 0, blur.shape[1] - 1.001)
    i0, j0 = gi.astype(int), gj.astype(int)
    ti, tj = (gi - i0)[None, :], (gj - j0)[:, None]
    I, J = i0[None, :], j0[:, None]
    f = ((blur[I, J] * (1 - ti) + blur[I + 1, J] * ti) * (1 - tj)
         + (blur[I, J + 1] * (1 - ti) + blur[I + 1, J + 1] * ti) * tj)
    f = np.clip((f - 0.3) / 0.4, 0, 1)
    return f * f * (3 - 2 * f)


def render_map(width: int, height: int, base: np.ndarray, bounds, fog: FogOfWar, *,
               agent=None, heading=None, markers: Sequence = (), tabs: Sequence = ()) -> np.ndarray:
    """Full-screen map image (height, width, 3) uint8.

    ``base`` is a (nx, nz, 3) uint8 colour image over ``bounds``; ``agent``
    and markers are world (x, z), with ``heading`` a world (dx, dz). Markers are
    drawn only on explored ground. ``tabs`` lists (rgb, selected), top first.
    """
    out = np.empty((height, width, 3), np.float32)
    out[:] = SCREEN
    m = max(2, round(0.025 * _unit(width, height)))
    left = (tab_boxes(width, height, 1)[0][2] + m) if tabs else m
    right = logo_box(width, height)[0] - m
    top, bottom = m, height - m
    if right - left < 8 or bottom - top < 8:
        left, right, top, bottom = 0, width, 0, height
    x0, z0, x1, z1 = map(float, bounds)
    scale = min((right - left) / (x1 - x0), (bottom - top) / (z1 - z0))   # pixels per metre
    mw, mh = max(1, int((x1 - x0) * scale)), max(1, int((z1 - z0) * scale))
    ox, oy = left + (right - left - mw) // 2, top + (bottom - top - mh) // 2
    wx = x0 + (np.arange(mw) + 0.5) / mw * (x1 - x0)
    wz = z0 + (np.arange(mh) + 0.5) / mh * (z1 - z0)
    bi = np.clip(((wx - x0) / (x1 - x0) * base.shape[0]).astype(int), 0, base.shape[0] - 1)
    bj = np.clip(((wz - z0) / (z1 - z0) * base.shape[1]).astype(int), 0, base.shape[1] - 1)
    img = base[bi[None, :], bj[:, None]].astype(np.float32)               # (mh, mw, 3)

    f = _fog_coverage(fog, wx, wz)
    cloud = (0.65 * value_noise(wx[None, :], wz[:, None], fog.cell * 6, 11)
             + 0.35 * value_noise(wx[None, :], wz[:, None], fog.cell * 2, 23))
    shadow = SHADOW * (0.7 + 0.6 * cloud)[..., None]
    # Explored ground darkens toward the fog edge, then fades into shadow.
    img = img * ((0.6 + 0.4 * f) * f)[..., None] + shadow * (1 - f)[..., None]

    # A faint coordinate grid helps relate map distances to travel.
    spacing = _nice_spacing(max(x1 - x0, z1 - z0))
    for k in np.arange(math.ceil(x0 / spacing) * spacing, x1, spacing):
        col = int((k - x0) * scale)
        if 0 < col < mw:
            img[:, col] = img[:, col] * 0.82 + PARCHMENT * 0.18
    for k in np.arange(math.ceil(z0 / spacing) * spacing, z1, spacing):
        row = int((k - z0) * scale)
        if 0 < row < mh:
            img[row, :] = img[row, :] * 0.82 + PARCHMENT * 0.18
    out[oy:oy + mh, ox:ox + mw] = img
    fc0, fr0, fc1, fr1 = max(ox - 1, 0), max(oy - 1, 0), min(ox + mw, width - 1), min(oy + mh, height - 1)
    out[fr0, fc0:fc1 + 1] = out[fr1, fc0:fc1 + 1] = PARCHMENT * 0.7
    out[fr0:fr1 + 1, fc0] = out[fr0:fr1 + 1, fc1] = PARCHMENT * 0.7

    def to_screen(x, z):
        return np.array((ox + (x - x0) * scale, oy + (z - z0) * scale))

    r = max(1.5, 0.018 * _unit(width, height))
    for marker in markers:
        x, z = marker[:2]
        if fog.is_explored(x, z):
            _draw_diamond(out, to_screen(x, z), r, marker[2] if len(marker) > 2 else LANDMARK)
    if agent is not None:
        _draw_agent(out, to_screen(*agent), heading, max(3.0, 0.04 * _unit(width, height)))
    for (c0, r0, c1, r1), (color, selected) in zip(tab_boxes(width, height, len(tabs)), tabs):
        out[r0:r1, c0:c1] = np.asarray(color, np.float32) * (1.0 if selected else 0.45)
        edge = ACCENT if selected else PARCHMENT * 0.5
        out[r0, c0:c1] = out[r1 - 1, c0:c1] = edge
        out[r0:r1, c0] = out[r0:r1, c1 - 1] = edge
    image = out.clip(0, 255).astype(np.uint8)
    draw_logo(image, active=True)
    return image


# --------------------------------------------------------------- UI state
class WorldMapUI:
    """Open/closed state, click handling and screen composition."""

    def __init__(self, layers: Sequence[str] = ("world",), tab_colors: Sequence = ()):
        self.layers = tuple(layers)
        self.tab_colors = tuple(tab_colors)
        self.open = False
        self.view_layer = 0

    def tab_order(self) -> list[int]:
        """Layer indices top to bottom (highest layer first)."""
        return list(range(len(self.layers)))[::-1]

    def click(self, u: float, v: float, width: int, height: int, current_layer: int = 0) -> list[dict]:
        """Apply one click; returns UI events (empty when nothing was hit)."""
        if _inside(logo_box(width, height), u, v, width, height):
            self.open = not self.open
            if self.open:
                self.view_layer = int(current_layer)
            return [dict(type="map_opened" if self.open else "map_closed",
                         layer=self.layers[self.view_layer])]
        if self.open and len(self.layers) > 1:
            for box, index in zip(tab_boxes(width, height, len(self.layers)), self.tab_order()):
                if _inside(box, u, v, width, height, pad=0.5):
                    self.view_layer = index
                    return [dict(type="map_layer_selected", layer=self.layers[index])]
        return []

    def tabs(self) -> list:
        if len(self.layers) < 2:
            return []
        return [(self.tab_colors[i] if i < len(self.tab_colors) else PARCHMENT, i == self.view_layer)
                for i in self.tab_order()]

    def compose(self, frame: np.ndarray, render_open: Callable[[int, int], np.ndarray]) -> np.ndarray:
        """What the agent sees: the map when open, else the frame plus the logo."""
        height, width = frame.shape[:2]
        if self.open:
            return render_open(width, height)
        screen = np.array(frame, copy=True)
        draw_logo(screen, active=False)
        return screen


# --------------------------------------------------------------- voxel maps
_TABLES: dict = {}


def _voxel_tables():
    if not _TABLES:
        from ..core import blocks as B
        from ..render.palette import map_color_table
        visible = np.array([b is not None and b.id != B.AIR and b.semantic != B.SEM_PLANT
                            for b in B.BLOCKS], bool)
        _TABLES.update(visible=visible, color=map_color_table().astype(np.float32))
    return _TABLES["visible"], _TABLES["color"]


def voxel_surface(world, lo: int = 0, hi: int | None = None) -> np.ndarray:
    """Top-down (size_x, size_z, 3) colours of the uppermost open ground whose
    standing height lies in [lo, hi]; columns without such ground are VOID."""
    from ..render.palette import voxel_hash

    visible, table = _voxel_tables()
    sy = world.size_y
    hi = sy if hi is None else min(int(hi), sy)
    ylo = max(0, int(lo) - 1)
    yhi = max(ylo + 1, hi)
    vis = visible[world.vox[:, ylo:min(yhi + 1, sy), :]]
    if yhi + 1 > sy:
        vis = np.concatenate([vis, np.zeros_like(vis[:, :1])], axis=1)
    cand = vis[:, :-1] & ~vis[:, 1:]
    found = cand.any(1)
    top = ylo + cand.shape[1] - 1 - np.argmax(cand[:, ::-1], axis=1)
    top = np.where(found, top, ylo)
    ix, iz = np.indices(found.shape)
    color = table[world.vox[ix, top, iz]]
    color *= (1.0 + 0.06 * voxel_hash(ix, top, iz))[..., None]
    h = top.astype(np.float32)
    gx, gz = np.gradient(h) if min(h.shape) > 1 else (np.zeros_like(h), np.zeros_like(h))
    shade = np.clip(1.0 + 0.11 * (gx + gz), 0.66, 1.3)
    lift = 0.88 + 0.24 * (h - ylo) / max(yhi - ylo, 1)
    color *= (shade * lift)[..., None]
    void = VOID * (0.85 + 0.3 * value_noise(ix, iz, 9.0, 5))[..., None]
    return np.where(found[..., None], color, void).clip(0, 255).astype(np.uint8)
