"""Heightfield route planning for the full-scale Teyvat world.

Routes are planned on the ``under`` traversal model from ``fullscale_reach``:
gentle ground, shallows, and open water, with buildings and closed gates
removed. That is the same graph the phase-1 proofs use to show a site is
reachable, so a planned route only crosses ground the proofs already accept.
Solid props (tree trunks, rocks) are also removed, with a margin for the
character. A clearance cost keeps routes away from cliffs and obstacles.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .fullscale_reach import build_graph, under_model

SWIM_COST = 2.0             # swimming is about half as fast as running
CLEARANCE_CELLS = 3.0       # prefer staying this many cells from obstacles
CLEARANCE_COST = 1.5
PROP_MARGIN = 0.9           # metres added to prop radius plus the character radius
SEGMENT_MAX = 36.0          # longest straight segment after smoothing (m)


class NavGrid:
    """Patch graph for one set of open facts."""

    def __init__(self, world, facts=(), char_radius: float = 0.4):
        t = world.terrain
        self.cell = float(t._cell)
        self.half = float(t._half)
        self.n = t.cfg.resolution - 1
        graph = build_graph(world, set(facts), under_model(self.cell))
        open_ = graph.rest & ~self._prop_blocked(world, char_radius)
        self.open = open_
        self.water = graph.water & open_
        clearance = ndimage.distance_transform_edt(open_)
        penalty = CLEARANCE_COST * np.clip(CLEARANCE_CELLS - clearance, 0.0, None) / CLEARANCE_CELLS
        self.cost = np.where(self.water, SWIM_COST, 1.0) + penalty
        self.label, _ = ndimage.label(open_, structure=np.ones((3, 3), bool))
        self._graph = self._build_sparse()

    # ----------------------------------------------------------- geometry
    def centre(self, i, j):
        return (-self.half + (i + 0.5) * self.cell, -self.half + (j + 0.5) * self.cell)

    def index(self, x, z):
        i = int(np.clip(np.floor((x + self.half) / self.cell), 0, self.n - 1))
        j = int(np.clip(np.floor((z + self.half) / self.cell), 0, self.n - 1))
        return i, j

    def _prop_blocked(self, world, char_radius):
        blocked = np.zeros((self.n, self.n), bool)
        props = world.props
        solid = np.flatnonzero(props.radius > 0.0)
        for k in solid:
            x, z = float(props.pos[k, 0]), float(props.pos[k, 2])
            reach = float(props.radius[k]) + char_radius + PROP_MARGIN
            i0, j0 = self.index(x - reach, z - reach)
            i1, j1 = self.index(x + reach, z + reach)
            ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
            cx, cz = self.centre(ii, jj)
            blocked[ii, jj] |= np.hypot(cx - x, cz - z) < reach
        return blocked

    def _build_sparse(self):
        n = self.n
        ids = np.arange(n * n).reshape(n, n)
        is_open = np.pad(self.open, 1)
        cost = np.pad(self.cost, 1, constant_values=np.inf)
        shifted = lambda a, di, dj: a[1 + di:n + 1 + di, 1 + dj:n + 1 + dj]
        rows, cols, weights = [], [], []
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
            ok = self.open & shifted(is_open, di, dj)
            if di and dj:           # no corner cutting on diagonals
                ok &= shifted(is_open, di, 0) & shifted(is_open, 0, dj)
            w = np.hypot(di, dj) * 0.5 * (self.cost + shifted(cost, di, dj))
            rows.append(ids[ok]); cols.append(ids[ok] + di * n + dj); weights.append(w[ok])
        rows, cols, weights = (np.concatenate(v) for v in (rows, cols, weights))
        m = coo_matrix((np.r_[weights, weights], (np.r_[rows, cols], np.r_[cols, rows])), shape=(n * n, n * n))
        return m.tocsr()

    # ------------------------------------------------------------ queries
    def snap(self, x, z, component=None, radius_cells=12):
        """Nearest open patch to (x, z), optionally inside one connected component."""
        i, j = self.index(x, z)
        best, best_d = None, np.inf
        for r in range(radius_cells + 1):
            i0, i1 = max(0, i - r), min(self.n, i + r + 1)
            j0, j1 = max(0, j - r), min(self.n, j + r + 1)
            window = self.open[i0:i1, j0:j1]
            if component is not None:
                window = window & (self.label[i0:i1, j0:j1] == component)
            if window.any():
                ii, jj = np.nonzero(window)
                ii, jj = ii + i0, jj + j0
                cx, cz = self.centre(ii, jj)
                d = np.hypot(cx - x, cz - z)
                k = int(np.argmin(d))
                if d[k] < best_d:
                    best, best_d = (int(ii[k]), int(jj[k])), float(d[k])
                if r * self.cell > best_d + self.cell:
                    break
        return best

    def plan(self, start_xz, goal_xz):
        """Smoothed waypoints from start to goal, or None if unconnected."""
        start = self.snap(*start_xz)
        if start is None:
            return None
        goal = self.snap(*goal_xz, component=self.label[start])
        if goal is None:
            return None
        n = self.n
        s, g = start[0] * n + start[1], goal[0] * n + goal[1]
        dist, pred = dijkstra(self._graph, indices=s, return_predecessors=True)
        if not np.isfinite(dist[g]):
            return None
        chain = [g]
        while chain[-1] != s:
            chain.append(int(pred[chain[-1]]))
        cells = [divmod(k, n) for k in reversed(chain)]
        points = [np.array(self.centre(i, j)) for i, j in cells]
        points[-1] = np.asarray(goal_xz, float) if self._segment_clear(points[-1], goal_xz) else points[-1]
        return self.smooth([np.asarray(start_xz, float)] + points)

    def _segment_clear(self, a, b, step=0.8):
        a, b = np.asarray(a, float), np.asarray(b, float)
        length = float(np.linalg.norm(b - a))
        for t in np.linspace(0.0, 1.0, max(2, int(length / step) + 1)):
            i, j = self.index(*(a + (b - a) * t))
            if not self.open[i, j] or self.water[i, j] != self.water[self.index(*a)]:
                return False
        return True

    def smooth(self, points):
        """String-pull the cell path, keeping segments on open ground of one medium."""
        out, k = [points[0]], 0
        while k < len(points) - 1:
            far = k + 1
            for m in range(len(points) - 1, k + 1, -1):
                if (np.linalg.norm(points[m] - points[k]) <= SEGMENT_MAX
                        and self._segment_clear(points[k], points[m])):
                    far = m
                    break
            out.append(points[far])
            k = far
        return out

    def connected(self, a_xz, b_xz):
        a, b = self.snap(*a_xz), self.snap(*b_xz)
        return a is not None and b is not None and self.label[a] == self.label[b]
