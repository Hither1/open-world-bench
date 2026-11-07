"""Axis-aligned box collision against the voxel grid."""

from __future__ import annotations

import numpy as np

from ..core import blocks as B

_EPS = 1e-3


def box_collides(world, center: np.ndarray, half: np.ndarray) -> bool:
    lo = np.floor(center - half + _EPS).astype(np.int64)
    hi = np.floor(center + half - _EPS).astype(np.int64)
    for x in range(lo[0], hi[0] + 1):
        for y in range(lo[1], hi[1] + 1):
            for z in range(lo[2], hi[2] + 1):
                if B.SOLID[world.get(x, y, z)]:
                    return True
    return False


def box_block_ids(world, center: np.ndarray, half: np.ndarray) -> set[int]:
    """Every distinct block id overlapping the box (including non-solid)."""
    lo = np.floor(center - half + _EPS).astype(np.int64)
    hi = np.floor(center + half - _EPS).astype(np.int64)
    out: set[int] = set()
    for x in range(lo[0], hi[0] + 1):
        for y in range(lo[1], hi[1] + 1):
            for z in range(lo[2], hi[2] + 1):
                out.add(world.get(x, y, z))
    return out


def axis_move(world, center: np.ndarray, half: np.ndarray, axis: int,
              d: float) -> tuple[np.ndarray, bool]:
    """Move one axis, snapping to the blocking voxel face on contact."""
    if abs(d) < 1e-9:
        return center, False
    cand = center.copy()
    cand[axis] += d
    if not box_collides(world, cand, half):
        return cand, False
    if d > 0:
        cand[axis] = np.floor(cand[axis] + half[axis]) - half[axis] - _EPS
    else:
        cand[axis] = np.ceil(cand[axis] - half[axis]) + half[axis] + _EPS
    if box_collides(world, cand, half):
        cand[axis] = center[axis]
    return cand, True


def move_and_collide(world, center: np.ndarray, half: np.ndarray,
                     delta: np.ndarray, step_height: float = 0.0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Sweep a box through the world.

    Horizontal axes are resolved first so that walking into a low ledge can
    auto-step up onto it (``step_height``), then the vertical axis.  Returns
    the new centre and a per-axis "blocked" flag array.
    """
    pos = np.asarray(center, dtype=np.float64).copy()
    delta = np.asarray(delta, dtype=np.float64)
    hit = np.zeros(3, dtype=bool)

    n_sub = int(max(1, np.ceil(np.abs(delta).max() / 0.45)))
    d = delta / n_sub
    for _ in range(n_sub):
        for axis in (0, 2):
            if abs(d[axis]) < 1e-9:
                continue
            moved, blocked = axis_move(world, pos, half, axis, d[axis])
            if blocked and step_height > 0:
                lifted = pos.copy()
                lifted[1] += step_height
                if not box_collides(world, lifted, half):
                    cand, b2 = axis_move(world, lifted, half, axis, d[axis])
                    if not b2:
                        # settle back down onto the ledge
                        drop = cand.copy()
                        drop[1] -= step_height
                        if box_collides(world, drop, half):
                            pos = cand
                            continue
                        lowered, _ = axis_move(world, cand, half, 1, -step_height)
                        pos = lowered
                        continue
            pos = moved
            hit[axis] |= blocked
        if abs(d[1]) > 1e-9:
            pos, blocked = axis_move(world, pos, half, 1, d[1])
            hit[1] |= blocked
    return pos.astype(np.float64), hit


def on_ground(world, center: np.ndarray, half: np.ndarray,
              probe: float = 0.08) -> bool:
    c = center.copy()
    c[1] -= probe
    return box_collides(world, c, half)


def wall_contact(world, center: np.ndarray, half: np.ndarray,
                 reach: float = 0.35) -> tuple[bool, np.ndarray, int]:
    """Is there a climbable wall within arm's reach of the box?

    Returns ``(found, outward_normal, block_id)`` where the normal points from
    the wall back toward the agent.  Vines count as climbable even where the
    surface behind them would not be.
    """
    # Vines occupy the air voxel against a wall, so the agent stands *inside*
    # them like a ladder -- check our own column before looking outward.
    cx, cz = int(np.floor(center[0])), int(np.floor(center[2]))
    for y in range(int(np.floor(center[1] - half[1])),
                   int(np.floor(center[1] + half[1])) + 1):
        if world.get(cx, y, cz) == B.VINE:
            return True, np.zeros(3), int(B.VINE)

    fallback = None
    wall = None
    for n in ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)):
        nv = np.array(n, dtype=np.float64)
        probe = center + nv * (half[0] + reach)
        px = int(np.floor(probe[0]))
        pz = int(np.floor(probe[2]))
        y_lo = int(np.floor(center[1] - half[1] * 0.6))
        y_hi = int(np.floor(center[1] + half[1] * 0.6))
        for y in range(y_lo, y_hi + 1):
            blk = world.get(px, y, pz)
            if blk == B.VINE:
                return True, -nv, int(blk)        # vines always win
            if B.SOLID[blk]:
                if B.CLIMBABLE[blk]:
                    if wall is None:
                        wall = (-nv, int(blk))
                elif fallback is None:
                    fallback = (-nv, int(blk))
    if wall is not None:
        return True, wall[0], wall[1]
    if fallback is not None:
        return False, fallback[0], fallback[1]
    return False, np.zeros(3), 0
