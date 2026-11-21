"""Apple trees: ripe apples fall now and then, and striking a trunk shakes
some loose. Fallen apples are ordinary pick-ups (``interact``); eating one
restores half a heart (2 health), as defined for the ``apple`` item.

Timing is a deterministic per-tree schedule derived from the world seed and
tree position; it never draws from the environment's random generator.
"""
from __future__ import annotations

import numpy as np

from ..core import blocks as B
from ..world.organic import unit

FALL_PERIOD = (60.0, 140.0)    # seconds between natural falls, per tree
REGROW = 240.0                 # seconds per apple ripening back on the tree
MAX_ON_GROUND = 3              # natural falls pause while this many lie below
STRIKE_REACH = 2.4


class Orchard:
    def __init__(self, world, trees, seed):
        self.world, self.seed = world, seed
        self.trees = trees

    @classmethod
    def from_world(cls, world, seed, share=0.35, region=None, band=None, include=None):
        """Mark a share of the natural trees (trunk on soil, leafy crown) as
        apple trees, chosen by hashing their position. ``band`` (a boolean per
        height) restricts the search to some layers of a large world; trunks at
        ``include`` (``(x, z)`` cells) are always apple trees."""
        vox = world.vox
        ylo, yhi = (0, vox.shape[1]) if band is None else (int(np.argmax(band)), int(len(band) - np.argmax(band[::-1])))
        ylo = max(1, ylo)
        log = vox[:, ylo:yhi, :] == B.LOG
        base = log & np.isin(vox[:, ylo - 1:yhi - 1, :], (B.GRASS, B.DIRT))
        trees = []
        for x, y, z in np.argwhere(base):
            y += ylo
            if region is not None and not region[x, z]:
                continue
            if (include is None or (int(x), int(z)) not in include) and unit("apple-tree", seed, int(x), int(z)) >= share:
                continue
            top = y
            while top + 1 < vox.shape[1] and vox[x, top + 1, z] == B.LOG:
                top += 1
            crown = vox[max(0, x - 2):x + 3, top:min(vox.shape[1], top + 4), max(0, z - 2):z + 3]
            if (crown == B.LEAVES).sum() < 6:
                continue
            identity = ("apple-tree", seed, int(x), int(z))
            trees.append(dict(id=len(trees), trunk=(float(x) + .5, float(y), float(z) + .5),
                              crown=float(top) + 1.0, capacity=int(4 + 3 * unit(*identity, "cap")),
                              apples=int(4 + 3 * unit(*identity, "cap")), drops=0,
                              fall=FALL_PERIOD[0] + (FALL_PERIOD[1] - FALL_PERIOD[0]) * unit(*identity, "t", 0),
                              ripen=REGROW, struck=0.0))
        return cls(world, trees, seed)

    def state(self):
        return [dict(id=t["id"], apples=t["apples"], drops=t["drops"], fall=round(t["fall"], 6),
                     ripen=round(t["ripen"], 6)) for t in self.trees]

    def _drop(self, tree, plants, count, cause):
        events = []
        x, y, z = tree["trunk"]
        for _ in range(min(count, tree["apples"])):
            k = tree["drops"]
            identity = ("apple-fall", self.seed, tree["id"], k)
            angle = 2 * np.pi * unit(*identity, "a")
            radius = 0.8 + 1.4 * unit(*identity, "r")
            px, pz = x + radius * np.cos(angle), z + radius * np.sin(angle)
            ix, iz = int(np.floor(px)), int(np.floor(pz))
            if not (0 <= ix < self.world.size_x and 0 <= iz < self.world.size_z):
                continue
            column = B.SOLID[self.world.vox[ix, :, iz]]
            below = np.flatnonzero(column[:int(tree["crown"])])
            ground = float(below[-1] + 1) if len(below) else y
            plants.append(dict(pos=(round(px, 3), ground, round(pz, 3)), item="apple", count=1,
                               source="apple_tree", tree=tree["id"]))
            tree["apples"] -= 1
            tree["drops"] += 1
            events.append(dict(type="apple_fell", tree=tree["id"], cause=cause,
                               pos=(round(px, 3), ground, round(pz, 3))))
        return events

    def step(self, dt, plants):
        events = []
        for tree in self.trees:
            tree["struck"] = max(0.0, tree["struck"] - dt)
            if tree["apples"] < tree["capacity"]:
                tree["ripen"] -= dt
                if tree["ripen"] <= 0:
                    tree["apples"] += 1
                    tree["ripen"] = REGROW
            tree["fall"] -= dt
            if tree["fall"] <= 0:
                waiting = sum(1 for p in plants if p.get("tree") == tree["id"] and not p.get("used"))
                if tree["apples"] and waiting < MAX_ON_GROUND:
                    events.extend(self._drop(tree, plants, 1, "ripe"))
                identity = ("apple-tree", self.seed, tree["id"], "t", tree["drops"])
                tree["fall"] = FALL_PERIOD[0] + (FALL_PERIOD[1] - FALL_PERIOD[0]) * unit(*identity)
        return events

    def strike(self, agent, plants):
        """Shake up to two apples loose from a trunk the agent faces."""
        best = None
        for tree in self.trees:
            x, y, z = tree["trunk"]
            delta = np.array((x - agent.pos[0], 0.0, z - agent.pos[2]))
            distance = float(np.linalg.norm(delta))
            if distance > STRIKE_REACH or abs(agent.pos[1] - y) > 2.5:
                continue
            if distance > 0.3 and np.dot(delta, agent.forward_flat()) < 0.5 * distance:
                continue
            if best is None or distance < best[0]:
                best = (distance, tree)
        if best is None:
            return []
        tree = best[1]
        if tree["struck"] > 0:
            return [dict(type="tree_struck", tree=tree["id"], apples=0)]
        tree["struck"] = 1.0
        events = self._drop(tree, plants, 2, "struck")
        return [dict(type="tree_struck", tree=tree["id"], apples=len(events))] + events
