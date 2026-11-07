"""Stable task-side architecture and shared structural collision volumes.

Buildings are authored at reset, independently of simulation RNG. The same solid
parts feed rendering and movement; terrain and task entities are never moved.
Roofs and facade trim are visual detail, not climbable/standable platforms.
"""
from dataclasses import dataclass
import hashlib

import numpy as np


KINDS = ("timber_house", "windmill", "pavilion", "market_stall", "ruin_gate", "watchtower")
# Conservative plan radii include eaves, sails, and ornaments.
RADII = dict(zip(KINDS, (5.2, 6.5, 5.2, 3.5, 5.4, 4.3)))


def _base_parts(kind):
    """(local centre, dimensions, material) for visible, colliding structure."""
    if kind == "timber_house":
        return [((0, 1.25, 0), (6, 5.5, 5), "plaster")]
    if kind == "windmill":
        return [((0, 2.8, 0), (4.6, 8.6, 4.6), "stone")]
    if kind == "watchtower":
        return [((0, 1.95, 0), (4.5, 6.9, 4.5), "stone")]
    if kind == "ruin_gate":
        return [((x, 1.25, 0), (1.5, 5.5, 1.8), "stone") for x in (-3.1, 3.1)] + [
            ((0, 4.35, 0), (7.7, .7, 1.8), "stone")]
    if kind in ("pavilion", "market_stall"):
        x, z, h = (2.6, 2.6, 3.8) if kind == "pavilion" else (1.8, 1.35, 2.9)
        parts = [((a*x, (h-1.4)/2, b*z), (.28, h+1.4, .28), "wood")
                 for a in (-1, 1) for b in (-1, 1)]
        parts += [((0, h-.15, b*z), (2*x+.35, .3, .3), "wood") for b in (-1, 1)]
        parts += [((a*x, h-.15, 0), (.3, .3, 2*z), "wood") for a in (-1, 1)]
        if kind == "market_stall":
            parts.append(((0, .45, 1.0), (3.4, 1.2, .6), "wood"))
        return parts
    raise ValueError(f"unknown architecture kind: {kind}")


def solid_parts(kind, foundation_depth=0.0):
    """Visible structural parts plus stone supports down to the original ground."""
    parts = _base_parts(kind)
    for center, size, _ in tuple(parts):
        bottom = center[1]-size[1]/2
        target = -foundation_depth-.25
        if bottom < -.5 and target < bottom:
            parts.append(((center[0], (target+bottom)/2, center[2]),
                          (size[0], bottom-target, size[2]), "stone"))
    return parts


@dataclass(frozen=True)
class Building:
    id: str
    kind: str
    pos: tuple[float, float, float]
    yaw: float
    appearance: int
    foundation_depth: float = 0.0

    @property
    def rotation(self):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))

    def metadata(self):
        return dict(id=self.id, kind=self.kind, position=list(self.pos), yaw=self.yaw,
                    appearance=self.appearance, foundation_depth=self.foundation_depth)


class Architecture:
    """Oriented solid boxes, independent of render visibility and camera angle."""
    def __init__(self, buildings=()):
        self.buildings = tuple(buildings)
        centers, halves, rotations = [], [], []
        for building in self.buildings:
            rotation = building.rotation
            for center, size, _ in solid_parts(building.kind, building.foundation_depth):
                centers.append(np.asarray(center) @ rotation.T + building.pos)
                halves.append(np.asarray(size)/2)
                rotations.append(rotation)
        self.centers = np.asarray(centers, dtype=float).reshape(-1, 3)
        self.halves = np.asarray(halves, dtype=float).reshape(-1, 3)
        self.rotations = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
        self.radii = np.linalg.norm(self.halves[:, [0, 2]], axis=1)

    def resolve(self, state, radius, height):
        if not len(self.centers):
            return
        delta = self.centers - state.pos
        nearby = ((delta[:, 0]**2 + delta[:, 2]**2 < (self.radii+radius)**2)
                  & (state.pos[1] < self.centers[:, 1]+self.halves[:, 1])
                  & (state.pos[1]+height > self.centers[:, 1]-self.halves[:, 1]))
        # Revisit adjacent parts after a push at a joint/corner.
        for _ in range(3):
            for i in np.flatnonzero(nearby):
                local = (state.pos-self.centers[i]) @ self.rotations[i]
                half = self.halves[i, [0, 2]]
                point = local[[0, 2]]
                nearest = np.clip(point, -half, half)
                offset = point-nearest
                distance = np.linalg.norm(offset)
                if distance >= radius:
                    continue
                if distance > 1e-8:
                    normal = offset/distance
                    push = radius-distance
                else:
                    margins = half-np.abs(point)
                    axis = int(np.argmin(margins))
                    normal = np.zeros(2)
                    normal[axis] = 1 if point[axis] >= 0 else -1
                    push = margins[axis]+radius
                world_normal = np.array((normal[0], 0, normal[1])) @ self.rotations[i].T
                state.pos += (world_normal*(push+1e-5)).astype(state.pos.dtype)
                into = float(state.vel @ world_normal)
                if into < 0:
                    state.vel -= (world_normal*into).astype(state.vel.dtype)

    def blocked(self, points, margin=.35):
        points = np.asarray(points)
        blocked = np.zeros(points.shape[:-1], bool)
        for center, half, rotation in zip(self.centers, self.halves, self.rotations):
            local = (points-center) @ rotation
            blocked |= np.all(np.abs(local) < half+margin, axis=-1)
        return blocked


def _distance_to_routes(point, segments):
    if not len(segments):
        return np.inf
    a, b = segments[:, 0], segments[:, 1]
    direction = b-a
    fraction = np.clip(np.sum((point-a)*direction, axis=1)
                       / np.maximum(np.sum(direction**2, axis=1), 1e-8), 0, 1)
    return float(np.min(np.linalg.norm(point-a-fraction[:, None]*direction, axis=1)))


def populate_architecture(world, spawn=None):
    """Fit architecture around task sites, leaving routes and mechanisms clear."""
    from .world import EntityKind
    course = getattr(world, "course", None)
    entities = list(world.entities.values())
    points = np.array([e.home[[0, 2]] for e in entities]).reshape(-1, 2)
    segments, anchors = [], []
    if course is not None:
        previous = None
        for milestone in course.milestones:
            for op in milestone.operations:
                current = [world.get(eid).home[[0, 2]] for eid in op.entities]
                # Protect every possible within-operation order (relay shuffle)
                # and every transition endpoint, not just one controller trace.
                for a in current:
                    for b in current:
                        segments.append((a, b))
                    if previous is not None:
                        for b in previous:
                            segments.append((a, b))
                if current:
                    previous = current
                family = {"lesson": "timber_house", "exchange": "market_stall",
                          "relay": "pavilion", "crossing": "windmill",
                          "sentinel": "ruin_gate", "fight": "watchtower",
                          "dive": "ruin_gate", "lift": "watchtower"}.get(op.kind)
                family = family or KINDS[len(anchors) % len(KINDS)]
                anchors.append((op.id, np.asarray(op.state["site"]), family))
    else:
        for e in entities:
            if e.kind == EntityKind.NPC:
                for j, kind in enumerate(("timber_house", "market_stall", "windmill")):
                    anchors.append((f"{e.id}-{j}", e.home[[0, 2]], kind))
            elif e.kind in (EntityKind.PILLAR, EntityKind.ALTAR, EntityKind.WAYPOINT):
                kind = "pavilion" if e.kind == EntityKind.WAYPOINT else (
                    "ruin_gate" if e.kind == EntityKind.PILLAR else "watchtower")
                anchors.append((e.id, e.home[[0, 2]], kind))
        # Keep direct approaches between objectives clear in the open-world map.
        segments = [(a, b) for i, a in enumerate(points) for b in points[i+1:]]
    if spawn is not None and len(points):
        segments.extend((np.asarray(spawn)[[0, 2]], p) for p in points[:1])
    routes = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    placed = []
    for identity, anchor, kind in anchors:
        radius = RADII[kind]
        digest = hashlib.blake2s(f"{world.cfg.seed}:{identity}:{kind}".encode(), digest_size=4).digest()
        seed = int.from_bytes(digest, "little")
        # Off-axis options keep buildings out of the course's serpentine streets.
        offsets = [(0, -18), (0, 18), (-18, -18), (18, 18), (18, -18), (-18, 18),
                   (-25, 0), (25, 0), (0, -28), (0, 28)]
        if course is None:
            # Natural islands need a wider search than the authored plateau.
            # Retain all clearance/terrain checks; never clear trees or flatten.
            phase = (seed % 24)*np.pi/12
            offsets += [(distance*np.cos(angle+phase), distance*np.sin(angle+phase))
                        for distance in (24, 32, 40, 52, 64, 80)
                        for angle in np.arange(24)*np.pi/12]
        for dx, dz in offsets[seed % 4:]+offsets[:seed % 4]:
            xz = anchor+(dx, dz)
            if np.any(np.abs(xz)+radius >= world.terrain.cfg.size/2-6):
                continue
            if len(points) and np.min(np.linalg.norm(points-xz, axis=1)) < radius+6:
                continue
            if _distance_to_routes(xz, routes) < radius+3:
                continue
            if any(np.linalg.norm(np.asarray(b.pos)[[0, 2]]-xz) < radius+RADII[b.kind]+2 for b in placed):
                continue
            props = world.props
            indices = props.query_radius(*xz, radius+3)
            if any(np.linalg.norm(props.pos[i, [0, 2]]-xz) < radius+props.radius[i]
                   for i in indices if props.radius[i] > 0):
                continue
            # Sample the footprint, not only the centre; no earthwork or flattening.
            offsets_ground = np.array([(a, b) for a in np.linspace(-radius, radius, 5) for b in np.linspace(-radius, radius, 5)])
            samples = xz+offsets_ground
            heights = world.terrain.height_at(samples[:, 0], samples[:, 1])
            if heights.min() < world.terrain.cfg.sea_level+.5 or np.ptp(heights) > (1.25 if course is not None else 3.2):
                continue
            y = float(heights.max())
            toward = anchor-xz
            yaw = float(np.arctan2(-toward[0], -toward[1]))
            placed.append(Building(f"architecture:{identity}", kind,
                                   (float(xz[0]), y, float(xz[1])), yaw, seed,
                                   float(np.ceil(np.ptp(heights)*10)/10)))
            break
    world.architecture = Architecture(placed)
    return world.architecture
