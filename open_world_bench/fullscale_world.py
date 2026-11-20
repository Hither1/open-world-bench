"""Full-scale Teyvat Task A world: authored regional terrain, structures and gates.

This module only constructs the level: no NPCs, items or milestones. Heights are
authored directly so that every region boundary the task depends on is a cliff
taller than one climb can scale under :mod:`open_world_bench.fullscale_physics`.
``fullscale_reach`` checks this on the actual heightfield.

Region layout (x east, z south, metres; north is -z):

* lowland: Windrest village, Reedmarsh and the cache islet, the eastern raider
  hills, the lower ferry and the western-gully ruin, bounded on the north by a
  river at the foot of the escarpment;
* plateau: upper ferry landing, highland outpost and the western glide edge,
  58 m up behind the escarpment;
* ridge: both signal relays and the summit approach, 104 m up behind a sheer
  wall that one ravine cuts through. A sealed gate closes the ravine mouth;
* summit pillar: the beacon platform, 156 m up, reached by the service lift.

Each seed shifts the site anchors by up to 18 m using a layout stream derived
from the seed. Neither the layout nor any visual detail draws from the
simulation RNG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import zlib

import numpy as np

from .architecture import Architecture, Building, RADII
from .campaign_env import GateCollisions
from .scatter import PropField, scatter_props
from .terrain import Terrain, TerrainConfig

SUITE_VERSION = "teyvat-fullscale-v1"
GEOGRAPHY_SCALE = 2.0
SIZE = 640.0 * GEOGRAPHY_SCALE
RESOLUTION = 513
MAX_HEIGHT = 160.0
SEA_FLOOR = -12.0

LOWLAND = 9.0
RIVER_BED = -2.5
MARSH = 1.2
CHANNEL_BED = -1.6
PLATEAU = 58.0
RIDGE = 104.0
DOME_RISE = 6.0
PILLAR_TOP = 156.0
PILLAR_RADIUS = 12.0
RAVINE_HALF_WIDTH = 6.0
RAVINE_LENGTH = 130.0
LAYOUT_JITTER = 18.0

# Facts that open each region gate, in task order.
FERRY_OPEN = "ferry_open"          # A06: the cable ferry lift runs
CORRIDOR_OPEN = "corridor_open"    # A07: the ravine gate opens
CAP_INSTALLED = "cap_installed"    # A11: the summit service lift runs
FACT_ORDER = (FERRY_OPEN, CORRIDOR_OPEN, CAP_INSTALLED)

# Design coordinates before seed jitter. Groups move together.
_GROUPS = {
    "village": {"spawn": (-400, 430), "keeper": (-385, 415), "yard_1": (-350, 395),
                "yard_2": (-275, 330), "yard_3": (-450, 300)},
    "marsh": {"junction_1": (-120, 390), "junction_2": (70, 450), "junction_3": (240, 370)},
    "cache": {"cache": (310, 470)},
    "camp": {"camp": (390, 180)},
    "outpost": {"outpost": (-90, -60)},
    "relay_1": {"relay_1": (130, -235)},
    "relay_2": {"relay_2": (-210, -310)},
    "ruin": {"ruin": (-470, 205)},
}
_GROUP_JITTER = {"cache": 10.0}
_FERRY_X, _CORRIDOR_X, _WEST_EDGE_X = 60.0, -20.0, -340.0
_SUMMIT = (0.0, -455.0)


def escarpment_z(x):
    """z of the escarpment face; the plateau lies north (smaller z)."""
    x = np.asarray(x, dtype=float)
    return 100.0 + 8.0 * np.sin(x / 95.0) + 4.0 * np.sin(x / 37.0 + 1.0)


def ridge_z(x):
    """z of the ridge wall; the ridge lies north (smaller z)."""
    x = np.asarray(x, dtype=float)
    return -150.0 + 10.0 * np.sin(x / 85.0 + 0.7)


def _step(signed, width):
    """Smooth 0..1 transition centred on signed == 0 over ``width`` metres."""
    t = np.clip(np.asarray(signed, dtype=float) / width + 0.5, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _stable_int(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


@dataclass(frozen=True)
class Layout:
    seed: int
    anchors: dict
    corridor_x: float
    summit: tuple

    def point(self, name) -> np.ndarray:
        return np.asarray(self.anchors[name], dtype=float)


def make_layout(seed: int) -> Layout:
    rng = np.random.default_rng([int(seed), 0x7E1A, 0x51])
    anchors = {}
    for group, points in _GROUPS.items():
        scale = _GROUP_JITTER.get(group, LAYOUT_JITTER)
        offset = rng.uniform(-scale, scale, 2)
        for name, xz in points.items():
            anchors[name] = (float(xz[0] + offset[0]), float(xz[1] + offset[1]))
    ferry_x = _FERRY_X + float(rng.uniform(-LAYOUT_JITTER, LAYOUT_JITTER))
    anchors["ferry_low"] = (ferry_x, float(escarpment_z(ferry_x)) + 48.0)
    anchors["ferry_high"] = (ferry_x - 2.0, float(escarpment_z(ferry_x - 2.0)) - 14.0)
    corridor_x = _CORRIDOR_X + float(rng.uniform(-LAYOUT_JITTER, LAYOUT_JITTER))
    anchors["corridor_mouth"] = (corridor_x, float(ridge_z(corridor_x)) + 10.0)
    anchors["ravine_top"] = (corridor_x, float(ridge_z(corridor_x)) - RAVINE_LENGTH - 6.0)
    west_x = _WEST_EDGE_X + float(rng.uniform(-LAYOUT_JITTER, LAYOUT_JITTER))
    anchors["west_edge"] = (west_x, float(escarpment_z(west_x)) - 6.0)
    summit = tuple(float(v) for v in np.asarray(_SUMMIT) + rng.uniform(-10.0, 10.0, 2))
    anchors["summit_top"] = summit
    anchors["summit_base"] = (summit[0], summit[1] + 28.0)
    return Layout(seed=int(seed), anchors=anchors, corridor_x=corridor_x, summit=summit)


@dataclass
class Lift:
    id: str
    bottom: tuple
    top: tuple
    opens: str


@dataclass
class FactState:
    """Facts that open region gates. ``gates`` feeds GateCollisions."""
    gates: list = field(default_factory=list)
    facts: set = field(default_factory=set)


@dataclass
class FullScaleWorld:
    seed: int
    layout: Layout
    terrain: Terrain
    props: PropField
    buildings: list
    lifts: list
    state: FactState
    architecture: GateCollisions

    @property
    def gates(self):
        return self.state.gates

    def set_facts(self, facts) -> None:
        self.state.facts = set(facts)

    def anchor(self, name) -> np.ndarray:
        x, z = self.layout.anchors[name]
        return np.array([x, float(self.terrain.height_at(x, z)), z])

    def grid(self):
        t = self.terrain
        axis = np.linspace(-t._half, t._half, t.cfg.resolution)
        return np.meshgrid(axis, axis, indexing="ij")

    def regions(self) -> dict:
        """Boolean grids of each region's walkable top, inset 4 m from its walls.

        Sea-cliff faces and the foot of each cliff are excluded by height, so a
        swimmer at the base of a sea cliff is not counted as being on top.
        """
        x, z = self.grid()
        h = self.terrain.height
        d_e = escarpment_z(x) - z               # > 0 north of the escarpment
        d_r = ridge_z(x) - z                    # > 0 north of the ridge wall
        cx = self.layout.corridor_x
        mouth = float(ridge_z(cx))
        ravine = ((np.abs(x - cx) < RAVINE_HALF_WIDTH - 1.0) & (z < mouth - 8.0)
                  & (z > mouth - RAVINE_LENGTH - 20.0) & (h > PLATEAU - 2.0))
        sx, sz = self.layout.summit
        pillar = np.hypot(x - sx, z - sz) < PILLAR_RADIUS - 3.0
        plateau = (d_e > 4.0) & (d_r < -4.0) & (h > PLATEAU - 6.0)
        ridge = (((d_r > 4.0) & ~pillar & (h > RIDGE - 6.0)) | ravine)
        lowland = (d_e < -4.0) & (h > 0.0)
        return dict(lowland=lowland, plateau=plateau, ridge=ridge, summit_top=pillar)

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.terrain.height).tobytes())
        h.update(np.ascontiguousarray(self.props.pos).tobytes())
        for b in self.buildings:
            h.update(repr(b.metadata()).encode())
        for g in self.gates:
            h.update(repr(sorted(g.items())).encode())
        return h.hexdigest()


def _coast_distance(x, z, seed):
    """Signed distance inside a rounded-square coast (positive inland)."""
    half, radius = 560.0, 140.0
    qx, qz = np.abs(x) - (half - radius), np.abs(z) - (half - radius)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, qz), 0)
    wobble = 7.0 * np.sin(x / 61.0 + seed % 7) * np.cos(z / 53.0) + 4.0 * np.sin((x - z) / 29.0)
    return radius - outside + wobble


def author_terrain(layout: Layout) -> Terrain:
    terrain = Terrain(TerrainConfig(seed=layout.seed, size=SIZE, resolution=RESOLUTION,
                                    max_height=MAX_HEIGHT, terrace_step=0.0,
                                    snow_line=125.0, tree_line=85.0))
    axis = np.linspace(-terrain._half, terrain._half, terrain.cfg.resolution)
    x, z = np.meshgrid(axis, axis, indexing="ij")
    p = layout.point

    # Lowland: gently rolling ground with the raider hills in the east.
    h = LOWLAND + 1.2 * np.sin(x / 53.0 + 1.0) * np.cos(z / 47.0) + 0.8 * np.sin((x + z) / 31.0)
    camp = p("camp") + (0.0, 25.0)
    h += 7.0 * np.exp(-((x - camp[0]) ** 2 + (z - camp[1]) ** 2) / (2 * 45.0 ** 2))
    moisture = np.full(h.shape, 0.5)

    # Reedmarsh: low wet ground cut by three swimmable channels.
    junctions = np.array([p("junction_1"), p("junction_2"), p("junction_3")])
    cx, cz = (junctions[:, 0].min() + p("cache")[0]) / 2.0, junctions[:, 1].mean() + 20.0
    rx, rz = (p("cache")[0] - junctions[:, 0].min()) / 2.0 + 60.0, 105.0
    ellipse = ((x - cx) / rx) ** 2 + ((z - cz) / rz) ** 2
    wet = _step((1.0 - ellipse) / 0.3, 1.0)
    h = h * (1 - wet) + (MARSH + 0.3 * np.sin(x / 23.0) * np.cos(z / 19.0)) * wet
    moisture = moisture * (1 - wet) + 0.85 * wet
    xs = np.sort(junctions[:, 0])
    for k, channel_x in enumerate([(xs[0] + xs[1]) / 2, (xs[1] + xs[2]) / 2, (xs[2] + p("cache")[0]) / 2 - 15]):
        centre = channel_x + 12.0 * np.sin(z / 30.0 + k)
        w = _step(7.0 - np.abs(x - centre), 3.0) * wet
        h = h * (1 - w) + CHANNEL_BED * w
    r_cache = np.hypot(x - p("cache")[0], z - p("cache")[1])
    ring = _step(40.0 - r_cache, 6.0)
    h = h * (1 - ring) + CHANNEL_BED * ring
    islet = _step(18.0 - r_cache, 12.0)          # gentle shore a swimmer can walk out onto
    h = h * (1 - islet) + 2.4 * islet

    # River along the escarpment foot, with walkable southern banks.
    d = z - escarpment_z(x)                      # > 0 south of the escarpment line
    river = 1.0 - _step(d - 39.0, 18.0)
    h = h * (1 - river) + RIVER_BED * river

    # Escarpment and plateau.
    plateau_h = PLATEAU + 1.0 * np.sin(x / 70.0) * np.cos(z / 55.0)
    plateau = _step(-d, 2.5)
    h = h * (1 - plateau) + plateau_h * plateau
    moisture = moisture * (1 - plateau) + 0.45 * plateau

    # Ridge, its ravine, the summit dome and the beacon pillar.
    ridge_h = RIDGE + 1.0 * np.sin(x / 61.0 + 0.4) * np.cos(z / 49.0)
    sx, sz = layout.summit
    r_summit = np.hypot(x - sx, z - sz)
    ridge_h = ridge_h + DOME_RISE * _step((120.0 - r_summit) / 120.0 - 0.5, 1.0)
    cxr = layout.corridor_x
    mouth = float(ridge_z(cxr))
    ramp = PLATEAU + (RIDGE - PLATEAU) * np.clip((mouth - z) / RAVINE_LENGTH, 0.0, 1.0)
    ravine = _step(RAVINE_HALF_WIDTH - np.abs(x - cxr), 1.5) * _step(z - (mouth - RAVINE_LENGTH - 20.0), 2.0)
    ridge_h = ridge_h * (1 - ravine) + ramp * ravine
    ridge = _step(ridge_z(x) - z, 2.5)
    h = h * (1 - ridge) + ridge_h * ridge
    moisture = moisture * (1 - ridge) + 0.3 * ridge
    pillar = _step(PILLAR_RADIUS - r_summit, 2.0)
    h = h * (1 - pillar) + PILLAR_TOP * pillar

    # Level pads under lowland and plateau sites so structures sit flat.
    for name in ("spawn", "keeper", "yard_1", "yard_2", "yard_3", "junction_1", "junction_2",
                 "junction_3", "camp", "ferry_low", "ferry_high", "outpost", "relay_1", "relay_2",
                 "ruin", "summit_base"):
        ax, az = p(name)
        i = int(round((ax + terrain._half) / terrain._cell))
        j = int(round((az + terrain._half) / terrain._cell))
        level = float(h[i, j])
        w = _step(9.0 - np.hypot(x - ax, z - az), 6.0)
        h = h * (1 - w) + level * w

    # Coast: gentle beaches below 20 m, sea cliffs above.
    coast = _coast_distance(x, z, layout.seed)
    width = np.where(h < 20.0, 90.0, 24.0)
    inland = _step(coast / width - 0.5, 1.0)
    h = SEA_FLOOR + (h - SEA_FLOOR) * inland

    terrain.height[:] = h.astype(np.float32)
    terrain.moisture[:] = np.clip(moisture, 0.0, 1.0).astype(np.float32)
    terrain.mountain_mask = np.clip((h - PLATEAU) / (PILLAR_TOP - PLATEAU), 0.0, 1.0).astype(np.float32)
    terrain._compute_normals()
    terrain._compute_biomes()
    return terrain


def _dilate(mask, cells):
    out = mask.copy()
    for _ in range(cells):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def scatter_scenery(terrain: Terrain, layout: Layout) -> PropField:
    """Vegetation and rocks, kept clear of cliffs, water and site pads."""
    props = scatter_props(terrain, layout.seed, 0.55)
    steep = _dilate(terrain.slope > np.radians(35.0), 4)
    cell = terrain._cell
    gi = np.clip(np.rint((props.pos[:, 0] + terrain._half) / cell).astype(int), 0, terrain.cfg.resolution - 1)
    gj = np.clip(np.rint((props.pos[:, 2] + terrain._half) / cell).astype(int), 0, terrain.cfg.resolution - 1)
    keep = ~steep[gi, gj] & (props.pos[:, 1] > 1.0)
    sites = np.array(list(layout.anchors.values()))
    near = np.min(np.hypot(props.pos[:, None, 0] - sites[None, :, 0], props.pos[:, None, 2] - sites[None, :, 1]), axis=1)
    keep &= near > 14.0
    keep &= ~((np.abs(props.pos[:, 0] - layout.corridor_x) < RAVINE_HALF_WIDTH + 8.0)
              & (props.pos[:, 2] < ridge_z(layout.corridor_x) + 12.0)
              & (props.pos[:, 2] > ridge_z(layout.corridor_x) - RAVINE_LENGTH - 30.0))
    return PropField(props.pos[keep], props.yaw[keep], props.scale[keep], props.kind[keep])


# (building kind, site anchor, distance from the anchor in metres)
_STRUCTURES = (
    ("timber_house", "spawn", 24.0), ("windmill", "spawn", 42.0), ("market_stall", "spawn", 17.0),
    ("timber_house", "keeper", 15.0), ("windmill", "yard_2", 30.0),
    ("windmill", "cache", 8.0),
    ("watchtower", "camp", 18.0), ("market_stall", "camp", 13.0), ("market_stall", "camp", 22.0),
    ("watchtower", "ferry_low", 14.0), ("watchtower", "ferry_high", 11.0),
    ("timber_house", "outpost", 16.0), ("watchtower", "outpost", 22.0),
    ("ruin_gate", "ruin", 10.0), ("watchtower", "ruin", 22.0),
)


def place_structures(terrain: Terrain, layout: Layout) -> list:
    buildings: list[Building] = []
    names = list(layout.anchors)
    for index, (kind, anchor, distance) in enumerate(_STRUCTURES):
        ax, az = layout.anchors[anchor]
        sites = np.array([layout.anchors[n] for n in names if n != anchor])
        r = RADII[kind]
        start = _stable_int(layout.seed, kind, anchor, index) % 12
        for k in range(12):
            angle = (start + k) * np.pi / 6.0
            xz = np.array((ax + np.cos(angle) * distance, az + np.sin(angle) * distance))
            samples = xz + np.array([(a, b) for a in (-r, 0, r) for b in (-r, 0, r)])
            hs = terrain.height_at(samples[:, 0], samples[:, 1])
            if hs.min() < 0.8 or np.ptp(hs) > 2.5:
                continue
            if np.max(terrain.slope_at(samples[:, 0], samples[:, 1])) > np.radians(30.0):
                continue
            others = [b for b in buildings if np.hypot(*(xz - np.asarray(b.pos)[[0, 2]])) < r + RADII[b.kind] + 2.0]
            clear = np.hypot(sites[:, 0] - xz[0], sites[:, 1] - xz[1]) > r + 3.0
            if others or not clear.all():
                continue
            yaw = float(np.arctan2(ax - xz[0], az - xz[1]))
            buildings.append(Building(f"fullscale:{anchor}:{kind}:{index}", kind,
                                      (float(xz[0]), float(hs.max()), float(xz[1])), yaw,
                                      _stable_int(layout.seed, "appearance", anchor, kind, index),
                                      float(np.ceil(np.ptp(hs) * 10) / 10)))
            break
    return buildings


def corridor_gate(layout: Layout) -> dict:
    """Sealed gate filling the ravine mouth from floor to rim, set into both walls."""
    cx = layout.corridor_x
    mouth = float(ridge_z(cx))
    return dict(id="corridor_gate", opens=CORRIDOR_OPEN,
                position=[cx, PLATEAU - 2.0, mouth - 3.0],
                size=[2 * RAVINE_HALF_WIDTH + 14.0, RIDGE - PLATEAU + 4.0, 4.0])


def build_world(seed: int = 7, facts=()) -> FullScaleWorld:
    layout = make_layout(seed)
    terrain = author_terrain(layout)
    props = scatter_scenery(terrain, layout)
    buildings = place_structures(terrain, layout)
    lifts = [Lift("ferry_lift", layout.anchors["ferry_low"], layout.anchors["ferry_high"], FERRY_OPEN),
             Lift("summit_lift", layout.anchors["summit_base"], layout.anchors["summit_top"], CAP_INSTALLED)]
    state = FactState(gates=[corridor_gate(layout)], facts=set(facts))
    architecture = GateCollisions(Architecture(buildings), state)
    return FullScaleWorld(seed=int(seed), layout=layout, terrain=terrain, props=props,
                          buildings=buildings, lifts=lifts, state=state, architecture=architecture)
