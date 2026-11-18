"""Full-scale Teyvat Task B world: the expedition island.

Level construction only: terrain, scenery, structures, the two gate boxes and
the vents. Mechanics live in :mod:`open_world_bench.fullscale_mechanics`.

Tiers (x east, z south, metres; north is -z):

* lowland, 9 m: Hearthvale village, the warden's arenas, the wind guide's
  training stacks, the expedition depot, Mirror Lake and the raider hills;
* upper valley, 58 m: north of the Great Scarp, reached through the Scarp
  Cut, a ravine ramp sealed by the causeway gate;
* highland, 104 m: north of the highland wall, reached only by the sky-lift's
  powered vents;
* crater floor, 104 m: inside a closed ring wall whose crest is 150 m, reached
  through the crater gate in a notch of the ring;
* the Crown, a 150 m mesa inside the crater, and its 196 m summit spire.

Rock stacks with flat tops are reached only by vent. Each vent lifts a
launched character to its column top (``apex``), from where a glide reaches
the next shelf. Every height step a gate depends on exceeds the proof climb
ceiling; ``fullscale_reach`` checks this on the actual heightfield.

Each seed shifts site groups by up to 18 m using a layout stream derived from
the seed. Neither the layout nor any visual detail draws from the simulation RNG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np

from .architecture import Architecture, Building, RADII
from .campaign_env import GateCollisions
from .fullscale_world import FactState, _dilate, _stable_int, _step
from .scatter import PropField, scatter_props
from .terrain import Terrain, TerrainConfig

SUITE_VERSION = "teyvat-fullscale-b-v1"
GEOGRAPHY_SCALE = 2.75
SIZE = 640.0 * GEOGRAPHY_SCALE
RESOLUTION = 705
MAX_HEIGHT = 220.0
SEA_FLOOR = -12.0
COAST_HALF = 780.0

LOWLAND = 9.0
LAKE_BED = -8.0
PLATEAU = 58.0
HIGHLAND = 104.0
RIM = 150.0
CROWN = 150.0
SPIRE = 196.0
CROWN_RADIUS = 85.0
SPIRE_RADIUS = 16.0
CRATER_INNER = 214.0
CRATER_OUTER = 244.0
CUT_HALF_WIDTH = 6.0
CUT_LENGTH = 130.0
NOTCH_HALF_WIDTH = 6.0
LAYOUT_JITTER = 18.0
GATE_DEPTH = 8.0                # at least three 2.5 m sample rows, so a full patch row is solid
UNDER_DROP = 1.0 / 2.125        # proof `under` glide: metres of descent per metre forward
APEX_MARGIN = 6.0               # column top above the under-model glide line to the target centre

# Facts in task order.
CAUSEWAY_OPEN = "causeway_open"          # B10: causeway gate at the Scarp Cut opens
RESERVE_POWER = "reserve_power"          # B14: depot reserve compartment unsealed
SKYLIFT_POWER = "skylift_power"          # B16: sky-lift vents powered
CRATER_GATE_OPEN = "crater_gate_open"    # B17: crater gate opens
CROWN_SIGNAL = "crown_signal"            # B20: Crown relays wake (needs relay B19)
SUMMIT_POWER = "summit_power"            # B23: spire vents powered
OUTPOST_CLEAR = "outpost_clear"          # B24: outpost sentinel defeated; outpost vents powered
FACT_ORDER = (CAUSEWAY_OPEN, RESERVE_POWER, SKYLIFT_POWER, CRATER_GATE_OPEN, CROWN_SIGNAL, SUMMIT_POWER,
              OUTPOST_CLEAR)

# Design coordinates before seed jitter. Groups move together.
_GROUPS = {
    "village": {"spawn": (-430, 640), "keeper": (-410, 610), "relay_b01": (-370, 575),
                "relay_b02": (-300, 640)},
    "arenas": {"warden": (-520, 520), "arena_b03": (-560, 455), "arena_b04": (-470, 420)},
    "wind": {"guide": (-265, 640), "pad_b05": (-240, 620), "stack_b05a": (-200, 600),
             "overlook_1": (-160, 625), "pad_b06": (-120, 660), "stack_b06a": (-80, 640),
             "stack_b06b": (5, 655), "overlook_2": (40, 625)},
    "depot": {"depot": (300, 665), "depot_statue": (275, 680)},
    "lake": {"lake": (210, 480)},
    "camp": {"camp": (450, 470)},
    "station": {"relay_b11": (-260, 150)},
    "lookout": {"pad_b12": (210, 260), "stack_b12a": (250, 240), "post": (330, 250)},
    "hollow": {"arena_b13": (-90, 40)},
    "reserve": {"relay_b14": (170, 30)},
    "skylift": {"skylift_house": (-330, 95), "relay_b16": (-375, 110), "pad_b16": (-330, 60),
                "stack_k1": (-330, -25)},
}
_CRATER_GROUPS = {
    # Crater sites move half as much, keeping their spacing to the ring.
    "shrine": {"pad_b18": (-120, -360), "stack_b18a": (-160, -395), "shrine": (-170, -455)},
    "court": {"relay_b19": (-60, -300)},
    "crown_approach": {"pad_b20": (-100, -320), "stack_b20a": (-110, -365), "stack_b20b": (-35, -400)},
    "outpost": {"stack_b24a": (-15, -545), "outpost": (-90, -585)},
}
_FIXED = {
    # Anchors that follow the crater and the Crown rather than a group offset.
    "crater_centre": (0.0, -430.0), "crown": (70.0, -440.0),
}
_CUT_X, _NOTCH_X = 40.0, 0.0


def scarp_z(x):
    """z of the Great Scarp face; the upper valley lies north (smaller z)."""
    x = np.asarray(x, dtype=float)
    return 300.0 + 9.0 * np.sin(x / 90.0) + 4.0 * np.sin(x / 41.0 + 1.0)


def wall_z(x):
    """z of the highland wall; the highland lies north (smaller z)."""
    x = np.asarray(x, dtype=float)
    return -80.0 + 10.0 * np.sin(x / 85.0 + 0.7)


# Stacks: name -> (radius m, height above the stack's own base tier). Rises are at
# least 40 m: the proof climb (30.8 m plus a 1 m pull-up) starts from ground up to
# 2 m above the nominal tier and the smoothed top edge lowers the rim by up to 3 m.
STACKS = {
    "stack_b05a": (8.0, 40.0), "overlook_1": (11.0, 50.0),
    "stack_b06a": (8.0, 40.0), "stack_b06b": (8.0, 46.0), "overlook_2": (11.0, 56.0),
    "stack_b12a": (9.0, 40.0), "post": (11.0, 50.0),
    "stack_k1": (10.0, 40.0),
    "stack_b18a": (8.0, 40.0), "shrine": (14.0, 46.0),
    "stack_b20a": (8.0, 40.0), "stack_b20b": (8.0, 42.0),
    "stack_b24a": (9.0, CROWN - HIGHLAND + 30.0), "outpost": (14.0, CROWN - HIGHLAND + 46.0),
}
_STACK_BASE = {"stack_b05a": LOWLAND, "overlook_1": LOWLAND, "stack_b06a": LOWLAND, "stack_b06b": LOWLAND,
               "overlook_2": LOWLAND, "stack_b12a": PLATEAU, "post": PLATEAU, "stack_k1": PLATEAU,
               "stack_b18a": HIGHLAND, "shrine": HIGHLAND, "stack_b20a": HIGHLAND, "stack_b20b": HIGHLAND,
               "stack_b24a": HIGHLAND, "outpost": HIGHLAND}

# Vent routes: (route, [(pad anchor, target anchor), ...], power fact or None).
# A leg's pad is the top of the previous shelf after the first leg.
ROUTES = (
    ("b05", (("pad_b05", "stack_b05a"), ("stack_b05a", "overlook_1")), None),
    ("b06", (("pad_b06", "stack_b06a"), ("stack_b06a", "stack_b06b"), ("stack_b06b", "overlook_2")), None),
    ("b12", (("pad_b12", "stack_b12a"), ("stack_b12a", "post")), None),
    ("b16", (("pad_b16", "stack_k1"), ("stack_k1", "highland_rim")), SKYLIFT_POWER),
    ("b18", (("pad_b18", "stack_b18a"), ("stack_b18a", "shrine")), None),
    ("b20", (("pad_b20", "stack_b20a"), ("stack_b20a", "stack_b20b"), ("stack_b20b", "crown_landing")), None),
    ("b23", (("spire_pad", "spire_top"),), SUMMIT_POWER),
    ("b24", (("outpost_pad", "stack_b24a"), ("stack_b24a", "outpost")), OUTPOST_CLEAR),
)


@dataclass(frozen=True)
class Layout:
    seed: int
    anchors: dict
    cut_x: float
    notch_x: float
    crater: tuple
    crown: tuple

    def point(self, name) -> np.ndarray:
        return np.asarray(self.anchors[name], dtype=float)


def make_layout(seed: int) -> Layout:
    rng = np.random.default_rng([int(seed), 0x7E1B, 0x52])
    anchors = {}
    groups = [(g, p, 1.0) for g, p in _GROUPS.items()] + [(g, p, 0.5) for g, p in _CRATER_GROUPS.items()]
    for group, points, scale in groups:
        offset = rng.uniform(-LAYOUT_JITTER, LAYOUT_JITTER, 2) * scale
        for name, xz in points.items():
            anchors[name] = (float(xz[0] + offset[0]), float(xz[1] + offset[1]))
    crater = tuple(float(v) for v in _FIXED["crater_centre"])
    crown = tuple(float(v) for v in np.asarray(_FIXED["crown"]) + rng.uniform(-8.0, 8.0, 2))
    anchors["crater_centre"], anchors["crown"] = crater, crown
    cut_x = _CUT_X + float(rng.uniform(-LAYOUT_JITTER, LAYOUT_JITTER))
    mouth = float(scarp_z(cut_x))
    anchors["cut_mouth"] = (cut_x, mouth + 14.0)                # mechanic and gate, lowland side
    anchors["cut_top"] = (cut_x, mouth - CUT_LENGTH - 10.0)     # upper valley statue
    notch_x = _NOTCH_X + float(rng.uniform(-10.0, 10.0))
    anchors["crater_gate"] = (notch_x, crater[1] + CRATER_OUTER + 14.0)   # outside, highland side
    anchors["crater_inside"] = (notch_x, crater[1] + CRATER_INNER - 16.0)
    anchors["highland_rim"] = (anchors["stack_k1"][0], float(wall_z(anchors["stack_k1"][0])) - 22.0)
    cx, cz = crown
    landing_dir = np.array([-0.95, 0.31])                       # toward the B20 stacks
    anchors["crown_landing"] = (cx + landing_dir[0] * (CROWN_RADIUS - 14.0), cz + landing_dir[1] * (CROWN_RADIUS - 14.0))
    anchors["relay_b21"] = (cx + 45.0, cz + 30.0)
    anchors["relay_b22"] = (cx - 10.0, cz + 55.0)
    anchors["relay_b23"] = (cx + 40.0, cz - 35.0)
    anchors["spire_pad"] = (cx + SPIRE_RADIUS + 8.0, cz + 10.0)   # east of the spire, away from the outpost
    anchors["spire_top"] = (cx, cz)
    out_dir = np.array([-0.62, -0.78])
    anchors["outpost_pad"] = (cx + out_dir[0] * (CROWN_RADIUS - 12.0), cz + out_dir[1] * (CROWN_RADIUS - 12.0))
    anchors["arena_b24"] = (anchors["outpost_pad"][0] + 14.0, anchors["outpost_pad"][1] + 10.0)
    anchors["arena_b17"] = (notch_x + 30.0, anchors["crater_gate"][1] + 25.0)
    anchors["relay_b17"] = (notch_x - 35.0, anchors["crater_gate"][1] + 20.0)
    anchors["skylift_top"] = anchors["highland_rim"]
    return Layout(seed=int(seed), anchors=anchors, cut_x=cut_x, notch_x=notch_x, crater=crater, crown=crown)


@dataclass(frozen=True)
class Vent:
    id: str
    route: str
    leg: int
    pad: tuple                  # (x, z)
    pad_height: float
    target: str                 # anchor name of the shelf it serves
    target_xz: tuple
    target_top: float
    apex: float                 # column top, absolute metres
    requires: str | None        # power fact, or None for unpowered vents


@dataclass
class ExpeditionWorld:
    seed: int
    layout: Layout
    terrain: Terrain
    props: PropField
    buildings: list
    lifts: list
    vents: list
    state: FactState
    architecture: GateCollisions
    tops: dict = field(default_factory=dict)        # stack name -> top height

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

    def route(self, name):
        return [v for v in self.vents if v.route == name]

    def regions(self) -> dict:
        """Walkable tops of each region and stack, inset from their walls."""
        x, z = self.grid()
        h = self.terrain.height
        lay = self.layout
        cx, cz = lay.crater
        r_crater = np.hypot(x - cx, z - cz)
        kx, kz = lay.crown
        r_crown = np.hypot(x - kx, z - kz)
        stacks = np.zeros(h.shape, bool)
        out = {}
        for name, (radius, _) in STACKS.items():
            sx, sz = lay.anchors[name]
            top = np.hypot(x - sx, z - sz) < radius - 2.0
            out[name] = top & (h > self.tops[name] - 3.0)
            stacks |= np.hypot(x - sx, z - sz) < radius + 4.0
        d_s = scarp_z(x) - z
        d_w = wall_z(x) - z
        cut = (np.abs(x - lay.cut_x) < CUT_HALF_WIDTH + 2.0) & (z > scarp_z(lay.cut_x) - CUT_LENGTH - 12.0)
        out["lowland"] = (d_s < -4.0) & (h > 0.5) & ~stacks
        out["plateau"] = (d_s > 4.0) & (d_w < -4.0) & (h > PLATEAU - 6.0) & ~stacks & ~cut
        out["highland"] = (d_w > 4.0) & (r_crater > CRATER_OUTER + 4.0) & (h > HIGHLAND - 6.0)
        out["crater_floor"] = (r_crater < CRATER_INNER - 4.0) & (r_crown > CROWN_RADIUS + 4.0) & ~stacks \
            & (h > HIGHLAND - 6.0) & (h < HIGHLAND + 6.0)
        out["crown"] = (r_crown < CROWN_RADIUS - 4.0) & (r_crown > SPIRE_RADIUS + 4.0) & (h > CROWN - 6.0)
        out["spire_top"] = (r_crown < SPIRE_RADIUS - 3.0) & (h > SPIRE - 3.0)
        return out

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.terrain.height).tobytes())
        h.update(np.ascontiguousarray(self.props.pos).tobytes())
        for b in self.buildings:
            h.update(repr(b.metadata()).encode())
        for g in self.gates:
            h.update(repr(sorted(g.items())).encode())
        for v in self.vents:
            h.update(repr(v).encode())
        return h.hexdigest()


def _coast_distance(x, z, seed):
    half, radius = COAST_HALF, 160.0
    qx, qz = np.abs(x) - (half - radius), np.abs(z) - (half - radius)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, qz), 0)
    wobble = 7.0 * np.sin(x / 67.0 + seed % 5) * np.cos(z / 59.0) + 4.0 * np.sin((x - z) / 31.0)
    return radius - outside + wobble


def _blend(h, target, w):
    return h * (1.0 - w) + target * w


def stack_top(name) -> float:
    radius, rise = STACKS[name]
    return _STACK_BASE[name] + rise


def author_terrain(layout: Layout) -> Terrain:
    terrain = Terrain(TerrainConfig(seed=layout.seed, size=SIZE, resolution=RESOLUTION,
                                    max_height=MAX_HEIGHT, terrace_step=0.0,
                                    snow_line=170.0, tree_line=125.0))
    axis = np.linspace(-terrain._half, terrain._half, terrain.cfg.resolution)
    x, z = np.meshgrid(axis, axis, indexing="ij")
    p = layout.point

    # Lowland: rolling meadow, the raider hills in the east.
    h = LOWLAND + 1.2 * np.sin(x / 53.0 + 1.0) * np.cos(z / 47.0) + 0.8 * np.sin((x + z) / 31.0)
    camp = p("camp") + (10.0, 20.0)
    h += 7.0 * np.exp(-((x - camp[0]) ** 2 + (z - camp[1]) ** 2) / (2 * 45.0 ** 2))
    moisture = np.full(h.shape, 0.5)

    # Mirror Lake: gentle shore down to the waterline, then a bowl to the bed.
    r_lake = np.hypot(x - p("lake")[0], z - p("lake")[1])
    shore = _step(110.0 - r_lake, 70.0)
    h = _blend(h, 0.6, shore)
    bowl = np.clip((62.0 - r_lake) / 50.0, 0.0, 1.0)
    h = np.where(r_lake < 62.0, 0.6 + (LAKE_BED - 0.6) * (bowl * (2 - bowl)), h)
    moisture = _blend(moisture, 0.9, shore)

    # Great Scarp and the upper valley, with the Scarp Cut ramp.
    plateau_h = PLATEAU + 1.0 * np.sin(x / 70.0) * np.cos(z / 55.0)
    mouth = float(scarp_z(layout.cut_x))
    ramp = LOWLAND + (PLATEAU - LOWLAND) * np.clip((mouth - z) / CUT_LENGTH, 0.0, 1.0)
    cut = _step(CUT_HALF_WIDTH - np.abs(x - layout.cut_x), 1.5) * _step(z - (mouth - CUT_LENGTH - 20.0), 2.0)
    plateau_h = _blend(plateau_h, ramp, cut)
    plateau = _step(scarp_z(x) - z, 2.5)
    h = _blend(h, plateau_h, plateau)
    moisture = _blend(moisture, 0.45, plateau)

    # Highland wall, the crater ring with its gate notch, the Crown and the spire.
    highland_h = HIGHLAND + 1.0 * np.sin(x / 61.0 + 0.4) * np.cos(z / 49.0)
    cx, cz = layout.crater
    r_crater = np.hypot(x - cx, z - cz)
    ring = _step(r_crater - CRATER_INNER, 3.0) * _step(CRATER_OUTER - r_crater, 3.0)
    notch = _step(NOTCH_HALF_WIDTH - np.abs(x - layout.notch_x), 1.5) * (z > cz + 100.0)
    highland_h = _blend(highland_h, RIM + 2.0 * np.sin(np.arctan2(z - cz, x - cx) * 7.0), ring * (1.0 - notch))
    floor = _step(CRATER_INNER - r_crater, 3.0)
    highland_h = _blend(highland_h, HIGHLAND + 0.6 * np.sin(x / 37.0) * np.cos(z / 41.0), floor)
    kx, kz = layout.crown
    r_crown = np.hypot(x - kx, z - kz)
    highland_h = _blend(highland_h, CROWN + 0.8 * np.sin(x / 29.0) * np.cos(z / 33.0), _step(CROWN_RADIUS - r_crown, 2.5))
    highland_h = _blend(highland_h, SPIRE, _step(SPIRE_RADIUS - r_crown, 2.0))
    highland = _step(wall_z(x) - z, 2.5)
    h = _blend(h, highland_h, highland)
    moisture = _blend(moisture, 0.3, highland)

    # Level pads under sites so structures, relays and arenas sit flat.
    pads = [n for n in layout.anchors if n not in STACKS and n not in ("crater_centre", "crown", "spire_top",
                                                                       "lake", "cut_mouth", "cut_top", "highland_rim")]
    for name in pads:
        ax, az = p(name)
        i = int(round((ax + terrain._half) / terrain._cell))
        j = int(round((az + terrain._half) / terrain._cell))
        level = float(h[i, j])
        radius = 16.0 if name.startswith(("arena", "relay")) else 9.0
        w = _step(radius - np.hypot(x - ax, z - az), 6.0)
        h = _blend(h, level, w)

    # Rock stacks reached only by vent: flat tops, sheer sides.
    for name, (radius, _) in STACKS.items():
        sx, sz = p(name)
        top = stack_top(name)
        w = _step(radius - np.hypot(x - sx, z - sz), 2.0)
        h = np.maximum(h, _blend(h, top, w))

    # Coast: beaches below 20 m, sea cliffs above.
    coast = _coast_distance(x, z, layout.seed)
    width = np.where(h < 20.0, 90.0, 24.0)
    inland = _step(coast / width - 0.5, 1.0)
    h = SEA_FLOOR + (h - SEA_FLOOR) * inland

    terrain.height[:] = h.astype(np.float32)
    terrain.moisture[:] = np.clip(moisture, 0.0, 1.0).astype(np.float32)
    terrain.mountain_mask = np.clip((h - PLATEAU) / (SPIRE - PLATEAU), 0.0, 1.0).astype(np.float32)
    terrain._compute_normals()
    terrain._compute_biomes()
    return terrain


def make_vents(terrain: Terrain, layout: Layout) -> list:
    vents = []
    for route, legs, power in ROUTES:
        for k, (pad, target) in enumerate(legs):
            px, pz = layout.anchors[pad]
            tx, tz = layout.anchors[target]
            pad_h = float(terrain.height_at(px, pz))
            top = float(terrain.height_at(tx, tz))
            d = float(np.hypot(tx - px, tz - pz))
            apex = max(top, pad_h) + UNDER_DROP * d + APEX_MARGIN
            vents.append(Vent(id=f"vent_{route}_{k + 1}", route=route, leg=k + 1, pad=(px, pz), pad_height=pad_h,
                              target=target, target_xz=(tx, tz), target_top=top, apex=round(apex, 3),
                              requires=power))
    return vents


def scatter_scenery(terrain: Terrain, layout: Layout) -> PropField:
    props = scatter_props(terrain, layout.seed, 0.5)
    steep = _dilate(terrain.slope > np.radians(35.0), 4)
    cell = terrain._cell
    gi = np.clip(np.rint((props.pos[:, 0] + terrain._half) / cell).astype(int), 0, terrain.cfg.resolution - 1)
    gj = np.clip(np.rint((props.pos[:, 2] + terrain._half) / cell).astype(int), 0, terrain.cfg.resolution - 1)
    keep = ~steep[gi, gj] & (props.pos[:, 1] > 1.0)
    sites = np.array(list(layout.anchors.values()))
    near = np.min(np.hypot(props.pos[:, None, 0] - sites[None, :, 0], props.pos[:, None, 2] - sites[None, :, 1]), axis=1)
    keep &= near > 16.0
    mouth = float(scarp_z(layout.cut_x))
    keep &= ~((np.abs(props.pos[:, 0] - layout.cut_x) < CUT_HALF_WIDTH + 8.0)
              & (props.pos[:, 2] < mouth + 20.0) & (props.pos[:, 2] > mouth - CUT_LENGTH - 30.0))
    nx = layout.notch_x
    cz = layout.crater[1]
    keep &= ~((np.abs(props.pos[:, 0] - nx) < NOTCH_HALF_WIDTH + 8.0) & (props.pos[:, 2] > cz + CRATER_INNER - 30.0)
              & (props.pos[:, 2] < cz + CRATER_OUTER + 30.0))
    return PropField(props.pos[keep], props.yaw[keep], props.scale[keep], props.kind[keep])


# (building kind, site anchor, distance from the anchor in metres)
_STRUCTURES = (
    ("timber_house", "spawn", 24.0), ("windmill", "spawn", 42.0), ("market_stall", "spawn", 17.0),
    ("timber_house", "keeper", 16.0), ("windmill", "relay_b02", 26.0), ("pavilion", "warden", 14.0),
    ("pavilion", "guide", 14.0),
    ("windmill", "depot", 12.0), ("watchtower", "depot", 20.0), ("market_stall", "depot", 22.0),
    ("watchtower", "camp", 18.0), ("market_stall", "camp", 13.0), ("market_stall", "camp", 22.0),
    ("watchtower", "cut_mouth", 16.0), ("watchtower", "cut_top", 14.0),
    ("ruin_gate", "relay_b11", 22.0), ("timber_house", "skylift_house", 0.0),
    ("watchtower", "crater_gate", 22.0), ("ruin_gate", "relay_b19", 20.0),
)


def place_structures(terrain: Terrain, layout: Layout) -> list:
    buildings: list[Building] = []
    names = list(layout.anchors)
    for index, (kind, anchor, distance) in enumerate(_STRUCTURES):
        ax, az = layout.anchors[anchor]
        sites = np.array([layout.anchors[n] for n in names if n != anchor])
        r = RADII[kind]
        start = _stable_int(layout.seed, kind, anchor, index) % 12
        for k in range(12 if distance > 0 else 1):
            angle = (start + k) * np.pi / 6.0
            xz = np.array((ax + np.cos(angle) * distance, az + np.sin(angle) * distance))
            samples = xz + np.array([(a, b) for a in (-r, 0, r) for b in (-r, 0, r)])
            hs = terrain.height_at(samples[:, 0], samples[:, 1])
            if distance > 0:
                if hs.min() < 0.8 or np.ptp(hs) > 2.5:
                    continue
                if np.max(terrain.slope_at(samples[:, 0], samples[:, 1])) > np.radians(30.0):
                    continue
                others = [b for b in buildings if np.hypot(*(xz - np.asarray(b.pos)[[0, 2]])) < r + RADII[b.kind] + 2.0]
                clear = np.hypot(sites[:, 0] - xz[0], sites[:, 1] - xz[1]) > r + 6.0
                if others or not clear.all():
                    continue
            yaw = float(np.arctan2(ax - xz[0], az - xz[1])) if distance > 0 else 0.0
            buildings.append(Building(f"fullscale-b:{anchor}:{kind}:{index}", kind,
                                      (float(xz[0]), float(hs.max()), float(xz[1])), yaw,
                                      _stable_int(layout.seed, "appearance", anchor, kind, index),
                                      float(np.ceil(np.ptp(hs) * 10) / 10)))
            break
    return buildings


def causeway_gate(layout: Layout) -> dict:
    """Gate filling the Scarp Cut mouth from floor to rim, set 7 m into both walls."""
    cx = layout.cut_x
    mouth = float(scarp_z(cx))
    return dict(id="causeway_gate", opens=CAUSEWAY_OPEN, position=[cx, LOWLAND - 2.0, mouth - 5.0],
                size=[2 * CUT_HALF_WIDTH + 14.0, PLATEAU - LOWLAND + 6.0, GATE_DEPTH])


def crater_gate(layout: Layout) -> dict:
    """Gate filling the ring notch from floor to crest, set 7 m into both sides."""
    cz = layout.crater[1]
    return dict(id="crater_gate", opens=CRATER_GATE_OPEN,
                position=[layout.notch_x, HIGHLAND - 2.0, cz + (CRATER_INNER + CRATER_OUTER) / 2.0],
                size=[2 * NOTCH_HALF_WIDTH + 14.0, RIM - HIGHLAND + 8.0, GATE_DEPTH])


def build_world(seed: int = 7, facts=()) -> ExpeditionWorld:
    layout = make_layout(seed)
    terrain = author_terrain(layout)
    props = scatter_scenery(terrain, layout)
    buildings = place_structures(terrain, layout)
    vents = make_vents(terrain, layout)
    state = FactState(gates=[causeway_gate(layout), crater_gate(layout)], facts=set(facts))
    architecture = GateCollisions(Architecture(buildings), state)
    tops = {name: float(terrain.height_at(*layout.anchors[name])) for name in STACKS}
    return ExpeditionWorld(seed=int(seed), layout=layout, terrain=terrain, props=props, buildings=buildings,
                           lifts=[], vents=vents, state=state, architecture=architecture, tops=tops)
