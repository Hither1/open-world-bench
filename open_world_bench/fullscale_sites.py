"""Site compounds and regional vegetation for the full-scale Teyvat Task B island.

Every site gets a recognisable group of structures instead of one generic
building, and vegetation grows in regional groves instead of an even scatter.
Everything here is authored from the seed and the layout, independently of the
simulation RNG. Structures collide as oriented boxes (``SiteArchitecture``);
their roofs and trim are visual detail and never traversal platforms, exactly
like the shared architecture. Meshes live in ``render/site_assets.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .architecture import Architecture, Building, RADII, solid_parts
from .scatter import PropField

# kind -> (plan radius m, [(local centre, size)] colliding boxes). y is metres above the piece's ground.
_PIECES = {
    "granary": (7.6, [((0, 6.5, 0), (7.0, 14.0, 7.0))]),
    "storehouse": (6.6, [((0, 2.2, 0), (11.0, 5.4, 5.5))]),
    "workshop": (6.2, [((x, 2.0, z), (0.36, 5.0, 0.36)) for x in (-4.2, 0.0, 4.2) for z in (-2.6, 2.6)]
                 + [((0, 0.45, 1.6), (3.0, 0.9, 0.9))]),
    "well": (1.7, [((0, 0.5, 0), (1.8, 1.2, 1.8))]),
    "fence": (2.3, [((0, 0.5, 0), (4.2, 1.1, 0.25))]),
    "palisade": (2.3, [((0, 1.5, 0), (4.2, 3.4, 0.5))]),
    "tent": (2.7, [((0, 1.2, 0), (3.2, 2.6, 3.2))]),
    "standing_stone": (0.9, [((0, 1.4, 0), (1.0, 3.2, 0.8))]),
    "signal_mast": (3.2, [((0, 2.0, 0), (3.4, 4.4, 3.4))]),
    "lift_pylon": (2.2, [((0, 5.0, 0), (1.4, 11.0, 1.4))]),
    "beacon_tower": (2.4, [((0, 4.5, 0), (2.6, 9.4, 2.6))]),
    "shrine_hall": (5.2, [((x, 2.0, z), (0.42, 4.6, 0.42)) for x in (-3.2, 3.2) for z in (-2.2, 2.2)]
                    + [((0, 1.6, 2.3), (6.4, 3.4, 0.4))]),
    "crates": (1.7, [((0, 0.6, 0), (2.2, 1.3, 1.6))]),
    "cart": (1.8, [((0, 0.6, 0), (2.6, 1.2, 1.4))]),
    "banner": (0.8, [((0, 2.5, 0), (0.3, 5.4, 0.3))]),
    "ruin_wall": (2.8, [((0, 1.3, 0), (5.0, 3.0, 0.9))]),
    "broken_column": (0.8, [((0, 1.2, 0), (0.9, 2.8, 0.9))]),
    "haystack": (1.5, [((0, 0.8, 0), (2.2, 1.7, 2.2))]),
}
PIECE_KINDS = tuple(_PIECES)
PIECE_RADII = {k: v[0] for k, v in _PIECES.items()}


def piece_parts(kind, foundation_depth=0.0):
    """Colliding boxes of a piece, extended down to the lowest ground under it."""
    parts = []
    for centre, size in _PIECES[kind][1]:
        bottom = centre[1] - size[1] / 2.0
        if bottom < 0.2 and foundation_depth > 0.0:
            centre = (centre[0], centre[1] - foundation_depth / 2.0, centre[2])
            size = (size[0], size[1] + foundation_depth, size[2])
        parts.append((centre, size))
    return parts


@dataclass(frozen=True)
class SitePiece:
    id: str
    kind: str
    pos: tuple
    yaw: float
    appearance: int
    foundation_depth: float = 0.0
    site: str = ""

    @property
    def rotation(self):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))

    def metadata(self):
        return dict(id=self.id, kind=self.kind, position=list(self.pos), yaw=self.yaw, appearance=self.appearance,
                    foundation_depth=self.foundation_depth, site=self.site)


class SiteArchitecture(Architecture):
    """Shared buildings plus site pieces as one set of oriented solid boxes.

    ``buildings`` keeps only the shared kinds, which the shared renderers draw;
    ``pieces`` are drawn by ``render.site_assets``."""

    def __init__(self, buildings=(), pieces=()):
        super().__init__(buildings)
        self.pieces = tuple(pieces)
        centres, halves, rotations = [], [], []
        for piece in self.pieces:
            rotation = piece.rotation
            for centre, size in piece_parts(piece.kind, piece.foundation_depth):
                centres.append(np.asarray(centre) @ rotation.T + piece.pos)
                halves.append(np.asarray(size) / 2.0)
                rotations.append(rotation)
        if centres:
            self.centers = np.concatenate([self.centers, np.asarray(centres, float)])
            self.halves = np.concatenate([self.halves, np.asarray(halves, float)])
            self.rotations = np.concatenate([self.rotations, np.asarray(rotations, float)])
            self.radii = np.linalg.norm(self.halves[:, [0, 2]], axis=1)


# ------------------------------------------------------------------ compounds
def _ring(kind, radius, count, skip=(), phase=0.0, face="out"):
    """Pieces on a circle around the site centre; ``skip`` leaves gaps (indices)."""
    out = []
    for k in range(count):
        if k in skip:
            continue
        a = phase + 2 * np.pi * k / count
        yaw = {"out": np.degrees(a) + 90.0, "tangent": np.degrees(a) + 90.0, "in": np.degrees(a) - 90.0}[face]
        out.append((kind, radius * np.cos(a), radius * np.sin(a), yaw))
    return out


# site anchor -> (rotate with the seed?, [(kind, dx, dz, yaw degrees)]). Offsets are metres from the anchor.
_COMPOUNDS = {
    # Hearthvale: houses around a green, a well, a mill, a fenced hayfield.
    "spawn": (True, [("timber_house", 22, 8, 250), ("timber_house", 14, -22, 330), ("timber_house", -12, -24, 30),
                     ("timber_house", -26, -4, 80), ("timber_house", -22, 20, 130), ("timber_house", 4, 30, 180),
                     ("timber_house", 36, -14, 290), ("timber_house", -40, -22, 60),
                     ("well", 6, -6, 0), ("market_stall", -8, 12, 150), ("market_stall", 12, 14, 200),
                     ("windmill", 48, 20, 240), ("banner", -3, -12, 0), ("banner", 14, 2, 0), ("cart", -14, 2, 40),
                     ("crates", 18, -8, 20), ("haystack", 52, -6, 0), ("haystack", 58, -1, 0), ("haystack", 55, 6, 0)]
              + [("fence", 46 + 4.4 * k, -14, 0) for k in range(5)] + [("fence", 46 + 4.4 * k, 12, 0) for k in range(2)]
              + [("fence", 66, -11.6 + 4.4 * k, 90) for k in range(4)]),
    "keeper": (True, [("workshop", 15, 4, 255), ("storehouse", -4, -19, 10), ("crates", 9, -10, 30),
                      ("cart", -13, 8, 300), ("banner", 6, 13, 0)]),
    "relay_b02": (True, [("windmill", 24, 12, 240), ("storehouse", -22, 12, 110), ("crates", -14, -16, 0),
                         ("haystack", 18, -18, 0), ("fence", 2, 23, 0), ("fence", 6.4, 23, 0)]),
    "warden": (True, [("pavilion", 14, 2, 260), ("storehouse", -6, -21, 10), ("banner", 9, 12, 0),
                      ("banner", -9, 12, 0), ("crates", -14, 6, 60)]),
    "arena_b03": (False, _ring("standing_stone", 15.5, 12, skip=(0, 6)) + [("banner", 18, 5, 0), ("banner", -18, -5, 0)]),
    "arena_b04": (False, _ring("standing_stone", 15.5, 12, skip=(3, 9), phase=0.26)
                  + [("banner", 5, 18, 0), ("banner", -5, -18, 0)]),
    "guide": (True, [("pavilion", 14, -2, 270), ("banner", 6, 12, 0), ("banner", -8, 10, 0), ("banner", -12, -8, 0),
                     ("crates", 2, -15, 0)]),
    # Depot: the granary with its crane is the island's tallest lowland landmark.
    "depot": (True, [("granary", 19, -8, 250), ("storehouse", -8, -24, 5), ("storehouse", -26, 2, 95),
                     ("watchtower", 30, 16, 230), ("market_stall", 4, 20, 180), ("crates", 9, -14, 0),
                     ("crates", 12.5, -11, 40), ("crates", -14, -12, 15), ("cart", -12, 16, 200), ("cart", 22, 6, 100),
                     ("banner", 10, 3, 0), ("banner", -6, 12, 0), ("well", -16, 9, 0)]
              + [("fence", -30 + 4.4 * k, -34, 0) for k in range(4)] + [("fence", 36, -18 + 4.4 * k, 90) for k in range(3)]),
    "camp": (False, _ring("palisade", 24.0, 34, skip=(0, 1, 33, 16, 17, 18), face="tangent")
             + [("tent", 12, 9, 220), ("tent", -13, 8, 130), ("tent", -4, -15, 20), ("tent", 15, -9, 300),
                ("watchtower", -11, -12, 60), ("crates", 6, 15, 0), ("crates", -8, 16, 30), ("cart", 17, 2, 80),
                ("banner", 27.5, 5, 0), ("banner", 27.5, -5, 0)]),
    "lake": (False, [("storehouse", 86, 22, 250), ("crates", 80, 34, 0), ("cart", 92, 12, 30), ("banner", 78, 28, 0),
                     ("tent", -84, 30, 110), ("crates", -80, 38, 0)]),
    "cut_mouth": (False, [("watchtower", -19, -6, 180), ("watchtower", 19, -6, 180), ("storehouse", 24, 14, 270),
                          ("crates", 12, 9, 0), ("cart", -14, 12, 20), ("banner", -10, 2, 0), ("banner", 10, 2, 0)]),
    "cut_top": (False, [("watchtower", 0, -14, 0), ("banner", -9, -4, 0), ("banner", 9, -4, 0), ("crates", 12, -12, 0)]),
    "relay_b11": (True, [("signal_mast", 17, 4, 0), ("storehouse", -8, -22, 10), ("ruin_gate", -22, 6, 90),
                         ("crates", 12, -12, 0), ("banner", 2, 16, 0), ("broken_column", -14, 18, 0)]),
    "pad_b12": (True, [("tent", 13, 8, 220), ("banner", -10, 8, 0), ("crates", 10, -10, 0)]),
    "arena_b13": (False, _ring("ruin_wall", 16.5, 8, skip=(1, 5), phase=0.2, face="tangent")
                  + _ring("broken_column", 21.5, 8, phase=0.6)),
    "relay_b14": (True, [("storehouse", 16, -6, 250), ("beacon_tower", -18, -10, 0), ("crates", 8, 14, 0),
                         ("crates", 11.4, 16, 50), ("banner", -6, 15, 0)]),
    "skylift_house": (False, [("timber_house", 0, 0, 0), ("lift_pylon", 12, -14, 0), ("lift_pylon", -12, -14, 0),
                              ("storehouse", 20, 8, 270), ("crates", -12, 6, 0), ("cart", -15, -4, 60),
                              ("banner", 9, 9, 0)]),
    "highland_rim": (False, [("lift_pylon", 13, 10, 0), ("lift_pylon", -13, 10, 0), ("banner", 10, -8, 0),
                             ("crates", -10, -9, 0)]),
    "crater_gate": (False, [("watchtower", -19, -5, 180), ("watchtower", 19, -5, 180), ("banner", -10, 4, 0),
                            ("banner", 10, 4, 0), ("broken_column", -14, 14, 0), ("broken_column", 15, 16, 0)]),
    "arena_b17": (False, _ring("standing_stone", 15.5, 12, skip=(2, 8), phase=0.1)),
    "relay_b17": (True, [("ruin_wall", 14, 4, 80), ("broken_column", -12, 10, 0), ("broken_column", -10, -13, 0),
                         ("banner", 4, -15, 0)]),
    "crater_inside": (False, [("broken_column", -11, -8, 0), ("broken_column", 11, -8, 0), ("banner", -9, -18, 0),
                              ("banner", 9, -18, 0)]),
    "relay_b19": (True, [("shrine_hall", 16, -2, 265), ("ruin_gate", -20, 4, 90), ("standing_stone", 8, 15, 30),
                         ("standing_stone", -6, 16, 80), ("standing_stone", -10, -15, 10), ("standing_stone", 6, -17, 50)]),
    "pad_b18": (True, [("banner", 9, 7, 0), ("broken_column", -10, 6, 0)]),
    "pad_b20": (True, [("banner", 9, -7, 0), ("broken_column", -9, -8, 0)]),
    "crown_landing": (False, [("beacon_tower", 12, 9, 0), ("banner", 6, -10, 0)]),
    "relay_b21": (True, [("broken_column", 12, 3, 0), ("broken_column", -3, 12, 0), ("broken_column", -12, -4, 0),
                         ("ruin_wall", 4, -15, 10)]),
    "relay_b22": (True, [("ruin_wall", 14, 2, 85), ("ruin_wall", -4, 15, 5), ("ruin_wall", -13, -6, 110),
                         ("beacon_tower", 10, -15, 0), ("broken_column", -16, 8, 0)]),
    "relay_b23": (True, _ring("standing_stone", 13.5, 6, phase=0.5)),
}
_SHARED = set(RADII)


def compound_extent(site) -> float:
    """Distance from a site's anchor to its farthest piece (0 for a site without a compound)."""
    if site not in _COMPOUNDS:
        return 0.0
    return max(float(np.hypot(dx, dz)) for _, dx, dz, _ in _COMPOUNDS[site][1])


def _unit(seed, *parts):
    from .fullscale_world import _stable_int
    return (_stable_int(seed, *parts) % 100003) / 100003.0


def place_sites(terrain, layout, reserved=()):
    """Place every compound. Returns (shared-kind Buildings, SitePieces, skipped ids).

    A piece is kept only on dry, near-level ground, clear of every other site
    anchor, of ``reserved`` discs ((x, z, radius): stacks, gates, vent pads) and
    of the pieces already placed."""
    from .fullscale_world import _stable_int
    buildings, pieces, skipped = [], [], []
    anchors = layout.anchors
    names = list(anchors)
    placed = []                                    # (x, z, radius)
    reserved = [tuple(map(float, r)) for r in reserved]
    for site, (rotates, items) in _COMPOUNDS.items():
        ax, az = anchors[site]
        turn = 2 * np.pi * _unit(layout.seed, "site-yaw", site) if rotates else 0.0
        others = np.array([anchors[n] for n in names if n != site])
        for index, (kind, dx, dz, yaw_deg) in enumerate(items):
            r = RADII[kind] if kind in _SHARED else PIECE_RADII[kind]
            ident = f"fullscale-b:{site}:{kind}:{index}"
            own = float(np.hypot(dx, dz))
            chain = kind in ("fence", "palisade")       # segments of one run touch end to end
            # A piece that does not fit swings around its anchor in 20 degree steps; runs stay put.
            swings = (0.0,) if chain or own < 1e-6 else tuple(np.radians(a) for a in
                                                               (0, 20, -20, 40, -40, 60, -60, 80, -80, 110, -110, 150, -150))
            why = ""
            for swing in swings:
                cs, sn = np.cos(turn + swing), np.sin(turn + swing)
                x, z = ax + dx * cs - dz * sn, az + dx * sn + dz * cs
                samples = np.array([(x + a * r, z + b * r) for a in (-1, 0, 1) for b in (-1, 0, 1)])
                hs = terrain.height_at(samples[:, 0], samples[:, 1])
                why = ""
                if hs.min() <= 0.8 or np.ptp(hs) >= 2.5:
                    why = "uneven or wet ground"
                elif float(np.max(terrain.slope_at(samples[:, 0], samples[:, 1]))) >= np.radians(24.0):
                    why = "slope"
                elif 1e-6 < own <= r + 5.0:
                    why = "too close to its own anchor"
                elif not np.all(np.hypot(others[:, 0] - x, others[:, 1] - z) > r + 8.0):
                    why = "another site's anchor"
                elif any(np.hypot(x - px, z - pz) <= (0.0 if chain and pk == kind else r + pr + 0.6)
                         for px, pz, pr, pk in placed):
                    why = "overlaps a placed piece"
                elif any(np.hypot(x - px, z - pz) <= r + pr for px, pz, pr in reserved):
                    why = "reserved ground"
                if not why:
                    break
            if why:
                skipped.append((ident, why))
                continue
            yaw = float(np.radians(yaw_deg) - turn - swing)
            appearance = _stable_int(layout.seed, "appearance", site, kind, index)
            depth = float(np.ceil(np.ptp(hs) * 10) / 10)
            pos = (float(x), float(hs.max()), float(z))
            if kind in _SHARED:
                buildings.append(Building(ident, kind, pos, yaw, appearance, depth))
            else:
                pieces.append(SitePiece(ident, kind, pos, yaw, appearance, depth, site))
            placed.append((x, z, r, kind))
    return buildings, pieces, skipped


# ----------------------------------------------------------------- vegetation
PINE, BROADLEAF, SAKURA, BAMBOO, ROCK, BOULDER, BUSH, GRASS, CRYSTAL = range(9)
GRID = 7.0                    # metres between candidate plants
JITTER = 1.7                  # so two trunks are never closer than GRID - 2 * JITTER
# zone -> (grove tree table, open-ground table, grove density, open density); tables are (kind, weight).
_ZONES = {
    "orchard": ([(BROADLEAF, .5), (SAKURA, .38), (BUSH, .12)], [(GRASS, .6), (BUSH, .2), (SAKURA, .08), (ROCK, .12)], .62, .13),
    "lakeside": ([(BAMBOO, .55), (BROADLEAF, .25), (BUSH, .2)], [(GRASS, .62), (BUSH, .24), (ROCK, .14)], .7, .16),
    "hills": ([(PINE, .45), (BROADLEAF, .4), (BUSH, .15)], [(GRASS, .5), (BUSH, .2), (ROCK, .2), (BOULDER, .1)], .6, .13),
    "plateau": ([(PINE, .78), (BROADLEAF, .1), (BUSH, .12)], [(GRASS, .5), (BUSH, .16), (ROCK, .22), (BOULDER, .12)], .68, .11),
    "highland": ([(PINE, .5), (BUSH, .5)], [(GRASS, .4), (BUSH, .2), (ROCK, .25), (BOULDER, .15)], .3, .1),
    "crater": ([(BOULDER, .4), (CRYSTAL, .22), (BUSH, .38)], [(ROCK, .5), (GRASS, .3), (BOULDER, .2)], .32, .09),
    "crown": ([(BUSH, .6), (CRYSTAL, .4)], [(GRASS, .55), (ROCK, .45)], .22, .08),
}
_SCALE = {PINE: (.8, 1.3), BROADLEAF: (.8, 1.25), SAKURA: (.8, 1.15), BAMBOO: (.8, 1.2), ROCK: (.7, 1.5),
          BOULDER: (.7, 1.3), BUSH: (.7, 1.3), GRASS: (.8, 1.3), CRYSTAL: (.6, 1.1)}


def _lattice(ix, iz, salt):
    h = (ix.astype(np.int64) * 374761393 + iz.astype(np.int64) * 668265263 + int(salt) * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF) / float(0x1000000)


def value_noise(x, z, scale, salt):
    """Smooth deterministic noise in [0, 1) with features about ``scale`` metres across."""
    u, v = np.asarray(x, float) / scale, np.asarray(z, float) / scale
    ix, iz = np.floor(u), np.floor(v)
    fx, fz = u - ix, v - iz
    fx, fz = fx * fx * (3 - 2 * fx), fz * fz * (3 - 2 * fz)
    a, b = _lattice(ix, iz, salt), _lattice(ix + 1, iz, salt)
    c, d = _lattice(ix, iz + 1, salt), _lattice(ix + 1, iz + 1, salt)
    return (a * (1 - fx) + b * fx) * (1 - fz) + (c * (1 - fx) + d * fx) * fz


def _segment_distance(px, pz, a, b):
    ax, az = a
    bx, bz = b
    dx, dz = bx - ax, bz - az
    t = np.clip(((px - ax) * dx + (pz - az) * dz) / max(dx * dx + dz * dz, 1e-9), 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), pz - (az + t * dz))


def scatter_biomes(terrain, layout, *, zones, trails=(), clear=(), stack_tops=()):
    """Regional groves on a jittered grid.

    ``zones(x, z, h)`` returns each point's zone name or ``""`` for bare ground.
    ``trails`` are (a, b) segments kept clear for 6 m on either side, ``clear``
    are (x, z, radius) discs, and ``stack_tops`` are (x, z, radius, top) shelves
    that get only low, non-colliding plants."""
    seed = int(layout.seed)
    half = terrain._half
    axis = np.arange(-half + GRID, half - GRID, GRID)
    gx, gz = np.meshgrid(axis, axis, indexing="ij")
    ix, iz = np.arange(gx.shape[0])[:, None] + 0 * gx, np.arange(gx.shape[1])[None, :] + 0 * gz
    x = (gx + (_lattice(ix, iz, seed * 11 + 1) - .5) * 2 * JITTER).ravel()
    z = (gz + (_lattice(ix, iz, seed * 11 + 2) - .5) * 2 * JITTER).ravel()
    ix, iz = ix.ravel(), iz.ravel()
    h = terrain.height_at(x, z)
    slope = terrain.slope_at(x, z)
    keep = (h > 1.2) & (slope < np.radians(26.0))
    # Keep plants off the faces' feet and rims: sample the surroundings.
    for dx, dz in ((4, 0), (-4, 0), (0, 4), (0, -4)):
        keep &= np.abs(terrain.height_at(x + dx, z + dz) - h) < 2.2
    for a, b in trails:
        keep &= _segment_distance(x, z, a, b) > 6.0
    for cx, cz, r in clear:
        keep &= np.hypot(x - cx, z - cz) > r
    on_top = np.zeros(x.shape, bool)
    for sx, sz, r, top in stack_tops:
        near = np.hypot(x - sx, z - sz) < r + 5.0
        shelf = (np.hypot(x - sx, z - sz) < r - 2.5) & (np.hypot(x - sx, z - sz) > 4.5) & (h > top - 1.0)
        on_top |= shelf
        keep &= ~near | shelf
    zone = zones(x, z, h)
    grove = (0.62 * value_noise(x, z, 95.0, seed * 3 + 5) + 0.38 * value_noise(x, z, 38.0, seed * 3 + 6))
    pick = _lattice(ix, iz, seed * 11 + 3)
    kind_roll = _lattice(ix, iz, seed * 11 + 4)
    kind = np.full(x.shape, -1, np.int64)
    for name, (trees, ground, dense, sparse) in _ZONES.items():
        here = keep & (zone == name) & ~on_top
        in_grove = here & (grove > 0.56)
        for mask, table, density in ((in_grove, trees, dense), (here & ~in_grove, ground, sparse)):
            chosen = mask & (pick < density)
            edges = np.cumsum([w for _, w in table])
            index = np.minimum(np.searchsorted(edges / edges[-1], kind_roll, side="right"), len(table) - 1)
            kind = np.where(chosen, np.array([k for k, _ in table])[index], kind)
    kind = np.where(keep & on_top & (pick < 0.5), np.where(kind_roll < 0.6, BUSH, GRASS), kind)
    sel = kind >= 0
    kind = kind[sel]
    lo = np.array([_SCALE[k][0] for k in range(9)])[kind]
    hi = np.array([_SCALE[k][1] for k in range(9)])[kind]
    scale = lo + (hi - lo) * _lattice(ix[sel], iz[sel], seed * 11 + 5)
    yaw = 2 * np.pi * _lattice(ix[sel], iz[sel], seed * 11 + 6)
    pos = np.stack([x[sel], h[sel], z[sel]], axis=1).astype(np.float32)
    return PropField(pos, yaw.astype(np.float32), scale.astype(np.float32), kind.astype(np.int32))
