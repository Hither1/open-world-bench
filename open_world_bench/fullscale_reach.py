"""Reachability proofs for the full-scale Teyvat world.

The ground the controller stands on is the bilinear surface between height
samples. Each square of four samples (a *patch*) is one graph node. For every
patch this module computes which part of it is walkable (gradient no steeper
than the walkable slope) and the height range of that part. Two traversal
models bracket the real :class:`~open_world_bench.fullscale_physics.FullScaleController`:

``over``
    More capable than the controller. It walks between neighbouring patches
    whenever their walkable heights meet, swims without limit, falls to any
    lower neighbour, glides without stamina limits at an optimistic ratio,
    and climbs up to ``climb_height_bound`` above the highest walkable point of
    the patch it starts from (a climb that starts mid-air, after a fall or a
    glide, gets only the climbing part of that bound, since launch skills need
    walkable ground). A climb may also move sideways for free, but only within
    the budget's distance of where it started (up to 1.41 times that along
    diagonals): the controller spends climbing stamina per second in every
    direction and cannot rest on a face, so no climb travels further than the
    budget. A face narrower than one patch, between two patches that each have
    somewhere to stand, can likewise be climbed if it is within the budget. A
    region this model cannot reach is unreachable.

``under``
    Less capable. It walks only across patches that are gentle everywhere,
    wades out of water only through gentle shallows, and never climbs, falls
    or glides. A region this model reaches is reachable. Lifts run when their
    fact holds. Designed glides are checked by flying the real controller
    (:mod:`open_world_bench.fullscale_routes`).

Patches under a building footprint or a closed gate block walking and
climbing. A glide may pass over them only above their top.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .character import CharacterConfig
from .fullscale_physics import VAULT_RISE, climb_height_bound
from .party import PartyConfig

SQRT2 = float(np.sqrt(2.0))
ALL8 = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
HALF8 = ((1, 0), (0, 1), (1, 1), (1, -1))
TOP_OUT = VAULT_RISE    # metres a climber can pull up onto the top of a face
LIFT_REACH = 4.0        # metres from a lift endpoint to its platform
SUBSAMPLES = 9          # samples per patch side when measuring walkable heights


@dataclass(frozen=True)
class TraversalModel:
    name: str
    walk_slope_deg: float       # gradient limit for standing
    whole_patch: bool           # under: a patch counts only if all of it is walkable
    water_exit: float           # lowest walkable height a swimmer can step onto (m above sea)
    climb_budget: float         # height gained on unwalkable faces from standing (0 = no climbing)
    air_climb_budget: float     # height gained after grabbing a face mid-air
    falls: bool                 # may drop to any lower neighbour
    glide_ratio: float          # metres forward per metre of descent
    glide_steps: int            # most patch steps in one glide
    glide_start: float          # glide height relative to the takeoff ground
    glide_clearance: float      # height a glide keeps above terrain and structures
    notes: dict = field(default_factory=dict, compare=False)


def over_model(cell: float, char_cfg: CharacterConfig | None = None,
               party: PartyConfig | None = None) -> TraversalModel:
    cfg = char_cfg or CharacterConfig()
    bound = climb_height_bound(cfg, party)
    ratio = cfg.glide_speed / abs(cfg.glide_fall_speed)
    refund = 25.0                              # largest launch-skill stamina refund in the roster
    glide_range = (cfg.stamina_max + refund) / cfg.glide_cost * cfg.glide_speed
    return TraversalModel(
        name="over", walk_slope_deg=cfg.max_walk_slope_deg + 1.0, whole_patch=False,
        water_exit=cell * np.tan(np.radians(cfg.max_walk_slope_deg)) * 1.05,
        climb_budget=bound.total, air_climb_budget=bound.climb * bound.margin, falls=True,
        # Octile steps overestimate Euclidean distance by up to 8.24%; widen to stay optimistic.
        glide_ratio=ratio * 1.25 * 1.0824, glide_steps=int(np.ceil(glide_range * 1.25 / cell)),
        glide_start=bound.start, glide_clearance=0.0,
        notes=dict(climb_bound=bound.total, glide_range=glide_range))


def under_model(cell: float, char_cfg: CharacterConfig | None = None) -> TraversalModel:
    """Walking, swimming and lifts only. Designed glides are checked with the
    real controller instead (see ``fullscale_routes``)."""
    cfg = char_cfg or CharacterConfig()
    ratio = cfg.glide_speed / abs(cfg.glide_fall_speed)
    return TraversalModel(
        name="under", walk_slope_deg=40.0, whole_patch=True, water_exit=1.5,
        climb_budget=0.0, air_climb_budget=0.0, falls=False, glide_ratio=min(2.2, ratio * 0.8),
        glide_steps=0, glide_start=-2.0, glide_clearance=1.5)


# ----------------------------------------------------------------- grid helpers
def _shift(a, di, dj, fill):
    """out[i, j] = a[i + di, j + dj], with ``fill`` outside the grid."""
    out = np.full_like(a, fill)
    n, m = a.shape
    out[max(-di, 0):n - max(di, 0), max(-dj, 0):m - max(dj, 0)] = \
        a[max(di, 0):n - max(-di, 0), max(dj, 0):m - max(-dj, 0)]
    return out


def _components(nodes, edges):
    """Connected-component labels by min-label propagation with pointer jumping.

    ``edges`` holds (di, dj, valid) where valid[i, j] marks an undirected edge
    between (i, j) and (i + di, j + dj). Non-nodes get label ``nodes.size``.
    """
    n = nodes.size
    lab = np.where(nodes, np.arange(n).reshape(nodes.shape), n)
    backs = [(-di, -dj, _shift(valid, -di, -dj, False)) for di, dj, valid in edges]
    while True:
        before = lab
        for (di, dj, valid), (bi, bj, back_valid) in zip(edges, backs):
            nb = _shift(lab, di, dj, n)
            lab = np.where(valid & (nb < lab), nb, lab)
            back = _shift(lab, bi, bj, n)
            lab = np.where(back_valid & (back < lab), back, lab)
        flat = lab.ravel().copy()
        while True:
            live = flat < n
            jumped = flat.copy()
            jumped[live] = flat[flat[live]]
            if np.array_equal(jumped, flat):
                break
            flat = jumped
        lab = flat.reshape(nodes.shape)
        if np.array_equal(lab, before):
            return lab


def _min_abs(a, b):
    """Smallest |value| of a linear function between a (t=0) and b (t=1), and where."""
    cross = np.sign(a) != np.sign(b)
    t_cross = np.clip(a / np.where(cross, a - b, 1.0), 0.0, 1.0)
    value = np.where(cross, 0.0, np.minimum(np.abs(a), np.abs(b)))
    where = np.where(cross, t_cross, np.where(np.abs(a) <= np.abs(b), 0.0, 1.0))
    return value, where


def patch_surface(h, cell, slope_deg):
    """Per-patch walkability of the bilinear surface.

    Returns (walk_some, walk_all, lo, hi): whether some or all of each patch is
    no steeper than ``slope_deg``, and the height range of its walkable part.
    The gradient along x depends only on the z fraction and vice versa, so the
    walkable part is convex and contains the point of least gradient. Sampling a
    grid that includes that point bounds the walkable heights to within one
    sub-cell, and that error is added to the range.
    """
    tan = np.tan(np.radians(slope_deg))
    f00, f10, f01, f11 = h[:-1, :-1], h[1:, :-1], h[:-1, 1:], h[1:, 1:]
    gx0, gx1 = (f10 - f00) / cell, (f11 - f01) / cell     # d/dx at z-fraction 0 and 1
    gz0, gz1 = (f01 - f00) / cell, (f11 - f10) / cell     # d/dz at x-fraction 0 and 1
    min_gx, tz_best = _min_abs(gx0, gx1)
    min_gz, tx_best = _min_abs(gz0, gz1)
    walk_some = np.hypot(min_gx, min_gz) <= tan
    max_grad = np.hypot(np.maximum(np.abs(gx0), np.abs(gx1)), np.maximum(np.abs(gz0), np.abs(gz1)))
    walk_all = max_grad <= tan
    fixed = np.linspace(0.0, 1.0, SUBSAMPLES)
    lo = np.full(f00.shape, np.inf)
    hi = np.full(f00.shape, -np.inf)
    tx_values = [np.full(f00.shape, v) for v in fixed] + [tx_best]
    tz_values = [np.full(f00.shape, v) for v in fixed] + [tz_best]
    for tx in tx_values:
        gz = gz0 * (1 - tx) + gz1 * tx
        top_row = f00 * (1 - tx) + f10 * tx
        bottom_row = f01 * (1 - tx) + f11 * tx
        for tz in tz_values:
            gx = gx0 * (1 - tz) + gx1 * tz
            ok = gx * gx + gz * gz <= tan * tan
            value = top_row * (1 - tz) + bottom_row * tz
            lo = np.where(ok, np.minimum(lo, value), lo)
            hi = np.where(ok, np.maximum(hi, value), hi)
    error = tan * SQRT2 * cell / (SUBSAMPLES - 1)
    lo, hi = lo - error, hi + error
    corners_lo = np.minimum(np.minimum(f00, f10), np.minimum(f01, f11))
    corners_hi = np.maximum(np.maximum(f00, f10), np.maximum(f01, f11))
    return walk_some, walk_all, lo, hi, corners_lo, corners_hi


def obstacle_grid(world, facts):
    """Blocked samples and obstacle tops from buildings and gates closed under ``facts``."""
    t = world.terrain
    x, z = world.grid()
    blocked = np.zeros(t.height.shape, bool)
    top = np.full(t.height.shape, -np.inf)
    base = getattr(world.architecture, "base", world.architecture)
    boxes = [(c, hf, r) for c, hf, r in zip(base.centers, base.halves, base.rotations)]
    for gate in world.gates:
        if gate["opens"] not in facts:
            size = np.asarray(gate["size"], float)
            centre = np.asarray(gate["position"], float) + (0.0, size[1] / 2, 0.0)
            boxes.append((centre, size / 2, np.eye(3)))
    pad = 0.4                                   # character radius
    for centre, half, rotation in boxes:
        reach = float(np.hypot(half[0], half[2])) + pad + t._cell
        i0, i1 = np.searchsorted(x[:, 0], [centre[0] - reach, centre[0] + reach])
        j0, j1 = np.searchsorted(z[0, :], [centre[2] - reach, centre[2] + reach])
        px, pz = x[i0:i1, j0:j1] - centre[0], z[i0:i1, j0:j1] - centre[2]
        lx = px * rotation[0, 0] + pz * rotation[2, 0]
        lz = px * rotation[0, 2] + pz * rotation[2, 2]
        inside = (np.abs(lx) <= half[0] + pad) & (np.abs(lz) <= half[2] + pad)
        blocked[i0:i1, j0:j1] |= inside
        tops = top[i0:i1, j0:j1]
        tops[inside] = np.maximum(tops[inside], centre[1] + half[1])
    return blocked, top


def _corners(a, reduce):
    return reduce(reduce(a[:-1, :-1], a[1:, :-1]), reduce(a[:-1, 1:], a[1:, 1:]))


@dataclass
class Graph:
    lo: np.ndarray              # lowest standing height per patch (0 for water)
    hi: np.ndarray              # highest standing height per patch
    corner_lo: np.ndarray
    corner_hi: np.ndarray
    blocked: np.ndarray
    glide_top: np.ndarray       # height a glide must exceed to pass over the patch
    water: np.ndarray
    rest: np.ndarray            # patches where one can stand or float
    steep: np.ndarray           # land patches with nowhere to stand
    walk_label: np.ndarray


def build_graph(world, facts, model: TraversalModel) -> Graph:
    t = world.terrain
    cell = t._cell
    h = t.height.astype(np.float64)
    walk_some, walk_all, lo, hi, corner_lo, corner_hi = patch_surface(h, cell, model.walk_slope_deg)
    blocked_nodes, box_top = obstacle_grid(world, facts)
    if model.whole_patch:
        blocked = _corners(blocked_nodes, np.logical_or)
        water = corner_hi < -0.3
        walk = walk_all & (corner_hi >= 0.0) & (corner_lo >= -2.0)    # gentle ground, shallows included
        lo, hi = corner_lo, corner_hi
        glide_top = np.maximum(corner_hi, _corners(box_top, np.maximum))
    else:
        blocked = _corners(blocked_nodes, np.logical_and)
        water = corner_lo < 0.0
        walk = walk_some & (hi >= 0.0)
        glide_top = np.maximum(corner_lo, np.where(blocked, _corners(box_top, np.maximum), -np.inf))
    water &= ~blocked
    walk &= ~blocked
    rest = water | walk
    lo = np.where(water, np.minimum(np.where(walk, lo, 0.0), 0.0), lo)
    hi = np.where(water, np.maximum(np.where(walk, hi, 0.0), 0.0), hi)
    steep = ~rest & ~blocked & (corner_hi >= 0.0)
    edges = []
    for di, dj in HALF8:
        nb_rest = _shift(rest, di, dj, False)
        nb_lo, nb_hi = _shift(lo, di, dj, np.inf), _shift(hi, di, dj, -np.inf)
        if model.whole_patch:
            nb_water, nb_walk = _shift(water, di, dj, False), _shift(walk, di, dj, False)
            # A gentle patch beside open water shares underwater corners with it.
            valid = (walk & nb_walk) | (water & nb_water) | (water & nb_walk) | (walk & nb_water)
        else:
            meet = (lo <= nb_hi) & (nb_lo <= hi)
            valid = rest & nb_rest & meet
            nb_water = _shift(water, di, dj, False)
            valid |= (water & nb_rest & (nb_lo <= model.water_exit)) | \
                     (rest & nb_water & (lo <= model.water_exit))
        edges.append((di, dj, valid & rest & nb_rest))
    walk_label = _components(rest, edges)
    return Graph(lo=lo, hi=hi, corner_lo=corner_lo, corner_hi=corner_hi, blocked=blocked,
                 glide_top=glide_top, water=water, rest=rest, steep=steep, walk_label=walk_label)


def _spread(seeds, steep, steps):
    """Carry each climb ceiling across up to ``steps`` neighbouring face patches."""
    ceiling = seeds
    for _ in range(steps):
        grown = ceiling
        for di, dj in ALL8:
            grown = np.maximum(grown, np.where(steep, _shift(ceiling, di, dj, -np.inf), -np.inf))
        if np.array_equal(grown, ceiling):
            break
        ceiling = grown
    return ceiling


def patch_centres(world):
    x, z = world.grid()
    return (x[:-1, :-1] + x[1:, 1:]) / 2.0, (z[:-1, :-1] + z[1:, 1:]) / 2.0


@dataclass
class Reach:
    model: TraversalModel
    facts: tuple
    graph: Graph
    rest: np.ndarray            # patches where the character can stand or float
    climb: np.ndarray           # unwalkable patches a climber can hang on
    ceiling: np.ndarray         # highest a climber can get on each face patch
    iterations: int

    @property
    def any(self):
        return self.rest | self.climb

    def at(self, world, name_or_xz, radius=3.0) -> bool:
        xz = world.layout.anchors[name_or_xz] if isinstance(name_or_xz, str) else name_or_xz
        x, z = patch_centres(world)
        near = np.hypot(x - xz[0], z - xz[1]) <= radius
        return bool((self.rest & near).any())

    def max_height(self):
        g = self.graph
        stand = np.where(self.rest & ~g.water, g.hi, -np.inf)
        hang = np.where(self.climb, np.minimum(self.ceiling, g.corner_hi), -np.inf)
        return float(max(stand.max(), hang.max()))


def region_patches(world) -> dict:
    """Region masks on patches: every corner of the patch lies in the region."""
    return {k: _corners(m, np.logical_and) for k, m in world.regions().items()}


def _nearest_rest(world, graph, xz):
    x, z = patch_centres(world)
    d = np.where(graph.rest & ~graph.water, np.hypot(x - xz[0], z - xz[1]), np.inf)
    index = np.unravel_index(int(np.argmin(d)), d.shape)
    if not np.isfinite(d[index]) or d[index] > 6.0:
        raise ValueError(f"no standable patch within 6 m of {xz}")
    return index


def _glide(start, carry, g, model, cell, climbing):
    """Spread glides from ``start`` heights; return the glide field, the face
    patches hit (with the hit height) and the largest ``carry`` value arriving."""
    glide, value = start, carry
    hit = np.full(start.shape, -np.inf)
    hit_value = np.full(start.shape, -np.inf)
    steep = g.steep
    for _ in range(model.glide_steps):
        incoming = np.full(start.shape, -np.inf)
        incoming_value = np.full(start.shape, -np.inf)
        for di, dj in ALL8:
            cost = cell * (SQRT2 if di and dj else 1.0) / model.glide_ratio
            incoming = np.maximum(incoming, _shift(glide, di, dj, -np.inf) - cost)
            incoming_value = np.maximum(incoming_value, _shift(value, di, dj, -np.inf))
        passable = incoming > g.glide_top + model.glide_clearance
        if climbing:
            touch = steep & (incoming >= g.corner_lo)
            hit = np.maximum(hit, np.where(touch, incoming, -np.inf))
            hit_value = np.maximum(hit_value, np.where(touch, incoming_value, -np.inf))
        grow = passable & (incoming > glide)
        if not grow.any():
            break
        glide = np.where(grow, incoming, glide)
        value = np.where(grow, np.maximum(value, incoming_value), value)
    return glide, hit, hit_value


def reach(world, facts=(), model: TraversalModel | None = None, start="spawn",
          use_lifts: bool = True, char_cfg=None, party=None) -> Reach:
    """Fixed point of everything ``model`` can reach from ``start``.

    Stamina recovers only while standing on walkable ground or swimming. A
    climb, fall or glide that begins on standing ground may grab a face with a
    full climb budget. One that begins on a face keeps the ceiling it already
    had and gains no new height.
    """
    cell = world.terrain._cell
    model = model or over_model(cell, char_cfg, party)
    facts = tuple(sorted(set(facts)))
    g = build_graph(world, set(facts), model)
    rest, steep = g.rest, g.steep
    reached = np.zeros(rest.shape, bool)
    xz = world.layout.anchors[start] if isinstance(start, str) else start
    reached[_nearest_rest(world, g, xz)] = True
    x, z = patch_centres(world)
    lifts = []
    if use_lifts:
        for lift in world.lifts:
            if lift.opens in facts:
                a = rest & ~g.water & (np.hypot(x - lift.bottom[0], z - lift.bottom[1]) <= LIFT_REACH)
                b = rest & ~g.water & (np.hypot(x - lift.top[0], z - lift.top[1]) <= LIFT_REACH)
                lifts.append((a, b))
    climbing = model.climb_budget > 0.0
    spread_steps = int(np.ceil(model.climb_budget / cell))
    none = np.full(rest.shape, -np.inf)
    seeds = none.copy()                          # climb ceilings where climbs begin
    memo = {}                                    # glide results keyed by their unchanged inputs
    iterations = 0
    while True:
        iterations += 1
        before_reached, before_seeds = reached.copy(), seeds.copy()
        labels = np.unique(g.walk_label[reached & rest])
        reached |= np.isin(g.walk_label, labels) & rest
        for a, b in lifts:
            if (reached & a).any():
                reached |= b
            if (reached & b).any():
                reached |= a
        stand = np.where(reached & rest, g.hi, -np.inf)
        if climbing:
            for di, dj in ALL8:
                seeds = np.maximum(seeds, np.where(steep, _shift(stand, di, dj, -np.inf) + model.climb_budget, -np.inf))
            ceiling = _spread(seeds, steep, spread_steps)
            hanging = steep & (g.corner_lo <= ceiling)
            hold = np.where(hanging, np.minimum(ceiling, g.corner_hi), -np.inf)
            held_ceiling = np.where(hanging, ceiling, -np.inf)
        else:
            hanging, hold, held_ceiling = np.zeros(rest.shape, bool), none, none
        if model.falls:
            for di, dj in ALL8:
                from_stand = _shift(stand, di, dj, -np.inf)
                from_hold = _shift(hold, di, dj, -np.inf)
                reached |= rest & ((g.lo < from_stand) | (g.lo < from_hold))
                if climbing:
                    seeds = np.maximum(seeds, np.where(steep & (g.corner_lo < from_stand),
                                                       from_stand + model.climb_budget, -np.inf))
                    seeds = np.maximum(seeds, np.where(steep & (g.corner_lo < from_hold),
                                                       _shift(held_ceiling, di, dj, -np.inf), -np.inf))
        if climbing:
            for di, dj in ALL8:
                reached |= rest & (g.lo <= _shift(held_ceiling, di, dj, -np.inf) + TOP_OUT)
                # A face inside two neighbouring standing patches, climbed from rest.
                reached |= rest & (g.lo <= _shift(stand, di, dj, -np.inf) + model.climb_budget + TOP_OUT)
        if model.glide_steps:
            if model.name == "under":
                launch = np.where(reached & rest & ~g.water, g.lo + model.glide_start, -np.inf)
            else:
                launch = np.where(reached & rest, g.hi + model.glide_start, -np.inf)
            landing = g.lo if model.name == "over" else g.hi
            if "stand" not in memo or not np.array_equal(memo["stand"][0], launch):
                memo["stand"] = (launch, _glide(launch, none, g, model, cell, climbing))
            glide, hit, _ = memo["stand"][1]
            reached |= rest & np.isfinite(glide) & (glide >= landing)
            if climbing:
                seeds = np.maximum(seeds, np.where(np.isfinite(hit), np.minimum(hit, g.corner_hi)
                                                   + model.air_climb_budget, -np.inf))
                if "hold" not in memo or not (np.array_equal(memo["hold"][0], hold)
                                              and np.array_equal(memo["hold"][1], held_ceiling)):
                    memo["hold"] = (hold, held_ceiling, _glide(hold, held_ceiling, g, model, cell, climbing))
                glide, _, hit_ceiling = memo["hold"][2]
                reached |= rest & np.isfinite(glide) & (glide >= landing)
                seeds = np.maximum(seeds, hit_ceiling)
        if np.array_equal(reached, before_reached) and np.array_equal(seeds, before_seeds):
            break
    ceiling = _spread(seeds, steep, spread_steps) if climbing else none
    climb = steep & (g.corner_lo <= ceiling)
    return Reach(model=model, facts=facts, graph=g, rest=reached, climb=climb, ceiling=ceiling,
                 iterations=iterations)


# ----------------------------------------------------------------- gate checks
LOWLAND_SITES = ("spawn", "keeper", "yard_1", "yard_2", "yard_3", "junction_1", "junction_2",
                 "junction_3", "cache", "camp", "ferry_low", "ruin")
PLATEAU_SITES = ("ferry_high", "outpost", "west_edge", "corridor_mouth")
RIDGE_SITES = ("relay_1", "relay_2", "summit_base", "ravine_top")
SUMMIT_SITES = ("summit_top",)


def check_world(world, char_cfg=None, party=None, controller_checks: bool = True) -> dict:
    """Prove each region gate and opened route, then confirm both with the controller.

    Gates are proven under ``over`` and opened routes under ``under``. The real
    controller then glides from the plateau's western edge to the ruin and makes
    the strongest single climb at the foot of every gate wall.
    """
    from .fullscale_world import FACT_ORDER, SUITE_VERSION
    from .fullscale_physics import PHYSICS_VERSION
    cell = world.terrain._cell
    over = over_model(cell, char_cfg, party)
    under = under_model(cell, char_cfg)
    regions = region_patches(world)
    stages = [
        ("start", (), LOWLAND_SITES, ("plateau", "ridge", "summit_top")),
        ("ferry_open", FACT_ORDER[:1], PLATEAU_SITES, ("ridge", "summit_top")),
        ("corridor_open", FACT_ORDER[:2], RIDGE_SITES, ("summit_top",)),
        ("cap_installed", FACT_ORDER[:3], SUMMIT_SITES, ()),
    ]
    bound = climb_height_bound(char_cfg, party)
    report = dict(suite=SUITE_VERSION, physics=PHYSICS_VERSION, seed=world.seed, world_digest=world.digest(),
                  climb_bound=dict(bound.__dict__),
                  models={m.name: {k: v for k, v in m.__dict__.items()} for m in (over, under)},
                  stages=[], ok=True)
    for name, facts, sites, sealed in stages:
        o = reach(world, facts, over)
        u = reach(world, facts, under)
        stage = dict(name=name, facts=list(facts),
                     sealed={r: int((o.any & regions[r]).sum()) for r in sealed},
                     region_min_height={r: round(float(o.graph.corner_lo[regions[r]].min()), 2) for r in sealed},
                     over_max_height=round(o.max_height(), 2),
                     sites={s: u.at(world, s) for s in sites},
                     iterations=dict(over=o.iterations, under=u.iterations))
        stage["ok"] = all(v == 0 for v in stage["sealed"].values()) and all(stage["sites"].values())
        report["ok"] &= stage["ok"]
        report["stages"].append(stage)
    if not controller_checks:
        return report
    from .fullscale_routes import climb_attempt, glide, wall_feet
    # The plateau's western edge reaches the ruin: glide down, then walk.
    flight = glide(world, "west_edge", "ruin", char_cfg=char_cfg)
    on_foot = reach(world, (), under, start=flight.landing, use_lifts=False).at(world, "ruin")
    report["west_glide"] = dict(flight.__dict__, ruin_on_foot=on_foot,
                                ok=flight.landed and flight.fall_damage == 0.0 and on_foot)
    report["ok"] &= report["west_glide"]["ok"]
    # No single climb at a gate wall exceeds the bound or reaches the top.
    climbs = []
    for kind, foot, toward in wall_feet(world):
        attempt = climb_attempt(world, foot, toward, char_cfg=char_cfg, party=party)
        climbs.append(dict(wall=kind, foot=attempt.foot, gained=round(attempt.gained, 3),
                           launched=attempt.launched, top_reached=attempt.top_reached))
    report["climbs"] = dict(attempts=climbs, max_gained=max(c["gained"] for c in climbs),
                            ok=all(c["launched"] and not c["top_reached"] and c["gained"] <= bound.unmargined
                                   for c in climbs))
    report["ok"] &= report["climbs"]["ok"]
    return report
