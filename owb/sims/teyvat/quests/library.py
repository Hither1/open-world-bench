"""The built-in questline: "The Windborne Ledger".

Six gated quests that together span the whole island and require every
traversal skill the character has: running, climbing, gliding, swimming,
combat, and being in the right place at the right time of day.

`build_questline` both *places* the entities a quest needs (using the terrain
to find sensible sites) and returns the matching :class:`QuestGraph`, so quest
design and world layout cannot drift apart.
"""
from __future__ import annotations

import numpy as np

from ..terrain import FOREST, MEADOW
from ..elements import Element, ElementalStatus
from ..world import Entity, EntityKind, World
from .schema import ObjKind, Objective, Quest, QuestGraph, Stage


def _ring(center: np.ndarray, radius: float, count: int, phase: float = 0.0) -> list[tuple[float, float]]:
    out = []
    for i in range(count):
        a = phase + 2.0 * np.pi * i / count
        out.append((float(center[0] + np.cos(a) * radius), float(center[2] + np.sin(a) * radius)))
    return out


def build_questline(world: World, rng: np.random.Generator | None = None) -> tuple[QuestGraph, np.ndarray]:
    """Populate `world` with quest entities. Returns (graph, player_spawn)."""
    rng = rng or np.random.default_rng(world.cfg.seed)
    T = world.terrain

    # ------------------------------------------------------------ the village
    village = T.sample_land(rng, 1, min_height=8.0, max_height=55.0, max_slope_deg=11.0)[0]
    vx, vz = float(village[0]), float(village[1])
    spawn = world.ground_pos(vx + 4.0, vz + 4.0)

    elder = world.spawn_at(
        "elder_mei", EntityKind.NPC, vx, vz, radius=3.2, label="Elder Mei",
        dialogue="The wind has gone quiet, traveller. Bring me three windwheel asters "
                 "and I will tell you what the ledger says.",
    )
    world.spawn_at("statue_windrest", EntityKind.WAYPOINT, vx + 7.0, vz - 5.0,
                   radius=3.0, label="Statue of the Seven")
    scholar = world.spawn_at(
        "scholar_arel", EntityKind.NPC, vx - 9.0, vz + 6.0, radius=3.0, label="Scholar Arel",
        dialogue="Four pillars stand in the old ruins. Touch them in the order the "
                 "mural shows: east, north, west, south.",
    )

    # ---------------------------------------------- Q1: asters around the vale
    aster_sites = T.sample_land(rng, 3, min_height=5.0, max_height=70.0, max_slope_deg=26.0,
                                radius=130.0, center=(vx, vz))
    for i, (ax, az) in enumerate(aster_sites):
        world.spawn_at(f"aster_{i}", EntityKind.COLLECTIBLE, float(ax), float(az),
                       lift=0.35, radius=2.4, item="windwheel_aster",
                       label=f"Windwheel Aster {i + 1}")

    # ------------------------------------------------ Q2: three wind beacons
    summit = T.highest_point()
    beacon_sites = []
    for i, (bx, bz) in enumerate(_ring(summit, 150.0, 3, phase=0.4)):
        p = T.sample_land(rng, 1, min_height=55.0, max_slope_deg=34.0, radius=55.0, center=(bx, bz))[0]
        beacon_sites.append(p)
        world.spawn_at(f"beacon_{i}", EntityKind.BEACON, float(p[0]), float(p[1]),
                       lift=0.0, radius=3.0, label=f"Wind Beacon {'ABC'[i]}")
        # Each beacon is guarded.
        gx, gz = float(p[0]) + 6.0, float(p[1]) + 4.0
        world.spawn_at(f"shardling_b{i}", EntityKind.ENEMY, gx, gz, hp=100.0, damage=8.0,
                       aggro_radius=16.0, label=f"Shardling Sentry {i + 1}",
                       elemental=ElementalStatus(
                           shield=40.0 if world.party else 0.0,
                           shield_element=(Element.EMBER, Element.TIDE, Element.FROST)[i],
                           resists=((Element.EMBER, Element.TIDE, Element.FROST)[i],),
                       ))

    # ------------------------------------------------- Q3: the sunken ledger
    dive_site = _find_dive_site(world, rng, near=(vx, vz))
    world.spawn(Entity(
        id="sunken_chest", kind=EntityKind.CHEST,
        pos=np.array([dive_site[0], dive_site[1] + 0.6, dive_site[2]], dtype=np.float32),
        radius=3.4, item="ledger_fragment", label="Barnacled Chest",
    ))
    for i, (ex, ez) in enumerate(_ring(np.array([dive_site[0], 0.0, dive_site[2]]), 22.0, 2)):
        p = T.sample_land(rng, 1, min_height=1.0, max_slope_deg=35.0, radius=30.0, center=(ex, ez))[0]
        world.spawn_at(f"shardling_w{i}", EntityKind.ENEMY, float(p[0]), float(p[1]),
                       hp=100.0, damage=10.0, label=f"Tidewatch Shardling {i + 1}")

    # ---------------------------------------------------- Q4: the first light
    altar = world.ground_pos(float(summit[0]) + 2.0, float(summit[2]) + 2.0)
    world.spawn(Entity(id="peak_altar", kind=EntityKind.ALTAR, pos=altar, radius=4.0,
                       label="Altar of First Light"))

    # ------------------------------------------------------- Q5: ruin pillars
    ruin_center = T.sample_land(rng, 1, min_height=12.0, max_height=85.0, max_slope_deg=14.0,
                                radius=240.0, center=(vx, vz))[0]
    rc = np.array([ruin_center[0], 0.0, ruin_center[1]], dtype=np.float32)
    pillar_order = ["pillar_east", "pillar_north", "pillar_west", "pillar_south"]
    offsets = {"pillar_east": (14.0, 0.0), "pillar_north": (0.0, -14.0),
               "pillar_west": (-14.0, 0.0), "pillar_south": (0.0, 14.0)}
    for pid, (dx, dz) in offsets.items():
        world.spawn_at(pid, EntityKind.PILLAR, float(rc[0] + dx), float(rc[2] + dz),
                       radius=3.0, label=pid.replace("pillar_", "").title() + " Pillar")
    chest_sites = _ring(rc, 26.0, 4, phase=0.78)
    for i, (cx, cz) in enumerate(chest_sites):
        world.spawn_at(f"ruin_chest_{i}", EntityKind.CHEST, float(cx), float(cz),
                       radius=3.0, item="relic_shard", label=f"Ruin Chest {i + 1}")

    # --------------------------------------------------------- Q6: the pact
    boss_pos = world.ground_pos(float(summit[0]) - 6.0, float(summit[2]) - 6.0)
    world.spawn(Entity(id="stormcaller", kind=EntityKind.BOSS, pos=boss_pos, hp=340.0,
                       damage=16.0, aggro_radius=26.0, label="The Stormcaller"))

    # ------------------------------------------------------------ the graph
    def pos3(e_id: str) -> tuple[float, float, float]:
        return tuple(float(v) for v in world.get(e_id).pos)  # type: ignore[return-value]

    q1 = Quest(
        id="awakening", title="Awakening at Windrest",
        summary="Elder Mei wants three windwheel asters before she will speak of the ledger.",
        stages=[
            Stage([Objective("a_talk", ObjKind.TALK, "Speak with Elder Mei in Windrest",
                             targets=("elder_mei",), position=pos3("elder_mei"), reward=1.0)]),
            Stage([Objective("a_collect", ObjKind.COLLECT, "Gather 3 windwheel asters around the vale",
                             item="windwheel_aster", count=3, position=pos3("aster_0"), reward=3.0)],
                  note="Asters grow in the open meadows nearby."),
            Stage([Objective("a_return", ObjKind.DELIVER, "Bring the asters back to Elder Mei",
                             targets=("elder_mei",), item="windwheel_aster", count=3,
                             position=pos3("elder_mei"), reward=2.0)]),
        ],
        reward=6.0, grants=("ledger_cover",),
    )

    q2 = Quest(
        id="beacons", title="Beacons of the Vale",
        summary="Three wind beacons ring the mountain. Light them all to wake the wind.",
        prerequisites=("awakening",),
        stages=[
            Stage([Objective("b_clear", ObjKind.DEFEAT, "Drive off the shardling sentries",
                             targets=tuple(f"shardling_b{i}" for i in range(3)), count=2,
                             position=pos3("shardling_b0"), reward=3.0)]),
            Stage([Objective("b_light", ObjKind.ACTIVATE, "Light the three wind beacons",
                             targets=("beacon_0", "beacon_1", "beacon_2"), count=3,
                             position=pos3("beacon_0"), reward=6.0)],
                  note="The beacons sit high: climb, or glide between them."),
        ],
        reward=8.0, grants=("wind_token",),
    )

    q3 = Quest(
        id="sunken", title="The Sunken Ledger",
        summary="A barnacled chest rests on the lakebed. The ledger's first pages are inside.",
        prerequisites=("awakening",),
        stages=[
            Stage([Objective("s_dive", ObjKind.DIVE, "Dive to the lakebed where the chest rests",
                             position=(float(dive_site[0]), float(dive_site[1]) + 1.2, float(dive_site[2])),
                             radius=7.0, reward=3.0)]),
            Stage([Objective("s_chest", ObjKind.OPEN_CHEST, "Open the barnacled chest",
                             targets=("sunken_chest",), position=pos3("sunken_chest"), reward=4.0)]),
            Stage([Objective("s_guards", ObjKind.DEFEAT, "Deal with the tidewatch shardlings",
                             targets=("shardling_w0", "shardling_w1"), count=2,
                             position=pos3("shardling_w0"), reward=3.0, optional=True)]),
        ],
        reward=8.0,
    )

    q4 = Quest(
        id="first_light", title="Peak of the First Light",
        summary="Stand on the summit altar at dawn and make the offering.",
        prerequisites=("beacons",),
        stages=[
            Stage([Objective("f_climb", ObjKind.ALTITUDE, "Reach the summit of Mount Aether at dawn",
                             min_height=float(summit[1]) - 12.0, position=pos3("peak_altar"),
                             radius=40.0, time_window=(0.18, 0.40), reward=6.0)],
                  note="Dawn is brief -- wait for it, or come back tomorrow."),
            Stage([Objective("f_offer", ObjKind.ACTIVATE, "Make the offering at the Altar of First Light",
                             targets=("peak_altar",), position=pos3("peak_altar"), reward=4.0)]),
        ],
        reward=10.0, grants=("dawn_sigil",),
    )

    q5 = Quest(
        id="ruins", title="Echoes in the Ruins",
        summary="Scholar Arel's mural names an order for the four pillars.",
        prerequisites=("sunken",),
        stages=[
            Stage([Objective("r_talk", ObjKind.TALK, "Ask Scholar Arel about the ruins",
                             targets=("scholar_arel",), position=pos3("scholar_arel"), reward=1.0)]),
            Stage([Objective("r_chests", ObjKind.OPEN_CHEST, "Search 3 of the 4 ruin chests",
                             targets=tuple(f"ruin_chest_{i}" for i in range(4)), count=3,
                             position=pos3("ruin_chest_0"), reward=4.0)]),
            Stage([Objective("r_seq", ObjKind.SEQUENCE,
                             "Touch the pillars in order: east, north, west, south",
                             targets=tuple(pillar_order), position=pos3("pillar_east"), reward=8.0)],
                  note="Touching the wrong pillar dims them all and resets the order."),
        ],
        reward=12.0, grants=("ledger_spine",),
    )

    q6 = Quest(
        id="pact", title="The Windborne Pact",
        summary="The Stormcaller holds the last page. End it, then bring the ledger home.",
        prerequisites=("first_light", "ruins"),
        stages=[
            Stage([Objective("p_boss", ObjKind.DEFEAT, "Defeat the Stormcaller at the summit",
                             targets=("stormcaller",), position=pos3("stormcaller"), reward=12.0)]),
            Stage([Objective("p_deliver", ObjKind.DELIVER, "Return the ledger fragment to Elder Mei",
                             targets=("elder_mei",), item="ledger_fragment", count=1,
                             position=pos3("elder_mei"), reward=8.0)]),
        ],
        reward=20.0,
    )

    del elder, scholar, beacon_sites, chest_sites
    return QuestGraph([q1, q2, q3, q4, q5, q6]), spawn


def _find_dive_site(world: World, rng: np.random.Generator, near: tuple[float, float]) -> np.ndarray:
    """A submerged spot deep enough to require diving, as close to home as possible."""
    T = world.terrain
    h = T.height
    half = T.cfg.size * 0.5
    cell = T.cfg.size / (T.cfg.resolution - 1)
    deep = (h < -3.5) & (h > -9.0)
    if not deep.any():
        deep = h < min(-1.0, float(h.min()) + 1.0)
    ij = np.argwhere(deep)
    xs = -half + ij[:, 0] * cell
    zs = -half + ij[:, 1] * cell
    d = (xs - near[0]) ** 2 + (zs - near[1]) ** 2
    k = int(np.argmin(d))
    return np.array([xs[k], float(h[ij[k, 0], ij[k, 1]]), zs[k]], dtype=np.float32)
