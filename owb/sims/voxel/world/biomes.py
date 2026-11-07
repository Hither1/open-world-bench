"""Biome table.

A biome bundles the surface materials, flora, fauna and climate feel of a
region.  Biomes are assigned from (temperature, moisture, elevation, slope) in
:mod:`owb.sims.voxel.world.worldgen` and drive both generation and gameplay (cold damage
in the snowfield, heat damage in the volcanic waste, which ingredients grow
where, which monsters camp there).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import blocks as B

PLAINS = 0
FOREST = 1
HILLS = 2
MOUNTAIN = 3
SNOWFIELD = 4
DESERT = 5
SAVANNA = 6
SWAMP = 7
BEACH = 8
LAKE = 9
VOLCANIC = 10
CANYON = 11
NUM_BIOMES = 12


@dataclass
class Biome:
    id: int
    name: str
    surface: int
    subsurface: int
    #: (block, per-column probability) ground cover
    cover: tuple[tuple[int, float], ...] = ()
    tree_kind: str = "none"          # none | oak | pine | palm | dead | mushroom
    tree_density: float = 0.0
    #: ingredient plants: (item id, relative weight)
    flora: tuple[tuple[str, float], ...] = ()
    #: monsters that camp here: (kind, relative weight)
    monsters: tuple[tuple[str, float], ...] = ()
    animals: tuple[tuple[str, float], ...] = ()
    #: ambient temperature offset in "comfort" units (-1 freezing, +1 scorching)
    climate: float = 0.0
    ore: tuple[tuple[int, float], ...] = ()
    map_color: tuple[int, int, int] = (120, 140, 100)
    tags: frozenset[str] = field(default_factory=frozenset)


BIOMES: list[Biome] = [None] * NUM_BIOMES  # type: ignore[list-item]


def _reg(b: Biome) -> None:
    BIOMES[b.id] = b


_reg(Biome(PLAINS, "grassland", B.GRASS, B.DIRT,
           cover=((B.TALL_GRASS, 0.22), (B.FLOWER, 0.03), (B.BUSH, 0.02)),
           tree_kind="oak", tree_density=0.004,
           flora=(("apple", 1.0), ("stambulb", 1.0), ("hearty_radish", 0.5),
                  ("swift_violet", 0.4), ("mighty_thistle", 0.5)),
           monsters=(("bokoblin", 1.0), ("chuchu", 0.6)),
           animals=(("boar", 1.0), ("bird", 0.8)),
           ore=((B.ORE_FLINT, 0.4),),
           map_color=(126, 168, 92), tags=frozenset({"temperate"})))

_reg(Biome(FOREST, "woodland", B.GRASS, B.DIRT,
           cover=((B.TALL_GRASS, 0.30), (B.BUSH, 0.08), (B.FLOWER, 0.04)),
           tree_kind="oak", tree_density=0.055,
           flora=(("mushroom", 1.2), ("apple", 1.0), ("berry", 1.0),
                  ("endura_carrot", 0.35), ("armoranth", 0.5)),
           monsters=(("bokoblin", 1.0), ("chuchu", 0.8), ("moblin", 0.25)),
           animals=(("deer", 1.0), ("bird", 1.0), ("boar", 0.8)),
           ore=((B.ORE_FLINT, 0.3),),
           map_color=(74, 126, 72), tags=frozenset({"temperate", "wooded"})))

_reg(Biome(HILLS, "highland", B.GRASS, B.DIRT,
           cover=((B.TALL_GRASS, 0.16), (B.FLOWER, 0.02)),
           tree_kind="pine", tree_density=0.012,
           flora=(("stambulb", 1.0), ("mighty_thistle", 0.8), ("berry", 0.6)),
           monsters=(("bokoblin", 1.0), ("moblin", 0.4), ("stalker", 0.2)),
           animals=(("goat", 1.0), ("bird", 0.6)),
           ore=((B.ORE_IRON, 0.8), (B.ORE_FLINT, 0.6)),
           climate=-0.1, map_color=(108, 148, 96), tags=frozenset({"temperate"})))

_reg(Biome(MOUNTAIN, "crags", B.STONE, B.STONE,
           cover=((B.GRAVEL, 0.10),),
           tree_kind="pine", tree_density=0.002,
           flora=(("mighty_thistle", 0.6),),
           monsters=(("stalker", 0.8), ("moblin", 0.4)),
           animals=(("goat", 0.8),),
           ore=((B.ORE_IRON, 1.6), (B.CRYSTAL, 0.5), (B.ORE_FLINT, 0.8)),
           climate=-0.35, map_color=(140, 138, 132),
           tags=frozenset({"rocky", "steep"})))

_reg(Biome(SNOWFIELD, "frostpeak", B.SNOW, B.SNOW_PACK,
           cover=((B.SNOW_PACK, 0.05),),
           tree_kind="pine", tree_density=0.010,
           flora=(("chillshroom", 1.0), ("spicy_pepper", 0.2)),
           monsters=(("ice_chuchu", 1.0), ("stalker", 0.5)),
           animals=(("goat", 0.5),),
           ore=((B.ORE_IRON, 0.8), (B.CRYSTAL, 0.7)),
           climate=-1.0, map_color=(232, 240, 248),
           tags=frozenset({"cold", "hazardous"})))

_reg(Biome(DESERT, "sunwaste", B.SAND, B.SAND,
           cover=((B.CACTUS, 0.012), (B.HAY, 0.02)),
           tree_kind="none", tree_density=0.0,
           flora=(("voltfruit", 1.0), ("cactus_fruit", 1.0), ("salt", 0.6)),
           monsters=(("electro_chuchu", 1.0), ("bokoblin", 0.7)),
           animals=(("bird", 0.3),),
           ore=((B.ORE_FLINT, 0.8),),
           climate=0.8, map_color=(226, 206, 148),
           tags=frozenset({"hot", "arid"})))

_reg(Biome(SAVANNA, "drylands", B.GRASS, B.DIRT,
           cover=((B.HAY, 0.20), (B.TALL_GRASS, 0.10)),
           tree_kind="dead", tree_density=0.008,
           flora=(("spicy_pepper", 1.0), ("mighty_thistle", 0.6), ("salt", 0.3)),
           monsters=(("bokoblin", 1.0), ("moblin", 0.3)),
           animals=(("boar", 0.8), ("bird", 0.5)),
           ore=((B.ORE_FLINT, 0.6),),
           climate=0.35, map_color=(192, 184, 110), tags=frozenset({"dry"})))

_reg(Biome(SWAMP, "mire", B.MUD, B.DIRT,
           cover=((B.TALL_GRASS, 0.24), (B.BUSH, 0.05)),
           tree_kind="mushroom", tree_density=0.02,
           flora=(("mushroom", 1.4), ("chillshroom", 0.5), ("berry", 0.6)),
           monsters=(("chuchu", 1.2), ("electro_chuchu", 0.4), ("bokoblin", 0.5)),
           animals=(("frog", 1.0), ("bird", 0.4)),
           ore=(),
           climate=0.1, map_color=(96, 110, 78), tags=frozenset({"wet"})))

_reg(Biome(BEACH, "shore", B.SAND, B.SAND,
           cover=((B.HAY, 0.02),),
           tree_kind="palm", tree_density=0.010,
           flora=(("cactus_fruit", 0.3), ("salt", 1.0)),
           monsters=(("chuchu", 0.5),),
           animals=(("bird", 0.8), ("crab", 1.0)),
           ore=(),
           climate=0.15, map_color=(226, 212, 162), tags=frozenset({"coastal"})))

_reg(Biome(LAKE, "waters", B.SAND, B.DIRT,
           cover=(), tree_kind="none", tree_density=0.0,
           flora=(("salt", 0.4),),
           monsters=(), animals=(("fish", 1.4),),
           ore=(), climate=0.0, map_color=(72, 132, 190),
           tags=frozenset({"water"})))

_reg(Biome(VOLCANIC, "cinderfell", B.BASALT, B.BASALT,
           cover=((B.ASH, 0.12), (B.GRAVEL, 0.06)),
           tree_kind="dead", tree_density=0.002,
           flora=(("spicy_pepper", 0.8), ("chillshroom", 0.6)),
           monsters=(("fire_chuchu", 1.2), ("stalker", 0.4)),
           animals=(),
           ore=((B.ORE_IRON, 1.4), (B.CRYSTAL, 0.8)),
           climate=1.0, map_color=(96, 76, 72),
           tags=frozenset({"hot", "hazardous"})))

_reg(Biome(CANYON, "redcut", B.GRAVEL, B.STONE,
           cover=((B.HAY, 0.06), (B.GRAVEL, 0.10)),
           tree_kind="dead", tree_density=0.004,
           flora=(("spicy_pepper", 0.6), ("salt", 0.4)),
           monsters=(("bokoblin", 0.8), ("stalker", 0.6)),
           animals=(("goat", 0.5),),
           ore=((B.ORE_IRON, 1.0), (B.ORE_FLINT, 1.0)),
           climate=0.3, map_color=(166, 124, 96),
           tags=frozenset({"rocky", "steep"})))

assert all(b is not None for b in BIOMES)
BIOME_NAMES = [b.name for b in BIOMES]
BIOME_BY_NAME = {b.name: b.id for b in BIOMES}
CLIMATE = np.array([b.climate for b in BIOMES], dtype=np.float32)
MAP_COLOR = np.array([b.map_color for b in BIOMES], dtype=np.uint8)
SURFACE = np.array([b.surface for b in BIOMES], dtype=np.uint8)
SUBSURFACE = np.array([b.subsurface for b in BIOMES], dtype=np.uint8)
COLD_BIOME = np.array([("cold" in b.tags) for b in BIOMES], dtype=bool)
HOT_BIOME = np.array([("hot" in b.tags) for b in BIOMES], dtype=bool)


def get(biome_id: int) -> Biome:
    return BIOMES[int(biome_id)]


def name_of(biome_id: int) -> str:
    return BIOMES[int(biome_id)].name


def region_names() -> dict[int, str]:
    return {b.id: b.name for b in BIOMES}
