"""Voxel block registry.

Every block type is an ``uint8`` id.  All per-block properties are also exposed
as NumPy arrays indexed by block id so that the renderer and the physics engine
can look properties up for millions of voxels at once, e.g.::

    solid_mask = SOLID[chunk]          # (X, Y, Z) bool
    rgb        = COLOR[chunk]          # (X, Y, Z, 3) uint8
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Block ids
# --------------------------------------------------------------------------
AIR = 0
STONE = 1
DIRT = 2
GRASS = 3
SAND = 4
SNOW = 5
ICE = 6
WATER = 7
LAVA = 8
LOG = 9
LEAVES = 10
TALL_GRASS = 11
ORE_IRON = 12
ORE_FLINT = 13
CRYSTAL = 14
METAL = 15
PLANK = 16
BRICK = 17
FIRE = 18
ASH = 19
CAMPFIRE = 20
COOKPOT = 21
FORGE = 22
BEACON = 23
BEACON_LIT = 24
PEDESTAL = 25
CHEST = 26
GATE = 27
MUD = 28
GRAVEL = 29
BUSH = 30
FLOWER = 31
CACTUS = 32
VINE = 33
GLASS = 34
RUNE_WALL = 35
PLATE = 36
TARGET = 37
BRIDGE = 38
RUBBER = 39
HAY = 40
CRACKED_WALL = 41
SNOW_PACK = 42
MOSS_STONE = 43
BASALT = 44
CAMPFIRE_LIT = 45
TORCH_WALL = 46
SIGNPOST = 47

NUM_BLOCKS = 48

# --------------------------------------------------------------------------
# Semantic classes (used for the segmentation observation)
# --------------------------------------------------------------------------
SEM_SKY = 0
SEM_TERRAIN = 1
SEM_ROCK = 2
SEM_WATER = 3
SEM_PLANT = 4
SEM_TREE = 5
SEM_ORE = 6
SEM_METAL = 7
SEM_STRUCTURE = 8
SEM_FIRE = 9
SEM_INTERACTIVE = 10
SEM_HAZARD = 11
SEM_MONSTER = 12
SEM_ANIMAL = 13
SEM_NPC = 14
SEM_ITEM = 15
SEM_SELF = 16
NUM_SEMANTIC = 17

SEMANTIC_NAMES = [
    "sky", "terrain", "rock", "water", "plant", "tree", "ore", "metal",
    "structure", "fire", "interactive", "hazard", "monster", "animal",
    "npc", "item", "self",
]

# Tool classes required to mine a block efficiently.
TOOL_ANY = 0
TOOL_AXE = 1
TOOL_PICKAXE = 2
TOOL_HAMMER = 3
TOOL_NONE = 4  # indestructible
TOOL_NAMES = ["any", "axe", "pickaxe", "hammer", "none"]


@dataclass
class Block:
    """Static description of one voxel type."""

    id: int
    name: str
    color: tuple[int, int, int]
    solid: bool = True
    liquid: bool = False
    opaque: bool = True
    climbable: bool = True
    flammability: float = 0.0          # 0 = fireproof, 1 = ignites instantly
    hardness: float = 1.0              # seconds of tool contact to break
    tool: int = TOOL_ANY
    conductive: bool = False           # carries lightning / shock
    emissive: float = 0.0              # self-illumination 0..1
    slippery: float = 0.0              # 0 normal ground, 1 frictionless ice
    semantic: int = SEM_TERRAIN
    drops: tuple[tuple[str, int], ...] = ()
    interactive: bool = False          # reacts to the `interact` action
    gloss: float = 0.0                 # specular highlight strength
    tags: frozenset[str] = field(default_factory=frozenset)


def _b(*args, **kw) -> Block:
    return Block(*args, **kw)


BLOCKS: list[Block] = [None] * NUM_BLOCKS  # type: ignore[list-item]


def _reg(block: Block) -> None:
    BLOCKS[block.id] = block


_reg(_b(AIR, "air", (0, 0, 0), solid=False, opaque=False, climbable=False,
        hardness=0.0, tool=TOOL_NONE, semantic=SEM_SKY))
_reg(_b(STONE, "stone", (128, 126, 122), hardness=2.0, tool=TOOL_PICKAXE,
        semantic=SEM_ROCK, drops=(("stone", 1),), tags=frozenset({"rock"})))
_reg(_b(DIRT, "dirt", (122, 92, 62), hardness=0.6, semantic=SEM_TERRAIN,
        drops=(("dirt", 1),)))
_reg(_b(GRASS, "grass", (122, 161, 79), hardness=0.6, flammability=0.25,
        semantic=SEM_TERRAIN, drops=(("dirt", 1),), tags=frozenset({"soil"})))
_reg(_b(SAND, "sand", (219, 203, 150), hardness=0.5, semantic=SEM_TERRAIN,
        drops=(("sand", 1),)))
_reg(_b(SNOW, "snow", (236, 242, 248), hardness=0.4, slippery=0.25,
        semantic=SEM_TERRAIN, drops=(("ice_shard", 1),), tags=frozenset({"cold"})))
_reg(_b(ICE, "ice", (176, 215, 232), hardness=1.0, tool=TOOL_HAMMER,
        slippery=0.9, opaque=False, semantic=SEM_TERRAIN, gloss=0.5,
        drops=(("ice_shard", 2),), tags=frozenset({"cold", "fragile"})))
_reg(_b(WATER, "water", (56, 120, 180), solid=False, liquid=True, opaque=False,
        climbable=False, hardness=0.0, tool=TOOL_NONE, conductive=True,
        semantic=SEM_WATER, gloss=0.8, tags=frozenset({"wet"})))
_reg(_b(LAVA, "lava", (226, 96, 38), solid=False, liquid=True, opaque=False,
        climbable=False, hardness=0.0, tool=TOOL_NONE, emissive=1.0,
        semantic=SEM_HAZARD, tags=frozenset({"hot", "hazard"})))
_reg(_b(LOG, "log", (112, 78, 48), hardness=1.4, tool=TOOL_AXE,
        flammability=0.5, semantic=SEM_TREE, drops=(("wood", 2),),
        tags=frozenset({"wood"})))
_reg(_b(LEAVES, "leaves", (84, 138, 66), hardness=0.2, flammability=0.9,
        opaque=False, semantic=SEM_TREE, drops=(("fiber", 1),),
        tags=frozenset({"plant"})))
_reg(_b(TALL_GRASS, "tall_grass", (150, 180, 90), solid=False, opaque=False,
        climbable=False, hardness=0.1, flammability=1.0, semantic=SEM_PLANT,
        drops=(("fiber", 1),), tags=frozenset({"plant"})))
_reg(_b(ORE_IRON, "iron_ore_vein", (146, 132, 124), hardness=3.0,
        tool=TOOL_PICKAXE, semantic=SEM_ORE, drops=(("iron_ore", 2),),
        gloss=0.2, tags=frozenset({"rock", "ore"})))
_reg(_b(ORE_FLINT, "flint_rock", (104, 104, 110), hardness=2.2,
        tool=TOOL_PICKAXE, semantic=SEM_ORE, drops=(("flint", 2),),
        tags=frozenset({"rock", "ore"})))
_reg(_b(CRYSTAL, "luminous_crystal", (152, 118, 214), hardness=2.5,
        tool=TOOL_HAMMER, emissive=0.8, opaque=False, semantic=SEM_ORE,
        drops=(("crystal", 1),), gloss=0.6, tags=frozenset({"rock", "ore"})))
_reg(_b(METAL, "metal_slab", (158, 164, 172), hardness=4.0, tool=TOOL_HAMMER,
        conductive=True, semantic=SEM_METAL, drops=(("metal_scrap", 1),),
        gloss=0.7, tags=frozenset({"metal"})))
_reg(_b(PLANK, "plank", (168, 126, 78), hardness=0.9, tool=TOOL_AXE,
        flammability=0.6, semantic=SEM_STRUCTURE, drops=(("wood", 1),),
        tags=frozenset({"wood"})))
_reg(_b(BRICK, "brick", (142, 122, 104), hardness=3.0, tool=TOOL_HAMMER,
        semantic=SEM_STRUCTURE, drops=(("stone", 1),), tags=frozenset({"rock"})))
_reg(_b(FIRE, "fire", (248, 166, 54), solid=False, opaque=False,
        climbable=False, hardness=0.0, tool=TOOL_NONE, emissive=1.0,
        semantic=SEM_FIRE, tags=frozenset({"hot", "hazard"})))
_reg(_b(ASH, "ash", (84, 78, 74), hardness=0.3, semantic=SEM_TERRAIN,
        drops=(("charcoal", 1),)))
_reg(_b(CAMPFIRE, "campfire", (96, 72, 54), hardness=0.4, flammability=0.8,
        opaque=False, semantic=SEM_INTERACTIVE, interactive=True,
        drops=(("wood", 1),), tags=frozenset({"wood", "ignitable"})))
_reg(_b(COOKPOT, "cooking_pot", (74, 74, 80), hardness=2.0, tool=TOOL_HAMMER,
        opaque=False, conductive=True, semantic=SEM_INTERACTIVE,
        interactive=True, gloss=0.5, tags=frozenset({"metal", "station"})))
_reg(_b(FORGE, "forge", (122, 92, 86), hardness=3.0, tool=TOOL_HAMMER,
        emissive=0.5, semantic=SEM_INTERACTIVE, interactive=True,
        tags=frozenset({"station", "hot"})))
_reg(_b(BEACON, "beacon", (108, 102, 96), hardness=0.0, tool=TOOL_NONE,
        opaque=False, semantic=SEM_INTERACTIVE, interactive=True,
        tags=frozenset({"ignitable", "objective"})))
_reg(_b(BEACON_LIT, "beacon_lit", (255, 196, 96), hardness=0.0, tool=TOOL_NONE,
        opaque=False, emissive=1.0, semantic=SEM_FIRE, interactive=True,
        tags=frozenset({"objective"})))
_reg(_b(PEDESTAL, "pedestal", (178, 172, 160), hardness=0.0, tool=TOOL_NONE,
        semantic=SEM_INTERACTIVE, interactive=True, gloss=0.3,
        tags=frozenset({"station", "objective"})))
_reg(_b(CHEST, "chest", (158, 112, 52), hardness=1.0, tool=TOOL_AXE,
        opaque=False, semantic=SEM_INTERACTIVE, interactive=True,
        tags=frozenset({"wood", "container"})))
_reg(_b(GATE, "gate", (120, 110, 96), hardness=0.0, tool=TOOL_NONE,
        semantic=SEM_STRUCTURE, interactive=True, tags=frozenset({"door"})))
_reg(_b(MUD, "mud", (96, 80, 58), hardness=0.4, semantic=SEM_TERRAIN,
        drops=(("dirt", 1),), tags=frozenset({"wet"})))
_reg(_b(GRAVEL, "gravel", (134, 130, 126), hardness=0.8, semantic=SEM_ROCK,
        drops=(("stone", 1), ("flint", 1)), tags=frozenset({"rock"})))
_reg(_b(BUSH, "berry_bush", (72, 120, 72), solid=False, opaque=False,
        climbable=False, hardness=0.15, flammability=0.8, semantic=SEM_PLANT,
        drops=(("berry", 2), ("fiber", 1)), tags=frozenset({"plant"})))
_reg(_b(FLOWER, "flower", (210, 134, 170), solid=False, opaque=False,
        climbable=False, hardness=0.05, flammability=0.9, semantic=SEM_PLANT,
        drops=(("petal", 1),), tags=frozenset({"plant"})))
_reg(_b(CACTUS, "cactus", (96, 142, 92), hardness=0.5, tool=TOOL_AXE,
        semantic=SEM_PLANT, drops=(("cactus_fruit", 1),),
        tags=frozenset({"plant", "spiky"})))
_reg(_b(VINE, "vine", (76, 124, 70), solid=False, opaque=False, climbable=True,
        hardness=0.1, flammability=0.8, semantic=SEM_PLANT,
        drops=(("fiber", 2),), tags=frozenset({"plant", "ladder"})))
_reg(_b(GLASS, "glass", (206, 228, 236), hardness=0.2, tool=TOOL_HAMMER,
        opaque=False, semantic=SEM_STRUCTURE, gloss=0.8,
        tags=frozenset({"fragile"})))
_reg(_b(RUNE_WALL, "rune_wall", (92, 104, 126), hardness=0.0, tool=TOOL_NONE,
        emissive=0.15, semantic=SEM_STRUCTURE, tags=frozenset({"indestructible"})))
_reg(_b(PLATE, "pressure_plate", (168, 158, 120), solid=True, hardness=0.0,
        tool=TOOL_NONE, opaque=False, semantic=SEM_INTERACTIVE,
        tags=frozenset({"objective", "sensor"})))
_reg(_b(TARGET, "target", (206, 92, 80), hardness=0.5, tool=TOOL_AXE,
        opaque=False, semantic=SEM_INTERACTIVE, drops=(("wood", 1),),
        tags=frozenset({"objective", "shootable"})))
_reg(_b(BRIDGE, "bridge_plank", (150, 116, 76), hardness=0.8, tool=TOOL_AXE,
        flammability=0.5, semantic=SEM_STRUCTURE, drops=(("wood", 1),),
        tags=frozenset({"wood"})))
_reg(_b(RUBBER, "insulator", (56, 52, 56), hardness=1.2, semantic=SEM_STRUCTURE,
        tags=frozenset({"insulator"})))
_reg(_b(HAY, "hay", (206, 178, 92), hardness=0.3, flammability=1.0,
        semantic=SEM_PLANT, drops=(("fiber", 2),), tags=frozenset({"plant"})))
_reg(_b(CRACKED_WALL, "cracked_wall", (132, 118, 110), hardness=6.0,
        tool=TOOL_HAMMER, semantic=SEM_STRUCTURE, drops=(("stone", 2),),
        tags=frozenset({"rock", "blastable"})))
_reg(_b(SNOW_PACK, "packed_snow", (214, 226, 238), hardness=0.8,
        semantic=SEM_TERRAIN, drops=(("ice_shard", 1),),
        tags=frozenset({"cold"})))
_reg(_b(MOSS_STONE, "moss_stone", (108, 122, 96), hardness=2.0,
        tool=TOOL_PICKAXE, semantic=SEM_ROCK, drops=(("stone", 1),),
        tags=frozenset({"rock"})))
_reg(_b(BASALT, "basalt", (72, 68, 72), hardness=2.6, tool=TOOL_PICKAXE,
        semantic=SEM_ROCK, drops=(("stone", 1),), tags=frozenset({"rock", "hot"})))
_reg(_b(CAMPFIRE_LIT, "campfire_lit", (242, 148, 58), solid=False, opaque=False,
        climbable=False, hardness=0.4, emissive=1.0, semantic=SEM_FIRE,
        interactive=True, tags=frozenset({"hot", "station"})))
_reg(_b(TORCH_WALL, "wall_torch", (240, 180, 96), solid=False, opaque=False,
        climbable=False, hardness=0.1, emissive=0.9, semantic=SEM_FIRE,
        drops=(("torch", 1),), tags=frozenset({"hot"})))
_reg(_b(SIGNPOST, "signpost", (146, 118, 84), solid=False, opaque=False,
        climbable=False, hardness=0.4, tool=TOOL_AXE, flammability=0.5,
        semantic=SEM_INTERACTIVE, interactive=True, drops=(("wood", 1),),
        tags=frozenset({"wood", "info"})))

assert all(b is not None for b in BLOCKS), "unregistered block id"
BY_NAME = {b.name: b.id for b in BLOCKS}

# --------------------------------------------------------------------------
# Vectorised property tables
# --------------------------------------------------------------------------
SOLID = np.array([b.solid for b in BLOCKS], dtype=bool)
LIQUID = np.array([b.liquid for b in BLOCKS], dtype=bool)
OPAQUE = np.array([b.opaque for b in BLOCKS], dtype=bool)
CLIMBABLE = np.array([b.climbable for b in BLOCKS], dtype=bool)
FLAMMABILITY = np.array([b.flammability for b in BLOCKS], dtype=np.float32)
HARDNESS = np.array([b.hardness for b in BLOCKS], dtype=np.float32)
TOOL = np.array([b.tool for b in BLOCKS], dtype=np.uint8)
CONDUCTIVE = np.array([b.conductive for b in BLOCKS], dtype=bool)
EMISSIVE = np.array([b.emissive for b in BLOCKS], dtype=np.float32)
SLIPPERY = np.array([b.slippery for b in BLOCKS], dtype=np.float32)
SEMANTIC = np.array([b.semantic for b in BLOCKS], dtype=np.uint8)
COLOR = np.array([b.color for b in BLOCKS], dtype=np.uint8)
GLOSS = np.array([b.gloss for b in BLOCKS], dtype=np.float32)
INTERACTIVE = np.array([b.interactive for b in BLOCKS], dtype=bool)
INDESTRUCTIBLE = TOOL == TOOL_NONE

#: blocks that stop a ray (water is handled separately by the renderer)
VIEW_BLOCKING = OPAQUE.copy()

#: blocks a walking agent may stand on
STANDABLE = SOLID.copy()

HOT = np.array(["hot" in b.tags for b in BLOCKS], dtype=bool)
COLD = np.array(["cold" in b.tags for b in BLOCKS], dtype=bool)
WET = np.array(["wet" in b.tags for b in BLOCKS], dtype=bool)
BLASTABLE = np.array(["blastable" in b.tags for b in BLOCKS], dtype=bool)
IGNITABLE = np.array(["ignitable" in b.tags for b in BLOCKS], dtype=bool)


def name_of(block_id: int) -> str:
    return BLOCKS[int(block_id)].name


def drops_of(block_id: int) -> tuple[tuple[str, int], ...]:
    return BLOCKS[int(block_id)].drops


def tool_for(block_id: int) -> str:
    return TOOL_NAMES[int(TOOL[int(block_id)])]
