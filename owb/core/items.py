"""Item registry: materials, food, tools, weapons, armour and rune gadgets.

Items are referenced by string id.  A concrete carried item is an
:class:`~owb.sim.inventory.ItemStack`, which may carry instance data (remaining
durability, cooked-dish effects, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Status effects
# --------------------------------------------------------------------------
EFF_HEAL = "heal"                 # instantaneous
EFF_STAMINA = "stamina"           # instantaneous
EFF_COLD_RESIST = "cold_resist"
EFF_HEAT_RESIST = "heat_resist"
EFF_SHOCK_RESIST = "shock_resist"
EFF_ATTACK_UP = "attack_up"
EFF_DEFENSE_UP = "defense_up"
EFF_SPEED_UP = "speed_up"
EFF_ENDURA = "endura"             # temporary bonus stamina
EFF_STEALTH = "stealth"

TIMED_EFFECTS = (
    EFF_COLD_RESIST, EFF_HEAT_RESIST, EFF_SHOCK_RESIST, EFF_ATTACK_UP,
    EFF_DEFENSE_UP, EFF_SPEED_UP, EFF_ENDURA, EFF_STEALTH,
)
ALL_EFFECTS = (EFF_HEAL, EFF_STAMINA) + TIMED_EFFECTS

# --------------------------------------------------------------------------
# Item kinds
# --------------------------------------------------------------------------
KIND_MATERIAL = "material"
KIND_FOOD = "food"
KIND_DISH = "dish"
KIND_WEAPON = "weapon"
KIND_TOOL = "tool"
KIND_BOW = "bow"
KIND_AMMO = "ammo"
KIND_SHIELD = "shield"
KIND_ARMOR = "armor"
KIND_GADGET = "gadget"
KIND_KEY = "key"
KIND_PLACEABLE = "placeable"

EQUIPPABLE = {KIND_WEAPON, KIND_TOOL, KIND_BOW, KIND_SHIELD, KIND_ARMOR}


@dataclass
class Item:
    id: str
    name: str
    kind: str
    stack: int = 99
    #: melee / projectile damage
    attack: float = 0.0
    #: uses before the item breaks (``0`` = indestructible)
    durability: int = 0
    #: tool class this item acts as: "axe" / "pickaxe" / "hammer" / "" .
    tool_class: str = ""
    #: block placed by the ``place`` action
    places: int | None = None
    #: instant + timed effects applied on ``eat``
    effects: dict[str, float] = field(default_factory=dict)
    #: duration in seconds for the timed effects above
    duration: float = 0.0
    #: free-form tags, used by the cooking system and by elemental logic
    tags: frozenset[str] = frozenset()
    #: cold / heat protection while worn (armour only)
    cold_resist: float = 0.0
    heat_resist: float = 0.0
    defense: float = 0.0
    weight: float = 1.0
    value: int = 1
    desc: str = ""


ITEMS: dict[str, Item] = {}


def _reg(item: Item) -> Item:
    ITEMS[item.id] = item
    return item


def _m(id_, name, *, tags=(), value=1, desc="", stack=99, places=None):
    return _reg(Item(id_, name, KIND_MATERIAL, stack=stack, tags=frozenset(tags),
                     value=value, desc=desc, places=places))


# ---------------------------- materials -----------------------------------
_m("wood", "wood", tags=("wood", "fuel", "flammable"), desc="A sturdy length of timber.")
_m("stone", "stone", tags=("rock",), places=1)
_m("dirt", "dirt", tags=("soil",), places=2)
_m("sand", "sand", tags=("soil",), places=4)
_m("flint", "flint", tags=("rock", "firestarter"), desc="Strike it to make sparks.")
_m("iron_ore", "iron ore", tags=("ore", "metal"))
_m("iron_ingot", "iron ingot", tags=("metal", "ingot"), value=6)
_m("metal_scrap", "metal scrap", tags=("metal",), places=15)
_m("crystal", "luminous crystal", tags=("ore", "glow"), value=8)
_m("charcoal", "charcoal", tags=("fuel",))
_m("fiber", "plant fiber", tags=("plant", "flammable"))
_m("ice_shard", "ice shard", tags=("cold",))
_m("feather", "feather", tags=("animal",))
_m("monster_horn", "monster horn", tags=("monster",), value=3)
_m("monster_fang", "monster fang", tags=("monster",), value=3)
_m("monster_guts", "monster guts", tags=("monster",), value=4)
_m("electro_gland", "electro gland", tags=("monster", "electric"), value=6)
_m("fire_gland", "fire gland", tags=("monster", "fire"), value=6)
_m("ice_gland", "frost gland", tags=("monster", "ice"), value=6)
_m("rope", "rope", tags=("plant",))
_m("petal", "flower petal", tags=("plant",))
_m("hide", "animal hide", tags=("animal",), value=3)
_m("plank_bundle", "plank bundle", tags=("wood",), places=16)
_m("bridge_kit", "bridge kit", tags=("wood",), places=38)

# ---------------------------- raw food ------------------------------------
_reg(Item("apple", "apple", KIND_FOOD, effects={EFF_HEAL: 2.0},
          tags=frozenset({"fruit"}), desc="Common roadside fruit."))
_reg(Item("berry", "wildberry", KIND_FOOD, effects={EFF_HEAL: 1.0},
          tags=frozenset({"fruit"})))
_reg(Item("mushroom", "mushroom", KIND_FOOD, effects={EFF_HEAL: 1.5},
          tags=frozenset({"mushroom"})))
_reg(Item("hearty_radish", "hearty radish", KIND_FOOD, effects={EFF_HEAL: 6.0},
          tags=frozenset({"vegetable", "hearty"}), value=8))
_reg(Item("spicy_pepper", "spicy pepper", KIND_FOOD,
          effects={EFF_HEAL: 1.0, EFF_COLD_RESIST: 1.0}, duration=90.0,
          tags=frozenset({"vegetable", "spicy"}),
          desc="Warms you from the inside. Key to surviving the snowfield."))
_reg(Item("chillshroom", "chillshroom", KIND_FOOD,
          effects={EFF_HEAL: 1.0, EFF_HEAT_RESIST: 1.0}, duration=90.0,
          tags=frozenset({"mushroom", "chilly"})))
_reg(Item("voltfruit", "voltfruit", KIND_FOOD,
          effects={EFF_HEAL: 1.5, EFF_SHOCK_RESIST: 1.0}, duration=90.0,
          tags=frozenset({"fruit", "electro"})))
_reg(Item("stambulb", "stambulb", KIND_FOOD, effects={EFF_STAMINA: 20.0},
          tags=frozenset({"vegetable", "energizing"}),
          desc="Restores stamina. Climbers' staple."))
_reg(Item("endura_carrot", "endura carrot", KIND_FOOD,
          effects={EFF_STAMINA: 10.0, EFF_ENDURA: 40.0}, duration=120.0,
          tags=frozenset({"vegetable", "enduring"}), value=10))
_reg(Item("mighty_thistle", "mighty thistle", KIND_FOOD,
          effects={EFF_ATTACK_UP: 1.0}, duration=90.0,
          tags=frozenset({"vegetable", "mighty"})))
_reg(Item("armoranth", "armoranth", KIND_FOOD, effects={EFF_DEFENSE_UP: 1.0},
          duration=90.0, tags=frozenset({"vegetable", "tough"})))
_reg(Item("swift_violet", "swift violet", KIND_FOOD, effects={EFF_SPEED_UP: 1.0},
          duration=90.0, tags=frozenset({"vegetable", "hasty"})))
_reg(Item("cactus_fruit", "cactus fruit", KIND_FOOD, effects={EFF_HEAL: 1.5},
          tags=frozenset({"fruit"})))
_reg(Item("meat_raw", "raw meat", KIND_FOOD, effects={EFF_HEAL: 1.0},
          tags=frozenset({"meat", "raw"})))
_reg(Item("meat_cooked", "seared steak", KIND_FOOD, effects={EFF_HEAL: 6.0},
          tags=frozenset({"meat", "cooked"}), value=6))
_reg(Item("fish_raw", "raw fish", KIND_FOOD, effects={EFF_HEAL: 1.0},
          tags=frozenset({"fish", "raw"})))
_reg(Item("fish_cooked", "roasted fish", KIND_FOOD, effects={EFF_HEAL: 4.0},
          tags=frozenset({"fish", "cooked"})))
_reg(Item("egg", "bird egg", KIND_FOOD, effects={EFF_HEAL: 2.0},
          tags=frozenset({"egg"})))
_reg(Item("honey", "sweet honey", KIND_FOOD,
          effects={EFF_HEAL: 4.0, EFF_STAMINA: 10.0}, tags=frozenset({"sweet"}),
          value=8))
_reg(Item("salt", "rock salt", KIND_MATERIAL, tags=frozenset({"seasoning"})))

# ---------------------------- tools & weapons ------------------------------
_reg(Item("axe_stone", "stone axe", KIND_TOOL, stack=1, attack=3.0,
          durability=40, tool_class="axe", tags=frozenset({"wood", "two_handed"}),
          desc="Fells trees. Snaps if you fight with it too long."))
_reg(Item("axe_iron", "iron axe", KIND_TOOL, stack=1, attack=5.0,
          durability=110, tool_class="axe",
          tags=frozenset({"metal", "two_handed"})))
_reg(Item("pickaxe_stone", "stone pickaxe", KIND_TOOL, stack=1, attack=2.5,
          durability=40, tool_class="pickaxe", tags=frozenset({"wood"})))
_reg(Item("pickaxe_iron", "iron pickaxe", KIND_TOOL, stack=1, attack=4.0,
          durability=120, tool_class="pickaxe", tags=frozenset({"metal"})))
_reg(Item("hammer", "sledgehammer", KIND_TOOL, stack=1, attack=6.0,
          durability=60, tool_class="hammer",
          tags=frozenset({"metal", "two_handed"})))
_reg(Item("torch", "torch", KIND_TOOL, stack=5, attack=1.0, durability=60,
          tool_class="", tags=frozenset({"wood", "flame", "light"}),
          places=46, desc="Carries fire. Also your lantern at night."))
_reg(Item("sword_wood", "wooden sword", KIND_WEAPON, stack=1, attack=4.0,
          durability=30, tags=frozenset({"wood"})))
_reg(Item("sword_iron", "iron sword", KIND_WEAPON, stack=1, attack=8.0,
          durability=90, tags=frozenset({"metal"})))
_reg(Item("sword_starlight", "starlight blade", KIND_WEAPON, stack=1,
          attack=18.0, durability=240, tags=frozenset({"metal", "legendary"}),
          value=100, desc="Forged from skyfall iron and a luminous crystal."))
_reg(Item("spear_wood", "wooden spear", KIND_WEAPON, stack=1, attack=5.0,
          durability=35, tags=frozenset({"wood", "reach"})))
_reg(Item("club_boko", "monster club", KIND_WEAPON, stack=1, attack=5.0,
          durability=25, tags=frozenset({"wood"})))
_reg(Item("bow_crude", "crude bow", KIND_BOW, stack=1, attack=3.0,
          durability=30, tags=frozenset({"wood"})))
_reg(Item("bow_hunter", "hunter's bow", KIND_BOW, stack=1, attack=6.0,
          durability=80, tags=frozenset({"wood"})))
_reg(Item("shield_wood", "wooden shield", KIND_SHIELD, stack=1, durability=40,
          defense=3.0, tags=frozenset({"wood"})))
_reg(Item("shield_metal", "metal shield", KIND_SHIELD, stack=1, durability=120,
          defense=6.0, tags=frozenset({"metal"})))

# ---------------------------- ammo & throwables ---------------------------
_reg(Item("arrow", "arrow", KIND_AMMO, stack=99, attack=3.0,
          tags=frozenset({"arrow"})))
_reg(Item("arrow_fire", "fire arrow", KIND_AMMO, stack=99, attack=4.0,
          tags=frozenset({"arrow", "fire"})))
_reg(Item("arrow_ice", "ice arrow", KIND_AMMO, stack=99, attack=4.0,
          tags=frozenset({"arrow", "ice"})))
_reg(Item("arrow_shock", "shock arrow", KIND_AMMO, stack=99, attack=4.0,
          tags=frozenset({"arrow", "electric"})))
_reg(Item("bomb", "remote bomb", KIND_PLACEABLE, stack=10, attack=12.0,
          tags=frozenset({"explosive"}),
          desc="Blows a hole in cracked walls. Mind the blast radius."))

# ---------------------------- armour --------------------------------------
_reg(Item("warm_doublet", "warm doublet", KIND_ARMOR, stack=1, durability=0,
          cold_resist=1.0, defense=2.0, tags=frozenset({"cloth"}),
          desc="Wool-lined. Shrugs off the cold without cooking a meal."))
_reg(Item("heat_cloak", "desert cloak", KIND_ARMOR, stack=1, heat_resist=1.0,
          defense=1.0, tags=frozenset({"cloth"})))
_reg(Item("rubber_helm", "insulated helm", KIND_ARMOR, stack=1, defense=2.0,
          tags=frozenset({"insulator"}),
          desc="Rubber lining. Lightning finds another target."))
_reg(Item("climb_gear", "climbing gear", KIND_ARMOR, stack=1, defense=1.0,
          tags=frozenset({"climb"}), desc="Climbing costs 35%% less stamina."))

# ---------------------------- gadgets (rune abilities) --------------------
_reg(Item("glider", "wing glider", KIND_GADGET, stack=1,
          tags=frozenset({"glide"}),
          desc="Hold to glide. Burns stamina while airborne."))
_reg(Item("rune_magnet", "magnet rune", KIND_GADGET, stack=1,
          tags=frozenset({"rune"}),
          desc="Grab and move metal objects at a distance."))
_reg(Item("rune_cryo", "cryo rune", KIND_GADGET, stack=1,
          tags=frozenset({"rune"}),
          desc="Raise a pillar of ice from any water surface."))
_reg(Item("rune_stasis", "stasis rune", KIND_GADGET, stack=1,
          tags=frozenset({"rune"}),
          desc="Freeze an object in time; stored impulses release at once."))
_reg(Item("rune_bomb", "bomb rune", KIND_GADGET, stack=1,
          tags=frozenset({"rune"}), desc="Conjures a remote bomb on demand."))

# ---------------------------- keys / quest items --------------------------
_reg(Item("key_bronze", "bronze key", KIND_KEY, stack=9,
          tags=frozenset({"key"})))
_reg(Item("trial_token", "trial token", KIND_KEY, stack=9,
          tags=frozenset({"key", "objective"}), value=20))
_reg(Item("beacon_ember", "beacon ember", KIND_KEY, stack=1,
          tags=frozenset({"objective", "flame"})))
_reg(Item("guardian_core", "guardian core", KIND_KEY, stack=9,
          tags=frozenset({"objective", "metal"}), value=50))

ITEM_IDS = tuple(ITEMS.keys())
ITEM_INDEX = {k: i for i, k in enumerate(ITEM_IDS)}
NUM_ITEMS = len(ITEM_IDS)


def get(item_id: str) -> Item:
    try:
        return ITEMS[item_id]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown item id {item_id!r}") from exc


def exists(item_id: str) -> bool:
    return item_id in ITEMS


def has_tag(item_id: str, tag: str) -> bool:
    return tag in ITEMS[item_id].tags


def is_metal(item_id: str) -> bool:
    return "metal" in ITEMS[item_id].tags


def display(item_id: str) -> str:
    it = ITEMS.get(item_id)
    return it.name if it else item_id
