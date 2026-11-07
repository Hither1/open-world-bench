"""Cooking and crafting.

Two systems, deliberately different in character:

* **Cooking** is open-ended -- throw up to five ingredients in a pot and the
  result is computed from their tags.  Mixing two different effect families
  spoils the dish, so the agent has to reason about *why* ingredients go
  together rather than memorise a recipe list.
* **Crafting** is a recipe table with station requirements (bare hands, a
  campfire, or a forge), which is what gives the long-horizon tasks their
  dependency chain: ore -> ingot -> blade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core import items as I
from .inventory import Inventory, ItemStack, make_stack

# --------------------------------------------------------------------------
# cooking
# --------------------------------------------------------------------------
#: ingredient tag -> (effect, base potency, seconds per ingredient)
EFFECT_TAGS: dict[str, tuple[str, float, float]] = {
    "spicy": (I.EFF_COLD_RESIST, 1.0, 60.0),
    "chilly": (I.EFF_HEAT_RESIST, 1.0, 60.0),
    "electro": (I.EFF_SHOCK_RESIST, 1.0, 60.0),
    "mighty": (I.EFF_ATTACK_UP, 1.0, 50.0),
    "tough": (I.EFF_DEFENSE_UP, 1.0, 50.0),
    "hasty": (I.EFF_SPEED_UP, 1.0, 50.0),
    "enduring": (I.EFF_ENDURA, 20.0, 90.0),
    "energizing": (I.EFF_STAMINA, 14.0, 0.0),
    "hearty": (I.EFF_HEAL, 4.0, 0.0),
}

MAX_INGREDIENTS = 5

_ADJECTIVE = {
    I.EFF_COLD_RESIST: "spicy", I.EFF_HEAT_RESIST: "chilly",
    I.EFF_SHOCK_RESIST: "electro", I.EFF_ATTACK_UP: "mighty",
    I.EFF_DEFENSE_UP: "tough", I.EFF_SPEED_UP: "hasty",
    I.EFF_ENDURA: "enduring", I.EFF_STAMINA: "energizing",
    I.EFF_HEAL: "hearty",
}


def _dish_noun(items: list[str]) -> str:
    tags = set()
    for it in items:
        tags |= I.get(it).tags
    if "meat" in tags and "fruit" in tags:
        return "meat and fruit skewer"
    if "meat" in tags:
        return "meat skewer"
    if "fish" in tags:
        return "fish fillet"
    if "mushroom" in tags:
        return "mushroom medley"
    if "fruit" in tags and "sweet" in tags:
        return "fruitcake"
    if "fruit" in tags:
        return "fruit bowl"
    if "vegetable" in tags:
        return "vegetable stew"
    if "egg" in tags:
        return "omelette"
    return "simmered dish"


@dataclass
class CookResult:
    ok: bool
    stack: ItemStack | None = None
    reason: str = ""
    effects: dict[str, float] = field(default_factory=dict)
    description: str = ""


def cook(items: list[str]) -> CookResult:
    """Combine up to five ingredients into a dish.

    Food value is the sum of the ingredients plus a cooking bonus.  A single
    effect family carries through to the dish and scales with how many
    ingredients of that family went in; two different families cancel out and
    you get dubious food.
    """
    items = [i for i in items if i]
    if not items:
        return CookResult(False, reason="nothing in the pot")
    if len(items) > MAX_INGREDIENTS:
        return CookResult(False, reason=f"at most {MAX_INGREDIENTS} ingredients")
    for it in items:
        if not I.exists(it):
            return CookResult(False, reason=f"unknown ingredient {it!r}")

    specs = [I.get(i) for i in items]
    edible = [s for s in specs if s.kind in (I.KIND_FOOD, I.KIND_DISH)
              or "seasoning" in s.tags]
    monster = [s for s in specs if "monster" in s.tags]
    other = [s for s in specs if s not in edible and s not in monster]

    if other:
        return CookResult(False, reason=f"{other[0].name} is not food")

    # --- elixirs: monster part + a gland/critter, no ordinary food ---------
    if monster and not [s for s in edible if "seasoning" not in s.tags]:
        eff = {}
        for s in monster:
            if "fire" in s.tags:
                eff[I.EFF_HEAT_RESIST] = 1.0
            elif "ice" in s.tags:
                eff[I.EFF_COLD_RESIST] = 1.0
            elif "electric" in s.tags:
                eff[I.EFF_SHOCK_RESIST] = 1.0
        if not eff:
            eff = {I.EFF_DEFENSE_UP: 1.0}
        name = _ADJECTIVE.get(next(iter(eff)), "murky") + " elixir"
        dur = 60.0 + 30.0 * len(monster)
        st = make_stack("mushroom", 1)      # carrier item id, renamed below
        st.item = "mushroom"
        st.data = dict(name=name, effects=dict(eff, **{I.EFF_HEAL: 1.0}),
                       duration=dur, dish=True)
        return CookResult(True, st, effects=st.data["effects"],
                          description=f"{name} ({', '.join(eff)}, {dur:.0f}s)")

    if monster:
        st = make_stack("mushroom", 1)
        st.data = dict(name="dubious food", effects={I.EFF_HEAL: 2.0},
                       duration=0.0, dish=True)
        return CookResult(True, st, effects={I.EFF_HEAL: 2.0},
                          reason="monster parts spoiled it",
                          description="dubious food (heals a little)")

    # --- effect family ----------------------------------------------------
    families: dict[str, int] = {}
    for s in specs:
        for tag in s.tags:
            if tag in EFFECT_TAGS:
                families[tag] = families.get(tag, 0) + 1
    timed = {t: c for t, c in families.items()
             if EFFECT_TAGS[t][0] in I.TIMED_EFFECTS}
    if len(timed) > 1:
        st = make_stack("mushroom", 1)
        st.data = dict(name="dubious food", effects={I.EFF_HEAL: 2.0},
                       duration=0.0, dish=True)
        return CookResult(True, st, effects={I.EFF_HEAL: 2.0},
                          reason="clashing ingredients",
                          description="dubious food (clashing effects)")

    heal = sum(s.effects.get(I.EFF_HEAL, 0.0) for s in specs)
    heal = heal * 1.45 + 1.0 * len(specs)
    effects: dict[str, float] = {}
    duration = 0.0
    adjective = ""
    for tag, count in families.items():
        eff, pot, per = EFFECT_TAGS[tag]
        if eff == I.EFF_HEAL:
            heal += pot * count * 1.5
            adjective = adjective or _ADJECTIVE[eff]
        elif eff == I.EFF_STAMINA:
            effects[eff] = effects.get(eff, 0.0) + pot * count
            adjective = adjective or _ADJECTIVE[eff]
        else:
            effects[eff] = max(effects.get(eff, 0.0),
                               min(3.0, 1.0 + 0.5 * (count - 1)) * pot)
            duration = max(duration, 30.0 + per * count)
            adjective = adjective or _ADJECTIVE[eff]
    for s in specs:
        if s.effects.get(I.EFF_STAMINA):
            effects[I.EFF_STAMINA] = effects.get(I.EFF_STAMINA, 0.0) \
                + s.effects[I.EFF_STAMINA] * 0.5
    effects[I.EFF_HEAL] = round(heal, 2)

    name = f"{adjective + ' ' if adjective else ''}{_dish_noun(items)}"
    st = make_stack("mushroom", 1)
    st.data = dict(name=name, effects=effects, duration=duration, dish=True,
                   ingredients=list(items))
    desc_bits = [f"heals {effects[I.EFF_HEAL]:.0f}"]
    for k, v in effects.items():
        if k == I.EFF_HEAL:
            continue
        desc_bits.append(f"{k} {v:.1f}" + (f" for {duration:.0f}s"
                                           if k in I.TIMED_EFFECTS else ""))
    return CookResult(True, st, effects=effects,
                      description=f"{name} ({', '.join(desc_bits)})")


# --------------------------------------------------------------------------
# crafting
# --------------------------------------------------------------------------
STATION_NONE = ""
STATION_FIRE = "campfire"
STATION_FORGE = "forge"
STATION_POT = "cookpot"


@dataclass
class Recipe:
    id: str
    inputs: dict[str, int]
    output: str
    count: int = 1
    station: str = STATION_NONE
    desc: str = ""


RECIPES: dict[str, Recipe] = {}


def _r(rid, inputs, output, count=1, station=STATION_NONE, desc=""):
    RECIPES[rid] = Recipe(rid, inputs, output, count, station, desc)


# hand crafting
_r("torch", {"wood": 1, "flint": 1}, "torch", 1, desc="Carry fire with you.")
_r("arrows", {"wood": 1, "flint": 1, "feather": 1}, "arrow", 5)
_r("rope", {"fiber": 4}, "rope", 1)
_r("sword_wood", {"wood": 3}, "sword_wood")
_r("spear_wood", {"wood": 2, "flint": 1}, "spear_wood")
_r("bow_crude", {"wood": 3, "fiber": 2}, "bow_crude")
_r("shield_wood", {"wood": 4}, "shield_wood")
_r("axe_stone", {"wood": 2, "stone": 2}, "axe_stone")
_r("pickaxe_stone", {"wood": 2, "stone": 3}, "pickaxe_stone")
_r("bridge_kit", {"wood": 6}, "bridge_kit", 3)
_r("climb_gear", {"fiber": 6, "hide": 2}, "climb_gear",
   desc="Climbing costs much less stamina.")
_r("warm_doublet", {"hide": 3, "fiber": 4}, "warm_doublet",
   desc="Cold protection without cooking.")
_r("heat_cloak", {"fiber": 6, "petal": 2}, "heat_cloak")
_r("rubber_helm", {"monster_guts": 4, "fiber": 2}, "rubber_helm",
   desc="Insulated: lightning looks elsewhere.")
_r("bomb", {"charcoal": 2, "flint": 2, "fiber": 1}, "bomb", 2)
_r("arrow_fire", {"arrow": 3, "fire_gland": 1}, "arrow_fire", 3)
_r("arrow_ice", {"arrow": 3, "ice_gland": 1}, "arrow_ice", 3)
_r("arrow_shock", {"arrow": 3, "electro_gland": 1}, "arrow_shock", 3)

# fire
_r("charcoal", {"wood": 2}, "charcoal", 2, STATION_FIRE)
_r("meat_cooked", {"meat_raw": 1}, "meat_cooked", 1, STATION_FIRE)
_r("fish_cooked", {"fish_raw": 1}, "fish_cooked", 1, STATION_FIRE)

# forge
_r("iron_ingot", {"iron_ore": 2, "charcoal": 1}, "iron_ingot", 1, STATION_FORGE,
   desc="Smelt ore into a usable ingot.")
_r("sword_iron", {"iron_ingot": 2, "wood": 1}, "sword_iron", 1, STATION_FORGE)
_r("axe_iron", {"iron_ingot": 2, "wood": 1}, "axe_iron", 1, STATION_FORGE)
_r("pickaxe_iron", {"iron_ingot": 2, "wood": 1}, "pickaxe_iron", 1,
   STATION_FORGE)
_r("hammer", {"iron_ingot": 2, "wood": 2}, "hammer", 1, STATION_FORGE)
_r("shield_metal", {"iron_ingot": 3}, "shield_metal", 1, STATION_FORGE)
_r("sword_starlight", {"iron_ingot": 4, "crystal": 2, "guardian_core": 1},
   "sword_starlight", 1, STATION_FORGE,
   desc="The legendary blade. Needs a guardian core.")

RECIPE_IDS = tuple(RECIPES)


def can_craft(recipe_id: str, inv: Inventory, station: str = STATION_NONE
              ) -> tuple[bool, str]:
    r = RECIPES.get(recipe_id)
    if r is None:
        return False, f"no recipe {recipe_id!r}"
    if r.station and r.station != station:
        return False, f"needs a {r.station}"
    for item, n in r.inputs.items():
        if inv.count(item) < n:
            return False, f"needs {n}x {I.display(item)}"
    return True, ""


def craft(recipe_id: str, inv: Inventory, station: str = STATION_NONE
          ) -> tuple[bool, str]:
    ok, why = can_craft(recipe_id, inv, station)
    if not ok:
        return False, why
    r = RECIPES[recipe_id]
    for item, n in r.inputs.items():
        inv.remove(item, n)
    inv.add(r.output, r.count)
    return True, f"crafted {r.count}x {I.display(r.output)}"


def available(inv: Inventory, station: str = STATION_NONE) -> list[str]:
    return [rid for rid in RECIPES if can_craft(rid, inv, station)[0]]


def recipes_for_station(station: str) -> list[Recipe]:
    return [r for r in RECIPES.values() if r.station == station]
