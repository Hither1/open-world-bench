"""Lemon tart: placement and milestones, with no simulator imports.

The kitchen that executes this is `LemonTart` in `owb.tasks.lemon_tart_env`.
The spec is `docs/mobile-manipulation-tasks.md`.
"""
from __future__ import annotations

from typing import Mapping, Sequence

INGREDIENTS = ("butter", "egg", "lemon", "flour", "sugar")
FRIDGE_INGREDIENTS = ("butter", "egg", "lemon")
DRAWER_INGREDIENTS = ("flour", "sugar")
HAND_TOOLS = ("bowl", "whisk", "saucepan")
BAKEWARE = ("baking_sheet", "tart_pan")
WASH_TOOLS = ("bowl", "whisk", "saucepan", "baking_sheet")
OBJECT_NAMES = INGREDIENTS + HAND_TOOLS + BAKEWARE

FRIDGE = "fridge"
DISHWASHER = "dishwasher"
TOASTER_OVEN = "toaster_oven"
STOVE = "stove"
COUNTER = "counter"
TART_PAN = "tart_pan"

KNOBS = ("function", "temperature", "time")
KNOB_ON = 0.25
KNOB_OFF = 0.05
MIN_DRAWERS = 4

MILESTONES = (
    "ingredients_out",
    "crust_baked",
    "knobs_zeroed",
    "curd_started",
    "crumble_baked",
    "assembled",
    "ingredients_restored",
    "tools_washed",
    "served",
)

RECIPE_INSTRUCTION = (
    "Make a lemon tart. Butter, egg, and lemon are in the fridge. Flour and sugar "
    "are in drawers. The tart pan and baking sheet start in the dishwasher; the bowl, "
    "whisk, and saucepan are split between drawers and the dishwasher.\n"
    "Crust: put the tart pan in the toaster oven and turn the function, temperature, "
    "and time knobs up, then turn all three back off.\n"
    "Curd: put the saucepan on the stove.\n"
    "Crumble: take the baking sheet back out if it is in the oven, then bake it with "
    "the same three knobs up.\n"
    "Set the lemon on the tart pan on the counter, then put all five ingredients back "
    "where they came from and close the doors. Load the bowl, whisk, saucepan, and "
    "baking sheet into the dishwasher and close it. Leave the tart on its pan on the "
    "counter."
)

SHORT_INSTRUCTION = (
    "Finish the lemon tart. Put the butter, egg, lemon, flour, and sugar back where "
    "they came from and close the doors. Load the bowl, whisk, saucepan, and baking "
    "sheet into the dishwasher and close it. Leave the tart on its pan on the counter."
)


def assign_homes(drawer_names: Sequence[str], rng) -> dict[str, str]:
    """Sample one episode's object homes.

    Butter, egg, and lemon are in the fridge. Flour and sugar get two different
    drawers. The tart pan and baking sheet are in the dishwasher. Exactly one of
    the bowl, whisk, and saucepan is in the dishwasher; the other two get the
    remaining drawers. Ingredients are never placed in the dishwasher.
    """
    drawers = list(drawer_names)
    if len(set(drawers)) != len(drawers):
        raise ValueError("drawer names must be unique")
    if len(drawers) < MIN_DRAWERS:
        raise ValueError(
            f"lemon tart needs at least {MIN_DRAWERS} drawers, got {len(drawers)}"
        )
    rng.shuffle(drawers)
    homes = {name: FRIDGE for name in FRIDGE_INGREDIENTS}
    homes["flour"] = drawers[0]
    homes["sugar"] = drawers[1]
    hand = list(HAND_TOOLS)
    rng.shuffle(hand)
    homes[hand[0]] = DISHWASHER
    homes[hand[1]] = drawers[2]
    homes[hand[2]] = drawers[3]
    homes["baking_sheet"] = DISHWASHER
    homes["tart_pan"] = DISHWASHER
    return homes


def _members(state: Mapping, key: str, name: str) -> set:
    raw = state.get(key, {}).get(name, ())
    return set(raw)


def _inside(state: Mapping, obj: str, fixture: str) -> bool:
    return fixture in _members(state, "inside", obj)


def _touching(state: Mapping, obj: str, other: str) -> bool:
    return other in _members(state, "touching", obj)


def _held(state: Mapping, obj: str) -> bool:
    return obj in set(state.get("held", ()))


def _open(state: Mapping, fixture: str) -> bool:
    return fixture in set(state.get("open", ()))


def _knob_values(state: Mapping) -> list[float]:
    knobs = state.get("knobs", {})
    return [float(knobs.get(name, 0.0)) for name in KNOBS]


def _knobs_on(state: Mapping) -> bool:
    return all(value >= KNOB_ON for value in _knob_values(state))


def _knobs_off(state: Mapping) -> bool:
    return all(value <= KNOB_OFF for value in _knob_values(state))


class LemonTartTracker:
    """Latches the nine lemon-tart milestones. Homes stay off the public state."""

    def __init__(self, homes: Mapping[str, str], storage_ids: Sequence[str]):
        missing = [name for name in OBJECT_NAMES if name not in homes]
        if missing:
            raise ValueError(f"homes missing {missing}")
        self.homes = dict(homes)
        self.storage_ids = tuple(storage_ids)
        self.reset()

    def reset(self) -> None:
        self.latched: list[str] = []
        self.sheet_cleared = False
        self.opens_during_restore = 0
        self.wrong_opens = 0
        self.pending_reward = 0
        self._prev_open: set[str] = set()
        self._state: dict = {}

    def update(self, state: Mapping) -> list[str]:
        """Advance from a sensed state. Returns milestones newly latched this call."""
        self._state = state
        before = set(self.latched)

        if self._want("ingredients_out") and self._ingredients_out(state):
            self.latched.append("ingredients_out")
        if self._want("crust_baked") and self._crust_baked(state):
            self.latched.append("crust_baked")
        if "crust_baked" in self.latched and not _inside(state, "baking_sheet", TOASTER_OVEN):
            self.sheet_cleared = True
        if self._want("knobs_zeroed") and self._has("crust_baked") and _knobs_off(state):
            self.latched.append("knobs_zeroed")
        if self._want("curd_started") and self._has("crust_baked") and _touching(state, "saucepan", STOVE):
            self.latched.append("curd_started")
        if self._want("crumble_baked") and self._crumble_baked(state):
            self.latched.append("crumble_baked")
        if self._want("assembled") and self._assembled(state):
            self.latched.append("assembled")
        if self._want("ingredients_restored") and self._ingredients_restored(state):
            self.latched.append("ingredients_restored")
        if self._want("tools_washed") and self._tools_washed(state):
            self.latched.append("tools_washed")
        if self._want("served") and self._served(state):
            self.latched.append("served")

        self._count_opens(state)
        newly = [name for name in MILESTONES if name in self.latched and name not in before]
        self.pending_reward += len(newly)
        return newly

    def consume_reward(self) -> int:
        value = self.pending_reward
        self.pending_reward = 0
        return value

    @property
    def recipe_visible(self) -> bool:
        return "crust_baked" not in self.latched

    @property
    def succeeded(self) -> bool:
        if not self._state:
            return False
        return (
            all(name in self.latched for name in MILESTONES)
            and self._ingredients_home(self._state)
            and self._wash_tools_loaded(self._state)
            and self._pan_on_counter(self._state)
        )

    def public_state(self) -> dict:
        return {
            "milestones": {name: name in self.latched for name in MILESTONES},
            "success": self.succeeded,
        }

    def evaluator_state(self) -> dict:
        state = self.public_state()
        state["homes"] = dict(self.homes)
        state["opens_during_restore"] = self.opens_during_restore
        state["wrong_opens"] = self.wrong_opens
        state["sheet_cleared"] = self.sheet_cleared
        return state

    def _has(self, name: str) -> bool:
        return name in self.latched

    def _want(self, name: str) -> bool:
        return name not in self.latched

    def _ingredients_out(self, state: Mapping) -> bool:
        for name in INGREDIENTS:
            home = self.homes[name]
            if _inside(state, name, home) or _open(state, home):
                return False
        return True

    def _crust_baked(self, state: Mapping) -> bool:
        return (
            self._has("ingredients_out")
            and _inside(state, TART_PAN, TOASTER_OVEN)
            and _knobs_on(state)
        )

    def _crumble_baked(self, state: Mapping) -> bool:
        return (
            self._has("knobs_zeroed")
            and self._has("curd_started")
            and self.sheet_cleared
            and _inside(state, "baking_sheet", TOASTER_OVEN)
            and _knobs_on(state)
        )

    def _assembled(self, state: Mapping) -> bool:
        return self._has("crumble_baked") and self._lemon_on_pan(state)

    def _ingredients_restored(self, state: Mapping) -> bool:
        return self._has("assembled") and self._ingredients_home(state)

    def _tools_washed(self, state: Mapping) -> bool:
        return self._has("ingredients_restored") and self._wash_tools_loaded(state)

    def _served(self, state: Mapping) -> bool:
        return self._has("tools_washed") and self._pan_on_counter(state)

    def _ingredients_home(self, state: Mapping) -> bool:
        for name in INGREDIENTS:
            home = self.homes[name]
            if not _inside(state, name, home) or _open(state, home) or _held(state, name):
                return False
        return True

    def _wash_tools_loaded(self, state: Mapping) -> bool:
        if _open(state, DISHWASHER):
            return False
        for name in WASH_TOOLS:
            if not _inside(state, name, DISHWASHER) or _held(state, name):
                return False
        return True

    def _lemon_on_pan(self, state: Mapping) -> bool:
        return _touching(state, "lemon", TART_PAN) and self._pan_on_counter(state)

    def _pan_on_counter(self, state: Mapping) -> bool:
        return (
            _touching(state, TART_PAN, COUNTER)
            and not _inside(state, TART_PAN, TOASTER_OVEN)
            and not _inside(state, TART_PAN, DISHWASHER)
            and not _held(state, TART_PAN)
        )

    def _count_opens(self, state: Mapping) -> None:
        now = set(state.get("open", ()))
        storage_now = now & set(self.storage_ids)
        storage_prev = self._prev_open & set(self.storage_ids)
        if self._has("ingredients_out") and not self._has("ingredients_restored"):
            opened = storage_now - storage_prev
            self.opens_during_restore += len(opened)
            outstanding = [
                name for name in INGREDIENTS
                if not _inside(state, name, self.homes[name])
            ]
            legal = {self.homes[name] for name in outstanding}
            self.wrong_opens += sum(1 for fixture in opened if fixture not in legal)
        self._prev_open = now
