"""Inventory, item stacks, durability and equipment slots."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from ..core import items as I

_UID = itertools.count(1)

SLOT_WEAPON = "weapon"
SLOT_BOW = "bow"
SLOT_SHIELD = "shield"
SLOT_ARMOR = "armor"
SLOT_HEAD = "head"
EQUIP_SLOTS = (SLOT_WEAPON, SLOT_BOW, SLOT_SHIELD, SLOT_ARMOR, SLOT_HEAD)


@dataclass
class ItemStack:
    """One stack of one item.

    ``data`` carries instance state -- most importantly the effects of a dish
    produced by the cooking pot, which are computed from its ingredients and so
    cannot live in the static item registry.
    """

    item: str
    count: int = 1
    durability: int | None = None
    data: dict = field(default_factory=dict)
    uid: int = field(default_factory=lambda: next(_UID))

    @property
    def spec(self) -> I.Item:
        return I.get(self.item)

    @property
    def name(self) -> str:
        return self.data.get("name") or I.display(self.item)

    def stackable_with(self, other: "ItemStack") -> bool:
        return (self.item == other.item and not self.data and not other.data
                and self.durability is None and other.durability is None)

    def effects(self) -> dict[str, float]:
        if "effects" in self.data:
            return dict(self.data["effects"])
        return dict(self.spec.effects)

    def duration(self) -> float:
        return float(self.data.get("duration", self.spec.duration))

    def attack(self) -> float:
        return float(self.data.get("attack", self.spec.attack))

    def to_dict(self) -> dict:
        d = dict(item=self.item, count=self.count)
        if self.durability is not None:
            d["durability"] = self.durability
        if self.data:
            d["data"] = dict(self.data)
        return d

    def __repr__(self) -> str:  # pragma: no cover
        dur = f" {self.durability}" if self.durability is not None else ""
        return f"<{self.name} x{self.count}{dur}>"


def make_stack(item: str, count: int = 1, **data) -> ItemStack:
    spec = I.get(item)
    dur = spec.durability if spec.durability > 0 else None
    return ItemStack(item=item, count=int(count), durability=dur,
                     data=dict(data))


class Inventory:
    """A bounded list of stacks plus equipment slots referencing them by uid."""

    def __init__(self, capacity: int = 32):
        self.capacity = int(capacity)
        self.stacks: list[ItemStack] = []
        self.equipped: dict[str, int | None] = {s: None for s in EQUIP_SLOTS}

    # ------------------------------------------------------------------
    def add(self, item: str, count: int = 1, **data) -> int:
        """Add items; returns how many actually fit."""
        if not I.exists(item):
            raise KeyError(f"unknown item {item!r}")
        spec = I.get(item)
        added = 0
        if not data and spec.durability == 0:
            for st in self.stacks:
                if st.item == item and not st.data and st.count < spec.stack:
                    room = spec.stack - st.count
                    take = min(room, count - added)
                    st.count += take
                    added += take
                    if added >= count:
                        return added
        while added < count and len(self.stacks) < self.capacity:
            take = min(spec.stack, count - added) if spec.durability == 0 else 1
            self.stacks.append(make_stack(item, take, **data))
            added += take
        return added

    def add_stack(self, stack: ItemStack) -> bool:
        if len(self.stacks) >= self.capacity:
            return False
        self.stacks.append(stack)
        return True

    def remove(self, item: str, count: int = 1) -> int:
        """Remove up to ``count``; returns how many were removed."""
        left = int(count)
        for st in list(self.stacks):
            if left <= 0:
                break
            if st.item != item:
                continue
            take = min(st.count, left)
            st.count -= take
            left -= take
            if st.count <= 0:
                self._drop_stack(st)
        return int(count) - left

    def _drop_stack(self, st: ItemStack) -> None:
        if st in self.stacks:
            self.stacks.remove(st)
        for slot, uid in self.equipped.items():
            if uid == st.uid:
                self.equipped[slot] = None

    # ------------------------------------------------------------------
    def count(self, item: str) -> int:
        return sum(st.count for st in self.stacks if st.item == item)

    def count_tag(self, tag: str) -> int:
        return sum(st.count for st in self.stacks if tag in st.spec.tags)

    def has(self, item: str, count: int = 1) -> bool:
        return self.count(item) >= count

    def has_all(self, reqs: dict[str, int]) -> bool:
        return all(self.count(k) >= v for k, v in reqs.items())

    def find(self, item: str) -> ItemStack | None:
        for st in self.stacks:
            if st.item == item:
                return st
        return None

    def by_uid(self, uid: int | None) -> ItemStack | None:
        if uid is None:
            return None
        for st in self.stacks:
            if st.uid == uid:
                return st
        return None

    def find_tool(self, tool_class: str) -> ItemStack | None:
        """Best (highest-attack) item acting as the given tool class."""
        best = None
        for st in self.stacks:
            if st.spec.tool_class == tool_class:
                if best is None or st.spec.attack > best.spec.attack:
                    best = st
        return best

    def find_best(self, kind: str) -> ItemStack | None:
        best = None
        for st in self.stacks:
            if st.spec.kind == kind:
                if best is None or st.spec.attack + st.spec.defense > \
                        best.spec.attack + best.spec.defense:
                    best = st
        return best

    @property
    def is_full(self) -> bool:
        return len(self.stacks) >= self.capacity

    # ------------------------------------------------------------------
    def equip(self, stack: ItemStack | None, slot: str | None = None) -> bool:
        if stack is None:
            return False
        spec = stack.spec
        if slot is None:
            slot = {I.KIND_WEAPON: SLOT_WEAPON, I.KIND_TOOL: SLOT_WEAPON,
                    I.KIND_BOW: SLOT_BOW, I.KIND_SHIELD: SLOT_SHIELD,
                    I.KIND_ARMOR: SLOT_ARMOR}.get(spec.kind)
        if slot is None or slot not in self.equipped:
            return False
        if slot == SLOT_ARMOR and "insulator" in spec.tags:
            slot = SLOT_HEAD
        self.equipped[slot] = stack.uid
        return True

    def equipped_stack(self, slot: str) -> ItemStack | None:
        return self.by_uid(self.equipped.get(slot))

    def unequip(self, slot: str) -> None:
        if slot in self.equipped:
            self.equipped[slot] = None

    def unequip_metal(self) -> list[str]:
        """Drop every metal item from the equipment slots (lightning safety)."""
        dropped = []
        for slot in EQUIP_SLOTS:
            st = self.equipped_stack(slot)
            if st is not None and "metal" in st.spec.tags:
                self.equipped[slot] = None
                dropped.append(st.item)
        return dropped

    def has_metal_equipped(self) -> bool:
        return any("metal" in st.spec.tags
                   for st in (self.equipped_stack(s) for s in EQUIP_SLOTS)
                   if st is not None)

    # ------------------------------------------------------------------
    def damage(self, slot: str, amount: int = 1) -> str | None:
        """Wear down an equipped item; returns its id if it broke."""
        st = self.equipped_stack(slot)
        if st is None or st.durability is None:
            return None
        st.durability -= int(amount)
        if st.durability <= 0:
            broken = st.item
            st.count -= 1
            if st.count <= 0:
                self._drop_stack(st)
            else:
                st.durability = st.spec.durability
            return broken
        return None

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for st in self.stacks:
            out[st.item] = out.get(st.item, 0) + st.count
        return out

    def describe(self) -> str:
        if not self.stacks:
            return "empty-handed"
        parts = []
        for st in self.stacks:
            tag = ""
            for slot, uid in self.equipped.items():
                if uid == st.uid:
                    tag = f" (equipped: {slot})"
            dur = f" [{st.durability}]" if st.durability is not None else ""
            cnt = f" x{st.count}" if st.count > 1 else ""
            parts.append(f"{st.name}{cnt}{dur}{tag}")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return dict(stacks=[s.to_dict() for s in self.stacks],
                    equipped={k: (self.by_uid(v).item if self.by_uid(v) else None)
                              for k, v in self.equipped.items()})
