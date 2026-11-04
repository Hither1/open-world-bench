"""Local, inventory-safe interactions for the Wildlands survival loop."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

import numpy as np

from ..core import blocks as B, items as I
from .crafting import cook


class SurvivalInteractions:
    """Inventory slots are zero-based; repeated cooking slots use multiple units."""

    def __init__(self, world, agent, elements, npcs, plants, chests):
        self.world, self.agent = world, agent
        self.elements, self.npcs = elements, npcs
        self.plants, self.chests = plants, chests

    def reachable(self, pos, radius=3.0):
        target = np.asarray(pos, dtype=np.float64) + (0, 0.5, 0)
        delta = target - self.agent.eye
        distance = float(np.linalg.norm(delta))
        if distance > radius:
            return False
        # Stop just short of the target, allowing solid pots and chests.
        hit = self.world.raycast(self.agent.eye, delta, max(0, distance - 0.7))
        return not hit.hit

    def nearby_blocks(self, block_ids):
        x, y, z = np.floor(self.agent.pos).astype(int)
        out = []
        for xx in range(max(0, x - 3), min(self.world.size_x, x + 4)):
            for yy in range(max(0, y - 2), min(self.world.size_y, y + 4)):
                for zz in range(max(0, z - 3), min(self.world.size_z, z + 4)):
                    if self.world.get(xx, yy, zz) in block_ids:
                        pos = (xx + 0.5, yy, zz + 0.5)
                        if self.reachable(pos):
                            out.append((xx, yy, zz))
        return out

    @staticmethod
    def failed(reason):
        return [dict(type="action_failed", reason=reason)]

    def interact(self):
        candidates = []
        for kind, objects in (("gather", self.plants), ("chest", self.chests)):
            for obj in objects:
                if not obj.get("used") and self.reachable(obj["pos"]):
                    distance = float(np.linalg.norm(np.asarray(obj["pos"]) - self.agent.pos))
                    candidates.append((distance, kind, obj))
        if candidates:
            _, kind, obj = min(candidates, key=lambda c: c[0])
            inv = self.agent.inventory
            if kind == "gather":
                remaining = obj.get("count", 1)
                added = inv.add(obj["item"], remaining)
                if not added:
                    return self.failed("inventory full")
                obj["count"] = remaining - added
                obj["used"] = obj["count"] == 0
                return [dict(type="gather", item=obj["item"], count=added)]
            if obj.get("locked_by_plate"):
                return self.failed("chest locked by a trial pressure plate")
            left, events = [], []
            for item, count in obj["items"]:
                added = inv.add(item, count)
                if added:
                    events.append(dict(type="loot", item=item, count=added))
                if added < count:
                    left.append([item, count - added])
            obj["items"] = left
            obj["used"] = not left
            return events or self.failed("inventory full")
        fires = self.nearby_blocks({B.CAMPFIRE})
        if fires:
            if self.elements.ignite(self.world, *fires[0]):
                return [dict(type="light_campfire", pos=fires[0])]
            return self.failed("fire disabled")
        return self.failed("nothing reachable")

    def stack(self, slot):
        stacks = self.agent.inventory.stacks
        return stacks[slot] if 0 <= slot < len(stacks) else None

    def eat(self, slot):
        st = self.stack(slot)
        if st is None or st.spec.kind not in (I.KIND_FOOD, I.KIND_DISH):
            return self.failed("select food to eat")
        a = self.agent
        effects, duration, name = st.effects(), st.duration(), st.name
        for effect, value in effects.items():
            if effect == I.EFF_HEAL:
                a.heal(value)
            elif effect == I.EFF_STAMINA:
                a.regen_stamina(value)
            elif effect in I.TIMED_EFFECTS and duration > 0:
                if effect == I.EFF_ENDURA:
                    a.bonus_stamina = max(a.bonus_stamina, value)
                    a.regen_stamina(value)
                a.add_effect(effect, value, duration)
        st.count -= 1
        if not st.count:
            a.inventory._drop_stack(st)
        return [dict(type="eat", name=name, effects=effects)]

    def cook(self, slots):
        if not self.nearby_blocks({B.COOKPOT}):
            return self.failed("a reachable cooking pot is required")
        if not 1 <= len(slots) <= 5:
            return self.failed("choose one to five ingredients")
        needed = Counter(slots)
        for slot, count in needed.items():
            st = self.stack(slot)
            if st is None or st.count < count:
                return self.failed("not enough ingredients in the selected slots")
            if st.data.get("dish"):
                return self.failed("cooked dishes cannot be used as raw ingredients")
        result = cook([self.stack(slot).item for slot in slots])
        if not result.ok:
            return self.failed(result.reason)
        # Prepare a transaction so a full pack never destroys ingredients.
        inv = deepcopy(self.agent.inventory)
        for slot in sorted(needed, reverse=True):
            st = inv.stacks[slot]
            st.count -= needed[slot]
            if not st.count:
                inv._drop_stack(st)
        if not inv.add_stack(result.stack):
            return self.failed("make room for the cooked dish")
        self.agent.inventory = inv
        return [dict(type="cook", name=result.stack.name, effects=result.effects)]

    def equip(self, slot):
        st = self.stack(slot)
        if not self.agent.inventory.equip(st):
            return self.failed("select equipment")
        return [dict(type="equip", item=st.item)]

    def attack(self, durability=True):
        a = self.agent
        if a.cooldown > 0:
            return self.failed("attack cooling down")
        a.cooldown = 0.5
        events = [dict(type="swing")]
        for npc in self.npcs.near(a.pos, 3.0):
            delta = npc.pos - a.pos
            delta[1] = 0
            if np.dot(delta, a.forward_flat()) < 0.5 * np.linalg.norm(delta):
                continue
            if not self.reachable(npc.pos):
                continue
            damage = a.attack_power()
            died = npc.hurt(damage)
            events.append(dict(type="hit", npc=npc.id, damage=damage))
            if died:
                events.extend(self.npcs.kill(npc, self.world, cause="melee"))
            if durability:
                broken = a.inventory.damage("weapon")
                if broken:
                    events.append(dict(type="item_broke", item=broken))
            break
        return events

    def dispatch(self, command, slot=0, ingredients=(), durability=True):
        if command == "interact":
            return self.interact()
        if command == "eat":
            return self.eat(slot)
        if command == "cook":
            return self.cook(list(ingredients))
        if command == "equip":
            return self.equip(slot)
        if command == "attack":
            return self.attack(durability)
        if command == "stow":
            return [dict(type="stow", items=self.agent.inventory.unequip_metal())]
        return []

    def render_list(self):
        out = []
        for kind, objects in (("plant", self.plants), ("chest", self.chests)):
            for obj in objects:
                if obj.get("used"):
                    continue
                out.append(dict(pos=tuple(np.asarray(obj["pos"]) + (0, 0.45, 0)),
                                radius=0.3 if kind == "plant" else 0.5,
                                height=0.7, semantic=B.SEM_ITEM,
                                color=(225, 125, 65) if kind == "plant" else (165, 115, 65)))
        return out
