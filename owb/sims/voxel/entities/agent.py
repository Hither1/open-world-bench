"""Agent state: body, vitals, status effects and what it is carrying."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import items as I
from ..core.config import SimConfig
from ..sim.inventory import (SLOT_ARMOR, SLOT_BOW, SLOT_HEAD, SLOT_SHIELD,
                             SLOT_WEAPON, Inventory)


@dataclass
class Effect:
    kind: str
    magnitude: float
    ttl: float          # seconds remaining


@dataclass
class AgentState:
    """Everything about the agent that the simulator advances."""

    pos: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32))
    vel: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32))
    yaw: float = 0.0
    pitch: float = 0.0

    health: float = 24.0
    max_health: float = 24.0
    stamina: float = 100.0
    max_stamina: float = 100.0
    bonus_stamina: float = 0.0        # from "enduring" dishes

    # movement state
    on_ground: bool = True
    in_water: bool = False
    submerged: bool = False
    climbing: bool = False
    gliding: bool = False
    sprinting: bool = False
    exhausted: bool = False           # stamina hit zero; must recover a little
    wetness: float = 0.0              # 0..1, from rain and swimming
    body_temp: float = 0.0            # -1 freezing .. +1 overheating
    last_ground_y: float = 0.0
    air_time: float = 0.0

    # progression / bookkeeping
    inventory: Inventory = field(default_factory=lambda: Inventory(32))
    effects: list[Effect] = field(default_factory=list)
    alive: bool = True
    deaths: int = 0
    damage_taken: float = 0.0
    distance_travelled: float = 0.0
    bow_charge: float = 0.0
    cooldown: float = 0.0             # attack / action cooldown
    held_body: int | None = None      # body id held by the magnet rune
    stasis_body: int | None = None
    flags: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def eye(self) -> np.ndarray:
        return self.pos + np.array([0.0, 1.6, 0.0], dtype=np.float32)

    def facing(self) -> np.ndarray:
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        return np.array([cp * np.cos(self.yaw), sp, cp * np.sin(self.yaw)],
                        dtype=np.float32)

    def forward_flat(self) -> np.ndarray:
        return np.array([np.cos(self.yaw), 0.0, np.sin(self.yaw)],
                        dtype=np.float32)

    # ------------------------------------------------------------------
    # status effects
    # ------------------------------------------------------------------
    def add_effect(self, kind: str, magnitude: float, duration: float) -> None:
        for e in self.effects:
            if e.kind == kind:
                e.magnitude = max(e.magnitude, magnitude)
                e.ttl = max(e.ttl, duration)
                return
        self.effects.append(Effect(kind, float(magnitude), float(duration)))

    def effect(self, kind: str) -> float:
        for e in self.effects:
            if e.kind == kind and e.ttl > 0:
                return e.magnitude
        return 0.0

    def has_effect(self, kind: str) -> bool:
        return self.effect(kind) > 0.0

    def tick_effects(self, dt: float) -> list[str]:
        expired = []
        for e in list(self.effects):
            e.ttl -= dt
            if e.ttl <= 0:
                self.effects.remove(e)
                expired.append(e.kind)
        if I.EFF_ENDURA in expired:
            self.bonus_stamina = 0.0
            self.stamina = min(self.stamina, self.max_stamina)
        return expired

    # ------------------------------------------------------------------
    # vitals
    # ------------------------------------------------------------------
    @property
    def stamina_cap(self) -> float:
        return self.max_stamina + self.bonus_stamina

    def heal(self, amount: float) -> None:
        self.health = float(min(self.max_health, self.health + amount))

    def hurt(self, amount: float, kind: str = "generic") -> float:
        amount = float(amount) * max(0.1, 1.0 - 0.12 * self.effect(I.EFF_DEFENSE_UP))
        shield = self.inventory.equipped_stack(SLOT_SHIELD)
        if shield is not None and kind in ("melee", "projectile"):
            amount = max(0.0, amount - shield.spec.defense * 0.5)
        armor = self.inventory.equipped_stack(SLOT_ARMOR)
        if armor is not None and kind in ("melee", "projectile"):
            amount = max(0.0, amount - armor.spec.defense * 0.25)
        self.health -= amount
        self.damage_taken += amount
        if self.health <= 0:
            self.health = 0.0
            self.alive = False
        return amount

    def spend_stamina(self, amount: float) -> bool:
        """Spend stamina; returns False (and empties the bar) if short."""
        if self.stamina < amount:
            self.stamina = 0.0
            self.exhausted = True
            return False
        self.stamina -= amount
        if self.stamina <= 0.0:
            self.exhausted = True
        return True

    def regen_stamina(self, amount: float) -> None:
        self.stamina = min(self.stamina_cap, self.stamina + amount)
        if self.stamina > self.stamina_cap * 0.22:
            self.exhausted = False

    # ------------------------------------------------------------------
    # resistances
    # ------------------------------------------------------------------
    def cold_resist(self) -> float:
        r = self.effect(I.EFF_COLD_RESIST)
        for slot in (SLOT_ARMOR, SLOT_HEAD):
            st = self.inventory.equipped_stack(slot)
            if st is not None:
                r += st.spec.cold_resist
        return r

    def heat_resist(self) -> float:
        r = self.effect(I.EFF_HEAT_RESIST)
        for slot in (SLOT_ARMOR, SLOT_HEAD):
            st = self.inventory.equipped_stack(slot)
            if st is not None:
                r += st.spec.heat_resist
        return r

    def shock_resist(self) -> float:
        r = self.effect(I.EFF_SHOCK_RESIST)
        st = self.inventory.equipped_stack(SLOT_HEAD)
        if st is not None and "insulator" in st.spec.tags:
            r += 1.0
        return r

    def climb_efficiency(self) -> float:
        """Multiplier on climbing stamina cost (lower is better)."""
        for slot in (SLOT_ARMOR, SLOT_HEAD):
            st = self.inventory.equipped_stack(slot)
            if st is not None and "climb" in st.spec.tags:
                return 0.65
        return 1.0

    def attack_power(self) -> float:
        st = self.inventory.equipped_stack(SLOT_WEAPON)
        base = st.attack() if st is not None else 1.0
        return base * (1.0 + 0.35 * self.effect(I.EFF_ATTACK_UP))

    def speed_mult(self) -> float:
        return 1.0 + 0.22 * self.effect(I.EFF_SPEED_UP)

    def has_glider(self) -> bool:
        return self.inventory.has("glider")

    def has_gadget(self, gadget: str) -> bool:
        return self.inventory.has(gadget)

    # ------------------------------------------------------------------
    def reset_vitals(self, cfg: SimConfig) -> None:
        self.max_health = cfg.max_health
        self.health = cfg.max_health
        self.max_stamina = cfg.max_stamina
        self.stamina = cfg.max_stamina
        self.bonus_stamina = 0.0
        self.effects.clear()
        self.alive = True
        self.wetness = 0.0
        self.body_temp = 0.0

    def proprio(self) -> np.ndarray:
        """Compact float vector of body state (for the proprio observation)."""
        return np.array([
            self.pos[0], self.pos[1], self.pos[2],
            self.vel[0], self.vel[1], self.vel[2],
            np.cos(self.yaw), np.sin(self.yaw), self.pitch,
            self.health / max(self.max_health, 1e-3),
            self.stamina / max(self.stamina_cap, 1e-3),
            float(self.on_ground), float(self.in_water), float(self.submerged),
            float(self.climbing), float(self.gliding), float(self.sprinting),
            float(self.exhausted), self.wetness, self.body_temp,
            self.cold_resist(), self.heat_resist(), self.shock_resist(),
            self.bow_charge, float(self.air_time),
        ], dtype=np.float32)


PROPRIO_DIM = 25
PROPRIO_NAMES = [
    "x", "y", "z", "vx", "vy", "vz", "cos_yaw", "sin_yaw", "pitch",
    "health_frac", "stamina_frac", "on_ground", "in_water", "submerged",
    "climbing", "gliding", "sprinting", "exhausted", "wetness", "body_temp",
    "cold_resist", "heat_resist", "shock_resist", "bow_charge", "air_time",
]
