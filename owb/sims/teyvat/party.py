"""The party: one agent, three characters, swapped on the fly.

Team play is the single-agent coordination problem this benchmark cares about.
The agent controls one character at a time; the others sit on the bench with
their own health, energy and cooldowns. Depth comes from the fact that a
reaction needs two different elements, so a strong play is: apply an aura with
one character, swap, and detonate it with the next. Party composition also
grants a passive **resonance**.

All characters, skills and names here are original to this project.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .elements import Element

MAX_ENERGY = 60.0


class SkillKind(IntEnum):
    STRIKE = 0     # damage in a cone in front
    FIELD = 1      # applies an element in a radius around the character
    BOLT = 2       # long-range single target
    LAUNCH = 3     # launches the character upward (traversal + element)
    WARD = 4       # shields the character
    DASH = 5       # short dash with damage along the way


@dataclass(frozen=True)
class Skill:
    name: str
    kind: SkillKind
    element: Element
    cooldown: float
    damage: float = 0.0
    radius: float = 0.0
    range: float = 0.0
    charge: float = 1.0           # elemental charge applied
    launch_speed: float = 0.0
    shield: float = 0.0
    heal: float = 0.0
    stamina_refund: float = 0.0
    energy_cost: float = 0.0      # >0 marks it as a burst
    description: str = ""


@dataclass(frozen=True)
class CharacterSpec:
    id: str
    name: str
    title: str
    element: Element
    max_hp: float
    attack: float
    skill: Skill
    burst: Skill
    tunic: tuple[float, float, float]
    hair: tuple[float, float, float]


# ---------------------------------------------------------------- the roster
ROSTER: dict[str, CharacterSpec] = {
    "kestrel": CharacterSpec(
        id="kestrel", name="Kestrel", title="the Torchbearer", element=Element.EMBER,
        max_hp=110.0, attack=34.0,
        skill=Skill("Cinder Lash", SkillKind.STRIKE, Element.EMBER, cooldown=6.0,
                    damage=46.0, radius=4.0, description="A sweeping arc of flame in front of you."),
        burst=Skill("Pyre Bloom", SkillKind.FIELD, Element.EMBER, cooldown=14.0,
                    damage=105.0, radius=8.0, charge=1.6, energy_cost=MAX_ENERGY,
                    description="Erupts ember in a wide ring."),
        tunic=(0.86, 0.33, 0.24), hair=(0.98, 0.76, 0.40)),
    "maren": CharacterSpec(
        id="maren", name="Maren", title="of the Still Water", element=Element.TIDE,
        max_hp=125.0, attack=28.0,
        skill=Skill("Tidecall", SkillKind.FIELD, Element.TIDE, cooldown=8.0,
                    damage=24.0, radius=6.5, charge=1.4,
                    description="Soaks everything nearby, priming reactions."),
        burst=Skill("Deluge", SkillKind.FIELD, Element.TIDE, cooldown=16.0,
                    damage=70.0, radius=9.0, charge=1.8, heal=45.0, energy_cost=MAX_ENERGY,
                    description="A wave that soaks enemies and mends the party."),
        tunic=(0.24, 0.52, 0.86), hair=(0.62, 0.86, 0.95)),
    "nym": CharacterSpec(
        id="nym", name="Nym", title="the Rimewright", element=Element.FROST,
        max_hp=105.0, attack=30.0,
        skill=Skill("Rime Spike", SkillKind.BOLT, Element.FROST, cooldown=7.0,
                    damage=52.0, range=18.0, description="A lance of ice at a distant target."),
        burst=Skill("Hoarfrost Veil", SkillKind.FIELD, Element.FROST, cooldown=15.0,
                    damage=80.0, radius=8.5, charge=1.7, energy_cost=MAX_ENERGY,
                    description="Freezes the ground and everything standing on it."),
        tunic=(0.58, 0.80, 0.92), hair=(0.92, 0.95, 0.99)),
    "vex": CharacterSpec(
        id="vex", name="Vex", title="the Stormstep", element=Element.SPARK,
        max_hp=100.0, attack=32.0,
        skill=Skill("Arc Step", SkillKind.DASH, Element.SPARK, cooldown=5.0,
                    damage=40.0, radius=3.2, range=7.0,
                    description="Dash forward, leaving a charged wake."),
        burst=Skill("Thunder Crown", SkillKind.FIELD, Element.SPARK, cooldown=14.0,
                    damage=95.0, radius=7.5, charge=1.6, energy_cost=MAX_ENERGY,
                    description="Calls down a ring of arcs."),
        tunic=(0.55, 0.35, 0.82), hair=(0.85, 0.72, 1.00)),
    "wren": CharacterSpec(
        id="wren", name="Wren", title="the Updraft", element=Element.GALE,
        max_hp=100.0, attack=27.0,
        skill=Skill("Updraft", SkillKind.LAUNCH, Element.GALE, cooldown=9.0,
                    damage=22.0, radius=5.0, launch_speed=13.0, stamina_refund=25.0,
                    description="A rising gust: launches you skyward, ready to glide."),
        burst=Skill("Squall Line", SkillKind.FIELD, Element.GALE, cooldown=13.0,
                    damage=65.0, radius=10.0, charge=1.5, energy_cost=MAX_ENERGY,
                    description="Scatters auras and enemies alike."),
        tunic=(0.34, 0.74, 0.58), hair=(0.90, 0.95, 0.80)),
    "boro": CharacterSpec(
        id="boro", name="Boro", title="the Bulwark", element=Element.STONE,
        max_hp=145.0, attack=31.0,
        skill=Skill("Bulwark", SkillKind.WARD, Element.STONE, cooldown=12.0,
                    damage=18.0, radius=4.0, shield=55.0,
                    description="Raises a stone ward around yourself."),
        burst=Skill("Fault Line", SkillKind.FIELD, Element.STONE, cooldown=18.0,
                    damage=110.0, radius=7.0, charge=1.5, shield=35.0, energy_cost=MAX_ENERGY,
                    description="Splits the ground open beneath your feet."),
        tunic=(0.72, 0.58, 0.28), hair=(0.45, 0.36, 0.28)),
}

DEFAULT_PARTY = ("kestrel", "maren", "wren")


class Resonance(IntEnum):
    NONE = 0
    BLAZE = 1        # two EMBER: more attack
    CURRENT = 2      # two TIDE: faster stamina recovery
    RIME = 3         # two FROST: tougher shields
    CHARGE = 4       # two SPARK: faster energy
    DRIFT = 5        # two GALE: cheaper traversal
    BEDROCK = 6      # two STONE: more health
    PRISM = 7        # all three different: stronger reactions


RESONANCE_NAMES = ["none", "blaze", "current", "rime", "charge", "drift", "bedrock", "prism"]
_RESONANCE_BY_ELEMENT = {
    Element.EMBER: Resonance.BLAZE, Element.TIDE: Resonance.CURRENT,
    Element.FROST: Resonance.RIME, Element.SPARK: Resonance.CHARGE,
    Element.GALE: Resonance.DRIFT, Element.STONE: Resonance.BEDROCK,
}


@dataclass
class MemberState:
    spec: CharacterSpec
    hp: float
    energy: float = 0.0
    skill_cd: float = 0.0
    burst_cd: float = 0.0
    downed: bool = False

    @property
    def alive(self) -> bool:
        return not self.downed and self.hp > 0.0


@dataclass
class PartyConfig:
    members: tuple[str, ...] = DEFAULT_PARTY
    swap_cooldown: float = 1.0
    passive_energy: float = 0.9      # energy per second for the active member
    bench_energy: float = 0.35       # benched members charge more slowly
    hit_energy: float = 2.5          # energy per landed hit
    revive_on_swap: bool = False


class Party:
    """Three characters, one of them active. Owns HP, energy and cooldowns."""

    def __init__(self, cfg: PartyConfig | None = None):
        self.cfg = cfg or PartyConfig()
        unknown = [m for m in self.cfg.members if m not in ROSTER]
        if unknown:
            raise ValueError(f"unknown party member(s) {unknown}; roster: {sorted(ROSTER)}")
        if not 1 <= len(self.cfg.members) <= 4:
            raise ValueError("party size must be between 1 and 4")
        self.members = [MemberState(ROSTER[m], hp=ROSTER[m].max_hp) for m in self.cfg.members]
        self.active_index = 0
        self.swap_timer = 0.0
        self.shield = 0.0
        self.shield_element = Element.NONE
        self.resonance = self._resonance()
        self._apply_resonance_statics()

    # ------------------------------------------------------------- resonance
    def _resonance(self) -> Resonance:
        els = [m.spec.element for m in self.members]
        for e in set(els):
            if els.count(e) >= 2:
                return _RESONANCE_BY_ELEMENT[e]
        return Resonance.PRISM if len(set(els)) == len(els) and len(els) >= 3 else Resonance.NONE

    def _apply_resonance_statics(self) -> None:
        if self.resonance == Resonance.BEDROCK:
            for m in self.members:
                m.hp *= 1.2

    @property
    def attack_multiplier(self) -> float:
        return 1.15 if self.resonance == Resonance.BLAZE else 1.0

    @property
    def reaction_multiplier(self) -> float:
        return 1.18 if self.resonance == Resonance.PRISM else 1.0

    @property
    def stamina_multiplier(self) -> float:
        """Multiplier on stamina *costs* (lower is better)."""
        return 0.8 if self.resonance == Resonance.DRIFT else 1.0

    @property
    def stamina_regen_multiplier(self) -> float:
        return 1.35 if self.resonance == Resonance.CURRENT else 1.0

    @property
    def shield_multiplier(self) -> float:
        return 1.3 if self.resonance == Resonance.RIME else 1.0

    @property
    def energy_multiplier(self) -> float:
        return 1.4 if self.resonance == Resonance.CHARGE else 1.0

    # ---------------------------------------------------------------- access
    @property
    def active(self) -> MemberState:
        return self.members[self.active_index]

    @property
    def element(self) -> Element:
        return self.active.spec.element

    @property
    def max_hp(self) -> float:
        return self.member_max_hp(self.active)

    def member_max_hp(self, member: MemberState) -> float:
        return member.spec.max_hp * (1.2 if self.resonance == Resonance.BEDROCK else 1.0)

    @property
    def any_alive(self) -> bool:
        return any(m.alive for m in self.members)

    def can_swap(self, index: int) -> bool:
        return (0 <= index < len(self.members) and index != self.active_index
                and self.swap_timer <= 0.0 and self.members[index].alive)

    def swap(self, index: int) -> bool:
        if not self.can_swap(index):
            return False
        self.active_index = index
        self.swap_timer = self.cfg.swap_cooldown
        return True

    def swap_to_next_alive(self) -> bool:
        for off in range(1, len(self.members)):
            idx = (self.active_index + off) % len(self.members)
            if self.members[idx].alive:
                self.active_index = idx
                self.swap_timer = self.cfg.swap_cooldown
                return True
        return False

    # ----------------------------------------------------------------- tick
    def tick(self, dt: float) -> None:
        self.swap_timer = max(0.0, self.swap_timer - dt)
        em = self.energy_multiplier
        for i, m in enumerate(self.members):
            m.skill_cd = max(0.0, m.skill_cd - dt)
            m.burst_cd = max(0.0, m.burst_cd - dt)
            rate = self.cfg.passive_energy if i == self.active_index else self.cfg.bench_energy
            if m.alive:
                m.energy = min(MAX_ENERGY, m.energy + rate * em * dt)
            if m.hp <= 0.0:
                m.downed = True

    def grant_energy(self, amount: float, to_all: bool = False) -> None:
        amount *= self.energy_multiplier
        targets = self.members if to_all else [self.active]
        for m in targets:
            if m.alive:
                m.energy = min(MAX_ENERGY, m.energy + amount)

    def heal(self, amount: float, to_all: bool = True) -> None:
        for m in (self.members if to_all else [self.active]):
            if m.alive:
                m.hp = min(self.member_max_hp(m), m.hp + amount)

    def revive(self) -> None:
        """Restore the party after a wipe, clearing combat resources."""
        for m in self.members:
            m.hp = self.member_max_hp(m)
            m.downed = False
            m.energy = m.skill_cd = m.burst_cd = 0.0
        self.shield = self.swap_timer = 0.0
        self.shield_element = Element.NONE

    def add_shield(self, amount: float, element: Element) -> None:
        amount *= self.shield_multiplier
        if amount >= self.shield:
            self.shield, self.shield_element = amount, element
        else:
            self.shield = max(self.shield, amount)

    def absorb(self, damage: float) -> float:
        """Run incoming damage through the party shield; returns what got through."""
        if self.shield <= 0.0:
            return damage
        taken = min(self.shield, damage)
        self.shield -= taken
        if self.shield <= 1e-6:
            self.shield, self.shield_element = 0.0, Element.NONE
        return max(0.0, damage - taken)

    # ---------------------------------------------------------------- skills
    def ready_skill(self) -> Skill | None:
        m = self.active
        return m.spec.skill if m.skill_cd <= 0.0 and m.alive else None

    def ready_burst(self) -> Skill | None:
        m = self.active
        if m.alive and m.burst_cd <= 0.0 and m.energy >= m.spec.burst.energy_cost:
            return m.spec.burst
        return None

    def spend_skill(self) -> None:
        m = self.active
        m.skill_cd = m.spec.skill.cooldown

    def spend_burst(self) -> None:
        m = self.active
        m.burst_cd = m.spec.burst.cooldown
        m.energy = max(0.0, m.energy - m.spec.burst.energy_cost)

    # ------------------------------------------------------------------ obs
    def as_vector(self) -> np.ndarray:
        """Flat party state for the observation: 7 numbers per member + 2."""
        rows = []
        for i, m in enumerate(self.members):
            rows.extend([
                float(i == self.active_index),
                m.hp / max(self.member_max_hp(m), 1e-6),
                m.energy / MAX_ENERGY,
                1.0 - min(1.0, m.skill_cd / max(m.spec.skill.cooldown, 1e-6)),
                1.0 - min(1.0, m.burst_cd / max(m.spec.burst.cooldown, 1e-6)),
                float(m.spec.element) / 6.0,
                float(m.alive),
            ])
        while len(rows) < 7 * 4:          # pad to a fixed 4-slot layout
            rows.extend([0.0] * 7)
        rows.append(self.shield / 100.0)
        rows.append(float(self.swap_timer <= 0.0))
        return np.asarray(rows, dtype=np.float32)

    def summary(self) -> list[dict]:
        return [{
            "name": m.spec.name, "element": m.spec.element.name.lower(),
            "hp": round(m.hp, 1), "energy": round(m.energy, 1),
            "skill_ready": m.alive and m.skill_cd <= 0.0,
            "burst_ready": m.alive and m.burst_cd <= 0.0 and m.energy >= m.spec.burst.energy_cost,
            "active": i == self.active_index, "downed": m.downed,
        } for i, m in enumerate(self.members)]
