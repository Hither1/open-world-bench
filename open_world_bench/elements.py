"""Elemental system: auras, reactions and damage.

This is an original design, not a copy of any commercial game's system. Six
elements can be *applied* to a target, where they linger as an **aura** with a
decaying charge. Applying a second, different element consumes both and fires a
**reaction**. Reactions are the combat depth of the benchmark: they are what
make the order of actions, and therefore planning across a party, matter.

    EMBER  heat            TIDE   water
    FROST  cold            SPARK  electricity
    GALE   wind            STONE  earth

Reaction table (symmetric unless noted):

    EMBER + TIDE  -> STEAMBURST   area damage around the target
    EMBER + FROST -> THAW         large single-target multiplier
    TIDE  + FROST -> GLACIATE     target is frozen in place
    TIDE  + SPARK -> CONDUIT      damage chains to nearby wet targets
    FROST + SPARK -> SHATTERCHARGE stuns and strips the aura
    EMBER + SPARK -> OVERPRESSURE knockback plus damage
    GALE  + any   -> SQUALL       spreads the existing aura to neighbours
    STONE + any   -> ENCASE       converts the aura into a shield for the player
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Element(IntEnum):
    NONE = 0
    EMBER = 1
    TIDE = 2
    FROST = 3
    SPARK = 4
    GALE = 5
    STONE = 6


ELEMENT_NAMES = ["none", "ember", "tide", "frost", "spark", "gale", "stone"]
# Display colours, also used for auras, skill effects and party UI.
ELEMENT_COLORS = {
    Element.NONE: (0.75, 0.75, 0.78),
    Element.EMBER: (1.00, 0.45, 0.22),
    Element.TIDE: (0.26, 0.62, 0.95),
    Element.FROST: (0.62, 0.90, 0.98),
    Element.SPARK: (0.78, 0.52, 1.00),
    Element.GALE: (0.52, 0.93, 0.74),
    Element.STONE: (0.93, 0.78, 0.36),
}


class Reaction(IntEnum):
    NONE = 0
    STEAMBURST = 1
    THAW = 2
    GLACIATE = 3
    CONDUIT = 4
    SHATTERCHARGE = 5
    OVERPRESSURE = 6
    SQUALL = 7
    ENCASE = 8


REACTION_NAMES = ["none", "steamburst", "thaw", "glaciate", "conduit",
                  "shattercharge", "overpressure", "squall", "encase"]

# (applied, aura) -> reaction. GALE and STONE are handled separately because
# they react with *any* aura rather than with one specific partner.
_PAIRS: dict[frozenset, Reaction] = {
    frozenset((Element.EMBER, Element.TIDE)): Reaction.STEAMBURST,
    frozenset((Element.EMBER, Element.FROST)): Reaction.THAW,
    frozenset((Element.TIDE, Element.FROST)): Reaction.GLACIATE,
    frozenset((Element.TIDE, Element.SPARK)): Reaction.CONDUIT,
    frozenset((Element.FROST, Element.SPARK)): Reaction.SHATTERCHARGE,
    frozenset((Element.EMBER, Element.SPARK)): Reaction.OVERPRESSURE,
}


@dataclass(frozen=True)
class ReactionEffect:
    """What a reaction does, in units the combat code can apply directly."""

    damage_multiplier: float = 1.0
    bonus_damage: float = 0.0
    radius: float = 0.0          # >0 means it also hits neighbours
    stun: float = 0.0            # seconds the target cannot act
    freeze: float = 0.0          # seconds the target cannot move
    knockback: float = 0.0       # metres
    spread_aura: bool = False    # re-applies the consumed aura to neighbours
    shield: float = 0.0          # shield points granted to the player
    energy: float = 6.0          # energy granted to the party for reacting


REACTION_EFFECTS: dict[Reaction, ReactionEffect] = {
    Reaction.STEAMBURST: ReactionEffect(damage_multiplier=1.7, radius=5.0, knockback=1.6, energy=8.0),
    Reaction.THAW: ReactionEffect(damage_multiplier=2.6, energy=8.0),
    Reaction.GLACIATE: ReactionEffect(damage_multiplier=1.1, freeze=3.0, energy=7.0),
    Reaction.CONDUIT: ReactionEffect(damage_multiplier=1.4, radius=7.5, energy=9.0),
    Reaction.SHATTERCHARGE: ReactionEffect(damage_multiplier=1.5, stun=1.6, energy=8.0),
    Reaction.OVERPRESSURE: ReactionEffect(damage_multiplier=1.6, radius=4.0, knockback=4.5, energy=8.0),
    Reaction.SQUALL: ReactionEffect(damage_multiplier=1.2, radius=8.0, spread_aura=True, knockback=2.5, energy=7.0),
    Reaction.ENCASE: ReactionEffect(damage_multiplier=1.0, shield=45.0, energy=6.0),
}

# Elements an enemy resists (takes 40% damage from) and is weak to (160%).
RESIST_MULT = 0.4
WEAK_MULT = 1.6


def resolve(applied: Element, aura: Element) -> Reaction:
    """Which reaction fires when `applied` lands on a target holding `aura`."""
    if applied == Element.NONE or aura == Element.NONE or applied == aura:
        return Reaction.NONE
    if applied == Element.GALE or aura == Element.GALE:
        return Reaction.SQUALL
    if applied == Element.STONE or aura == Element.STONE:
        return Reaction.ENCASE
    return _PAIRS.get(frozenset((applied, aura)), Reaction.NONE)


@dataclass
class Aura:
    """A decaying elemental charge attached to one target."""

    element: Element = Element.NONE
    charge: float = 0.0
    decay: float = 0.55          # charge per second

    def tick(self, dt: float) -> None:
        if self.element == Element.NONE:
            return
        self.charge -= self.decay * dt
        if self.charge <= 0.0:
            self.element = Element.NONE
            self.charge = 0.0

    def apply(self, element: Element, charge: float = 1.0) -> Reaction:
        """Apply an element. Returns the reaction that fired, if any."""
        if element == Element.NONE:
            return Reaction.NONE
        if self.element == Element.NONE:
            self.element, self.charge = element, charge
            return Reaction.NONE
        if self.element == element:
            self.charge = max(self.charge, charge)
            return Reaction.NONE
        reaction = resolve(element, self.element)
        if reaction == Reaction.SQUALL:
            # Gale blows the *existing* aura outward and leaves nothing behind.
            self.element, self.charge = Element.NONE, 0.0
        elif reaction == Reaction.ENCASE:
            self.element, self.charge = Element.NONE, 0.0
        else:
            self.element, self.charge = Element.NONE, 0.0
        return reaction

    def as_vector(self) -> np.ndarray:
        v = np.zeros(len(Element), dtype=np.float32)
        v[int(self.element)] = float(np.clip(self.charge, 0.0, 2.0))
        return v


@dataclass
class ElementalStatus:
    """Per-enemy combat state: aura, shield, and crowd control timers."""

    aura: Aura = field(default_factory=Aura)
    shield: float = 0.0
    shield_element: Element = Element.NONE
    frozen: float = 0.0
    stunned: float = 0.0
    resists: tuple[Element, ...] = ()
    weak_to: tuple[Element, ...] = ()

    def tick(self, dt: float) -> None:
        self.aura.tick(dt)
        self.frozen = max(0.0, self.frozen - dt)
        self.stunned = max(0.0, self.stunned - dt)

    @property
    def disabled(self) -> bool:
        return self.frozen > 0.0 or self.stunned > 0.0

    def multiplier_for(self, element: Element) -> float:
        if element in self.weak_to:
            return WEAK_MULT
        if element in self.resists:
            return RESIST_MULT
        return 1.0


def damage_against_shield(status: ElementalStatus, element: Element, damage: float) -> tuple[float, float, bool]:
    """Split incoming damage between an elemental shield and health.

    An elemental shield only yields quickly to the element that opposes it,
    which is what forces the player to bring the right party member.

    Returns (damage_to_health, damage_to_shield, shield_broke).
    """
    if status.shield <= 0.0:
        return damage, 0.0, False
    opposing = resolve(element, status.shield_element) != Reaction.NONE
    efficiency = 2.5 if opposing else 0.35
    to_shield = min(status.shield, damage * efficiency)
    status.shield -= to_shield
    broke = status.shield <= 1e-6
    if broke:
        status.shield = 0.0
        status.shield_element = Element.NONE
    # Only overflow past a broken shield reaches health.
    leftover = max(0.0, damage - to_shield / max(efficiency, 1e-6))
    return (leftover if broke else 0.0), to_shield, broke
