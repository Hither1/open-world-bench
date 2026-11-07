"""The living world: interactable entities, combat, day/night cycle."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

from . import math3d as mth
from .character import CharacterConfig, CharacterController, CharacterState, MoveState
from .elements import Element, ElementalStatus, Reaction, REACTION_EFFECTS, damage_against_shield
from .events import Event
from .party import Party, SkillKind
from .scatter import PropField, scatter_props
from .terrain import Terrain, TerrainConfig


class EntityKind(IntEnum):
    NPC = 0
    CHEST = 1
    BEACON = 2          # activatable wind beacon
    PILLAR = 3          # activatable puzzle pillar
    COLLECTIBLE = 4     # pickable plant / ore
    ENEMY = 5
    BOSS = 6
    ALTAR = 7
    WAYPOINT = 8        # respawn / fast-travel statue


ENTITY_NAMES = ["npc", "chest", "beacon", "pillar", "collectible", "enemy", "boss", "altar", "waypoint"]
# Kinds the player can trigger with the interact button.
INTERACTABLE = (EntityKind.NPC, EntityKind.CHEST, EntityKind.BEACON, EntityKind.PILLAR,
                EntityKind.COLLECTIBLE, EntityKind.ALTAR, EntityKind.WAYPOINT)
HOSTILE = (EntityKind.ENEMY, EntityKind.BOSS)


@dataclass
class Entity:
    id: str
    kind: EntityKind
    pos: np.ndarray
    label: str = ""
    yaw: float = 0.0
    radius: float = 2.6              # interaction radius
    item: str | None = None          # what a collectible or chest yields
    amount: int = 1
    hp: float = 100.0
    max_hp: float = 0.0
    damage: float = 9.0
    aggro_radius: float = 15.0
    used: bool = False               # opened / lit / picked
    alive: bool = True
    dialogue: str = ""
    home: np.ndarray | None = None
    attack_cd: float = 0.0
    bob: float = 0.0
    elemental: ElementalStatus = field(default_factory=ElementalStatus)
    visual_role: str = ""          # geometry cues; never used by physics or task logic
    visual_version: int = 1

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=np.float32)
        if self.home is None:
            self.home = self.pos.copy()
        if not self.label:
            self.label = self.id.replace("_", " ").title()
        if self.max_hp <= 0.0:
            self.max_hp = self.hp


@dataclass
class WorldConfig:
    seed: int = 0
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    prop_density: float = 2.4
    day_length: float = 480.0        # seconds of sim time for a full day
    start_time_of_day: float = 0.32  # morning
    player_attack_range: float = 2.8
    player_attack_arc_deg: float = 100.0
    player_attack_damage: float = 34.0
    player_attack_cooldown: float = 0.42
    enemy_attack_cooldown: float = 1.3
    enemy_speed: float = 3.4
    respawn_on_death: bool = True


class World:
    """Owns the terrain, the props and every entity in it."""

    def __init__(self, cfg: WorldConfig | None = None,
                 terrain: Terrain | None = None, props: PropField | None = None,
                 party: Party | None = None):
        """Build a world. `terrain`/`props` may be passed in to reuse a cached
        island across episode resets, which is most of the construction cost."""
        self.cfg = cfg or WorldConfig()
        tcfg = self.cfg.terrain
        if tcfg.seed != self.cfg.seed:
            tcfg = TerrainConfig(**{**tcfg.__dict__, "seed": self.cfg.seed})
        self.terrain = terrain if terrain is not None else Terrain(tcfg)
        self.props = props if props is not None else scatter_props(
            self.terrain, self.cfg.seed, self.cfg.prop_density)
        self.rng = np.random.default_rng(self.cfg.seed * 104729 + 7)
        self.entities: dict[str, Entity] = {}
        self.time_of_day = float(self.cfg.start_time_of_day)
        self.elapsed = 0.0
        self._player_attack_cd = 0.0
        self.last_interact_hint: str | None = None
        self.party = party
        self.combat = CombatSystem(self, party)

    # ----------------------------------------------------------------- spawn
    def ground_pos(self, x: float, z: float, lift: float = 0.0) -> np.ndarray:
        x, z = self.terrain.clamp_to_bounds(float(x), float(z), margin=4.0)
        y = float(self.terrain.height_at(x, z)) + lift
        return np.array([x, y, z], dtype=np.float32)

    def spawn(self, entity: Entity) -> Entity:
        if entity.id in self.entities:
            raise ValueError(f"duplicate entity id {entity.id!r}")
        self.entities[entity.id] = entity
        return entity

    def spawn_at(self, id: str, kind: EntityKind, x: float, z: float, lift: float = 0.0, **kw: Any) -> Entity:
        return self.spawn(Entity(id=id, kind=kind, pos=self.ground_pos(x, z, lift), **kw))

    def get(self, id: str) -> Entity:
        return self.entities[id]

    def of_kind(self, *kinds: EntityKind) -> list[Entity]:
        ks = set(kinds)
        return [e for e in self.entities.values() if e.kind in ks]

    # -------------------------------------------------------------- lighting
    @property
    def sun_direction(self) -> np.ndarray:
        """Unit vector pointing *toward* the sun."""
        ang = (self.time_of_day - 0.25) * 2.0 * np.pi      # 0.25 -> sunrise
        return mth.norm(np.array([np.cos(ang), np.sin(ang), -0.35], dtype=np.float32))

    @property
    def daylight(self) -> float:
        """0 at night, 1 at midday, with soft twilight shoulders."""
        return float(np.clip(self.sun_direction[1] * 1.6 + 0.22, 0.0, 1.0))

    def time_label(self) -> str:
        t = self.time_of_day % 1.0
        for lo, name in ((0.05, "night"), (0.22, "dawn"), (0.42, "morning"),
                         (0.58, "midday"), (0.72, "afternoon"), (0.85, "dusk")):
            if t < lo:
                return name
        return "night"

    # ----------------------------------------------------------------- update
    def update(self, char: CharacterState, dt: float, char_cfg: CharacterConfig) -> list[Event]:
        events: list[Event] = []
        self.elapsed += dt
        self.time_of_day = (self.time_of_day + dt / max(self.cfg.day_length, 1e-3)) % 1.0
        self._player_attack_cd = max(0.0, self._player_attack_cd - dt)
        self.combat.tick(dt)
        events += self._handle_defeat(char, char_cfg)
        if char.hp <= 0.0:
            return events

        for e in self.entities.values():
            e.bob = (e.bob + dt) % 1000.0
            if e.kind in HOSTILE and e.alive:
                events += self._update_enemy(e, char, dt)

        events += self._handle_defeat(char, char_cfg)
        self.last_interact_hint = self._hint(char)
        return events

    def swap_character(self, char: CharacterState, index: int) -> list[Event]:
        party = self.party
        if party is None or char.hp <= 0.0:
            return []
        party.active.hp = char.hp
        if not party.swap(index):
            return []
        char.hp = party.active.hp
        return [Event("party_swap", {"slot": index, "character": party.active.spec.id})]

    def _handle_defeat(self, char: CharacterState, char_cfg: CharacterConfig) -> list[Event]:
        party = self.party
        if party is not None:
            party.active.hp = char.hp
        if char.hp > 0.0:
            return []
        events = []
        if party is not None:
            if party.active.downed:
                return []
            party.active.downed = True
            events.append(Event("member_downed", {"character": party.active.spec.id}))
            if party.swap_to_next_alive():
                char.hp = party.active.hp
                events.append(Event("party_swap", {"slot": party.active_index,
                                                    "character": party.active.spec.id, "automatic": True}))
                return events
        events.append(Event("player_defeated", {"pos": char.pos.copy()}))
        if self.cfg.respawn_on_death:
            if party is not None:
                party.revive()
            self._respawn(char, char_cfg)
        return events

    def _update_enemy(self, e: Entity, char: CharacterState, dt: float) -> list[Event]:
        events: list[Event] = []
        e.attack_cd = max(0.0, e.attack_cd - dt)
        if e.elemental.disabled:
            return events
        to_player = char.pos - e.pos
        dist = float(np.linalg.norm(to_player[[0, 2]]))
        if dist < e.aggro_radius:
            if dist > 1.9:
                step = mth.norm(np.array([to_player[0], 0.0, to_player[2]], dtype=np.float32))
                speed = self.cfg.enemy_speed * (1.25 if e.kind == EntityKind.BOSS else 1.0)
                nxt = e.pos + step * speed * dt
                e.pos = self.ground_pos(float(nxt[0]), float(nxt[2]))
                e.yaw = float(np.arctan2(step[0], -step[2]))
            elif e.attack_cd <= 0.0 and char.hp > 0.0 and abs(float(to_player[1])) <= 3.0:
                damage = self.party.absorb(e.damage) if self.party else e.damage
                absorbed = e.damage - damage
                damage = min(char.hp, damage)
                char.hp = max(0.0, char.hp - damage)
                e.attack_cd = self.cfg.enemy_attack_cooldown
                events.append(Event("player_hit", {"entity": e.id, "damage": damage,
                                                   "absorbed": absorbed}))
        elif dist > e.aggro_radius * 1.8:
            # Drift home so enemies cannot be kited across the whole island.
            back = e.home - e.pos
            if float(np.linalg.norm(back[[0, 2]])) > 1.0:
                step = mth.norm(np.array([back[0], 0.0, back[2]], dtype=np.float32))
                nxt = e.pos + step * self.cfg.enemy_speed * 0.6 * dt
                e.pos = self.ground_pos(float(nxt[0]), float(nxt[2]))
        return events

    def _respawn(self, char: CharacterState, char_cfg: CharacterConfig) -> None:
        waypoints = self.of_kind(EntityKind.WAYPOINT)
        if waypoints:
            dists = [float(np.linalg.norm((w.pos - char.pos)[[0, 2]])) for w in waypoints]
            target = waypoints[int(np.argmin(dists))]
            char.pos = self.ground_pos(float(target.pos[0]) + 1.5, float(target.pos[2]))
        char.vel[:] = 0.0
        char.hp = self.party.active.hp if self.party else char_cfg.max_hp
        char.stamina = char_cfg.stamina_max
        char.state = MoveState.GROUNDED
        char.grounded = True
        char.air_time = char._peak_fall_speed = char._regen_timer = 0.0
        char._jump_was_down = False

    # ------------------------------------------------------------ interaction
    def _candidates(self, char: CharacterState) -> list[tuple[float, Entity]]:
        out = []
        for e in self.entities.values():
            if e.kind not in INTERACTABLE:
                continue
            if e.used and e.kind != EntityKind.NPC:
                continue
            d = e.pos - char.pos
            dist = float(np.linalg.norm(d[[0, 2]]))
            if dist <= e.radius and abs(float(d[1])) <= 4.5:
                out.append((dist, e))
        out.sort(key=lambda t: t[0])
        return out

    _VERBS = {
        EntityKind.NPC: "Talk to", EntityKind.CHEST: "Open", EntityKind.BEACON: "Light",
        EntityKind.PILLAR: "Touch", EntityKind.COLLECTIBLE: "Pick",
        EntityKind.ALTAR: "Make an offering at", EntityKind.WAYPOINT: "Attune to",
    }

    def _hint(self, char: CharacterState) -> str | None:
        cands = self._candidates(char)
        if not cands:
            return None
        e = cands[0][1]
        return f"{self._VERBS[e.kind]} {e.label}"

    def interact(self, char: CharacterState) -> list[Event]:
        cands = self._candidates(char)
        if not cands:
            return []
        e = cands[0][1]
        k = e.kind
        if k == EntityKind.NPC:
            return [Event("talk", {"entity": e.id, "text": e.dialogue or f"{e.label} nods at you."})]
        if k == EntityKind.COLLECTIBLE:
            e.used = True
            return [Event("collect", {"entity": e.id, "item": e.item or e.id, "amount": e.amount})]
        if k == EntityKind.CHEST:
            e.used = True
            evs = [Event("open_chest", {"entity": e.id})]
            if e.item:
                evs.append(Event("collect", {"entity": e.id, "item": e.item, "amount": e.amount}))
            return evs
        if k in (EntityKind.BEACON, EntityKind.PILLAR):
            e.used = True
            return [Event("activate", {"entity": e.id})]
        if k == EntityKind.ALTAR:
            e.used = True
            return [Event("offer", {"entity": e.id}), Event("activate", {"entity": e.id})]
        if k == EntityKind.WAYPOINT:
            e.used = True
            return [Event("attune", {"entity": e.id})]
        return []

    def attack(self, char: CharacterState) -> list[Event]:
        return self.combat.attack(char)

    # --------------------------------------------------------------- queries
    def nearest(self, pos: np.ndarray, *kinds: EntityKind, only_unused: bool = True) -> Entity | None:
        best, best_d = None, 1e18
        pool = self.of_kind(*kinds) if kinds else list(self.entities.values())
        for e in pool:
            if only_unused and (e.used or not e.alive):
                continue
            d = float(np.linalg.norm((e.pos - pos)[[0, 2]]))
            if d < best_d:
                best, best_d = e, d
        return best

    def state_dict(self, char: CharacterState) -> dict:
        return {
            "player_pos": char.pos.copy(),
            "player_hp": char.hp,
            "time_of_day": self.time_of_day,
            "daylight": self.daylight,
        }


@dataclass
class CombatEffect:
    pos: np.ndarray
    element: Element
    radius: float
    remaining: float = 0.45


class CombatSystem:
    def __init__(self, world, party: Party | None):
        self.world = world
        self.party = party
        self.effects: list[CombatEffect] = []

    def tick(self, dt: float) -> None:
        for effect in self.effects:
            effect.remaining -= dt
        self.effects[:] = [e for e in self.effects if e.remaining > 0.0]
        for enemy in self.world.of_kind(*HOSTILE):
            enemy.elemental.tick(dt)
        if self.party is not None:
            self.party.tick(dt)

    def _targets(self, char: CharacterState, reach: float, *, cone: bool = False) -> list:
        facing = mth.yaw_pitch_to_dir(char.yaw, 0.0)
        threshold = np.cos(np.radians(self.world.cfg.player_attack_arc_deg * 0.5))
        targets = []
        for enemy in self.world.of_kind(*HOSTILE):
            delta = enemy.pos - char.pos
            dist = float(np.linalg.norm(delta))
            horizontal = float(np.linalg.norm(delta[[0, 2]]))
            if not enemy.alive or dist > reach:
                continue
            if cone and horizontal > 1e-6:
                direction = np.array([delta[0], 0.0, delta[2]]) / horizontal
                if float(np.dot(direction, facing)) < threshold:
                    continue
            targets.append((dist, enemy.id, enemy))
        return [e for _, _, e in sorted(targets, key=lambda row: row[:2])]

    def attack(self, char: CharacterState) -> list[Event]:
        world, party = self.world, self.party
        if char.hp <= 0.0 or world._player_attack_cd > 0.0:
            return []
        world._player_attack_cd = world.cfg.player_attack_cooldown
        damage = party.active.spec.attack * party.attack_multiplier if party else world.cfg.player_attack_damage
        events = [Event("swing")]
        for enemy in self._targets(char, world.cfg.player_attack_range, cone=True):
            events += self._hit(enemy, char, damage, Element.NONE)
        return events

    def cast(self, char: CharacterState, controller: CharacterController, *, burst: bool = False) -> list[Event]:
        party = self.party
        if party is None or char.hp <= 0.0:
            return []
        ability = party.ready_burst() if burst else party.ready_skill()
        if ability is None:
            return []
        party.active.hp = char.hp
        if burst:
            party.spend_burst()
        else:
            party.spend_skill()
        events = [Event("burst" if burst else "skill", {
            "character": party.active.spec.id, "name": ability.name,
            "element": ability.element.name.lower(),
        })]

        if ability.kind == SkillKind.DASH:
            start, end = controller.dash(char, ability.range)
            segment = end - start
            denom = float(np.dot(segment, segment))
            targets = []
            for enemy in self.world.of_kind(*HOSTILE):
                t = float(np.clip(np.dot(enemy.pos - start, segment) / max(denom, 1e-8), 0.0, 1.0))
                if enemy.alive and np.linalg.norm(enemy.pos - (start + t * segment)) <= ability.radius:
                    targets.append(enemy)
        else:
            cone = ability.kind in (SkillKind.STRIKE, SkillKind.BOLT)
            targets = self._targets(char, ability.range or ability.radius, cone=cone)
            if ability.kind == SkillKind.BOLT:
                targets = targets[:1]

        if ability.launch_speed:
            char.vel[1] = ability.launch_speed
            char.grounded = False
            char.state = MoveState.AIRBORNE
        if ability.shield:
            party.add_shield(ability.shield, ability.element)
        if ability.heal:
            party.heal(ability.heal)
            char.hp = party.active.hp
        char.stamina = min(controller.cfg.stamina_max, char.stamina + ability.stamina_refund)

        self.effects.append(CombatEffect(char.pos.copy(), ability.element, ability.radius or 1.0))
        for enemy in targets:
            if enemy.alive:
                events += self._hit(enemy, char, ability.damage * party.attack_multiplier,
                                    ability.element, ability.charge)
        return events

    def _damage(self, enemy, damage: float, element: Element) -> list[Event]:
        if not enemy.alive:
            return []
        damage *= enemy.elemental.multiplier_for(element)
        health, shield, broke = damage_against_shield(enemy.elemental, element, damage)
        actual = min(enemy.hp, health)
        enemy.hp = max(0.0, enemy.hp - health)
        events = [Event("enemy_hit", {
            "entity": enemy.id, "hp": enemy.hp, "damage": actual,
            "shield_damage": shield, "element": element.name.lower(),
        })]
        if broke:
            events.append(Event("shield_broken", {"entity": enemy.id}))
        if enemy.hp <= 0.0:
            enemy.alive = False
            events.append(Event("defeat", {"entity": enemy.id, "kind": int(enemy.kind)}))
        return events

    def _hit(self, enemy, char: CharacterState, damage: float, element: Element,
             charge: float = 1.0) -> list[Event]:
        status = enemy.elemental
        previous = status.aura.element
        reaction = status.aura.apply(element, charge)
        party = self.party
        if party is not None:
            party.grant_energy(party.cfg.hit_energy)
        if reaction == Reaction.NONE:
            return self._damage(enemy, damage, element)

        effect = REACTION_EFFECTS[reaction]
        multiplier = party.reaction_multiplier if party else 1.0
        total = (damage * effect.damage_multiplier + effect.bonus_damage) * multiplier
        center = enemy.pos.copy()
        events = [Event("reaction", {
            "entity": enemy.id, "reaction": reaction.name.lower(),
            "element": element.name.lower(), "aura": previous.name.lower(),
        })]
        events += self._damage(enemy, total, element)
        status.frozen = max(status.frozen, effect.freeze)
        status.stunned = max(status.stunned, effect.stun)
        if enemy.alive and effect.knockback:
            direction = mth.norm(np.array([center[0] - char.pos[0], 0.0, center[2] - char.pos[2]]))
            point = center + direction * effect.knockback
            enemy.pos = self.world.ground_pos(float(point[0]), float(point[2]))
        if party is not None:
            party.grant_energy(effect.energy, to_all=True)
            if effect.shield:
                party.add_shield(effect.shield, previous)
        self.effects.append(CombatEffect(center, element, effect.radius or 1.5))

        # A reaction has one bounded splash pass: secondary hits cannot recurse.
        if effect.radius:
            spread = previous if previous not in (Element.GALE, Element.STONE) else element
            for other in self.world.of_kind(*HOSTILE):
                if other is enemy or not other.alive or np.linalg.norm(other.pos - center) > effect.radius:
                    continue
                if reaction == Reaction.CONDUIT and other.elemental.aura.element != Element.TIDE:
                    continue
                events += self._damage(other, total * 0.5, spread if effect.spread_aura else element)
                if effect.spread_aura and other.alive:
                    other.elemental.aura.element = spread
                    other.elemental.aura.charge = charge
        return events
